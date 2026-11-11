"""Threaded IR-to-hands perception loop for the hand modes (F-C).

:class:`HandPerception` owns the analysis thread. The source delivers decoded
IR/depth frames from its own thread; the loop keeps only the newest pending
frame, so a slow DPU never builds a backlog and the hands shown are the ones
the camera saw last. Callbacks run on the analysis thread, in order of the
frames actually processed.

The loop lives in the application layer because it composes the vision
pipeline with the camera source port; the IR model input preprocessing (OpenCV
CLAHE) arrives as the injected ``preprocess`` callable from the composition
root.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cpsa.application.slot import LatestSlot
from cpsa.ports import PoseEstimator
from cpsa.vision.calibration import DepthModel, Intrinsics
from cpsa.vision.hand3d import Hand3D, HandFrame, lift_keypoints
from cpsa.vision.irdepth import IrDepthFrame
from cpsa.vision.types import PersonPose

logger = logging.getLogger(__name__)

__all__ = ["TIMING_KEYS", "HandPerception"]

TIMING_KEYS: tuple[str, ...] = (
    "decode_ms",
    "preprocess_ms",
    "dpu_ms",
    "post_ms",
    "lift_ms",
)
"""The ``HandFrame.timing`` keys, in pipeline order."""

_SLOT_TIMEOUT_S = 0.1
_CAMERA_STALE_S = 1.0
_RATE_WINDOW = 30
_BOX_VALUES = 4


class IrDepthSourceLike(Protocol):
    """The part of an IR/depth source this loop drives."""

    def start(self, on_frame: Callable[[IrDepthFrame], None]) -> None: ...

    def stop(self) -> None: ...


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _box(values: NDArray[np.generic]) -> tuple[float, float, float, float]:
    box = np.asarray(values, dtype=np.float64).reshape(-1)
    if box.size != _BOX_VALUES:
        raise ValueError(f"a hand box needs {_BOX_VALUES} values, got {box.size}")
    x1, y1, x2, y2 = (float(value) for value in box)
    return (x1, y1, x2, y2)


class HandPerception:
    """Run an IR hand pose model over an IR/depth source and deliver frames.

    Args:
        source: IR/depth source, for example
            :class:`cpsa.adapters.camera.irdepth.IrDepthSource`.
        estimator: Pose estimator producing 21-keypoint hands from the
            preprocessed IR image.
        intrinsics: IR intrinsics used by the lift.
        depth: Raw Z16 depth to metres model.
        preprocess: ``(H, W)`` uint8 IR image to the ``(H, W, 3)`` model input;
            the composition root passes the adapter that applies the CLAHE.
        max_fps: Upper bound on the frames processed per second.

    Raises:
        ValueError: If ``max_fps`` is not positive.
    """

    def __init__(
        self,
        source: IrDepthSourceLike,
        estimator: PoseEstimator,
        intrinsics: Intrinsics,
        depth: DepthModel,
        *,
        preprocess: Callable[[NDArray[np.uint8]], NDArray[np.uint8]],
        max_fps: float = 30.0,
    ) -> None:
        if max_fps <= 0.0:
            raise ValueError(f"max_fps must be positive, got {max_fps}")
        self._source = source
        self._estimator = estimator
        self._intrinsics = intrinsics
        self._depth = depth
        self._preprocess = preprocess
        self._period = 1.0 / float(max_fps)
        self._lifecycle = threading.Lock()
        self._lock = threading.Lock()
        self._slot: LatestSlot[IrDepthFrame] = LatestSlot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._processed: deque[float] = deque(maxlen=_RATE_WINDOW)
        self._last_arrival = 0.0
        self._last_dpu_ms = 0.0

    @property
    def running(self) -> bool:
        """True while the analysis thread is active."""
        return self._running and not self._stop.is_set()

    def start(self, on_frame: Callable[[HandFrame], None]) -> None:
        """Open the source and deliver :class:`HandFrame` objects to ``on_frame``."""

        with self._lifecycle:
            if self._running:
                raise RuntimeError("HandPerception is already started")
            with self._lock:
                self._stop = threading.Event()
                self._slot = LatestSlot()
                self._processed.clear()
                self._last_arrival = 0.0
                self._last_dpu_ms = 0.0
                self._running = True
            try:
                self._source.start(self._ingest)
            except Exception:
                with self._lock:
                    self._running = False
                raise
            thread = threading.Thread(
                target=self._run,
                args=(on_frame,),
                name="cpsa-hand-perception",
                daemon=True,
            )
            with self._lock:
                self._thread = thread
            try:
                thread.start()
            except Exception:
                with self._lock:
                    self._thread = None
                    self._running = False
                self._source.stop()
                raise

    def stop(self, timeout_s: float) -> bool:
        """Stop the loop and the source; return whether the loop stopped in time.

        The wait for the analysis thread is bounded by ``timeout_s``; closing
        the source afterwards is bounded by the source's own stop.
        """

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._lock:
            running = self._running
            if not running:
                return True
            self._running = False
            self._stop.set()
            thread = self._thread
        stopped = True
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            stopped = not thread.is_alive()
            if not stopped:
                logger.warning("hand perception did not stop within %.2f s", timeout_s)
        try:
            self._source.stop()
        except Exception:
            logger.exception("IR/depth source failed to stop")
        with self._lock:
            self._thread = None
        return stopped

    def health(self) -> Mapping[str, Any]:
        """Report ``camera`` (frames flowing), ``fps`` and the last ``dpu_ms``."""

        with self._lock:
            running = self._running
            stamps = tuple(self._processed)
            last_arrival = self._last_arrival
            dpu_ms = self._last_dpu_ms
        fresh = last_arrival > 0.0 and (time.monotonic() - last_arrival) <= _CAMERA_STALE_S
        fps = 0.0
        if len(stamps) >= 2 and stamps[-1] > stamps[0]:
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        return {"camera": running and fresh, "fps": float(fps), "dpu_ms": float(dpu_ms)}

    def _ingest(self, frame: IrDepthFrame) -> None:
        with self._lock:
            self._last_arrival = time.monotonic()
        self._slot.put(frame)

    def _run(self, on_frame: Callable[[HandFrame], None]) -> None:
        next_due = 0.0
        while not self._stop.is_set():
            frame = self._slot.get(timeout=_SLOT_TIMEOUT_S)
            if frame is None:
                continue
            now = time.perf_counter()
            if now < next_due:
                continue
            next_due = now + self._period
            try:
                hand_frame = self._process(frame)
            except Exception:
                logger.exception("hand perception failed on frame %d", frame.seq)
                continue
            if self._stop.is_set():
                return
            with self._lock:
                self._processed.append(time.perf_counter())
            try:
                on_frame(hand_frame)
            except Exception:
                logger.exception("hand frame callback failed")

    def _process(self, frame: IrDepthFrame) -> HandFrame:
        started = time.perf_counter()
        model_input = self._preprocess(frame.ir)
        preprocess_ms = _elapsed_ms(started)

        started = time.perf_counter()
        persons = list(self._estimator.estimate(model_input))
        dpu_ms = _elapsed_ms(started)

        started = time.perf_counter()
        keypoints = [_keypoints_2d(person) for person in persons]
        post_ms = _elapsed_ms(started)

        started = time.perf_counter()
        hands = tuple(
            Hand3D(
                score=float(person.score),
                box=_box(person.box),
                kpts2d=kpts2d,
                kpts3d=lift_keypoints(kpts2d, frame.depth_raw, self._intrinsics, self._depth),
            )
            for person, kpts2d in zip(persons, keypoints, strict=True)
        )
        lift_ms = _elapsed_ms(started)

        timing = {
            "decode_ms": float(frame.decode_ms),
            "preprocess_ms": preprocess_ms,
            "dpu_ms": dpu_ms,
            "post_ms": post_ms,
            "lift_ms": lift_ms,
        }
        return HandFrame(
            seq=frame.seq,
            t=frame.t,
            width=frame.width,
            height=frame.height,
            hands=hands,
            timing=timing,
        )


def _keypoints_2d(person: PersonPose) -> NDArray[np.float32]:
    """Stack a person's ``xy`` and ``conf`` into ``(K, 3)`` float32 keypoints."""

    xy = np.asarray(person.xy, dtype=np.float32).reshape(-1, 2)
    conf = np.asarray(person.conf, dtype=np.float32).reshape(-1)
    if conf.size != xy.shape[0]:
        raise ValueError(f"got {xy.shape[0]} keypoint positions but {conf.size} confidences")
    return np.column_stack((xy, conf)).astype(np.float32)
