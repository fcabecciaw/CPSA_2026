"""The detection loop: camera frames in, overlaid video and states out.

One loop processes frames from a :class:`~cpsa.camera.UvcCamera` with **one**
DPU model at a time (R2-G): :class:`~cpsa.yolox.YoloxDetector` for the boxes or
:class:`~cpsa.movenet.DpuMovenet` for the 17-keypoint skeleton. ``model.set``
swaps the loaded model: the loop waits for the frame in flight, releases the
current runner (graph and buffers) and builds the other one, then resumes. The
tracker stabilizes the YOLOX boxes and the keypoint stabilizer the MoveNet
skeleton. The loop publishes the overlay of the active
model into the shared :class:`~cpsa.channel.FrameChannel` and the snapshot
messages into the :class:`~cpsa.channel.MessageChannel`; ``model.set`` arrives
the other way through the :class:`~cpsa.channel.CommandChannel`.

The 60 fps target needs the DPU and the CPU to work on different frames:

```text
submit the model for frame N (preprocess + execute_async)
decode + draw + publish frame N-1   (while the DPU runs on N)
next iteration: submit N+1, then wait N
```

The detectors expose that split as ``submit``/``wait``; a detector with only
``detect`` (a fake in a test, a CPU stand-in) runs the simple synchronous path.
The camera slot keeps only the newest frame, so a slow frame is dropped, never
queued.

A booth stands still for minutes at a time, so the YOLOX view has a motion gate
in front of the DPU: the loop compares each frame's 80x60 grey image with the
one it last inferred, and while nothing moves it draws the previous boxes again
instead of preprocess, DPU and decode (:class:`~cpsa.motion.StillGate`). The
tracker's boxes and the published objects are reused untouched, so a still
scene costs the camera conversion, the motion check and the draw only; a full
detection still runs at least every ``tracker.still.force_every_s`` seconds, so
a new object that appears without moving the frame cannot go unnoticed.

The loop also owns the process status: ``status`` once per second and on change
(state, camera, errors, ``active_model``, ``switching``) into the message
channel, ``showcase.state`` at most ten times per second while YOLOX is loaded,
``pose.state`` at most ``movenet.publish_hz`` times per second while MoveNet is
loaded, and the per-stage timing line every five seconds. A camera that stops
(unplugged, a driver error) is reopened with a capped backoff, which is how a
re-plug recovers without restarting the process.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from cpsa import protocol
from cpsa.camera import CameraError, CameraNotFound, UvcCamera, VideoFrame
from cpsa.channel import CommandChannel, FrameChannel, MessageChannel
from cpsa.config import CameraConfig, ModelConfig, MovenetConfig, ShowcaseConfig
from cpsa.keypoints import KeypointStabilizer
from cpsa.motion import StillGate
from cpsa.movenet import DpuMovenet, MovenetPose
from cpsa.overlay import draw_overlay, draw_skeleton
from cpsa.tracker import Tracker
from cpsa.yolox import Detection, YoloxDetector, class_thresholds, label_for, labels_for

logger = logging.getLogger("cpsa.pipeline")

__all__ = [
    "STATE_PERIOD_S",
    "STATUS_PERIOD_S",
    "TIMING_PERIOD_S",
    "Pipeline",
    "StartError",
    "cap_admits",
]

STATE_PERIOD_S = 0.1
"""``showcase.state`` is published at most 10 Hz; the video is not throttled."""

STATUS_PERIOD_S = 1.0
"""``status`` is published at most once per second, and on every change."""

TIMING_PERIOD_S = 5.0
"""The per-stage timing line is logged once per this many seconds."""

CAMERA_STALE_S = 1.0
"""``status.camera`` is no longer ``ok`` when no frame arrived for this long."""

CAMERA_RETRY_S = 1.0
"""First delay before a camera that stopped is reopened; it doubles to the cap."""

CAMERA_RETRY_MAX_S = 10.0
"""Cap of the camera reopen backoff."""

_SWITCH_INTERVAL_S = 0.001
"""``sys.setswitchinterval`` in the pipeline process, in seconds."""

_RATE_WINDOW = 30
"""Frames the rolling ``fps`` is measured over."""

_SLOT_TIMEOUT_S = 0.1

_CAP_SLACK_S = 0.002
"""A frame up to this early on the frame cadence is still processed."""


def cap_admits(now: float, next_due: float, period: float) -> tuple[bool, float]:
    """Decide whether a frame arriving at ``now`` fits the frame cadence.

    The cadence is fixed (``next_due`` advances by ``period``) with
    :data:`_CAP_SLACK_S` of slack: camera frames arrive with jitter around their
    period, so a strict ``now < next_due`` test would drop every frame that is a
    hair early and roughly halve the rate when the cap equals the camera rate.
    A late frame restarts the cadence from its own arrival, so it never builds
    credit for a burst.

    Returns:
        ``(admit, next_due)`` with the updated due time.
    """

    if now < next_due - _CAP_SLACK_S:
        return False, next_due
    return True, max(next_due + period, now + period - 2.0 * _CAP_SLACK_S)


_CAMERA_CODES = {"missing": "camera_missing", "error": "camera_error"}


class StartError(Exception):
    """The process cannot start, with the F-K1 error code that explains why.

    ``model_missing``, ``dpu_error``, ``camera_missing``, ``camera_error`` and
    ``internal`` are the codes ``status.errors`` carries.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Detector(Protocol):
    """What the loop needs from a detector: ``stage_ms`` and ``submit``/``wait``."""

    @property
    def stage_ms(self) -> dict[str, float]: ...

    def submit(self, rgb: NDArray[np.uint8]) -> int: ...

    def wait(self, token: int) -> Sequence[Detection]: ...


class PoseDetector(Protocol):
    """What the loop needs from the MoveNet runner (``stage_ms`` + the two calls)."""

    @property
    def stage_ms(self) -> dict[str, float]: ...

    def submit(self, rgb: NDArray[np.uint8]) -> int: ...

    def wait(self, token: int) -> MovenetPose: ...


class Camera(Protocol):
    """What the loop needs from a camera: a lifecycle, frames and errors."""

    @property
    def running(self) -> bool: ...

    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class _PendingFrame:
    """A frame already on the DPU, waiting to be decoded and drawn."""

    seq: int
    width: int
    height: int
    rgb: NDArray[np.uint8]
    token: int
    started: float
    camera_ms: float
    model: str
    """Which runner owns ``token``: ``yolox`` or ``movenet``."""
    t: float
    """The frame's monotonic capture time: the keypoint filter measures ``dt`` from it."""


class _LatestSlot:
    """Holds at most one frame; a newer frame replaces an untaken one."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: VideoFrame | None = None

    def put(self, frame: VideoFrame) -> None:
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def get(self, timeout: float) -> VideoFrame | None:
        with self._cond:
            if not self._cond.wait_for(lambda: self._frame is not None, timeout):
                return None
            frame, self._frame = self._frame, None
            return frame


class Pipeline:
    """Camera, one DPU model, overlay and status of the showcase.

    Args:
        config: The validated F-K3 configuration.
        channel: The shared frame the server shows, already attached.
        messages: The message channel the server drains.
        monitor: Where ``status`` and ``showcase.state`` go; defaults to
            ``messages``. The process gives the pipeline both the same channel
            and keeps the interface separate for tests.
        detector_factory: Builds the YOLOX runner from the model config;
            defaults to :class:`~cpsa.yolox.YoloxDetector`.
        movenet_factory: Builds the MoveNet runner from the ``movenet`` config;
            defaults to :class:`~cpsa.movenet.DpuMovenet`.
        camera_factory: Builds the camera from the camera config; defaults to
            :class:`~cpsa.camera.UvcCamera`.
        commands: The control channel the server writes ``model.set`` into; the
            loop drains it every frame. ``None`` means the model never switches.
        labels: Label table; defaults to the configured language.

    The pipeline does not create or release the channel: the process owns it.
    """

    def __init__(
        self,
        config: ShowcaseConfig,
        *,
        channel: FrameChannel,
        messages: MessageChannel,
        detector_factory: Callable[[ModelConfig], Detector] | None = None,
        movenet_factory: Callable[[MovenetConfig], PoseDetector] | None = None,
        camera_factory: Callable[[CameraConfig], Camera] | None = None,
        commands: CommandChannel | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        self._config = config
        self._channel = channel
        self._messages = messages
        self._detector_factory = (
            detector_factory if detector_factory is not None else _default_detector
        )
        self._movenet_factory = movenet_factory if movenet_factory is not None else _default_movenet
        self._camera_factory = camera_factory if camera_factory is not None else _default_camera
        self._commands = commands
        self._labels: tuple[str, ...] = (
            tuple(labels) if labels is not None else labels_for(config.labels)
        )
        self._tracker = Tracker(config.tracker)
        still = config.tracker.still
        self._still: StillGate | None = (
            StillGate(still.motion_thr, still.force_every_s) if still.enabled else None
        )
        self._last_objects: list[protocol.ShowcaseObject] = []
        """The objects last decoded from the DPU; a skipped frame publishes them."""
        self._stabilizer = KeypointStabilizer(config.movenet.smoothing)
        self._lock = threading.Lock()
        self._slot = _LatestSlot()
        self._stop = threading.Event()
        self._runner: Detector | PoseDetector | None = None
        self._camera: Camera | None = None
        self._running = False
        self._camera_error: str | None = None
        self._errors: list[str] = []
        self._processed: deque[float] = deque(maxlen=_RATE_WINDOW)
        self._last_arrival = 0.0
        self._last_dpu_ms: float | None = None
        self._last_state = 0.0
        self._last_pose = 0.0
        self._frame_error = False
        """A frame failed; the next good frame clears the ``dpu_error`` it raised."""
        self._swap_error = False
        """A ``model.set`` load failed: ``dpu_error`` stays until a swap succeeds."""
        self._active_model = protocol.ACTIVE_MODELS[0]
        self._requested_model: str | None = None
        self._switching = False
        self._swap_ms: float | None = None
        self._last_status = 0.0
        self._status_key: tuple[str, str, str, bool, tuple[str, ...]] | None = None
        self._started_at = 0.0
        self._next_camera_attempt = 0.0
        self._camera_retry_s = CAMERA_RETRY_S
        self._timing_sums: dict[str, float] = {}
        self._timing_frames = 0
        self._timing_since = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Open the camera and the default DPU model (YOLOX).

        Raises:
            StartError: With ``model_missing`` before the camera is touched (the
                YOLOX or the enabled MoveNet ``.xmodel``),
                ``dpu_error``/``internal`` for a runner, and
                ``camera_missing``/``camera_error`` for the camera. A failed
                start releases whatever it acquired.
        """

        model = self._config.model
        if not model.path.is_file():
            raise StartError("model_missing", f"model not found: {model.path}")
        movenet = self._config.movenet
        if movenet.enabled and not movenet.path.is_file():
            raise StartError("model_missing", f"movenet model not found: {movenet.path}")

        active = protocol.ACTIVE_MODELS[0]
        try:
            runner = self._build_runner(active)
        except FileNotFoundError as exc:
            raise StartError("model_missing", str(exc)) from exc
        except RuntimeError as exc:
            raise StartError("dpu_error", str(exc)) from exc
        except Exception as exc:
            logger.exception("the %s runner could not be built", active)
            raise StartError("internal", str(exc)) from exc

        camera = self._camera_factory(self._config.camera)
        try:
            camera.start(self._ingest, self._camera_failed)
        except CameraNotFound as exc:
            raise StartError("camera_missing", str(exc)) from exc
        except CameraError as exc:
            raise StartError("camera_error", str(exc)) from exc
        except Exception as exc:
            logger.exception("the camera could not be opened")
            _release_quietly(camera)
            self._release_runner(runner)
            raise StartError("camera_error", str(exc)) from exc

        with self._lock:
            self._runner = runner
            self._camera = camera
            self._started_at = time.monotonic()
            self._last_arrival = time.monotonic()
            self._processed.clear()
            self._errors = []
            self._camera_error = None
            self._status_key = None
            self._tracker.reset()
            self._reset_still()
            self._active_model = active
            self._requested_model = None
            self._switching = False
            self._swap_ms = None
            self._swap_error = False
            self._frame_error = False
            self._last_state = 0.0
            self._last_pose = 0.0
            self._timing_sums = {}
            self._timing_frames = 0
            self._timing_since = time.monotonic()
            self._camera_retry_s = CAMERA_RETRY_S
            self._next_camera_attempt = 0.0
        logger.info(
            "showcase started: model=%s movenet=%s camera=%dx%d@%g",
            model.name,
            movenet.name if movenet.enabled else "off",
            self._config.camera.width,
            self._config.camera.height,
            self._config.camera.fps,
        )

    def stop(self) -> None:
        """Stop the loop, then release the camera and the DPU runner."""
        with self._lock:
            self._running = False
            self._stop.set()
            camera = self._camera
            self._camera = None
            runner = self._runner
            self._runner = None
            self._requested_model = None
            self._errors = []
            self._camera_error = None
            self._processed.clear()
            self._last_arrival = 0.0
            self._last_dpu_ms = None
        _release_quietly(camera)
        self._release_runner(runner)

    def _reset_still(self) -> None:
        """Forget the still gate's reference frame, as a new camera or model needs."""
        if self._still is not None:
            self._still.reset()
        self._last_objects = []

    def status(self) -> dict[str, Any]:
        """Return the F-K1 ``status`` payload for the current instant."""
        with self._lock:
            running = self._running
            camera_error = self._camera_error
            errors = list(self._errors)
            stamps = tuple(self._processed)
            last_arrival = self._last_arrival
            dpu_ms = self._last_dpu_ms
            started_at = self._started_at
            active_model = self._active_model
            switching = self._switching
            swap_ms = self._swap_ms
        if not running:
            camera = "missing"
        elif camera_error is not None:
            camera = camera_error
        elif last_arrival > 0.0 and (time.monotonic() - last_arrival) <= CAMERA_STALE_S:
            camera = "ok"
        else:
            camera = "missing"
        return protocol.status(
            state="running" if running else "starting",
            camera=camera,
            model=self._model_name(active_model),
            fps=_rate(stamps) if running else None,
            dpu_ms=dpu_ms,
            errors=errors,
            uptime_s=max(0.0, time.monotonic() - started_at) if running else 0.0,
            active_model=active_model,
            switching=switching,
            swap_ms=swap_ms,
        )

    def publish(self, state: str, errors: Sequence[str], camera: str) -> None:
        """Publish one ``status`` snapshot, bypassing the rate limit.

        The process uses this to report a start failure before exiting; a
        running pipeline publishes from :meth:`run`.
        """

        self._status_key = (state, camera, self._active_model, self._switching, tuple(errors))
        self._last_status = time.monotonic()
        self._messages.put(
            "status",
            protocol.status(
                state=state,
                camera=camera,
                model=self._model_name(self._active_model),
                fps=None,
                dpu_ms=None,
                errors=list(errors),
                uptime_s=0.0,
                active_model=self._active_model,
                switching=self._switching,
                swap_ms=self._swap_ms,
            ),
        )

    def set_model(self, model: str) -> bool:
        """Ask the loop to load ``model`` in place of the current one (R2-G).

        The swap itself happens in the loop thread, between frames: it waits for
        the job in flight, releases the current runner and builds the other one,
        so `status.switching` is true while it runs. Asking for the model that
        is already loaded is a no-op, and it also drops a swap still waiting for
        the loop: two quick clicks end on the model of the second one.

        Returns:
            ``False`` for an unknown model or when MoveNet is disabled, so the
            caller can answer ``error bad_command``.
        """

        if model not in protocol.ACTIVE_MODELS:
            return False
        if model == "movenet" and not self._config.movenet.enabled:
            return False
        with self._lock:
            if model == self._active_model and not self._switching:
                # Asking for the loaded model drops a swap still queued for the
                # loop; during a swap it queues the way back instead, so the
                # second of two quick clicks is the one that wins.
                self._requested_model = None
            else:
                self._requested_model = model
        return True

    # -- frame loop --------------------------------------------------------

    def run(self, stop: threading.Event | None = None) -> None:
        """Process frames until ``stop`` is set (or :meth:`stop` is called).

        Args:
            stop: External stop request, the process' signal handler; ``None``
                means only :meth:`stop` ends the loop.
        """

        stop = stop if stop is not None else self._stop
        period = 1.0 / self._config.camera.fps
        next_due = 0.0
        pending: _PendingFrame | None = None
        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(_SWITCH_INTERVAL_S)
        with self._lock:
            self._running = True
            self._started_at = time.monotonic()
            self._timing_since = time.monotonic()
        self._publish_status(force=True)
        try:
            while not stop.is_set() and not self._stop.is_set():
                self._keep_camera_alive(stop)
                self._drain_commands()
                pending = self._apply_model_request(pending)
                frame = self._slot.get(timeout=_SLOT_TIMEOUT_S)
                if frame is not None:
                    admit, next_due = cap_admits(time.perf_counter(), next_due, period)
                    if admit:
                        try:
                            pending = self._step(frame, pending)
                        except Exception:
                            logger.exception("the detector failed on frame %d", frame.seq)
                            self._frame_error = True
                            self._note_error("dpu_error")
                        else:
                            with self._lock:
                                # A frame clears the ``dpu_error`` of a previous
                                # frame, never the one a failed model swap left
                                # behind: only a swap that works clears that.
                                if (
                                    not self._frame_error
                                    and not self._swap_error
                                    and "dpu_error" in self._errors
                                ):
                                    self._errors = [
                                        code for code in self._errors if code != "dpu_error"
                                    ]
                self._publish_status()
        finally:
            with self._lock:
                runner = self._runner
            if pending is not None:
                _drain_quietly(runner, pending.token)
            sys.setswitchinterval(previous_interval)
            self.stop()
            logger.info("showcase stopped")

    def _step(self, frame: VideoFrame, pending: _PendingFrame | None) -> _PendingFrame | None:
        """Start frame N on the DPU, then finish frame N-1 while it runs."""

        with self._lock:
            runner = self._runner
            model = self._active_model
        if runner is None:
            return None
        self._frame_error = False

        started = time.perf_counter()
        rgb = frame.decode()
        camera_ms = (time.perf_counter() - started) * 1000.0

        if model == "yolox" and self._still is not None:
            motion_started = time.perf_counter()
            infer = self._still.update(rgb, time.monotonic())
            motion_ms = (time.perf_counter() - motion_started) * 1000.0
            if not infer:
                # The scene has not moved: the boxes of the last inference are
                # still the truth, so frame N-1 is finished and frame N is drawn
                # from them without touching the DPU.
                self._finish_quietly(pending)
                self._finish_still(frame.width, frame.height, rgb, started, camera_ms, motion_ms)
                return None

        submit = getattr(runner, "submit", None)
        wait = getattr(runner, "wait", None)
        if not callable(submit) or not callable(wait):
            detect = getattr(runner, "detect")  # noqa: B009 - a test double may only have detect
            self._finish(
                frame.seq,
                frame.width,
                frame.height,
                rgb,
                runner,
                model,
                detect(rgb),
                started,
                camera_ms,
                frame.t,
            )
            return None

        token = submit(rgb)
        current = _PendingFrame(
            seq=frame.seq,
            width=frame.width,
            height=frame.height,
            rgb=rgb,
            token=token,
            started=started,
            camera_ms=camera_ms,
            model=model,
            t=frame.t,
        )
        self._finish_quietly(pending)
        return current

    def _finish_quietly(self, pending: _PendingFrame | None) -> None:
        """Finish a frame in flight; a failure costs an error, never its token."""
        if pending is None:
            return
        try:
            self._finish_pending(pending)
        except Exception:
            logger.exception("the detector failed on frame %d", pending.seq)
            self._frame_error = True
            self._note_error("dpu_error")

    def _finish_pending(self, pending: _PendingFrame) -> None:
        """Wait for, draw and publish a frame the DPU has been working on."""

        with self._lock:
            runner = self._runner
        if runner is None:
            return
        if pending.model != self._active_model:
            # A swap drains the frame in flight before it releases a runner, so
            # this is only reachable after an external stop().
            _drain_quietly(runner, pending.token)
            return
        self._finish(
            pending.seq,
            pending.width,
            pending.height,
            pending.rgb,
            runner,
            pending.model,
            runner.wait(pending.token),
            pending.started,
            pending.camera_ms,
            pending.t,
        )

    def _finish(
        self,
        seq: int,
        width: int,
        height: int,
        rgb: NDArray[np.uint8],
        runner: Detector | PoseDetector,
        model: str,
        output: Sequence[Detection] | MovenetPose,
        started: float,
        camera_ms: float,
        timestamp: float,
    ) -> None:
        """Draw, publish and time one decoded frame."""

        track_ms = 0.0
        pose: MovenetPose | None = None
        objects: list[protocol.ShowcaseObject] = []
        if model == "movenet":
            stabilize_started = time.perf_counter()
            pose = self._stabilizer.update(
                cast(MovenetPose, output), timestamp, size=(width, height)
            )
            track_ms = (time.perf_counter() - stabilize_started) * 1000.0
        else:
            detections = cast(Sequence[Detection], output)
            track_started = time.perf_counter()
            tracked = self._tracker.update(detections)
            track_ms = (time.perf_counter() - track_started) * 1000.0
            objects = _objects(tracked, self._labels)

        stage = runner.stage_ms
        self._render(
            width=width,
            height=height,
            rgb=rgb,
            model=model,
            objects=objects,
            pose=pose,
            started=started,
            camera_ms=camera_ms,
            track_ms=track_ms,
            stage=stage,
            dpu_ms=stage.get("dpu"),
            skipped=False,
        )

    def _finish_still(
        self,
        width: int,
        height: int,
        rgb: NDArray[np.uint8],
        started: float,
        camera_ms: float,
        motion_ms: float,
    ) -> None:
        """Draw the last inferred boxes on a frame the still gate kept off the DPU.

        The tracker is not touched: its boxes, labels and scores are the ones the
        last real inference produced, so the overlay and the published objects
        are the same pixels and the same counts as that frame. Only the frame
        itself is new, which is what keeps the video live while nothing moves.
        """

        with self._lock:
            last_dpu = self._last_dpu_ms
        self._render(
            width=width,
            height=height,
            rgb=rgb,
            model="yolox",
            objects=self._last_objects,
            pose=None,
            started=started,
            camera_ms=camera_ms,
            track_ms=0.0,
            stage={"motion": motion_ms},
            dpu_ms=last_dpu,
            skipped=True,
        )

    def _render(
        self,
        *,
        width: int,
        height: int,
        rgb: NDArray[np.uint8],
        model: str,
        objects: Sequence[protocol.ShowcaseObject],
        pose: MovenetPose | None,
        started: float,
        camera_ms: float,
        track_ms: float,
        stage: Mapping[str, float],
        dpu_ms: float | None,
        skipped: bool,
    ) -> None:
        """Draw the overlay, publish the video and the states, record the timings."""

        draw_started = time.perf_counter()
        if pose is not None:
            overlay = draw_skeleton(rgb, pose.keypoints.tolist())
        else:
            overlay = draw_overlay(rgb, objects)
        draw_ms = (time.perf_counter() - draw_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        self._channel.write(overlay)
        now = time.monotonic()
        with self._lock:
            self._processed.append(time.perf_counter())
            self._last_dpu_ms = None if dpu_ms is None else float(dpu_ms)
            if model == "yolox" and not skipped:
                self._last_objects = list(objects)
            stamps = tuple(self._processed)
            last_dpu = self._last_dpu_ms
            publish_state = model == "yolox" and now - self._last_state >= STATE_PERIOD_S
            pose_period = _pose_period(self._config.movenet.publish_hz)
            publish_pose = pose is not None and now - self._last_pose >= pose_period
            if publish_state:
                self._last_state = now
            if publish_pose:
                self._last_pose = now
        self._record_timings(
            {
                "camera": camera_ms,
                "preprocess": stage.get("preprocess", 0.0),
                "dpu": stage.get("dpu", 0.0),
                "decode": stage.get("decode", 0.0),
                "motion": stage.get("motion", 0.0),
                "track": track_ms,
                "draw": draw_ms,
                "total": total_ms,
                "skipped": 1.0 if skipped else 0.0,
            },
            stamps,
        )
        # F-E1: each state belongs to the model that produced it -- the objects
        # pane only receives showcase.state while YOLOX is loaded, and the
        # skeleton only pose.state while MoveNet is loaded.
        if publish_state:
            self._messages.put(
                "showcase.state",
                protocol.showcase_state(
                    self._config.model.name,
                    width,
                    height,
                    objects,
                    fps=_rate(stamps),
                    dpu_ms=last_dpu,
                    total_ms=total_ms,
                ),
            )
        if publish_pose and pose is not None:
            self._messages.put(
                "pose.state",
                protocol.pose_state(
                    self._config.movenet.name,
                    width,
                    height,
                    pose.keypoints.tolist(),
                    score=pose.score,
                    fps=_rate(stamps),
                    dpu_ms=stage.get("dpu"),
                    total_ms=total_ms,
                ),
            )

    # -- commands ----------------------------------------------------------

    def _drain_commands(self) -> None:
        """Apply the ``model.set`` commands the server queued (R2-G)."""
        channel = self._commands
        if channel is None:
            return
        for topic, data in channel.drain():
            if topic != "model.set":
                logger.warning("ignoring unknown pipeline command %r", topic)
                continue
            model = data.get("model")
            if not isinstance(model, str) or not self.set_model(model):
                logger.warning("ignoring model.set with model %r", model)

    # -- model swap --------------------------------------------------------

    def _apply_model_request(self, pending: _PendingFrame | None) -> _PendingFrame | None:
        """Perform a queued ``model.set`` between frames (R2-G)."""

        with self._lock:
            requested = self._requested_model
            self._requested_model = None
            active = self._active_model
        if requested is None or requested == active:
            return pending
        if pending is not None:
            # The job in flight belongs to the runner about to be released:
            # wait for it (the frame is dropped, its model is going away) so the
            # DPU is idle and its buffers are free before the graph disappears.
            _drain_quietly(self._runner, pending.token)
            pending = None
        self._swap_model(requested)
        return None

    def _swap_model(self, target: str) -> None:
        """Release the loaded runner and build ``target`` in its place.

        A failed load restores the previous runner, so a bad ``model.set``
        never leaves the showcase without a model.
        """

        with self._lock:
            previous = self._active_model
        logger.info("model swap: %s -> %s", previous, target)
        with self._lock:
            self._switching = True
        self._publish_status(force=True)

        started = time.perf_counter()
        with self._lock:
            old = self._runner
            self._runner = None
        self._release_runner(old)

        loaded = target
        failed = False
        try:
            runner: Detector | PoseDetector | None = self._build_runner(target)
        except Exception:
            logger.exception("the %s runner could not be loaded; restoring %s", target, previous)
            failed = True
            loaded = previous
            try:
                runner = self._build_runner(previous)
            except Exception:
                logger.exception("the %s runner could not be restored", previous)
                runner = None

        swap_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._runner = runner
            self._active_model = loaded
            self._switching = False
            self._swap_ms = swap_ms
            self._swap_error = failed
            self._frame_error = False
            self._last_state = 0.0
            self._last_pose = 0.0
            if loaded == "yolox":
                self._tracker.reset()
                self._reset_still()
            self._stabilizer.reset()
        if failed:
            # Stays in ``status.errors`` until a swap succeeds: frames keep
            # coming from the restored model, so they must not clear it.
            self._note_error("dpu_error")
        self._publish_status(force=True)
        logger.info("model swap to %s done in %.1f ms", loaded, swap_ms)

    def _build_runner(self, model: str) -> Detector | PoseDetector:
        """Build the runner of ``model`` from the configuration."""

        if model == "movenet":
            config = self._config.movenet
            if not config.enabled:
                raise ValueError("movenet is disabled in the configuration")
            return self._movenet_factory(config)
        return self._detector_factory(self._config.model)

    def _release_runner(self, runner: Detector | PoseDetector | None) -> None:
        """Release a runner's graph and buffers without masking a failure."""

        if runner is None:
            return
        close = getattr(runner, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            logger.exception("failed to release the DPU runner")

    def _model_name(self, model: str) -> str:
        """The ``.xmodel`` stem of the loaded model."""

        if model == "movenet":
            return self._config.movenet.name
        return self._config.model.name

    # -- camera recovery ---------------------------------------------------

    def _ingest(self, frame: VideoFrame) -> None:
        """Keep only the newest frame: a slow loop never builds a backlog."""
        with self._lock:
            self._last_arrival = time.monotonic()
        self._slot.put(frame)

    def _camera_failed(self, error: Exception) -> None:
        """Record a capture failure reported by the camera thread."""
        camera = "missing" if isinstance(error, CameraNotFound) else "error"
        logger.error("the camera stopped (%s): %s", camera, error)
        with self._lock:
            self._camera_error = camera
            self._next_camera_attempt = time.monotonic() + self._camera_retry_s
            self._camera_retry_s = min(self._camera_retry_s * 2.0, CAMERA_RETRY_MAX_S)
            self._add_error(_CAMERA_CODES[camera])

    def _keep_camera_alive(self, stop: threading.Event) -> None:
        """Reopen the camera after a stop or a re-plug, with a capped backoff."""

        with self._lock:
            camera = self._camera
            camera_error = self._camera_error
            ready_at = self._next_camera_attempt
        if camera_error is None or camera is None:
            return
        if camera.running:
            return
        if time.monotonic() < ready_at or stop.is_set():
            return
        try:
            camera.start(self._ingest, self._camera_failed)
        except CameraError as exc:
            logger.warning("the camera is still unavailable: %s", exc)
            with self._lock:
                self._next_camera_attempt = time.monotonic() + self._camera_retry_s
                self._camera_retry_s = min(self._camera_retry_s * 2.0, CAMERA_RETRY_MAX_S)
                self._camera_error = "missing" if isinstance(exc, CameraNotFound) else "error"
                self._add_error(_CAMERA_CODES[self._camera_error])
            return
        logger.info("the camera was reopened")
        with self._lock:
            self._camera_error = None
            self._camera_retry_s = CAMERA_RETRY_S
            self._last_arrival = time.monotonic()
            self._errors = [e for e in self._errors if e not in _CAMERA_CODES.values()]

    # -- publishing and timing ---------------------------------------------

    def _publish_status(self, *, force: bool = False) -> None:
        """Publish ``status`` at 1 Hz and whenever the situation changes."""
        payload = self.status()
        key: tuple[str, str, str, bool, tuple[str, ...]] = (
            str(payload["state"]),
            str(payload["camera"]),
            str(payload["active_model"]),
            bool(payload["switching"]),
            tuple(str(code) for code in payload["errors"]),
        )
        now = time.monotonic()
        with self._lock:
            last = self._last_status
            if not force and key == self._status_key and now - last < STATUS_PERIOD_S:
                return
            self._last_status = now
            self._status_key = key
        self._messages.put("status", payload)

    def _note_error(self, code: str) -> None:
        with self._lock:
            self._add_error(code)

    def _add_error(self, code: str) -> None:
        """Append an error code once, preserving order (lock held by the caller)."""
        if code not in self._errors:
            self._errors.append(code)

    def _record_timings(self, samples: Mapping[str, float], stamps: Sequence[float]) -> None:
        """Accumulate the per-stage timings and log the means once every ~5 s."""
        with self._lock:
            for name, milliseconds in samples.items():
                self._timing_sums[name] = self._timing_sums.get(name, 0.0) + float(milliseconds)
            self._timing_frames += 1
            elapsed = time.monotonic() - self._timing_since
            if elapsed < TIMING_PERIOD_S:
                return
            means = {name: total / self._timing_frames for name, total in self._timing_sums.items()}
            counted = self._timing_frames
            self._timing_sums = {}
            self._timing_frames = 0
            self._timing_since = time.monotonic()
        logger.info(
            "showcase stages over %d frames: camera=%.2f preprocess=%.2f dpu=%.2f "
            "decode=%.2f motion=%.3f track=%.2f draw=%.2f total=%.2f skipped=%.0f%% "
            "total_fps=%.1f",
            counted,
            means.get("camera", 0.0),
            means.get("preprocess", 0.0),
            means.get("dpu", 0.0),
            means.get("decode", 0.0),
            means.get("motion", 0.0),
            means.get("track", 0.0),
            means.get("draw", 0.0),
            means.get("total", 0.0),
            100.0 * means.get("skipped", 0.0),
            _rate(stamps),
        )


def _objects(
    detections: Sequence[Detection],
    labels: Sequence[str],
) -> list[protocol.ShowcaseObject]:
    """Map the detector output to the labelled objects the wire and overlay use."""
    return [
        protocol.ShowcaseObject(
            label=label_for(int(detection.class_id), labels),
            class_id=int(detection.class_id),
            score=float(detection.score),
            box=(
                float(detection.box[0]),
                float(detection.box[1]),
                float(detection.box[2]),
                float(detection.box[3]),
            ),
        )
        for detection in detections
    ]


def _rate(stamps: Sequence[float]) -> float:
    """Rolling frames per second over the recorded timestamps; ``0.0`` when unknown."""
    if len(stamps) < 2 or stamps[-1] <= stamps[0]:
        return 0.0
    return (len(stamps) - 1) / (stamps[-1] - stamps[0])


def _pose_period(publish_hz: float) -> float:
    """Seconds between two ``pose.state`` messages; never longer than the frame."""
    return 1.0 / max(float(publish_hz), 1e-6)


def _default_detector(model: ModelConfig) -> Detector:
    """Build the DPU detector of F-S1 (imported here: it binds the VART runtime)."""
    return YoloxDetector(
        model.path,
        conf_thr=model.conf_thr,
        iou_thr=model.iou_thr,
        class_thr=class_thresholds(
            model.conf_thr,
            class_conf=model.class_conf,
            object_conf=model.object_conf,
        ),
    )


def _default_movenet(config: MovenetConfig) -> PoseDetector:
    """Build the MoveNet runner of F-E2 (it binds the VART runtime too)."""
    return DpuMovenet(config.path, hide_score=config.smoothing.hide_score)


def _default_camera(config: CameraConfig) -> Camera:
    """Build the UVC camera of F-K3."""
    return UvcCamera(config)


def _release_quietly(camera: Camera | None) -> None:
    """Release a camera without masking the failure that is being reported."""
    if camera is None:
        return
    try:
        camera.stop()
    except Exception:
        logger.exception("failed to release the camera")


def _drain_quietly(detector: Detector | PoseDetector | None, token: int) -> None:
    """Wait for a submitted DPU job so the buffers are free before release."""
    if detector is None:
        return
    try:
        detector.wait(token)
    except Exception:
        logger.warning("the last DPU job did not finish cleanly", exc_info=True)
