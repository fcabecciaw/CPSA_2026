"""The detection loop: camera frames in, overlaid video and states out.

One loop processes frames from a :class:`~cpsa.camera.UvcCamera` with two DPU
models: :class:`~cpsa.yolox.YoloxDetector` for the boxes and
:class:`~cpsa.movenet.DpuMovenet` for the 17-keypoint skeleton (R2-G). Both run
on every frame, each with its own two-slot pipelining, and the tracker stabilizes
the YOLOX boxes. The loop publishes the overlay of the active view into the
shared :class:`~cpsa.channel.FrameChannel` and the snapshot messages into the
:class:`~cpsa.channel.MessageChannel`; ``model.set`` arrives the other way
through the :class:`~cpsa.channel.CommandChannel`.

The 60 fps target needs the DPU and the CPU to work on different frames:

```text
submit YOLOX and MoveNet for frame N (preprocess + execute_async)
decode + draw + publish frame N-1   (while the DPU runs on N)
next iteration: submit N+1, then wait N
```

The detectors expose that split as ``submit``/``wait``; a detector with only
``detect`` (a fake in a test, a CPU stand-in) runs the simple synchronous path.
The camera slot keeps only the newest frame, so a slow frame is dropped, never
queued.

The loop also owns the process status: ``status`` once per second and on change
(state, camera, errors, ``active_model``) into the message channel,
``showcase.state`` at most ten times per second while the YOLOX view is active,
``pose.state`` at most ``movenet.publish_hz`` times per second, and the per-stage
timing line every five seconds. A camera that stops (unplugged, a driver error)
is reopened with a capped backoff, which is how a re-plug recovers without
restarting the process.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cpsa import protocol
from cpsa.camera import CameraError, CameraNotFound, UvcCamera, VideoFrame
from cpsa.channel import CommandChannel, FrameChannel, MessageChannel
from cpsa.config import CameraConfig, ModelConfig, MovenetConfig, ShowcaseConfig
from cpsa.movenet import DpuMovenet, MovenetPose
from cpsa.overlay import draw_overlay, draw_skeleton
from cpsa.tracker import Tracker
from cpsa.yolox import Detection, YoloxDetector, label_for, labels_for

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
    movenet_token: int | None = None
    """MoveNet job of the same frame, ``None`` when the pose model is off."""


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
    """Camera, detector, overlay and status of the showcase.

    Args:
        config: The validated F-K3 configuration.
        channel: The shared frame the server shows, already attached.
        messages: The message channel the server drains.
        monitor: Where ``status`` and ``showcase.state`` go; defaults to
            ``messages``. The process gives the pipeline both the same channel
            and keeps the interface separate for tests.
        detector_factory: Builds the detector from the model config; defaults
            to :class:`~cpsa.yolox.YoloxDetector`.
        movenet_factory: Builds the MoveNet runner from the ``movenet`` config;
            defaults to :class:`~cpsa.movenet.DpuMovenet`.
        camera_factory: Builds the camera from the camera config; defaults to
            :class:`~cpsa.camera.UvcCamera`.
        commands: The control channel the server writes ``model.set`` into; the
            loop drains it every frame. ``None`` means the view never switches.
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
        self._lock = threading.Lock()
        self._slot = _LatestSlot()
        self._stop = threading.Event()
        self._detector: Detector | None = None
        self._pose_detector: PoseDetector | None = None
        self._camera: Camera | None = None
        self._running = False
        self._camera_error: str | None = None
        self._errors: list[str] = []
        self._processed: deque[float] = deque(maxlen=_RATE_WINDOW)
        self._last_arrival = 0.0
        self._last_dpu_ms: float | None = None
        self._last_state = 0.0
        self._last_pose = 0.0
        self._pose_failed = False
        self._active_model = protocol.ACTIVE_MODELS[0]
        self._last_status = 0.0
        self._status_key: tuple[str, str, str, tuple[str, ...]] | None = None
        self._started_at = 0.0
        self._next_camera_attempt = 0.0
        self._camera_retry_s = CAMERA_RETRY_S
        self._timing_sums: dict[str, float] = {}
        self._timing_frames = 0
        self._timing_since = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Open the camera and both DPU models.

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
        try:
            detector = self._detector_factory(model)
        except FileNotFoundError as exc:
            raise StartError("model_missing", str(exc)) from exc
        except RuntimeError as exc:
            raise StartError("dpu_error", str(exc)) from exc
        except Exception as exc:
            logger.exception("the detector could not be built")
            raise StartError("internal", str(exc)) from exc

        pose_detector: PoseDetector | None = None
        movenet = self._config.movenet
        if movenet.enabled:
            if not movenet.path.is_file():
                raise StartError("model_missing", f"movenet model not found: {movenet.path}")
            try:
                pose_detector = self._movenet_factory(movenet)
            except FileNotFoundError as exc:
                raise StartError("model_missing", str(exc)) from exc
            except RuntimeError as exc:
                raise StartError("dpu_error", str(exc)) from exc
            except Exception as exc:
                logger.exception("the MoveNet runner could not be built")
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
            raise StartError("camera_error", str(exc)) from exc

        with self._lock:
            self._detector = detector
            self._pose_detector = pose_detector
            self._camera = camera
            self._started_at = time.monotonic()
            self._last_arrival = time.monotonic()
            self._processed.clear()
            self._errors = []
            self._camera_error = None
            self._status_key = None
            self._tracker.reset()
            self._active_model = protocol.ACTIVE_MODELS[0]
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
        """Stop the loop, then release the camera and the detectors."""
        with self._lock:
            self._running = False
            self._stop.set()
            camera = self._camera
            self._camera = None
            self._detector = None
            self._pose_detector = None
            self._errors = []
            self._camera_error = None
            self._processed.clear()
            self._last_arrival = 0.0
            self._last_dpu_ms = None
        _release_quietly(camera)

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
            model=self._config.model.name,
            fps=_rate(stamps) if running else None,
            dpu_ms=dpu_ms,
            errors=errors,
            uptime_s=max(0.0, time.monotonic() - started_at) if running else 0.0,
            active_model=active_model,
        )

    def publish(self, state: str, errors: Sequence[str], camera: str) -> None:
        """Publish one ``status`` snapshot, bypassing the rate limit.

        The process uses this to report a start failure before exiting; a
        running pipeline publishes from :meth:`run`.
        """

        self._status_key = (state, camera, self._active_model, tuple(errors))
        self._last_status = time.monotonic()
        self._messages.put(
            "status",
            protocol.status(
                state=state,
                camera=camera,
                model=self._config.model.name,
                fps=None,
                dpu_ms=None,
                errors=list(errors),
                uptime_s=0.0,
                active_model=self._active_model,
            ),
        )

    def set_model(self, model: str) -> bool:
        """Switch the dashboard view; both models keep running (R2-G).

        ``yolox`` draws the tracked boxes, ``movenet`` the glowing skeleton, and
        ``status.active_model`` tells the dashboard which pane to show. This
        never stops or restarts a DPU runner.

        Returns:
            ``False`` for an unknown model or when MoveNet is disabled, so the
            caller can answer ``error bad_command``.
        """

        if model not in protocol.ACTIVE_MODELS:
            return False
        with self._lock:
            if model == "movenet" and self._pose_detector is None:
                return False
            self._active_model = model
        self._publish_status(force=True)
        logger.info("dashboard view switched to %s", model)
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
                frame = self._slot.get(timeout=_SLOT_TIMEOUT_S)
                if frame is not None:
                    admit, next_due = cap_admits(time.perf_counter(), next_due, period)
                    if admit:
                        try:
                            pending = self._step(frame, pending)
                        except Exception:
                            logger.exception("the detector failed on frame %d", frame.seq)
                            self._note_error("dpu_error")
                        else:
                            with self._lock:
                                if not self._pose_failed and "dpu_error" in self._errors:
                                    self._errors = [
                                        code for code in self._errors if code != "dpu_error"
                                    ]
                self._publish_status()
        finally:
            if pending is not None:
                _drain_quietly(self._detector, pending.token)
                _drain_pose_quietly(self._pose_detector, pending.movenet_token)
            sys.setswitchinterval(previous_interval)
            self.stop()
            logger.info("showcase stopped")

    def _step(self, frame: VideoFrame, pending: _PendingFrame | None) -> _PendingFrame | None:
        """Start frame N on the DPU, then finish frame N-1 while it runs."""

        with self._lock:
            detector = self._detector
            pose_detector = self._pose_detector
        if detector is None:
            return None
        self._pose_failed = False

        started = time.perf_counter()
        rgb = frame.decode()
        camera_ms = (time.perf_counter() - started) * 1000.0

        submit = getattr(detector, "submit", None)
        wait = getattr(detector, "wait", None)
        if not callable(submit) or not callable(wait):
            detections = detector.detect(rgb)  # type: ignore[attr-defined]
            pose = None
            if pose_detector is not None:
                pose = pose_detector.wait(pose_detector.submit(rgb))
            self._finish(
                frame.seq,
                frame.width,
                frame.height,
                rgb,
                detections,
                pose,
                detector.stage_ms,
                _stage(pose_detector),
                started,
                camera_ms,
            )
            return None

        token = submit(rgb)
        movenet_token: int | None = None
        if pose_detector is not None:
            try:
                movenet_token = pose_detector.submit(rgb)
            except Exception:
                # A MoveNet failure must never strand the YOLOX token, which
                # still has to be waited for on the next frame.
                logger.exception("MoveNet failed to submit frame %d", frame.seq)
                self._pose_failed = True
                self._note_error("dpu_error")
        current = _PendingFrame(
            seq=frame.seq,
            width=frame.width,
            height=frame.height,
            rgb=rgb,
            token=token,
            started=started,
            camera_ms=camera_ms,
            movenet_token=movenet_token,
        )
        if pending is not None:
            # A failure while finishing frame N-1 must not cost frame N its
            # token, or the slot would stay busy for ever.
            try:
                self._finish_pending(detector, pose_detector, pending)
            except Exception:
                logger.exception("the detector failed on frame %d", pending.seq)
        return current

    def _finish_pending(
        self,
        detector: Detector,
        pose_detector: PoseDetector | None,
        pending: _PendingFrame,
    ) -> None:
        """Wait for, draw and publish a frame the DPU has been working on."""
        # MoveNet is submitted after YOLOX, so waiting for it first also waits
        # for the YOLOX job; each wait is isolated so one failure cannot leave
        # the other runner's slot busy.
        pose: MovenetPose | None = None
        if pose_detector is not None and pending.movenet_token is not None:
            try:
                pose = pose_detector.wait(pending.movenet_token)
            except Exception:
                logger.exception("MoveNet failed on frame %d", pending.seq)
                self._pose_failed = True
                self._note_error("dpu_error")
        detections = detector.wait(pending.token)
        self._finish(
            pending.seq,
            pending.width,
            pending.height,
            pending.rgb,
            detections,
            pose,
            detector.stage_ms,
            _stage(pose_detector),
            pending.started,
            pending.camera_ms,
        )

    def _finish(
        self,
        seq: int,
        width: int,
        height: int,
        rgb: NDArray[np.uint8],
        detections: Sequence[Detection],
        pose: MovenetPose | None,
        stage: Mapping[str, float],
        pose_stage: Mapping[str, float],
        started: float,
        camera_ms: float,
    ) -> None:
        """Draw, publish and time one decoded frame."""

        track_started = time.perf_counter()
        tracked = self._tracker.update(detections)
        track_ms = (time.perf_counter() - track_started) * 1000.0
        objects = _objects(tracked, self._labels)
        with self._lock:
            active_model = self._active_model
        draw_started = time.perf_counter()
        if active_model == "movenet" and pose is not None:
            overlay = draw_skeleton(rgb, pose.keypoints.tolist())
        else:
            overlay = draw_overlay(rgb, objects)
        draw_ms = (time.perf_counter() - draw_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        dpu_ms = stage.get("dpu")
        self._channel.write(overlay)
        now = time.monotonic()
        with self._lock:
            self._processed.append(time.perf_counter())
            self._last_dpu_ms = None if dpu_ms is None else float(dpu_ms)
            stamps = tuple(self._processed)
            last_dpu = self._last_dpu_ms
            active_model = self._active_model
            publish_state = now - self._last_state >= STATE_PERIOD_S
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
                "movenet_dpu": pose_stage.get("dpu", 0.0),
                "movenet_decode": pose_stage.get("decode", 0.0),
                "track": track_ms,
                "draw": draw_ms,
                "total": total_ms,
            },
            stamps,
        )
        # F-E1: the objects pane only receives states while YOLOX is the view;
        # R2-G keeps the pose flowing in both views so gestures never stop.
        if publish_state and active_model == "yolox":
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
                    dpu_ms=pose_stage.get("dpu"),
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
        key: tuple[str, str, str, tuple[str, ...]] = (
            str(payload["state"]),
            str(payload["camera"]),
            str(payload["active_model"]),
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
            "decode=%.2f movenet_dpu=%.2f movenet_decode=%.2f track=%.2f draw=%.2f "
            "total=%.2f total_fps=%.1f",
            counted,
            means.get("camera", 0.0),
            means.get("preprocess", 0.0),
            means.get("dpu", 0.0),
            means.get("decode", 0.0),
            means.get("movenet_dpu", 0.0),
            means.get("movenet_decode", 0.0),
            means.get("track", 0.0),
            means.get("draw", 0.0),
            means.get("total", 0.0),
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


def _stage(detector: PoseDetector | None) -> Mapping[str, float]:
    """The stage timings of a MoveNet runner, or zeros when it is off."""
    return detector.stage_ms if detector is not None else _NO_STAGE


_NO_STAGE: Mapping[str, float] = {"preprocess": 0.0, "dpu": 0.0, "decode": 0.0}


def _default_detector(model: ModelConfig) -> Detector:
    """Build the DPU detector of F-S1 (imported here: it binds the VART runtime)."""
    return YoloxDetector(model.path, conf_thr=model.conf_thr, iou_thr=model.iou_thr)


def _default_movenet(config: MovenetConfig) -> PoseDetector:
    """Build the MoveNet runner of F-E2 (it binds the VART runtime too)."""
    return DpuMovenet(config.path)


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


def _drain_quietly(detector: Detector | None, token: int) -> None:
    """Wait for the last submitted DPU job so the buffers are free before release."""
    if detector is None:
        return
    try:
        detector.wait(token)
    except Exception:
        logger.warning("the last DPU job did not finish cleanly", exc_info=True)


def _drain_pose_quietly(pose_detector: PoseDetector | None, token: int | None) -> None:
    """Wait for the last MoveNet job so its buffers are free before release."""
    if pose_detector is None or token is None:
        return
    try:
        pose_detector.wait(token)
    except Exception:
        logger.warning("the last MoveNet job did not finish cleanly", exc_info=True)
