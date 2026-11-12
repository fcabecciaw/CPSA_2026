"""The ``showcase`` supervisor mode: "Cosa vede l'IA" (F-S2, F-S3).

One pre-trained model-zoo detector (YOLOX-nano, COCO 80 classes) runs on the
KV260 DPU over the F200 RGB camera. Every processed frame is drawn with its
boxes and labels and offered to ``/video.mjpeg`` (F14, left pane), and the
detections are published as ``showcase.state`` at most 10 times per second
(right pane). The overlay is drawn here, on the board, so the video and the boxes
can never drift apart however fast the browser is.

The RGB camera settings are the ones of ``vision.camera``, the same node the
medical mode uses: one booth setup, no duplicated camera section. A missing model
file fails the start with ``model_missing`` before the camera is touched; a
missing RGB node fails it with ``camera_missing``; a node that is present but
cannot be opened or configured fails it with ``camera_error``. :meth:`stop`
always releases the camera and the DPU runner.

The detector, the camera and the label table arrive as factories: tests inject a
fake, and the defaults (the DPU detector of F-S1 and the COCO tables of
:mod:`cpsa.vision.coco`) are imported only when the mode starts, so booting the
supervisor never loads the VART bindings. The types are structural on purpose: a
detector exposing ``detect``/``stage_ms`` as
:class:`~cpsa.vision.detection.ObjectDetector` specifies them satisfies
:class:`ShowcaseDetector` without this module importing the adapter.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from cpsa.application.config import (
    Config,
    ShowcaseConfig,
    ShowcaseModelConfig,
    VisionCameraConfig,
)
from cpsa.application.slot import LatestSlot
from cpsa.protocol.showcase import ShowcaseObject, showcase_state
from cpsa.supervisor.contract import ModeContext, ModeStartError
from cpsa.vision.types import VideoFrame

logger = logging.getLogger("cpsa.board.showcase")

__all__ = ["STATE_PERIOD_S", "ShowcaseMode", "class_color", "draw_overlay"]

STATE_PERIOD_S = 0.1
"""``showcase.state`` is published at most 10 Hz; the video is not throttled."""

CAMERA_STALE_S = 1.0
"""``health()["camera"]`` is not ``ok`` when no frame arrived for this long."""

_RATE_WINDOW = 30
"""Frames the rolling ``fps`` is measured over."""

_SLOT_TIMEOUT_S = 0.1
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1
_LABEL_PADDING = 3
_LABEL_GAP = 4

_CAMERA_CODES = {"camera_missing": "missing", "camera_error": "error"}

_PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (245, 130, 48),
    (0, 130, 200),
    (240, 50, 230),
    (0, 128, 128),
    (170, 110, 40),
    (145, 30, 180),
    (70, 240, 240),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 0),
)
"""Box colours in RGB, one per class modulo the table length: stable across frames."""


class ShowcaseDetection(Protocol):
    """One detection as the mode reads it (the F-S1 :class:`Detection` shape).

    The fields are read-only properties so a frozen dataclass satisfies them.
    """

    @property
    def box(self) -> tuple[float, float, float, float]:
        """``x1, y1, x2, y2`` in source-image pixels."""

    @property
    def score(self) -> float:
        """Confidence in ``0..1``."""

    @property
    def class_id(self) -> int:
        """COCO-80 class index; 0 is ``person``."""


class ShowcaseDetector(Protocol):
    """The part of :class:`~cpsa.vision.detection.ObjectDetector` the mode drives."""

    @property
    def stage_ms(self) -> Mapping[str, float]:
        """Per-stage milliseconds of the last :meth:`detect` call."""

    def detect(self, rgb: NDArray[np.uint8]) -> Sequence[ShowcaseDetection]:
        """Return the detections of one RGB frame in source pixels, best first."""


class ShowcaseSource(Protocol):
    """The part of an RGB video source the mode drives."""

    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None: ...

    def stop(self) -> None: ...


SourceFactory = Callable[[VisionCameraConfig], ShowcaseSource]
DetectorFactory = Callable[[ShowcaseModelConfig], ShowcaseDetector]
LabelsFactory = Callable[[str], tuple[str, ...]]


def _default_source(camera: VisionCameraConfig) -> ShowcaseSource:
    """Build the RGB source of the booth camera (``vision.camera``)."""
    from cpsa.adapters.camera.uvc import UvcVideoSource

    return UvcVideoSource(
        camera.device,
        width=camera.width,
        height=camera.height,
        fps=camera.fps,
        exposure=camera.exposure,
        gain=camera.gain,
        power_line_hz=camera.power_line_hz,
        match=camera.match,
    )


def _default_detector(model: ShowcaseModelConfig) -> ShowcaseDetector:
    """Build the F-S1 DPU detector of ``model`` (imported when the mode starts).

    The import is deferred to the first start so that booting the supervisor,
    listing the modes or running a different mode never loads the VART bindings.
    """
    adapter = import_module("cpsa.adapters.detect.yolox_dpu")
    detector: ShowcaseDetector = adapter.DpuYoloxDetector(
        model.path, conf_thr=model.conf_thr, iou_thr=model.iou_thr
    )
    return detector


def _default_labels(language: str) -> tuple[str, ...]:
    """Return the COCO-80 label table of the configured language."""
    coco = import_module("cpsa.vision.coco")
    labels: tuple[str, ...] = coco.COCO80_EN if language == "en" else coco.COCO80_IT
    return labels


def _require_showcase(config: Config) -> ShowcaseConfig:
    """Return the ``showcase`` section the mode needs to exist at all."""
    if config.showcase is None:
        raise ValueError("the showcase mode needs a 'showcase' configuration section")
    return config.showcase


class ShowcaseMode:
    """Run one COCO object detector over the RGB camera and publish what it sees."""

    name = "showcase"

    def __init__(
        self,
        config: Config,
        *,
        source_factory: SourceFactory | None = None,
        detector_factory: DetectorFactory | None = None,
        labels_factory: LabelsFactory | None = None,
    ) -> None:
        self._config = config
        self._showcase = _require_showcase(config)
        self._source_factory: SourceFactory = (
            source_factory if source_factory is not None else _default_source
        )
        self._detector_factory: DetectorFactory = (
            detector_factory if detector_factory is not None else _default_detector
        )
        self._labels_factory: LabelsFactory = (
            labels_factory if labels_factory is not None else _default_labels
        )
        self._lock = threading.Lock()
        self._source: ShowcaseSource | None = None
        self._detector: ShowcaseDetector | None = None
        self._ctx: ModeContext | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._slot: LatestSlot[VideoFrame] = LatestSlot()
        self._labels: tuple[str, ...] = ()
        self._running = False
        self._camera_error: str | None = None
        self._errors: list[str] = []
        self._processed: deque[float] = deque(maxlen=_RATE_WINDOW)
        self._last_arrival = 0.0
        self._last_dpu_ms: float | None = None
        self._last_publish = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self, ctx: ModeContext) -> None:
        """Open the RGB camera and the detector, then return once frames flow.

        Raises :class:`ModeStartError` with ``model_missing``, ``camera_missing``,
        ``camera_error``, ``dpu_error``, ``busy`` or ``internal``; a failed start
        releases whatever it acquired.
        """
        with self._lock:
            if self._running:
                raise ModeStartError("busy", "showcase mode is already running")
            self._camera_error = None
            self._errors = []
        if ctx.stop_event.is_set():
            logger.info("Stop requested before showcase mode started")
            return

        showcase = self._showcase
        model = showcase.model()
        if not model.path.is_file():
            self._fail("model_missing")
            raise ModeStartError("model_missing", f"showcase model not found: {model.path}")
        try:
            detector = self._detector_factory(model)
        except FileNotFoundError as exc:
            self._fail("model_missing")
            raise ModeStartError("model_missing", str(exc)) from exc
        except RuntimeError as exc:
            self._fail("dpu_error")
            raise ModeStartError("dpu_error", str(exc)) from exc
        except Exception as exc:
            logger.exception("showcase mode could not build the detector")
            self._fail("internal")
            raise ModeStartError("internal", str(exc)) from exc
        try:
            labels = tuple(self._labels_factory(showcase.labels))
        except Exception as exc:
            logger.exception("showcase mode could not load the label table")
            self._fail("internal")
            raise ModeStartError("internal", str(exc)) from exc

        camera = self._config.vision.camera
        try:
            source = self._source_factory(camera)
        except Exception as exc:
            logger.exception("showcase mode could not build the RGB source")
            self._fail("camera_error")
            raise ModeStartError("camera_error", str(exc)) from exc

        with self._lock:
            self._stop = threading.Event()
            self._slot = LatestSlot()
            self._processed.clear()
            self._last_arrival = 0.0
            self._last_dpu_ms = None
            self._last_publish = 0.0
        try:
            source.start(self._ingest, self._camera_failed)
        except FileNotFoundError as exc:
            _release_quietly(source)
            self._fail("camera_missing")
            raise ModeStartError("camera_missing", str(exc)) from exc
        except Exception as exc:
            logger.exception("showcase mode could not open the RGB camera")
            _release_quietly(source)
            self._fail("camera_error")
            raise ModeStartError("camera_error", str(exc)) from exc

        thread = threading.Thread(
            target=self._run,
            name="cpsa-showcase",
            daemon=True,
        )
        with self._lock:
            self._source = source
            self._detector = detector
            self._labels = labels
            self._ctx = ctx
            self._thread = thread
            self._running = True
        self._publish(
            ctx,
            width=camera.width,
            height=camera.height,
            objects=(),
            fps=None,
            dpu_ms=None,
            total_ms=None,
        )
        try:
            thread.start()
        except Exception as exc:
            self._fail("internal")
            self._release()
            raise ModeStartError("internal", str(exc)) from exc
        if ctx.stop_event.is_set():
            # A stop requested while the camera opened must release it now (F4).
            logger.info("Stop requested while showcase mode was starting")
            self.stop(0.0)
            return
        logger.info("Showcase mode started with model %r", model.name)

    def stop(self, timeout_s: float) -> bool:
        """Stop the loop and release the camera and the DPU runner."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._lock:
            running = self._running
            self._running = False
            self._stop.set()
            thread = self._thread
        stopped = True
        if running and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            stopped = not thread.is_alive()
            if not stopped:
                logger.warning("showcase mode did not stop within %.2f s", timeout_s)
        with self._lock:
            source = self._source
            self._source = None
            self._detector = None
            self._ctx = None
            self._thread = None
            self._labels = ()
            self._camera_error = None
            self._errors = []
            self._processed.clear()
            self._last_arrival = 0.0
            self._last_dpu_ms = None
            self._last_publish = 0.0
        _release_quietly(source)
        logger.info("Showcase mode stopped")
        return stopped

    # -- health ------------------------------------------------------------

    def health(self) -> Mapping[str, Any]:
        """Return the F4 keys: ``imu``, ``camera``, ``model``, ``fps``, ``dpu_ms``, ``errors``.

        ``imu`` is always ``"n/a"``: this mode drives no BlueCoin. ``camera`` is
        ``ok`` while frames keep arriving, ``missing``/``error`` after a camera
        failure and ``None`` when the mode is not running.
        """
        with self._lock:
            running = self._running
            camera_error = self._camera_error
            stamps = tuple(self._processed)
            last_arrival = self._last_arrival
            dpu_ms = self._last_dpu_ms
            errors = list(self._errors)
        camera: str | None
        if camera_error is not None:
            camera = camera_error
        elif not running:
            camera = None
        elif last_arrival > 0.0 and (time.monotonic() - last_arrival) <= CAMERA_STALE_S:
            camera = "ok"
        else:
            camera = "missing"
        fps: float | None = None
        if running:
            fps = 0.0
            if len(stamps) >= 2 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        return {
            "imu": "n/a",
            "camera": camera,
            "model": self._showcase.model().name,
            "fps": fps,
            "dpu_ms": dpu_ms,
            "errors": errors,
        }

    # -- frame loop --------------------------------------------------------

    def _ingest(self, frame: VideoFrame) -> None:
        """Keep only the newest frame: a slow detector never builds a backlog."""
        with self._lock:
            self._last_arrival = time.monotonic()
        self._slot.put(frame)

    def _run(self) -> None:
        period = 1.0 / self._showcase.max_fps
        next_due = 0.0
        while not self._stop.is_set():
            frame = self._slot.get(timeout=_SLOT_TIMEOUT_S)
            if frame is None:
                continue
            now = time.perf_counter()
            if now < next_due:
                continue
            next_due = now + period
            try:
                self._process(frame)
            except Exception:
                logger.exception("showcase mode failed on frame %d", frame.seq)

    def _process(self, frame: VideoFrame) -> None:
        """Detect, draw and publish one frame; runs on the showcase thread."""
        with self._lock:
            ctx = self._ctx
            detector = self._detector
        if ctx is None or detector is None:
            return
        started = time.perf_counter()
        rgb = frame.decode()
        objects = _objects(detector.detect(rgb), self._labels)
        overlay = draw_overlay(rgb, objects)
        total_ms = (time.perf_counter() - started) * 1000.0
        stage = detector.stage_ms
        dpu_ms = stage.get("dpu")
        self._offer_video(ctx, frame.seq, overlay)
        with self._lock:
            self._processed.append(time.perf_counter())
            self._last_dpu_ms = None if dpu_ms is None else float(dpu_ms)
            stamps = tuple(self._processed)
            last_dpu = self._last_dpu_ms
            last_publish = self._last_publish
        now = time.monotonic()
        if now - last_publish < STATE_PERIOD_S:
            return
        with self._lock:
            self._last_publish = now
        fps = 0.0
        if len(stamps) >= 2 and stamps[-1] > stamps[0]:
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        self._publish(
            ctx,
            width=frame.width,
            height=frame.height,
            objects=objects,
            fps=fps,
            dpu_ms=last_dpu,
            total_ms=total_ms,
        )

    def _offer_video(
        self,
        ctx: ModeContext,
        seq: int,
        image: NDArray[np.uint8],
    ) -> None:
        """Offer the drawn frame to ``/video.mjpeg`` (F14); never blocks."""
        offer = getattr(ctx.publisher, "offer_video_frame", None)
        if not callable(offer):
            return
        try:
            offer(seq, image)
        except Exception:
            logger.exception("Publisher rejected a showcase video frame")

    def _camera_failed(self, error: Exception) -> None:
        """Record a capture failure reported by the source thread."""
        logger.error("Showcase camera failed: %s", error)
        with self._lock:
            self._camera_error = "error"
            if "camera_error" not in self._errors:
                self._errors.append("camera_error")

    # -- publishing --------------------------------------------------------

    def _publish(
        self,
        ctx: ModeContext,
        *,
        width: int,
        height: int,
        objects: Sequence[ShowcaseObject],
        fps: float | None,
        dpu_ms: float | None,
        total_ms: float | None,
    ) -> None:
        payload = showcase_state(
            self._showcase.model().name,
            width,
            height,
            objects,
            fps=fps,
            dpu_ms=dpu_ms,
            total_ms=total_ms,
        )
        try:
            ctx.publisher.state("showcase.state", payload)
        except Exception:
            logger.exception("Publisher rejected showcase.state")

    # -- helpers -----------------------------------------------------------

    def _fail(self, code: str) -> None:
        with self._lock:
            self._errors = [code]
            self._camera_error = _CAMERA_CODES.get(code)

    def _release(self) -> None:
        """Release a start that failed after the camera opened."""
        with self._lock:
            source = self._source
            self._source = None
            self._detector = None
            self._ctx = None
            self._thread = None
            self._running = False
            self._stop.set()
        _release_quietly(source)


def _objects(
    detections: Sequence[ShowcaseDetection],
    labels: Sequence[str],
) -> list[ShowcaseObject]:
    """Map the detector output to the labelled objects the wire and the overlay use."""

    objects: list[ShowcaseObject] = []
    for detection in detections:
        class_id = int(detection.class_id)
        box = tuple(float(value) for value in detection.box)
        if len(box) != 4:
            logger.debug("Dropping a detection whose box has %d values", len(box))
            continue
        objects.append(
            ShowcaseObject(
                label=_label(class_id, labels),
                class_id=class_id,
                score=float(detection.score),
                box=(box[0], box[1], box[2], box[3]),
            )
        )
    return objects


def _label(class_id: int, labels: Sequence[str]) -> str:
    """Return the label of a COCO class; an unknown index keeps its number."""
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class {class_id}"


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return the RGB colour of a class: stable across frames and runs."""
    return _PALETTE[abs(int(class_id)) % len(_PALETTE)]


def draw_overlay(
    rgb: NDArray[np.uint8],
    objects: Sequence[ShowcaseObject],
) -> NDArray[np.uint8]:
    """Return a copy of ``rgb`` with a 2 px box and a ``"<label> <score>%"`` tag.

    The image is RGB (the order :meth:`WebServer.offer_video_frame` expects) and
    the copy is what the encoder gets, so the frame being analyzed can never
    change under it.
    """
    overlay = np.array(rgb, copy=True)
    height, width = overlay.shape[:2]
    for item in objects:
        colour = class_color(item.class_id)
        x1, y1, x2, y2 = _clip(item.box, width, height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, _BOX_THICKNESS)
        text = f"{item.label} {int(round(item.score * 100.0))}%"
        (text_width, text_height), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        label_height = text_height + 2 * _LABEL_GAP
        top = y1 - label_height
        if top < 0:
            # No room above the box: put the tag just inside its top edge.
            top = y1 + _LABEL_GAP
        top = max(0, min(top, height - label_height))
        bottom = min(height - 1, top + label_height)
        right = min(width - 1, x1 + text_width + 2 * _LABEL_PADDING)
        cv2.rectangle(overlay, (x1, top), (right, bottom), colour, -1)
        cv2.putText(
            overlay,
            text,
            (x1 + _LABEL_PADDING, bottom - _LABEL_GAP),
            _FONT,
            _FONT_SCALE,
            _text_color(colour),
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return overlay


def _clip(
    box: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Clip a float box to the image and return integer corners inside it."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
        return (0, 0, 0, 0)
    left = min(max(int(round(x1)), 0), max(width - 1, 0))
    top = min(max(int(round(y1)), 0), max(height - 1, 0))
    right = min(max(int(round(x2)), left), max(width - 1, 0))
    bottom = min(max(int(round(y2)), top), max(height - 1, 0))
    return (left, top, right, bottom)


def _text_color(colour: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black on a light box colour, white on a dark one."""
    red, green, blue = colour
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 140.0 else (255, 255, 255)


def _release_quietly(source: ShowcaseSource | None) -> None:
    """Release a source without masking the failure that is being reported."""
    if source is None:
        return
    try:
        source.stop()
    except Exception:
        logger.exception("Failed to release the showcase RGB source")
