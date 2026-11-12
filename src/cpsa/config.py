"""The one configuration file of the showcase (F-K3).

``config/showcase.yaml`` is the single source: the YAML carries every value,
this module only validates it and names the offending dotted key on error. The
sections are exactly the showcase ones — ``camera``, ``tracker``, ``model``,
``labels``, ``web``, ``log`` — and nothing else: a stale key is rejected instead
of silently ignored, so a tree cut from an older demo cannot boot with half its
settings ignored.

```yaml
camera:  { device: auto, match: RealSense, width: 640, height: 480, fps: 60,
           exposure: 100, gain: 200, white_balance: 4000,
           backlight_compensation: 0, saturation: 110, power_line_hz: 50 }
tracker: { enabled: true, iou_match: 0.3, alpha: 0.35, fast_alpha: 0.8,
           deadband_iou: 0.9, min_hits: 2, max_misses: 6 }
model:   { path: models/yolox_nano.xmodel, conf_thr: 0.35, iou_thr: 0.45 }
movenet: { path: models/movenet.xmodel, enabled: true, publish_hz: 30,
           smoothing: { enabled: true, min_cutoff: 1.0, beta: 0.02,
                        d_cutoff: 1.0, deadband_px: 1.5, show_score: 0.35,
                        hide_score: 0.25, show_frames: 2, hide_frames: 4 } }
labels: it
web:     { host: 0.0.0.0, port: 8080, video_fps: 60, jpeg_quality: 70,
           static_dir: web/dist }
log:     { level: INFO }
```
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = [
    "LABEL_LANGUAGES",
    "LOG_LEVELS",
    "WHITE_BALANCE_AUTO",
    "CameraConfig",
    "ConfigError",
    "LogConfig",
    "ModelConfig",
    "MovenetConfig",
    "ShowcaseConfig",
    "SmoothingConfig",
    "TrackerConfig",
    "WebConfig",
    "load",
    "parse",
]

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
"""Values of ``log.level``."""

LABEL_LANGUAGES = ("it", "en")
"""Values of the top-level ``labels`` key; Italian is the booth default."""

WHITE_BALANCE_AUTO = "auto"
"""``camera.white_balance`` value that leaves the ISP auto white balance on."""

GAIN_RANGE = (64, 2540)
"""Manual ``gain`` range the F200 RGB node accepts."""

WHITE_BALANCE_RANGE = (2500, 6500)
"""``white_balance_temperature`` range in Kelvin, in 100 K steps."""

BACKLIGHT_COMPENSATION_RANGE = (0, 4)
"""``backlight_compensation`` range the F200 RGB node accepts."""

SATURATION_RANGE = (0, 255)
"""V4L2 ``saturation`` range of the F200 RGB node; the camera default is 128."""


DEFAULT_CAMERA_MATCH = "RealSense"
"""Card-name fragment ``device: auto`` looks for when ``camera.match`` is absent.

The kernel truncates the F200 card name to 31 characters
(``"Intel(R) RealSense(TM) 3D Camer"``), so it never contains ``"F200"``.
"""


class ConfigError(Exception):
    """The configuration is missing, unreadable or invalid at ``key``."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(f"invalid configuration at '{key}': {reason}")
        self.key = key
        self.reason = reason


@dataclass(frozen=True)
class CameraConfig:
    """V4L2 capture settings of the RGB node.

    ``device`` is a ``/dev/videoN`` path or ``auto``; with ``auto`` the node
    whose card name contains ``match`` is selected at every start, so a USB
    re-plug that renumbers the nodes cannot break the demo. ``exposure`` is the
    manual ``exposure_time_absolute`` in 100 us units and ``gain`` the manual
    V4L2 gain; ``power_line_hz`` is 0 (off), 50 or 60.

    ``white_balance`` is a colour temperature in Kelvin (2500..6500 in 100 K
    steps) or ``"auto"``, ``backlight_compensation`` is the F200 ISP strength
    0..4 and ``saturation`` the V4L2 saturation 0..255. All of them are applied
    at every open: the ISP's own automatic adjustments are what made the booth
    image drift.
    """

    device: str
    match: str
    width: int
    height: int
    fps: float
    exposure: int
    gain: int
    white_balance: int | str
    backlight_compensation: int
    saturation: int
    power_line_hz: int
    mirror: bool = True


@dataclass(frozen=True)
class TrackerConfig:
    """Box stabilization between the detector and the overlay.

    ``enabled: false`` publishes the raw detector boxes unchanged. Otherwise a
    detection joins a track of the same class when their IoU is at least
    ``iou_match``; the track filters its box with an EMA of ``alpha``
    (``fast_alpha`` when the new box barely overlaps the filtered one, so a
    real move is followed at once) and only moves its displayed box when it
    leaves a ``deadband_iou`` overlap or shifts by more than 4% of its diagonal
    or changes a side by more than 8%. A track is shown after ``min_hits``
    consecutive detections (immediately at score 0.6) and survives
    ``max_misses`` missed frames.
    """

    enabled: bool
    iou_match: float
    alpha: float
    fast_alpha: float
    deadband_iou: float
    min_hits: int
    max_misses: int


@dataclass(frozen=True)
class SmoothingConfig:
    """Keypoint stabilization between the MoveNet decoder and the overlay.

    ``enabled: false`` publishes the raw keypoints unchanged. Otherwise every
    keypoint runs a One Euro filter on ``x, y``: ``min_cutoff`` Hz at rest, raised
    by ``beta`` pixels of measured speed so a real swing is followed, with
    ``d_cutoff`` Hz on the speed estimate itself. The published position leaves
    a ``deadband_px`` deadband (widened to 1 % of the shoulder width when that
    is wider) and a keypoint is shown after ``show_frames`` frames at
    ``show_score`` and hidden only after ``hide_frames`` frames under
    ``hide_score``.
    """

    enabled: bool
    min_cutoff: float
    beta: float
    d_cutoff: float
    deadband_px: float
    show_score: float
    hide_score: float
    show_frames: int
    hide_frames: int


@dataclass(frozen=True)
class ModelConfig:
    """The compiled detector this build runs.

    ``path`` is the compiled ``.xmodel`` the DPU loads (relative paths resolve
    against the process working directory, which the unit sets to the repository
    root); :attr:`name` is what ``status.model`` and ``showcase.state`` report.
    """

    path: Path
    conf_thr: float
    iou_thr: float

    @property
    def name(self) -> str:
        """The model name on the wire: the file's stem (``yolox_nano``)."""
        return self.path.stem


@dataclass(frozen=True)
class MovenetConfig:
    """The single-pose skeleton model that ``model.set`` loads in place of YOLOX (R2-G).

    ``path`` is the compiled ``.xmodel`` the DPU runner loads on a swap;
    ``enabled: false`` rejects ``model.set movenet`` and leaves the process
    YOLOX-only, ``publish_hz`` caps ``pose.state``, independently of the frame
    rate, and ``smoothing`` holds the keypoint stabilization.
    """

    path: Path
    enabled: bool
    publish_hz: float
    smoothing: SmoothingConfig

    @property
    def name(self) -> str:
        """The model name on the wire: the file's stem (``movenet``)."""
        return self.path.stem


@dataclass(frozen=True)
class WebConfig:
    """HTTP and WebSocket endpoint (F-K2)."""

    host: str
    port: int
    video_fps: float
    jpeg_quality: int
    static_dir: Path


@dataclass(frozen=True)
class LogConfig:
    """Log verbosity; the unit's journal keeps the lines."""

    level: str


@dataclass(frozen=True)
class ShowcaseConfig:
    """Everything the one process needs, validated."""

    camera: CameraConfig
    tracker: TrackerConfig
    model: ModelConfig
    movenet: MovenetConfig
    labels: str
    web: WebConfig
    log: LogConfig


def load(path: Path) -> ShowcaseConfig:
    """Read and validate the YAML configuration at ``path``."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(str(path), f"cannot read file: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(str(path), f"invalid YAML: {exc}") from exc
    return parse(data)


def parse(data: object) -> ShowcaseConfig:
    """Validate an already parsed configuration document."""

    root = _Section(data, "")
    camera = _camera_config(root.section("camera"))
    tracker = _tracker_config(root.section("tracker"))
    model = _model_config(root.section("model"))
    movenet = _movenet_config(root.section("movenet"))
    labels = root.choice("labels", LABEL_LANGUAGES)
    web = _web_config(root.section("web"))
    log = LogConfig(level=root.section("log").choice("level", LOG_LEVELS))
    root.reject_unknown(("camera", "tracker", "model", "movenet", "labels", "web", "log"))
    return ShowcaseConfig(
        camera=camera,
        tracker=tracker,
        model=model,
        movenet=movenet,
        labels=labels,
        web=web,
        log=log,
    )


def _camera_config(camera: _Section) -> CameraConfig:
    config = CameraConfig(
        device=camera.text("device"),
        match=camera.optional_text("match", DEFAULT_CAMERA_MATCH),
        width=camera.even_positive_int("width"),
        height=camera.even_positive_int("height"),
        fps=camera.positive("fps"),
        exposure=camera.integer("exposure", 1, 10000),
        gain=camera.integer("gain", *GAIN_RANGE),
        white_balance=camera.white_balance("white_balance"),
        backlight_compensation=camera.integer(
            "backlight_compensation", *BACKLIGHT_COMPENSATION_RANGE
        ),
        saturation=camera.integer("saturation", *SATURATION_RANGE),
        power_line_hz=camera.power_line_hz("power_line_hz"),
        mirror=camera.optional_boolean("mirror", True),
    )
    camera.reject_unknown()
    return config


def _tracker_config(tracker: _Section) -> TrackerConfig:
    config = TrackerConfig(
        enabled=tracker.boolean("enabled"),
        iou_match=tracker.fraction("iou_match"),
        alpha=tracker.fraction("alpha"),
        fast_alpha=tracker.fraction("fast_alpha"),
        deadband_iou=tracker.fraction("deadband_iou"),
        min_hits=tracker.integer("min_hits", 0, 100),
        max_misses=tracker.integer("max_misses", 0, 100),
    )
    tracker.reject_unknown()
    return config


def _model_config(model: _Section) -> ModelConfig:
    config = ModelConfig(
        path=Path(model.text("path")).expanduser(),
        conf_thr=model.fraction("conf_thr"),
        iou_thr=model.fraction("iou_thr"),
    )
    model.reject_unknown()
    return config


def _movenet_config(movenet: _Section) -> MovenetConfig:
    config = MovenetConfig(
        path=Path(movenet.text("path")).expanduser(),
        enabled=movenet.boolean("enabled"),
        publish_hz=movenet.positive("publish_hz"),
        smoothing=_smoothing_config(movenet.section("smoothing")),
    )
    movenet.reject_unknown()
    return config


def _smoothing_config(smoothing: _Section) -> SmoothingConfig:
    config = SmoothingConfig(
        enabled=smoothing.boolean("enabled"),
        min_cutoff=smoothing.positive("min_cutoff"),
        beta=smoothing.positive("beta"),
        d_cutoff=smoothing.positive("d_cutoff"),
        deadband_px=smoothing.positive("deadband_px"),
        show_score=smoothing.fraction("show_score"),
        hide_score=smoothing.fraction("hide_score"),
        show_frames=smoothing.integer("show_frames", 1, 100),
        hide_frames=smoothing.integer("hide_frames", 1, 100),
    )
    smoothing.reject_unknown()
    if config.hide_score >= config.show_score:
        raise ConfigError(
            smoothing._key("hide_score"),
            "expected a value below show_score",
        )
    return config


def _web_config(web: _Section) -> WebConfig:
    config = WebConfig(
        host=web.text("host"),
        port=web.integer("port", 1, 65535),
        video_fps=web.positive("video_fps"),
        jpeg_quality=web.integer("jpeg_quality", 1, 100),
        static_dir=Path(web.text("static_dir")).expanduser(),
    )
    web.reject_unknown()
    return config


class _Section:
    """A mapping in the document, tracking which keys were consumed."""

    def __init__(self, data: object, path: str) -> None:
        if not isinstance(data, Mapping):
            raise ConfigError(path or "<root>", "expected a mapping")
        self._data = data
        self._path = path
        self._used: set[str] = set()

    def _key(self, name: str) -> str:
        return f"{self._path}.{name}" if self._path else name

    def _get(self, name: str) -> object:
        if name not in self._data:
            raise ConfigError(self._key(name), "missing required key")
        self._used.add(name)
        return self._data[name]

    def section(self, name: str) -> _Section:
        return _Section(self._get(name), self._key(name))

    def text(self, name: str) -> str:
        value = self._get(name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(self._key(name), "expected a non-empty string")
        return value

    def optional_text(self, name: str, default: str) -> str:
        """Return the non-empty string at ``name``, or ``default`` when absent."""
        if name not in self._data:
            return default
        return self.text(name)

    def choice(self, name: str, options: tuple[str, ...]) -> str:
        value = self.text(name)
        if value not in options:
            raise ConfigError(self._key(name), f"expected one of {', '.join(options)}")
        return value

    def positive(self, name: str) -> float:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(self._key(name), "expected a positive number")
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if not math.isfinite(number) or number <= 0:
            raise ConfigError(self._key(name), "expected a finite positive number")
        return number

    def fraction(self, name: str) -> float:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(self._key(name), "expected a number between 0 and 1")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ConfigError(self._key(name), "expected a number between 0 and 1")
        return number

    def integer(self, name: str, minimum: int, maximum: int) -> int:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(self._key(name), "expected an integer")
        if not minimum <= value <= maximum:
            raise ConfigError(
                self._key(name), f"expected an integer between {minimum} and {maximum}"
            )
        return value

    def boolean(self, name: str) -> bool:
        value = self._get(name)
        if not isinstance(value, bool):
            raise ConfigError(self._key(name), "expected true or false")
        return value

    def optional_boolean(self, name: str, default: bool) -> bool:
        if name not in self._data:
            return default
        return self.boolean(name)

    def white_balance(self, name: str) -> int | str:
        """A Kelvin integer in 100 K steps, or :data:`WHITE_BALANCE_AUTO`."""
        value = self._get(name)
        if isinstance(value, str):
            if value.strip().casefold() == WHITE_BALANCE_AUTO:
                return WHITE_BALANCE_AUTO
            raise ConfigError(self._key(name), "expected 'auto' or a Kelvin integer")
        minimum, maximum = WHITE_BALANCE_RANGE
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(self._key(name), "expected 'auto' or a Kelvin integer")
        if not minimum <= value <= maximum or value % 100:
            raise ConfigError(
                self._key(name),
                f"expected 'auto' or a Kelvin integer between {minimum} and {maximum} "
                "in steps of 100",
            )
        return value

    def even_positive_int(self, name: str) -> int:
        """A positive even integer: YUYV packs two pixels per four-byte group."""
        value = self.integer(name, 2, 16384)
        if value % 2:
            raise ConfigError(self._key(name), "expected an even integer")
        return value

    def power_line_hz(self, name: str) -> int:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 50, 60):
            raise ConfigError(self._key(name), "expected 0, 50 or 60")
        return value

    def reject_unknown(self, expected: tuple[str, ...] | None = None) -> None:
        unknown = sorted(str(key) for key in self._data if key not in self._used)
        if unknown:
            hint = f" (expected {', '.join(expected)})" if expected else ""
            raise ConfigError(self._key(unknown[0]), f"unknown key{hint}")
