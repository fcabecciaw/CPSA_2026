"""The one configuration file of the showcase (F-K3).

``config/showcase.yaml`` is the single source: the YAML carries every value,
this module only validates it and names the offending dotted key on error. The
sections are exactly the F-K3 ones — ``camera``, ``model``, ``labels``, ``web``,
``log`` — and nothing else: a stale key is rejected instead of silently ignored,
so a tree cut from an older demo cannot boot with half its settings ignored.

```yaml
camera: { device: auto, match: RealSense, width: 640, height: 480, fps: 60,
          exposure: 150, gain: 320, power_line_hz: 50 }
model:  { path: models/yolox_nano.xmodel, conf_thr: 0.35, iou_thr: 0.45 }
labels: it
web:    { host: 0.0.0.0, port: 8080, video_fps: 60, jpeg_quality: 70,
          static_dir: web/dist }
log:    { level: INFO }
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
    "CameraConfig",
    "ConfigError",
    "LogConfig",
    "ModelConfig",
    "ShowcaseConfig",
    "WebConfig",
    "load",
    "parse",
]

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
"""Values of ``log.level``."""

LABEL_LANGUAGES = ("it", "en")
"""Values of the top-level ``labels`` key; Italian is the booth default."""

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
    """

    device: str
    match: str
    width: int
    height: int
    fps: float
    exposure: int
    gain: int
    power_line_hz: int


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
    model: ModelConfig
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
    model = _model_config(root.section("model"))
    labels = root.choice("labels", LABEL_LANGUAGES)
    web = _web_config(root.section("web"))
    log = LogConfig(level=root.section("log").choice("level", LOG_LEVELS))
    root.reject_unknown(("camera", "model", "labels", "web", "log"))
    return ShowcaseConfig(camera=camera, model=model, labels=labels, web=web, log=log)


def _camera_config(camera: _Section) -> CameraConfig:
    config = CameraConfig(
        device=camera.text("device"),
        match=camera.optional_text("match", DEFAULT_CAMERA_MATCH),
        width=camera.even_positive_int("width"),
        height=camera.even_positive_int("height"),
        fps=camera.positive("fps"),
        exposure=camera.integer("exposure", 1, 10000),
        gain=camera.integer("gain", 1, 65535),
        power_line_hz=camera.power_line_hz("power_line_hz"),
    )
    camera.reject_unknown()
    return config


def _model_config(model: _Section) -> ModelConfig:
    config = ModelConfig(
        path=Path(model.text("path")).expanduser(),
        conf_thr=model.fraction("conf_thr"),
        iou_thr=model.fraction("iou_thr"),
    )
    model.reject_unknown()
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
