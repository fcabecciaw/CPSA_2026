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
model:   { path: models/yolox_nano.xmodel, conf_thr: 0.35,
           class_conf: {person: 0.40}, object_conf: 0.25, iou_thr: 0.45,
           zoom: 1.0, zoom_center: [0.5, 0.5], alternate: false }
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = [
    "LABEL_LANGUAGES",
    "LOG_LEVELS",
    "WHITE_BALANCE_AUTO",
    "ZOOM_RANGE",
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


ZOOM_RANGE = (1.0, 2.0)
"""``model.zoom`` range: 1.0 is the whole frame, 2.0 the central quarter."""

DETECTOR_KINDS = ("yolox", "yolov5")
"""Values of ``model.kind``: the object detector the showcase loads at start."""

DEFAULT_YOLOV5_PATH = Path("models/yolov5_nano.xmodel")
"""Where ``deploy.sh --yolov5-model`` installs YOLOv5-nano on the board."""


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

    ``replay_dir`` replaces the V4L2 node with :class:`cpsa.camera.ReplayCamera`,
    which loops the images of that directory at ``fps``: the fps of the whole
    app is then measurable on the board with nobody in front of the camera.
    ``replay_hold_frames`` frames are shown per image and every frame is shifted
    by up to ``replay_jitter_px`` pixels, so the scene keeps changing for the
    still gate at roughly ``fps / replay_hold_frames`` changes per second.
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
    replay_dir: Path | None = None
    replay_hold_frames: int = 6
    replay_jitter_px: int = 2


@dataclass(frozen=True)
class StillConfig:
    """The motion gate that keeps a still scene off the DPU.

    ``enabled: false`` runs the detector on every frame, as before. Otherwise
    the loop measures how much the 80x60 grey image moved since the last frame
    it actually inferred; when that mean difference stays below ``motion_thr``
    for three frames the frames are drawn from the last boxes without touching
    the DPU, and a full detection is forced every ``force_every_s`` seconds
    (or at once, when the scene moves) so a new still object cannot be missed
    for longer than that.
    """

    enabled: bool
    motion_thr: float
    force_every_s: float


@dataclass(frozen=True)
class TrackerConfig:
    """Box stabilization between the detector and the overlay.

    ``enabled: false`` publishes the raw detector boxes unchanged. Otherwise a
    detection joins a track when their IoU is at least ``iou_match`` and the
    class agrees, or at 0.5 when it does not (an agreeing class always wins);
    the track filters its box with an EMA of ``alpha`` (``fast_alpha`` when the
    new box barely overlaps the filtered one, so a real move is followed at
    once) and only moves its displayed box when it leaves a ``deadband_iou``
    overlap or shifts by more than 4% of its diagonal or changes a side by more
    than 8%. Its class is the majority of the last 15 matched frames, relabelled
    only when a new class holds 70% of them for 0.5 s, so a detector that
    flickers between two classes no longer kills and re-creates the track. A
    track whose displayed box has held for a second is *settled*: it tolerates a
    wider deadband, a score quantized to 5% and up to 30 missed frames. A track
    is shown after ``min_hits`` consecutive detections (immediately at score
    0.6) and otherwise survives ``max_misses`` missed frames.

    ``still`` is the motion gate in front of the DPU (:class:`StillConfig`).
    """

    enabled: bool
    iou_match: float
    alpha: float
    fast_alpha: float
    deadband_iou: float
    min_hits: int
    max_misses: int
    still: StillConfig


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

    ``conf_thr`` is the class-independent minimum ``objectness * class`` score
    and stays the default. ``object_conf`` is the threshold of every object
    class, and ``class_conf`` overrides single classes by their canonical
    COCO-80 English name (``{"person": 0.40}``); both are optional, so a config
    without them keeps ``conf_thr`` for every class. The precedence is
    ``class_conf``, then ``object_conf``, then ``conf_thr``. The names are
    checked against the COCO-80 table when the detector is built
    (:func:`cpsa.yolox.class_thresholds`), so a typo cannot reach the board.

    ``zoom`` centre-crops the frame before the letterbox: ``1.0`` (the default)
    feeds the whole frame, ``1.3``-``1.5`` feeds the central 77%-44% into the
    same 416x416 input, so the objects held in front of the camera arrive
    larger and stop flickering between classes. ``zoom_center`` moves that crop
    (``[0.5, 0.5]`` is the middle); boxes are mapped back to full-frame pixels.

    A zoom hides the frame edges, which is where a person walks in, so
    ``alternate`` (on by default whenever ``zoom`` is above 1.0) splits the
    passes: even frames run the whole frame and report only persons, odd frames
    run the crop and report every other class. One inference per frame, no
    extra DPU work, and each class group is only aged on its own passes.
    ``alternate: true`` with ``zoom: 1.0`` is refused: both passes would be the
    same frame and the split would only halve each group's update rate.

    ``kind`` picks the object detector: ``yolox`` (``path``, the default) or
    ``yolov5`` (``yolov5_path``, YOLOv5-nano at 640x640). YOLOv5 shares the
    thresholds and ignores the zoom: it sees the whole frame at full size. The
    dashboard's Calibrazione card switches between them at run time.
    """

    path: Path
    conf_thr: float
    iou_thr: float
    class_conf: Mapping[str, float] = field(default_factory=dict)
    object_conf: float | None = None
    zoom: float = 1.0
    zoom_center: tuple[float, float] = (0.5, 0.5)
    alternate: bool = False
    kind: str = DETECTOR_KINDS[0]
    yolov5_path: Path = DEFAULT_YOLOV5_PATH

    @property
    def detector_path(self) -> Path:
        """The ``.xmodel`` of the selected ``kind``."""
        return self.yolov5_path if self.kind == "yolov5" else self.path

    @property
    def name(self) -> str:
        """The model name on the wire: the stem of the selected file (``yolox_nano``)."""
        return self.detector_path.stem


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
        replay_dir=_optional_path(camera, "replay_dir"),
        replay_hold_frames=camera.optional_integer("replay_hold_frames", 6, 1, 600),
        replay_jitter_px=camera.optional_integer("replay_jitter_px", 2, 0, 64),
    )
    camera.reject_unknown()
    return config


def _optional_path(section: _Section, name: str) -> Path | None:
    """Return the path at ``name``, or ``None`` when the key is absent."""
    text = section.optional_text(name, "")
    return Path(text).expanduser() if text else None


def _tracker_config(tracker: _Section) -> TrackerConfig:
    config = TrackerConfig(
        enabled=tracker.boolean("enabled"),
        iou_match=tracker.fraction("iou_match"),
        alpha=tracker.fraction("alpha"),
        fast_alpha=tracker.fraction("fast_alpha"),
        deadband_iou=tracker.fraction("deadband_iou"),
        min_hits=tracker.integer("min_hits", 0, 100),
        max_misses=tracker.integer("max_misses", 0, 100),
        still=_still_config(tracker.section("still")),
    )
    tracker.reject_unknown()
    return config


def _still_config(still: _Section) -> StillConfig:
    config = StillConfig(
        enabled=still.boolean("enabled"),
        motion_thr=still.positive("motion_thr"),
        force_every_s=still.positive("force_every_s"),
    )
    still.reject_unknown()
    return config


def _model_config(model: _Section) -> ModelConfig:
    zoom = model.optional_number("zoom", 1.0, *ZOOM_RANGE)
    alternate = model.optional_boolean("alternate", zoom > 1.0)
    if alternate and zoom <= 1.0:
        raise ConfigError(
            model._key("alternate"),
            "expected the key to be absent or false when model.zoom is 1.0",
        )
    config = ModelConfig(
        path=Path(model.text("path")).expanduser(),
        conf_thr=model.fraction("conf_thr"),
        iou_thr=model.fraction("iou_thr"),
        class_conf=model.fraction_map("class_conf"),
        object_conf=model.optional_fraction("object_conf"),
        zoom=zoom,
        zoom_center=model.fraction_pair("zoom_center", (0.5, 0.5)),
        alternate=alternate,
        kind=model.optional_choice("kind", DETECTOR_KINDS, DETECTOR_KINDS[0]),
        yolov5_path=Path(model.optional_text("yolov5_path", str(DEFAULT_YOLOV5_PATH))).expanduser(),
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

    def optional_choice(self, name: str, options: tuple[str, ...], default: str) -> str:
        """Return the option at ``name``, or ``default`` when absent."""
        if name not in self._data:
            return default
        return self.choice(name, options)

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

    def optional_fraction(self, name: str) -> float | None:
        """Return the fraction at ``name``, or ``None`` when the key is absent."""
        if name not in self._data:
            return None
        return self.fraction(name)

    def optional_number(self, name: str, default: float, minimum: float, maximum: float) -> float:
        """Return the number at ``name`` within ``[minimum, maximum]``, or ``default``."""
        if name not in self._data:
            return default
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(self._key(name), f"expected a number between {minimum} and {maximum}")
        number = float(value)
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ConfigError(self._key(name), f"expected a number between {minimum} and {maximum}")
        return number

    def fraction_pair(self, name: str, default: tuple[float, float]) -> tuple[float, float]:
        """Return the ``[x, y]`` fraction pair at ``name``, or ``default`` when absent."""
        if name not in self._data:
            return default
        value = self._get(name)
        if isinstance(value, (str, Mapping)) or not isinstance(value, Sequence):
            raise ConfigError(self._key(name), "expected two numbers between 0 and 1")
        pair = list(value)
        if len(pair) != 2:
            raise ConfigError(self._key(name), "expected two numbers between 0 and 1")
        numbers: list[float] = []
        for item in pair:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ConfigError(self._key(name), "expected two numbers between 0 and 1")
            number = float(item)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ConfigError(self._key(name), "expected two numbers between 0 and 1")
            numbers.append(number)
        return (numbers[0], numbers[1])

    def fraction_map(self, name: str) -> dict[str, float]:
        """Return a ``class name -> fraction`` mapping, empty when the key is absent."""
        if name not in self._data:
            return {}
        value = self._get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(self._key(name), "expected a mapping of class names to fractions")
        fractions: dict[str, float] = {}
        for class_name, raw_threshold in value.items():
            key = f"{self._key(name)}.{class_name}"
            if not isinstance(class_name, str) or not class_name.strip():
                raise ConfigError(key, "expected a non-empty COCO class name")
            if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
                raise ConfigError(key, "expected a number between 0 and 1")
            number = float(raw_threshold)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ConfigError(key, "expected a number between 0 and 1")
            fractions[class_name] = number
        return fractions

    def integer(self, name: str, minimum: int, maximum: int) -> int:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(self._key(name), "expected an integer")
        if not minimum <= value <= maximum:
            raise ConfigError(
                self._key(name), f"expected an integer between {minimum} and {maximum}"
            )
        return value

    def optional_integer(self, name: str, default: int, minimum: int, maximum: int) -> int:
        """Return the integer at ``name`` within ``[minimum, maximum]``, or ``default``."""
        if name not in self._data:
            return default
        return self.integer(name, minimum, maximum)

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
