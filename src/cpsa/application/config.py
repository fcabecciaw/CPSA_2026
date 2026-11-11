"""Typed, validated board configuration.

Every key of the hardware sections is required and there are no code defaults:
the YAML file is the single source of configuration. ``server`` and ``supervisor``
are optional sections; when a section is absent its dataclass defaults apply.
Validation failures raise :class:`ConfigError` naming the dotted key.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, TypeVar, cast

import yaml

from cpsa.domain.model import FaceRegion, HandShape
from cpsa.vision.assessment import AnalysisConfig, SeverityConfig
from cpsa.vision.contacts2d import ContactConfig2D
from cpsa.vision.handface import FeatureConfig
from cpsa.vision.handshape import HandShapeConfig
from cpsa.vision.layouts import get_layout
from cpsa.vision.oscillation import OscillationConfig
from cpsa.vision.skeleton import SkeletonConfig
from cpsa.vision.subject import SubjectConfig
from cpsa.vision.tracker2d import Tracker2DConfig
from cpsa.vision.trigger import TriggerConfig

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
VISION_MODEL_KINDS = ("onnx", "xmodel")
SUPERVISOR_MODES = ("idle", "medical")
MEDICAL_TRIGGERS = ("imu", "vision")
"""Episode trigger sources (FR-026)."""

TriggerMode = Literal["imu", "vision"]
DEFAULT_CAMERA_MATCH = "F200"
"""Card-name fragment used when ``vision.camera.device`` is ``auto``.

The adapters own the discovery itself; the application layer may not import
them, so the default the demo camera reports is repeated here.
"""
EnumT = TypeVar("EnumT", bound=Enum)


class ConfigError(Exception):
    """The configuration is missing, unreadable or invalid at ``key``."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(f"invalid configuration at '{key}': {reason}")
        self.key = key
        self.reason = reason


@dataclass(frozen=True)
class LogConfig:
    base_path: Path
    level: str


@dataclass(frozen=True)
class BlueCoinsConfig:
    left: str
    right: str
    scan_timeout_s: float
    discovery_retries: int
    retry_interval_s: float


@dataclass(frozen=True)
class SyncConfig:
    max_skew_ms: float
    stale_ms: float


@dataclass(frozen=True)
class DispatchConfig:
    cooldown_s: float
    verification_deadline_s: float


@dataclass(frozen=True)
class RetryConfig:
    scan_timeout_s: float
    fast_attempts: int
    interval_s: float
    sleep_s: float


@dataclass(frozen=True)
class MetaMotionConfig:
    enable: bool


@dataclass(frozen=True)
class SpeakerConfig:
    enable: bool
    mac: str


@dataclass(frozen=True)
class LedStripConfig:
    enable: bool
    duration_ms: int


@dataclass(frozen=True)
class ActuatorsConfig:
    retry: RetryConfig
    metamotion: MetaMotionConfig
    speaker: SpeakerConfig
    led_strip: LedStripConfig


@dataclass(frozen=True)
class XuConfig:
    laser_power: int
    accuracy: int
    filter: int
    confidence: int


@dataclass(frozen=True)
class CameraConfig:
    rgb_device: str
    depth_device: str
    width: int
    height: int
    fps: float
    rgb_exposure: int
    xu: XuConfig


@dataclass(frozen=True)
class VisionModelConfig:
    """Pose-model settings: an ONNX export or a compiled DPU ``.xmodel``."""

    kind: str
    path: Path
    input_width: int
    input_height: int
    conf_thr: float
    iou_thr: float


@dataclass(frozen=True)
class VisionCameraConfig:
    """RGB-only UVC camera settings used by the vision pipeline.

    ``device`` may be ``auto``; ``match`` is the card-name fragment that
    :attr:`cpsa.adapters.camera.devices.resolve_device` looks for then.
    """

    device: str
    match: str
    width: int
    height: int
    fps: float
    exposure: int


@dataclass(frozen=True)
class VisionEvidenceConfig:
    """Evidence location, clip length and retention."""

    enabled: bool
    base_path: Path
    clip_s: float
    retention_days: int


@dataclass(frozen=True)
class VisionConfig:
    """2D vision pipeline settings (FR-001..FR-025).

    ``tracker`` and ``analysis`` are the validated Task B/C dataclasses, so the
    YAML is the single source of the feature, contact and severity thresholds.
    ``fast_budget_s`` must stay below ``dispatch.verification_deadline_s``; that
    cross-section check lives in :func:`parse`.
    """

    enabled: bool
    layout: str
    model: VisionModelConfig
    camera: VisionCameraConfig
    preroll_s: float
    lookback_s: float
    fast_after_s: float
    fast_budget_s: float
    idle_timeout_s: float
    stale_after_s: float
    analysis_fps: float
    queue_max: int
    imu_buffer_s: float
    evidence: VisionEvidenceConfig
    tracker: Tracker2DConfig
    analysis: AnalysisConfig

    def __post_init__(self) -> None:
        if self.model.kind not in VISION_MODEL_KINDS:
            raise ValueError(f"model kind must be one of {', '.join(VISION_MODEL_KINDS)}")
        if not 0.0 <= self.model.conf_thr <= 1.0:
            raise ValueError("model conf_thr must be in [0, 1]")
        if not 0.0 <= self.model.iou_thr <= 1.0:
            raise ValueError("model iou_thr must be in [0, 1]")
        for name in (
            "preroll_s",
            "lookback_s",
            "fast_budget_s",
            "idle_timeout_s",
            "stale_after_s",
            "analysis_fps",
            "imu_buffer_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if not math.isfinite(self.fast_after_s) or self.fast_after_s < 0.0:
            raise ValueError("fast_after_s must be a finite number >= 0")
        if self.analysis_fps <= 0.0:
            raise ValueError("analysis_fps must be positive")
        if self.evidence.retention_days < 0:
            raise ValueError("evidence retention_days must be >= 0")


@dataclass(frozen=True)
class ServerConfig:
    """HTTP and WebSocket endpoint served by ``cpsa-supervisor`` (IF-101)."""

    host: str = "0.0.0.0"
    port: int = 8080
    static_dir: Path = Path("web/dist")
    idle_timeout_s: float = 30.0


@dataclass(frozen=True)
class SupervisorConfig:
    """Which mode the supervisor starts at boot (FR-205)."""

    default_mode: str = "medical"


@dataclass(frozen=True)
class MedicalConfig:
    """Episode trigger source and the vision-trigger thresholds (FR-026..FR-030).

    A document without a ``medical`` section means ``trigger: imu``. With
    ``trigger: vision`` the medical mode runs camera-only: it never scans for or
    connects to a BlueCoin. The thresholds are the validated Task F7 dataclass;
    ``vision_trigger.refractory_s`` defaults to ``dispatch.cooldown_s``.
    """

    trigger: TriggerMode = "imu"
    vision_trigger: TriggerConfig = field(default_factory=TriggerConfig)


@dataclass(frozen=True)
class Config:
    log: LogConfig
    bluecoins: BlueCoinsConfig
    sync: SyncConfig
    dispatch: DispatchConfig
    actuators: ActuatorsConfig
    camera: CameraConfig
    vision: VisionConfig
    medical: MedicalConfig = field(default_factory=MedicalConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)


def load(path: Path) -> Config:
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


def parse(data: object) -> Config:
    """Validate an already parsed configuration document."""
    root = _Section(data, "")
    log = root.section("log")
    bluecoins = root.section("bluecoins")
    sync = root.section("sync")
    dispatch = root.section("dispatch")
    actuators = root.section("actuators")
    retry = actuators.section("retry")
    metamotion = actuators.section("metamotion")
    speaker = actuators.section("speaker")
    led_strip = actuators.section("led_strip")
    camera = root.section("camera")
    xu = camera.section("xu")
    vision = _vision_config(root.section("vision"))
    server_section = root.optional_section("server")
    supervisor_section = root.optional_section("supervisor")
    server = _server_config(server_section) if server_section is not None else ServerConfig()
    supervisor = (
        _supervisor_config(supervisor_section)
        if supervisor_section is not None
        else SupervisorConfig()
    )
    cooldown_s = dispatch.positive("cooldown_s")
    verification_deadline_s = dispatch.positive("verification_deadline_s")
    medical = _medical_config(root.optional_section("medical"), cooldown_s)

    if vision.fast_budget_s >= verification_deadline_s:
        raise ConfigError(
            "vision.fast_budget_s",
            "must be less than dispatch.verification_deadline_s",
        )
    if medical.trigger == "vision" and not vision.enabled:
        raise ConfigError("medical.trigger", "vision requires vision.enabled: true")

    config = Config(
        log=LogConfig(
            base_path=Path(log.text("base_path")).expanduser(),
            level=log.choice("level", LOG_LEVELS),
        ),
        bluecoins=BlueCoinsConfig(
            left=bluecoins.text("left"),
            right=bluecoins.text("right"),
            scan_timeout_s=bluecoins.positive("scan_timeout_s"),
            discovery_retries=bluecoins.count("discovery_retries"),
            retry_interval_s=bluecoins.positive("retry_interval_s"),
        ),
        sync=SyncConfig(
            max_skew_ms=sync.positive("max_skew_ms"),
            stale_ms=sync.positive("stale_ms"),
        ),
        dispatch=DispatchConfig(
            cooldown_s=cooldown_s,
            verification_deadline_s=verification_deadline_s,
        ),
        actuators=ActuatorsConfig(
            retry=RetryConfig(
                scan_timeout_s=retry.positive("scan_timeout_s"),
                fast_attempts=retry.count("fast_attempts"),
                interval_s=retry.positive("interval_s"),
                sleep_s=retry.positive("sleep_s"),
            ),
            metamotion=MetaMotionConfig(enable=metamotion.flag("enable")),
            speaker=SpeakerConfig(enable=speaker.flag("enable"), mac=speaker.text("mac")),
            led_strip=LedStripConfig(
                enable=led_strip.flag("enable"),
                duration_ms=led_strip.positive_int("duration_ms"),
            ),
        ),
        camera=CameraConfig(
            rgb_device=camera.text("rgb_device"),
            depth_device=camera.text("depth_device"),
            width=camera.positive_int("width"),
            height=camera.positive_int("height"),
            fps=camera.positive("fps"),
            rgb_exposure=camera.integer("rgb_exposure", 83, 10000),
            xu=XuConfig(
                laser_power=xu.integer("laser_power", 0, 16),
                accuracy=xu.integer("accuracy", 1, 3),
                filter=xu.integer("filter", 0, 7),
                confidence=xu.integer("confidence", 0, 15),
            ),
        ),
        vision=vision,
        medical=medical,
        server=server,
        supervisor=supervisor,
    )
    sections = (
        root,
        log,
        bluecoins,
        sync,
        dispatch,
        actuators,
        retry,
        metamotion,
        speaker,
        led_strip,
        camera,
        xu,
    )
    for section in sections:
        section.reject_unknown()
    return config


def _server_config(server: _Section) -> ServerConfig:
    """Build the validated optional ``server`` section (defaults apply when absent)."""
    config = ServerConfig(
        host=server.text("host"),
        port=server.integer("port", 1, 65535),
        static_dir=Path(server.text("static_dir")).expanduser(),
        idle_timeout_s=server.positive("idle_timeout_s"),
    )
    server.reject_unknown()
    return config


def _supervisor_config(supervisor: _Section) -> SupervisorConfig:
    """Build the validated optional ``supervisor`` section."""
    config = SupervisorConfig(default_mode=supervisor.choice("default_mode", SUPERVISOR_MODES))
    supervisor.reject_unknown()
    return config


def _medical_config(medical: _Section | None, cooldown_s: float) -> MedicalConfig:
    """Build the validated optional ``medical`` section (F9).

    An absent section is the IMU trigger with the default thresholds. A null
    ``vision_trigger.refractory_s`` means ``dispatch.cooldown_s``.
    """
    if medical is None:
        return MedicalConfig(vision_trigger=TriggerConfig(refractory_s=cooldown_s))
    trigger = medical.choice("trigger", MEDICAL_TRIGGERS)
    thresholds = medical.section("vision_trigger")
    config = MedicalConfig(
        trigger=cast(TriggerMode, trigger),
        vision_trigger=TriggerConfig(
            min_contacts=thresholds.positive_int("min_contacts"),
            contact_window_s=thresholds.positive("contact_window_s"),
            oscillation_min_s=thresholds.positive("oscillation_min_s"),
            refractory_s=thresholds.optional_non_negative("refractory_s", cooldown_s),
        ),
    )
    for section in (medical, thresholds):
        section.reject_unknown()
    return config


def _vision_config(vision: _Section) -> VisionConfig:
    """Build the validated vision configuration from its YAML section."""
    layout_name = vision.text("layout")
    try:
        get_layout(layout_name)
    except ValueError as exc:
        raise ConfigError("vision.layout", str(exc)) from exc
    model = vision.section("model")
    camera = vision.section("camera")
    evidence = vision.section("evidence")
    tracker = vision.section("tracker")
    subject = tracker.section("subject")
    skeleton = tracker.section("skeleton")
    analysis = vision.section("analysis")
    features = analysis.section("features")
    contacts = analysis.section("contacts")
    hand_shape = analysis.section("hand_shape")
    oscillation = analysis.section("oscillation")
    severity = analysis.section("severity")

    try:
        tracker_config = Tracker2DConfig(
            subject=SubjectConfig(
                rule=subject.choice("rule", ("largest", "central")),
                full_body_ratio=subject.positive("full_body_ratio"),
                face_anchor_conf=subject.fraction("face_anchor_conf"),
                wrist_conf=subject.fraction("wrist_conf"),
                face_width_source=subject.choice(
                    "face_width_source", ("ears", "eyes", "jaw", "auto")
                ),
                eye_to_face_ratio=subject.positive("eye_to_face_ratio"),
                face_width_window=subject.positive_int("face_width_window"),
            ),
            skeleton=SkeletonConfig(
                max_ratio=skeleton.positive("max_ratio"),
                ref_conf=skeleton.fraction("ref_conf"),
                ref_frames=skeleton.positive_int("ref_frames"),
                ref_min_frames=skeleton.positive_int("ref_min_frames"),
                check_conf=skeleton.fraction("check_conf"),
            ),
            min_conf=tracker.fraction("min_conf"),
            gate_chi2=tracker.positive("gate_chi2"),
            max_predict_frames=tracker.count("max_predict_frames"),
            meas_std_px=tracker.positive("meas_std_px"),
            jerk_std_pxps3=tracker.positive("jerk_std_pxps3"),
            init_pos_std_px=tracker.positive("init_pos_std_px"),
            init_vel_std_pxps=tracker.positive("init_vel_std_pxps"),
            init_acc_std_pxps2=tracker.positive("init_acc_std_pxps2"),
        )
        analysis_config = AnalysisConfig(
            features=FeatureConfig(
                hand_conf=features.fraction("hand_conf"),
                face_conf=features.fraction("face_conf"),
                occlusion_conf=features.fraction("occlusion_conf"),
                forehead_margin_fw=features.positive("forehead_margin_fw"),
            ),
            contacts=ContactConfig2D(
                contact_enter_fw=contacts.positive("contact_enter_fw"),
                contact_exit_fw=contacts.positive("contact_exit_fw"),
                occlusion_contact=contacts.fraction("occlusion_contact"),
                occlusion_near_fw=contacts.positive("occlusion_near_fw"),
                min_duration_s=contacts.non_negative("min_duration_s"),
                approach_window_s=contacts.positive("approach_window_s"),
                recoil_window_s=contacts.positive("recoil_window_s"),
            ),
            hand_shape=HandShapeConfig(
                extended_ratio=hand_shape.positive("extended_ratio"),
                pinch_ratio=hand_shape.positive("pinch_ratio"),
                min_conf=hand_shape.fraction("min_conf"),
                min_keypoints=hand_shape.count("min_keypoints"),
            ),
            oscillation=OscillationConfig(
                rocking_band_hz=oscillation.pair("rocking_band_hz"),
                flapping_band_hz=oscillation.pair("flapping_band_hz"),
                rocking_min_strength=oscillation.fraction("rocking_min_strength"),
                rocking_min_amp_fw=oscillation.positive("rocking_min_amp_fw"),
                flapping_min_strength=oscillation.fraction("flapping_min_strength"),
                flapping_height_margin_fw=oscillation.positive("flapping_height_margin_fw"),
                min_conf=oscillation.fraction("min_conf"),
                min_samples=oscillation.positive_int("min_samples"),
            ),
            severity=SeverityConfig(
                dangerous_regions=frozenset(
                    _enum_values(
                        FaceRegion, severity.texts("dangerous_regions"), "dangerous_regions"
                    )
                ),
                dangerous_shapes=frozenset(
                    _enum_values(HandShape, severity.texts("dangerous_shapes"), "dangerous_shapes")
                ),
                shape_speed_fwps=severity.positive("shape_speed_fwps"),
                strike_speed_fwps=severity.positive("strike_speed_fwps"),
                strike_decel_fwps2=severity.positive("strike_decel_fwps2"),
                recoil_fw=severity.positive("recoil_fw"),
            ),
            approach_fw=analysis.positive("approach_fw"),
            approach_speed_fwps=analysis.positive("approach_speed_fwps"),
            min_framed_fraction=analysis.fraction("min_framed_fraction"),
            max_gap_s=analysis.positive("max_gap_s"),
            activity_window_s=analysis.positive("activity_window_s"),
            repetition_band_hz=analysis.pair("repetition_band_hz"),
        )
        config = VisionConfig(
            enabled=vision.flag("enabled"),
            layout=layout_name,
            model=VisionModelConfig(
                kind=model.choice("kind", VISION_MODEL_KINDS),
                path=Path(model.text("path")).expanduser(),
                input_width=model.positive_int("input_width"),
                input_height=model.positive_int("input_height"),
                conf_thr=model.fraction("conf_thr"),
                iou_thr=model.fraction("iou_thr"),
            ),
            camera=VisionCameraConfig(
                device=camera.text("device"),
                match=camera.optional_text("match", DEFAULT_CAMERA_MATCH),
                width=camera.positive_int("width"),
                height=camera.positive_int("height"),
                fps=camera.positive("fps"),
                exposure=camera.integer("exposure", 1, 10000),
            ),
            preroll_s=vision.positive("preroll_s"),
            lookback_s=vision.positive("lookback_s"),
            fast_after_s=vision.non_negative("fast_after_s"),
            fast_budget_s=vision.positive("fast_budget_s"),
            idle_timeout_s=vision.positive("idle_timeout_s"),
            stale_after_s=vision.positive("stale_after_s"),
            analysis_fps=vision.positive("analysis_fps"),
            queue_max=vision.positive_int("queue_max"),
            imu_buffer_s=vision.positive("imu_buffer_s"),
            evidence=VisionEvidenceConfig(
                enabled=evidence.flag("enabled"),
                base_path=Path(evidence.text("base_path")).expanduser(),
                clip_s=evidence.positive("clip_s"),
                retention_days=evidence.count("retention_days"),
            ),
            tracker=tracker_config,
            analysis=analysis_config,
        )
    except ValueError as exc:
        raise ConfigError("vision", str(exc)) from exc
    sections = (
        vision,
        model,
        camera,
        evidence,
        tracker,
        subject,
        skeleton,
        analysis,
        features,
        contacts,
        hand_shape,
        oscillation,
        severity,
    )
    for section in sections:
        section.reject_unknown()
    return config


def _enum_values(enum: type[EnumT], names: tuple[str, ...], key: str) -> list[EnumT]:
    """Map YAML names to enum members, naming the offending key on error."""
    values: list[EnumT] = []
    for name in names:
        try:
            values.append(enum[name.upper()])
        except KeyError:
            message = f"unknown value {name!r}"
            raise ConfigError(f"vision.analysis.severity.{key}", message) from None
    return values


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

    def optional_section(self, name: str) -> _Section | None:
        """Return the subsection, or ``None`` when the optional section is absent."""
        if name not in self._data:
            return None
        return self.section(name)

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

    def flag(self, name: str) -> bool:
        value = self._get(name)
        if not isinstance(value, bool):
            raise ConfigError(self._key(name), "expected true or false")
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

    def non_negative(self, name: str) -> float:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(self._key(name), "expected a non-negative number")
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if not math.isfinite(number) or number < 0.0:
            raise ConfigError(self._key(name), "expected a finite non-negative number")
        return number

    def fraction(self, name: str) -> float:
        number = self.non_negative(name)
        if number > 1.0:
            raise ConfigError(self._key(name), "expected a number between 0 and 1")
        return number

    def optional_non_negative(self, name: str, default: float) -> float:
        """Return the non-negative number at ``name``, or ``default`` when it is null or absent."""
        if name not in self._data or self._data[name] is None:
            self._used.add(name)
            return default
        return self.non_negative(name)

    def pair(self, name: str) -> tuple[float, float]:
        value = self._get(name)
        if not isinstance(value, list) or len(value) != 2:
            raise ConfigError(self._key(name), "expected a list of two numbers")
        first, second = value
        numbers: list[float] = []
        for item in (first, second):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ConfigError(self._key(name), "expected a list of two numbers")
            number = float(item)
            if not math.isfinite(number):
                raise ConfigError(self._key(name), "expected finite numbers")
            numbers.append(number)
        if numbers[0] >= numbers[1]:
            raise ConfigError(self._key(name), "expected low < high")
        return (numbers[0], numbers[1])

    def texts(self, name: str) -> tuple[str, ...]:
        value = self._get(name)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ConfigError(self._key(name), "expected a list of strings")
        return tuple(value)

    def positive_int(self, name: str) -> int:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(self._key(name), "expected a positive integer")
        return value

    def count(self, name: str) -> int:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(self._key(name), "expected an integer >= 0")
        return value

    def integer(self, name: str, minimum: int, maximum: int) -> int:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(self._key(name), "expected an integer")
        if not minimum <= value <= maximum:
            raise ConfigError(
                self._key(name), f"expected an integer between {minimum} and {maximum}"
            )
        return value

    def reject_unknown(self) -> None:
        unknown = sorted(str(k) for k in self._data if k not in self._used)
        if unknown:
            raise ConfigError(self._key(unknown[0]), "unknown key")
