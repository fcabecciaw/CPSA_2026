import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from cpsa.application.config import ConfigError, load, parse
from cpsa.domain.model import FaceRegion

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "config" / "board.yaml"


def valid() -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(SHIPPED.read_text()))


def test_shipped_config_is_valid() -> None:
    config = load(SHIPPED)
    assert config.bluecoins.left == "CPSA_L2"
    assert config.dispatch.verification_deadline_s == 1.0
    assert config.actuators.retry.fast_attempts == 5
    assert config.actuators.led_strip.duration_ms == 2500


def test_shipped_camera_defaults() -> None:
    config = load(SHIPPED)
    assert config.camera.rgb_device == "/dev/video0"
    assert config.camera.depth_device == "/dev/video2"
    assert (config.camera.width, config.camera.height) == (640, 480)
    assert config.camera.fps == 30
    assert config.camera.rgb_exposure == 300
    assert config.camera.xu.laser_power == 16
    assert config.camera.xu.accuracy == 2
    assert config.camera.xu.filter == 6
    assert config.camera.xu.confidence == 1


def test_home_is_expanded_in_log_base_path() -> None:
    assert not str(parse(valid()).log.base_path).startswith("~")


def _without(data: dict[str, Any], dotted: str) -> dict[str, Any]:
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node[part]
    del node[leaf]
    return data


def _with(data: dict[str, Any], dotted: str, value: object) -> dict[str, Any]:
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node[part]
    node[leaf] = value
    return data


@pytest.mark.parametrize(
    "key",
    [
        "log",
        "bluecoins.right",
        "dispatch.verification_deadline_s",
        "actuators.retry.sleep_s",
        "camera",
        "camera.depth_device",
        "camera.xu.laser_power",
    ],
)
def test_missing_key_is_named(key: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse(_without(valid(), key))
    assert info.value.key == key
    assert key in str(info.value)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sync.max_skew_ms", 0),
        ("dispatch.cooldown_s", -1),
        ("dispatch.verification_deadline_s", "fast"),
        ("bluecoins.discovery_retries", 1.5),
        ("bluecoins.discovery_retries", -1),
        ("actuators.retry.fast_attempts", True),
        ("actuators.metamotion.enable", "yes"),
        ("actuators.led_strip.duration_ms", 2.5),
        ("actuators.speaker.mac", ""),
        ("log.level", "VERBOSE"),
        ("bluecoins", ["CPSA_L2"]),
        ("camera.width", 0),
        ("camera.height", -1),
        ("camera.fps", 0),
        ("camera.fps", "fast"),
        ("camera.rgb_exposure", 82),
        ("camera.rgb_exposure", 10001),
        ("camera.rgb_exposure", 300.5),
        ("camera.xu.laser_power", -1),
        ("camera.xu.laser_power", 17),
        ("camera.xu.accuracy", 0),
        ("camera.xu.accuracy", 4),
        ("camera.xu.filter", 8),
        ("camera.xu.confidence", 16),
        ("camera.xu.confidence", True),
    ],
)
def test_invalid_value_is_named(key: str, value: object) -> None:
    with pytest.raises(ConfigError) as info:
        parse(_with(valid(), key, value))
    assert info.value.key == key


def test_zero_retries_are_allowed() -> None:
    assert parse(_with(valid(), "bluecoins.discovery_retries", 0)).bluecoins.discovery_retries == 0


def test_unknown_key_is_named() -> None:
    with pytest.raises(ConfigError) as info:
        parse(_with(valid(), "sync.max_skew", 30))
    assert info.value.key == "sync.max_skew"


def test_unknown_camera_key_is_named() -> None:
    with pytest.raises(ConfigError) as info:
        parse(_with(valid(), "camera.xu.gain", 3))
    assert info.value.key == "camera.xu.gain"


def test_non_mapping_document_is_rejected() -> None:
    with pytest.raises(ConfigError) as info:
        parse(["not", "a", "mapping"])
    assert info.value.key == "<root>"


def test_unreadable_and_malformed_files_name_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match="missing.yaml"):
        load(missing)
    broken = tmp_path / "broken.yaml"
    broken.write_text("log: [unclosed\n")
    with pytest.raises(ConfigError, match="broken.yaml"):
        load(broken)


@pytest.mark.parametrize("literal", [".nan", ".inf", "-.inf", "1.0e+400"])
def test_nonfinite_numbers_are_rejected(literal: str) -> None:
    text = SHIPPED.read_text().replace("max_skew_ms: 30", f"max_skew_ms: {literal}")
    with pytest.raises(ConfigError) as info:
        parse(yaml.safe_load(text))
    assert info.value.key == "sync.max_skew_ms"


def test_shipped_vision_defaults() -> None:
    config = load(SHIPPED)
    vision = config.vision

    assert vision.enabled is False
    assert vision.layout == "HALFBUST_62"
    assert vision.model.kind == "onnx"
    assert (vision.model.input_width, vision.model.input_height) == (640, 480)
    assert vision.camera.device == "/dev/video0"
    assert (vision.camera.width, vision.camera.height) == (640, 480)
    assert vision.fast_budget_s == 0.8
    assert vision.fast_budget_s < config.dispatch.verification_deadline_s
    assert vision.analysis_fps == 10.0
    assert vision.evidence.base_path == Path("~/Desktop/CPSA_evidence").expanduser()
    assert vision.tracker.min_conf == 0.3
    assert vision.analysis.approach_fw == 0.4
    assert vision.analysis.severity.dangerous_regions == frozenset({FaceRegion.EYES})
    assert vision.analysis.repetition_band_hz == (0.2, 5.0)


@pytest.mark.parametrize(
    "key",
    [
        "vision",
        "vision.layout",
        "vision.model",
        "vision.camera",
        "vision.fast_budget_s",
        "vision.tracker",
        "vision.tracker.subject",
        "vision.analysis",
        "vision.analysis.contacts",
        "vision.evidence.retention_days",
    ],
)
def test_missing_vision_key_is_named(key: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse(_without(valid(), key))
    assert info.value.key == key


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("vision.enabled", "yes"),
        ("vision.layout", "NO_SUCH_LAYOUT"),
        ("vision.model.kind", "tflite"),
        ("vision.model.conf_thr", 1.5),
        ("vision.model.iou_thr", -0.1),
        ("vision.camera.width", 0),
        ("vision.camera.exposure", 0),
        ("vision.preroll_s", 0),
        ("vision.lookback_s", -1),
        ("vision.fast_after_s", -0.1),
        ("vision.fast_budget_s", 0),
        ("vision.idle_timeout_s", 0),
        ("vision.stale_after_s", float("nan")),
        ("vision.analysis_fps", 0),
        ("vision.queue_max", 0),
        ("vision.imu_buffer_s", 0),
        ("vision.evidence.enabled", "true"),
        ("vision.evidence.retention_days", -1),
        ("vision.tracker.min_conf", 1.5),
        ("vision.tracker.subject.rule", "biggest"),
        ("vision.analysis.approach_fw", 0),
        ("vision.analysis.severity.dangerous_regions", ["jaw"]),
        ("vision.analysis.repetition_band_hz", [5.0, 0.2]),
    ],
)
def test_invalid_vision_value_is_named(key: str, value: object) -> None:
    with pytest.raises(ConfigError) as info:
        parse(_with(valid(), key, value))
    assert info.value.key == key


def test_vision_fast_budget_must_fit_the_dispatch_deadline() -> None:
    with pytest.raises(ConfigError) as info:
        parse(_with(valid(), "vision.fast_budget_s", 1.0))
    assert info.value.key == "vision.fast_budget_s"


def test_unknown_vision_key_is_named() -> None:
    with pytest.raises(ConfigError) as info:
        parse(_with(valid(), "vision.analysis_stride", 2))
    assert info.value.key == "vision.analysis_stride"


def test_server_and_supervisor_defaults() -> None:
    config = load(SHIPPED)

    assert config.server.host == "0.0.0.0"
    assert config.server.port == 8080
    assert config.server.static_dir == Path("web/dist")
    assert config.server.idle_timeout_s == 30.0
    assert config.supervisor.default_mode == "medical"


def test_server_and_supervisor_sections_are_loaded() -> None:
    data = valid()
    data["server"] = {
        "host": "127.0.0.1",
        "port": 9000,
        "static_dir": "~/web/build",
        "idle_timeout_s": 2.5,
    }
    data["supervisor"] = {"default_mode": "idle"}

    config = parse(data)

    assert config.server.host == "127.0.0.1"
    assert config.server.port == 9000
    assert config.server.static_dir == Path("~/web/build").expanduser()
    assert config.server.idle_timeout_s == 2.5
    assert config.supervisor.default_mode == "idle"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("host", ""),
        ("port", 0),
        ("port", 65536),
        ("port", "8080"),
        ("port", True),
        ("static_dir", ""),
        ("idle_timeout_s", 0),
        ("extra", True),
    ],
)
def test_invalid_server_value_is_named(key: str, value: object) -> None:
    data = valid()
    data["server"] = {
        "host": "0.0.0.0",
        "port": 8080,
        "static_dir": "web/dist",
        "idle_timeout_s": 30,
        key: value,
    }

    with pytest.raises(ConfigError) as info:
        parse(data)
    assert info.value.key == f"server.{key}"


def test_missing_server_key_is_named_when_the_section_is_present() -> None:
    data = valid()
    data["server"] = {"host": "0.0.0.0", "static_dir": "web/dist", "idle_timeout_s": 30}

    with pytest.raises(ConfigError) as info:
        parse(data)
    assert info.value.key == "server.port"


@pytest.mark.parametrize("value", ["lab", "MEDICAL", "", 1])
def test_invalid_default_mode_is_named(value: object) -> None:
    data = valid()
    data["supervisor"] = {"default_mode": value}

    with pytest.raises(ConfigError) as info:
        parse(data)
    assert info.value.key == "supervisor.default_mode"


def test_unknown_supervisor_key_is_named() -> None:
    data = valid()
    data["supervisor"] = {"default_mode": "medical", "fallback": "idle"}

    with pytest.raises(ConfigError) as info:
        parse(data)
    assert info.value.key == "supervisor.fallback"
