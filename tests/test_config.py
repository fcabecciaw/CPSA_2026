"""The one configuration file: defaults, validation and error keys (F-K3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cpsa.config import ConfigError, load, parse

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "config" / "showcase.yaml"

DOCUMENT = {
    "camera": {
        "device": "auto",
        "match": "RealSense",
        "width": 640,
        "height": 480,
        "fps": 60,
        "exposure": 150,
        "gain": 320,
        "power_line_hz": 50,
    },
    "model": {"path": "models/yolox_nano.xmodel", "conf_thr": 0.35, "iou_thr": 0.45},
    "labels": "it",
    "web": {
        "host": "0.0.0.0",
        "port": 8080,
        "video_fps": 60,
        "jpeg_quality": 70,
        "static_dir": "web/dist",
    },
    "log": {"level": "INFO"},
}


def document() -> dict[str, object]:
    import copy

    return copy.deepcopy(DOCUMENT)


def test_the_shipped_configuration_parses_and_names_the_machine() -> None:
    config = load(SHIPPED)

    assert config.model.name == "yolox_nano"
    assert config.model.path == Path("models/yolox_nano.xmodel")
    assert config.camera.device == "auto"
    assert config.camera.width == 640 and config.camera.height == 480
    assert config.camera.fps == 60.0
    assert config.camera.exposure == 150
    assert config.camera.gain == 320
    assert config.camera.power_line_hz == 50
    assert config.web.port == 8080
    assert config.web.video_fps == 60.0
    assert config.web.jpeg_quality == 70
    assert config.web.static_dir == Path("web/dist")
    assert config.labels == "it"
    assert config.log.level == "INFO"


def test_a_document_parses_with_the_default_camera_match() -> None:
    data = document()
    del data["camera"]["match"]  # type: ignore[index]

    assert parse(data).camera.match == "RealSense"


@pytest.mark.parametrize(
    "section, key",
    [
        ("camera", "width"),
        ("camera", "exposure"),
        ("model", "path"),
        ("web", "jpeg_quality"),
        ("log", "level"),
    ],
)
def test_a_missing_key_is_named(section: str, key: str) -> None:
    data = document()
    del data[section][key]  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == f"{section}.{key}"


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("camera", "width", 639),  # YUYV needs an even width
        ("camera", "width", 0),
        ("camera", "fps", 0),
        ("camera", "fps", float("nan")),
        ("camera", "exposure", 0),
        ("camera", "gain", 0),
        ("camera", "power_line_hz", 30),
        ("camera", "device", ""),
        ("model", "conf_thr", 1.5),
        ("model", "iou_thr", -0.1),
        ("web", "port", 0),
        ("web", "port", 70000),
        ("web", "jpeg_quality", 101),
        ("web", "video_fps", -1),
        ("log", "level", "LOUD"),
    ],
)
def test_an_invalid_value_is_named(section: str, key: str, value: object) -> None:
    data = document()
    data[section][key] = value  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == f"{section}.{key}"


def test_labels_are_italian_or_english() -> None:
    data = document()
    data["labels"] = "de"  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "labels"


def test_an_unknown_key_is_rejected_instead_of_ignored() -> None:
    data = document()
    data["supervisor"] = {"default_mode": "showcase"}  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "supervisor"

    data = document()
    data["camera"]["depth_device"] = "auto"  # type: ignore[index]
    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "camera.depth_device"


def test_a_non_mapping_document_is_rejected() -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse([1, 2, 3])
    assert excinfo.value.key == "<root>"

    with pytest.raises(ConfigError) as excinfo:
        parse({"camera": "auto"})
    assert excinfo.value.key == "camera"


def test_unreadable_and_malformed_files_name_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError) as excinfo:
        load(missing)
    assert "cannot read file" in excinfo.value.reason

    broken = tmp_path / "broken.yaml"
    broken.write_text("camera: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load(broken)
    assert "invalid YAML" in excinfo.value.reason


def test_the_model_name_is_the_file_stem(tmp_path: Path) -> None:
    data = document()
    data["model"]["path"] = str(tmp_path / "other_detector.xmodel")  # type: ignore[index]

    assert parse(data).model.name == "other_detector"
