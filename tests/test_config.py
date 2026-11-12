"""The one configuration file: defaults, validation and error keys (F-K3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cpsa.config import ConfigError, load, parse
from cpsa.yolox import COCO80_EN, class_thresholds

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "config" / "showcase.yaml"

DOCUMENT = {
    "camera": {
        "device": "auto",
        "match": "RealSense",
        "width": 640,
        "height": 480,
        "fps": 60,
        "exposure": 100,
        "gain": 200,
        "white_balance": 4000,
        "backlight_compensation": 0,
        "saturation": 110,
        "power_line_hz": 50,
    },
    "tracker": {
        "enabled": True,
        "iou_match": 0.3,
        "alpha": 0.35,
        "fast_alpha": 0.8,
        "deadband_iou": 0.9,
        "min_hits": 2,
        "max_misses": 6,
        "still": {"enabled": True, "motion_thr": 1.0, "force_every_s": 0.5},
    },
    "model": {"path": "models/yolox_nano.xmodel", "conf_thr": 0.35, "iou_thr": 0.45},
    "movenet": {
        "path": "models/movenet.xmodel",
        "enabled": True,
        "publish_hz": 30,
        "smoothing": {
            "enabled": True,
            "min_cutoff": 1.0,
            "beta": 0.02,
            "d_cutoff": 1.0,
            "deadband_px": 1.5,
            "show_score": 0.35,
            "hide_score": 0.25,
            "show_frames": 2,
            "hide_frames": 4,
        },
    },
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


def test_the_camera_mirrors_by_default_and_can_be_turned_off(tmp_path: Path) -> None:
    shipped = load(SHIPPED)
    assert shipped.camera.mirror is True

    text = SHIPPED.read_text(encoding="utf-8").replace("mirror: true ", "mirror: false", 1)
    custom = tmp_path / "showcase.yaml"
    custom.write_text(text, encoding="utf-8")
    assert load(custom).camera.mirror is False


def test_the_shipped_configuration_parses_and_names_the_machine() -> None:
    config = load(SHIPPED)

    assert config.model.name == "yolox_nano"
    assert config.model.path == Path("models/yolox_nano.xmodel")
    assert config.model.conf_thr == 0.35
    assert config.model.class_conf == {"person": 0.40}
    assert config.model.object_conf == 0.25
    assert config.model.zoom == 1.3
    assert config.model.zoom_center == (0.5, 0.5)
    assert config.movenet.name == "movenet"
    assert config.movenet.path == Path("models/movenet.xmodel")
    assert config.movenet.enabled is True
    assert config.movenet.publish_hz == 30.0
    assert config.movenet.smoothing.enabled is True
    assert config.movenet.smoothing.min_cutoff == 1.0
    assert config.movenet.smoothing.beta == 0.02
    assert config.movenet.smoothing.d_cutoff == 1.0
    assert config.movenet.smoothing.deadband_px == 1.5
    assert config.movenet.smoothing.show_score == 0.35
    assert config.movenet.smoothing.hide_score == 0.25
    assert config.movenet.smoothing.show_frames == 2
    assert config.movenet.smoothing.hide_frames == 4
    assert config.camera.device == "auto"
    assert config.camera.width == 640 and config.camera.height == 480
    assert config.camera.fps == 60.0
    assert config.camera.exposure == 100
    assert config.camera.gain == 200
    assert config.camera.white_balance == 4000
    assert config.camera.backlight_compensation == 0
    assert config.camera.saturation == 110
    assert config.camera.power_line_hz == 50
    assert config.tracker.enabled is True
    assert config.tracker.iou_match == 0.3
    assert config.tracker.alpha == 0.35
    assert config.tracker.fast_alpha == 0.8
    assert config.tracker.deadband_iou == 0.9
    assert config.tracker.min_hits == 2
    assert config.tracker.max_misses == 6
    assert config.tracker.still.enabled is True
    assert config.tracker.still.motion_thr == 1.0
    assert config.tracker.still.force_every_s == 0.5
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


def test_a_document_without_class_thresholds_keeps_the_global_default() -> None:
    model = parse(document()).model

    assert model.class_conf == {}
    assert model.object_conf is None
    assert model.zoom == 1.0
    assert model.zoom_center == (0.5, 0.5)
    assert class_thresholds(model.conf_thr, class_conf=model.class_conf) == {
        class_id: model.conf_thr for class_id in range(len(COCO80_EN))
    }


@pytest.mark.parametrize("zoom", [1.0, 1.3, 1.5, 2.0])
def test_the_model_zoom_parses(zoom: float) -> None:
    data = document()
    data["model"]["zoom"] = zoom  # type: ignore[index]

    assert parse(data).model.zoom == zoom


def test_the_zoom_center_parses_and_must_be_a_fraction_pair() -> None:
    data = document()
    data["model"]["zoom_center"] = [0.4, 0.6]  # type: ignore[index]

    assert parse(data).model.zoom_center == (0.4, 0.6)


@pytest.mark.parametrize(
    "value",
    [0.9, 2.1, 0, "1.3", True, [1.3], None],
)
def test_an_invalid_zoom_is_named(value: object) -> None:
    data = document()
    data["model"]["zoom"] = value  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "model.zoom"


@pytest.mark.parametrize(
    "value",
    [[0.5], [0.5, 0.5, 0.5], [1.5, 0.5], [-0.1, 0.5], "centre", {"x": 0.5}, [0.5, True]],
)
def test_an_invalid_zoom_center_is_named(value: object) -> None:
    data = document()
    data["model"]["zoom_center"] = value  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "model.zoom_center"


def test_the_class_thresholds_resolve_to_the_showcase_classes() -> None:
    model = load(SHIPPED).model

    thresholds = class_thresholds(
        model.conf_thr, class_conf=model.class_conf, object_conf=model.object_conf
    )

    assert len(thresholds) == len(COCO80_EN)
    assert thresholds[COCO80_EN.index("person")] == 0.40
    assert thresholds[COCO80_EN.index("bottle")] == 0.25
    assert set(thresholds.values()) == {0.40, 0.25}


@pytest.mark.parametrize(
    "section, key",
    [
        ("camera", "width"),
        ("camera", "exposure"),
        ("camera", "white_balance"),
        ("camera", "backlight_compensation"),
        ("camera", "saturation"),
        ("tracker", "deadband_iou"),
        ("tracker", "enabled"),
        ("model", "path"),
        ("movenet", "path"),
        ("movenet", "enabled"),
        ("movenet", "publish_hz"),
        ("movenet", "smoothing"),
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
        ("camera", "gain", 63),  # the F200 offers 64..2540
        ("camera", "gain", 2541),
        ("camera", "white_balance", "manual"),
        ("camera", "white_balance", 2450),
        ("camera", "white_balance", 6501),
        ("camera", "white_balance", 4050),  # not a 100 K step
        ("camera", "white_balance", True),
        ("camera", "backlight_compensation", -1),
        ("camera", "backlight_compensation", 5),
        ("camera", "saturation", -1),
        ("camera", "saturation", 256),
        ("camera", "power_line_hz", 30),
        ("camera", "device", ""),
        ("tracker", "enabled", "yes"),
        ("tracker", "iou_match", 1.5),
        ("tracker", "alpha", -0.1),
        ("tracker", "fast_alpha", 2),
        ("tracker", "deadband_iou", -1),
        ("tracker", "min_hits", -1),
        ("tracker", "max_misses", -1),
        ("model", "conf_thr", 1.5),
        ("model", "iou_thr", -0.1),
        ("model", "object_conf", 1.5),
        ("movenet", "enabled", "yes"),
        ("movenet", "publish_hz", 0),
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


@pytest.mark.parametrize(
    ("class_conf", "key"),
    [
        (["person"], "model.class_conf"),
        ({"person": 1.5}, "model.class_conf.person"),
        ({"person": "high"}, "model.class_conf.person"),
        ({"": 0.4}, "model.class_conf."),
    ],
)
def test_an_invalid_class_threshold_is_named(class_conf: object, key: str) -> None:
    data = document()
    data["model"]["class_conf"] = class_conf  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == key


def test_white_balance_accepts_auto_and_the_control_range() -> None:
    data = document()
    data["camera"]["white_balance"] = "auto"  # type: ignore[index]
    assert parse(data).camera.white_balance == "auto"

    for temperature in (2500, 4000, 6500):
        data["camera"]["white_balance"] = temperature  # type: ignore[index]
        assert parse(data).camera.white_balance == temperature


def test_the_pose_model_can_be_disabled_or_slowed() -> None:
    data = document()
    data["movenet"]["enabled"] = False  # type: ignore[index]
    data["movenet"]["publish_hz"] = 15  # type: ignore[index]

    config = parse(data)

    assert config.movenet.enabled is False
    assert config.movenet.publish_hz == 15.0


def test_the_keypoint_smoothing_can_be_disabled() -> None:
    data = document()
    data["movenet"]["smoothing"]["enabled"] = False  # type: ignore[index]

    assert parse(data).movenet.smoothing.enabled is False


def test_a_missing_smoothing_key_is_named() -> None:
    data = document()
    del data["movenet"]["smoothing"]["beta"]  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "movenet.smoothing.beta"


@pytest.mark.parametrize(
    "key, value",
    [
        ("enabled", "yes"),
        ("min_cutoff", 0),
        ("beta", 0),
        ("d_cutoff", -1),
        ("deadband_px", 0),
        ("show_score", 1.5),
        ("hide_score", -0.1),
        ("show_frames", 0),
        ("hide_frames", 101),
        ("show_frames", 2.5),
    ],
)
def test_an_invalid_smoothing_value_is_named(key: str, value: object) -> None:
    data = document()
    data["movenet"]["smoothing"][key] = value  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == f"movenet.smoothing.{key}"


def test_the_hide_score_must_stay_below_the_show_score() -> None:
    data = document()
    data["movenet"]["smoothing"]["hide_score"] = 0.35  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "movenet.smoothing.hide_score"


def test_an_unknown_smoothing_key_is_rejected() -> None:
    data = document()
    data["movenet"]["smoothing"]["kalman"] = True  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "movenet.smoothing.kalman"


def test_the_tracker_can_be_disabled() -> None:
    data = document()
    data["tracker"]["enabled"] = False  # type: ignore[index]

    assert parse(data).tracker.enabled is False


def test_the_still_gate_is_on_by_default_and_can_be_disabled() -> None:
    data = document()
    data["tracker"]["still"]["enabled"] = False  # type: ignore[index]

    assert parse(data).tracker.still.enabled is False


def test_a_missing_still_key_is_named() -> None:
    data = document()
    del data["tracker"]["still"]["force_every_s"]  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "tracker.still.force_every_s"


def test_a_missing_still_block_is_named() -> None:
    data = document()
    del data["tracker"]["still"]  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "tracker.still"


@pytest.mark.parametrize(
    "key, value",
    [
        ("enabled", "yes"),
        ("motion_thr", 0),
        ("motion_thr", -1),
        ("force_every_s", 0),
        ("force_every_s", "half"),
    ],
)
def test_an_invalid_still_value_is_named(key: str, value: object) -> None:
    data = document()
    data["tracker"]["still"][key] = value  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == f"tracker.still.{key}"


def test_an_unknown_still_key_is_rejected() -> None:
    data = document()
    data["tracker"]["still"]["frames"] = 3  # type: ignore[index]

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == "tracker.still.frames"


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
