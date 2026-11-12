"""Live calibration: validation, the overrides file and the worker (debug panel)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from cpsa.calibration import Calibration, CalibrationStore, LatestWorker, parse_update
from cpsa.config import load

REPO_ROOT = Path(__file__).resolve().parents[1]

SHIPPED = load(REPO_ROOT / "config" / "showcase.yaml")


def test_the_yaml_is_the_base_calibration() -> None:
    base = Calibration.from_config(SHIPPED)

    assert base.zoom == SHIPPED.model.zoom
    assert base.zoom_center == SHIPPED.model.zoom_center
    assert (base.gain, base.exposure, base.saturation) == (
        SHIPPED.camera.gain,
        SHIPPED.camera.exposure,
        SHIPPED.camera.saturation,
    )
    assert base.applied_to(SHIPPED) == SHIPPED


def test_a_calibration_lands_in_the_model_and_camera_sections_only() -> None:
    tuned = Calibration(zoom=1.7, zoom_center=(0.3, 0.6), gain=900, exposure=200, saturation=40)

    config = tuned.applied_to(SHIPPED)

    assert (config.model.zoom, config.model.zoom_center) == (1.7, (0.3, 0.6))
    assert (config.camera.gain, config.camera.exposure, config.camera.saturation) == (900, 200, 40)
    assert config.model.path == SHIPPED.model.path
    assert config.camera.white_balance == SHIPPED.camera.white_balance
    assert config.tracker == SHIPPED.tracker


def test_any_subset_is_accepted_and_normalized() -> None:
    assert parse_update({"zoom": 1}) == {"zoom": 1.0}
    assert parse_update({"zoom_center": [0, 1]}) == {"zoom_center": (0.0, 1.0)}
    assert parse_update({"gain": 640.0, "exposure": 200, "saturation": 0}) == {
        "gain": 640,
        "exposure": 200,
        "saturation": 0,
    }
    assert parse_update({"zoom": 2.0, "gain": 2540, "exposure": 100, "saturation": 255}) == {
        "zoom": 2.0,
        "gain": 2540,
        "exposure": 100,
        "saturation": 255,
    }


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"zoom": 0.99},
        {"zoom": 2.01},
        {"zoom": True},
        {"zoom": float("inf")},
        {"zoom_center": "0.5,0.5"},
        {"zoom_center": [0.5, 0.5, 0.5]},
        {"zoom_center": [-0.1, 0.5]},
        {"gain": 63},
        {"gain": 2541},
        {"gain": 64.5},
        {"exposure": 300},
        {"exposure": "100"},
        {"saturation": -1},
        {"saturation": None},
        {"zoom": 1.2, "white_balance": 4000},
    ],
)
def test_a_bad_update_is_rejected(data: Any) -> None:
    with pytest.raises(ValueError):
        parse_update(data)


def test_the_store_round_trips_atomically(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path / "nested" / "calibration.json")
    tuned = Calibration(zoom=1.55, zoom_center=(0.5, 0.45), gain=320, exposure=200, saturation=110)

    assert store.load() == {}
    store.save(tuned)

    assert Calibration.from_config(SHIPPED).updated(store.load()) == tuned
    assert json.loads(store.path.read_text(encoding="utf-8"))["zoom_center"] == [0.5, 0.45]
    assert [path.name for path in store.path.parent.iterdir()] == ["calibration.json"]

    store.remove()
    assert not store.path.exists()
    store.remove()  # a second reset is harmless


@pytest.mark.parametrize("text", ["{not json", '{"gain": 1}', '["zoom"]'])
def test_a_broken_file_is_ignored(tmp_path: Path, text: str) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(text, encoding="utf-8")

    assert CalibrationStore(path).load() == {}


def test_a_partial_file_overrides_only_its_keys(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text('{"gain": 1000}', encoding="utf-8")

    calibration = Calibration.from_config(SHIPPED).updated(CalibrationStore(path).load())

    assert calibration.gain == 1000
    assert calibration.zoom == SHIPPED.model.zoom


def test_the_worker_applies_the_newest_value_off_the_caller_thread() -> None:
    gate = threading.Event()
    started = threading.Event()
    applied: list[int] = []
    threads: set[str] = set()

    def apply(value: int) -> None:
        threads.add(threading.current_thread().name)
        started.set()
        gate.wait(2.0)
        applied.append(value)

    worker: LatestWorker[int] = LatestWorker(apply, name="test-calibration")
    worker.submit(1)
    assert started.wait(2.0)
    for value in range(2, 6):
        worker.submit(value)  # collapse while the first one is applied
    gate.set()

    assert worker.wait_idle(2.0)
    assert applied == [1, 5]
    assert threads == {"test-calibration"}

    worker.close()
    worker.submit(9)
    assert applied[-1] == 5


def test_a_failing_application_does_not_stop_the_worker() -> None:
    applied: list[int] = []

    def apply(value: int) -> None:
        if value == 1:
            raise OSError("disk full")
        applied.append(value)

    worker: LatestWorker[int] = LatestWorker(apply, name="test-calibration")
    worker.submit(1)
    assert worker.wait_idle(2.0)
    worker.submit(2)
    assert worker.wait_idle(2.0)
    worker.close()

    assert applied == [2]


def test_the_detector_is_validated_persisted_and_applied(tmp_path: Path) -> None:
    from dataclasses import replace

    assert parse_update({"detector": "yolov5"}) == {"detector": "yolov5"}
    for bad in ("yolov8", 5, None, ["yolox"]):
        with pytest.raises(ValueError, match="detector"):
            parse_update({"detector": bad})

    store = CalibrationStore(tmp_path / "calibration.json")
    tuned = replace(Calibration.from_config(SHIPPED), detector="yolov5")
    store.save(tuned)

    assert store.load()["detector"] == "yolov5"
    assert Calibration.from_config(SHIPPED).updated(store.load()) == tuned
    assert tuned.applied_to(SHIPPED).model.kind == "yolov5"


def test_yolov5_is_available_only_when_its_model_is_installed(tmp_path: Path) -> None:
    from dataclasses import replace

    from cpsa.calibration import available_detectors

    missing = replace(SHIPPED, model=replace(SHIPPED.model, yolov5_path=tmp_path / "none.xmodel"))
    installed_path = tmp_path / "yolov5_nano.xmodel"
    installed_path.write_bytes(b"x")
    installed = replace(SHIPPED, model=replace(SHIPPED.model, yolov5_path=installed_path))

    assert available_detectors(missing) == ("yolox",)
    assert available_detectors(installed) == ("yolox", "yolov5")


def test_the_shipped_thresholds_are_the_base_and_held_keeps_its_offsets() -> None:
    base = Calibration.from_config(SHIPPED)
    assert (base.person_conf, base.held_conf, base.object_conf) == (0.40, 0.30, 0.45)

    model = (
        base.updated({"person_conf": 0.6, "held_conf": 0.2, "object_conf": 0.7})
        .applied_to(SHIPPED)
        .model
    )

    assert model.object_conf == 0.7
    assert model.class_conf["person"] == 0.6
    # bottle and cup were the loosest (0.30), cell phone and book 0.05 above.
    assert {name: model.class_conf[name] for name in ("bottle", "cup", "cell phone", "book")} == {
        "bottle": 0.2,
        "cup": 0.2,
        "cell phone": 0.25,
        "book": 0.25,
    }
    # Applying twice is stable (the offsets survive a shift).
    again = base.updated({"held_conf": 0.2}).applied_to(SHIPPED)
    assert Calibration.from_config(again).held_conf == 0.2
    assert again.model.class_conf["book"] == 0.25


def test_the_thresholds_are_validated_and_persisted(tmp_path: Path) -> None:
    assert parse_update({"person_conf": 0.1, "held_conf": 0.9, "object_conf": 0.5}) == {
        "person_conf": 0.1,
        "held_conf": 0.9,
        "object_conf": 0.5,
    }
    for key in ("person_conf", "held_conf", "object_conf"):
        for bad in (0.05, 0.95, "0.5", True):
            with pytest.raises(ValueError, match=key):
                parse_update({key: bad})

    store = CalibrationStore(tmp_path / "calibration.json")
    tuned = Calibration.from_config(SHIPPED).updated({"person_conf": 0.55, "object_conf": 0.65})
    store.save(tuned)
    assert Calibration.from_config(SHIPPED).updated(store.load()) == tuned
