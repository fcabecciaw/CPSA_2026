"""Session-level bone references, plausibility and foreshortening (AC-005)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa.domain.model import Side
from cpsa.vision.layouts import HALFBUST_62
from cpsa.vision.skeleton import SkeletonConfig, SkeletonModel
from cpsa.vision.types import KeypointLayout

_LAYOUT: KeypointLayout = HALFBUST_62
_SIZE = len(_LAYOUT)
_FACE_WIDTH_PX = 80.0


def _body(scale: float = 1.0) -> dict[str, tuple[float, float]]:
    """Rigid upper body in which every reference bone is one face width."""
    return {
        "left_shoulder": (300.0, 250.0),
        "right_shoulder": (300.0 + 80.0 * scale, 250.0),
        "left_elbow": (300.0, 250.0 + 80.0 * scale),
        "right_elbow": (300.0 + 80.0 * scale, 250.0 + 80.0 * scale),
        "left_wrist": (300.0, 250.0 + 160.0 * scale),
        "right_wrist": (300.0 + 80.0 * scale, 250.0 + 160.0 * scale),
    }


def _arrays(
    points: dict[str, tuple[float, float]],
    conf: float | dict[str, float] = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.full((_SIZE, 2), np.nan, dtype=np.float64)
    confidences = np.zeros(_SIZE, dtype=np.float64)
    for name, value in points.items():
        index = _LAYOUT.index(name)
        xy[index] = value
        confidences[index] = conf[name] if isinstance(conf, dict) else conf
    return xy, confidences


def _model(ref_min_frames: int = 5, ref_frames: int = 10) -> SkeletonModel:
    return SkeletonModel(
        SkeletonConfig(ref_min_frames=ref_min_frames, ref_frames=ref_frames),
        _LAYOUT,
    )


def _collect(model: SkeletonModel, frames: int = 5, scale: float = 1.0) -> None:
    for _ in range(frames):
        xy, conf = _arrays(_body(scale))
        model.observe(xy, conf, _FACE_WIDTH_PX)


def test_references_freeze_and_stay_stable() -> None:
    model = _model(ref_min_frames=5)
    _collect(model, frames=1)
    running = dict(model.reference_fw)
    assert running
    _collect(model, frames=4)
    references = dict(model.reference_fw)
    assert set(references) == {
        "shoulder_width",
        "left_upper_arm",
        "left_forearm",
        "right_upper_arm",
        "right_forearm",
    }
    assert references == pytest.approx({name: 1.0 for name in references})

    _collect(model, frames=10, scale=2.0)
    assert dict(model.reference_fw) == references


def test_injected_bone_break_is_excluded() -> None:
    model = _model()
    _collect(model)
    normal_xy, normal_conf = _arrays(_body())
    normal = model.observe(normal_xy, normal_conf, _FACE_WIDTH_PX)
    assert bool(normal[_LAYOUT.index("left_wrist")])

    broken = _body()
    broken["left_wrist"] = (300.0, 650.0)
    conf = {name: 0.9 for name in broken}
    conf["left_wrist"] = 0.4
    xy, confidences = _arrays(broken, conf)
    accepted = model.observe(xy, confidences, _FACE_WIDTH_PX)

    assert not bool(accepted[_LAYOUT.index("left_wrist")])
    assert bool(accepted[_LAYOUT.index("left_elbow")])
    assert bool(accepted[_LAYOUT.index("right_wrist")])
    assert bool(accepted[_LAYOUT.index("right_elbow")])


def test_broken_bone_rejects_only_the_lower_confidence_joint() -> None:
    model = _model()
    _collect(model)
    broken = _body()
    broken["left_shoulder"] = (-100.0, 250.0)
    conf = {name: 0.9 for name in broken}
    conf["left_shoulder"] = 0.4
    xy, confidences = _arrays(broken, conf)
    accepted = model.observe(xy, confidences, _FACE_WIDTH_PX)

    assert not bool(accepted[_LAYOUT.index("left_shoulder")])
    assert bool(accepted[_LAYOUT.index("right_shoulder")])
    assert bool(accepted[_LAYOUT.index("left_elbow")])


def test_foreshortening_drops_for_a_shortened_forearm() -> None:
    model = _model()
    _collect(model)
    xy, _ = _arrays(_body())
    ratios = model.foreshortening(xy)
    assert ratios[Side.LEFT] == pytest.approx(1.0)
    assert ratios[Side.RIGHT] == pytest.approx(1.0)

    shortened = _body()
    shortened["left_wrist"] = (300.0, 370.0)  # forearm 40 px = 0.5 fw
    short_xy, _ = _arrays(shortened)
    short_ratios = model.foreshortening(short_xy)
    assert short_ratios[Side.LEFT] is not None
    assert short_ratios[Side.LEFT] < 1.0
    assert short_ratios[Side.LEFT] == pytest.approx(0.5)
    assert short_ratios[Side.RIGHT] == pytest.approx(1.0)

    missing = _body()
    del missing["left_elbow"]
    missing_xy, _ = _arrays(missing)
    assert model.foreshortening(missing_xy)[Side.LEFT] is None


def test_foreshortening_is_none_without_references() -> None:
    model = _model()
    xy, _ = _arrays(_body())
    assert model.foreshortening(xy) == {Side.LEFT: None, Side.RIGHT: None}


def test_no_face_width_accepts_without_collecting() -> None:
    model = _model()
    xy, conf = _arrays(_body())
    accepted = model.observe(xy, conf, None)
    for name in _body():
        assert bool(accepted[_LAYOUT.index(name)])
    assert dict(model.reference_fw) == {}


def test_references_are_returned_as_a_copy() -> None:
    model = _model()
    _collect(model)
    references = dict(model.reference_fw)
    references["left_forearm"] = 99.0
    assert model.reference_fw["left_forearm"] == pytest.approx(1.0)


def test_invalid_shapes_raise() -> None:
    model = _model()
    with pytest.raises(ValueError):
        model.observe(np.zeros((_SIZE - 1, 2)), np.zeros(_SIZE), _FACE_WIDTH_PX)
    with pytest.raises(ValueError):
        model.observe(np.zeros((_SIZE, 2)), np.zeros(_SIZE - 1), _FACE_WIDTH_PX)
    with pytest.raises(ValueError):
        model.foreshortening(np.zeros((_SIZE - 1, 2)))


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SkeletonConfig(max_ratio=1.0)
    with pytest.raises(ValueError):
        SkeletonConfig(ref_min_frames=10, ref_frames=5)
    with pytest.raises(ValueError):
        SkeletonConfig(ref_conf=0.0)
    with pytest.raises(ValueError):
        SkeletonConfig(check_conf=1.5)
