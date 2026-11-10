"""Hand-to-face features in face-width units (FR-012, FR-013, AC-006)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa.domain.model import FaceRegion, HandShape, Side
from cpsa.vision.handface import FeatureConfig, hand_face_samples
from cpsa.vision.types import KeypointStatus
from tests.unit import vision_fixtures as fx

CFG = FeatureConfig()
CHIN = fx.face_anchor("face_chin")


def _samples(frame, layout=fx.LAYOUT, cfg=CFG):
    return hand_face_samples(frame, layout, cfg)


def _one(frame, side: Side = Side.RIGHT):
    samples = [sample for sample in _samples(frame) if sample.side is side]
    assert len(samples) == 1
    return samples[0]


def test_distance_closing_speed_and_time_to_contact() -> None:
    frames = fx.distance_profile(
        [(0.0, 1.0), (0.5, 0.2)],
        side=Side.RIGHT,
        shape=HandShape.FIST,
        anchor=CHIN,
    )

    sample = _one(frames[0])

    assert sample.distance_fw == pytest.approx(1.0, abs=1e-9)
    # The hand closes 0.8 fw in 0.5 s.
    assert sample.closing_speed_fwps == pytest.approx(1.6, rel=0.02)
    assert sample.approach_angle_deg == pytest.approx(0.0, abs=1e-6)
    assert sample.time_to_contact_s == pytest.approx(1.0 / 1.6, rel=0.03)


def test_receding_hand_has_negative_closing_and_no_time_to_contact() -> None:
    frames = fx.distance_profile(
        [(0.0, 0.2), (0.5, 1.0)],
        side=Side.RIGHT,
        shape=HandShape.OPEN_PALM,
        anchor=CHIN,
    )

    sample = _one(frames[0])

    assert sample.closing_speed_fwps == pytest.approx(-1.6, rel=0.02)
    assert sample.approach_angle_deg == pytest.approx(180.0, abs=1e-6)
    assert sample.time_to_contact_s is None


@pytest.mark.parametrize(
    ("anchor", "region"),
    [
        ("left_eye", FaceRegion.EYES),
        ("face_mouth_left", FaceRegion.MOUTH),
        ("left_ear", FaceRegion.EAR),
        ("face_jaw_left", FaceRegion.NOSE_CHEEK),
    ],
)
def test_region_comes_from_the_nearest_anchor(anchor: str, region: FaceRegion) -> None:
    xy = fx.neutral_xy()
    fx.set_hand_distance(xy, Side.RIGHT, fx.face_anchor(anchor), 0.1, HandShape.FIST)

    sample = _one(fx.make_frame(0, 0.0, xy))

    assert sample.nearest_region is region


def test_point_above_the_brow_line_is_forehead_temple() -> None:
    xy = fx.neutral_xy()
    brow_y = fx.face_anchor("face_brow_left")[1]
    fx.set_hand(
        xy,
        Side.RIGHT,
        (fx.FACE_CENTER[0], brow_y - 0.2 * fx.FACE_WIDTH_PX),
        HandShape.FIST,
    )

    sample = _one(fx.make_frame(0, 0.0, xy))

    assert sample.nearest_region is FaceRegion.FOREHEAD_TEMPLE
    assert sample.distance_fw == pytest.approx(0.32, abs=0.02)


def test_occlusion_fraction_counts_missing_and_low_confidence_anchors() -> None:
    xy = fx.neutral_xy()
    status = np.full(fx.KEYPOINT_COUNT, int(KeypointStatus.MEASURED), dtype=np.int8)
    conf = np.full(fx.KEYPOINT_COUNT, 0.9, dtype=np.float64)
    lost = ["left_eye", "right_eye", "left_ear", "face_chin"]
    for name in lost:
        status[fx.LAYOUT.index(name)] = int(KeypointStatus.LOST)
    for name in ("face_lip_upper", "face_lip_lower"):
        conf[fx.LAYOUT.index(name)] = 0.1
    frame = fx.make_frame(0, 0.0, xy, conf=conf, status=status)

    sample = _one(frame)

    assert sample.occlusion_fraction == pytest.approx(6.0 / len(fx.LAYOUT.face))


def test_unknown_face_width_leaves_metric_fields_none() -> None:
    frames = fx.distance_profile(
        [(0.0, 0.5), (0.5, 0.1)],
        side=Side.RIGHT,
        shape=HandShape.FIST,
        anchor=CHIN,
    )
    source = frames[1]
    frame = fx.make_frame(
        source.seq,
        source.t,
        source.xy,
        vel=source.vel,
        acc=source.acc,
        face_width_px=None,
    )

    sample = _one(frame)

    assert sample.distance_fw is None
    assert sample.closing_speed_fwps is None
    assert sample.closing_accel_fwps2 is None
    assert sample.time_to_contact_s is None
    assert sample.approach_angle_deg == pytest.approx(0.0, abs=1e-6)
    assert sample.nearest_region is FaceRegion.MOUTH


def test_features_are_scale_invariant() -> None:
    def samples_at(scale: float):
        fw = fx.FACE_WIDTH_PX * scale
        frames = fx.distance_profile(
            [(0.0, 1.0), (0.5, 0.2)],
            side=Side.RIGHT,
            shape=HandShape.FIST,
            anchor=fx.face_anchor("face_chin", fw=fw),
            fw=fw,
            unit=fx.HAND_UNIT_PX * scale,
        )
        return [_one(frame) for frame in frames]

    half = samples_at(0.5)
    double = samples_at(2.0)

    for small, large in zip(half, double, strict=True):
        assert small.distance_fw == pytest.approx(large.distance_fw, rel=1e-9)
        assert small.closing_speed_fwps == pytest.approx(large.closing_speed_fwps, rel=1e-9)
        assert small.closing_accel_fwps2 == pytest.approx(
            large.closing_accel_fwps2, rel=1e-9, abs=1e-9
        )
        assert small.approach_angle_deg == pytest.approx(large.approach_angle_deg, rel=1e-9)
        assert small.occlusion_fraction == pytest.approx(large.occlusion_fraction)


def test_samples_are_ordered_left_then_right() -> None:
    frame = fx.make_frame(0, 0.0, fx.neutral_xy())

    samples = _samples(frame)

    assert [sample.side for sample in samples] == [Side.LEFT, Side.RIGHT]
    assert samples[0].distance_fw == pytest.approx(samples[1].distance_fw)


def test_layout_without_hand_keypoints_falls_back_to_the_wrist() -> None:
    full = fx.neutral_xy()
    body = full[: len(fx.BODY_NAMES)]
    frame = fx.make_frame(0, 0.0, body)

    samples = hand_face_samples(frame, fx.WRIST_ONLY_LAYOUT, CFG)

    assert [sample.side for sample in samples] == [Side.LEFT, Side.RIGHT]
    assert all(sample.distance_fw is not None for sample in samples)
    assert all(sample.distance_fw > 2.0 for sample in samples if sample.distance_fw is not None)


def test_lost_hands_give_no_samples() -> None:
    status = np.full(fx.KEYPOINT_COUNT, int(KeypointStatus.LOST), dtype=np.int8)
    frame = fx.make_frame(0, 0.0, fx.neutral_xy(), status=status)

    assert _samples(frame) == ()
