"""Hand-to-face contact detection (FR-014, AC-007)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa.domain.model import FaceRegion, HandShape, Side
from cpsa.vision.assessment import AnalysisConfig
from cpsa.vision.contacts2d import ContactConfig2D, detect_contacts
from cpsa.vision.types import KeypointStatus
from tests.unit import vision_fixtures as fx

CFG = AnalysisConfig()
CHIN = fx.face_anchor("face_chin")
LEFT_EYE = fx.face_anchor("left_eye")


def _contact_profile(shape: HandShape, anchor=CHIN, side: Side = Side.RIGHT):
    return fx.distance_profile(
        [(10.0, 1.0), (10.3, 0.05), (10.4, 0.05), (10.45, 1.2), (10.6, 1.2)],
        side=side,
        shape=shape,
        anchor=anchor,
    )


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (HandShape.FIST, HandShape.FIST),
        (HandShape.POINT_PINCH, HandShape.POINT_PINCH),
        (HandShape.OPEN_PALM, HandShape.OPEN_PALM),
    ],
)
def test_contact_reports_side_region_shape_and_kinematics(
    shape: HandShape, expected: HandShape
) -> None:
    contacts = detect_contacts(_contact_profile(shape), fx.LAYOUT, CFG)

    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.side is Side.RIGHT
    assert contact.region is FaceRegion.MOUTH
    assert contact.t_start == pytest.approx(10.3, abs=fx.DT / 2)
    assert contact.duration_s == pytest.approx(0.1, abs=fx.DT / 2)
    assert contact.min_distance_fw == pytest.approx(0.05, abs=1e-6)
    assert contact.peak_closing_speed_fwps == pytest.approx(3.17, rel=0.05)
    assert contact.decel_fwps2 is not None
    assert contact.decel_fwps2 > 4.0
    assert contact.hand_shape is expected
    assert contact.head_recoil_fw == pytest.approx(0.0, abs=1e-9)


def test_contact_on_the_eye_region() -> None:
    contacts = detect_contacts(_contact_profile(HandShape.FIST, LEFT_EYE), fx.LAYOUT, CFG)

    assert len(contacts) == 1
    assert contacts[0].region is FaceRegion.EYES


def test_contact_on_the_left_hand() -> None:
    frames = fx.distance_profile(
        [(10.0, 1.0), (10.3, 0.05), (10.4, 0.05), (10.45, 1.2)],
        side=Side.LEFT,
        shape=HandShape.FIST,
        anchor=CHIN,
    )

    contacts = detect_contacts(frames, fx.LAYOUT, CFG)

    assert [contact.side for contact in contacts] == [Side.LEFT]


def test_distance_hysteresis_needs_a_crossing_into_the_enter_band() -> None:
    outside = fx.distance_profile(
        [(0.0, 1.0), (0.2, 0.2), (0.5, 0.18), (0.8, 0.2), (1.0, 1.0)],
        side=Side.RIGHT,
        shape=HandShape.FIST,
        anchor=CHIN,
    )
    inside = fx.distance_profile(
        [(0.0, 1.0), (0.3, 0.1), (0.5, 0.2), (0.7, 0.1), (1.0, 1.0)],
        side=Side.RIGHT,
        shape=HandShape.FIST,
        anchor=CHIN,
    )

    assert detect_contacts(outside, fx.LAYOUT, CFG) == []
    # 0.2 is above contact_enter_fw but below contact_exit_fw: one contact.
    assert len(detect_contacts(inside, fx.LAYOUT, CFG)) == 1


def test_too_short_touch_is_discarded() -> None:
    frames = fx.distance_profile(
        [(0.0, 1.0), (0.4667, 1.0), (0.5, 0.05), (0.5333, 1.0), (1.0, 1.0)],
        side=Side.RIGHT,
        shape=HandShape.FIST,
        anchor=CHIN,
    )

    assert detect_contacts(frames, fx.LAYOUT, CFG) == []


def test_occlusion_contact_counts_when_the_hand_is_near() -> None:
    xy = fx.neutral_xy()
    fx.set_hand_distance(xy, Side.RIGHT, CHIN, 0.4, HandShape.FIST)
    status = np.full(fx.KEYPOINT_COUNT, int(KeypointStatus.MEASURED), dtype=np.int8)
    kept = {"nose", "face_chin", "face_jaw_right", "face_jaw_left"}
    for name in (*fx.BODY_NAMES[:5], *fx.FACE_NAMES):
        if name not in kept:
            status[fx.LAYOUT.index(name)] = int(KeypointStatus.LOST)
    times = 10.0 + np.arange(10, dtype=np.float64) * fx.DT

    frames = [
        fx.make_frame(index, float(t), xy.copy(), status=status) for index, t in enumerate(times)
    ]
    contacts = detect_contacts(frames, fx.LAYOUT, CFG)

    assert len(contacts) == 1
    assert contacts[0].min_distance_fw == pytest.approx(0.4, abs=1e-6)
    assert contacts[0].region is FaceRegion.MOUTH


def test_head_recoil_after_onset() -> None:
    times = 10.0 + np.arange(18, dtype=np.float64) * fx.DT
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        shift = min(1.0, max(0.0, (float(t) - 10.2) / 0.1)) * 0.15
        fx.shift_face(xy, (shift, 0.0))
        fx.set_hand_distance(
            xy,
            Side.RIGHT,
            fx.face_anchor("face_chin") + np.array([shift * fx.FACE_WIDTH_PX, 0.0]),
            0.1,
            HandShape.FIST,
        )
        positions.append(xy)

    contacts = detect_contacts(fx.frames_from_positions(times, positions), fx.LAYOUT, CFG)

    assert len(contacts) == 1
    assert contacts[0].head_recoil_fw == pytest.approx(0.15, abs=0.01)
    assert contacts[0].head_recoil_speed_fwps is not None
    assert contacts[0].head_recoil_speed_fwps > 1.0


def test_repeated_contacts_are_separate() -> None:
    frames = fx.distance_profile(
        [
            (0.0, 1.0),
            (0.2, 0.05),
            (0.3, 0.05),
            (0.4, 1.0),
            (1.2, 1.0),
            (1.4, 0.05),
            (1.5, 0.05),
            (1.6, 1.0),
            (2.4, 1.0),
            (2.6, 0.05),
            (2.7, 0.05),
            (2.8, 1.0),
        ],
        side=Side.RIGHT,
        shape=HandShape.FIST,
        anchor=CHIN,
    )

    contacts = detect_contacts(frames, fx.LAYOUT, CFG)

    assert len(contacts) == 3
    assert [contact.t_start for contact in contacts] == pytest.approx([0.2, 1.4, 2.6], abs=0.05)


def test_unknown_face_width_gives_no_contacts() -> None:
    xy = fx.neutral_xy()
    fx.set_hand_distance(xy, Side.RIGHT, CHIN, 0.05, HandShape.FIST)
    times = 0.0 + np.arange(10, dtype=np.float64) * fx.DT
    frames = [
        fx.make_frame(index, float(t), xy.copy(), face_width_px=None)
        for index, t in enumerate(times)
    ]

    assert detect_contacts(frames, fx.LAYOUT, CFG) == []


def test_empty_input_returns_no_contacts() -> None:
    assert detect_contacts([], fx.LAYOUT, CFG) == []


def test_invalid_configuration_raises() -> None:
    bad = AnalysisConfig(contacts=ContactConfig2D(contact_enter_fw=0.3, contact_exit_fw=0.2))
    with pytest.raises(ValueError, match="contact_exit_fw"):
        detect_contacts([], fx.LAYOUT, bad)
