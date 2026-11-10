"""Hand-shape classification fixtures (FR-014, AC-007)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa.domain.model import HandShape
from cpsa.vision.handshape import HandShapeConfig, classify_hand_shape
from tests.unit import vision_fixtures as fx

CFG = HandShapeConfig()
CONF = np.full(21, 0.9, dtype=np.float64)


@pytest.mark.parametrize(
    ("shape", "pinch", "expected"),
    [
        (HandShape.FIST, False, HandShape.FIST),
        (HandShape.POINT_PINCH, False, HandShape.POINT_PINCH),
        (HandShape.POINT_PINCH, True, HandShape.POINT_PINCH),
        (HandShape.OPEN_PALM, False, HandShape.OPEN_PALM),
    ],
)
def test_fist_point_pinch_and_open_palm(shape: HandShape, pinch: bool, expected: HandShape) -> None:
    points = fx.hand_points((500.0, 300.0), shape, pinch=pinch)

    assert classify_hand_shape(points, CONF, CFG) is expected


def test_mixed_extension_is_unknown() -> None:
    points = fx.hand_points((500.0, 300.0), HandShape.UNKNOWN)

    assert classify_hand_shape(points, CONF, CFG) is HandShape.UNKNOWN


def test_low_confidence_is_unknown() -> None:
    points = fx.hand_points((500.0, 300.0), HandShape.FIST)
    confidence = np.full(21, 0.1, dtype=np.float64)

    assert classify_hand_shape(points, confidence, CFG) is HandShape.UNKNOWN


def test_too_few_confident_keypoints_is_unknown() -> None:
    points = fx.hand_points((500.0, 300.0), HandShape.FIST)
    confidence = np.full(21, 0.9, dtype=np.float64)
    confidence[[1, 2, 3, 4, 18, 19, 20]] = 0.05

    assert classify_hand_shape(points, confidence, CFG) is HandShape.UNKNOWN


def test_degenerate_hand_is_unknown() -> None:
    points = np.zeros((21, 2), dtype=np.float64)

    assert classify_hand_shape(points, CONF, CFG) is HandShape.UNKNOWN


def test_malformed_input_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        classify_hand_shape(np.zeros((20, 2)), np.full(20, 0.9), CFG)
    with pytest.raises(ValueError, match="shape"):
        classify_hand_shape(np.zeros((21, 2)), np.full(20, 0.9), CFG)
