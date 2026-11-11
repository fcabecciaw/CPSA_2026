"""3D lift: depth median, invalid windows and border clipping (F-B)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa.vision.calibration import DepthModel, Intrinsics
from cpsa.vision.hand3d import lift_keypoints

INTRINSICS = Intrinsics(fx=8.0, fy=8.0, cx=4.0, cy=3.0, width=8, height=6)
DEPTH = DepthModel(scale_m=1e-3, offset_m=0.0)


def depth_map() -> np.ndarray:
    """A 6x8 map with nonzero samples only around the centre and one corner."""

    raw = np.zeros((6, 8), dtype=np.uint16)
    raw[2, 3] = 100
    raw[2, 4] = 200
    raw[2, 5] = 300
    raw[3, 4] = 400
    raw[0, 0] = 500
    raw[0, 1] = 500
    return raw


def test_valid_window_uses_the_median_of_nonzero_depth_and_projects_the_pixel() -> None:
    kpts = np.array([[6.0, 5.0, 0.9], [4.0, 3.0, 0.8]], dtype=np.float32)

    lifted = lift_keypoints(kpts, depth_map(), INTRINSICS, DEPTH)

    assert lifted.shape == (2, 4)
    assert lifted.dtype == np.float32
    # (6, 5): window rows 3..5, columns 4..7 holds 400 and a zero -> 0.4 m.
    np.testing.assert_allclose(lifted[0], [0.1, 0.1, 0.4, 1.0], atol=1e-6)
    # (4, 3): window rows 1..5, columns 2..6 holds 100/200/300/400 -> 0.25 m.
    np.testing.assert_allclose(lifted[1], [0.0, 0.0, 0.25, 1.0], atol=1e-6)


def test_window_without_depth_is_invalid_at_the_origin() -> None:
    kpts = np.array([[7.0, 5.0, 0.9]], dtype=np.float32)
    raw = depth_map()
    raw[3:, 5:] = 0

    lifted = lift_keypoints(kpts, raw, INTRINSICS, DEPTH)

    np.testing.assert_array_equal(lifted, np.zeros((1, 4), dtype=np.float32))


def test_an_all_zero_map_leaves_every_hand_at_the_origin() -> None:
    kpts = np.array([[0.0, 0.0, 1.0], [3.0, 2.0, 1.0]], dtype=np.float32)

    lifted = lift_keypoints(kpts, np.zeros((6, 8), dtype=np.uint16), INTRINSICS, DEPTH)

    assert lifted.shape == (2, 4)
    np.testing.assert_array_equal(lifted, np.zeros((2, 4), dtype=np.float32))


def test_border_keypoints_clip_the_window_instead_of_panicking() -> None:
    kpts = np.array([[0.0, 0.0, 1.0], [7.0, 5.0, 1.0]], dtype=np.float32)

    lifted = lift_keypoints(kpts, depth_map(), INTRINSICS, DEPTH)

    # (0, 0): the window is clipped to rows 0..2, columns 0..2 -> 0.5 m.
    np.testing.assert_allclose(lifted[0], [-0.25, -0.1875, 0.5, 1.0], atol=1e-6)
    # (7, 5): the window is clipped to rows 3..5, columns 5..7, all zero.
    np.testing.assert_array_equal(lifted[1], np.zeros(4, dtype=np.float32))


def test_window_size_controls_the_median_footprint() -> None:
    kpts = np.array([[4.0, 3.0, 1.0]], dtype=np.float32)

    narrow = lift_keypoints(kpts, depth_map(), INTRINSICS, DEPTH, window=1)
    wide = lift_keypoints(kpts, depth_map(), INTRINSICS, DEPTH, window=5)

    assert narrow[0, 2] == pytest.approx(0.4)
    assert wide[0, 2] == pytest.approx(0.25)


def test_no_keypoints_lift_to_an_empty_result() -> None:
    lifted = lift_keypoints(np.zeros((0, 3), dtype=np.float32), depth_map(), INTRINSICS, DEPTH)
    assert lifted.shape == (0, 4)


def test_non_finite_keypoints_are_invalid() -> None:
    kpts = np.array([[np.nan, 3.0, 1.0], [4.0, np.inf, 1.0]], dtype=np.float32)

    lifted = lift_keypoints(kpts, depth_map(), INTRINSICS, DEPTH)

    np.testing.assert_array_equal(lifted, np.zeros((2, 4), dtype=np.float32))


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="kpts2d must be"):
        lift_keypoints(np.zeros((2, 2), dtype=np.float32), depth_map(), INTRINSICS, DEPTH)
    with pytest.raises(ValueError, match="depth_raw must be"):
        lift_keypoints(np.zeros((1, 3), dtype=np.float32), np.zeros(6), INTRINSICS, DEPTH)
    with pytest.raises(ValueError, match="window must be"):
        lift_keypoints(np.zeros((1, 3), dtype=np.float32), depth_map(), INTRINSICS, DEPTH, window=4)
    with pytest.raises(ValueError, match="positive focal lengths"):
        lift_keypoints(
            np.zeros((1, 3), dtype=np.float32),
            depth_map(),
            Intrinsics(fx=0.0, fy=8.0, cx=4.0, cy=3.0, width=8, height=6),
            DEPTH,
        )
