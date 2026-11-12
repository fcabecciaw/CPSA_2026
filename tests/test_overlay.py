"""The overlay: the box drawing and the MoveNet skeleton (R2-G)."""

from __future__ import annotations

import numpy as np

from cpsa.overlay import COCO_SKELETON, SKELETON_MIN_SCORE, draw_skeleton
from cpsa.protocol import NUM_KEYPOINTS

DIM_FACTOR = 0.35
"""The MoveNet view dims the camera frame to this brightness."""


def _frame(value: int = 200, *, width: int = 64, height: int = 48) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _keypoints(score: float = 0.9) -> list[list[float]]:
    return [[4.0 + 3.0 * (index % 10), 4.0 + 2.0 * index, score] for index in range(17)]


def _dimmed(frame: np.ndarray) -> np.ndarray:
    return np.rint(frame * DIM_FACTOR)


def test_the_skeleton_bones_cover_the_coco_17_keypoints() -> None:
    assert len(COCO_SKELETON) == 16
    assert {joint for bone in COCO_SKELETON for joint in bone} == set(range(NUM_KEYPOINTS))


def test_the_skeleton_is_drawn_on_a_dimmed_copy_of_the_frame() -> None:
    frame = _frame()

    overlay = draw_skeleton(frame, _keypoints())

    assert overlay is not frame
    assert np.array_equal(frame, _frame())  # the analysed frame is untouched
    assert overlay.dtype == np.uint8
    # The video stays visible but recedes: the skeleton has to win the frame.
    background = overlay[:, 45:]  # right of every keypoint and bone
    assert background.max() <= _dimmed(frame).max() + 1
    assert background.mean() < frame.mean() * 0.5


def test_the_skeleton_uses_a_neon_palette_per_body_part() -> None:
    frame = _frame(value=20)

    overlay = draw_skeleton(frame, _keypoints())

    colours = {tuple(int(channel) for channel in pixel) for pixel in overlay.reshape(-1, 3)}
    bright = [colour for colour in colours if max(colour) >= 200]
    # Head, arms, torso and legs each bring their own neon colour.
    assert len(bright) >= 4
    assert any(red > 200 and blue > 200 for red, _, blue in bright)  # magenta head
    assert any(red < 60 and green > 200 and blue > 200 for red, green, blue in bright)
    assert any(green > 200 and blue < 100 for _, green, blue in bright)


def test_a_low_score_keypoint_leaves_only_the_dimmed_video() -> None:
    frame = _frame()
    low = _keypoints(score=SKELETON_MIN_SCORE - 0.01)

    overlay = draw_skeleton(frame, low)

    np.testing.assert_allclose(overlay, _dimmed(frame), atol=1)


def test_an_out_of_frame_joint_is_clipped() -> None:
    frame = _frame()
    keypoints = _keypoints()
    keypoints[0] = [-100.0, -100.0, 0.9]
    keypoints[1] = [10_000.0, 10_000.0, 0.9]

    overlay = draw_skeleton(frame, keypoints)

    assert overlay.shape == frame.shape
    assert not np.array_equal(overlay, _dimmed(frame))
