"""The overlay: the box drawing and the MoveNet skeleton (R2-G)."""

from __future__ import annotations

import numpy as np

from cpsa.overlay import COCO_SKELETON, SKELETON_MIN_SCORE, draw_skeleton
from cpsa.protocol import NUM_KEYPOINTS


def _frame(value: int = 20, *, width: int = 64, height: int = 48) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _keypoints(score: float = 0.9) -> list[list[float]]:
    return [[4.0 + 3.0 * (index % 10), 4.0 + 2.0 * index, score] for index in range(17)]


def test_the_skeleton_bones_cover_the_coco_17_keypoints() -> None:
    assert len(COCO_SKELETON) == 16
    assert {joint for bone in COCO_SKELETON for joint in bone} == set(range(NUM_KEYPOINTS))


def test_the_skeleton_is_drawn_on_a_copy_of_the_frame() -> None:
    frame = _frame()

    overlay = draw_skeleton(frame, _keypoints())

    assert overlay is not frame
    assert np.array_equal(frame, _frame())  # the analysed frame is untouched
    assert not np.array_equal(overlay, frame)


def test_a_low_score_keypoint_is_not_drawn() -> None:
    frame = _frame()
    low = _keypoints(score=SKELETON_MIN_SCORE - 0.01)

    overlay = draw_skeleton(frame, low)

    assert np.array_equal(overlay, frame)


def test_an_out_of_frame_joint_is_clipped() -> None:
    frame = _frame()
    keypoints = _keypoints()
    keypoints[0] = [-100.0, -100.0, 0.9]
    keypoints[1] = [10_000.0, 10_000.0, 0.9]

    overlay = draw_skeleton(frame, keypoints)

    assert overlay.shape == frame.shape
    assert not np.array_equal(overlay, frame)
