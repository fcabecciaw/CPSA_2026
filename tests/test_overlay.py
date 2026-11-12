"""The overlay: the box drawing and the MoveNet skeleton (R2-G)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa import overlay
from cpsa.overlay import (
    COCO_SKELETON,
    MAX_DRAWN_OBJECTS,
    SKELETON_MIN_SCORE,
    class_color,
    draw_overlay,
    draw_skeleton,
)
from cpsa.protocol import NUM_KEYPOINTS, ShowcaseObject

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


# -- the boxes (R2-G) -------------------------------------------------------- #


def _object(
    label: str = "persona",
    class_id: int = 0,
    score: float = 0.87,
    box: tuple[float, float, float, float] = (10.2, 12.6, 60.0, 50.0),
) -> ShowcaseObject:
    return ShowcaseObject(label=label, class_id=class_id, score=score, box=box)


def test_the_overlay_draws_the_box_edge_in_the_class_colour() -> None:
    frame = _frame(value=200)

    picture = draw_overlay(frame, [_object()])

    assert picture is not frame
    assert np.array_equal(frame, _frame(value=200))  # the analysed frame is untouched
    # ``(10.2, 12.6)`` rounds to the box's top-left corner, which the 2 px
    # rectangle of the class colour covers; the inside of the box is the frame.
    assert tuple(int(channel) for channel in picture[13, 10]) == class_color(0)
    assert tuple(int(channel) for channel in picture[13, 60]) == class_color(0)
    # Below the tag on the top edge, inside the box, the frame still shows.
    assert tuple(int(channel) for channel in picture[45, 30]) == (200, 200, 200)


def test_a_box_outside_the_frame_is_clipped() -> None:
    frame = _frame()

    picture = draw_overlay(frame, [_object(box=(-100.0, -80.0, 40.0, 40.0))])

    assert picture.shape == frame.shape
    assert tuple(int(channel) for channel in picture[0, 0]) == class_color(0)
    assert tuple(int(channel) for channel in picture[40, 40]) != (200, 200, 200)


def test_at_most_twelve_boxes_are_drawn_persons_first(monkeypatch: pytest.MonkeyPatch) -> None:
    drawn: list[tuple[str, int]] = []
    sprite = overlay._label_sprite

    def recording(label: str, percent: int, colour: tuple[int, int, int]) -> np.ndarray:
        drawn.append((label, percent))
        return sprite(label, percent, colour)

    monkeypatch.setattr(overlay, "_label_sprite", recording)
    objects = [
        _object(label="persona", class_id=0, score=0.50 + 0.05 * index, box=(5, 5, 30, 30))
        for index in range(8)
    ] + [
        _object(label="bottiglia", class_id=39, score=0.90 + 0.01 * index, box=(5, 5, 30, 30))
        for index in range(8)
    ]

    draw_overlay(_frame(), objects)

    # The eight persons beat the bottles whatever their score, then the four
    # best bottles fill the frame: twelve tags, the best score first in each group.
    assert drawn == [("persona", 85 - 5 * index) for index in range(8)] + [
        ("bottiglia", 97),
        ("bottiglia", 96),
        ("bottiglia", 95),
        ("bottiglia", 94),
    ]
    assert len(drawn) == MAX_DRAWN_OBJECTS


def test_a_label_is_rasterized_once_and_reused_across_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(overlay, "_LABEL_SPRITES", {})
    objects = [_object(label="tazza", class_id=41, score=0.62), _object(label="libro", score=0.44)]

    draw_overlay(_frame(), objects)
    cached = dict(overlay._LABEL_SPRITES)
    draw_overlay(_frame(), objects)

    assert set(cached) == {("tazza", 62, class_color(41)), ("libro", 44, class_color(0))}
    assert cached == overlay._LABEL_SPRITES  # the second frame rendered nothing


def test_a_label_sprite_is_keyed_by_label_score_and_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(overlay, "_LABEL_SPRITES", {})

    draw_overlay(
        _frame(),
        [
            _object(label="tazza", class_id=41, score=0.62),
            _object(label="tazza", class_id=41, score=0.63),
            _object(label="tazza", class_id=39, score=0.62),
        ],
    )

    assert len(overlay._LABEL_SPRITES) == 3
