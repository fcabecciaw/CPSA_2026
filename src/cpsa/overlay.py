"""Drawing the detections on the analysed frame.

The overlay is drawn on the board, on the same frame the detector saw, so the
video and the boxes can never drift apart however fast the browser is. Each box
is a 2 px rectangle in the class colour plus a ``"<label> <score>%"`` tag; the
colours come from a fixed palette indexed by class, so a class keeps its colour
across frames and runs.

In the MoveNet view the frame is dimmed to 35 % and the COCO-17 skeleton is
drawn on top in a neon palette, one colour per body part (magenta head, cyan
arms, lime torso, an electric purple left leg and an orange right one): every
bone is a thick darker halo under a thin bright core, every reliable joint a
coloured dot with a white centre.
Keypoints below :data:`SKELETON_MIN_SCORE` are dropped, as F-E4 requires.

The whole overlay is a handful of OpenCV calls -- the dim is one
``convertScaleAbs`` into the output frame and each part is one ``polylines`` and
one ``fillPoly`` pass -- instead of two calls per bone plus one per joint: the
per-call rasterizer setup, not the pixels, was the board's 2.8 ms. R2-G budgets
1.5 ms for dim plus draw.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from cpsa.protocol import NUM_KEYPOINTS, ShowcaseObject

__all__ = [
    "COCO_SKELETON",
    "SKELETON_MIN_SCORE",
    "class_color",
    "draw_overlay",
    "draw_skeleton",
]

_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1
_LABEL_PADDING = 3
_LABEL_GAP = 4

SKELETON_MIN_SCORE = 0.3
"""A keypoint below this score is not drawn (F-E4 #25)."""

_GLOW_THICKNESS = 10
_BONE_THICKNESS = 3
_JOINT_RADIUS = 3
_CORE_RADIUS = 1

_DIM_FACTOR = 0.35
"""Brightness of the video under the skeleton: the pose has to win the frame."""

_GLOW_FACTOR = 0.4
"""A bone's halo is its neon colour at 40 %."""

_JOINT_CORE_COLOR = (255, 255, 255)
"""White centre of a joint dot."""

#: ``(keypoints, bones, neon colour)``, in skeleton drawing order. The keypoint
#: lists decide each dot's colour: a joint shared by two parts keeps the colour
#: of the part it appears in first.
_Part = tuple[tuple[int, ...], tuple[tuple[int, int], ...], tuple[int, int, int]]
_PARTS: tuple[_Part, ...] = (
    ((0, 1, 2, 3, 4), ((0, 1), (0, 2), (1, 3), (2, 4)), (255, 20, 255)),  # head
    (
        (5, 6, 7, 8, 9, 10),
        ((5, 7), (7, 9), (6, 8), (8, 10)),
        (0, 245, 255),
    ),  # arms
    ((5, 6, 11, 12), ((5, 6), (5, 11), (6, 12), (11, 12)), (120, 255, 45)),  # torso
    ((11, 13, 15), ((11, 13), (13, 15)), (185, 40, 255)),  # left leg, purple
    ((12, 14, 16), ((12, 14), (14, 16)), (255, 145, 0)),  # right leg, orange
)

COCO_SKELETON: tuple[tuple[int, int], ...] = tuple(bone for _, bones, _ in _PARTS for bone in bones)
"""COCO-17 bone pairs, in :data:`cpsa.protocol.KEYPOINTS` order."""

_DOT_OFFSETS = np.array(
    (
        (_JOINT_RADIUS, 0),
        (2, 2),
        (0, _JOINT_RADIUS),
        (-2, 2),
        (-_JOINT_RADIUS, 0),
        (-2, -2),
        (0, -_JOINT_RADIUS),
        (2, -2),
    ),
    dtype=np.int32,
)
"""Octagon around a joint cell: a filled circle in one ``fillPoly`` call."""

_CORE_OFFSETS = np.array(
    (
        (_CORE_RADIUS, 0),
        (_CORE_RADIUS, _CORE_RADIUS),
        (0, _CORE_RADIUS),
        (-_CORE_RADIUS, _CORE_RADIUS),
        (-_CORE_RADIUS, 0),
        (-_CORE_RADIUS, -_CORE_RADIUS),
        (0, -_CORE_RADIUS),
        (_CORE_RADIUS, -_CORE_RADIUS),
    ),
    dtype=np.int32,
)
"""The white centre of a dot, in the same shape as :data:`_DOT_OFFSETS`."""

_PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (245, 130, 48),
    (0, 130, 200),
    (240, 50, 230),
    (0, 128, 128),
    (170, 110, 40),
    (145, 30, 180),
    (70, 240, 240),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 0),
)
"""Box colours in RGB, one per class modulo the table length."""


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return the RGB colour of a class: stable across frames and runs."""
    return _PALETTE[abs(int(class_id)) % len(_PALETTE)]


def draw_overlay(
    rgb: NDArray[np.uint8],
    objects: Sequence[ShowcaseObject],
) -> NDArray[np.uint8]:
    """Return a copy of ``rgb`` with a 2 px box and a ``"<label> <score>%"`` tag.

    The image is RGB (the order the JPEG encoder and the shared frame channel
    expect) and the copy is what the video shows, so the frame being analyzed
    can never change under it.
    """
    overlay = np.array(rgb, copy=True)
    height, width = overlay.shape[:2]
    for item in objects:
        colour = class_color(item.class_id)
        x1, y1, x2, y2 = _clip(item.box, width, height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, _BOX_THICKNESS)
        text = f"{item.label} {int(round(item.score * 100.0))}%"
        (text_width, text_height), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        label_height = text_height + 2 * _LABEL_GAP
        top = y1 - label_height
        if top < 0:
            # No room above the box: put the tag just inside its top edge.
            top = y1 + _LABEL_GAP
        top = max(0, min(top, height - label_height))
        bottom = min(height - 1, top + label_height)
        right = min(width - 1, x1 + text_width + 2 * _LABEL_PADDING)
        cv2.rectangle(overlay, (x1, top), (right, bottom), colour, -1)
        cv2.putText(
            overlay,
            text,
            (x1 + _LABEL_PADDING, bottom - _LABEL_GAP),
            _FONT,
            _FONT_SCALE,
            _text_color(colour),
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return overlay


def draw_skeleton(
    rgb: NDArray[np.uint8],
    keypoints: Sequence[Sequence[float]],
) -> NDArray[np.uint8]:
    """Return a copy of ``rgb`` with the glowing COCO-17 skeleton drawn on it.

    Args:
        rgb: ``(H, W, 3)`` RGB uint8 frame.
        keypoints: 17 ``(x, y, score)`` triples in source pixels,
            :data:`cpsa.protocol.KEYPOINTS` order.

    Returns:
        The frame the video shows: the camera frame dimmed to
        :data:`_DIM_FACTOR`, plus the neon skeleton and a dot on every joint
        whose score is at least :data:`SKELETON_MIN_SCORE`.
    """

    overlay = np.empty_like(rgb)
    cv2.convertScaleAbs(rgb, dst=overlay, alpha=_DIM_FACTOR)
    height, width = overlay.shape[:2]
    points, drawable = _skeleton_points(keypoints, width, height)
    if not drawable.any():
        return overlay

    for _, bones, colour in _PARTS:
        segments: list[NDArray[np.int32]] = [
            np.stack((points[first], points[second]))
            for first, second in bones
            if first < len(drawable)
            and second < len(drawable)
            and drawable[first]
            and drawable[second]
        ]
        if segments:
            halo = tuple(int(channel * _GLOW_FACTOR) for channel in colour)
            # The halo is an aliased pass: at 10 px its edge is invisible, while
            # the thin bright core keeps LINE_AA. On the A53 the AA rasterizer
            # was most of the board's old 2.8 ms, not the drawing itself.
            cv2.polylines(overlay, segments, False, halo, _GLOW_THICKNESS, cv2.LINE_8)
            cv2.polylines(overlay, segments, False, colour, _BONE_THICKNESS, cv2.LINE_AA)

    for joints, _, colour in _PARTS:
        visible = [index for index in joints if index < len(drawable) and drawable[index]]
        if not visible:
            continue
        dots: NDArray[np.int32] = points[visible]
        circles: NDArray[np.int32] = dots[:, None, :] + _DOT_OFFSETS[None, :, :]
        cv2.fillPoly(overlay, list(circles), colour, cv2.LINE_8)
    all_dots: NDArray[np.int32] = points[drawable]
    core: NDArray[np.int32] = all_dots[:, None, :] + _CORE_OFFSETS[None, :, :]
    cv2.fillPoly(overlay, list(core), _JOINT_CORE_COLOR, cv2.LINE_8)
    return overlay


def _skeleton_points(
    keypoints: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    """Clip the keypoints to the image and say which ones may be drawn."""

    array = np.asarray(keypoints, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3 or not len(array):
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=bool)
    count = min(len(array), NUM_KEYPOINTS)
    coordinates = np.rint(array[:count, :2])
    drawable = (array[:count, 2] >= SKELETON_MIN_SCORE) & np.isfinite(array[:count]).all(axis=1)
    np.clip(coordinates, 0.0, (max(width - 1, 0), max(height - 1, 0)), out=coordinates)
    coordinates[~drawable] = 0.0
    return coordinates.astype(np.int32), drawable


def _clip(
    box: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Clip a float box to the image and return integer corners inside it."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
        return (0, 0, 0, 0)
    left = min(max(int(round(x1)), 0), max(width - 1, 0))
    top = min(max(int(round(y1)), 0), max(height - 1, 0))
    right = min(max(int(round(x2)), left), max(width - 1, 0))
    bottom = min(max(int(round(y2)), top), max(height - 1, 0))
    return (left, top, right, bottom)


def _text_color(colour: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black on a light box colour, white on a dark one."""
    red, green, blue = colour
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 140.0 else (255, 255, 255)
