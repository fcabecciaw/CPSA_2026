"""Drawing the detections on the analysed frame.

The overlay is drawn on the board, on the same frame the detector saw, so the
video and the boxes can never drift apart however fast the browser is. Each box
is a 2 px rectangle in the class colour plus a ``"<label> <score>%"`` tag; the
colours come from a fixed palette indexed by class, so a class keeps its colour
across frames and runs.

A busy frame draws at most :data:`MAX_DRAWN_OBJECTS` boxes -- the persons first,
then the best scores -- because a 640x480 video cannot show fifty tags anyway;
every detection still reaches ``showcase.state``. Each label is rasterized once
and cached as an RGB sprite, so a frame pays one blit per box instead of a
``getTextSize``/``putText`` pair, and the boxes of one colour go out in a single
``polylines`` call. Boxes, labels and the frame copy together cost 0.08 ms on
the host and are budgeted 1.5 ms on the board (R2-G): the label rasterizer alone
was 5 ms there.

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

import heapq
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from cpsa.protocol import NUM_KEYPOINTS, ShowcaseObject

__all__ = [
    "COCO_SKELETON",
    "MAX_DRAWN_OBJECTS",
    "PERSON_CLASS",
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

MAX_DRAWN_OBJECTS = 12
"""Boxes drawn on one frame: persons first, then the best scores.

The video is 640x480 and a crowd would only bury itself; the rest of the
detections still reach ``showcase.state`` and the dashboard.
"""

PERSON_CLASS = 0
"""COCO-80 ``person``: the class the drawing order keeps when the frame is full."""

_LABEL_SPRITE_LIMIT = 512
"""Label sprites kept before the cache is dropped (a few dozen labels x scores)."""

_LABEL_SPRITES: dict[tuple[str, int, tuple[int, int, int]], NDArray[np.uint8]] = {}
"""``(label, score %, colour) -> RGB sprite``: the label rasterized once (R2-G)."""

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

    At most :data:`MAX_DRAWN_OBJECTS` boxes are drawn -- the persons first, then
    by score -- and each label is a sprite rendered once per label and score and
    then blitted: the board's overlay was 5 ms of ``putText``/``getTextSize``
    rasterizer setup, not of pixels. Every detection still reaches
    ``showcase.state``: this caps the drawing, not the measurement.
    """
    overlay = np.array(rgb, copy=True)
    if not objects:
        return overlay
    height, width = overlay.shape[:2]
    drawn = _drawn(objects)
    corners = _clipped_corners(drawn, width, height)
    colours = [class_color(item.class_id) for item in drawn]
    _draw_boxes(overlay, corners, colours)
    for item, position, colour in zip(drawn, corners, colours, strict=True):
        sprite = _label_sprite(item.label, int(round(item.score * 100.0)), colour)
        _blit_label(overlay, sprite, int(position[0]), int(position[1]))
    return overlay


def _drawn(objects: Sequence[ShowcaseObject]) -> list[ShowcaseObject]:
    """The objects to draw: persons first, then by score, at most :data:`MAX_DRAWN_OBJECTS`."""
    if len(objects) <= MAX_DRAWN_OBJECTS:
        return list(objects)
    return heapq.nsmallest(MAX_DRAWN_OBJECTS, objects, key=_draw_priority)


def _draw_priority(item: ShowcaseObject) -> tuple[int, float]:
    """Sort key of the drawing order: a person before any other class, then the score."""
    return (0 if item.class_id == PERSON_CLASS else 1, -float(item.score))


def _clipped_corners(
    items: Sequence[ShowcaseObject],
    width: int,
    height: int,
) -> NDArray[np.int32]:
    """Clip every box to the image in one numpy pass: ``(x1, y1, x2, y2)`` rows."""
    boxes = np.asarray([item.box for item in items], dtype=np.float64).reshape(-1, 4)
    finite = np.isfinite(boxes).all(axis=1)
    boxes[~finite] = 0.0
    np.rint(boxes, out=boxes)
    corners = np.empty((boxes.shape[0], 4), dtype=np.int32)
    np.clip(boxes[:, 0], 0, max(width - 1, 0), out=boxes[:, 0])
    np.clip(boxes[:, 1], 0, max(height - 1, 0), out=boxes[:, 1])
    np.clip(boxes[:, 2], 0, max(width - 1, 0), out=boxes[:, 2])
    np.clip(boxes[:, 3], 0, max(height - 1, 0), out=boxes[:, 3])
    corners[:, 0] = boxes[:, 0]
    corners[:, 1] = boxes[:, 1]
    corners[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0])
    corners[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1])
    return corners


def _draw_boxes(
    overlay: NDArray[np.uint8],
    corners: NDArray[np.int32],
    colours: Sequence[tuple[int, int, int]],
) -> None:
    """Draw every box of a colour in one ``polylines`` call (one per colour)."""
    if corners.size == 0:
        return
    quads = np.empty((corners.shape[0], 4, 2), dtype=np.int32)
    quads[:, 0, 0] = corners[:, 0]
    quads[:, 0, 1] = corners[:, 1]
    quads[:, 1, 0] = corners[:, 2]
    quads[:, 1, 1] = corners[:, 1]
    quads[:, 2, 0] = corners[:, 2]
    quads[:, 2, 1] = corners[:, 3]
    quads[:, 3, 0] = corners[:, 0]
    quads[:, 3, 1] = corners[:, 3]
    groups: dict[tuple[int, int, int], list[NDArray[np.int32]]] = {}
    for quad, colour in zip(quads, colours, strict=True):
        groups.setdefault(colour, []).append(quad)
    for colour, group in groups.items():
        cv2.polylines(overlay, group, True, colour, _BOX_THICKNESS)


def _label_sprite(
    label: str,
    percent: int,
    colour: tuple[int, int, int],
) -> NDArray[np.uint8]:
    """The ``"<label> <score>%"`` tag as an RGB sprite, rendered once and cached.

    The sprite is exactly the filled rectangle plus the text the overlay used to
    draw into the frame, so blitting it is pixel-for-pixel the old label. The
    cache is keyed by label, rounded score and colour: a frame usually hits it
    for every box and only pays a blit.
    """
    key = (label, percent, colour)
    sprite = _LABEL_SPRITES.get(key)
    if sprite is not None:
        return sprite
    text = f"{label} {percent}%"
    (text_width, text_height), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    sprite = np.empty(
        (text_height + 2 * _LABEL_GAP, text_width + 2 * _LABEL_PADDING, 3), dtype=np.uint8
    )
    sprite[:, :] = colour
    cv2.putText(
        sprite,
        text,
        (_LABEL_PADDING, text_height + _LABEL_GAP),
        _FONT,
        _FONT_SCALE,
        _text_color(colour),
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )
    if len(_LABEL_SPRITES) >= _LABEL_SPRITE_LIMIT:
        _LABEL_SPRITES.clear()
    _LABEL_SPRITES[key] = sprite
    return sprite


def _blit_label(
    overlay: NDArray[np.uint8],
    sprite: NDArray[np.uint8],
    left: int,
    bottom: int,
) -> None:
    """Copy a label sprite above (or, when there is no room, inside) a box."""
    height, width = overlay.shape[:2]
    sprite_height, sprite_width = sprite.shape[:2]
    top = bottom - sprite_height
    if top < 0:
        # No room above the box: put the tag just inside its top edge.
        top = bottom + _LABEL_GAP
    top = max(0, min(top, max(height - sprite_height, 0)))
    visible_height = min(sprite_height, height - top)
    visible_width = min(sprite_width, width - left)
    if visible_height <= 0 or visible_width <= 0:
        return
    overlay[top : top + visible_height, left : left + visible_width] = sprite[
        :visible_height, :visible_width
    ]


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


def _text_color(colour: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black on a light box colour, white on a dark one."""
    red, green, blue = colour
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 140.0 else (255, 255, 255)
