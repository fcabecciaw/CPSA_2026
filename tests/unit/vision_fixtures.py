"""Synthetic tracked frames and layouts for the 2-D vision feature tests.

The Task C tests build :class:`~cpsa.vision.types.TrackedFrame` fixtures
directly, so they do not depend on the Task B tracker or the Task A layouts.
The layout mirrors the HALFBUST_62 keypoint names, but only the keypoints the
feature code reads carry meaningful geometry. All lengths are in pixels and all
fixtures are deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cpsa.domain.model import FaceRegion, HandShape, Side
from cpsa.vision.types import FramingState, KeypointLayout, KeypointStatus, TrackedFrame

DT = 1.0 / 30.0
FACE_WIDTH_PX = 80.0
FACE_CENTER = np.array([480.0, 300.0])
HAND_UNIT_PX = 25.0

BODY_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
FACE_NAMES = (
    "face_jaw_right",
    "face_jaw_left",
    "face_chin",
    "face_brow_right",
    "face_brow_left",
    "face_mouth_right",
    "face_mouth_left",
    "face_lip_upper",
    "face_lip_lower",
)
HAND_NAMES = tuple(
    [f"left_hand_{index}" for index in range(21)] + [f"right_hand_{index}" for index in range(21)]
)

_LEFT_HAND = tuple(range(len(BODY_NAMES) + len(FACE_NAMES), len(BODY_NAMES) + len(FACE_NAMES) + 21))
_RIGHT_HAND = tuple(start + 21 for start in _LEFT_HAND)
_FACE = (0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19)
_FACE_REGIONS = (
    (0, FaceRegion.NOSE_CHEEK),
    (1, FaceRegion.EYES),
    (2, FaceRegion.EYES),
    (3, FaceRegion.EAR),
    (4, FaceRegion.EAR),
    (11, FaceRegion.NOSE_CHEEK),
    (12, FaceRegion.NOSE_CHEEK),
    (13, FaceRegion.MOUTH),
    (14, FaceRegion.EYES),
    (15, FaceRegion.EYES),
    (16, FaceRegion.MOUTH),
    (17, FaceRegion.MOUTH),
    (18, FaceRegion.MOUTH),
    (19, FaceRegion.MOUTH),
)

LAYOUT = KeypointLayout(
    name="halfbust62_test",
    names=BODY_NAMES + FACE_NAMES + HAND_NAMES,
    face=_FACE,
    face_regions=_FACE_REGIONS,
    left_hand=_LEFT_HAND,
    right_hand=_RIGHT_HAND,
    edges=((5, 6), (5, 9), (6, 10), (9, 7), (10, 8)),
)

WRIST_ONLY_LAYOUT = KeypointLayout(
    name="wrist_only_test",
    names=BODY_NAMES,
    face=(0, 1, 2, 3, 4),
    face_regions=_FACE_REGIONS[:5],
    left_hand=(),
    right_hand=(),
    edges=((5, 6), (5, 9), (6, 10)),
)

KEYPOINT_COUNT = len(LAYOUT)

# Face-anchor offsets in face-width units, relative to the face centre.
_FACE_OFFSETS: Mapping[str, tuple[float, float]] = {
    "nose": (0.0, 0.0),
    "left_eye": (-0.28, -0.15),
    "right_eye": (0.28, -0.15),
    "left_ear": (-0.5, 0.0),
    "right_ear": (0.5, 0.0),
    "face_jaw_right": (-0.45, 0.25),
    "face_jaw_left": (0.45, 0.25),
    "face_chin": (0.0, 0.45),
    "face_brow_right": (-0.25, -0.3),
    "face_brow_left": (0.25, -0.3),
    "face_mouth_right": (-0.2, 0.25),
    "face_mouth_left": (0.2, 0.25),
    "face_lip_upper": (0.0, 0.2),
    "face_lip_lower": (0.0, 0.3),
}

_FINGERS: Mapping[int, tuple[float, float]] = {
    5: (-0.12, -0.38),  # index MCP
    9: (0.0, -0.40),  # middle MCP
    13: (0.12, -0.38),  # ring MCP
    17: (0.22, -0.33),  # little MCP
}
_EXTENDED_FINGERS: Mapping[HandShape, tuple[int, ...]] = {
    HandShape.FIST: (),
    HandShape.POINT_PINCH: (5,),
    HandShape.OPEN_PALM: (5, 9, 13, 17),
    HandShape.UNKNOWN: (5, 13),
}
_INDEX_TIP = np.array([-0.12, -0.98])
_THUMB_TIP = {
    HandShape.FIST: np.array([-0.30, -0.20]),
    HandShape.POINT_PINCH: np.array([-0.50, -0.38]),
    HandShape.OPEN_PALM: np.array([-0.50, -0.38]),
    HandShape.UNKNOWN: np.array([-0.50, -0.38]),
}


def face_anchor(
    name: str,
    face_center: ArrayLike = FACE_CENTER,
    fw: float = FACE_WIDTH_PX,
) -> NDArray[np.float64]:
    """Return one face anchor in pixels for a face of width ``fw``."""
    offset = np.asarray(_FACE_OFFSETS[name], dtype=np.float64)
    return np.asarray(face_center, dtype=np.float64) + offset * fw


def neutral_xy(
    face_center: ArrayLike = FACE_CENTER,
    fw: float = FACE_WIDTH_PX,
) -> NDArray[np.float64]:
    """Return a relaxed seated scene: face centred, arms down, hands at the wrists."""
    xy = np.zeros((KEYPOINT_COUNT, 2), dtype=np.float64)
    center = np.asarray(face_center, dtype=np.float64)
    for name, offset in _FACE_OFFSETS.items():
        xy[LAYOUT.index(name)] = center + np.asarray(offset, dtype=np.float64) * fw
    for name, sign in (("left", -1.0), ("right", 1.0)):
        for anchor, height in (("shoulder", 1.6), ("elbow", 2.4), ("wrist", 3.2)):
            point = center + np.asarray([sign * 1.0, height], dtype=np.float64) * fw
            xy[LAYOUT.index(f"{name}_{anchor}")] = point
    for side in (Side.LEFT, Side.RIGHT):
        wrist = xy[LAYOUT.index(f"{side.value}_wrist")]
        set_hand(xy, side, wrist, HandShape.OPEN_PALM)
    return xy


def xy_hand_center(xy: NDArray[np.float64], side: Side) -> NDArray[np.float64]:
    """Return the current centroid of one hand's keypoints."""
    return np.mean(xy[np.asarray(LAYOUT.hand(side), dtype=np.int64)], axis=0)


def nearest_anchor(
    xy: NDArray[np.float64],
    side: Side,
    fw: float = FACE_WIDTH_PX,
) -> tuple[float, str]:
    """Return the ``(distance_fw, anchor_name)`` nearest to one hand centroid."""
    origin = xy[LAYOUT.index("nose")]
    center = xy_hand_center(xy, side)
    distances = {
        name: float(np.linalg.norm(center - (origin + np.asarray(offset) * fw)))
        for name, offset in _FACE_OFFSETS.items()
    }
    name = min(distances, key=distances.__getitem__)
    return distances[name] / fw, name


def face_centroid(xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the centroid of the layout face anchors of a scene."""
    return np.mean(xy[np.asarray(LAYOUT.face, dtype=np.int64)], axis=0)


def hand_points(
    center: ArrayLike,
    shape: HandShape,
    *,
    unit: float = HAND_UNIT_PX,
    pinch: bool = False,
) -> NDArray[np.float64]:
    """Return 21 hand keypoints whose centroid sits at ``center``.

    ``unit`` is the pixel length of one local hand unit; the wrist-to-middle-MCP
    distance is ``0.4 * unit``. ``pinch`` moves the thumb tip next to the index
    tip, so ``POINT_PINCH`` covers both the point and the pinch variants.
    """
    local = np.zeros((21, 2), dtype=np.float64)
    extended = _EXTENDED_FINGERS[shape]
    for mcp_index, mcp in _FINGERS.items():
        mcp_point = np.asarray(mcp, dtype=np.float64)
        if mcp_index in extended:
            chain = [
                mcp_point,
                mcp_point + np.array([0.0, -0.25]),
                mcp_point + np.array([0.0, -0.45]),
                mcp_point + np.array([0.0, -0.60]),
            ]
        else:
            chain = [
                mcp_point,
                mcp_point + np.array([0.01, 0.15]),
                mcp_point + np.array([0.03, 0.20]),
                mcp_point + np.array([0.06, 0.20]),
            ]
        local[mcp_index : mcp_index + 4] = chain
    local[0] = (0.0, 0.0)
    local[1] = (-0.12, -0.05)
    local[2] = (-0.30, -0.15)
    local[3] = (-0.42, -0.28)
    local[4] = _INDEX_TIP + np.array([0.04, 0.05]) if pinch else _THUMB_TIP[shape]
    points = local * unit
    points += np.asarray(center, dtype=np.float64) - np.mean(points, axis=0)
    return points


def set_hand(
    xy: NDArray[np.float64],
    side: Side,
    center: ArrayLike,
    shape: HandShape,
    *,
    unit: float = HAND_UNIT_PX,
    pinch: bool = False,
) -> None:
    """Place one hand (and its body wrist) with the hand centroid at ``center``."""
    hand = np.asarray(LAYOUT.hand(side), dtype=np.int64)
    xy[hand] = hand_points(center, shape, unit=unit, pinch=pinch)
    xy[LAYOUT.index(f"{side.value}_wrist")] = np.asarray(center, dtype=np.float64)


def set_hand_distance(
    xy: NDArray[np.float64],
    side: Side,
    anchor: ArrayLike,
    distance_fw: float,
    shape: HandShape,
    *,
    fw: float = FACE_WIDTH_PX,
    unit: float = HAND_UNIT_PX,
    direction: tuple[float, float] = (0.0, 1.0),
    pinch: bool = False,
) -> NDArray[np.float64]:
    """Place the hand ``distance_fw`` from ``anchor`` along ``direction``."""
    offset = np.asarray(direction, dtype=np.float64)
    offset = offset / np.linalg.norm(offset)
    center = np.asarray(anchor, dtype=np.float64) + offset * distance_fw * fw
    set_hand(xy, side, center, shape, unit=unit, pinch=pinch)
    return center


def make_frame(
    seq: int,
    t: float,
    xy: NDArray[np.float64],
    *,
    conf: ArrayLike = 0.9,
    status: NDArray[np.int8] | None = None,
    vel: NDArray[np.float64] | None = None,
    acc: NDArray[np.float64] | None = None,
    face_width_px: float | None = FACE_WIDTH_PX,
    framing: FramingState = FramingState.HALF_BODY,
) -> TrackedFrame:
    """Build one tracked frame from keypoint positions."""
    points = np.asarray(xy, dtype=np.float64)
    confidence = np.asarray(conf, dtype=np.float64)
    if confidence.ndim == 0:
        confidence = np.full(points.shape[0], float(confidence), dtype=np.float64)
    if status is None:
        status_array = np.full(points.shape[0], int(KeypointStatus.MEASURED), dtype=np.int8)
    else:
        status_array = np.asarray(status, dtype=np.int8)
    zeros = np.zeros_like(points)
    return TrackedFrame(
        seq=seq,
        t=float(t),
        framing=framing,
        box=np.array([0.0, 0.0, float(points[:, 0].max()), float(points[:, 1].max())]),
        xy=points,
        vel=zeros if vel is None else np.asarray(vel, dtype=np.float64),
        acc=zeros if acc is None else np.asarray(acc, dtype=np.float64),
        xy_std=np.zeros_like(points),
        vel_std=np.zeros_like(points),
        acc_std=np.ones_like(points),
        status=status_array,
        conf=confidence,
        face_width_px=face_width_px,
        foreshortening={Side.LEFT: None, Side.RIGHT: None},
    )


def frames_from_positions(
    times: ArrayLike,
    positions: Sequence[NDArray[np.float64]],
    *,
    face_width_px: ArrayLike | float | None = FACE_WIDTH_PX,
    framing: FramingState = FramingState.HALF_BODY,
    conf: ArrayLike = 0.9,
) -> list[TrackedFrame]:
    """Build tracked frames with finite-difference velocity and acceleration."""
    timestamps = np.asarray(times, dtype=np.float64)
    stack = np.stack([np.asarray(position, dtype=np.float64) for position in positions])
    if timestamps.size >= 2:
        velocity = np.gradient(stack, timestamps, axis=0)
        acceleration = np.gradient(velocity, timestamps, axis=0)
    else:
        velocity = np.zeros_like(stack)
        acceleration = np.zeros_like(stack)
    if np.ndim(face_width_px) == 0:
        widths: list[float | None] = [None if face_width_px is None else float(face_width_px)]
        widths = widths * timestamps.size
    else:
        widths = [
            None if not np.isfinite(float(value)) else float(value) for value in face_width_px
        ]
    return [
        make_frame(
            index,
            float(timestamps[index]),
            stack[index],
            conf=conf,
            vel=velocity[index],
            acc=acceleration[index],
            face_width_px=widths[index],
            framing=framing,
        )
        for index in range(timestamps.size)
    ]


def distance_profile(
    keyframes: Sequence[tuple[float, float]],
    *,
    side: Side,
    shape: HandShape,
    anchor: ArrayLike,
    fw: float = FACE_WIDTH_PX,
    face_center: ArrayLike = FACE_CENTER,
    unit: float = HAND_UNIT_PX,
    pinch: bool = False,
    conf: ArrayLike = 0.9,
    direction: tuple[float, float] = (0.0, 1.0),
) -> list[TrackedFrame]:
    """Build a one-hand trajectory from ``(time, distance_fw)`` keyframes.

    The hand approaches ``anchor`` along ``direction`` (image-down by default),
    so the target anchor stays the nearest one near contact.
    """
    first, last = keyframes[0][0], keyframes[-1][0]
    times = np.arange(first, last + 0.5 * DT, DT, dtype=np.float64)
    distances = np.interp(
        times,
        [keyframe[0] for keyframe in keyframes],
        [keyframe[1] for keyframe in keyframes],
    )
    positions = []
    for distance in distances:
        xy = neutral_xy(face_center, fw)
        set_hand_distance(
            xy,
            side,
            anchor,
            float(distance),
            shape,
            fw=fw,
            unit=unit,
            direction=direction,
            pinch=pinch,
        )
        positions.append(xy)
    return frames_from_positions(times, positions, face_width_px=fw, conf=conf)


def shift_face(
    xy: NDArray[np.float64],
    offset_fw: tuple[float, float],
    fw: float = FACE_WIDTH_PX,
) -> None:
    """Translate every face anchor by ``offset_fw`` face-widths."""
    offset = np.asarray(offset_fw, dtype=np.float64) * fw
    face = np.asarray(LAYOUT.face, dtype=np.int64)
    xy[face] = xy[face] + offset
