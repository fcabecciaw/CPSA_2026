"""Hand-shape classification from 21 keypoints (FR-014, AC-007).

The keypoint order is COCO-WholeBody: 0 wrist, 1-4 thumb, 5-8 index, 9-12
middle, 13-16 ring, 17-20 little finger. The classifier is scale free: every
distance is normalized by the wrist-to-middle-MCP distance of the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cpsa.domain.model import HandShape

_EPS = 1e-9
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_MCP = 9
_FINGER_MCP = (5, 9, 13, 17)
_FINGER_TIP = (8, 12, 16, 20)


@dataclass(frozen=True)
class HandShapeConfig:
    """Thresholds of the hand-shape rules.

    A finger is extended when the tip-to-wrist distance is at least
    ``extended_ratio`` times the MCP-to-wrist distance. A pinch is a thumb tip
    closer than ``pinch_ratio`` hand scales to the index tip. ``min_conf`` is the
    confidence below which a keypoint is unusable, and ``min_keypoints`` is the
    number of usable keypoints required before any classification is attempted.
    """

    extended_ratio: float = 1.4
    pinch_ratio: float = 0.3
    min_conf: float = 0.3
    min_keypoints: int = 15


def _distance(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(first - second))


def classify_hand_shape(
    xy: ArrayLike,
    conf: ArrayLike,
    cfg: HandShapeConfig,
) -> HandShape:
    """Classify one hand as fist, point/pinch or open palm (FR-014).

    A fist is all four fingers curled. A point/pinch is an extended index with
    the other three curled, or a thumb tip close to the index tip. An open palm
    is all four fingers extended. Missing or low-confidence keypoints, a
    degenerate hand and every other configuration give ``UNKNOWN``.
    """
    points = np.asarray(xy, dtype=np.float64)
    confidence = np.asarray(conf, dtype=np.float64).reshape(-1)
    if points.shape != (21, 2):
        raise ValueError(f"xy must have shape (21, 2), got {points.shape}")
    if confidence.shape != (21,):
        raise ValueError(f"conf must have shape (21,), got {confidence.shape}")

    usable = np.all(np.isfinite(points), axis=1) & np.isfinite(confidence)
    usable &= confidence >= cfg.min_conf
    if int(np.count_nonzero(usable)) < cfg.min_keypoints:
        return HandShape.UNKNOWN

    required = (_WRIST, _MIDDLE_MCP, _THUMB_TIP, _INDEX_TIP, *_FINGER_MCP, *_FINGER_TIP)
    if not bool(np.all(usable[list(required)])):
        return HandShape.UNKNOWN

    wrist = points[_WRIST]
    scale = _distance(points[_MIDDLE_MCP], wrist)
    if scale <= _EPS:
        return HandShape.UNKNOWN

    extended: list[bool] = []
    for mcp, tip in zip(_FINGER_MCP, _FINGER_TIP, strict=True):
        reference = _distance(points[mcp], wrist)
        if reference <= _EPS:
            return HandShape.UNKNOWN
        extended.append(_distance(points[tip], wrist) / reference >= cfg.extended_ratio)

    index_extended = extended[0]
    other_curled = not any(extended[1:])
    pinch = _distance(points[_THUMB_TIP], points[_INDEX_TIP]) / scale <= cfg.pinch_ratio

    if not any(extended):
        return HandShape.FIST
    if (index_extended and other_curled) or pinch:
        return HandShape.POINT_PINCH
    if all(extended):
        return HandShape.OPEN_PALM
    return HandShape.UNKNOWN
