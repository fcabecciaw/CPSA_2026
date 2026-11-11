"""3D hands: the frames the hand modes consume and the 2D-to-3D lift (F-B).

A hand is detected by the pose model on the IR image, so its keypoints are IR
pixels. :func:`lift_keypoints` reads the aligned raw depth around each keypoint,
converts the median to metres and projects the pixel back with the IR pinhole
model. A keypoint without a depth measurement stays at the origin and is
flagged invalid, so consumers never mistake a missing depth for zero metres.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cpsa.vision.calibration import DepthModel, Intrinsics

__all__ = ["Hand3D", "HandFrame", "lift_keypoints"]

_COORDINATES_2D = 3
_COORDINATES_3D = 4


@dataclass(frozen=True)
class Hand3D:
    """One hand in one frame.

    ``box`` is ``(x1, y1, x2, y2)`` in IR pixels; ``kpts2d`` is ``(K, 3)`` with
    ``u, v, conf`` in IR pixels; ``kpts3d`` is ``(K, 4)`` with ``x, y, z`` in
    metres in the IR camera frame and a ``0/1`` valid flag. Invalid keypoints
    have ``x = y = z = 0``.
    """

    score: float
    box: tuple[float, float, float, float]
    kpts2d: NDArray[np.float32]
    kpts3d: NDArray[np.float32]


@dataclass(frozen=True)
class HandFrame:
    """Pose-model output for one IR frame, ready for the protocol builders.

    ``timing`` holds ``decode_ms``, ``preprocess_ms``, ``dpu_ms``, ``post_ms``
    and ``lift_ms``.
    """

    seq: int
    t: float
    width: int
    height: int
    hands: tuple[Hand3D, ...]
    timing: Mapping[str, float]


def _window_median(depth_raw: NDArray[np.uint16], row: int, column: int, half: int) -> float | None:
    """Median of the nonzero raw depth in the window around ``(row, column)``."""

    height, width = depth_raw.shape
    top = max(0, row - half)
    bottom = min(height, row + half + 1)
    left = max(0, column - half)
    right = min(width, column + half + 1)
    patch = depth_raw[top:bottom, left:right]
    measured = patch[patch > 0]
    if measured.size == 0:
        return None
    return float(np.median(measured))


def lift_keypoints(
    kpts2d: NDArray[np.float32],
    depth_raw: NDArray[np.uint16],
    intrinsics: Intrinsics,
    depth: DepthModel,
    window: int = 5,
) -> NDArray[np.float32]:
    """Lift ``(K, 3)`` IR-pixel keypoints to ``(K, 4)`` metric ones.

    For each keypoint the raw depth is taken as the median of the nonzero
    samples in a ``window x window`` box around ``(round(u), round(v))``, clipped
    to the image at the borders. The median is converted to metres with
    ``depth`` and the keypoint is projected with the IR pinhole model. A
    keypoint whose window has no nonzero sample, or whose pixel is not finite,
    is marked invalid and left at the origin.

    Args:
        kpts2d: ``(K, 3)`` array of ``u, v, conf`` in IR pixels.
        depth_raw: ``(H, W)`` raw Z16 depth, pixel-aligned with the IR image.
        intrinsics: IR intrinsics; only the pinhole terms are used.
        depth: Raw depth to metres model.
        window: Odd side of the median window, in pixels.

    Raises:
        ValueError: If the shapes disagree, ``window`` is not a positive odd
            number, or the focal lengths are not positive.
    """

    kpts = np.asarray(kpts2d, dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[1] < _COORDINATES_2D:
        raise ValueError(f"kpts2d must be (K, 3), got shape {kpts.shape}")
    raw = np.asarray(depth_raw)
    if raw.ndim != 2:
        raise ValueError(f"depth_raw must be (H, W), got shape {raw.shape}")
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd number, got {window}")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError(
            f"intrinsics need positive focal lengths, got fx={intrinsics.fx}, fy={intrinsics.fy}"
        )

    count = kpts.shape[0]
    lifted = np.zeros((count, _COORDINATES_3D), dtype=np.float32)
    if count == 0:
        return lifted
    height, width = raw.shape
    half = window // 2
    scale = float(depth.scale_m)
    offset = float(depth.offset_m)

    for index in range(count):
        u = float(kpts[index, 0])
        v = float(kpts[index, 1])
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        column = min(max(int(round(u)), 0), width - 1)
        row = min(max(int(round(v)), 0), height - 1)
        median = _window_median(raw, row, column, half)
        if median is None:
            continue
        z_m = median * scale + offset
        if not np.isfinite(z_m) or z_m <= 0.0:
            continue
        lifted[index, 0] = (u - intrinsics.cx) * z_m / intrinsics.fx
        lifted[index, 1] = (v - intrinsics.cy) * z_m / intrinsics.fy
        lifted[index, 2] = z_m
        lifted[index, 3] = 1.0
    return lifted
