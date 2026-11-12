"""F-G crop convention: square, upright hand crops around a rotated box.

A crop is the tuple ``(cx, cy, side, angle)``: the centre in full-frame pixels,
the side of the square in full-frame pixels and the rotation in radians. Let
``u`` be the hand's up direction in the image (wrist to fingers); ``angle`` is
the rotation that maps ``u`` onto image up, ``(0, -1)``. The forward map of a
full-frame point ``p`` to the ``N x N`` crop is

``q = R(angle) @ (p - c) * (N / side) + (N/2, N/2)``

with ``R(a) = [[cos a, -sin a], [sin a, cos a]]`` in image coordinates (``y``
points down). :func:`crop_matrix` returns the 2x3 OpenCV matrix of that map, so
a crop is ``cv2.warpAffine(img, crop_matrix(...), (N, N))``; :func:`to_crop`
and :func:`from_crop` apply the forward and inverse maps to points, for example
to move crop keypoints back to the full frame.

This module is pure NumPy so the training dataset builder and the runtime
localizer share one implementation of the convention.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_CROP_SIDE",
    "angle_from_up",
    "crop_matrix",
    "from_crop",
    "to_crop",
]

DEFAULT_CROP_SIDE = 192
"""Side of the square model input, in pixels (F-G)."""

_COORDINATES = 2
_IMAGE_UP = np.array([0.0, -1.0])
"""Up direction of the image (``y`` points down)."""


def crop_matrix(
    cx: float,
    cy: float,
    side: float,
    angle: float,
    n: int = DEFAULT_CROP_SIDE,
) -> NDArray[np.float64]:
    """Return the 2x3 forward matrix of the F-G crop as ``(cx, cy, side, angle)``.

    Args:
        cx: Crop centre column in full-frame pixels.
        cy: Crop centre row in full-frame pixels.
        side: Side of the crop in full-frame pixels.
        angle: Rotation that maps the hand up direction onto image up, radians.
        n: Side of the square crop in pixels.

    Returns:
        ``(2, 3)`` matrix mapping full-frame points to crop pixels.

    Raises:
        ValueError: If ``side`` or ``n`` is not positive, or a centre is not
            finite.
    """

    if not math.isfinite(cx) or not math.isfinite(cy):
        raise ValueError(f"crop centre must be finite, got ({cx}, {cy})")
    if side <= 0.0:
        raise ValueError(f"crop side must be positive, got {side}")
    if n <= 0:
        raise ValueError(f"crop size must be positive, got {n}")

    scale = float(n) / float(side)
    cos, sin = math.cos(angle), math.sin(angle)
    linear = scale * np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    half = float(n) / 2.0
    offset = np.array([half, half], dtype=np.float64) - linear @ np.array(
        [cx, cy], dtype=np.float64
    )
    return np.hstack([linear, offset[:, np.newaxis]])


def to_crop(
    points: ArrayLike,
    cx: float,
    cy: float,
    side: float,
    angle: float,
    n: int = DEFAULT_CROP_SIDE,
) -> NDArray[np.float64]:
    """Map full-frame ``(..., 2)`` points to crop pixels (F-G forward map).

    Args:
        points: Full-frame ``x, y`` pixels, shape ``(K, 2)`` or ``(2,)``.
        cx: Crop centre column in full-frame pixels.
        cy: Crop centre row in full-frame pixels.
        side: Side of the crop in full-frame pixels.
        angle: Rotation onto image up, radians.
        n: Side of the square crop in pixels.

    Returns:
        Crop pixels with the same shape as ``points``.

    Raises:
        ValueError: If ``points`` does not have two coordinates per point, or
            the crop is invalid (see :func:`crop_matrix`).
    """

    array = _as_points(points)
    matrix = crop_matrix(cx, cy, side, angle, n)
    return np.asarray(
        array @ matrix[:, :_COORDINATES].T + matrix[:, _COORDINATES], dtype=np.float64
    )


def from_crop(
    points: ArrayLike,
    cx: float,
    cy: float,
    side: float,
    angle: float,
    n: int = DEFAULT_CROP_SIDE,
) -> NDArray[np.float64]:
    """Map crop ``(..., 2)`` points back to full-frame pixels (F-G inverse map).

    Args:
        points: Crop ``x, y`` pixels, shape ``(K, 2)`` or ``(2,)``.
        cx: Crop centre column in full-frame pixels.
        cy: Crop centre row in full-frame pixels.
        side: Side of the crop in full-frame pixels.
        angle: Rotation onto image up, radians.
        n: Side of the square crop in pixels.

    Returns:
        Full-frame pixels with the same shape as ``points``.

    Raises:
        ValueError: If ``points`` does not have two coordinates per point, or
            the crop is invalid (see :func:`crop_matrix`).
    """

    array = _as_points(points)
    matrix = crop_matrix(cx, cy, side, angle, n)
    # Invert q = L p + t, where t already carries -L @ c: p = L^-1 (q - t).
    scale = float(n) / float(side)
    cos, sin = math.cos(angle), math.sin(angle)
    inverse = (1.0 / scale) * np.array([[cos, sin], [-sin, cos]], dtype=np.float64)
    return np.asarray((array - matrix[:, _COORDINATES]) @ inverse.T, dtype=np.float64)


def angle_from_up(up: ArrayLike) -> float:
    """Return the F-G ``angle`` that maps ``up`` onto image up.

    Args:
        up: Hand up direction in the image (wrist to fingers), any non-zero
            length.

    Returns:
        Rotation in radians in ``(-pi, pi]``.

    Raises:
        ValueError: If ``up`` is not a 2-vector or has zero length.
    """

    vector = _as_points(up)
    if vector.shape != (_COORDINATES,):
        raise ValueError(f"up must be one 2-vector, got shape {vector.shape}")
    length = float(np.hypot(vector[0], vector[1]))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"up must be a non-zero vector, got {vector}")
    # R(angle) @ up ~ image up; with R a rotation, angle is the polar angle of
    # the image-up vector expressed in the rotated frame.
    return math.atan2(-float(vector[0]), -float(vector[1]))


def _as_points(points: ArrayLike) -> NDArray[np.float64]:
    """Return ``points`` as a float array whose last axis holds ``x, y``."""

    array = np.asarray(points, dtype=np.float64)
    if array.shape[-1:] != (_COORDINATES,):
        raise ValueError(f"points must be (..., 2), got shape {array.shape}")
    return array
