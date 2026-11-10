"""Camera calibration model, projection and depth lifting.

All point arrays follow the OpenCV camera convention: ``x`` right, ``y`` down,
``z`` forward along the optical axis, in metres. Image coordinates are pixels
with the origin at the top-left corner of the pixel grid.

Distortion uses the Brown-Conrady model with OpenCV's coefficient order
``(k1, k2, p1, p2, k3)``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

_DIST_LENGTH = 5
_UNDISTORT_ITERATIONS = 20
_UNDISTORT_TOLERANCE = 1e-14
_ROTATION_TOLERANCE = 1e-6
_DEFAULT_Z_MIN_M = 0.2
_MAX_ROI_FRACTION = 0.5
_LOCAL_BOUND_MARGIN_PX = 12.0
_DISTORTION_TILE_PX = 64
_ROI_RANGE_TOLERANCE_M = 1e-3


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics with Brown-Conrady distortion (OpenCV order)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Extrinsics:
    """Rigid transform ``p_target = R @ p_source + t_m``."""

    R: NDArray[np.float64]
    t_m: NDArray[np.float64]


@dataclass(frozen=True)
class DepthModel:
    """Raw Z16 depth to metres: ``depth_m = raw * scale_m + offset_m``."""

    scale_m: float
    offset_m: float


@dataclass(frozen=True)
class Calibration:
    """The ``calibration.json`` v1 content used by the vision pipeline.

    ``T_ir_rgb`` maps a point from the IR camera frame into the RGB camera frame:
    ``p_rgb = R @ p_ir + t_m``. ``depth`` applies to the raw IR depth image.
    """

    rgb: Intrinsics
    ir: Intrinsics
    T_ir_rgb: Extrinsics
    depth: DepthModel


def _mapping(value: object, key: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"calibration key '{key}' must be an object")
    return cast(Mapping[str, Any], value)


def _get(obj: Mapping[str, Any], key: str, path: str | None = None) -> Any:
    if key not in obj:
        raise ValueError(f"calibration key '{path or key}' is missing")
    return obj[key]


def _number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"calibration key '{key}' must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"calibration key '{key}' must be finite")
    return number


def _integer(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"calibration key '{key}' must be an integer")
    if value <= 0:
        raise ValueError(f"calibration key '{key}' must be positive")
    return value


def _vector(value: object, key: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"calibration key '{key}' must be a list of {length} numbers")
    return [_number(item, key) for item in value]


def _matrix3(value: object, key: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"calibration key '{key}' must be a 3x3 matrix")
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"calibration key '{key}' must be a 3x3 matrix")
        rows.append([_number(item, key) for item in row])
    return rows


def _rotation(value: object, key: str) -> list[list[float]]:
    """Validate that ``value`` is a proper rotation matrix."""
    rows = _matrix3(value, key)
    rotation = np.asarray(rows, dtype=np.float64)
    orthonormal_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    if orthonormal_error > _ROTATION_TOLERANCE:
        raise ValueError(f"calibration key '{key}' must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > _ROTATION_TOLERANCE:
        raise ValueError(f"calibration key '{key}' must have determinant +1")
    return rows


def _parse_intrinsics(root: Mapping[str, Any], name: str) -> Intrinsics:
    obj = _mapping(_get(root, name), name)
    width = _integer(_get(obj, "w", f"{name}.w"), f"{name}.w")
    height = _integer(_get(obj, "h", f"{name}.h"), f"{name}.h")
    K = _matrix3(_get(obj, "K", f"{name}.K"), f"{name}.K")
    dist = _vector(_get(obj, "dist", f"{name}.dist"), f"{name}.dist", _DIST_LENGTH)
    if K[0][1] != 0.0 or K[1][0] != 0.0 or K[2] != [0.0, 0.0, 1.0]:
        raise ValueError(f"calibration key '{name}.K' must be a pinhole matrix")
    if K[0][0] <= 0.0:
        raise ValueError(f"calibration key '{name}.K' must have fx > 0")
    if K[1][1] <= 0.0:
        raise ValueError(f"calibration key '{name}.K' must have fy > 0")
    return Intrinsics(
        fx=K[0][0],
        fy=K[1][1],
        cx=K[0][2],
        cy=K[1][2],
        width=width,
        height=height,
        dist=(dist[0], dist[1], dist[2], dist[3], dist[4]),
    )


def _parse_rms(root: Mapping[str, Any]) -> None:
    rms = _mapping(_get(root, "rms"), "rms")
    for key in ("rgb_px", "ir_px", "stereo_px", "depth_mm"):
        value = _number(_get(rms, key, f"rms.{key}"), f"rms.{key}")
        if value < 0.0:
            raise ValueError(f"calibration key 'rms.{key}' must be non-negative")


def _parse_board(root: Mapping[str, Any]) -> None:
    board = _mapping(_get(root, "board"), "board")
    squares = _get(board, "squares", "board.squares")
    if not isinstance(squares, list) or len(squares) != 2:
        raise ValueError("calibration key 'board.squares' must be a list of 2 integers")
    for item in squares:
        _integer(item, "board.squares")
    for key in ("square_mm", "marker_mm"):
        value = _number(_get(board, key, f"board.{key}"), f"board.{key}")
        if value <= 0.0:
            raise ValueError(f"calibration key 'board.{key}' must be positive")
    dictionary = _get(board, "dictionary", "board.dictionary")
    if not isinstance(dictionary, str) or not dictionary:
        raise ValueError("calibration key 'board.dictionary' must be a non-empty string")


def load_calibration(path: str | Path) -> Calibration:
    """Load and validate a ``calibration.json`` v1 file.

    Raises ``ValueError`` naming the offending key when the schema is invalid.
    """
    with Path(path).open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    root = _mapping(data, "calibration")
    version = _get(root, "schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError(f"calibration key 'schema_version' must be the integer 1, got {version!r}")

    rgb = _parse_intrinsics(root, "rgb")
    ir = _parse_intrinsics(root, "ir")

    extrinsic = _mapping(_get(root, "T_ir_rgb"), "T_ir_rgb")
    rotation = _rotation(_get(extrinsic, "R", "T_ir_rgb.R"), "T_ir_rgb.R")
    translation = _vector(_get(extrinsic, "t_m", "T_ir_rgb.t_m"), "T_ir_rgb.t_m", 3)

    depth = _mapping(_get(root, "depth"), "depth")
    scale_m = _number(_get(depth, "scale_m", "depth.scale_m"), "depth.scale_m")
    if scale_m <= 0.0:
        raise ValueError("calibration key 'depth.scale_m' must be positive")
    offset_m = _number(_get(depth, "offset_m", "depth.offset_m"), "depth.offset_m")
    _parse_rms(root)
    _parse_board(root)

    return Calibration(
        rgb=rgb,
        ir=ir,
        T_ir_rgb=Extrinsics(
            R=np.asarray(rotation, dtype=np.float64),
            t_m=np.asarray(translation, dtype=np.float64),
        ),
        depth=DepthModel(scale_m=scale_m, offset_m=offset_m),
    )


def _distort_normalized(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dist: tuple[float, float, float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def _distort_jacobian(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dist: tuple[float, float, float, float, float],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    dradial = 2.0 * (k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2))
    dradial_dx = dradial * x
    dradial_dy = dradial * y
    j11 = radial + dradial_dx * x + 2.0 * p1 * y + 6.0 * p2 * x
    j12 = dradial_dy * x + 2.0 * p1 * x + 2.0 * p2 * y
    j21 = dradial_dx * y + 2.0 * p1 * x + 2.0 * p2 * y
    j22 = radial + dradial_dy * y + 6.0 * p1 * y + 2.0 * p2 * x
    return j11, j12, j21, j22


def _undistort_normalized(
    xd: NDArray[np.float64],
    yd: NDArray[np.float64],
    dist: tuple[float, float, float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Invert Brown-Conrady distortion with Newton-Raphson.

    Converges to ``|distorted(x) - (xd, yd)| < 1e-14`` in normalised coordinates
    for coefficients produced by a real calibration; typical residual after
    reprojection is far below 1e-6 px. Strongly out-of-range points are clamped
    only by the iteration limit, so callers should stay inside the calibrated
    image.
    """
    x = xd.copy()
    y = yd.copy()
    if not any(dist):
        return x, y
    for _ in range(_UNDISTORT_ITERATIONS):
        xd_est, yd_est = _distort_normalized(x, y, dist)
        ex = xd_est - xd
        ey = yd_est - yd
        if ex.size == 0 or max(float(np.max(np.abs(ex))), float(np.max(np.abs(ey)))) < (
            _UNDISTORT_TOLERANCE
        ):
            break
        j11, j12, j21, j22 = _distort_jacobian(x, y, dist)
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-12, 1.0, det)
        step_x = (j22 * ex - j12 * ey) / det
        step_y = (-j21 * ex + j11 * ey) / det
        x = x - step_x
        y = y - step_y
    return x, y


def project(points_cam: ArrayLike, intr: Intrinsics) -> NDArray[np.float64]:
    """Project camera-frame points ``(..., 3)`` to distorted pixels ``(..., 2)``.

    Points with ``z <= 0`` are outside the pinhole model; their projection is
    undefined and yields ``inf``/``nan`` after a divide-by-zero.
    """
    points = np.asarray(points_cam, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError(f"expected camera points with shape (..., 3), got {points.shape}")
    z = points[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = points[..., 0] / z
        y = points[..., 1] / z
        if any(intr.dist):
            xd, yd = _distort_normalized(x, y, intr.dist)
        else:
            xd, yd = x, y
        u = intr.fx * xd + intr.cx
        v = intr.fy * yd + intr.cy
    return np.stack((u, v), axis=-1)


def project_jacobian(points_cam: ArrayLike, intr: Intrinsics) -> NDArray[np.float64]:
    """Jacobian ``d(u, v) / d(X, Y, Z)`` for :func:`project`, shape ``(..., 2, 3)``."""
    points = np.asarray(points_cam, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError(f"expected camera points with shape (..., 3), got {points.shape}")
    z = points[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = points[..., 0] / z
        y = points[..., 1] / z
        j11, j12, j21, j22 = _distort_jacobian(x, y, intr.dist)
        inv_z = 1.0 / z
        jacobian = np.empty(points.shape[:-1] + (2, 3), dtype=np.float64)
        jacobian[..., 0, 0] = intr.fx * j11 * inv_z
        jacobian[..., 0, 1] = intr.fx * j12 * inv_z
        jacobian[..., 0, 2] = -intr.fx * (j11 * x + j12 * y) * inv_z
        jacobian[..., 1, 0] = intr.fy * j21 * inv_z
        jacobian[..., 1, 1] = intr.fy * j22 * inv_z
        jacobian[..., 1, 2] = -intr.fy * (j21 * x + j22 * y) * inv_z
    return jacobian


def deproject(uv: ArrayLike, z_m: ArrayLike, intr: Intrinsics) -> NDArray[np.float64]:
    """Lift distorted pixels ``(..., 2)`` at depths ``z_m`` (...,) to camera points.

    Distortion is inverted iteratively (Newton-Raphson); the residual is below
    ``1e-6 px`` for coefficients from a real calibration, including the image
    corners.
    """
    pixels = np.asarray(uv, dtype=np.float64)
    if pixels.shape[-1:] != (2,):
        raise ValueError(f"expected pixels with shape (..., 2), got {pixels.shape}")
    depths = np.asarray(z_m, dtype=np.float64)
    xd = (pixels[..., 0] - intr.cx) / intr.fx
    yd = (pixels[..., 1] - intr.cy) / intr.fy
    x, y = _undistort_normalized(xd, yd, intr.dist)
    depth = np.broadcast_to(depths, x.shape)
    return np.stack((x * depth, y * depth, depth), axis=-1)


def depth_to_m(raw: ArrayLike, depth_model: DepthModel) -> NDArray[np.float32]:
    """Convert raw Z16 depth to metres, using ``NaN`` for invalid (zero) pixels."""
    raw_array = np.asarray(raw)
    depth: NDArray[np.float32] = raw_array.astype(np.float32) * np.float32(
        depth_model.scale_m
    ) + np.float32(depth_model.offset_m)
    depth = np.where(raw_array == 0, np.float32("nan"), depth).astype(np.float32)
    return depth


@lru_cache(maxsize=4)
def _ir_rays(intr: Intrinsics) -> NDArray[np.float64]:
    """Normalised, undistorted IR rays ``(height, width, 2)`` for every pixel.

    Undistortion depends only on the pixel grid and intrinsics, so it is computed
    once per calibration and reused across frames. The returned array is
    read-only; callers must not mutate it.
    """
    v, u = np.indices((intr.height, intr.width), dtype=np.float64)
    xd = (u - intr.cx) / intr.fx
    yd = (v - intr.cy) / intr.fy
    x, y = _undistort_normalized(xd, yd, intr.dist)
    rays = np.stack((x, y), axis=-1)
    rays.setflags(write=False)
    return rays


@lru_cache(maxsize=8)
def _distortion_bound_pixels(intr: Intrinsics) -> float:
    """Conservative bound on ``|distorted_pixel - pinhole_pixel|`` per axis.

    The bound is evaluated over the image plus a margin that is expanded until
    it contains its own displacement bound. This self-consistent fixed point
    covers every pixel that can be registered into the image, including points
    whose pinhole coordinate lies just outside it. Returns ``inf`` when the
    bound cannot be established or exceeds a quarter of the smaller image
    dimension.
    """
    if not any(intr.dist):
        return 0.0
    limit = 0.25 * float(min(intr.width, intr.height))
    step = 2.0
    margin = 4.0
    while margin <= limit:
        columns = np.arange(-margin, intr.width + margin, step, dtype=np.float64)
        rows = np.arange(-margin, intr.height + margin, step, dtype=np.float64)
        u, v = np.meshgrid(columns, rows)
        x = (u - intr.cx) / intr.fx
        y = (v - intr.cy) / intr.fy
        xd, yd = _distort_normalized(x, y, intr.dist)
        displacement_u = np.abs(intr.fx * (xd - x))
        displacement_v = np.abs(intr.fy * (yd - y))
        j11, j12, j21, j22 = _distort_jacobian(x, y, intr.dist)
        with np.errstate(invalid="ignore", over="ignore"):
            # Jacobian of the pixel displacement with respect to pixel coordinates.
            du_du = j11 - 1.0
            du_dv = intr.fx * j12 / intr.fy
            dv_du = intr.fy * j21 / intr.fx
            dv_dv = j22 - 1.0
            lipschitz = np.sqrt(du_du**2 + du_dv**2 + dv_du**2 + dv_dv**2)
            if not (
                np.all(np.isfinite(displacement_u))
                and np.all(np.isfinite(displacement_v))
                and np.all(np.isfinite(lipschitz))
            ):
                return float("inf")
            padding = 1.0 + 1.1 * float(np.max(lipschitz)) * step / np.sqrt(2.0)
            bound = max(float(np.max(displacement_u)), float(np.max(displacement_v))) + padding
        if not np.isfinite(bound):
            return float("inf")
        if bound <= margin:
            return float(bound)
        if bound > limit:
            return float("inf")
        margin = min(max(bound * 1.05, margin * 2.0), limit)
    return float("inf")


def _pinhole_project(points: NDArray[np.float64], intr: Intrinsics) -> NDArray[np.float64]:
    """Project camera points without applying lens distortion."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack(
            (
                intr.fx * points[..., 0] / points[..., 2] + intr.cx,
                intr.fy * points[..., 1] / points[..., 2] + intr.cy,
            ),
            axis=-1,
        )


def _distortion_padding_on_box(
    intr: Intrinsics,
    left: float,
    right: float,
    top: float,
    bottom: float,
    max_margin: float,
) -> float | None:
    """Local distortion-displacement bound around a pinhole pixel rectangle.

    Expands the rectangle until the sampled displacement bound fits inside the
    expanded margin, so the result covers every distorted pixel whose pinhole
    coordinate lies in the original rectangle. Returns ``None`` when no bound
    below ``max_margin`` can be established.
    """
    if not any(intr.dist):
        return 0.0
    margin = 2.0
    step = 1.0 if max(right - left, bottom - top) <= 64.0 else 2.0
    while margin <= max_margin:
        columns = np.arange(left - margin, right + margin + step, step, dtype=np.float64)
        rows = np.arange(top - margin, bottom + margin + step, step, dtype=np.float64)
        if columns.size == 0 or rows.size == 0:
            return None
        u, v = np.meshgrid(columns, rows)
        x = (u - intr.cx) / intr.fx
        y = (v - intr.cy) / intr.fy
        xd, yd = _distort_normalized(x, y, intr.dist)
        displacement_u = np.abs(intr.fx * (xd - x))
        displacement_v = np.abs(intr.fy * (yd - y))
        j11, j12, j21, j22 = _distort_jacobian(x, y, intr.dist)
        with np.errstate(invalid="ignore", over="ignore"):
            du_du = j11 - 1.0
            du_dv = intr.fx * j12 / intr.fy
            dv_du = intr.fy * j21 / intr.fx
            dv_dv = j22 - 1.0
            lipschitz = np.sqrt(du_du**2 + du_dv**2 + dv_du**2 + dv_dv**2)
            if not (
                np.all(np.isfinite(displacement_u))
                and np.all(np.isfinite(displacement_v))
                and np.all(np.isfinite(lipschitz))
            ):
                return None
            padding = 1.0 + 1.1 * float(np.max(lipschitz)) * step / np.sqrt(2.0)
            bound = max(float(np.max(displacement_u)), float(np.max(displacement_v))) + padding
        if not np.isfinite(bound):
            return None
        if bound <= margin:
            return float(bound)
        if bound > max_margin:
            return None
        margin = min(max(bound * 1.05, margin * 1.5), max_margin)
    return None


@lru_cache(maxsize=8)
def _distortion_tile_bounds(
    intr: Intrinsics,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Per-tile local distortion bounds covering the image plus one tile."""
    origin_u = np.arange(
        -_DISTORTION_TILE_PX, intr.width + _DISTORTION_TILE_PX, _DISTORTION_TILE_PX
    )
    origin_v = np.arange(
        -_DISTORTION_TILE_PX, intr.height + _DISTORTION_TILE_PX, _DISTORTION_TILE_PX
    )
    bounds = np.full((origin_v.size, origin_u.size), np.nan, dtype=np.float64)
    for row, top in enumerate(origin_v):
        for column, left in enumerate(origin_u):
            local = _distortion_padding_on_box(
                intr,
                float(left),
                float(left + _DISTORTION_TILE_PX - 1),
                float(top),
                float(top + _DISTORTION_TILE_PX - 1),
                _LOCAL_BOUND_MARGIN_PX,
            )
            if local is not None:
                bounds[row, column] = local
    bounds.setflags(write=False)
    return bounds, origin_u, origin_v


def _tile_distortion_bound(
    intr: Intrinsics,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> float | None:
    """Look up a conservative distortion bound for a pixel rectangle."""
    if not any(intr.dist):
        return 0.0
    bounds, origin_u, origin_v = _distortion_tile_bounds(intr)
    margin = _LOCAL_BOUND_MARGIN_PX
    column0 = int(np.floor((left - margin - float(origin_u[0])) / _DISTORTION_TILE_PX))
    column1 = int(np.floor((right + margin - float(origin_u[0])) / _DISTORTION_TILE_PX))
    row0 = int(np.floor((top - margin - float(origin_v[0])) / _DISTORTION_TILE_PX))
    row1 = int(np.floor((bottom + margin - float(origin_v[0])) / _DISTORTION_TILE_PX))
    if column0 < 0 or row0 < 0 or column1 >= bounds.shape[1] or row1 >= bounds.shape[0]:
        return None
    selected = bounds[row0 : row1 + 1, column0 : column1 + 1]
    if not np.all(np.isfinite(selected)):
        return None
    return float(np.max(selected))


def _keypoint_roi_mask(
    pixels: NDArray[np.float64],
    calib: Calibration,
    window: int,
    z_min: float,
    z_max: float,
) -> NDArray[np.bool_] | None:
    """Conservative IR-pixel mask containing every sample a keypoint can use.

    For each RGB keypoint window, the four corners of the (distortion-expanded)
    pinhole window are mapped into the IR frame at the nearest and furthest
    valid depths ``z_min``/``z_max``. Translation and rotation are both applied,
    so the convex hull of the resulting quadrilaterals contains every IR point
    whose registered RGB pixel could fall inside the window. Distortion is
    handled by expanding both image spaces with a per-calibration pixel bound.
    Returns ``None`` when the calibration geometry is uncertain or when the
    union would cover more than half of the image, in which case the caller must
    use full registration.
    """
    rgb = calib.rgb
    ir = calib.ir
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min <= 0.0 or z_max < z_min:
        return None
    rotation = np.asarray(calib.T_ir_rgb.R, dtype=np.float64)
    translation = np.asarray(calib.T_ir_rgb.t_m, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        return None
    if not (np.all(np.isfinite(rotation)) and np.all(np.isfinite(translation))):
        return None
    if (
        float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))) > _ROTATION_TOLERANCE
        or abs(float(np.linalg.det(rotation)) - 1.0) > _ROTATION_TOLERANCE
    ):
        return None
    if rgb.fx <= 0.0 or rgb.fy <= 0.0 or ir.fx <= 0.0 or ir.fy <= 0.0:
        return None

    rgb_bound = _distortion_bound_pixels(rgb)
    ir_bound = _distortion_bound_pixels(ir)
    limit = 0.25 * float(min(rgb.width, rgb.height, ir.width, ir.height))
    if not np.isfinite(rgb_bound) or not np.isfinite(ir_bound):
        return None
    if rgb_bound > limit or ir_bound > limit:
        return None

    finite = np.all(np.isfinite(pixels), axis=1)
    if not np.any(finite):
        return np.zeros((ir.height, ir.width), dtype=np.bool_)
    key_u = pixels[:, 0]
    key_v = pixels[:, 1]
    window_left = np.maximum(0.0, key_u - window)
    window_right = np.minimum(float(rgb.width - 1), key_u + window)
    window_top = np.maximum(0.0, key_v - window)
    window_bottom = np.minimum(float(rgb.height - 1), key_v + window)
    usable = finite & (window_left <= window_right) & (window_top <= window_bottom)
    if not np.any(usable):
        return np.zeros((ir.height, ir.width), dtype=np.bool_)

    # Tighter local distortion bounds keep the parallax region small; the global
    # bound remains the conservative fallback near strong distortion.
    rgb_bounds = np.full(pixels.shape[0], rgb_bound, dtype=np.float64)
    for index in np.flatnonzero(usable):
        local = _tile_distortion_bound(
            rgb,
            float(window_left[index]),
            float(window_right[index]),
            float(window_top[index]),
            float(window_bottom[index]),
        )
        if local is not None:
            rgb_bounds[index] = local
    left = window_left - rgb_bounds
    right = window_right + rgb_bounds
    top = window_top - rgb_bounds
    bottom = window_bottom + rgb_bounds

    directions = np.empty((pixels.shape[0], 4, 3), dtype=np.float64)
    corners_u = np.stack((left, right, left, right), axis=1)
    corners_v = np.stack((top, top, bottom, bottom), axis=1)
    directions[..., 0] = (corners_u - rgb.cx) / rgb.fx
    directions[..., 1] = (corners_v - rgb.cy) / rgb.fy
    directions[..., 2] = 1.0
    directions_ir = directions @ rotation
    forward = directions_ir[..., 2]
    if np.any(forward[usable] <= 0.0):
        return None
    translation_ir = rotation.T @ translation
    translation_z = float(translation_ir[2])
    near_scale = (z_min + translation_z) / forward
    far_scale = (z_max + translation_z) / forward
    if (
        np.any(near_scale[usable] <= 0.0)
        or np.any(far_scale[usable] <= 0.0)
        or not np.all(np.isfinite(near_scale[usable]))
        or not np.all(np.isfinite(far_scale[usable]))
    ):
        return None
    near_points = directions_ir * near_scale[..., None] - translation_ir[None, None, :]
    far_points = directions_ir * far_scale[..., None] - translation_ir[None, None, :]
    near_pixels = _pinhole_project(near_points, ir)
    far_pixels = _pinhole_project(far_points, ir)
    if not (np.all(np.isfinite(near_pixels[usable])) and np.all(np.isfinite(far_pixels[usable]))):
        return None
    combined = np.concatenate((near_pixels, far_pixels), axis=1)
    pinhole_u0 = np.min(combined[..., 0], axis=1)
    pinhole_u1 = np.max(combined[..., 0], axis=1)
    pinhole_v0 = np.min(combined[..., 1], axis=1)
    pinhole_v1 = np.max(combined[..., 1], axis=1)
    ir_bounds = np.full(pixels.shape[0], ir_bound, dtype=np.float64)
    for index in np.flatnonzero(usable):
        local = _tile_distortion_bound(
            ir,
            float(pinhole_u0[index]),
            float(pinhole_u1[index]),
            float(pinhole_v0[index]),
            float(pinhole_v1[index]),
        )
        if local is not None:
            ir_bounds[index] = local
    u0 = np.floor(pinhole_u0 - ir_bounds).astype(np.int64)
    u1 = np.ceil(pinhole_u1 + ir_bounds).astype(np.int64)
    v0 = np.floor(pinhole_v0 - ir_bounds).astype(np.int64)
    v1 = np.ceil(pinhole_v1 + ir_bounds).astype(np.int64)
    u0 = np.maximum(0, u0)
    v0 = np.maximum(0, v0)
    u1 = np.minimum(ir.width - 1, u1)
    v1 = np.minimum(ir.height - 1, v1)
    usable &= (u0 <= u1) & (v0 <= v1)
    if not np.any(usable):
        return np.zeros((ir.height, ir.width), dtype=np.bool_)

    mask = np.zeros((ir.height, ir.width), dtype=np.bool_)
    for index in np.flatnonzero(usable):
        mask[v0[index] : v1[index] + 1, u0[index] : u1[index] + 1] = True

    if float(np.count_nonzero(mask)) > _MAX_ROI_FRACTION * mask.size:
        return None
    return mask


def _lift_keypoints_core(
    pixels: NDArray[np.float64],
    depth_ir: NDArray[np.float32],
    calib: Calibration,
    window: int,
    z_min: float,
    roi_mask: NDArray[np.bool_] | None,
) -> NDArray[np.float64]:
    """Shared lifting implementation; ``roi_mask=None`` registers every pixel."""
    flat = pixels.reshape(-1, 2)
    lifted = np.full((flat.shape[0], 3), np.nan, dtype=np.float64)

    valid_depth = np.isfinite(depth_ir) & (depth_ir >= z_min)
    if roi_mask is not None:
        valid_depth &= roi_mask
    rows, cols = np.nonzero(valid_depth)
    if rows.size == 0:
        return lifted.reshape(pixels.shape[:-1] + (3,))

    depth_valid = depth_ir[rows, cols].astype(np.float64)
    rays = _ir_rays(calib.ir)
    points_ir = np.empty((rows.size, 3), dtype=np.float64)
    points_ir[:, 0] = rays[rows, cols, 0] * depth_valid
    points_ir[:, 1] = rays[rows, cols, 1] * depth_valid
    points_ir[:, 2] = depth_valid
    points_rgb = points_ir @ calib.T_ir_rgb.R.T + calib.T_ir_rgb.t_m

    depth_rgb = points_rgb[:, 2]
    positive = depth_rgb > 0.0
    points_rgb = points_rgb[positive]
    depth_rgb = depth_rgb[positive]
    registered = project(points_rgb, calib.rgb)

    width = calib.rgb.width
    height = calib.rgb.height
    u = registered[:, 0]
    v = registered[:, 1]
    inside = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    u = u[inside]
    v = v[inside]
    depth_rgb = depth_rgb[inside]
    if u.size == 0:
        return lifted.reshape(pixels.shape[:-1] + (3,))

    # Mark the cells each keypoint can see and drop every registered sample that
    # cannot contribute to any keypoint. The remaining samples are bucketed by
    # RGB pixel cell, so each keypoint scans only a (2 * window + 1) square.
    bounds = np.full((flat.shape[0], 4), -1, dtype=np.int64)
    allowed = np.zeros((height, width), dtype=np.bool_)
    for index in range(flat.shape[0]):
        key_u, key_v = flat[index]
        if not (np.isfinite(key_u) and np.isfinite(key_v)):
            continue
        u0 = max(0, int(np.floor(key_u)) - window)
        u1 = min(width - 1, int(np.floor(key_u)) + window)
        v0 = max(0, int(np.floor(key_v)) - window)
        v1 = min(height - 1, int(np.floor(key_v)) + window)
        if u0 > u1 or v0 > v1:
            continue
        bounds[index] = (u0, u1, v0, v1)
        allowed[v0 : v1 + 1, u0 : u1 + 1] = True

    cell_column = np.floor(u).astype(np.int64)
    cell_row = np.floor(v).astype(np.int64)
    keep = allowed[cell_row, cell_column]
    u = u[keep]
    v = v[keep]
    depth_rgb = depth_rgb[keep]
    if u.size == 0:
        return lifted.reshape(pixels.shape[:-1] + (3,))

    cell = cell_row[keep] * width + cell_column[keep]
    order = np.argsort(cell, kind="stable")
    cell_sorted = cell[order]
    u_sorted = u[order]
    v_sorted = v[order]
    depth_sorted = depth_rgb[order]
    counts = np.bincount(cell_sorted, minlength=height * width)
    starts = np.empty(counts.size + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, out=starts[1:])

    median_depths = np.full(flat.shape[0], np.nan, dtype=np.float64)
    for index in range(flat.shape[0]):
        u0, u1, v0, v1 = (int(value) for value in bounds[index])
        if u0 < 0:
            continue

        near_u: list[NDArray[np.float64]] = []
        near_v: list[NDArray[np.float64]] = []
        near_depth: list[NDArray[np.float64]] = []
        for row in range(v0, v1 + 1):
            row_start = row * width
            first = starts[row_start + u0]
            last = starts[row_start + u1 + 1]
            if last > first:
                near_u.append(u_sorted[first:last])
                near_v.append(v_sorted[first:last])
                near_depth.append(depth_sorted[first:last])
        if not near_u:
            continue

        candidate_u = np.concatenate(near_u)
        candidate_v = np.concatenate(near_v)
        candidate_depth = np.concatenate(near_depth)
        in_window = (np.abs(candidate_u - flat[index, 0]) <= window) & (
            np.abs(candidate_v - flat[index, 1]) <= window
        )
        if np.any(in_window):
            median_depths[index] = float(np.median(candidate_depth[in_window]))

    found = np.isfinite(median_depths)
    if np.any(found):
        lifted[found] = deproject(flat[found], median_depths[found], calib.rgb)

    return lifted.reshape(pixels.shape[:-1] + (3,))


def lift_keypoints(
    uv_rgb: ArrayLike,
    depth_raw: NDArray[np.uint16],
    calib: Calibration,
    window: int = 5,
    z_min: float = _DEFAULT_Z_MIN_M,
) -> NDArray[np.float64]:
    """Lift RGB keypoints to 3D camera-frame points using registered depth.

    Depth pixels live in the IR frame, so every valid depth pixel is deprojected,
    transformed with ``T_ir_rgb`` and projected into the RGB image. For each
    keypoint, the median depth of the registered samples inside the
    ``window``-pixel square around it is used, then the keypoint is deprojected
    with the RGB intrinsics. A keypoint without registered depth yields ``NaN``.

    Depths below ``z_min`` metres are treated as invalid; the default ``0.2``
    matches the F200 minimum range and bounds the translation parallax used to
    restrict registration to a conservative IR region of interest. The square
    window includes a pixel whose centre is at most ``window`` pixels away in
    both axes. Undistortion of the fixed IR pixel grid is cached per calibration,
    so the per-frame cost is the projection of the valid depth pixels inside the
    ROI. Full registration is used when the calibration geometry makes the ROI
    uncertain or when the union covers more than half of the image.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    if not np.isfinite(z_min) or z_min <= 0.0:
        raise ValueError(f"z_min must be a finite positive depth, got {z_min}")
    depth_image = np.asarray(depth_raw)
    if depth_image.ndim != 2:
        raise ValueError(f"expected a 2D depth image, got shape {depth_image.shape}")
    expected = (calib.ir.height, calib.ir.width)
    if depth_image.shape != expected:
        raise ValueError(
            f"depth image shape {depth_image.shape} does not match IR intrinsics {expected}"
        )

    pixels = np.asarray(uv_rgb, dtype=np.float64)
    if pixels.shape[-1:] != (2,):
        raise ValueError(f"expected keypoints with shape (..., 2), got {pixels.shape}")
    flat = pixels.reshape(-1, 2)
    depth_ir = depth_to_m(depth_image, calib.depth)
    valid_depth = np.isfinite(depth_ir) & (depth_ir >= z_min)
    if not np.any(valid_depth):
        return _lift_keypoints_core(pixels, depth_ir, calib, window, z_min, None)
    # Every sample above the configured floor is at least this far away, so
    # the observed depth range is a tighter (still conservative) parallax bound.
    depth_values = depth_ir[valid_depth]
    low_depth = max(z_min, float(np.min(depth_values)))
    high_depth = float(np.max(depth_values))
    roi_mask = _keypoint_roi_mask(flat, calib, window, low_depth, high_depth)
    if roi_mask is not None:
        # Every contributing pixel is inside the mask, so the depth range of the
        # valid pixels inside it still bounds all contributing depths. Iterating
        # this contraction tightens the ROI without losing any sample.
        for _ in range(3):
            in_roi = valid_depth & roi_mask
            if not np.any(in_roi):
                break
            values = depth_ir[in_roi]
            new_low = max(z_min, float(np.min(values)))
            new_high = float(np.max(values))
            if (
                new_low <= low_depth + _ROI_RANGE_TOLERANCE_M
                and new_high >= high_depth - _ROI_RANGE_TOLERANCE_M
            ):
                break
            tightened = _keypoint_roi_mask(flat, calib, window, new_low, new_high)
            if tightened is None:
                roi_mask = None
                break
            if np.array_equal(tightened, roi_mask):
                break
            roi_mask = tightened
            low_depth, high_depth = new_low, new_high
    return _lift_keypoints_core(pixels, depth_ir, calib, window, z_min, roi_mask)
