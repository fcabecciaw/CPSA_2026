"""Depth localizer and detection gate for the hand demos (F-H).

The aligned IR/depth frame gives the table and the hands without a network. The
empty table is calibrated once (:func:`calibrate_table`): a per-pixel median
background and a RANSAC table plane define a *zone* of table pixels, and the
per-pixel height above that background turns the frame into a foreground band
between :attr:`LocatorConfig.min_height_m` and
:attr:`LocatorConfig.max_height_m`. Morphology and connected components give
one blob per hand plus its arm; the blob end that touches the zone border is the
*entry* (the wrist side), the farthest point is the *tip*, the distance
transform peak within :attr:`LocatorConfig.hand_length_m` of the tip is the
palm centre (:func:`locate_hands`).

Its runtime role is the *gate* of the full-frame hand model
(:func:`gate_detections`): a detection box survives only if enough of it is
foreground above the table and it is not too large to be a hand, which is what
removes the boxes the model finds on the user's bust, arm and background.

Everything is computed on a 2x-downsampled mask (320x240) and returned in
full-frame coordinates, which keeps :func:`locate_hands` and the gate well under
10 ms on the host CPU. OpenCV is used for median filtering, morphology, the
distance transform and connected components; the layer is ``adapters`` so that
is allowed.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - depends on the installed extra
    import cv2
except ImportError:  # pragma: no cover - depends on the installed extra
    cv2 = None  # type: ignore[assignment]

from cpsa.vision.calibration import Intrinsics
from cpsa.vision.hand_crop import angle_from_up

__all__ = [
    "HAND_FILL",
    "HandRoi",
    "LocatorConfig",
    "TableModel",
    "calibrate_table",
    "foreground_mask",
    "gate_detections",
    "locate_hands",
]

HAND_FILL = 0.6
"""Fraction of the crop side the hand extent fills (F-G: ``side = extent / 0.6``)."""

_RANSAC_ITERATIONS = 240
"""Plane hypotheses tried on the calibration frames."""

_RANSAC_SEED = 0
"""Fixed seed so a calibration is reproducible."""

_PLANE_THRESHOLD_M = 0.005
"""RANSAC inlier distance of the table plane."""

_ZONE_TOLERANCE_M = 0.02
"""Depth band around the table plane that counts as table surface."""

_ZONE_OPEN_PX = 3
"""Opening kernel of the raw zone candidate, full resolution."""

_ZONE_CLOSE_PX = 15
"""Closing kernel that seals holes in the calibrated zone."""

_CALIBRATION_STRIDE = 2
"""Pixel stride of the RANSAC sample, full resolution."""

_PIXEL_AREA_FLOOR = 100
"""Half-resolution pixels below which a component cannot reach ``min_area_m2``."""

_MIN_SIDE_PX = 16.0
"""Crops thinner than this are degenerated blobs, not hands."""

_TIP_DEPTH_RADIUS_PX = 3
"""Half-resolution radius of the depth window around the tip."""

_PALM_DEPTH_RADIUS_PX = 3
"""Half-resolution radius of the depth window around the palm."""

_CONTACT_DILATION_PX = 5
"""Dilation that lets an opened blob reach the zone border ring again."""


@dataclass(frozen=True)
class TableModel:
    """The empty-table calibration the localizer runs against.

    ``background_m`` is the ``(H, W)`` per-pixel median depth in metres,
    ``NaN`` where no calibration frame measured the pixel. ``plane`` is
    ``(n_x, n_y, n_z, d)`` in camera metres with a unit normal pointing toward
    the camera and ``n . X + d = 0`` on the table, so ``n . X + d`` is the
    height of ``X`` above the table. ``zone`` is the ``(H, W)`` boolean table
    region where hands are searched.
    """

    background_m: NDArray[np.float32]
    plane: tuple[float, float, float, float]
    zone: NDArray[np.bool_]


@dataclass(frozen=True)
class HandRoi:
    """One hand found by the localizer, as an F-G crop plus its geometry.

    ``cx, cy`` are the palm centre in full-frame pixels, ``side`` the crop side
    in full-frame pixels, ``angle`` the rotation onto image up in radians.
    ``entry`` is where the arm enters the zone (or the near image edge when the
    blob does not touch the border) and ``tip`` the farthest point from it, both
    in full-frame pixels. ``palm_depth_m`` is the depth at the palm and
    ``area_m2`` the real area of the blob.
    """

    cx: float
    cy: float
    side: float
    angle: float
    palm_depth_m: float
    area_m2: float
    tip: tuple[float, float]
    entry: tuple[float, float]


@dataclass(frozen=True)
class LocatorConfig:
    """Tuning of the depth localizer.

    The defaults are the F-H ones plus the gate ones. ``min_height_m`` is 5 cm
    rather than 2 cm because the 2026-11-12 empty capture showed that the F200
    depth stream needs about 0.1 s to settle (its first four frames carry
    2.4 cm of noise against 0.6 cm afterwards) and a 2 cm band turned that
    transient into four phantom ROIs. That is still far below the 6-20 cm a
    hand hovers above the table; a hand lying flat on the table is not found by
    this localizer.

    ``min_overlap``, ``max_hand_size_m`` and ``min_score`` configure
    :func:`gate_detections`: how much of a detection box must be foreground,
    how large the box may be in metres, and the lowest detector score to keep.
    """

    min_height_m: float = 0.05
    max_height_m: float = 0.35
    min_area_m2: float = 0.002
    hand_length_m: float = 0.20
    max_hands: int = 2
    downsample: int = 2
    median_px: int = 5
    open_px: int = 5
    close_px: int = 7
    min_overlap: float = 0.15
    max_hand_size_m: float = 0.30
    min_score: float = 0.0


def calibrate_table(
    depth_frames: Sequence[NDArray[np.uint16]],
    *,
    scale_m: float,
    intrinsics: Intrinsics,
) -> TableModel:
    """Build the empty-table model from a run of raw Z16 depth frames.

    The per-pixel median of the frames is the background; a RANSAC plane over
    the background pixels that have a measurement gives the table plane, and
    the zone is the largest connected region of the background that lies within
    :data:`_ZONE_TOLERANCE_M` of that plane, with holes sealed so a hand over an
    unmeasured pixel is still searched.

    Args:
        depth_frames: ``(H, W)`` raw Z16 frames of the empty table; zero means
            "no measurement".
        scale_m: Raw Z16 units to metres.
        intrinsics: IR intrinsics of the frames.

    Returns:
        The table model, ready for :func:`locate_hands` and
        :func:`gate_detections`.

    Raises:
        ValueError: If ``depth_frames`` is empty, the frames are not 2-D and of
            equal shape, ``scale_m`` is not positive, or the focal lengths are
            not positive.
    """

    _require_cv2()
    frames = [np.asarray(frame) for frame in depth_frames]
    if not frames:
        raise ValueError("calibrate_table needs at least one depth frame")
    shape = frames[0].shape
    if any(frame.ndim != 2 or frame.shape != shape for frame in frames):
        raise ValueError("depth frames must all be 2-D arrays of one shape")
    if scale_m <= 0.0:
        raise ValueError(f"scale_m must be positive, got {scale_m}")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError(
            f"intrinsics need positive focal lengths, got fx={intrinsics.fx}, fy={intrinsics.fy}"
        )

    stack = np.stack([frame.astype(np.float32) * float(scale_m) for frame in frames])
    stack[stack <= 0.0] = np.nan
    with warnings.catch_warnings():
        # Pixels never measured in any frame stay NaN; that is the contract.
        warnings.simplefilter("ignore", RuntimeWarning)
        background = np.nanmedian(stack, axis=0).astype(np.float32)

    plane = _ransac_plane(background, intrinsics)
    zone = _table_zone(background, plane, intrinsics)
    return TableModel(background_m=background, plane=plane, zone=zone)


def locate_hands(
    depth: NDArray[np.uint16],
    table: TableModel,
    *,
    scale_m: float,
    intrinsics: Intrinsics,
    config: LocatorConfig,
) -> list[HandRoi]:
    """Find up to ``config.max_hands`` hands, largest blob first.

    Args:
        depth: ``(H, W)`` raw Z16 depth of the frame, pixel-aligned with the
            calibration.
        table: The empty-table model from :func:`calibrate_table`.
        scale_m: Raw Z16 units to metres.
        intrinsics: IR intrinsics of the frame.
        config: Tuning and the hand cap.

    Returns:
        One :class:`HandRoi` per accepted blob, sorted by real area, largest
        first. An empty table gives ``[]``.

    Raises:
        ValueError: If the frame or the model do not match, or a parameter is
            invalid.
    """

    _require_cv2()
    raw = _validate_frame(depth, table, scale_m=scale_m, intrinsics=intrinsics, config=config)
    return _locate(raw, table, scale_m=scale_m, intrinsics=intrinsics, config=config)


def foreground_mask(
    depth: NDArray[np.uint16],
    table: TableModel,
    *,
    scale_m: float,
    intrinsics: Intrinsics,
    config: LocatorConfig,
) -> NDArray[np.bool_]:
    """Return the full-resolution foreground mask :func:`locate_hands` works on.

    This is the debug and preview view of the localizer: valid depth inside the
    calibrated zone whose height above the background is inside the configured
    band, after median filtering and the configured morphology, upsampled to
    full resolution.

    Args:
        depth: ``(H, W)`` raw Z16 depth of the frame.
        table: The empty-table model from :func:`calibrate_table`.
        scale_m: Raw Z16 units to metres.
        intrinsics: IR intrinsics of the frame.
        config: Tuning and the hand cap.

    Returns:
        ``(H, W)`` boolean mask.

    Raises:
        ValueError: If the frame or the model do not match, or a parameter is
            invalid.
    """

    _require_cv2()
    raw = _validate_frame(depth, table, scale_m=scale_m, intrinsics=intrinsics, config=config)
    half_mask = _half_resolution_mask(
        raw, table, scale_m=scale_m, intrinsics=intrinsics, config=config
    )
    if int(config.downsample) == 1:
        return half_mask
    mask = half_mask.astype(np.uint8) * np.uint8(255)
    full = cv2.resize(mask, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(full > 0, dtype=np.bool_)


def gate_detections(
    boxes_xyxy: NDArray[np.float64],
    scores: NDArray[np.float64],
    depth: NDArray[np.uint16],
    table: TableModel,
    *,
    scale_m: float,
    intrinsics: Intrinsics,
    config: LocatorConfig,
) -> NDArray[np.bool_]:
    """Keep the detections that sit on a foreground blob above the table.

    This is the runtime role of the depth localizer: the full-frame hand model
    proposes boxes, and the depth frame decides which ones can be a hand. A box
    is kept when all of these hold:

    * at least ``config.min_overlap`` of its area is foreground, that is inside
      the calibrated table zone and above the background band;
    * its real size ``max(w, h) * z / fx`` is at most
      ``config.max_hand_size_m``, with ``z`` the median valid depth inside the
      box (a bust, a sleeve or a whole arm is much larger than a hand);
    * its score is at least ``config.min_score``.

    A box with no depth measurement inside cannot be sized and is dropped, as
    is a box outside the image.

    Args:
        boxes_xyxy: ``(N, 4)`` detection boxes ``x1, y1, x2, y2`` in full-frame
            pixels.
        scores: ``(N,)`` detection scores.
        depth: ``(H, W)`` raw Z16 depth of the frame, pixel-aligned with the
            calibration.
        table: The empty-table model from :func:`calibrate_table`.
        scale_m: Raw Z16 units to metres.
        intrinsics: IR intrinsics of the frame.
        config: Tuning, including the gate thresholds.

    Returns:
        ``(N,)`` boolean keep mask, in box order.

    Raises:
        ValueError: If the boxes or scores have the wrong shape, the frame or
            the model do not match, or a threshold is invalid.
    """

    _require_cv2()
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros(0, dtype=np.bool_)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be (N, 4) xyxy, got shape {boxes.shape}")
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.shape != (boxes.shape[0],):
        raise ValueError(f"scores must be ({boxes.shape[0]},), got shape {scores_array.shape}")
    _validate_thresholds(config)
    raw = _validate_frame(depth, table, scale_m=scale_m, intrinsics=intrinsics, config=config)
    mask = foreground_mask(raw, table, scale_m=scale_m, intrinsics=intrinsics, config=config)
    depth_m = raw.astype(np.float32) * np.float32(scale_m)
    return _gate(boxes, scores_array, mask, depth_m, intrinsics=intrinsics, config=config)


def _gate(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float32],
    *,
    intrinsics: Intrinsics,
    config: LocatorConfig,
) -> NDArray[np.bool_]:
    """Score the boxes against the foreground mask and the box depth."""

    rows, columns = mask.shape
    keep = np.zeros(boxes.shape[0], dtype=np.bool_)
    for index, box in enumerate(boxes):
        if not np.isfinite(box).all() or scores[index] < config.min_score:
            continue
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        if width <= 0.0 or height <= 0.0:
            continue
        left = int(np.clip(np.floor(box[0]), 0, columns))
        right = int(np.clip(np.ceil(box[2]), 0, columns))
        top = int(np.clip(np.floor(box[1]), 0, rows))
        bottom = int(np.clip(np.ceil(box[3]), 0, rows))
        if right <= left or bottom <= top:
            continue
        coverage = float(mask[top:bottom, left:right].sum()) / (width * height)
        window = depth_m[top:bottom, left:right]
        measured = window[window > 0.0]
        if measured.size == 0:
            continue
        size_m = max(width, height) * float(np.median(measured)) / intrinsics.fx
        keep[index] = coverage >= config.min_overlap and size_m <= config.max_hand_size_m
    return keep


def _validate_thresholds(config: LocatorConfig) -> None:
    """Check the gate thresholds of one configuration."""

    if not 0.0 <= config.min_overlap <= 1.0:
        raise ValueError(f"min_overlap must be in [0, 1], got {config.min_overlap}")
    if config.max_hand_size_m <= 0.0:
        raise ValueError(f"max_hand_size_m must be positive, got {config.max_hand_size_m}")


def _locate(
    raw: NDArray[np.uint16],
    table: TableModel,
    *,
    scale_m: float,
    intrinsics: Intrinsics,
    config: LocatorConfig,
) -> list[HandRoi]:
    """Run the components and the per-hand geometry on an already validated frame."""

    factor = int(config.downsample)
    half_mask = _half_resolution_mask(
        raw, table, scale_m=scale_m, intrinsics=intrinsics, config=config
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        half_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    if count <= 1:
        return []
    labels_array = np.asarray(labels, dtype=np.int32)

    half_depth = _half_depth(raw, scale_m=scale_m, config=config)
    zone_half = np.ascontiguousarray(table.zone[::factor, ::factor])
    border = _zone_border(zone_half)
    intrinsics_half = _scaled_intrinsics(intrinsics, factor)

    found: list[tuple[float, HandRoi]] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < _PIXEL_AREA_FLOOR:
            continue
        roi = _component_roi(
            index,
            labels_array,
            half_depth,
            border,
            factor=factor,
            intrinsics_half=intrinsics_half,
            config=config,
        )
        if roi is not None:
            found.append((roi.area_m2, roi))

    found.sort(key=lambda item: item[0], reverse=True)
    return [roi for _, roi in found[: max(0, int(config.max_hands))]]


def _component_roi(
    index: int,
    labels: NDArray[np.int32],
    half_depth: NDArray[np.float32],
    border: NDArray[np.bool_],
    *,
    factor: int,
    intrinsics_half: Intrinsics,
    config: LocatorConfig,
) -> HandRoi | None:
    """Turn one connected component into a hand ROI, or ``None`` if it is no hand."""

    component = labels == index
    ys, xs = np.nonzero(component)
    if xs.size == 0:
        return None
    z_median = float(np.median(half_depth[ys, xs]))
    if not math.isfinite(z_median) or z_median <= 0.0:
        return None
    area_m2 = float(xs.size) * (z_median / intrinsics_half.fx) ** 2
    if area_m2 < config.min_area_m2:
        return None

    entry_x, entry_y = _entry_point(component, border, xs, ys)
    distances = np.hypot(xs - entry_x, ys - entry_y)
    tip_index = int(np.argmax(distances))
    tip_x, tip_y = float(xs[tip_index]), float(ys[tip_index])

    up = np.array([tip_x - entry_x, tip_y - entry_y])
    length = float(np.hypot(up[0], up[1]))
    if length <= 0.0:
        return None
    up /= length

    z_hand = _local_depth(half_depth, ys, xs, tip_x, tip_y, _TIP_DEPTH_RADIUS_PX)
    if z_hand is None or z_hand <= 0.0:
        z_hand = z_median
    hand_length_px = float(config.hand_length_m) * intrinsics_half.fx / z_hand
    hand_selector = distances <= hand_length_px
    if not hand_selector.any():
        hand_selector = np.ones_like(distances, dtype=bool)

    palm_x, palm_y = _palm_point(component, xs, ys, hand_selector, tip_x, tip_y)
    palm_depth = _local_depth(half_depth, ys, xs, palm_x, palm_y, _PALM_DEPTH_RADIUS_PX)
    if palm_depth is None:
        palm_depth = z_hand

    extent = _aligned_extent(xs, ys, up, hand_selector) * float(factor)
    side = extent / HAND_FILL
    if side < _MIN_SIDE_PX:
        return None

    return HandRoi(
        cx=palm_x * factor,
        cy=palm_y * factor,
        side=float(side),
        angle=angle_from_up((tip_x - entry_x, tip_y - entry_y)),
        palm_depth_m=float(palm_depth),
        area_m2=area_m2,
        tip=(tip_x * factor, tip_y * factor),
        entry=(entry_x * factor, entry_y * factor),
    )


def _entry_point(
    component: NDArray[np.bool_],
    border: NDArray[np.bool_],
    xs: NDArray[np.intp],
    ys: NDArray[np.intp],
) -> tuple[float, float]:
    """Return the wrist-side entry of one component, in half-resolution pixels.

    The blob touches the zone border where the arm crosses the table edge: the
    centroid of that contact is the entry. A blob fully inside the zone (a hand
    held over the table without touching an edge) falls back to its point
    nearest the near image edge, that is the largest row.
    """

    reach = cv2.dilate(
        component.astype(np.uint8),
        np.ones((_CONTACT_DILATION_PX, _CONTACT_DILATION_PX), np.uint8),
        iterations=1,
    )
    contact = (reach > 0) & border
    if contact.any():
        contact_ys, contact_xs = np.nonzero(contact)
        return float(contact_xs.mean()), float(contact_ys.mean())
    bottom = int(np.max(ys))
    selected = xs[ys == bottom]
    return float(selected.mean()), float(bottom)


def _palm_point(
    component: NDArray[np.bool_],
    xs: NDArray[np.intp],
    ys: NDArray[np.intp],
    hand_selector: NDArray[np.bool_],
    tip_x: float,
    tip_y: float,
) -> tuple[float, float]:
    """Return the distance-transform peak of the hand part, farthest from the tip on ties."""

    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    values = distance[ys[hand_selector], xs[hand_selector]]
    best = float(np.max(values))
    candidates = np.nonzero(hand_selector)[0][values >= best - 1e-6]
    # The palm is the widest part; the tip-side candidate wins a tie so a
    # forearm and a palm of equal width do not flip the ROI back to the arm.
    if candidates.size > 1:
        tip_distance = np.hypot(xs[candidates] - tip_x, ys[candidates] - tip_y)
        chosen = int(candidates[int(np.argmin(tip_distance))])
    else:
        chosen = int(candidates[0])
    return float(xs[chosen]), float(ys[chosen])


def _aligned_extent(
    xs: NDArray[np.intp],
    ys: NDArray[np.intp],
    up: NDArray[np.float64],
    selector: NDArray[np.bool_],
) -> float:
    """Larger side of the ``up``-aligned bbox of the selected half-resolution pixels."""

    along = xs[selector] * up[0] + ys[selector] * up[1]
    across = -xs[selector] * up[1] + ys[selector] * up[0]
    largest = max(float(along.max() - along.min()), float(across.max() - across.min()))
    return largest + 1.0


def _local_depth(
    half_depth: NDArray[np.float32],
    ys: NDArray[np.intp],
    xs: NDArray[np.intp],
    x: float,
    y: float,
    radius: int,
) -> float | None:
    """Median depth of the component pixels within ``radius`` of ``(x, y)``."""

    near = (np.abs(xs - x) <= radius) & (np.abs(ys - y) <= radius)
    if not near.any():
        return None
    values = half_depth[ys[near], xs[near]]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _half_resolution_mask(
    raw: NDArray[np.uint16],
    table: TableModel,
    *,
    scale_m: float,
    intrinsics: Intrinsics,
    config: LocatorConfig,
) -> NDArray[np.bool_]:
    """Build the half-resolution foreground mask."""

    factor = int(config.downsample)
    depth_m = _half_depth(raw, scale_m=scale_m, config=config)
    background = np.ascontiguousarray(
        np.asarray(table.background_m, dtype=np.float32)[::factor, ::factor]
    )
    intrinsics_half = _scaled_intrinsics(intrinsics, factor)
    plane_depth = _plane_depth(table.plane, depth_m.shape, intrinsics_half)
    height = np.where(np.isfinite(background), background - depth_m, plane_depth - depth_m)
    band = (
        (depth_m > 0.0)
        & np.isfinite(height)
        & (height >= np.float32(config.min_height_m))
        & (height <= np.float32(config.max_height_m))
    )
    zone = np.ascontiguousarray(table.zone[::factor, ::factor])
    mask = band & zone
    binary = mask.astype(np.uint8)
    if int(config.close_px) > 0:
        binary = np.asarray(
            cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(config.close_px),) * 2),
            ),
            dtype=np.uint8,
        )
    if int(config.open_px) > 0:
        binary = np.asarray(
            cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(config.open_px),) * 2),
            ),
            dtype=np.uint8,
        )
    return np.asarray(binary > 0, dtype=np.bool_)


def _zone_border(zone: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return the ring of zone pixels adjacent to the outside.

    The image edge counts as outside: a table region that reaches the frame
    border has its border there too, so an arm leaving the frame is an entry.
    """

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(
        zone.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.asarray(zone & (eroded == 0), dtype=np.bool_)


def _half_depth(
    raw: NDArray[np.uint16], *, scale_m: float, config: LocatorConfig
) -> NDArray[np.float32]:
    """Median-filtered half-resolution depth in metres (``0`` stays invalid)."""

    factor = int(config.downsample)
    filtered = np.ascontiguousarray(raw[::factor, ::factor])
    if int(config.median_px) >= 3:
        filtered = np.asarray(cv2.medianBlur(filtered, int(config.median_px)), dtype=np.uint16)
    return filtered.astype(np.float32) * np.float32(scale_m)


def _scaled_intrinsics(intrinsics: Intrinsics, factor: int) -> Intrinsics:
    """Intrinsics of the downsampled grid."""

    width = max(1, int(intrinsics.width) // factor)
    height = max(1, int(intrinsics.height) // factor)
    return Intrinsics(
        fx=float(intrinsics.fx) / factor,
        fy=float(intrinsics.fy) / factor,
        cx=float(intrinsics.cx) / factor,
        cy=float(intrinsics.cy) / factor,
        width=width,
        height=height,
        dist=intrinsics.dist,
    )


def _validate_frame(
    depth: NDArray[np.uint16],
    table: TableModel,
    *,
    scale_m: float,
    intrinsics: Intrinsics,
    config: LocatorConfig,
) -> NDArray[np.uint16]:
    """Check one frame against its table model and return it as an array."""

    raw = np.asarray(depth)
    if raw.ndim != 2:
        raise ValueError(f"depth must be (H, W), got shape {raw.shape}")
    if table.zone.shape != raw.shape or table.background_m.shape != raw.shape:
        raise ValueError(
            "depth frame and table model disagree: "
            f"frame {raw.shape}, background {table.background_m.shape}, zone {table.zone.shape}"
        )
    if scale_m <= 0.0:
        raise ValueError(f"scale_m must be positive, got {scale_m}")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError(
            f"intrinsics need positive focal lengths, got fx={intrinsics.fx}, fy={intrinsics.fy}"
        )
    if int(config.downsample) < 1:
        raise ValueError(f"downsample must be at least 1, got {config.downsample}")
    if config.max_height_m <= config.min_height_m:
        raise ValueError(
            "max_height_m must exceed min_height_m, got "
            f"{config.min_height_m} and {config.max_height_m}"
        )
    return raw


def _ransac_plane(
    background: NDArray[np.float32], intrinsics: Intrinsics
) -> tuple[float, float, float, float]:
    """Fit the table plane to the background pixels and return ``(nx, ny, nz, d)``.

    RANSAC with a fixed seed gives a robust hypothesis and a total-least-squares
    refit on its inliers gives the accurate plane. The normal points toward the
    camera, so ``n . X + d`` is positive above the table and ``d`` is the
    camera-to-plane distance.
    """

    stride = _CALIBRATION_STRIDE
    sampled = background[::stride, ::stride]
    rows, columns = np.nonzero(np.isfinite(sampled))
    if rows.size < 3:
        raise ValueError("the calibration frames have too few measured pixels to fit a plane")
    z = sampled[rows, columns].astype(np.float64)
    points = np.stack(
        [
            (columns.astype(np.float64) * stride - intrinsics.cx) / intrinsics.fx * z,
            (rows.astype(np.float64) * stride - intrinsics.cy) / intrinsics.fy * z,
            z,
        ],
        axis=1,
    )

    generator = np.random.default_rng(_RANSAC_SEED)
    best_inliers = np.zeros(points.shape[0], dtype=bool)
    for _ in range(_RANSAC_ITERATIONS):
        picks = generator.choice(points.shape[0], size=3, replace=False)
        edges = points[picks]
        normal = np.cross(edges[1] - edges[0], edges[2] - edges[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            continue
        normal /= norm
        offset = -float(normal @ edges[0])
        inliers = np.abs(points @ normal + offset) <= _PLANE_THRESHOLD_M
        if int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers

    if int(best_inliers.sum()) < 3:
        raise ValueError("no table plane found in the calibration frames")
    centroid = points[best_inliers].mean(axis=0)
    _, _, right = np.linalg.svd(points[best_inliers] - centroid, full_matrices=False)
    normal = right[-1] / float(np.linalg.norm(right[-1]))
    if normal[2] > 0.0:
        normal = -normal
    offset = -float(normal @ centroid)
    return (float(normal[0]), float(normal[1]), float(normal[2]), offset)


def _table_zone(
    background: NDArray[np.float32],
    plane: tuple[float, float, float, float],
    intrinsics: Intrinsics,
) -> NDArray[np.bool_]:
    """Return the table region: background pixels close to the table plane."""

    plane_depth = _plane_depth(plane, background.shape, intrinsics)
    distance = np.abs(background - plane_depth)
    candidate = (
        np.isfinite(background)
        & np.isfinite(distance)
        & (distance <= _ZONE_TOLERANCE_M)
        & (plane_depth > 0.0)
    ).astype(np.uint8)
    candidate = np.asarray(
        cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ZONE_OPEN_PX, _ZONE_OPEN_PX)),
        ),
        dtype=np.uint8,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8, ltype=cv2.CV_32S
    )
    if count <= 1:
        raise ValueError("the calibration frames show no table surface")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    zone = (labels == largest).astype(np.uint8)
    zone = np.asarray(
        cv2.morphologyEx(
            zone,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ZONE_CLOSE_PX, _ZONE_CLOSE_PX)),
            iterations=1,
        ),
        dtype=np.uint8,
    )
    rows, columns = zone.shape
    flood = zone.copy()
    mask = np.zeros((rows + 2, columns + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)
    # Everything the flood could not reach is a hole inside the table.
    zone = zone | (1 - flood)
    return np.asarray(zone > 0, dtype=np.bool_)


def _plane_depth(
    plane: tuple[float, float, float, float],
    shape: tuple[int, int],
    intrinsics: Intrinsics,
) -> NDArray[np.float32]:
    """Depth of the table plane at every pixel, ``NaN`` where it is behind."""

    n_x, n_y, n_z, d = (float(value) for value in plane)
    rows, columns = shape
    columns_m = (np.arange(columns, dtype=np.float64) - intrinsics.cx) / intrinsics.fx
    rows_m = (np.arange(rows, dtype=np.float64) - intrinsics.cy) / intrinsics.fy
    denominator = n_x * columns_m[np.newaxis, :] + n_y * rows_m[:, np.newaxis] + n_z
    valid = denominator < 0.0
    depth = np.full((rows, columns), np.nan, dtype=np.float32)
    depth[valid] = (-d / denominator[valid]).astype(np.float32)
    return depth


def _require_cv2() -> None:
    """Raise a helpful error when the vision extra is not installed."""

    if cv2 is None:  # pragma: no cover - depends on the installed extra
        raise RuntimeError(
            "OpenCV is required for the depth hand localizer; "
            "install the 'vision' extra (uv sync --extra vision)"
        )
