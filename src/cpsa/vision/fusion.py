"""Event-level IMU-vision fusion: clock offset, depth occlusion and contacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cpsa.domain.model import ContactEvent, FaceRegion, Side
from cpsa.vision.tracking import Track

_MAX_COMMON_RATE_HZ = 500.0
_MIN_PEARSON_DENOMINATOR = 1e-12


@dataclass(frozen=True)
class ContactConfig:
    """Thresholds for hand-face contact detection (tuned in Task 11)."""

    contact_distance_m: float
    imu_peak_threshold_g: float
    imu_peak_window_s: float
    min_occluder_fraction: float
    min_separation_s: float
    approach_window_s: float = 0.3


@dataclass(frozen=True)
class IMUSignal:
    """Wrist acceleration magnitude in g, on the IMU clock.

    Raw ``imu.csv`` samples are recorded in mg, so callers must divide them by
    1000 before constructing this signal.
    """

    t_s: NDArray[np.float64]
    acc_norm_g: NDArray[np.float64]


def _clean_series(
    values: ArrayLike,
    times: ArrayLike,
    name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    series = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(times, dtype=np.float64)
    if series.ndim != 1 or timestamps.ndim != 1:
        raise ValueError(f"{name} values and timestamps must be 1D")
    if series.size != timestamps.size:
        raise ValueError(f"{name} values and timestamps must have the same length")
    finite = np.isfinite(series) & np.isfinite(timestamps)
    series = series[finite]
    timestamps = timestamps[finite]
    if series.size < 3:
        raise ValueError(f"{name} needs at least 3 finite samples")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{name} timestamps must be strictly increasing")
    return timestamps, series


def _pearson(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    centered_first = first - float(np.mean(first))
    centered_second = second - float(np.mean(second))
    denominator = float(
        np.sqrt(float(centered_first @ centered_first) * float(centered_second @ centered_second))
    )
    if denominator <= _MIN_PEARSON_DENOMINATOR:
        return 0.0
    return float((centered_first @ centered_second) / denominator)


def estimate_clock_offset(
    imu_t: ArrayLike,
    imu_acc_norm: ArrayLike,
    cam_t: ArrayLike,
    cam_acc_norm: ArrayLike,
    max_lag_s: float = 0.3,
) -> tuple[float, float]:
    """Estimate the lag between wrist-IMU and camera acceleration magnitude.

    Returns ``(offset_s, peak_corr)`` where ``offset_s`` must be *added* to IMU
    timestamps to place them on the camera clock. Both signals are linearly
    resampled to the faster of the two native rates, then normalised
    cross-correlation is evaluated over integer lags in ``[-max_lag_s,
    max_lag_s]``. The peak is refined with parabolic interpolation, so the
    result is not limited to the resampling grid.

    Degenerate constant signals return ``(0.0, 0.0)``.
    """
    if max_lag_s <= 0.0:
        raise ValueError(f"max_lag_s must be positive, got {max_lag_s}")
    imu_time, imu_values = _clean_series(imu_acc_norm, imu_t, "imu")
    cam_time, cam_values = _clean_series(cam_acc_norm, cam_t, "camera")

    imu_step = float(np.median(np.diff(imu_time)))
    cam_step = float(np.median(np.diff(cam_time)))
    rate_hz = min(1.0 / min(imu_step, cam_step), _MAX_COMMON_RATE_HZ)
    step = 1.0 / rate_hz

    start = max(float(imu_time[0]), float(cam_time[0])) + max_lag_s
    stop = min(float(imu_time[-1]), float(cam_time[-1])) - max_lag_s
    if stop <= start:
        raise ValueError("signals do not overlap by more than 2 * max_lag_s on both sides")
    sample_count = int(np.floor((stop - start) / step)) + 1
    if sample_count < 5:
        raise ValueError("signals are too short for clock-offset estimation")
    grid = start + np.arange(sample_count, dtype=np.float64) * step
    imu_grid = np.interp(grid, imu_time, imu_values)

    lag_count = int(np.floor(max_lag_s / step + 1e-9))
    lags = np.arange(-lag_count, lag_count + 1, dtype=np.float64) * step
    correlations = np.empty(lags.size, dtype=np.float64)
    for index, lag in enumerate(lags):
        cam_shifted = np.interp(grid + lag, cam_time, cam_values)
        correlations[index] = _pearson(imu_grid, cam_shifted)

    best = int(np.argmax(correlations))
    peak_corr = float(correlations[best])
    if peak_corr == 0.0:
        return 0.0, 0.0
    offset = float(lags[best])
    if 0 < best < correlations.size - 1:
        left = float(correlations[best - 1])
        middle = peak_corr
        right = float(correlations[best + 1])
        denominator = left - 2.0 * middle + right
        if abs(denominator) > _MIN_PEARSON_DENOMINATOR:
            delta = 0.5 * (left - right) / denominator
            if abs(delta) <= 1.0:
                offset += delta * step
    return offset, peak_corr


def occluder_fraction(
    depth_m: ArrayLike,
    face_box: ArrayLike,
    face_depth_m: float,
    margin_m: float = 0.03,
) -> float:
    """Fraction of valid face-box depth pixels in front of the face depth.

    ``face_box`` is ``(x0, y0, x1, y1)`` in pixels, with ``x1``/``y1``
    exclusive; the box is clipped to the image. The denominator is the number
    of valid (non-zero, finite) pixels inside the box: missing depth is not
    evidence of either occlusion or absence, so it is ignored. Returns ``0.0``
    for an empty or fully invalid box.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"expected a 2D depth image, got shape {depth.shape}")
    box = np.asarray(face_box, dtype=np.float64).reshape(-1)
    if box.size != 4:
        raise ValueError(f"face_box must have 4 values, got {box.size}")
    if not np.all(np.isfinite(box)):
        raise ValueError("face_box must be finite")
    if not np.isfinite(face_depth_m):
        raise ValueError("face_depth_m must be finite")

    height, width = depth.shape
    x0, y0, x1, y1 = (float(value) for value in box)
    left = max(0, int(np.floor(min(x0, x1))))
    right = min(width, int(np.ceil(max(x0, x1))))
    top = max(0, int(np.floor(min(y0, y1))))
    bottom = min(height, int(np.ceil(max(y0, y1))))
    if right <= left or bottom <= top:
        return 0.0

    crop = depth[top:bottom, left:right]
    valid = np.isfinite(crop) & (crop > 0.0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return 0.0
    occluded = valid & (crop <= face_depth_m - margin_m)
    return float(np.count_nonzero(occluded)) / float(valid_count)


def _normalise_anchors(
    anchors: Mapping[FaceRegion, ArrayLike] | Iterable[tuple[FaceRegion, ArrayLike]],
) -> list[tuple[FaceRegion, NDArray[np.float64]]]:
    items: Iterable[tuple[FaceRegion, ArrayLike]] = (
        anchors.items() if isinstance(anchors, Mapping) else anchors
    )
    normalised: list[tuple[FaceRegion, NDArray[np.float64]]] = []
    for region, points in items:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"anchor {region!r} must have shape (T, 3), got {array.shape}")
        normalised.append((region, array))
    if not normalised:
        raise ValueError("at least one face anchor is required")
    return normalised


def _approach_speed(
    t: NDArray[np.float64],
    pos: NDArray[np.float64],
    vel: NDArray[np.float64],
    valid: NDArray[np.bool_],
    anchor_points: NDArray[np.float64],
    anchor_velocity: NDArray[np.float64],
    contact_frame: int,
    keypoint: int,
    anchor: int,
    window_s: float,
) -> float:
    start = float(t[contact_frame]) - window_s
    frames = np.flatnonzero((t >= start) & (t <= t[contact_frame]))
    frames = frames[valid[frames, keypoint]]
    if frames.size == 0:
        return 0.0
    offset = pos[frames, keypoint] - anchor_points[frames, anchor]
    relative_velocity = vel[frames, keypoint] - anchor_velocity[frames, anchor]
    finite = np.all(np.isfinite(offset), axis=1) & np.all(np.isfinite(relative_velocity), axis=1)
    if not np.any(finite):
        return 0.0
    offset = offset[finite]
    relative_velocity = relative_velocity[finite]
    distance = np.linalg.norm(offset, axis=1)
    safe_distance = np.maximum(distance, 1e-9)
    closing = -np.einsum("ij,ij->i", offset, relative_velocity) / safe_distance
    closing = closing[np.isfinite(closing)]
    if closing.size == 0:
        return 0.0
    return float(max(0.0, float(np.max(closing))))


def _max_peak(first: float | None, second: float | None) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)


def _impact_peak(
    imu: IMUSignal | None,
    contact_t: float,
    cfg: ContactConfig,
    clock_offset_s: float,
) -> float | None:
    if imu is None:
        return None
    times = np.asarray(imu.t_s, dtype=np.float64) + clock_offset_s
    values = np.asarray(imu.acc_norm_g, dtype=np.float64) - 1.0
    if times.shape != values.shape or times.ndim != 1:
        raise ValueError("IMUSignal t_s and acc_norm_g must be 1D and equal length")
    window = cfg.imu_peak_window_s
    selected = (times >= contact_t - window) & (times <= contact_t + window)
    selected &= np.isfinite(times) & np.isfinite(values)
    if not np.any(selected):
        return None
    peak = float(np.max(values[selected]))
    if peak < cfg.imu_peak_threshold_g:
        return None
    return peak


def _episodes(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    chunks = np.split(indices, breaks)
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks]


def detect_contacts(
    track: Mapping[Side, Track],
    anchors: Mapping[FaceRegion, ArrayLike] | Iterable[tuple[FaceRegion, ArrayLike]],
    imu: IMUSignal | None,
    occluder: ArrayLike | None,
    cfg: ContactConfig,
    *,
    clock_offset_s: float = 0.0,
) -> list[ContactEvent]:
    """Detect hand-to-face contacts from smoothed 3D tracks and depth evidence.

    ``track`` maps each side to its smoothed :class:`~cpsa.vision.tracking.Track`
    (positions, velocities, valid mask and timestamps are used); the keypoint
    layout is not interpreted, so no keypoint schema is required.
    Each side is evaluated independently against every face anchor. A contact
    episode is a maximal run of frames where the minimum keypoint-anchor
    distance is at most ``cfg.contact_distance_m``; its event time is the first
    frame of the run. The episode's region comes from the nearest anchor at the
    closest frame, and ``min_distance_m`` is the minimum over the episode.

    ``occluder`` is the per-frame :func:`occluder_fraction` array; an event is
    flagged when the maximum fraction in its episode reaches
    ``cfg.min_occluder_fraction``. ``impact_peak_g`` is the largest
    ``|acc| - 1 g`` within ``cfg.imu_peak_window_s`` of the event (IMU
    timestamps plus ``clock_offset_s``), or ``None`` when there is no IMU, no
    sample in the window, or the peak is below ``cfg.imu_peak_threshold_g``.
    Events on the same side closer than ``cfg.min_separation_s`` are merged into
    the earlier event: the merged event keeps the earliest time and region and
    takes the maximum impact peak, the logical OR of the occluder flags, the
    minimum distance and the maximum approach speed.
    """
    if not np.isfinite(clock_offset_s):
        raise ValueError("clock_offset_s must be finite")
    if not np.isfinite(cfg.contact_distance_m) or cfg.contact_distance_m <= 0.0:
        raise ValueError("cfg.contact_distance_m must be positive")

    sides = list(track.items())
    if not sides:
        raise ValueError("track must contain at least one side")
    anchor_items = _normalise_anchors(anchors)
    anchor_points = np.stack([points for _, points in anchor_items], axis=1)  # (T, A, 3)
    regions = [region for region, _ in anchor_items]
    frame_count = anchor_points.shape[0]
    anchor_count = len(anchor_items)
    if frame_count < 1:
        raise ValueError("at least one frame is required")
    if not np.isfinite(cfg.imu_peak_window_s) or cfg.imu_peak_window_s < 0.0:
        raise ValueError("cfg.imu_peak_window_s must be non-negative")
    if not np.isfinite(cfg.min_separation_s) or cfg.min_separation_s < 0.0:
        raise ValueError("cfg.min_separation_s must be non-negative")
    if not np.isfinite(cfg.approach_window_s) or cfg.approach_window_s < 0.0:
        raise ValueError("cfg.approach_window_s must be non-negative")
    if not np.isfinite(cfg.imu_peak_threshold_g):
        raise ValueError("cfg.imu_peak_threshold_g must be finite")
    if not 0.0 <= cfg.min_occluder_fraction <= 1.0:
        raise ValueError("cfg.min_occluder_fraction must be in [0, 1]")

    timestamps = np.asarray(sides[0][1].t, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size != frame_count:
        raise ValueError(
            f"track timestamps must have shape ({frame_count},), got {timestamps.shape}"
        )
    if timestamps.size >= 2 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("track timestamps must be strictly increasing")
    for side, side_track in sides:
        if not np.array_equal(np.asarray(side_track.t, dtype=np.float64), timestamps):
            raise ValueError(f"track timestamps differ for side {side!r}")
        positions_shape = np.shape(side_track.pos)
        if len(positions_shape) != 3 or positions_shape[0] != frame_count:
            raise ValueError(f"track positions for side {side!r} have an invalid shape")
        if positions_shape[2] != 3:
            raise ValueError(f"track positions for side {side!r} have an invalid shape")
        if np.shape(side_track.vel) != positions_shape:
            raise ValueError(f"track velocities for side {side!r} have an invalid shape")
        if np.shape(side_track.valid) != positions_shape[:2]:
            raise ValueError(f"track valid mask for side {side!r} has an invalid shape")

    if occluder is not None:
        occluder_values = np.asarray(occluder, dtype=np.float64)
        if occluder_values.shape != (frame_count,):
            raise ValueError(
                f"occluder must have shape ({frame_count},), got {occluder_values.shape}"
            )
    else:
        occluder_values = None

    anchor_velocity = np.zeros((frame_count, anchor_count, 3), dtype=np.float64)
    if frame_count >= 2:
        anchor_velocity = np.asarray(
            np.gradient(anchor_points, timestamps, axis=0, edge_order=1),
            dtype=np.float64,
        )

    events: list[ContactEvent] = []
    for side, side_track in sides:
        positions = np.asarray(side_track.pos, dtype=np.float64)
        velocities = np.asarray(side_track.vel, dtype=np.float64)
        valid = np.asarray(side_track.valid, dtype=np.bool_)
        keypoint_count = positions.shape[1]

        difference = positions[:, :, None, :] - anchor_points[:, None, :, :]
        distances = np.linalg.norm(difference, axis=-1)  # (T, K, A)
        distances = np.where(np.isfinite(distances), distances, np.inf)
        distances[~valid] = np.inf
        flat = distances.reshape(frame_count, keypoint_count * anchor_count)
        nearest = np.argmin(flat, axis=1)
        minimum_distance = flat[np.arange(frame_count), nearest]
        nearest_keypoint = nearest // anchor_count
        nearest_anchor = nearest % anchor_count

        last_event_t: float | None = None
        for start, end in _episodes(minimum_distance <= cfg.contact_distance_m):
            episode_distance = minimum_distance[start : end + 1]
            closest = start + int(np.argmin(episode_distance))
            event_t = float(timestamps[start])
            keypoint = int(nearest_keypoint[closest])
            anchor = int(nearest_anchor[closest])
            approach = _approach_speed(
                timestamps,
                positions,
                velocities,
                valid,
                anchor_points,
                anchor_velocity,
                start,
                keypoint,
                anchor,
                cfg.approach_window_s,
            )
            occluded = False
            if occluder_values is not None:
                window = occluder_values[start : end + 1]
                finite = window[np.isfinite(window)]
                if finite.size:
                    occluded = bool(float(np.max(finite)) >= cfg.min_occluder_fraction)
            peak = _impact_peak(imu, event_t, cfg, clock_offset_s)
            candidate = ContactEvent(
                t=event_t,
                side=side,
                region=regions[anchor],
                min_distance_m=float(minimum_distance[closest]),
                approach_speed_mps=approach,
                impact_peak_g=peak,
                occluder=occluded,
            )
            if last_event_t is not None and event_t - last_event_t < cfg.min_separation_s:
                previous = events[-1]
                events[-1] = ContactEvent(
                    t=previous.t,
                    side=previous.side,
                    region=previous.region,
                    min_distance_m=min(previous.min_distance_m, candidate.min_distance_m),
                    approach_speed_mps=max(
                        previous.approach_speed_mps, candidate.approach_speed_mps
                    ),
                    impact_peak_g=_max_peak(previous.impact_peak_g, candidate.impact_peak_g),
                    occluder=previous.occluder or candidate.occluder,
                )
                continue
            events.append(candidate)
            last_event_t = event_t

    events.sort(key=lambda event: (event.t, event.side.value, event.region.value))
    return events
