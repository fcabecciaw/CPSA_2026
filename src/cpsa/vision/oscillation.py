"""Oscillation analysis for rocking and hand flapping (FR-019, AC-010).

The periodicity estimator detrends the signal, resamples it on a uniform grid
and takes the normalized autocorrelation peak inside the requested band. Its
strength is the normalized peak, so a clean oscillation is close to 1 and an
aperiodic signal stays low.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cpsa.domain.model import Side
from cpsa.vision.types import KeypointLayout, KeypointStatus, TrackedFrame

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from cpsa.vision.assessment import AnalysisConfig

_EPS = 1e-9
_LOST = int(KeypointStatus.LOST)


@dataclass(frozen=True)
class OscillationConfig:
    """Bands and thresholds of the rocking and flapping detectors.

    ``rocking_min_amp_fw`` is a sine amplitude (half peak-to-peak) in
    face-widths. ``flapping_height_margin_fw`` is how far below the shoulder
    midpoint the wrist may sit and still count as near shoulder height.
    """

    rocking_band_hz: tuple[float, float] = (0.5, 2.0)
    flapping_band_hz: tuple[float, float] = (2.0, 6.0)
    rocking_min_strength: float = 0.5
    rocking_min_amp_fw: float = 0.05
    flapping_min_strength: float = 0.5
    flapping_height_margin_fw: float = 0.5
    min_conf: float = 0.3
    min_samples: int = 8


def _detrend(t: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove the mean and a least-squares linear trend."""
    centered = t - float(np.mean(t))
    mean_value = float(np.mean(values))
    variance = float(centered @ centered)
    if variance <= _EPS:
        return values - mean_value
    slope = float(centered @ (values - mean_value)) / variance
    return values - mean_value - slope * centered


def _normalized_autocorrelation(values: NDArray[np.float64], lag: int) -> float:
    head = values[:-lag]
    tail = values[lag:]
    denominator = float(np.sqrt(float(head @ head) * float(tail @ tail)))
    if denominator <= _EPS:
        return 0.0
    return float((head @ tail) / denominator)


def periodicity(
    t: ArrayLike,
    signal: ArrayLike,
    band_hz: tuple[float, float],
) -> tuple[float | None, float]:
    """Return ``(frequency, strength)`` of the strongest periodic component.

    The signal is linearly detrended and resampled on a uniform grid before the
    autocorrelation is evaluated. ``frequency`` is the peak in ``band_hz`` (refined
    by parabolic interpolation) and is ``None`` when the band holds no positive
    correlation or the signal is too short. ``strength`` is the normalized
    autocorrelation peak in ``[0, 1]``; ``0.0`` means no periodicity.
    """
    times = np.asarray(t, dtype=np.float64).reshape(-1)
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if times.size != values.size:
        raise ValueError("t and signal must have the same length")
    low, high = float(band_hz[0]), float(band_hz[1])
    if not (np.isfinite(low) and np.isfinite(high)) or low <= 0.0 or high <= low:
        raise ValueError(f"band_hz must be positive and increasing, got {band_hz!r}")
    if times.size and np.any(np.diff(times) <= 0.0):
        raise ValueError("t must be strictly increasing")

    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]
    if times.size < 4:
        return None, 0.0
    step = float(np.median(np.diff(times)))
    if not np.isfinite(step) or step <= 0.0:
        return None, 0.0
    grid = np.arange(times[0], times[-1] + 0.5 * step, step, dtype=np.float64)
    if grid.size < 4:
        return None, 0.0
    detrended = _detrend(grid, np.interp(grid, times, values))
    if float(detrended @ detrended) <= _EPS:
        return None, 0.0

    rate_hz = 1.0 / step
    min_lag = max(1, int(np.ceil(rate_hz / high)))
    max_lag = min(int(np.floor(rate_hz / low)), grid.size - max(8, grid.size // 4))
    if max_lag < min_lag:
        return None, 0.0

    correlations = {
        lag: _normalized_autocorrelation(detrended, lag) for lag in range(min_lag, max_lag + 1)
    }
    best_lag = max(correlations, key=lambda lag: correlations[lag])
    peak = correlations[best_lag]
    if peak <= 0.0:
        return None, 0.0

    refined = float(best_lag)
    left = correlations.get(best_lag - 1)
    right = correlations.get(best_lag + 1)
    if left is not None and right is not None:
        denominator = left - 2.0 * peak + right
        if abs(denominator) > _EPS:
            delta = 0.5 * (left - right) / denominator
            if abs(delta) <= 1.0:
                refined += delta
    strength = float(np.clip(peak, 0.0, 1.0))
    if refined <= _EPS:
        return None, strength
    frequency = float(np.clip(rate_hz / refined, low, high))
    return frequency, strength


def _series(
    frames: list[TrackedFrame],
    layout: KeypointLayout,
    names: tuple[str, ...],
    min_conf: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Collect a time series of tracked keypoints with a known face width.

    Returns ``(t, points, face_width_px)`` with ``points`` shaped
    ``(n, len(names), 2)``. Frames with a missing face width, a lost or
    low-confidence keypoint, or a non-increasing timestamp are skipped.
    """
    indices = np.asarray([layout.index(name) for name in names], dtype=np.int64)
    times: list[float] = []
    points: list[NDArray[np.float64]] = []
    widths: list[float] = []
    for frame in frames:
        width = frame.face_width_px
        if width is None or not np.isfinite(width) or width <= 0.0:
            continue
        if times and frame.t <= times[-1]:
            continue
        status = np.asarray(frame.status, dtype=np.int8)[indices]
        conf = np.asarray(frame.conf, dtype=np.float64)[indices]
        xy = np.asarray(frame.xy, dtype=np.float64)[indices]
        if bool(np.any(status == _LOST)) or bool(np.any(conf < min_conf)):
            continue
        if not bool(np.all(np.isfinite(xy))):
            continue
        times.append(float(frame.t))
        points.append(xy)
        widths.append(float(width))
    if not times:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, len(names), 2), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(points, dtype=np.float64),
        np.asarray(widths, dtype=np.float64),
    )


def detect_rocking(
    frames: list[TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
) -> bool:
    """Detect rocking from the shoulder-midpoint horizontal oscillation (FR-019).

    The signal is the midpoint ``x`` of the two shoulders in face-widths. The
    frame counts as rocking when the periodicity strengthens by at least
    ``cfg.oscillation.rocking_min_strength`` inside ``rocking_band_hz`` and the
    sine amplitude reaches ``rocking_min_amp_fw``.
    """
    osc = cfg.oscillation
    names = ("left_shoulder", "right_shoulder")
    if not all(layout.has(name) for name in names):
        return False
    times, points, widths = _series(frames, layout, names, osc.min_conf)
    if times.size < osc.min_samples:
        return False
    signal = np.mean(points[:, :, 0], axis=1) / widths
    frequency, strength = periodicity(times, signal, osc.rocking_band_hz)
    if frequency is None or strength < osc.rocking_min_strength:
        return False
    detrended = _detrend(times, signal)
    amplitude = 0.5 * float(np.max(detrended) - np.min(detrended))
    return amplitude >= osc.rocking_min_amp_fw


def detect_flapping(
    frames: list[TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
) -> bool:
    """Detect hand flapping from the wrist oscillation near the shoulders (FR-019).

    For each side the signal is the vertical wrist offset from the
    shoulder-midpoint in face-widths. The hand must sit near or above the
    shoulders (median offset at most ``flapping_height_margin_fw``) and the
    periodicity inside ``flapping_band_hz`` must reach
    ``flapping_min_strength``. Either side qualifies.
    """
    osc = cfg.oscillation
    shoulder_names = ("left_shoulder", "right_shoulder")
    if not all(layout.has(name) for name in shoulder_names):
        return False
    for side in (Side.LEFT, Side.RIGHT):
        wrist_name = f"{side.value}_wrist"
        if not layout.has(wrist_name):
            continue
        times, points, widths = _series(frames, layout, (*shoulder_names, wrist_name), osc.min_conf)
        if times.size < osc.min_samples:
            continue
        shoulder_mid_y = np.mean(points[:, :2, 1], axis=1)
        relative = (points[:, 2, 1] - shoulder_mid_y) / widths
        if float(np.median(relative)) > osc.flapping_height_margin_fw:
            continue
        frequency, strength = periodicity(times, relative, osc.flapping_band_hz)
        if frequency is not None and strength >= osc.flapping_min_strength:
            return True
    return False
