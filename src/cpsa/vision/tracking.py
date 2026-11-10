"""Per-keypoint 3D constant-velocity EKF and RTS smoother.

Each keypoint is filtered independently with a 6D state ``[position, velocity]``
in the camera frame. A valid 3D observation is used directly; otherwise a 2D
pixel observation is fused through the camera projection model. Measurements
that fail a chi-square gate are treated as missing (predict only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cpsa.vision.calibration import Intrinsics, project, project_jacobian

_MIN_CONF = 1e-3
_STATE_DIM = 6
_EPS = 1e-9


@dataclass(frozen=True)
class TrackerConfig:
    """Noise and gating parameters for the per-keypoint EKF."""

    accel_std: float
    meas_std_px: float
    meas_std_depth_rel: float
    gate_chi2_3d: float = 11.34
    gate_chi2_2d: float = 9.21
    max_missed: int = 15


@dataclass(frozen=True)
class TrackState:
    """One causal filter step for ``K`` keypoints."""

    t: float
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    cov: NDArray[np.float64]
    valid: NDArray[np.bool_]
    missed: NDArray[np.int64]
    accepted: NDArray[np.bool_]
    gated: NDArray[np.bool_]


@dataclass(frozen=True)
class Track:
    """A whole-sequence track for ``K`` keypoints.

    ``accepted``, ``gated`` and ``missed`` are forward-pass diagnostics:
    ``accepted`` marks frames whose measurement updated the state, ``gated``
    marks frames whose measurement was rejected by the chi-square gate, and
    ``missed`` counts consecutive frames without an accepted update.
    """

    t: NDArray[np.float64]
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    cov: NDArray[np.float64]
    valid: NDArray[np.bool_]
    accepted: NDArray[np.bool_]
    gated: NDArray[np.bool_]
    missed: NDArray[np.int64]


def _transition(dt: float) -> NDArray[np.float64]:
    transition = np.eye(_STATE_DIM, dtype=np.float64)
    transition[:3, 3:] = np.eye(3, dtype=np.float64) * dt
    return transition


def _process_noise(dt: float, accel_std: float) -> NDArray[np.float64]:
    """Discrete white-noise-acceleration process noise for one step."""
    q = accel_std * accel_std
    covariance = np.zeros((_STATE_DIM, _STATE_DIM), dtype=np.float64)
    covariance[:3, :3] = np.eye(3, dtype=np.float64) * (q * dt**3 / 3.0)
    covariance[:3, 3:] = np.eye(3, dtype=np.float64) * (q * dt**2 / 2.0)
    covariance[3:, :3] = covariance[:3, 3:]
    covariance[3:, 3:] = np.eye(3, dtype=np.float64) * (q * dt)
    return covariance


def _confidence_scale(conf: float) -> float:
    if not np.isfinite(conf):
        conf = 0.0
    return 1.0 / max(conf, _MIN_CONF)


def _measurement_3d_covariance(
    depth_m: float,
    conf: float,
    intrinsics: Intrinsics,
    cfg: TrackerConfig,
) -> NDArray[np.float64]:
    depth = abs(depth_m)
    lateral_std_x = cfg.meas_std_px * depth / intrinsics.fx
    lateral_std_y = cfg.meas_std_px * depth / intrinsics.fy
    depth_std = cfg.meas_std_depth_rel * depth
    scale = _confidence_scale(conf)
    return (
        np.diag(
            np.array(
                [
                    lateral_std_x * lateral_std_x,
                    lateral_std_y * lateral_std_y,
                    depth_std * depth_std,
                ],
                dtype=np.float64,
            )
        )
        * scale
    )


def _measurement_2d_covariance(conf: float, cfg: TrackerConfig) -> NDArray[np.float64]:
    variance = cfg.meas_std_px * cfg.meas_std_px * _confidence_scale(conf)
    return np.diag(np.array([variance, variance], dtype=np.float64))


class _KeypointFilter:
    """One 6D constant-velocity EKF."""

    def __init__(self, cfg: TrackerConfig, intrinsics: Intrinsics) -> None:
        self._cfg = cfg
        self._intrinsics = intrinsics
        self.x = np.zeros(_STATE_DIM, dtype=np.float64)
        self.P = np.zeros((_STATE_DIM, _STATE_DIM), dtype=np.float64)
        self.initialized = False
        self.valid = False
        self.missed = 0

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        transition = _transition(dt)
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + _process_noise(dt, self._cfg.accel_std)

    def update(
        self,
        obs3d: NDArray[np.float64],
        obs2d: NDArray[np.float64],
        conf: float,
    ) -> tuple[bool, bool, bool]:
        """Apply one measurement; return ``(accepted, gated, reset)``."""
        has_3d = bool(np.all(np.isfinite(obs3d))) and bool(obs3d[2] > 0.0)
        if has_3d:
            if not self.initialized or not self.valid:
                self._initialize_3d(obs3d, conf)
                return True, False, True
            covariance = _measurement_3d_covariance(
                float(obs3d[2]), conf, self._intrinsics, self._cfg
            )
            jacobian = np.zeros((3, _STATE_DIM), dtype=np.float64)
            jacobian[:, :3] = np.eye(3, dtype=np.float64)
            if self._apply(
                obs3d,
                self.x[:3],
                jacobian,
                covariance,
                self._cfg.gate_chi2_3d,
            ):
                self._accept()
                return True, False, False
            self._miss()
            return False, True, False

        has_2d = bool(np.all(np.isfinite(obs2d)))
        if has_2d and self.initialized and self.valid:
            predicted_pixel = project(self.x[:3], self._intrinsics)
            if bool(np.all(np.isfinite(predicted_pixel))) and bool(self.x[2] > _EPS):
                covariance = _measurement_2d_covariance(conf, self._cfg)
                jacobian_pixel = project_jacobian(self.x[:3], self._intrinsics)
                jacobian = np.zeros((2, _STATE_DIM), dtype=np.float64)
                jacobian[:, :3] = jacobian_pixel
                if self._apply(
                    obs2d,
                    predicted_pixel,
                    jacobian,
                    covariance,
                    self._cfg.gate_chi2_2d,
                ):
                    self._accept()
                    return True, False, False
                self._miss()
                return False, True, False

        # A 2D-only observation cannot initialise a 3D track.
        if self.initialized:
            self._miss()
        return False, False, False

    def _initialize_3d(self, obs3d: NDArray[np.float64], conf: float) -> None:
        self.x = np.zeros(_STATE_DIM, dtype=np.float64)
        self.x[:3] = obs3d
        self.P = np.zeros((_STATE_DIM, _STATE_DIM), dtype=np.float64)
        self.P[:3, :3] = _measurement_3d_covariance(
            float(obs3d[2]), conf, self._intrinsics, self._cfg
        )
        # One second of acceleration uncertainty gives the initial velocity room
        # to adapt without making the first frames numerically singular.
        self.P[3:, 3:] = np.eye(3, dtype=np.float64) * (self._cfg.accel_std**2)
        self.initialized = True
        self.valid = True
        self.missed = 0

    def _apply(
        self,
        measurement: NDArray[np.float64],
        predicted: NDArray[np.float64],
        jacobian: NDArray[np.float64],
        measurement_covariance: NDArray[np.float64],
        gate: float,
    ) -> bool:
        innovation = measurement - predicted
        innovation_covariance = jacobian @ self.P @ jacobian.T + measurement_covariance
        try:
            solved = np.linalg.solve(innovation_covariance, innovation)
        except np.linalg.LinAlgError:
            return False
        distance = float(innovation @ solved)
        if not np.isfinite(distance) or distance > gate:
            return False

        gain = np.linalg.solve(innovation_covariance, jacobian @ self.P).T
        self.x = self.x + gain @ innovation
        identity = np.eye(_STATE_DIM, dtype=np.float64)
        residual = identity - gain @ jacobian
        self.P = residual @ self.P @ residual.T + gain @ measurement_covariance @ gain.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def _accept(self) -> None:
        self.missed = 0
        self.valid = True

    def _miss(self) -> None:
        self.missed += 1
        self.valid = self.missed <= self._cfg.max_missed


class CausalTracker:
    """Online tracker for ``K`` keypoints.

    ``update`` accepts one frame at a time. Timestamps must be non-decreasing;
    the filter predicts with the elapsed ``dt`` before each update.
    """

    def __init__(self, cfg: TrackerConfig, intrinsics: Intrinsics, n_keypoints: int) -> None:
        if n_keypoints < 1:
            raise ValueError(f"n_keypoints must be positive, got {n_keypoints}")
        self._n_keypoints = n_keypoints
        self._filters = [_KeypointFilter(cfg, intrinsics) for _ in range(n_keypoints)]
        self._last_t: float | None = None

    def update(
        self,
        t: float,
        obs3d: ArrayLike,
        obs2d: ArrayLike,
        conf: ArrayLike,
    ) -> TrackState:
        timestamp = float(t)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        observations_3d = np.asarray(obs3d, dtype=np.float64)
        observations_2d = np.asarray(obs2d, dtype=np.float64)
        confidences = np.asarray(conf, dtype=np.float64)
        if observations_3d.shape != (self._n_keypoints, 3):
            raise ValueError(
                f"obs3d must have shape ({self._n_keypoints}, 3), got {observations_3d.shape}"
            )
        if observations_2d.shape != (self._n_keypoints, 2):
            raise ValueError(
                f"obs2d must have shape ({self._n_keypoints}, 2), got {observations_2d.shape}"
            )
        if confidences.shape != (self._n_keypoints,):
            raise ValueError(
                f"conf must have shape ({self._n_keypoints},), got {confidences.shape}"
            )

        if self._last_t is None:
            dt = 0.0
        else:
            dt = timestamp - self._last_t
            if dt < 0.0:
                raise ValueError(
                    f"timestamps must be non-decreasing, got {timestamp} after {self._last_t}"
                )

        accepted = np.zeros(self._n_keypoints, dtype=np.bool_)
        gated = np.zeros(self._n_keypoints, dtype=np.bool_)
        for index, keypoint_filter in enumerate(self._filters):
            keypoint_filter.predict(dt)
            accepted[index], gated[index], _ = keypoint_filter.update(
                observations_3d[index],
                observations_2d[index],
                float(confidences[index]),
            )

        self._last_t = timestamp
        return TrackState(
            t=timestamp,
            pos=np.stack([f.x[:3] if f.initialized else np.full(3, np.nan) for f in self._filters]),
            vel=np.stack([f.x[3:] if f.initialized else np.full(3, np.nan) for f in self._filters]),
            cov=np.stack([f.P for f in self._filters]),
            valid=np.array([f.valid for f in self._filters], dtype=np.bool_),
            missed=np.array([f.missed for f in self._filters], dtype=np.int64),
            accepted=accepted,
            gated=gated,
        )


def smooth(
    t: ArrayLike,
    obs3d: ArrayLike,
    obs2d: ArrayLike,
    conf: ArrayLike,
    intrinsics: Intrinsics,
    cfg: TrackerConfig,
) -> Track:
    """Forward EKF plus RTS backward pass over a whole sequence.

    ``t`` is ``(T,)``; ``obs3d`` is ``(T, K, 3)`` with ``NaN`` for missing;
    ``obs2d`` is ``(T, K, 2)`` and ``conf`` is ``(T, K)``. Each keypoint is
    initialised by its first valid 3D observation; earlier frames stay invalid
    with ``NaN`` position. Missing or gated measurements are bridged by the
    motion model for up to ``max_missed`` consecutive frames.
    """
    timestamps = np.asarray(t, dtype=np.float64)
    observations_3d = np.asarray(obs3d, dtype=np.float64)
    observations_2d = np.asarray(obs2d, dtype=np.float64)
    confidences = np.asarray(conf, dtype=np.float64)

    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError(f"t must be a non-empty 1D array, got {timestamps.shape}")
    if observations_3d.ndim != 3 or observations_3d.shape[-1] != 3:
        raise ValueError(f"obs3d must have shape (T, K, 3), got {observations_3d.shape}")
    frame_count = timestamps.size
    keypoint_count = observations_3d.shape[1]
    if observations_3d.shape[0] != frame_count:
        raise ValueError(f"obs3d has {observations_3d.shape[0]} frames but t has {frame_count}")
    if observations_2d.shape != (frame_count, keypoint_count, 2):
        raise ValueError(
            f"obs2d must have shape ({frame_count}, {keypoint_count}, 2), got "
            f"{observations_2d.shape}"
        )
    if confidences.shape != (frame_count, keypoint_count):
        raise ValueError(
            f"conf must have shape ({frame_count}, {keypoint_count}), got {confidences.shape}"
        )
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("t must be finite")
    if np.any(np.diff(timestamps) < 0.0):
        raise ValueError("t must be non-decreasing")

    filters = [_KeypointFilter(cfg, intrinsics) for _ in range(keypoint_count)]
    state_shape = (frame_count, keypoint_count, _STATE_DIM)
    covariance_shape = (frame_count, keypoint_count, _STATE_DIM, _STATE_DIM)
    prior_mean = np.full(state_shape, np.nan, dtype=np.float64)
    prior_covariance = np.full(covariance_shape, np.nan, dtype=np.float64)
    posterior_mean = np.full(state_shape, np.nan, dtype=np.float64)
    posterior_covariance = np.full(covariance_shape, np.nan, dtype=np.float64)
    valid = np.zeros((frame_count, keypoint_count), dtype=np.bool_)
    accepted = np.zeros((frame_count, keypoint_count), dtype=np.bool_)
    gated = np.zeros((frame_count, keypoint_count), dtype=np.bool_)
    missed = np.zeros((frame_count, keypoint_count), dtype=np.int64)
    reset = np.zeros((frame_count, keypoint_count), dtype=np.bool_)

    for frame in range(frame_count):
        dt = 0.0 if frame == 0 else float(timestamps[frame] - timestamps[frame - 1])
        for index, keypoint_filter in enumerate(filters):
            keypoint_filter.predict(dt)
            if keypoint_filter.initialized:
                prior_mean[frame, index] = keypoint_filter.x
                prior_covariance[frame, index] = keypoint_filter.P
            accepted[frame, index], gated[frame, index], reset[frame, index] = (
                keypoint_filter.update(
                    observations_3d[frame, index],
                    observations_2d[frame, index],
                    float(confidences[frame, index]),
                )
            )
            if keypoint_filter.initialized:
                posterior_mean[frame, index] = keypoint_filter.x
                posterior_covariance[frame, index] = keypoint_filter.P
            valid[frame, index] = keypoint_filter.valid
            missed[frame, index] = keypoint_filter.missed

    smoothed_mean = posterior_mean.copy()
    smoothed_covariance = posterior_covariance.copy()
    for frame in range(frame_count - 2, -1, -1):
        dt = float(timestamps[frame + 1] - timestamps[frame])
        transition = _transition(dt)
        for index in range(keypoint_count):
            # A track that was re-initialised after ``max_missed`` starts a new
            # segment. Its old prediction is unrelated to the new state, so the
            # RTS recursion must not cross that boundary.
            if reset[frame + 1, index]:
                continue
            if (
                not np.all(np.isfinite(posterior_mean[frame, index]))
                or not np.all(np.isfinite(prior_mean[frame + 1, index]))
                or not np.all(np.isfinite(smoothed_mean[frame + 1, index]))
            ):
                continue
            covariance = posterior_covariance[frame, index]
            try:
                smoother_gain = (
                    covariance @ transition.T @ np.linalg.inv(prior_covariance[frame + 1, index])
                )
            except np.linalg.LinAlgError:
                continue
            smoothed_mean[frame, index] = posterior_mean[frame, index] + smoother_gain @ (
                smoothed_mean[frame + 1, index] - prior_mean[frame + 1, index]
            )
            smooth_cov = (
                covariance
                + smoother_gain
                @ (smoothed_covariance[frame + 1, index] - prior_covariance[frame + 1, index])
                @ smoother_gain.T
            )
            smoothed_covariance[frame, index] = 0.5 * (smooth_cov + smooth_cov.T)

    return Track(
        t=timestamps,
        pos=smoothed_mean[..., :3],
        vel=smoothed_mean[..., 3:],
        cov=smoothed_covariance,
        valid=valid,
        accepted=accepted,
        gated=gated,
        missed=missed,
    )
