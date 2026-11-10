"""Trajectory smoothing: EKF/RTS outlier rejection, gap bridging and causal lag."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.vision.calibration import Intrinsics, project
from cpsa.vision.tracking import CausalTracker, Track, TrackerConfig, smooth

INTRINSICS = Intrinsics(
    fx=600.0,
    fy=600.0,
    cx=320.0,
    cy=240.0,
    width=640,
    height=480,
)


def _config(**overrides: float) -> TrackerConfig:
    values: dict[str, float] = {
        "accel_std": 2.0,
        "meas_std_px": 2.0,
        "meas_std_depth_rel": 0.01,
        "max_missed": 15,
    }
    values.update(overrides)
    return TrackerConfig(**values)  # type: ignore[arg-type]


def _piecewise_velocity(
    frame_count: int,
    dt: float,
    start: NDArray[np.float64],
    velocity_before: NDArray[np.float64],
    impact_frame: int,
    velocity_after: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    velocities = np.empty((frame_count, 3))
    velocities[:impact_frame] = velocity_before
    velocities[impact_frame:] = velocity_after
    positions = np.empty((frame_count, 3))
    positions[0] = start
    for frame in range(1, frame_count):
        positions[frame] = positions[frame - 1] + velocities[frame - 1] * dt
    return positions, velocities


def _impact_observations(
    frame_count: int = 150,
    dt: float = 1.0 / 30.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    timestamps = np.arange(frame_count, dtype=np.float64) * dt
    positions, velocities = _piecewise_velocity(
        frame_count,
        dt,
        np.array([0.0, 0.0, 0.6]),
        np.array([0.2, -0.1, 0.05]),
        impact_frame=75,
        velocity_after=np.array([-0.15, 0.2, 0.1]),
    )
    rng = np.random.default_rng(7)
    observed = positions + rng.normal(0.0, 0.004, positions.shape)
    outlier_frames = np.array([10, 30, 50, 70, 90, 120, 140])
    observed[outlier_frames] += np.array([0.3, 0.0, 0.0])
    gap = np.zeros(frame_count, dtype=np.bool_)
    gap[100:111] = True
    observed[gap] = np.nan

    obs3d = observed[:, None, :]
    obs2d = project(obs3d, INTRINSICS)
    conf = np.full((frame_count, 1), 0.9)
    return timestamps, positions, obs3d, obs2d, conf, gap


def test_rts_rejects_outliers_and_halves_position_rmse() -> None:
    timestamps, truth, obs3d, obs2d, conf, gap = _impact_observations()
    cfg = _config()

    track = smooth(timestamps, obs3d, obs2d, conf, INTRINSICS, cfg)

    measured = obs3d[:, 0, :]
    measured_finite = np.all(np.isfinite(measured), axis=1)
    raw_error = np.linalg.norm(measured[measured_finite] - truth[measured_finite], axis=1)
    raw_rmse = float(np.sqrt(np.mean(raw_error**2)))

    estimated = track.pos[:, 0, :]
    estimated_finite = np.all(np.isfinite(estimated), axis=1)
    smooth_error = np.linalg.norm(estimated[estimated_finite] - truth[estimated_finite], axis=1)
    smooth_rmse = float(np.sqrt(np.mean(smooth_error**2)))

    assert smooth_rmse <= 0.5 * raw_rmse
    assert smooth_rmse < 0.01

    outlier_frames = np.array([10, 30, 50, 70, 90, 120, 140])
    assert not gap[outlier_frames].any()
    assert track.gated[outlier_frames, 0].all()
    assert int(track.gated.sum()) >= outlier_frames.size
    assert int(track.accepted.sum()) == int(measured_finite.sum()) - int(track.gated.sum())
    assert track.valid[:, 0].all()


@pytest.mark.parametrize("gap_length", [15, 17])
def test_gaps_up_to_max_missed_are_bridged(gap_length: int) -> None:
    frame_count = 90
    dt = 1.0 / 30.0
    timestamps = np.arange(frame_count, dtype=np.float64) * dt
    velocity = np.array([0.15, -0.05, 0.02])
    truth = np.array([0.0, 0.0, 0.6]) + timestamps[:, None] * velocity
    rng = np.random.default_rng(11)
    observed = truth + rng.normal(0.0, 1e-4, truth.shape)
    gap_start = 40
    gap = np.zeros(frame_count, dtype=np.bool_)
    gap[gap_start : gap_start + gap_length] = True
    observed[gap] = np.nan

    obs3d = observed[:, None, :]
    obs2d = project(obs3d, INTRINSICS)
    conf = np.full((frame_count, 1), 1.0)
    track = smooth(timestamps, obs3d, obs2d, conf, INTRINSICS, _config())

    expected_valid = gap_length <= 15
    assert bool(track.valid[gap][-1, 0]) is expected_valid
    if expected_valid:
        error = np.linalg.norm(track.pos[gap, 0, :] - truth[gap], axis=1)
        assert np.max(error) < 0.005
    else:
        assert not track.valid[gap][-1, 0]


def test_smooth_uses_2d_measurements_when_depth_is_missing() -> None:
    frame_count = 60
    dt = 1.0 / 30.0
    timestamps = np.arange(frame_count, dtype=np.float64) * dt
    velocity = np.array([0.12, 0.03, 0.0])
    truth = np.array([-0.1, 0.05, 0.7]) + timestamps[:, None] * velocity
    obs3d = truth[:, None, :].copy()
    obs3d[1:] = np.nan
    obs2d = project(truth[:, None, :], INTRINSICS)
    conf = np.full((frame_count, 1), 1.0)

    track = smooth(timestamps, obs3d, obs2d, conf, INTRINSICS, _config())

    assert track.valid[:, 0].all()
    assert track.accepted[1:, 0].all()
    assert int(track.gated.sum()) == 0
    error = track.pos[:, 0, :] - truth
    # Pixels constrain lateral position directly; absolute depth is only weakly
    # observable without depth measurements, so it may drift a little.
    assert float(np.max(np.linalg.norm(error[:, :2], axis=1))) < 0.01
    assert float(np.max(np.linalg.norm(error, axis=1))) < 0.05


def test_smooth_supports_variable_frame_intervals() -> None:
    frame_count = 50
    steps = np.resize(np.array([1.0 / 30.0, 1.0 / 15.0, 1.0 / 60.0]), frame_count - 1)
    timestamps = np.concatenate(([0.0], np.cumsum(steps)))
    velocity = np.array([0.18, -0.04, 0.03])
    truth = np.array([0.02, -0.01, 0.55]) + timestamps[:, None] * velocity
    obs3d = truth[:, None, :]
    obs2d = project(obs3d, INTRINSICS)
    conf = np.ones((frame_count, 1))

    track = smooth(timestamps, obs3d, obs2d, conf, INTRINSICS, _config())

    assert track.valid[:, 0].all()
    error = np.linalg.norm(track.pos[:, 0, :] - truth, axis=1)
    assert float(np.max(error)) < 0.01


def test_causal_tracker_lag_at_impact_is_bounded() -> None:
    frame_count = 90
    dt = 1.0 / 30.0
    impact_frame = 45
    timestamps = np.arange(frame_count, dtype=np.float64) * dt
    truth, _ = _piecewise_velocity(
        frame_count,
        dt,
        np.array([-0.2, 0.0, 0.6]),
        np.array([0.25, 0.0, 0.0]),
        impact_frame=impact_frame,
        velocity_after=np.array([-0.2, 0.1, 0.0]),
    )
    rng = np.random.default_rng(19)
    observed = truth + rng.normal(0.0, 0.002, truth.shape)
    tracker = CausalTracker(_config(), INTRINSICS, n_keypoints=1)

    estimates = []
    for frame in range(frame_count):
        state = tracker.update(
            float(timestamps[frame]),
            observed[frame][None, :],
            project(observed[frame][None, :], INTRINSICS),
            np.array([1.0]),
        )
        estimates.append(state.pos[0])
    estimates_array = np.asarray(estimates)

    error = np.linalg.norm(estimates_array - truth, axis=1)
    impact_window = error[impact_frame : impact_frame + 15]  # 0.5 s
    settled_window = error[impact_frame + 6 : impact_frame + 15]  # after 0.2 s
    assert float(np.max(impact_window)) < 0.05
    assert float(np.max(settled_window)) < 0.015


def test_causal_tracker_initialises_on_first_3d_measurement() -> None:
    tracker = CausalTracker(_config(), INTRINSICS, n_keypoints=1)
    point = np.array([[0.05, -0.02, 0.65]])

    first = tracker.update(
        t=0.0,
        obs3d=np.array([[np.nan, np.nan, np.nan]]),
        obs2d=project(point, INTRINSICS),
        conf=np.array([1.0]),
    )
    assert not first.valid[0]
    assert not first.accepted[0]

    second = tracker.update(
        t=1.0 / 30.0,
        obs3d=point,
        obs2d=project(point, INTRINSICS),
        conf=np.array([1.0]),
    )
    assert second.valid[0]
    assert second.accepted[0]
    np.testing.assert_allclose(second.pos[0], point[0], atol=1e-9)


def test_smooth_rejects_non_monotonic_timestamps() -> None:
    frame_count = 4
    obs3d = np.zeros((frame_count, 1, 3))
    obs3d[..., 2] = 0.6
    obs2d = project(obs3d, INTRINSICS)
    conf = np.ones((frame_count, 1))
    timestamps = np.array([0.0, 0.1, 0.05, 0.2])

    with pytest.raises(ValueError, match="non-decreasing"):
        smooth(timestamps, obs3d, obs2d, conf, INTRINSICS, _config())


def test_causal_tracker_rejects_decreasing_time() -> None:
    tracker = CausalTracker(_config(), INTRINSICS, n_keypoints=1)
    point = np.array([[0.0, 0.0, 0.6]])
    tracker.update(1.0, point, project(point, INTRINSICS), np.array([1.0]))
    with pytest.raises(ValueError, match="non-decreasing"):
        tracker.update(0.5, point, project(point, INTRINSICS), np.array([1.0]))


def test_smooth_returns_nan_before_initialisation() -> None:
    frame_count = 5
    timestamps = np.arange(frame_count) / 30.0
    obs3d = np.zeros((frame_count, 1, 3))
    obs3d[..., 2] = 0.6
    obs3d[:2] = np.nan
    obs2d = project(obs3d, INTRINSICS)
    conf = np.ones((frame_count, 1))

    track: Track = smooth(timestamps, obs3d, obs2d, conf, INTRINSICS, _config())

    assert np.all(np.isnan(track.pos[:2, 0, :]))
    assert not track.valid[:2, 0].any()
    assert track.valid[2:, 0].all()
    assert math.isfinite(float(track.pos[2, 0, 2]))


def test_rts_does_not_cross_a_reinitialisation_boundary() -> None:
    timestamps = np.arange(5, dtype=np.float64) / 30.0
    obs3d = np.full((5, 1, 3), np.nan)
    obs3d[0, 0] = [0.0, 0.0, 1.0]
    obs3d[3, 0] = [1.0, 0.0, 1.0]
    obs3d[4, 0] = [1.0, 0.0, 1.0]
    obs2d = project(obs3d, INTRINSICS)
    conf = np.ones((5, 1))

    track = smooth(timestamps, obs3d, obs2d, conf, INTRINSICS, _config(max_missed=1))

    # The first segment must be smoothed on its own: the later re-initialisation
    # at x = 1 m must not invent motion in frames 0-1.
    np.testing.assert_allclose(track.pos[:2, 0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(track.pos[:2, 0, 2], 1.0, atol=1e-12)
    np.testing.assert_allclose(track.pos[3:, 0, 0], 1.0, atol=1e-9)


def test_2d_mahalanobis_gate_rejects_inconsistent_pixels() -> None:
    frame_count = 5
    timestamps = np.arange(frame_count, dtype=np.float64) / 30.0
    truth = np.array([0.0, 0.0, 0.6])
    obs3d = np.full((frame_count, 1, 3), np.nan)
    obs3d[0, 0] = truth
    obs2d = np.tile(project(truth, INTRINSICS), (frame_count, 1, 1))
    obs2d[1, 0, 0] += 50.0
    obs2d[2, 0, 1] -= 50.0
    conf = np.ones((frame_count, 1))

    track = smooth(timestamps, obs3d, obs2d, conf, INTRINSICS, _config(accel_std=0.05))

    assert track.accepted[0, 0]
    assert track.gated[1, 0]
    assert track.gated[2, 0]
    assert not track.accepted[1, 0]
    np.testing.assert_allclose(track.pos[1, 0], truth, atol=2e-3)


def test_rts_rejects_depth_only_outliers() -> None:
    frame_count = 120
    dt = 1.0 / 30.0
    timestamps = np.arange(frame_count, dtype=np.float64) * dt
    truth = np.array([-0.1, 0.04, 0.65]) + timestamps[:, None] * np.array([0.08, -0.02, 0.0])
    rng = np.random.default_rng(31)
    observed = truth + rng.normal(0.0, 0.003, truth.shape)
    outlier_frames = np.array([15, 45, 75, 105])
    observed[outlier_frames, 2] += 0.5

    obs3d = observed[:, None, :]
    obs2d = project(obs3d, INTRINSICS)
    conf = np.full((frame_count, 1), 0.9)
    track = smooth(timestamps, obs3d, obs2d, conf, INTRINSICS, _config())

    assert track.gated[outlier_frames, 0].all()
    raw_error = np.linalg.norm(observed - truth, axis=1)
    smooth_error = np.linalg.norm(track.pos[:, 0, :] - truth, axis=1)
    raw_rmse = float(np.sqrt(np.mean(raw_error**2)))
    smooth_rmse = float(np.sqrt(np.mean(smooth_error**2)))
    assert smooth_rmse <= 0.5 * raw_rmse
    assert smooth_rmse < 0.01


def test_confidence_scales_measurement_covariance() -> None:
    point = np.array([[0.02, -0.01, 0.8]])
    low_tracker = CausalTracker(_config(), INTRINSICS, n_keypoints=1)
    high_tracker = CausalTracker(_config(), INTRINSICS, n_keypoints=1)
    low = low_tracker.update(0.0, point, project(point, INTRINSICS), np.array([0.01]))
    high = high_tracker.update(0.0, point, project(point, INTRINSICS), np.array([1.0]))

    np.testing.assert_allclose(low.cov[0, :3, :3], 100.0 * high.cov[0, :3, :3])


def test_lateral_covariance_uses_each_focal_length() -> None:
    intrinsics = Intrinsics(
        fx=800.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    point = np.array([[0.0, 0.0, 1.0]])
    tracker = CausalTracker(_config(), intrinsics, n_keypoints=1)

    state = tracker.update(0.0, point, project(point, intrinsics), np.array([1.0]))

    assert state.cov[0, 0, 0] == pytest.approx((2.0 * 1.0 / 800.0) ** 2)
    assert state.cov[0, 1, 1] == pytest.approx((2.0 * 1.0 / 400.0) ** 2)
    assert state.cov[0, 1, 1] / state.cov[0, 0, 0] == pytest.approx(4.0)


def test_smooth_tracks_through_nonzero_lens_distortion() -> None:
    intrinsics = Intrinsics(
        fx=600.0,
        fy=590.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        dist=(-0.12, 0.03, 0.001, -0.0006, 0.005),
    )
    frame_count = 60
    dt = 1.0 / 30.0
    timestamps = np.arange(frame_count, dtype=np.float64) * dt
    truth = np.array([-0.1, 0.04, 0.7]) + timestamps[:, None] * np.array([0.16, -0.03, 0.0])
    obs3d = np.full((frame_count, 1, 3), np.nan)
    obs3d[0, 0] = truth[0]
    obs2d = project(truth[:, None, :], intrinsics)
    conf = np.ones((frame_count, 1))

    track = smooth(timestamps, obs3d, obs2d, conf, intrinsics, _config())

    assert track.valid[:, 0].all()
    assert track.accepted[1:, 0].all()
    assert not track.gated[:, 0].any()
    lateral_error = np.linalg.norm(track.pos[:, 0, :2] - truth[:, :2], axis=1)
    assert float(np.max(lateral_error)) < 0.01
