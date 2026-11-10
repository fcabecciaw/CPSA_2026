"""IMU-camera clock offset, depth occlusion and hand-face contact detection."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.domain.model import FaceRegion, Side
from cpsa.vision.fusion import (
    ContactConfig,
    IMUSignal,
    detect_contacts,
    estimate_clock_offset,
    occluder_fraction,
)
from cpsa.vision.tracking import Track

DT = 1.0 / 30.0


def _signal(t: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 + 0.4 * np.sin(2.0 * np.pi * 1.3 * t) + 0.2 * np.sin(2.0 * np.pi * 3.1 * t + 0.4)


@pytest.mark.parametrize("lag", [-0.25, -0.1, 0.0, 0.1, 0.25])
def test_clock_offset_recovers_known_lag(lag: float) -> None:
    rng = np.random.default_rng(5)
    cam_t = np.arange(0.0, 4.0, 1.0 / 30.0)
    cam_values = _signal(cam_t) + rng.normal(0.0, 0.01, cam_t.size)
    imu_t = np.arange(-0.5, 4.5, 0.01)
    # The IMU timestamp is ``lag`` seconds behind the physical event, so adding
    # ``lag`` maps it onto the camera clock.
    imu_values = _signal(imu_t + lag) + rng.normal(0.0, 0.01, imu_t.size)

    offset, peak_corr = estimate_clock_offset(imu_t, imu_values, cam_t, cam_values)

    assert abs(offset - lag) <= 1.0 / 30.0
    assert peak_corr > 0.9


def test_clock_offset_handles_imu_arriving_late() -> None:
    cam_t = np.arange(0.0, 4.0, 1.0 / 30.0)
    cam_values = _signal(cam_t)
    imu_t = np.arange(-0.5, 4.5, 0.01)
    imu_values = _signal(imu_t - 0.12)

    offset, peak_corr = estimate_clock_offset(imu_t, imu_values, cam_t, cam_values)

    assert offset == pytest.approx(-0.12, abs=1.0 / 30.0)
    assert peak_corr > 0.9


def test_clock_offset_returns_zero_for_constant_signals() -> None:
    imu_t = np.arange(0.0, 2.0, 0.01)
    cam_t = np.arange(0.0, 2.0, 1.0 / 30.0)
    offset, peak_corr = estimate_clock_offset(
        imu_t, np.ones(imu_t.size), cam_t, np.ones(cam_t.size)
    )
    assert offset == 0.0
    assert peak_corr == 0.0


def test_clock_offset_rejects_too_short_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        estimate_clock_offset(
            np.array([0.0, 0.01, 0.02]),
            np.ones(3),
            np.array([0.0, 0.1, 0.2]),
            np.ones(3),
            max_lag_s=0.3,
        )


def test_occluder_fraction_counts_valid_pixels_in_front() -> None:
    depth = np.full((20, 20), 0.6)
    depth[2:7, 2:7] = 0.5

    fraction = occluder_fraction(depth, (2.0, 2.0, 12.0, 12.0), 0.6)

    assert fraction == pytest.approx(25.0 / 100.0)


def test_occluder_fraction_uses_margin_and_ignores_invalid_depth() -> None:
    depth = np.full((20, 20), 0.6)
    depth[2:7, 2:7] = 0.5
    depth[10, 10] = 0.0
    depth[3, 3] = 0.58  # only 2 cm in front: inside the 3 cm margin

    fraction = occluder_fraction(depth, (2.0, 2.0, 12.0, 12.0), 0.6)

    assert fraction == pytest.approx(24.0 / 99.0)


def test_occluder_fraction_is_zero_for_empty_or_invalid_box() -> None:
    depth = np.full((10, 10), 0.5)
    assert occluder_fraction(depth, (20.0, 20.0, 30.0, 30.0), 0.6) == 0.0
    assert occluder_fraction(np.zeros((10, 10)), (0.0, 0.0, 10.0, 10.0), 0.6) == 0.0


def _track(
    timestamps: NDArray[np.float64],
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64] | None = None,
    valid: NDArray[np.bool_] | None = None,
) -> Track:
    positions = np.asarray(positions, dtype=np.float64)
    frame_count, keypoint_count = positions.shape[:2]
    if velocities is None:
        if frame_count >= 2:
            velocities = np.gradient(positions, timestamps, axis=0, edge_order=1)
        else:
            velocities = np.zeros_like(positions)
    if valid is None:
        valid = np.ones((frame_count, keypoint_count), dtype=np.bool_)
    return Track(
        t=timestamps,
        pos=positions,
        vel=velocities,
        cov=np.zeros((frame_count, keypoint_count, 6, 6)),
        valid=valid,
        accepted=np.ones((frame_count, keypoint_count), dtype=np.bool_),
        gated=np.zeros((frame_count, keypoint_count), dtype=np.bool_),
        missed=np.zeros((frame_count, keypoint_count), dtype=np.int64),
    )


def _contact_scene(
    duration_s: float = 2.0,
    start_x: float = 0.2,
    velocity_x: float = -0.11,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], Track]:
    timestamps = np.arange(0.0, duration_s, DT)
    positions = np.zeros((timestamps.size, 1, 3))
    positions[:, 0, 0] = start_x + velocity_x * timestamps
    positions[:, 0, 2] = 0.6
    velocities = np.zeros_like(positions)
    velocities[:, 0, 0] = velocity_x
    return timestamps, positions, velocities, _track(timestamps, positions, velocities)


def _config(**overrides: float) -> ContactConfig:
    values: dict[str, float] = {
        "contact_distance_m": 0.05,
        "imu_peak_threshold_g": 0.5,
        "imu_peak_window_s": 0.15,
        "min_occluder_fraction": 0.05,
        "min_separation_s": 0.2,
        "approach_window_s": 0.3,
    }
    values.update(overrides)
    return ContactConfig(**values)  # type: ignore[arg-type]


def test_contact_present_reports_region_approach_impact_and_occluder() -> None:
    timestamps, positions, velocities, track = _contact_scene()
    anchor = np.tile(np.array([0.0, 0.0, 0.6]), (timestamps.size, 1))
    occluder = np.where(np.abs(positions[:, 0, 0]) <= 0.08, 0.25, 0.0)

    contact_time = 1.45
    imu_t = np.arange(-0.2, 2.3, 0.01) + 0.07
    imu_values = 1.0 + 1.2 * np.exp(-(((imu_t - 0.07 - contact_time) / 0.03) ** 2))
    imu = IMUSignal(t_s=imu_t, acc_norm_g=imu_values)

    events = detect_contacts(
        {Side.LEFT: track},
        {FaceRegion.EYES: anchor},
        imu,
        occluder,
        _config(),
        clock_offset_s=-0.07,
    )

    assert len(events) == 1
    event = events[0]
    assert event.side is Side.LEFT
    assert event.region is FaceRegion.EYES
    assert event.min_distance_m < 0.05
    assert event.approach_speed_mps == pytest.approx(0.11, abs=0.02)
    assert event.impact_peak_g == pytest.approx(1.2, abs=0.05)
    assert event.occluder is True
    assert event.t == pytest.approx(timestamps[np.argmax(np.abs(positions[:, 0, 0]) <= 0.05)])


def test_no_contact_when_hand_stays_away() -> None:
    timestamps, positions, velocities, track = _contact_scene(start_x=0.3, velocity_x=-0.075)
    anchor = np.tile(np.array([0.0, 0.0, 0.6]), (timestamps.size, 1))
    imu = IMUSignal(
        t_s=np.arange(-0.2, 2.3, 0.01),
        acc_norm_g=1.0 + 1.5 * np.exp(-(((np.arange(-0.2, 2.3, 0.01) - 1.4) / 0.03) ** 2)),
    )

    events = detect_contacts(
        {Side.LEFT: track},
        {FaceRegion.EYES: anchor},
        imu,
        None,
        _config(),
    )

    assert events == []


def test_imu_peak_without_vision_contact_produces_no_event() -> None:
    timestamps, positions, velocities, track = _contact_scene(start_x=0.3, velocity_x=-0.075)
    anchor = np.tile(np.array([0.0, 0.0, 0.6]), (timestamps.size, 1))
    imu_t = np.arange(-0.2, 2.3, 0.01)
    imu = IMUSignal(
        t_s=imu_t,
        acc_norm_g=1.0 + 2.0 * np.exp(-(((imu_t - 1.0) / 0.02) ** 2)),
    )

    events = detect_contacts(
        {Side.LEFT: track},
        {FaceRegion.EYES: anchor},
        imu,
        np.zeros(timestamps.size),
        _config(),
    )

    assert events == []


def test_contact_without_imu_or_occluder_has_no_optional_evidence() -> None:
    timestamps, positions, velocities, track = _contact_scene()
    anchor = np.tile(np.array([0.0, 0.0, 0.6]), (timestamps.size, 1))

    events = detect_contacts(
        {Side.LEFT: track},
        {FaceRegion.MOUTH: anchor},
        None,
        None,
        _config(),
    )

    assert len(events) == 1
    assert events[0].impact_peak_g is None
    assert events[0].occluder is False
    assert events[0].region is FaceRegion.MOUTH


def test_invalid_high_velocity_state_is_excluded_from_approach_speed() -> None:
    timestamps = np.arange(0.0, 3.0, DT)
    positions = np.zeros((timestamps.size, 1, 3))
    positions[:, 0, 0] = 0.2
    positions[:, 0, 2] = 0.6
    positions[10:12, 0, 0] = 0.04
    positions[20:22, 0, 0] = 0.01
    velocities = np.zeros_like(positions)
    velocities[:11, 0, 0] = -0.1
    velocities[11:, 0, 0] = -0.2
    valid = np.ones((timestamps.size, 1), dtype=np.bool_)
    valid[5, 0] = False
    positions[5, 0] = [0.6, 0.0, 0.6]
    velocities[5, 0] = [-100.0, 0.0, 0.0]
    track = _track(timestamps, positions, velocities, valid)
    anchor = np.tile(np.array([0.0, 0.0, 0.6]), (timestamps.size, 1))

    events = detect_contacts(
        {Side.LEFT: track},
        {FaceRegion.EYES: anchor},
        None,
        None,
        _config(min_separation_s=0.0),
    )

    assert len(events) == 2
    assert events[0].t == pytest.approx(timestamps[10])
    assert events[0].approach_speed_mps == pytest.approx(0.1, abs=0.02)


def test_min_separation_merges_evidence_into_the_existing_event() -> None:
    timestamps = np.arange(0.0, 3.0, DT)
    positions = np.zeros((timestamps.size, 1, 3))
    positions[:, 0, 0] = 0.2
    positions[:, 0, 2] = 0.6
    positions[10:12, 0, 0] = 0.04
    positions[20:22, 0, 0] = 0.01
    velocities = np.zeros_like(positions)
    velocities[:11, 0, 0] = -0.1
    velocities[11:, 0, 0] = -0.2
    track = _track(timestamps, positions, velocities)
    anchor = np.tile(np.array([0.0, 0.0, 0.6]), (timestamps.size, 1))

    imu_t = np.arange(-0.2, 3.2, 0.01)
    imu_values = (
        1.0
        + 0.8 * np.exp(-(((imu_t - timestamps[10]) / 0.02) ** 2))
        + 1.5 * np.exp(-(((imu_t - timestamps[20]) / 0.02) ** 2))
    )
    occluder = np.zeros(timestamps.size)
    occluder[20:22] = 0.2

    events = detect_contacts(
        {Side.LEFT: track},
        {FaceRegion.EYES: anchor},
        IMUSignal(t_s=imu_t, acc_norm_g=imu_values),
        occluder,
        _config(min_separation_s=0.5),
    )

    assert len(events) == 1
    event = events[0]
    assert event.t == pytest.approx(timestamps[10])
    assert event.region is FaceRegion.EYES
    assert event.min_distance_m == pytest.approx(0.01, abs=1e-6)
    assert event.approach_speed_mps == pytest.approx(0.2, abs=0.02)
    assert event.impact_peak_g == pytest.approx(1.5, abs=0.05)
    assert event.occluder is True


def test_min_separation_merges_a_second_episode() -> None:
    timestamps = np.arange(0.0, 3.0, DT)
    positions = np.zeros((timestamps.size, 1, 3))
    positions[:, 0, 0] = 0.2
    positions[15:20, 0, 0] = 0.0
    positions[26:31, 0, 0] = 0.0
    positions[:, 0, 2] = 0.6
    track = _track(timestamps, positions, velocities=np.zeros_like(positions))
    anchor = np.tile(np.array([0.0, 0.0, 0.6]), (timestamps.size, 1))

    merged = detect_contacts(
        {Side.LEFT: track},
        {FaceRegion.EYES: anchor},
        None,
        None,
        _config(min_separation_s=0.5),
    )
    separate = detect_contacts(
        {Side.LEFT: track},
        {FaceRegion.EYES: anchor},
        None,
        None,
        _config(min_separation_s=0.0),
    )

    assert len(merged) == 1
    assert len(separate) == 2


@pytest.mark.parametrize("lag", [-0.117, 0.083])
def test_clock_offset_refines_between_resampling_samples(lag: float) -> None:
    rng = np.random.default_rng(17)
    cam_t = np.arange(0.0, 6.0, 1.0 / 30.0)
    cam_values = _signal(cam_t) + rng.normal(0.0, 0.005, cam_t.size)
    imu_t = np.arange(-0.5, 6.5, 1.0 / 200.0)
    imu_values = _signal(imu_t + lag) + rng.normal(0.0, 0.005, imu_t.size)

    offset, peak_corr = estimate_clock_offset(imu_t, imu_values, cam_t, cam_values)

    # One camera frame is 1/30 s; the parabolic refinement should be much finer.
    assert offset == pytest.approx(lag, abs=0.25 / 30.0)
    assert peak_corr > 0.9


def test_anchors_accept_a_list_of_region_arrays() -> None:
    timestamps, positions, velocities, track = _contact_scene()
    anchor = np.tile(np.array([0.0, 0.0, 0.6]), (timestamps.size, 1))

    events = detect_contacts(
        {Side.LEFT: track},
        [(FaceRegion.NOSE_CHEEK, anchor)],
        None,
        None,
        _config(),
    )

    assert len(events) == 1
    assert events[0].region is FaceRegion.NOSE_CHEEK


def test_impact_peak_below_threshold_is_not_reported() -> None:
    timestamps, positions, velocities, track = _contact_scene()
    anchor = np.tile(np.array([0.0, 0.0, 0.6]), (timestamps.size, 1))
    imu = IMUSignal(
        t_s=np.arange(-0.2, 2.3, 0.01),
        acc_norm_g=1.0 + 0.1 * np.ones(250),
    )

    events = detect_contacts(
        {Side.LEFT: track},
        {FaceRegion.EYES: anchor},
        imu,
        None,
        _config(),
    )

    assert len(events) == 1
    assert events[0].impact_peak_g is None
