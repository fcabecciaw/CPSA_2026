"""The keypoint stabilizer: jitter, motion, dropouts, hysteresis and cost."""

from __future__ import annotations

import time

import numpy as np
import pytest

from cpsa.config import SmoothingConfig
from cpsa.keypoints import OUTLIER_FRACTION, RESET_AFTER_S, KeypointStabilizer
from cpsa.movenet import KEYPOINTS, MovenetPose
from cpsa.overlay import _skeleton_points

NUM = len(KEYPOINTS)
FRAME = (640, 480)
DIAGONAL = float(np.hypot(*FRAME))
PERIOD = 1.0 / 22.0
"""The board decodes MoveNet at about 22 Hz."""
WRIST = KEYPOINTS.index("left_wrist")
STILL = np.array(
    [[100.0 + 20.0 * (index % 4), 100.0 + 30.0 * (index // 4)] for index in range(NUM)]
)
"""A plausible standing body: every joint a fixed pixel, both shoulders 20 px apart."""


def smoothing(**overrides: object) -> SmoothingConfig:
    settings: dict[str, object] = {
        "enabled": True,
        "min_cutoff": 1.0,
        "beta": 0.02,
        "d_cutoff": 1.0,
        "deadband_px": 1.5,
        "show_score": 0.35,
        "hide_score": 0.25,
        "show_frames": 2,
        "hide_frames": 4,
    }
    settings.update(overrides)
    return SmoothingConfig(**settings)  # type: ignore[arg-type]


def pose(
    points: object = None,
    *,
    score: float = 0.9,
    scores: object = None,
) -> MovenetPose:
    """Build a pose from ``(17, 2)`` points, with one score or one per joint."""
    coordinates = STILL if points is None else np.asarray(points, dtype=np.float64)
    keypoints = np.empty((NUM, 3), dtype=np.float32)
    keypoints[:, :2] = coordinates
    keypoints[:, 2] = 0.9 if scores is None else scores
    return MovenetPose(keypoints=keypoints, score=score)


def scores_for(index: int, value: float, *, base: float = 0.9) -> np.ndarray:
    """One joint at ``value``, the rest confident."""
    scores = np.full(NUM, base, dtype=np.float64)
    scores[index] = value
    return scores


class Driver:
    """Feed frames one period apart, as the camera does."""

    def __init__(self, stabilizer: KeypointStabilizer, *, start: float = 0.0) -> None:
        self.stabilizer = stabilizer
        self.now = start

    def push(self, frame: MovenetPose) -> MovenetPose:
        self.now += PERIOD
        return self.stabilizer.update(frame, self.now, size=FRAME)

    def push_all(self, frames: object) -> list[MovenetPose]:
        return [self.push(frame) for frame in frames]  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #


def test_a_still_body_publishes_pixel_identical_keypoints() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)
    rng = np.random.default_rng(7)
    jittered = [pose(STILL + rng.uniform(-0.4, 0.4, STILL.shape)) for _ in range(30)]

    outputs = driver.push_all(jittered)

    shown = outputs[2:]
    assert all(item.keypoints[WRIST, 2] >= 0.35 for item in shown)
    reference = shown[0].keypoints[:, :2].copy()
    for item in shown[1:]:
        np.testing.assert_array_equal(item.keypoints[:, :2], reference)
    # Sub-pixel jitter is invisible but the joints still sit on the body.
    assert np.abs(reference - STILL).max() < 1.0
    assert not np.array_equal(jittered[0].keypoints[:, :2], jittered[-1].keypoints[:, :2])


def test_a_forty_pixel_swing_per_frame_is_followed_within_a_frame() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)
    driver.push_all([pose()] * 4)
    swing = []
    for step in range(1, 7):
        points = STILL.copy()
        points[WRIST, 0] += 40.0 * step
        swing.append(pose(points))

    outputs = driver.push_all(swing)

    published = [float(item.keypoints[WRIST, 0]) for item in outputs]
    truth = [STILL[WRIST, 0] + 40.0 * step for step in range(1, 7)]
    # "One frame of lag" is one swing step: the drawn wrist is always closer to
    # the current pose than to the previous one.
    for drawn, current, previous in zip(
        published, truth, [STILL[WRIST, 0], *truth[:-1]], strict=True
    ):
        assert abs(drawn - current) < abs(drawn - previous)
    assert published[0] > STILL[WRIST, 0] + 10.0


def test_a_fast_vertical_drop_is_followed_too() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)
    driver.push_all([pose()] * 4)
    ankle = KEYPOINTS.index("left_ankle")
    points = STILL.copy()
    for step in range(1, 5):
        points[ankle, 1] = STILL[ankle, 1] + 8.0 * step
        driver.push(pose(points))

    published = driver.stabilizer.update(pose(points), driver.now + PERIOD, size=FRAME)

    assert abs(float(published.keypoints[ankle, 1]) - (STILL[ankle, 1] + 32.0)) < 8.0


# --------------------------------------------------------------------------- #
# Dropouts and hysteresis
# --------------------------------------------------------------------------- #


def test_a_short_dropout_holds_the_last_position() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)
    driver.push_all([pose()] * 4)
    drifted = STILL.copy()
    drifted[WRIST, 0] += 10.0  # the decoder wanders off while the joint is weak
    weak = scores_for(WRIST, 0.1)

    held = driver.push_all([pose(drifted, scores=weak)] * 3)

    for item in held:
        assert item.keypoints[WRIST, 2] >= 0.35
        np.testing.assert_array_equal(item.keypoints[WRIST, :2], STILL[WRIST])
    gone = driver.push(pose(drifted, scores=weak))
    assert gone.keypoints[WRIST, 2] == 0.0
    np.testing.assert_array_equal(gone.keypoints[WRIST, :2], STILL[WRIST])
    assert not stabilizer.visible[WRIST]
    # The bones to that joint go with it.
    _, drawable = _skeleton_points(gone.keypoints, *FRAME)
    assert not drawable[WRIST]
    assert drawable.sum() == NUM - 1


def test_a_joint_appears_after_two_confident_frames() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)

    first = driver.push(pose())
    second = driver.push(pose())

    assert first.keypoints[WRIST, 2] == 0.0
    assert second.keypoints[WRIST, 2] >= 0.35
    assert stabilizer.visible.all()
    _, drawable = _skeleton_points(second.keypoints, *FRAME)
    assert drawable.all()


def test_a_score_inside_the_hysteresis_band_keeps_a_joint_shown() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)
    driver.push_all([pose()] * 4)
    band = scores_for(WRIST, 0.3)

    outputs = driver.push_all([pose(scores=band)] * 8)

    for item in outputs:
        assert item.keypoints[WRIST, 2] >= 0.35
    assert stabilizer.visible[WRIST]


def test_a_joint_that_never_reaches_the_show_score_never_appears() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)

    outputs = driver.push_all([pose(scores=scores_for(WRIST, 0.3))] * 10)

    assert all(item.keypoints[WRIST, 2] == 0.0 for item in outputs)
    assert not stabilizer.visible[WRIST]


def test_the_whole_skeleton_needs_two_frames_and_survives_a_short_dip() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)

    weak = pose(scores=np.full(NUM, 0.2), score=0.2)
    first = driver.push(weak)
    assert first.score == 0.0
    assert not stabilizer.present

    present = driver.push_all([pose()] * 2)
    assert stabilizer.present
    assert present[-1].score >= 0.35

    kept = driver.push_all([weak] * 3)
    for item in kept:
        assert item.score >= 0.34  # float32 of 0.35
        assert item.keypoints[:, 2].min() >= 0.34
    assert stabilizer.present

    gone = driver.push(weak)
    assert not stabilizer.present
    assert gone.score == 0.0
    assert not gone.keypoints[:, 2].any()


# --------------------------------------------------------------------------- #
# Resets
# --------------------------------------------------------------------------- #


def test_a_model_swap_restarts_the_filters() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)
    driver.push_all([pose()] * 6)

    stabilizer.reset()  # a model swap

    assert not stabilizer.present
    assert not stabilizer.visible.any()
    shifted = STILL + 1.0  # well inside the old deadband
    outputs = driver.push_all([pose(shifted)] * 2)
    np.testing.assert_array_equal(outputs[-1].keypoints[:, :2], shifted.astype(np.float32))


def test_the_filters_restart_when_the_person_returns_after_a_second() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)
    driver.push_all([pose()] * 6)
    absent = pose(scores=np.zeros(NUM), score=0.0)
    driver.push_all([absent] * int((RESET_AFTER_S + 0.5) / PERIOD))

    shifted = STILL + 1.0
    outputs = driver.push_all([pose(shifted)] * 2)

    np.testing.assert_array_equal(outputs[-1].keypoints[:, :2], shifted.astype(np.float32))


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #


def test_a_one_frame_outlier_is_held_and_discarded() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)
    driver.push_all([pose()] * 4)
    spike = STILL.copy()
    spike[WRIST, 0] += OUTLIER_FRACTION * DIAGONAL + 40.0

    held = driver.push(pose(spike))
    back = driver.push(pose())

    np.testing.assert_array_equal(held.keypoints[WRIST, :2], STILL[WRIST])
    np.testing.assert_array_equal(back.keypoints[WRIST, :2], STILL[WRIST])
    # Only the jumping joint is held: the rest of the body is untouched.
    np.testing.assert_array_equal(held.keypoints[0, :2], STILL[0])


def test_a_jump_confirmed_by_a_second_frame_is_accepted_at_once() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    driver = Driver(stabilizer)
    driver.push_all([pose()] * 4)
    far = STILL.copy()
    far[WRIST, 0] += OUTLIER_FRACTION * DIAGONAL + 40.0

    held = driver.push(pose(far))
    snapped = driver.push(pose(far))

    np.testing.assert_array_equal(held.keypoints[WRIST, :2], STILL[WRIST])
    np.testing.assert_array_equal(snapped.keypoints[WRIST, :2], far[WRIST].astype(np.float32))


# --------------------------------------------------------------------------- #
# Disabled, costs
# --------------------------------------------------------------------------- #


def test_disabled_publishes_the_raw_pose() -> None:
    stabilizer = KeypointStabilizer(smoothing(enabled=False))
    raw = pose()

    assert stabilizer.update(raw, 0.0, size=FRAME) is raw
    assert stabilizer.update(pose(), PERIOD, size=FRAME) is not raw


@pytest.mark.benchmark
def test_a_frame_costs_less_than_the_skeleton_budget() -> None:
    stabilizer = KeypointStabilizer(smoothing())
    rng = np.random.default_rng(0)
    frames = [pose(STILL + rng.uniform(-20.0, 20.0, STILL.shape)) for _ in range(60)]
    for index, frame in enumerate(frames):  # warm the filters and the allocator
        stabilizer.update(frame, index * PERIOD, size=FRAME)

    started = time.perf_counter()
    for index in range(600):
        stabilizer.update(frames[index % len(frames)], PERIOD * (100 + index), size=FRAME)
    per_frame_ms = (time.perf_counter() - started) / 600.0 * 1000.0

    assert per_frame_ms < 0.3
