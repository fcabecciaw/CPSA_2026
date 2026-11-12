"""The still gate: one cheap motion score in front of the DPU (F-S6)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.motion import MOTION_SIZE, STILL_FRAMES, StillGate, small_gray

WIDTH, HEIGHT = 640, 480


def frame(value: int = 90, *, noise: float = 0.0, seed: int = 0) -> NDArray[np.uint8]:
    """One 640x480 RGB frame, flat or with gaussian sensor noise."""
    rgb = np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)
    if noise:
        generator = np.random.default_rng(seed)
        noisy = rgb.astype(np.float64) + generator.normal(0.0, noise, (HEIGHT, WIDTH, 1))
        rgb = np.clip(noisy, 0, 255).astype(np.uint8)
    return rgb


def gate(motion_thr: float = 1.0, force_every_s: float = 0.5) -> StillGate:
    return StillGate(motion_thr, force_every_s)


def test_the_grey_image_is_the_downsampled_frame() -> None:
    small = small_gray(frame(90))

    assert small.shape == (MOTION_SIZE[1], MOTION_SIZE[0])
    assert small.dtype == np.uint8


def test_the_first_frame_is_always_inferred() -> None:
    still = gate()

    assert still.update(frame(90), 0.0) is True
    assert still.score == float("inf")


def test_the_score_is_the_mean_absolute_difference_in_grey_levels() -> None:
    still = gate()
    still.update(frame(0), 0.0)

    still.update(frame(10), 0.02)

    assert still.score == pytest.approx(10.0)


def test_three_still_frames_then_the_gate_skips() -> None:
    still = gate()
    decisions = [still.update(frame(90), 0.02 * index) for index in range(6)]

    assert decisions[:STILL_FRAMES] == [True] * STILL_FRAMES
    assert decisions[STILL_FRAMES:] == [False, False, False]
    assert still.still_frames == 5
    assert still.score == 0.0


def test_motion_is_inferred_at_once_and_resets_the_count() -> None:
    still = gate()
    for index in range(5):
        still.update(frame(90), 0.02 * index)
    assert still.update(frame(90), 0.1) is False

    assert still.update(frame(200), 0.12) is True
    assert still.still_frames == 0
    assert still.score > 1.0


def test_a_still_scene_still_gets_a_forced_detection() -> None:
    still = gate(force_every_s=0.5)
    decisions = [still.update(frame(90), 0.02 * index) for index in range(5)]

    assert decisions[-1] is False

    # Half a second after the last real detection the next frame goes through.
    assert still.update(frame(90), 0.62) is True
    assert still.update(frame(90), 0.64) is False


def test_sensor_noise_on_a_still_scene_counts_as_still() -> None:
    still = gate(motion_thr=1.0)

    for index in range(6):
        decision = still.update(frame(90, noise=2.0, seed=index), 0.02 * index)

    assert still.score < 1.0
    assert decision is False


def test_a_gate_with_a_tighter_threshold_calls_the_same_noise_motion() -> None:
    still = gate(motion_thr=0.1)
    still.update(frame(90), 0.0)

    assert still.update(frame(90, noise=2.0, seed=7), 0.02) is True


def test_reset_forgets_the_reference_frame() -> None:
    still = gate()
    still.update(frame(90), 0.0)
    still.update(frame(90), 0.02)
    still.reset()

    assert still.score == float("inf")
    assert still.update(frame(90), 10.0) is True


@pytest.mark.benchmark
def test_one_motion_score_costs_a_fraction_of_a_millisecond() -> None:
    """Host micro-benchmark of the gate: two resizes, one small absolute difference.

    The bound is wall clock, not ``process_time``: OpenCV's resize runs its box
    filter on several threads, so the CPU time of one call is a multiple of the
    time the loop actually waits for it.
    """

    still = gate()
    frames = [frame(90, noise=2.0, seed=index) for index in range(20)]
    for rgb in frames:
        still.update(rgb, 0.0)

    samples = []
    for index in range(300):
        started = time.perf_counter()
        still.update(frames[index % len(frames)], 0.02 * index)
        samples.append((time.perf_counter() - started) * 1000.0)

    per_frame_ms = min(samples)
    assert per_frame_ms < 0.5, f"{per_frame_ms:.3f} ms per motion score"
