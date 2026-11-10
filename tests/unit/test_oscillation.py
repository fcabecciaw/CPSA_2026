"""Periodicity, rocking and hand-flapping detection (FR-019, AC-010)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa.domain.model import HandShape, Side
from cpsa.vision.assessment import AnalysisConfig
from cpsa.vision.oscillation import (
    OscillationConfig,
    detect_flapping,
    detect_rocking,
    periodicity,
)
from tests.unit import vision_fixtures as fx

CFG = AnalysisConfig()
TIME = 10.0 + np.arange(180, dtype=np.float64) * fx.DT


def _rocking_frames(amplitude_fw: float = 0.15, frequency_hz: float = 1.0, seconds: float = 6.0):
    times = 10.0 + np.arange(int(seconds / fx.DT), dtype=np.float64) * fx.DT
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        offset = amplitude_fw * fx.FACE_WIDTH_PX * np.sin(2.0 * np.pi * frequency_hz * (t - 10.0))
        for name in ("left_shoulder", "right_shoulder"):
            xy[fx.LAYOUT.index(name), 0] += offset
        positions.append(xy)
    return fx.frames_from_positions(times, positions)


def _flapping_frames(
    frequency_hz: float = 3.0,
    amplitude_fw: float = 0.2,
    seconds: float = 2.0,
    offset_fw: float = 0.1,
):
    times = 10.0 + np.arange(int(seconds / fx.DT), dtype=np.float64) * fx.DT
    shoulder_y = fx.FACE_CENTER[1] + 1.6 * fx.FACE_WIDTH_PX
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        wrist_y = (
            shoulder_y
            + offset_fw * fx.FACE_WIDTH_PX
            + amplitude_fw * fx.FACE_WIDTH_PX * np.sin(2.0 * np.pi * frequency_hz * (t - 10.0))
        )
        fx.set_hand(
            xy,
            Side.RIGHT,
            (fx.FACE_CENTER[0] + 0.9 * fx.FACE_WIDTH_PX, wrist_y),
            HandShape.OPEN_PALM,
        )
        positions.append(xy)
    return fx.frames_from_positions(times, positions)


def test_periodicity_recovers_a_known_frequency() -> None:
    rng = np.random.default_rng(7)
    values = 0.2 * np.sin(2.0 * np.pi * 1.2 * (TIME - TIME[0])) + rng.normal(0.0, 0.01, TIME.size)

    frequency, strength = periodicity(TIME, values, (0.5, 2.0))

    assert frequency == pytest.approx(1.2, abs=0.05)
    assert strength > 0.85


def test_periodicity_rejects_an_aperiodic_signal() -> None:
    rng = np.random.default_rng(11)

    _, strength = periodicity(TIME, rng.normal(0.0, 1.0, TIME.size), (0.5, 2.0))

    assert strength < 0.3


def test_periodicity_of_constant_or_short_signals() -> None:
    assert periodicity(TIME, np.ones(TIME.size), (0.5, 2.0)) == (None, 0.0)
    assert periodicity(TIME[:3], np.ones(3), (0.5, 2.0)) == (None, 0.0)


def test_periodicity_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="band_hz"):
        periodicity(TIME, np.zeros(TIME.size), (2.0, 0.5))
    with pytest.raises(ValueError, match="same length"):
        periodicity(TIME, np.zeros(3), (0.5, 2.0))
    with pytest.raises(ValueError, match="increasing"):
        periodicity(TIME[::-1], np.zeros(TIME.size), (0.5, 2.0))


def test_rocking_is_detected_from_the_shoulder_midpoint() -> None:
    assert detect_rocking(_rocking_frames(), fx.LAYOUT, CFG) is True


def test_small_shoulder_wobble_is_not_rocking() -> None:
    assert detect_rocking(_rocking_frames(amplitude_fw=0.01), fx.LAYOUT, CFG) is False


def test_still_shoulders_are_not_rocking() -> None:
    frames = fx.frames_from_positions(TIME[:90], [fx.neutral_xy() for _ in TIME[:90]])

    assert detect_rocking(frames, fx.LAYOUT, CFG) is False


def test_layout_without_shoulders_is_not_rocking() -> None:
    full = fx.neutral_xy()
    body = full[: len(fx.BODY_NAMES)]
    frames = fx.frames_from_positions(TIME[:90], [body for _ in TIME[:90]])

    assert detect_rocking(frames, fx.WRIST_ONLY_LAYOUT, CFG) is False


def test_flapping_is_detected_near_shoulder_height() -> None:
    assert detect_flapping(_flapping_frames(), fx.LAYOUT, CFG) is True


def test_flapping_needs_the_wrist_near_the_shoulders() -> None:
    assert detect_flapping(_flapping_frames(offset_fw=3.0), fx.LAYOUT, CFG) is False


def test_slow_wrist_motion_is_not_flapping() -> None:
    assert detect_flapping(_flapping_frames(frequency_hz=1.0), fx.LAYOUT, CFG) is False


def test_bands_and_thresholds_are_configurable() -> None:
    configured = AnalysisConfig(
        oscillation=OscillationConfig(rocking_min_amp_fw=0.2, rocking_min_strength=0.5)
    )
    frames = _rocking_frames(amplitude_fw=0.15, frequency_hz=1.0)

    assert detect_rocking(frames, fx.LAYOUT, CFG) is True
    assert detect_rocking(frames, fx.LAYOUT, configured) is False
