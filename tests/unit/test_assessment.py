"""Stage-1 verdict, severity and stage-2 report (FR-015..FR-019, FR-023)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from cpsa.domain.model import FaceRegion, HandShape, Side, StereotypyLabel, Tag, Verdict
from cpsa.vision.assessment import (
    AnalysisConfig,
    assess_fast,
    build_report,
    severity_of,
    severity_of_approach,
)
from cpsa.vision.fusion import IMUSignal
from cpsa.vision.handface import HandFaceSample
from cpsa.vision.types import Contact, KeypointStatus
from tests.unit import vision_fixtures as fx

CFG = AnalysisConfig()
CHIN = fx.face_anchor("face_chin")
LEFT_EYE = fx.face_anchor("left_eye")


def _window(frames):
    return frames[0].t, frames[-1].t


def _contact(shape: HandShape = HandShape.FIST, anchor=CHIN, speed: float = 2.5):
    """A hand that closes on ``anchor`` and stays on it for a tenth of a second."""
    return fx.distance_profile(
        [
            (10.0, 1.0),
            (10.0 + 0.95 / speed, 0.05),
            (10.0 + 0.95 / speed + 0.1, 0.05),
            (10.0 + 0.95 / speed + 0.15, 1.2),
        ],
        side=Side.RIGHT,
        shape=shape,
        anchor=anchor,
    )


def _slow_cheek_touch():
    return fx.distance_profile(
        [
            (10.0, 1.0),
            (12.4, 0.28),
            (12.8, 0.12),
            (13.0, 0.10),
            (13.4, 0.10),
            (14.0, 1.0),
        ],
        side=Side.RIGHT,
        shape=HandShape.OPEN_PALM,
        anchor=CHIN,
    )


def _approach_only(shape: HandShape = HandShape.FIST, anchor=CHIN, speed: float = 1.5):
    """A hand that closes to 0.3 fw from ``anchor`` and retreats without touching."""
    reach = (1.0 - 0.3) / speed
    return fx.distance_profile(
        [
            (10.0, 1.0),
            (10.0 + reach, 0.3),
            (10.0 + reach + 0.2, 0.3),
            (10.0 + reach + 0.2 + 0.7 / speed, 1.0),
        ],
        side=Side.RIGHT,
        shape=shape,
        anchor=anchor,
    )


def _rejected_frames(seconds: float = 1.0):
    times = 10.0 + np.arange(int(seconds / fx.DT), dtype=np.float64) * fx.DT
    return fx.frames_from_positions(times, [fx.neutral_xy() for _ in times])


def _face_hidden(frames):
    status = np.full(fx.KEYPOINT_COUNT, int(KeypointStatus.MEASURED), dtype=np.int8)
    for name in (*fx.BODY_NAMES[:5], *fx.FACE_NAMES):
        status[fx.LAYOUT.index(name)] = int(KeypointStatus.LOST)
    hidden = []
    for frame in frames:
        hidden.append(
            fx.make_frame(
                frame.seq,
                frame.t,
                frame.xy,
                status=status,
                vel=frame.vel,
                acc=frame.acc,
                face_width_px=frame.face_width_px,
                framing=frame.framing,
            )
        )
    return hidden


# ---------------------------------------------------------------------------
# Stage 1: verdict and severity
# ---------------------------------------------------------------------------


def test_fist_to_the_eye_confirms_and_grades_dangerous() -> None:
    frames = _contact(HandShape.FIST, LEFT_EYE)
    t_from, t_to = _window(frames)

    assessment = assess_fast(frames, t_from, t_to, fx.LAYOUT, CFG)

    assert assessment.verdict is Verdict.CONFIRMED
    assert assessment.severity is Tag.DANGEROUS
    assert assessment.episode_id is None
    assert assessment.features["region_code"] == 1.0
    assert assessment.features["hand_shape_code"] == 1.0
    assert assessment.features["coverage_ok"] == 1.0
    assert assessment.features["coverage_end_t"] == pytest.approx(t_to)


def test_slow_open_palm_touch_on_the_cheek_is_non_dangerous() -> None:
    frames = _slow_cheek_touch()
    t_from, t_to = _window(frames)

    assessment = assess_fast(frames, t_from, t_to, fx.LAYOUT, CFG)

    assert assessment.verdict is Verdict.CONFIRMED
    assert assessment.severity is Tag.NON_DANGEROUS
    assert assessment.features["region_code"] == 2.0
    assert assessment.features["hand_shape_code"] == 3.0


@pytest.mark.parametrize(
    ("anchor", "shape", "expected"),
    [
        (CHIN, HandShape.FIST, Tag.DANGEROUS),
        (CHIN, HandShape.OPEN_PALM, Tag.NON_DANGEROUS),
    ],
)
def test_approach_only_confirmation_always_sets_a_severity(
    anchor, shape: HandShape, expected: Tag
) -> None:
    frames = _approach_only(shape=shape, anchor=anchor, speed=1.2)
    t_from, t_to = _window(frames)

    assessment = assess_fast(frames, t_from, t_to, fx.LAYOUT, CFG)

    assert assessment.verdict is Verdict.CONFIRMED
    assert assessment.severity is expected
    assert assessment.features["decel_fwps2"] == -1.0
    assert assessment.features["head_recoil_fw"] == -1.0


def test_hands_away_from_the_face_are_rejected() -> None:
    frames = _rejected_frames()
    t_from, t_to = _window(frames)

    assessment = assess_fast(frames, t_from, t_to, fx.LAYOUT, CFG)

    assert assessment.verdict is Verdict.REJECTED
    assert assessment.severity is None
    assert assessment.features["framed_fraction"] == 1.0
    assert assessment.features["min_distance_fw"] > CFG.contacts.contact_exit_fw


def test_hidden_face_is_undecided() -> None:
    frames = _face_hidden(_rejected_frames())
    t_from, t_to = _window(frames)

    assessment = assess_fast(frames, t_from, t_to, fx.LAYOUT, CFG)

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.severity is None
    assert assessment.features["framed_fraction"] == 0.0


def test_incomplete_coverage_is_undecided_not_rejected() -> None:
    frames = _rejected_frames()
    t_from = frames[0].t
    t_to = frames[-1].t + 0.5

    assessment = assess_fast(frames, t_from, t_to, fx.LAYOUT, CFG)

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.features["coverage_ok"] == 0.0
    assert assessment.features["coverage_end_t"] == pytest.approx(frames[-1].t)


def test_unhealthy_source_is_undecided() -> None:
    frames = _rejected_frames()
    t_from, t_to = _window(frames)

    assessment = assess_fast(frames, t_from, t_to, fx.LAYOUT, CFG, source_healthy=False)

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.features["coverage_ok"] == 0.0


def test_gap_in_the_analyzed_frames_is_undecided() -> None:
    frames = _rejected_frames(seconds=1.0)
    t_from, t_to = _window(frames)
    gapped = frames[:5] + frames[20:]

    assessment = assess_fast(gapped, t_from, t_to, fx.LAYOUT, CFG)

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.features["coverage_ok"] == 0.0


def test_uncovered_window_start_is_undecided() -> None:
    frames = _rejected_frames(seconds=1.0)
    t_from = frames[10].t + 0.25
    t_to = frames[-1].t
    gapped = frames[:5] + frames[25:]

    assessment = assess_fast(gapped, t_from, t_to, fx.LAYOUT, CFG)

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.features["coverage_ok"] == 0.0


def test_no_frames_is_undecided() -> None:
    assessment = assess_fast([], 10.0, 11.0, fx.LAYOUT, CFG)

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.features["coverage_ok"] == 0.0
    assert assessment.features["coverage_end_t"] == 10.0


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError, match="window"):
        assess_fast([], 11.0, 10.0, fx.LAYOUT, CFG)


def test_repeated_contacts_over_a_long_window_stay_within_the_fast_budget() -> None:
    frames = _repeated_contacts()

    started = time.perf_counter()
    assessment = assess_fast(frames, frames[0].t, frames[-1].t, fx.LAYOUT, CFG)
    elapsed = time.perf_counter() - started

    assert assessment.verdict is Verdict.CONFIRMED
    assert elapsed < 0.8


def test_features_expose_the_documented_codes() -> None:
    frames = _contact(HandShape.OPEN_PALM, CHIN)
    t_from, t_to = _window(frames)

    features = assess_fast(frames, t_from, t_to, fx.LAYOUT, CFG).features

    assert set(features) == {
        "min_distance_fw",
        "peak_closing_speed_fwps",
        "decel_fwps2",
        "occlusion_max",
        "head_recoil_fw",
        "region_code",
        "hand_shape_code",
        "framed_fraction",
        "coverage_end_t",
        "coverage_ok",
    }
    assert features["region_code"] == 2.0
    assert features["hand_shape_code"] == 3.0


# ---------------------------------------------------------------------------
# Severity rules
# ---------------------------------------------------------------------------


def _contact_sample(
    region: FaceRegion = FaceRegion.NOSE_CHEEK,
    shape: HandShape = HandShape.OPEN_PALM,
    peak: float | None = 0.2,
    decel: float | None = 0.5,
    recoil: float | None = 0.0,
) -> Contact:
    return Contact(
        side=Side.RIGHT,
        region=region,
        t_start=0.0,
        duration_s=0.1,
        min_distance_fw=0.1,
        peak_closing_speed_fwps=peak,
        decel_fwps2=decel,
        hand_shape=shape,
        head_recoil_fw=recoil,
        head_recoil_speed_fwps=0.0,
    )


@pytest.mark.parametrize(
    ("contact", "expected"),
    [
        (_contact_sample(region=FaceRegion.EYES, peak=0.1), Tag.DANGEROUS),
        (_contact_sample(shape=HandShape.FIST, peak=1.0), Tag.DANGEROUS),
        (
            _contact_sample(shape=HandShape.POINT_PINCH, peak=1.0),
            Tag.DANGEROUS,
        ),
        (_contact_sample(shape=HandShape.FIST, peak=0.5), Tag.NON_DANGEROUS),
        (_contact_sample(shape=HandShape.OPEN_PALM, peak=2.0), Tag.DANGEROUS),
        (_contact_sample(decel=4.0), Tag.DANGEROUS),
        (_contact_sample(recoil=0.05), Tag.DANGEROUS),
        (_contact_sample(peak=None, decel=None, recoil=None), Tag.NON_DANGEROUS),
    ],
)
def test_severity_of_contact_rules(contact: Contact, expected: Tag) -> None:
    assert severity_of(contact, CFG) is expected


def _approach_sample(
    distance: float = 0.3,
    closing: float | None = 1.2,
    region: FaceRegion = FaceRegion.NOSE_CHEEK,
) -> HandFaceSample:
    return HandFaceSample(
        t=0.0,
        side=Side.RIGHT,
        distance_fw=distance,
        closing_speed_fwps=closing,
        closing_accel_fwps2=None,
        approach_angle_deg=0.0,
        time_to_contact_s=None,
        occlusion_fraction=0.0,
        nearest_region=region,
    )


@pytest.mark.parametrize(
    ("samples", "shape", "expected"),
    [
        ([_approach_sample(region=FaceRegion.EYES)], HandShape.OPEN_PALM, Tag.DANGEROUS),
        ([_approach_sample(closing=1.0)], HandShape.FIST, Tag.DANGEROUS),
        ([_approach_sample(closing=0.5)], HandShape.FIST, Tag.NON_DANGEROUS),
        ([_approach_sample(closing=2.5)], HandShape.OPEN_PALM, Tag.DANGEROUS),
        ([_approach_sample(closing=None)], HandShape.OPEN_PALM, Tag.NON_DANGEROUS),
        ([], HandShape.UNKNOWN, Tag.NON_DANGEROUS),
    ],
)
def test_severity_of_approach_rules(
    samples: list[HandFaceSample], shape: HandShape, expected: Tag
) -> None:
    assert severity_of_approach(samples, shape, CFG) is expected


# ---------------------------------------------------------------------------
# Stage 2: episode report
# ---------------------------------------------------------------------------


def _repeated_contacts():
    return fx.distance_profile(
        [
            (0.0, 1.0),
            (0.2, 0.05),
            (0.3, 0.05),
            (0.4, 1.0),
            (1.2, 1.0),
            (1.4, 0.05),
            (1.5, 0.05),
            (1.6, 1.0),
            (2.4, 1.0),
            (2.6, 0.05),
            (2.7, 0.05),
            (2.8, 1.0),
        ],
        side=Side.RIGHT,
        shape=HandShape.FIST,
        anchor=CHIN,
    )


def test_report_counts_and_frequencies_of_repeated_contacts() -> None:
    frames = _repeated_contacts()

    report = build_report("ep-1", 0.0, frames, fx.LAYOUT, CFG, now=3.0)

    assert report.episode_id == "ep-1"
    assert report.contact_count == 3
    assert len(report.contacts) == 3
    assert report.repetition_hz == pytest.approx(1.0 / 1.2, rel=0.05)
    assert report.interval_cv == pytest.approx(0.0, abs=1e-6)
    assert report.speed_trend_fwps_per_s == pytest.approx(0.0, abs=1e-3)
    assert report.peak_severity is Tag.DANGEROUS
    assert StereotypyLabel.HAND_TO_FACE in report.labels
    assert report.active is True
    assert report.t_end == pytest.approx(2.8)


def test_report_activity_window_ends_the_episode() -> None:
    frames = _repeated_contacts()

    report = build_report("ep-1", 0.0, frames, fx.LAYOUT, CFG, now=10.0)

    assert report.active is False


def test_report_labels_rocking_without_raising_severity() -> None:
    times = 10.0 + np.arange(180, dtype=np.float64) * fx.DT
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        offset = 0.15 * fx.FACE_WIDTH_PX * np.sin(2.0 * np.pi * (t - 10.0))
        for name in ("left_shoulder", "right_shoulder"):
            xy[fx.LAYOUT.index(name), 0] += offset
        positions.append(xy)
    frames = fx.frames_from_positions(times, positions)

    report = build_report("ep-2", times[0], frames, fx.LAYOUT, CFG, now=times[-1])

    assert report.labels == frozenset({StereotypyLabel.ROCKING})
    assert report.peak_severity is None
    assert report.contact_count == 0
    assert report.active is True
    assert assess_fast(frames, times[0], times[-1], fx.LAYOUT, CFG).severity is None


def test_report_labels_hand_flapping() -> None:
    times = 10.0 + np.arange(60, dtype=np.float64) * fx.DT
    shoulder_y = fx.FACE_CENTER[1] + 1.6 * fx.FACE_WIDTH_PX
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        wrist_y = (
            shoulder_y
            + 0.1 * fx.FACE_WIDTH_PX
            + 0.2 * fx.FACE_WIDTH_PX * np.sin(2.0 * np.pi * 3.0 * (t - 10.0))
        )
        fx.set_hand(
            xy,
            Side.RIGHT,
            (fx.FACE_CENTER[0] + 0.9 * fx.FACE_WIDTH_PX, wrist_y),
            HandShape.OPEN_PALM,
        )
        positions.append(xy)
    frames = fx.frames_from_positions(times, positions)

    report = build_report("ep-3", times[0], frames, fx.LAYOUT, CFG, now=times[-1])

    assert StereotypyLabel.HAND_FLAPPING in report.labels
    assert report.peak_severity is None


def test_report_recovers_the_injected_clock_offset() -> None:
    lag = 0.12
    fw = fx.FACE_WIDTH_PX

    def signal(t: np.ndarray) -> np.ndarray:
        return 1.0 + 0.4 * np.sin(2.0 * np.pi * 1.3 * t) + 0.2 * np.sin(2.0 * np.pi * 3.1 * t + 0.4)

    times = 0.5 + np.arange(120, dtype=np.float64) * fx.DT
    base = fx.neutral_xy()
    frames = []
    for index, t in enumerate(times):
        acc = np.zeros((fx.KEYPOINT_COUNT, 2), dtype=np.float64)
        for side in (Side.LEFT, Side.RIGHT):
            acc[fx.LAYOUT.index(f"{side.value}_wrist")] = (float(signal(t)) * fw, 0.0)
        frames.append(fx.make_frame(index, float(t), base.copy(), acc=acc))
    imu_time = np.arange(0.0, 5.0, 0.01)
    # The IMU timestamp is ``lag`` seconds behind the physical event, so adding
    # ``lag`` maps it onto the camera clock.
    imu = IMUSignal(t_s=imu_time, acc_norm_g=signal(imu_time + lag))

    report = build_report("ep-4", float(times[0]), frames, fx.LAYOUT, CFG, now=5.0, imu=imu)

    assert report.clock_offset_s == pytest.approx(lag, abs=fx.DT)
    assert report.imu_consistency is not None
    assert report.imu_consistency > 0.8


def test_report_without_imu_has_no_clock_fields() -> None:
    frames = _rejected_frames()

    report = build_report("ep-5", 10.0, frames, fx.LAYOUT, CFG, now=11.0)

    assert report.clock_offset_s is None
    assert report.imu_consistency is None
    assert report.labels == frozenset()
    assert report.active is False
