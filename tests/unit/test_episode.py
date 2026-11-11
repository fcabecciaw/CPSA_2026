"""Episode state, frame selection and the P-04 follow-up rules (FR-002, FR-003, FR-015..FR-020)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np
import pytest

import cpsa.vision.episode as episode_module
from cpsa.domain.model import HandShape, Side, StereotypyLabel, Tag, Verdict
from cpsa.vision.assessment import AnalysisConfig, severity_of
from cpsa.vision.contacts2d import detect_contacts
from cpsa.vision.episode import EpisodeAnalyzer, select_frames
from cpsa.vision.handface import hand_face_samples as hand_face_samples_impl
from cpsa.vision.tracker2d import Tracker2DConfig
from cpsa.vision.types import KeypointStatus, PoseFrame, TrackedFrame
from tests.unit import vision_fixtures as fx

TRACKER = Tracker2DConfig()
ANALYSIS = AnalysisConfig()
EYE = fx.face_anchor("left_eye")
CHIN = fx.face_anchor("face_chin")


class _Timed:
    def __init__(self, t: float) -> None:
        self.t = t


class _ScriptedTracker:
    """Return the prepared tracked frames in order, ignoring the pose input."""

    def __init__(self, frames: Sequence[TrackedFrame]) -> None:
        self._frames = list(frames)
        self._index = 0
        self.resets = 0

    def reset(self) -> None:
        self._index = 0
        self.resets += 1

    def update(self, _frame: PoseFrame) -> TrackedFrame:
        tracked = self._frames[self._index]
        self._index += 1
        return tracked


def _pose(seq: int, t: float) -> PoseFrame:
    return PoseFrame(seq, t, 640, 480, ())


def _with_contact(
    times: np.ndarray,
    window: tuple[float, float],
    anchor: np.ndarray,
    shape: HandShape,
    keyframes: Sequence[tuple[float, float]],
) -> list[TrackedFrame]:
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        if window[0] <= t <= window[1]:
            distance = float(
                np.interp(t, [item[0] for item in keyframes], [item[1] for item in keyframes])
            )
            fx.set_hand_distance(xy, Side.RIGHT, anchor, distance, shape, direction=(0, 1))
        positions.append(xy)
    return fx.frames_from_positions(times, positions)


def _episode_frames() -> list[TrackedFrame]:
    """One trajectory: a fist to the eye, then a slow open palm on the chin."""
    times = np.arange(9.0, 21.0, fx.DT, dtype=np.float64)
    frames = _with_contact(
        times,
        (9.3, 10.8),
        EYE,
        HandShape.FIST,
        [(9.3, 1.0), (9.5, 0.05), (10.5, 0.05), (10.8, 1.2)],
    )
    second = _with_contact(
        times,
        (11.6, 13.8),
        CHIN,
        HandShape.OPEN_PALM,
        [(11.6, 1.0), (12.2, 0.10), (12.9, 0.10), (13.4, 1.2)],
    )
    # Keep the neutral frames outside the second contact and the second-contact
    # frames inside its window.
    merged = []
    for index, t in enumerate(times):
        merged.append(second[index] if 11.6 <= t <= 13.8 else frames[index])
    return merged


def _analyzer(
    frames: Sequence[TrackedFrame], *, start: float = 9.2, until: float | None = None
) -> EpisodeAnalyzer:
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=_ScriptedTracker(frames))
    analyzer.start("ep-1", start)
    for frame in frames:
        if frame.t >= start and (until is None or frame.t <= until):
            analyzer.push(_pose(frame.seq, frame.t))
    return analyzer


def _push_frames(analyzer: EpisodeAnalyzer, frames: Sequence[TrackedFrame]) -> None:
    for frame in frames:
        analyzer.push(_pose(frame.seq, frame.t))


def _long_held_contact_frames(start: float = 9.2, stop: float = 15.0) -> list[TrackedFrame]:
    """One long-held chin contact whose distance and shape change over time."""
    times = np.arange(start, stop, fx.DT, dtype=np.float64)
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        if t >= start + 0.3:
            phase = min(t - (start + 0.3), 5.0)
            distance = 0.05 + 0.02 * phase
            shape = HandShape.FIST if phase < 2.5 else HandShape.OPEN_PALM
            fx.set_hand_distance(xy, Side.RIGHT, CHIN, distance, shape, direction=(0, 1))
        positions.append(xy)
    return fx.frames_from_positions(times, positions)


def _hide_right_hand(frame: TrackedFrame) -> TrackedFrame:
    """Make the right hand invisible without moving it, ending a held contact."""
    hand = np.asarray(fx.LAYOUT.hand(Side.RIGHT), dtype=np.int64)
    wrist = np.asarray([fx.LAYOUT.index("right_wrist")], dtype=np.int64)
    conf = np.array(frame.conf, dtype=np.float64)
    status = np.array(frame.status, dtype=np.int8)
    conf[hand] = 0.0
    conf[wrist] = 0.0
    status[hand] = int(KeypointStatus.LOST)
    status[wrist] = int(KeypointStatus.LOST)
    return dataclasses.replace(frame, conf=conf, status=status)


def _missing_sample_mid_contact() -> list[TrackedFrame]:
    """A held chin contact whose right-hand sample disappears mid-run (R-15).

    ``detect_contacts`` grades per-side sample runs, so the missing samples are
    bridged and the contact stays one run; the incremental state must match.
    """
    frames = _long_held_contact_frames(9.2, 12.0)
    return [
        _hide_right_hand(frame) if 10.1 <= frame.t <= 10.2 + 1e-9 else frame for frame in frames
    ]


def _long_contact_then_neutral() -> list[TrackedFrame]:
    """A 7.2 s chin contact followed by neutral frames covering its end window (R-14)."""
    times = np.arange(9.0, 22.0, fx.DT, dtype=np.float64)
    return _with_contact(
        times,
        (10.0, 17.2),
        CHIN,
        HandShape.FIST,
        [(9.95, 1.0), (10.0, 0.05), (17.2, 0.05), (17.25, 1.2)],
    )


def _short_contact_with_late_recoil(*, recoil: bool = True) -> list[TrackedFrame]:
    """A 0.2 s open-palm chin contact whose head recoil lands after it ended (R-10).

    The approach is slow enough to stay below the approach-speed rule and it
    ends because the hand leaves the face, so only the late head recoil can
    make the contact ``DANGEROUS``.
    """
    keyframes = [(9.7, 0.5), (10.3, 0.1), (10.45, 0.1)]
    times = np.arange(9.2, 11.0, fx.DT, dtype=np.float64)
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        if t >= 9.7:
            distance = float(
                np.interp(t, [key[0] for key in keyframes], [key[1] for key in keyframes])
            )
            fx.set_hand_distance(
                xy, Side.RIGHT, CHIN, distance, HandShape.OPEN_PALM, direction=(0, 1)
            )
        if recoil and t >= 10.5:
            fx.shift_face(xy, (0.0, 0.15))
        positions.append(xy)
    frames = fx.frames_from_positions(times, positions)
    return [frame if frame.t < 10.45 - 1e-9 else _hide_right_hand(frame) for frame in frames]


# ---------------------------------------------------------------------------
# P-06 frame selection
# ---------------------------------------------------------------------------


def test_select_frames_keeps_the_first_frame_at_or_after_each_grid_point() -> None:
    frames = [_Timed(t) for t in (0.0, 0.03, 0.11, 0.19, 0.21)]

    selected = select_frames(frames, 10.0, t0=0.0)

    assert [frame.t for frame in selected] == [0.0, 0.11, 0.21]


def test_select_frames_defaults_to_the_first_timestamp_and_sorts() -> None:
    frames = [_Timed(t) for t in (10.25, 10.0, 10.05, 10.15, 10.35)]

    selected = select_frames(frames, 5.0)

    assert [frame.t for frame in selected] == [10.0, 10.25]


def test_select_frames_keeps_a_frame_at_most_once() -> None:
    frames = [_Timed(t) for t in (0.0, 0.0, 0.0, 0.5)]

    selected = select_frames(frames, 100.0, t0=0.0)

    assert len(selected) == 2


def test_select_frames_is_empty_for_no_frames() -> None:
    assert select_frames([], 10.0) == []


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_select_frames_rejects_invalid_rates(fps: float) -> None:
    with pytest.raises(ValueError):
        select_frames([_Timed(0.0)], fps)


def test_select_frames_rejects_a_non_finite_start() -> None:
    with pytest.raises(ValueError):
        select_frames([_Timed(0.0)], 10.0, t0=float("nan"))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_analyzer_rejects_use_before_start_and_after_end() -> None:
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS)
    with pytest.raises(RuntimeError):
        analyzer.push(_pose(0, 0.0))
    with pytest.raises(RuntimeError):
        analyzer.assess(0.0, 1.0)
    with pytest.raises(RuntimeError):
        analyzer.report(0.0)

    frames = _episode_frames()
    analyzer = _analyzer(frames)
    analyzer.end()
    with pytest.raises(RuntimeError):
        analyzer.push(_pose(0, frames[-1].t + 0.1))
    with pytest.raises(RuntimeError):
        analyzer.assess(9.2, 10.0)
    assert not analyzer.active(10.0)


def test_analyzer_rejects_out_of_order_frames() -> None:
    frames = _episode_frames()
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=_ScriptedTracker(frames))
    analyzer.start("ep-1", 9.2)
    analyzer.push(_pose(0, 10.0))
    with pytest.raises(ValueError):
        analyzer.push(_pose(1, 9.9))


def test_start_resets_the_tracker_and_rejects_empty_ids() -> None:
    tracker = _ScriptedTracker(_episode_frames())
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=tracker)
    analyzer.start("ep-1", 1.0)
    analyzer.start("ep-2", 2.0)
    assert tracker.resets == 2
    with pytest.raises(ValueError):
        analyzer.start("", 3.0)
    with pytest.raises(ValueError):
        analyzer.start("ep-3", float("nan"))


def test_analyzer_exposes_frames_and_grade() -> None:
    frames = _episode_frames()
    analyzer = _analyzer(frames)

    assert analyzer.episode_id == "ep-1"
    assert analyzer.t_start == 9.2
    assert analyzer.started
    assert len(analyzer.frames) == len([frame for frame in frames if frame.t >= 9.2])
    # Pushed frames grade themselves; the pushed episode ends with the mild contact.
    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[1] is Tag.NON_DANGEROUS


# ---------------------------------------------------------------------------
# F17: an episode window over one running stream (plan S)
# ---------------------------------------------------------------------------


def _stream(
    frames: Sequence[TrackedFrame], start: float = 9.2
) -> tuple[EpisodeAnalyzer, _ScriptedTracker]:
    """Start an idle stream and push exactly ``frames`` through it, in order."""
    tracker = _ScriptedTracker(frames)
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=tracker)
    analyzer.start("stream-1", start)
    _push_frames(analyzer, frames)
    return analyzer, tracker


def test_open_window_reports_only_events_after_its_start() -> None:
    """F17: a window excludes the events of the idle stream before ``t_start``."""
    frames = [frame for frame in _episode_frames() if 9.2 <= frame.t <= 11.55]
    analyzer, tracker = _stream(frames)
    # Contact 1 (9.3..10.8 s) still keeps the idle stream active at 11.55 s.
    assert analyzer.active(11.55)
    assert analyzer.contact_peak is not None
    resets = tracker.resets

    analyzer.open_window("ep-1", 11.5)

    assert analyzer.window_open
    assert analyzer.episode_id == "ep-1"
    assert analyzer.t_start == 11.5
    assert tracker.resets == resets  # the window keeps the tracker
    assert analyzer.contact_events == ()
    assert analyzer.contact_peak is None
    assert not analyzer.active(11.55)
    assessment = analyzer.assess(11.5, 11.55, now=11.55)
    assert assessment.verdict is not Verdict.CONFIRMED
    assert "from_episode" not in assessment.features
    report = analyzer.report(11.55)
    assert (report.episode_id, report.t_start) == ("ep-1", 11.5)
    assert all(contact.t_start >= report.t_start for contact in report.contacts)

    analyzer.close_window()

    assert not analyzer.window_open
    assert analyzer.episode_id == "stream-1"
    assert analyzer.t_start == 9.2
    assert analyzer.active(11.55)  # the stream still owns the old event


def test_open_window_requires_an_active_stream_and_no_open_window() -> None:
    frames = [frame for frame in _episode_frames() if frame.t <= 10.0]
    tracker = _ScriptedTracker(frames)
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=tracker)
    with pytest.raises(RuntimeError):
        analyzer.open_window("ep-1", 9.2)

    analyzer.start("stream-1", 9.2)
    _push_frames(analyzer, frames)
    analyzer.open_window("ep-1", 9.5)
    with pytest.raises(RuntimeError, match="already open"):
        analyzer.open_window("ep-2", 9.8)
    analyzer.close_window()
    with pytest.raises(ValueError):
        analyzer.open_window("", 9.8)
    with pytest.raises(ValueError):
        analyzer.open_window("ep-3", float("nan"))
    analyzer.end()
    with pytest.raises(RuntimeError):
        analyzer.open_window("ep-4", 9.8)


def test_a_window_and_its_close_keep_the_held_contact_run() -> None:
    """F17: a contact held across the window boundary stays one stream event."""
    frames = _long_held_contact_frames(9.2, 12.0)
    analyzer, tracker = _stream(frames)
    first = analyzer.contact_events
    assert [event[0] for event in first] == [pytest.approx(9.5, abs=fx.DT)]
    resets = tracker.resets

    analyzer.open_window("ep-1", 10.5)
    analyzer.close_window()

    assert tracker.resets == resets
    assert [event[0] for event in analyzer.contact_events] == [pytest.approx(9.5, abs=fx.DT)]
    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[0] >= 11.9  # the run kept growing after the window


def test_prune_drops_frames_events_and_grades_before_the_cut() -> None:
    frames = [frame for frame in _episode_frames() if 9.2 <= frame.t <= 14.5]
    analyzer, _ = _stream(frames)
    assert len(analyzer.contact_events) == 2

    analyzer.prune(11.0)

    assert analyzer.frames
    assert all(frame.t >= 11.0 - 1e-9 for frame in analyzer.frames)
    assert [event[0] for event in analyzer.contact_events] == [pytest.approx(12.2, abs=2 * fx.DT)]
    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[0] >= 11.0
    assert analyzer.report(14.5).t_end == pytest.approx(analyzer.frames[-1].t)


def test_prune_never_cuts_inside_an_open_window() -> None:
    frames = [frame for frame in _episode_frames() if 9.2 <= frame.t <= 13.0]
    analyzer, _ = _stream(frames)
    analyzer.open_window("ep-1", 11.5)

    analyzer.prune(12.9)

    assert all(frame.t >= 11.5 - fx.DT for frame in analyzer.frames)
    assert all(event[0] >= 11.5 - fx.DT for event in analyzer.contact_events)


def test_prune_without_a_window_keeps_the_unspecified_history() -> None:
    frames = [frame for frame in _episode_frames() if 9.2 <= frame.t <= 12.0]
    analyzer, _ = _stream(frames)

    analyzer.prune(9.0)  # before every recorded frame

    assert len(analyzer.frames) == len(frames)


def test_prune_bounds_a_contact_held_past_the_retained_history() -> None:
    """F17: a long hold keeps grading from the retained frames, not from all of them."""
    times = np.arange(20.0, 20.6, fx.DT, dtype=np.float64)
    frames = _long_held_contact_frames(9.2, 20.0) + fx.frames_from_positions(
        times, [fx.neutral_xy() for _ in times]
    )
    tracker = _ScriptedTracker(frames)
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=tracker)
    analyzer.start("stream-1", 9.2)
    peak = 0
    runs = 0
    for frame in frames:
        analyzer.push(_pose(frame.seq, frame.t))
        analyzer.prune(frame.t - 3.0)
        peak = max(peak, len(analyzer.frames))
        runs = max(runs, len(analyzer._contact_runs))

    assert peak <= int(3.0 / fx.DT) + 1
    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[0] >= 19.9  # the held contact stayed active
    # The hold stayed one contact run: pruning never split it.
    assert runs == 1


# ---------------------------------------------------------------------------
# P-03 coverage and source health
# ---------------------------------------------------------------------------


def test_no_frames_is_undecided() -> None:
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=_ScriptedTracker([]))
    analyzer.start("ep-1", 10.0)

    assessment = analyzer.assess(10.0, 11.0)

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.features["coverage_ok"] == 0.0
    assert assessment.episode_id == "ep-1"


def test_incomplete_coverage_is_undecided_not_rejected() -> None:
    times = np.array([10.0, 10.1, 10.2])
    frames = fx.frames_from_positions(times, [fx.neutral_xy() for _ in times])
    analyzer = _analyzer(frames, start=10.0)

    assessment = analyzer.assess(10.0, 11.0)

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.features["coverage_ok"] == 0.0
    assert assessment.features["coverage_end_t"] < 11.0


def test_unhealthy_source_is_undecided_even_with_a_contact() -> None:
    analyzer = _analyzer(_episode_frames())

    assessment = analyzer.assess(9.2, 10.4, now=10.2, source_healthy=False)

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.features["coverage_ok"] == 0.0


# ---------------------------------------------------------------------------
# P-04 revised: episode grade, follow-ups, activity-only and rejection
# ---------------------------------------------------------------------------


def test_contact_in_the_window_confirms_and_records_the_latest_grade() -> None:
    analyzer = _analyzer(_episode_frames(), until=10.5)

    assessment = analyzer.assess(9.2, 10.4, now=10.2)

    assert assessment.verdict is Verdict.CONFIRMED
    assert assessment.severity is Tag.DANGEROUS
    assert assessment.episode_id == "ep-1"
    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[1] is Tag.DANGEROUS
    # The recency reference is the event end, not its 9.4 s start (R-14).
    assert 10.2 <= analyzer.latest_grade[0] <= 10.4


def test_follow_up_inside_the_activity_window_uses_the_latest_grade() -> None:
    analyzer = _analyzer(_episode_frames(), until=11.6)
    analyzer.assess(9.2, 10.4, now=10.2)

    follow_up = analyzer.assess(10.6, 11.4, now=11.2)

    assert follow_up.verdict is Verdict.CONFIRMED
    assert follow_up.severity is Tag.DANGEROUS
    assert follow_up.features["from_episode"] == 1.0


def test_an_event_between_assessment_windows_becomes_the_latest_grade() -> None:
    times = np.arange(9.0, 13.0, fx.DT, dtype=np.float64)
    frames = _with_contact(
        times,
        (10.0, 10.3),
        EYE,
        HandShape.FIST,
        [(10.0, 1.0), (10.05, 0.05), (10.2, 0.05), (10.3, 1.2)],
    )
    selected = [frame for frame in frames if frame.t >= 9.2]
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=_ScriptedTracker(selected))
    analyzer.start("ep-1", 9.2)
    pushed = 0
    while selected[pushed].t <= 9.9:
        analyzer.push(_pose(selected[pushed].seq, selected[pushed].t))
        pushed += 1

    before = analyzer.assess(9.2, 9.9, now=9.5)

    assert before.verdict is not Verdict.CONFIRMED
    assert analyzer.latest_grade is None

    while selected[pushed].t <= 10.7:
        analyzer.push(_pose(selected[pushed].seq, selected[pushed].t))
        pushed += 1

    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[1] is Tag.DANGEROUS
    # Recency uses the event end, not its 10.0 s start (R-14).
    assert abs(analyzer.latest_grade[0] - 10.2) <= fx.DT

    follow_up = analyzer.assess(10.4, 10.6, now=10.7)

    assert follow_up.verdict is Verdict.CONFIRMED
    assert follow_up.severity is Tag.DANGEROUS
    assert follow_up.features["from_episode"] == 1.0


def test_window_verdict_uses_the_maximum_but_latest_grade_uses_the_latest_event() -> None:
    times = np.arange(9.0, 12.0, fx.DT, dtype=np.float64)
    severe = _with_contact(
        times,
        (9.0, 9.2),
        EYE,
        HandShape.FIST,
        [(9.0, 0.05), (9.15, 0.05), (9.2, 1.2)],
    )
    # A decelerating open palm reaches the chin after the eye contact: the window
    # maximum stays DANGEROUS, but the latest event is graded NON_DANGEROUS.
    mild = _with_contact(
        times,
        (9.3, 10.9),
        CHIN,
        HandShape.OPEN_PALM,
        [
            (9.3, 1.0),
            (9.4, 0.767),
            (9.5, 0.580),
            (9.6, 0.435),
            (9.7, 0.326),
            (9.8, 0.248),
            (9.9, 0.195),
            (10.0, 0.163),
            (10.1, 0.147),
            (10.2, 0.141),
            (10.3, 0.140),
            (10.6, 0.140),
            (10.9, 1.2),
        ],
    )
    frames = [mild[index] if 9.3 <= t <= 10.9 else severe[index] for index, t in enumerate(times)]
    analyzer = _analyzer(frames, start=9.0)

    assessment = analyzer.assess(9.0, 10.5, now=10.5)

    assert assessment.verdict is Verdict.CONFIRMED
    assert assessment.severity is Tag.DANGEROUS
    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[1] is Tag.NON_DANGEROUS
    assert analyzer.latest_grade[0] > 9.6


def test_follow_up_after_inactivity_rejects_and_stops_follow_ups() -> None:
    analyzer = _analyzer(_episode_frames())
    analyzer.assess(9.2, 10.4, now=10.2)

    late = analyzer.assess(19.2, 20.4, now=20.0)

    assert late.verdict is Verdict.REJECTED
    assert late.severity is None
    assert "from_episode" not in late.features


def test_a_later_non_dangerous_contact_downgrades_the_grade() -> None:
    analyzer = _analyzer(_episode_frames())
    analyzer.assess(9.2, 10.4, now=10.2)

    assessment = analyzer.assess(11.9, 13.1, now=12.9)

    assert assessment.verdict is Verdict.CONFIRMED
    assert assessment.severity is Tag.NON_DANGEROUS
    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[1] is Tag.NON_DANGEROUS

    follow_up = analyzer.assess(12.9, 14.1, now=13.9)
    assert follow_up.verdict is Verdict.CONFIRMED
    assert follow_up.severity is Tag.NON_DANGEROUS


# ---------------------------------------------------------------------------
# R-10/R-11: pending recoil tail and incremental contact grading
# ---------------------------------------------------------------------------


def test_a_short_contact_with_a_late_recoil_uses_the_finalized_grade() -> None:
    frames = _short_contact_with_late_recoil()
    expected = detect_contacts(frames, fx.LAYOUT, ANALYSIS)
    assert len(expected) == 1
    contact = expected[0]
    mild = detect_contacts(_short_contact_with_late_recoil(recoil=False), fx.LAYOUT, ANALYSIS)
    assert len(mild) == 1
    assert severity_of(mild[0], ANALYSIS) is Tag.NON_DANGEROUS
    assert severity_of(contact, ANALYSIS) is Tag.DANGEROUS
    assert contact.head_recoil_fw is not None

    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=_ScriptedTracker(frames))
    analyzer.start("ep-1", float(frames[0].t))
    split = (
        max(index for index, frame in enumerate(frames) if frame.t <= contact.t_start + 0.25) + 1
    )
    _push_frames(analyzer, frames[:split])

    # The contact ended but its recoil window is still open, so the grade only
    # reflects the frames pushed so far. Recency uses the contact end (R-14).
    contact_end = contact.t_start + contact.duration_s
    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[0] == pytest.approx(contact_end)
    assert analyzer.latest_grade[1] is Tag.NON_DANGEROUS
    assert analyzer.contact_peak == (1, contact.t_start)

    _push_frames(analyzer, frames[split:])

    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[0] == pytest.approx(contact_end)
    assert analyzer.latest_grade[1] is Tag.DANGEROUS
    assert analyzer.contact_peak == (2, contact.t_start)
    follow_up = analyzer.assess(
        contact.t_start + 0.4, contact.t_start + 0.6, now=contact.t_start + 0.6
    )
    assert follow_up.verdict is Verdict.CONFIRMED
    assert follow_up.severity is Tag.DANGEROUS
    assert follow_up.features["from_episode"] == 1.0

    report = analyzer.report(contact.t_start + 0.6)
    graded = [item for item in report.contacts if item.t_start == contact.t_start]
    assert graded
    assert severity_of(graded[0], ANALYSIS) is Tag.DANGEROUS


@pytest.mark.parametrize(
    "builder",
    [_long_held_contact_frames, _short_contact_with_late_recoil, _missing_sample_mid_contact],
    ids=["long-held-contact", "short-late-recoil", "missing-sample"],
)
def test_incremental_grading_matches_detect_contacts_at_every_prefix(
    builder: Callable[[], list[TrackedFrame]],
) -> None:
    """The incremental grade equals ``detect_contacts`` for every pushed prefix.

    ``latest_grade`` carries the per-contact end time and severity and
    ``contact_peak`` the report's peak rule over all contacts seen so far, so
    together they cover every graded result the analyzer derives from a contact.
    """
    frames = builder()
    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=_ScriptedTracker(frames))
    analyzer.start("ep-1", float(frames[0].t))
    for frame in frames:
        analyzer.push(_pose(frame.seq, frame.t))

        contacts = detect_contacts(analyzer.frames, fx.LAYOUT, ANALYSIS)
        peaks = [
            (1 if severity_of(contact, ANALYSIS) is Tag.NON_DANGEROUS else 2, contact.t_start)
            for contact in contacts
        ]
        assert analyzer.contact_peak == (max(peaks) if peaks else None)
        if contacts:
            latest = max(contacts, key=lambda contact: contact.t_start + contact.duration_s)
            assert analyzer.latest_grade is not None
            assert analyzer.latest_grade[0] == pytest.approx(latest.t_start + latest.duration_s)
            assert analyzer.latest_grade[1] is severity_of(latest, ANALYSIS)
        else:
            assert analyzer.latest_grade is None


def test_push_work_does_not_grow_with_contact_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held contact must not re-detect the whole run on every frame (R-11).

    Counting hand-to-face sample computations keeps this deterministic; a
    wall-clock bound would depend on machine load.
    """
    frames = _long_held_contact_frames(0.0, 20.0)
    calls = {"count": 0}

    def counted(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return hand_face_samples_impl(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("cpsa.vision.episode.hand_face_samples", counted)
    monkeypatch.setattr("cpsa.vision.contacts2d.hand_face_samples", counted)

    # The per-frame regrade must also stay bounded by the recoil window, not the run.
    centroid_slices: list[int] = []
    face_centroids_impl = episode_module._face_centroids

    def sized(frames_arg: Sequence[TrackedFrame], *args: object) -> object:
        centroid_slices.append(len(frames_arg))
        return face_centroids_impl(frames_arg, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(episode_module, "_face_centroids", sized)

    analyzer = EpisodeAnalyzer(fx.LAYOUT, TRACKER, ANALYSIS, tracker=_ScriptedTracker(frames))
    analyzer.start("ep-1", 0.0)
    per_push = []
    for frame in frames:
        before = calls["count"]
        analyzer.push(_pose(frame.seq, frame.t))
        per_push.append(calls["count"] - before)

    step = int(round(1.0 / fx.DT))
    first = float(np.mean(per_push[:step]))
    last = float(np.mean(per_push[-step:]))
    assert first >= 1.0
    assert last <= 2.0 * first
    recoil_frames = int(np.ceil(ANALYSIS.contacts.recoil_window_s / fx.DT)) + 2
    assert centroid_slices
    assert max(centroid_slices) <= recoil_frames


# ---------------------------------------------------------------------------
# R-14: latest-grade recency uses the event end, like the report activity
# ---------------------------------------------------------------------------


def test_long_contact_recency_uses_its_end_time() -> None:
    """Follow-ups stay confirmed until one activity window after the contact end (R-14)."""
    frames = _long_contact_then_neutral()
    contact = detect_contacts(frames, fx.LAYOUT, ANALYSIS)[0]
    end = contact.t_start + contact.duration_s
    assert end - contact.t_start > ANALYSIS.activity_window_s
    assert severity_of(contact, ANALYSIS) is Tag.DANGEROUS

    # While the contact is still held it counts as active now, even though its
    # start is already older than the activity window.
    held_until = contact.t_start + ANALYSIS.activity_window_s + 1.0
    held = _analyzer(frames, start=9.2, until=held_until)
    held_now = held.frames[-1].t
    during = held.assess(held_now - 0.3, held_now - 0.1, now=held_now)
    assert during.verdict is Verdict.CONFIRMED
    assert during.severity is Tag.DANGEROUS
    assert during.features["from_episode"] == 1.0
    assert held.latest_grade is not None
    assert held.latest_grade[0] == pytest.approx(held_now)
    assert held.report(held_now).active

    # A follow-up inside the activity window of the contact end keeps the grade.
    analyzer = _analyzer(frames, start=9.2)
    within = end + 2.0
    follow_up = analyzer.assess(within - 0.9, within, now=within)
    assert follow_up.verdict is Verdict.CONFIRMED
    assert follow_up.severity is Tag.DANGEROUS
    assert follow_up.features["from_episode"] == 1.0
    assert analyzer.latest_grade is not None
    assert analyzer.latest_grade[0] == pytest.approx(end)
    assert analyzer.latest_grade[1] is Tag.DANGEROUS
    assert analyzer.report(within).active

    # Once the window closes the episode is inactive and follow-ups are rejected.
    after = end + ANALYSIS.activity_window_s + 0.5
    late = analyzer.assess(after - 0.9, after, now=after)
    assert late.verdict is Verdict.REJECTED
    assert late.severity is None
    assert "from_episode" not in late.features
    assert not analyzer.report(after).active


def test_activity_only_rocking_is_undecided_not_rejected() -> None:
    times = 10.0 + np.arange(180, dtype=np.float64) * fx.DT
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        offset = 0.15 * fx.FACE_WIDTH_PX * np.sin(2.0 * np.pi * (t - 10.0))
        for name in ("left_shoulder", "right_shoulder"):
            xy[fx.LAYOUT.index(name), 0] += offset
        positions.append(xy)
    frames = fx.frames_from_positions(times, positions)
    analyzer = _analyzer(frames, start=float(times[0]))

    assessment = analyzer.assess(float(times[0]), float(times[-1]), now=float(times[-1]))

    assert assessment.verdict is Verdict.UNDECIDED
    assert assessment.severity is None
    assert assessment.features["activity_only"] == 1.0
    assert "from_episode" not in assessment.features
    assert analyzer.active(float(times[-1]))


def test_stage_two_report_describes_the_contact() -> None:
    analyzer = _analyzer(_episode_frames())
    analyzer.assess(9.2, 10.4, now=10.2)

    report = analyzer.report(10.4)

    assert report.episode_id == "ep-1"
    assert report.t_start == 9.2
    assert report.contact_count == 2
    assert report.peak_severity is Tag.DANGEROUS
    assert StereotypyLabel.HAND_TO_FACE in report.labels
    assert report.active
