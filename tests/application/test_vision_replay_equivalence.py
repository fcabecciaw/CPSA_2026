"""AC-014/AC-018: replay reproduces the live episode exactly (FR-025, P-06, P-07)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from cpsa.application.vision_verifier import (
    EpisodeTrace,
    TraceFrame,
    episode_report_json,
    wrist_acceleration,
)
from cpsa.board.vision_replay import replay_audit
from cpsa.domain.model import HandShape, Side, Tag, Verdict, VisionAssessment
from cpsa.vision.episode import EpisodeAnalyzer
from cpsa.vision.fusion import IMUSignal
from cpsa.vision.types import PersonPose, PoseFrame, TrackedFrame, VideoFrame
from tests.application import test_vision_verifier as live
from tests.fakes import FakeClock
from tests.unit import vision_fixtures as fx


def live_frames() -> list[TrackedFrame]:
    """The known episode timeline: a fist to the eye that confirms at 10.1 s."""
    return [frame for frame in live.episode_frames() if frame.t <= 11.4]


def video_frames(tracked: list[TrackedFrame]) -> list[VideoFrame]:
    """Wrap the synthetic tracked frames as lazy camera frames for the replay."""
    return [
        VideoFrame(
            seq=frame.seq,
            t=frame.t,
            width=640,
            height=480,
            decode=live._payload(int(frame.seq)),
        )
        for frame in tracked
    ]


def _assert_values_equal(left, right) -> None:
    if isinstance(left, float) and isinstance(right, float):
        assert left == pytest.approx(right, abs=1e-9)
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_values_equal(first, second)
    elif isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for name in left:
            _assert_values_equal(left[name], right[name])
    else:
        assert left == right


def _assert_same_assessment(left: VisionAssessment, right: VisionAssessment) -> None:
    assert left.verdict is right.verdict
    assert left.severity is right.severity
    assert left.episode_id == right.episode_id
    assert set(left.features) == set(right.features)
    for name, value in left.features.items():
        assert value == pytest.approx(right.features[name], abs=1e-9)


def _assert_same_report(left, right) -> None:
    first = episode_report_json(left)
    second = episode_report_json(right)
    first.pop("trace")
    second.pop("trace")
    _assert_values_equal(first, second)


def _replay(frames, config, trace, *, imu=None):
    return replay_audit(
        video_frames(frames), live.PoseBook(frames), config, fx.LAYOUT, trace, imu=imu
    )


def run_live(
    tmp_path: Path,
    *,
    delay: float = 0.0,
    queue_max: int = 8,
    fast_budget_s: float | None = None,
    wait_for_frames: bool = True,
):
    """Run one live episode over the pre-roll and return its outputs."""
    frames = live_frames()
    overrides: dict[str, object] = {"queue_max": queue_max}
    if fast_budget_s is not None:
        overrides["fast_budget_s"] = fast_budget_s
    config = live._config(tmp_path, **overrides)
    clock = FakeClock(10.0)
    estimator = live.PoseBook(frames, delay=delay)
    verifier, source, _, _ = live.make_verifier(
        tmp_path, config=config, estimator=estimator, clock=clock
    )
    verifier.start()
    try:
        live.emit_range(source, frames, 9.0, 10.4)
        verifier.observe(live.detection(Tag.DANGEROUS, 10.2, clock))
        if wait_for_frames:
            live.wait_until(
                lambda: (verifier.analyzed_until_t or 0.0) >= 10.4 - 1e-9,
                timeout=10.0,
            )
        clock.t = 10.2
        result = verifier.verify(live.detection(Tag.DANGEROUS, 10.2, clock))
        clock.t = 10.4
    finally:
        verifier.stop()
    trace = verifier.last_trace
    report = verifier.last_report
    assert trace is not None and report is not None
    return frames, config, result, trace, report


# ---------------------------------------------------------------------------
# AC-014: live and replay are identical under estimator delays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delay",
    [0.0, fx.DT / 2.0, 2.0 * fx.DT],
    ids=["no-delay", "half-frame", "double-frame"],
)
def test_replay_matches_live_for_every_estimator_delay(tmp_path: Path, delay: float) -> None:
    frames, config, result, trace, report = run_live(tmp_path, delay=delay)

    replay = _replay(frames, config, trace)

    assert result.verdict is Verdict.CONFIRMED
    assert result.severity is Tag.DANGEROUS
    assert len(replay.assessments) == 1
    _assert_same_assessment(result, replay.assessments[0])
    assert replay.report is not None
    _assert_same_report(report, replay.report)


def test_replay_matches_live_after_the_queue_dropped_frames(tmp_path: Path) -> None:
    frames = live_frames()
    config = live._config(tmp_path, queue_max=2)
    clock = FakeClock(9.0)
    gate = threading.Event()
    estimator = live.PoseBook(frames, gate=gate)
    verifier, source, _, _ = live.make_verifier(
        tmp_path, config=config, estimator=estimator, clock=clock
    )
    verifier.start()
    try:
        verifier.observe(live.detection(Tag.DANGEROUS, 9.2, clock))
        source.emit(frames[0])
        live.wait_until(estimator.started.is_set)
        for frame in frames:
            if 9.1 <= frame.t <= 9.5 and abs(frame.t * 10 - round(frame.t * 10)) < 1e-6:
                source.emit(frame)
        gate.set()
        live.wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 9.5 - 1e-9)
        clock.t = 9.5
        result = verifier.verify(live.detection(Tag.DANGEROUS, 9.3, clock))
        clock.t = 9.6
    finally:
        verifier.stop()
    trace = verifier.last_trace
    report = verifier.last_report
    assert trace is not None and report is not None
    assert len(trace.frames) < len([frame for frame in frames if 9.0 <= frame.t <= 9.5])

    replay = _replay(frames, config, trace)

    assert len(replay.assessments) == len(trace.assessments)
    _assert_same_assessment(result, replay.assessments[0])
    assert replay.report is not None
    _assert_same_report(report, replay.report)


def test_replay_matches_a_timed_out_live_assessment(tmp_path: Path) -> None:
    frames, config, result, trace, report = run_live(
        tmp_path, delay=0.2, fast_budget_s=0.05, wait_for_frames=False
    )

    replay = _replay(frames, config, trace)

    assert result.verdict is Verdict.UNDECIDED
    assert result.features == {"timeout": 1.0}
    assert any(record.timeout for record in trace.assessments)
    assert len(replay.assessments) == len(trace.assessments)
    _assert_same_assessment(result, replay.assessments[-1])
    assert replay.report is not None
    _assert_same_report(report, replay.report)


def test_replay_matches_an_unhealthy_source(tmp_path: Path) -> None:
    frames = live_frames()
    config = live._config(tmp_path)
    clock = FakeClock(10.0)
    verifier, source, _, _ = live.make_verifier(
        tmp_path, config=config, estimator=live.PoseBook(frames), clock=clock
    )
    verifier.start()
    try:
        live.emit_range(source, frames, 9.0, 10.4)
        source.fail(RuntimeError("camera unplugged"))
        result = verifier.verify(live.detection(Tag.DANGEROUS, 10.2, clock))
        clock.t = 10.4
    finally:
        verifier.stop()
    trace = verifier.last_trace
    report = verifier.last_report
    assert trace is not None and report is not None
    assert result.verdict is Verdict.UNDECIDED
    assert trace.assessments[-1].source_healthy is False

    replay = _replay(frames, config, trace)

    _assert_same_assessment(result, replay.assessments[-1])
    assert replay.report is not None
    _assert_same_report(report, replay.report)


def test_replay_matches_a_source_that_went_stale_before_completion(tmp_path: Path) -> None:
    frames = live_frames()
    config = live._config(tmp_path, stale_after_s=0.5, queue_max=64)
    clock = FakeClock(9.0)
    gate = threading.Event()
    gate_seq = next(frame.seq for frame in frames if frame.t >= 10.2 - 1e-9)
    estimator = live.PoseBook(frames, gate=gate, gate_on=gate_seq)
    verifier, source, _, _ = live.make_verifier(
        tmp_path, config=config, estimator=estimator, clock=clock
    )
    verifier.start()
    try:
        verifier.observe(live.detection(Tag.DANGEROUS, 9.2, clock))
        live.emit_range(source, frames, 9.0, 10.25)
        live.wait_until(estimator.started.is_set)
        # The last capture is 10.2 s; by completion 0.6 s have passed, more than
        # the 0.5 s freshness window, so the episode is no longer healthy.
        clock.t = 10.8
        gate.set()
        result = verifier.verify(live.detection(Tag.DANGEROUS, 10.0, clock))
    finally:
        gate.set()
        verifier.stop()
    trace = verifier.last_trace
    report = verifier.last_report
    assert trace is not None and report is not None
    assert result.verdict is Verdict.UNDECIDED
    assert result.features["coverage_ok"] == 0.0
    assert trace.assessments[-1].source_healthy is False

    replay = _replay(frames, config, trace)

    _assert_same_assessment(result, replay.assessments[-1])
    assert replay.report is not None
    _assert_same_report(report, replay.report)


def test_replay_matches_a_timeout_followed_by_a_grade_use(tmp_path: Path) -> None:
    from cpsa.vision import episode as episode_module

    frames = [frame for frame in live.episode_frames() if frame.t <= 15.1]
    config = live._config(tmp_path, fast_budget_s=0.05, queue_max=64)
    clock = FakeClock(9.0)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    gate = {"armed": True}
    original = episode_module.EpisodeAnalyzer.assess

    def gated(self, t_from, t_to, now=None, *, source_healthy=True):
        if gate["armed"]:
            gate["armed"] = False
            entered.set()
            release.wait(timeout=5.0)
            try:
                return original(self, t_from, t_to, now, source_healthy=source_healthy)
            finally:
                finished.set()
        return original(self, t_from, t_to, now, source_healthy=source_healthy)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(episode_module.EpisodeAnalyzer, "assess", gated)
    verifier, source, _, _ = live.make_verifier(
        tmp_path, config=config, estimator=live.PoseBook(frames), clock=clock
    )
    verifier.start()
    timed_out: list[VisionAssessment] = []
    try:
        verifier.observe(live.detection(Tag.DANGEROUS, 9.2, clock))
        live.emit_range(source, frames, 9.0, 15.1)
        live.wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 15.1 - 1e-9)
        # The first assessment is still running when its request times out; the
        # computed result is discarded but must not leave any episode state behind.
        thread = threading.Thread(
            target=lambda: timed_out.append(
                verifier.verify(live.detection(Tag.DANGEROUS, 12.9, clock))
            )
        )
        thread.start()
        live.wait_until(entered.is_set)
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        release.set()
        live.wait_until(finished.is_set)
        clock.t = 14.9
        result = verifier.verify(live.detection(Tag.DANGEROUS, 14.9, clock))
        clock.t = 15.1
    finally:
        release.set()
        verifier.stop()
        monkeypatch.undo()
    trace = verifier.last_trace
    report = verifier.last_report
    assert trace is not None and report is not None
    assert timed_out and timed_out[0].verdict is Verdict.UNDECIDED
    assert timed_out[0].features == {"timeout": 1.0}
    assert result.verdict is Verdict.CONFIRMED
    assert result.severity is Tag.NON_DANGEROUS
    assert result.features["from_episode"] == 1.0
    assert any(record.timeout for record in trace.assessments)

    replay = _replay(frames, config, trace)

    assert len(replay.assessments) == len(trace.assessments)
    _assert_same_assessment(result, replay.assessments[-1])
    assert replay.report is not None
    _assert_same_report(report, replay.report)


# ---------------------------------------------------------------------------
# AC-018: the injected IMU clock offset survives the replay
# ---------------------------------------------------------------------------


def _oscillating_frames() -> list[TrackedFrame]:
    times = np.arange(0.0, 4.0, fx.DT, dtype=np.float64)
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        offset = 30.0 * np.sin(2.0 * np.pi * 2.0 * (t - times[0]))
        fx.set_hand(
            xy,
            Side.RIGHT,
            (
                fx.FACE_CENTER[0] + 0.9 * fx.FACE_WIDTH_PX + offset,
                fx.FACE_CENTER[1] + 3.0 * fx.FACE_WIDTH_PX,
            ),
            HandShape.OPEN_PALM,
        )
        positions.append(xy)
    return fx.frames_from_positions(times, positions)


def _tracked(frames, config):
    analyzer = EpisodeAnalyzer(fx.LAYOUT, config.tracker, config.analysis)
    analyzer.start("ep-imu", float(frames[0].t))
    for frame in frames:
        person = PersonPose(
            box=np.array([0.0, 0.0, 640.0, 480.0]),
            score=0.9,
            xy=frame.xy.copy(),
            conf=frame.conf.copy(),
        )
        analyzer.push(PoseFrame(frame.seq, frame.t, 640, 480, (person,)))
    return analyzer.frames


def _session_episode_two(*, outlier_every: int | None) -> list[TrackedFrame]:
    """A later episode in the same session, with optional stretched right forearms."""
    frames: list[TrackedFrame] = []
    for index, source in enumerate(_oscillating_frames()):
        xy = source.xy.copy()
        conf = np.full(xy.shape[0], 0.9, dtype=np.float64)
        if outlier_every is not None and index % outlier_every == 0:
            elbow = fx.LAYOUT.index("right_elbow")
            wrist = fx.LAYOUT.index("right_wrist")
            direction = xy[wrist] - xy[elbow]
            length = float(np.linalg.norm(direction))
            if length < 1e-9:
                direction = np.array([0.0, 1.0])
                length = 1.0
            xy[wrist] = xy[elbow] + direction / length * 6.0 * fx.FACE_WIDTH_PX
            conf[wrist] = 0.4
        frames.append(fx.make_frame(1000 + index, 20.0 + source.t, xy, conf=conf))
    return frames


def test_replay_restores_the_session_skeleton_references(tmp_path: Path) -> None:
    from cpsa.ports import SampleKind

    first = [frame for frame in live.episode_frames() if frame.t <= 13.5]
    second = _session_episode_two(outlier_every=9)
    clean = _session_episode_two(outlier_every=None)
    frames = first + second
    config = live._config(tmp_path, queue_max=256)
    clock = FakeClock(9.0)
    verifier, source, _, _ = live.make_verifier(
        tmp_path, config=config, estimator=live.PoseBook(frames), clock=clock
    )
    verifier.start()
    try:
        live.emit_range(source, first, 9.0, 9.5)
        clock.t = 9.3
        verifier.verify(live.detection(Tag.DANGEROUS, 9.3, clock))
        live.emit_range(source, first, 9.53, 13.5)
        live.wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 13.5 - 1e-9)
        clock.t = 13.3
    finally:
        verifier.stop()
    first_trace = verifier.last_trace
    assert first_trace is not None
    assert first_trace.skeleton is not None

    tracked_clean = _tracked(clean, config)
    imu_t, imu_acc = wrist_acceleration(tracked_clean, fx.LAYOUT, config.analysis, Side.RIGHT)
    assert imu_t.size > 3
    imu = IMUSignal(t_s=imu_t, acc_norm_g=imu_acc)
    clock.t = 20.0
    verifier.start()
    try:
        verifier.observe(live.detection(Tag.DANGEROUS, 20.2, clock))
        for sample_t, sample_acc in zip(imu_t, imu_acc, strict=True):
            verifier.on_imu_sample(
                Side.RIGHT, SampleKind.ACC, (1000.0 * float(sample_acc), 0.0, 0.0), float(sample_t)
            )
        live.emit_range(source, second, 20.0, 24.0)
        live.wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 23.9)
        clock.t = 23.9
        result = verifier.verify(live.detection(Tag.DANGEROUS, 23.7, clock))
    finally:
        verifier.stop()
    trace = verifier.last_trace
    report = verifier.last_report
    assert trace is not None and report is not None
    assert trace.skeleton is not None
    assert trace.skeleton["frozen"]["right_forearm"] is True
    assert report.imu_consistency is not None

    replay = replay_audit(
        video_frames(second),
        live.PoseBook(second),
        config,
        fx.LAYOUT,
        trace,
        imu={Side.RIGHT: imu},
    )

    _assert_same_assessment(result, replay.assessments[-1])
    assert replay.report is not None
    _assert_same_report(report, replay.report)


def test_injected_imu_offset_is_recovered_through_replay(tmp_path: Path) -> None:
    frames = _oscillating_frames()
    config = live._config(tmp_path)
    tracked = _tracked(frames, config)
    cam_t, cam_acc = wrist_acceleration(tracked, fx.LAYOUT, config.analysis, Side.RIGHT)
    assert cam_t.size > 30
    injected = 0.1
    imu = IMUSignal(t_s=cam_t + injected, acc_norm_g=cam_acc.copy())
    trace = EpisodeTrace(
        episode_id="ep-imu",
        t_start=float(frames[0].t),
        layout=fx.LAYOUT.name,
        analysis_fps=config.analysis_fps,
        report_now=float(frames[-1].t),
        frames=tuple(TraceFrame(frame.seq, frame.t) for frame in frames),
        assessments=(),
    )

    result = replay_audit(
        video_frames(frames),
        live.PoseBook(frames),
        config,
        fx.LAYOUT,
        trace,
        imu={Side.RIGHT: imu},
    )

    assert result.report is not None
    assert result.report.clock_offset_s == pytest.approx(-injected, abs=fx.DT)
    assert result.report.imu_consistency == pytest.approx(1.0, abs=0.1)
