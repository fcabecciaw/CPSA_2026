"""Vision verifier runtime: P-05 threading, AC-001, AC-011..AC-013.

The fake source, estimator and clock make every episode deterministic: frames
are emitted by the test, the estimator maps a frame payload to its prepared pose,
and replay of the same frames is what AC-014 checks in the companion module.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from cpsa.application.config import VisionConfig, load
from cpsa.application.dispatcher import Dispatcher
from cpsa.application.slot import LatestSlot
from cpsa.application.vision_verifier import VisionVerifier
from cpsa.domain.model import (
    ActuatorKind,
    Detection,
    FaceRegion,
    HandShape,
    Side,
    Tag,
    Verdict,
    VisionAssessment,
)
from cpsa.vision.contacts2d import detect_contacts
from cpsa.vision.types import EpisodeReport, PersonPose, TrackedFrame, VideoFrame
from tests.application.test_config import SHIPPED
from tests.fakes import FakeClock, MemorySink, RecordingActuator, RecordingGui
from tests.unit import vision_fixtures as fx

STRONG_META = {"duty": 100, "duration": 900}
MILD_META = {"duty": 40, "duration": 700}
EYE = fx.face_anchor("left_eye")
CHIN = fx.face_anchor("face_chin")


def _config(tmp_path: Path, **overrides: object) -> VisionConfig:
    base = load(SHIPPED).vision
    values: dict[str, object] = {
        "enabled": True,
        "preroll_s": 3.0,
        "lookback_s": 1.0,
        "fast_after_s": 0.2,
        "fast_budget_s": 1.0,
        "idle_timeout_s": 100.0,
        "stale_after_s": 0.5,
        "analysis_fps": 10.0,
        "queue_max": 8,
        "imu_buffer_s": 10.0,
        "evidence": replace(base.evidence, enabled=False, base_path=tmp_path / "evidence"),
    }
    values.update(overrides)
    return replace(base, **values)


def _trajectory(
    window: tuple[float, float],
    anchor: np.ndarray,
    shape: HandShape,
    keyframes: Sequence[tuple[float, float]],
    times: np.ndarray,
) -> list[np.ndarray]:
    positions = []
    for t in times:
        xy = fx.neutral_xy()
        if window[0] <= t <= window[1]:
            distance = float(
                np.interp(t, [item[0] for item in keyframes], [item[1] for item in keyframes])
            )
            fx.set_hand_distance(xy, Side.RIGHT, anchor, distance, shape, direction=(0, 1))
        positions.append(xy)
    return positions


def episode_frames() -> list[TrackedFrame]:
    """Neutral frames from 9 s to 21 s with a fist and a slow open-palm contact."""
    times = np.arange(9.0, 21.0, fx.DT, dtype=np.float64)
    first = _trajectory(
        (9.3, 10.8),
        EYE,
        HandShape.FIST,
        [(9.3, 1.0), (9.5, 0.05), (10.5, 0.05), (10.8, 1.2)],
        times,
    )
    second = _trajectory(
        (11.6, 13.8),
        CHIN,
        HandShape.OPEN_PALM,
        [(11.6, 1.0), (12.6, 0.12), (13.2, 0.12), (13.8, 1.2)],
        times,
    )
    positions = [
        second[index] if 11.6 <= t <= 13.8 else first[index] for index, t in enumerate(times)
    ]
    return fx.frames_from_positions(times, positions)


def _payload(seq: int) -> Callable[[], np.ndarray]:
    def decode() -> np.ndarray:
        array = np.zeros((1, 2, 3), dtype=np.uint8)
        array[0, 0, :] = seq % 256
        array[0, 1, :] = (seq // 256) % 256
        return array

    return decode


class ManualVideoSource:
    """A ``VideoSource`` driven frame by frame from the test thread."""

    def __init__(self) -> None:
        self.on_frame: Callable[[VideoFrame], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None
        self.start_calls = 0
        self.stop_calls = 0
        self.running = False

    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.start_calls += 1
        self.on_frame = on_frame
        self.on_error = on_error
        self.running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def emit(self, tracked: TrackedFrame) -> None:
        assert self.on_frame is not None
        frame = VideoFrame(
            seq=int(tracked.seq),
            t=float(tracked.t),
            width=640,
            height=480,
            decode=_payload(int(tracked.seq)),
        )
        self.on_frame(frame)

    def fail(self, error: Exception) -> None:
        assert self.on_error is not None
        self.on_error(error)


class PoseBook:
    """Map the payload sequence number to a prepared pose; optional delay and failure."""

    def __init__(
        self,
        frames: Sequence[TrackedFrame],
        *,
        delay: float = 0.0,
        fail_on: set[int] | None = None,
        gate: threading.Event | None = None,
        gate_on: int | None = None,
    ) -> None:
        self._by_seq = {int(frame.seq): frame for frame in frames}
        self.delay = delay
        self.fail_on = fail_on or set()
        self.gate = gate
        self.gate_on = gate_on
        self.started = threading.Event()
        self.calls: list[int] = []

    def estimate(self, rgb: np.ndarray) -> list[PersonPose]:
        seq = int(rgb[0, 0, 0]) | (int(rgb[0, 1, 0]) << 8)
        self.calls.append(seq)
        if self.gate is not None and (self.gate_on is None or seq == self.gate_on):
            self.started.set()
            self.gate.wait(timeout=5.0)
        if self.delay:
            time.sleep(self.delay)
        if seq in self.fail_on:
            raise RuntimeError("pose model failed")
        tracked = self._by_seq[seq]
        return [
            PersonPose(
                box=np.array([0.0, 0.0, 640.0, 480.0]),
                score=0.9,
                xy=tracked.xy.copy(),
                conf=tracked.conf.copy(),
            )
        ]


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.005)


def detection(tag: Tag, window_end_ts: float, clock: FakeClock) -> Detection:
    return Detection(tag, clock.now(), window_end_ts, (0.5,) * 18)


def make_verifier(
    tmp_path: Path,
    *,
    config: VisionConfig | None = None,
    estimator: PoseBook | None = None,
    clock: FakeClock | None = None,
    evidence: object | None = None,
) -> tuple[VisionVerifier, ManualVideoSource, PoseBook, FakeClock]:
    frames = episode_frames()
    source = ManualVideoSource()
    book = estimator if estimator is not None else PoseBook(frames)
    fake_clock = clock if clock is not None else FakeClock(9.0)
    verifier = VisionVerifier(
        source,
        book,
        evidence,
        config if config is not None else _config(tmp_path),
        fake_clock,
        fx.LAYOUT,
        log_dir=tmp_path / "logs",
    )
    return verifier, source, book, fake_clock


def emit_range(
    source: ManualVideoSource,
    frames: Sequence[TrackedFrame],
    start: float,
    end: float,
) -> list[TrackedFrame]:
    emitted = [frame for frame in frames if start <= frame.t <= end]
    for frame in emitted:
        source.emit(frame)
    return emitted


# ---------------------------------------------------------------------------
# AC-001: no inference while idle, pre-roll, and inference stops after idle
# ---------------------------------------------------------------------------


def test_no_inference_without_a_detection_and_preroll_is_bounded(tmp_path: Path) -> None:
    verifier, source, book, _ = make_verifier(tmp_path, config=_config(tmp_path, preroll_s=1.0))
    frames = episode_frames()
    verifier.start()
    try:
        emit_range(source, frames, 9.0, 10.5)

        assert book.calls == []
        newest = max(frame.t for frame in frames if frame.t <= 10.5)
        assert verifier.preroll_len == len(
            [
                frame
                for frame in frames
                if frame.t <= newest + 1e-9 and newest - frame.t <= 1.0 + 1e-9
            ]
        )
    finally:
        verifier.stop()


def test_detection_analyzes_the_preroll_and_inference_stops_after_idle(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        preroll_s=5.0,
        idle_timeout_s=0.2,
        analysis=replace(load(SHIPPED).vision.analysis, activity_window_s=0.5),
    )
    clock = FakeClock(10.0)
    verifier, source, book, _ = make_verifier(tmp_path, config=config, clock=clock)
    frames = episode_frames()
    verifier.start()
    try:
        emit_range(source, frames, 9.0, 10.4)
        verifier.observe(detection(Tag.DANGEROUS, 10.2, clock))
        wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 10.4 - 1e-9)

        analyzed = {frame.seq for frame in frames if 9.2 <= frame.t <= 10.4}
        assert book.calls
        assert set(book.calls).issubset(analyzed)
        assert all(frame.t >= 9.2 for frame in frames if frame.seq in book.calls)

        clock.t = 30.0
        wait_until(lambda: verifier.last_report is not None)
        calls_at_idle = list(book.calls)
        emit_range(source, frames, 10.5, 12.0)
        time.sleep(0.1)
        assert book.calls == calls_at_idle
    finally:
        verifier.stop()


def test_a_stale_observation_after_the_episode_does_not_start_a_new_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config(tmp_path, preroll_s=2.0, idle_timeout_s=0.5)
    clock = FakeClock(10.0)
    verifier, source, book, _ = make_verifier(tmp_path, config=config, clock=clock)
    frames = episode_frames()
    verifier.start()
    try:
        emit_range(source, frames, 9.0, 9.5)
        with caplog.at_level(logging.DEBUG, logger="cpsa.vision"):
            verifier.observe(detection(Tag.DANGEROUS, 9.0, clock))
        time.sleep(0.05)

        assert verifier.episode_id is None
        assert book.calls == []
        assert "stale vision observation" in caplog.text

        verifier.observe(detection(Tag.DANGEROUS, 9.6, clock))
        wait_until(lambda: verifier.episode_id is not None)
        wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 9.5 - 1e-9)
        assert book.calls
    finally:
        verifier.stop()


# ---------------------------------------------------------------------------
# P-05 lifecycle contract
# ---------------------------------------------------------------------------


def test_start_and_stop_are_idempotent_and_verify_needs_a_start(tmp_path: Path) -> None:
    verifier, source, _, clock = make_verifier(tmp_path)
    with pytest.raises(RuntimeError):
        verifier.verify(detection(Tag.DANGEROUS, 10.2, clock))

    verifier.start()
    verifier.start()
    verifier.stop()
    verifier.stop()

    assert source.start_calls == 1
    assert source.stop_calls == 1
    with pytest.raises(RuntimeError):
        verifier.verify(detection(Tag.DANGEROUS, 10.2, clock))


# ---------------------------------------------------------------------------
# P-03: the fast budget, backlog and source health
# ---------------------------------------------------------------------------


def test_expired_stage_one_budget_returns_timeout_undecided(tmp_path: Path) -> None:
    frames = episode_frames()
    clock = FakeClock(10.0)
    estimator = PoseBook(frames, delay=0.2)
    config = _config(tmp_path, fast_budget_s=0.05, stale_after_s=5.0)
    verifier, source, _, _ = make_verifier(
        tmp_path, config=config, estimator=estimator, clock=clock
    )
    verifier.start()
    try:
        emit_range(source, frames, 9.0, 10.4)
        result = verifier.verify(detection(Tag.DANGEROUS, 10.2, clock))
    finally:
        verifier.stop()

    assert result.verdict is Verdict.UNDECIDED
    assert result.features == {"timeout": 1.0}
    trace = verifier.last_trace
    assert trace is not None
    assert any(record.timeout for record in trace.assessments)


def test_queue_drops_oldest_frames_under_backlog(tmp_path: Path) -> None:
    frames = episode_frames()
    clock = FakeClock(9.0)
    gate = threading.Event()
    estimator = PoseBook(frames, gate=gate)
    config = _config(tmp_path, queue_max=2)
    verifier, source, _, _ = make_verifier(
        tmp_path, config=config, estimator=estimator, clock=clock
    )
    verifier.start()
    try:
        verifier.observe(detection(Tag.DANGEROUS, 9.2, clock))
        source.emit(frames[0])
        wait_until(estimator.started.is_set)
        for frame in frames:
            if 9.1 <= frame.t <= 9.5 and abs(frame.t * 10 - round(frame.t * 10)) < 1e-6:
                source.emit(frame)
        gate.set()
        wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 9.5 - 1e-9)
    finally:
        verifier.stop()

    trace = verifier.last_trace
    assert trace is not None
    times = [record.t for record in trace.frames]
    assert times == pytest.approx([9.0, 9.4, 9.5])


# ---------------------------------------------------------------------------
# AC-013: source and estimator failures fail open
# ---------------------------------------------------------------------------


def test_source_failure_gives_undecided_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    frames = episode_frames()
    clock = FakeClock(10.0)
    verifier, source, _, _ = make_verifier(tmp_path, clock=clock)
    verifier.start()
    try:
        assert verifier.health_error is None
        emit_range(source, frames, 9.0, 10.4)
        source.fail(RuntimeError("camera unplugged"))
        assert isinstance(verifier.health_error, RuntimeError)
        result = verifier.verify(detection(Tag.DANGEROUS, 10.2, clock))
    finally:
        verifier.stop()

    assert result.verdict is Verdict.UNDECIDED
    assert result.features["coverage_ok"] == 0.0
    assert "Vision video source failed" in caplog.text
    trace = verifier.last_trace
    assert trace is not None
    assert trace.assessments[-1].source_healthy is False


def test_a_camera_that_cannot_open_is_reported_through_health_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class UnopenableSource(ManualVideoSource):
        def start(
            self,
            on_frame: Callable[[VideoFrame], None],
            on_error: Callable[[Exception], None] | None = None,
        ) -> None:
            raise FileNotFoundError("/dev/video0")

    source = UnopenableSource()
    verifier = VisionVerifier(
        source,
        PoseBook(episode_frames()),
        None,
        _config(tmp_path),
        FakeClock(9.0),
        fx.LAYOUT,
        log_dir=tmp_path / "logs",
    )
    verifier.start()
    try:
        assert isinstance(verifier.health_error, FileNotFoundError)
        assert "continuing without frames" in caplog.text
    finally:
        verifier.stop()


def test_estimator_failure_gives_undecided_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    frames = episode_frames()
    clock = FakeClock(10.0)
    failing = int(frames[30].seq)
    gate = threading.Event()
    estimator = PoseBook(frames, fail_on={failing}, gate=gate, gate_on=failing)
    verifier, source, _, _ = make_verifier(
        tmp_path,
        config=_config(tmp_path, stale_after_s=5.0, queue_max=64),
        estimator=estimator,
        clock=clock,
    )
    verifier.start()
    results: list[VisionAssessment] = []
    try:
        verifier.observe(detection(Tag.DANGEROUS, 9.2, clock))
        emit_range(source, frames, 9.0, 10.4)
        # Block on the failing frame so the request is registered before the
        # failure force-serves every pending request.
        wait_until(estimator.started.is_set)
        thread = threading.Thread(
            target=lambda: results.append(verifier.verify(detection(Tag.DANGEROUS, 10.2, clock)))
        )
        thread.start()
        wait_until(lambda: bool(verifier._requests))
        gate.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        result = results[0]
    finally:
        gate.set()
        verifier.stop()

    assert result.verdict is Verdict.UNDECIDED
    assert result.features == {"error": 1.0}
    assert "Pose estimation failed" in caplog.text
    trace = verifier.last_trace
    assert trace is not None
    assert any(record.error for record in trace.assessments)


# ---------------------------------------------------------------------------
# AC-012: evidence, trace and report cross-references
# ---------------------------------------------------------------------------


def test_episode_writes_clip_keyframe_trace_and_linked_report(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    clock = FakeClock(10.0)
    from cpsa.adapters.evidence import FileEvidenceSink

    config = _config(
        tmp_path,
        evidence=replace(
            load(SHIPPED).vision.evidence, enabled=True, base_path=tmp_path / "evidence"
        ),
    )
    frames = episode_frames()
    evidence = FileEvidenceSink(tmp_path / "evidence", 7, fx.LAYOUT)
    verifier, source, _, _ = make_verifier(tmp_path, config=config, clock=clock, evidence=evidence)
    verifier.start()
    try:
        emit_range(source, frames, 9.0, 10.4)
        result = verifier.verify(detection(Tag.DANGEROUS, 10.2, clock))
        assert result.verdict is Verdict.CONFIRMED
        clock.t = 10.4
    finally:
        verifier.stop()

    report = verifier.last_report
    assert report is not None
    episode_folder = (
        tmp_path / "evidence" / datetime.now().astimezone().date().isoformat() / report.episode_id
    )
    assert (episode_folder / "keyframe.jpg").is_file()
    assert (episode_folder / "clip.mp4").is_file() or (episode_folder / "clip.avi").is_file()
    trace_path = (
        tmp_path
        / "evidence"
        / clock.now().astimezone().date().isoformat()
        / report.episode_id
        / "trace.jsonl"
    )
    assert trace_path.is_file()

    reports = list((tmp_path / "logs").rglob("reports.jsonl"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8").splitlines()[0])
    assert payload["event"] == "episode_report"
    assert payload["episode_id"] == report.episode_id
    assert payload["trace"] == str(trace_path)
    assert Path(payload["trace"]).is_file()


def test_report_without_evidence_keeps_the_trace_under_the_log_dir(tmp_path: Path) -> None:
    clock = FakeClock(10.0)
    verifier, source, _, _ = make_verifier(tmp_path, clock=clock)
    verifier.start()
    try:
        emit_range(source, episode_frames(), 9.0, 10.4)
        verifier.verify(detection(Tag.DANGEROUS, 10.2, clock))
        clock.t = 10.4
    finally:
        verifier.stop()

    trace = verifier.last_trace
    assert trace is not None
    traces = list((tmp_path / "logs").rglob("trace.jsonl"))
    assert len(traces) == 1
    assert trace.episode_id in str(traces[0])
    assert "vision" in traces[0].parts


class RecordingEvidence:
    """Capture the clip window the verifier writes for each episode."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[np.ndarray, TrackedFrame | None]]] = []
        self.keyframes: list[int] = []
        self.purges: list[datetime] = []

    def write_episode(
        self,
        episode_id: str,
        frames: Sequence[tuple[np.ndarray, TrackedFrame | None]],
        fps: float,
        keyframe: int,
    ) -> None:
        self.calls.append(list(frames))
        self.keyframes.append(keyframe)

    def purge(self, now: datetime) -> None:
        self.purges.append(now)


def test_evidence_around_an_early_peak_survives_a_long_episode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock(9.0)
    base = load(SHIPPED).vision
    config = _config(
        tmp_path,
        idle_timeout_s=0.5,
        analysis=replace(base.analysis, activity_window_s=1.0),
        evidence=replace(
            base.evidence,
            enabled=True,
            base_path=tmp_path / "evidence",
            clip_s=2.0,
        ),
    )
    times = np.arange(9.0, 16.0, fx.DT, dtype=np.float64)
    frames = fx.frames_from_positions(
        times,
        _trajectory(
            (10.0, 11.4),
            EYE,
            HandShape.FIST,
            [(10.0, 1.0), (10.1, 0.05), (11.0, 0.05), (11.4, 1.2)],
            times,
        ),
    )
    evidence = RecordingEvidence()
    verifier, source, _, _ = make_verifier(tmp_path, config=config, clock=clock, evidence=evidence)
    verifier.start()
    try:
        emit_range(source, frames, 9.0, 10.0)
        clock.t = 10.0
        verifier.verify(detection(Tag.DANGEROUS, 9.8, clock))
        previous = 10.0
        t = 10.3
        while t <= 15.0 + 1e-9:
            clock.t = t
            emit_range(source, frames, previous + 0.5 * fx.DT, t)
            wait_until(lambda t=t: (verifier.analyzed_until_t or 0.0) >= t - 1e-9)
            verifier.verify(detection(Tag.DANGEROUS, t - 0.2, clock))
            previous = t
            t += 0.3
        clock.t = 15.0
    finally:
        verifier.stop()

    report = verifier.last_report
    assert report is not None
    peak = next(contact for contact in report.contacts if contact.region is FaceRegion.EYES)
    # The rolling buffer only keeps clip + activity + idle + 1 s, so by 15 s the
    # frames around the 10 s peak are long gone from it.
    assert peak.t_start + config.evidence.clip_s + 1.0 < 15.0
    assert len(evidence.calls) == 1
    clip = evidence.calls[0]
    tracked = [frame for _, frame in clip if frame is not None]
    assert tracked
    assert min(frame.t for frame in tracked) <= peak.t_start - 0.5
    assert max(frame.t for frame in tracked) >= peak.t_start + 0.5
    assert max(frame.t for frame in tracked) - min(frame.t for frame in tracked) <= (
        config.evidence.clip_s + 1e-6
    )
    keyframe = evidence.keyframes[0]
    assert abs(float(tracked[keyframe].t) - peak.t_start) <= fx.DT + 1e-9
    assert "No retained frames cover the peak window" not in caplog.text


def test_trace_round_trips_through_json_lines(tmp_path: Path) -> None:
    clock = FakeClock(10.0)
    verifier, source, _, _ = make_verifier(tmp_path, clock=clock)
    verifier.start()
    try:
        emit_range(source, episode_frames(), 9.0, 10.4)
        verifier.verify(detection(Tag.DANGEROUS, 10.2, clock))
    finally:
        verifier.stop()

    trace = verifier.last_trace
    assert trace is not None
    path = list((tmp_path / "logs").rglob("trace.jsonl"))[0]
    read = type(trace).read(path)
    assert read == trace


# ---------------------------------------------------------------------------
# AC-011 at dispatcher level: latest severity, upgrade, downgrade, rejection
# ---------------------------------------------------------------------------


def test_follow_ups_use_the_latest_severity_at_dispatcher_level(tmp_path: Path) -> None:
    clock = FakeClock(9.0)
    config = _config(tmp_path, idle_timeout_s=100.0, queue_max=64)
    verifier, source, _, _ = make_verifier(tmp_path, config=config, clock=clock)
    slot: LatestSlot[Detection] = LatestSlot()
    actuator = RecordingActuator("meta_AA", ActuatorKind.METAMOTION)
    sink = MemorySink()
    gui = RecordingGui()
    dispatcher = Dispatcher(
        slot,
        verifier,
        [actuator],
        sink,
        gui,
        clock,
        cooldown_s=0.2,
        deadline_s=5.0,
    )
    frames = episode_frames()
    verifier.start()
    dispatcher.start()
    try:
        emit_range(source, frames, 9.0, 10.4)
        clock.t = 10.2
        first = dispatcher.handle(detection(Tag.DANGEROUS, 10.2, clock))
        assert first is not None and first.decision is not None
        assert first.decision.verdict is Verdict.CONFIRMED
        assert first.decision.severity is Tag.DANGEROUS
        assert actuator.executed[-1] == STRONG_META

        emit_range(source, frames, 10.5, 11.4)
        wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 11.4 - 1e-9)
        clock.t = 11.2
        follow_up = dispatcher.handle(detection(Tag.DANGEROUS, 11.2, clock))
        assert follow_up is not None and follow_up.decision is not None
        assert follow_up.decision.verdict is Verdict.CONFIRMED
        assert follow_up.decision.severity is Tag.DANGEROUS
        assert follow_up.decision.assessment is not None
        assert follow_up.decision.assessment.features["from_episode"] == 1.0

        emit_range(source, frames, 11.5, 13.1)
        wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 13.1 - 1e-9)
        clock.t = 12.9
        downgrade = dispatcher.handle(detection(Tag.DANGEROUS, 12.9, clock))
        assert downgrade is not None and downgrade.decision is not None
        assert downgrade.decision.verdict is Verdict.CONFIRMED
        assert downgrade.decision.severity is Tag.NON_DANGEROUS
        assert actuator.executed[-1] == MILD_META

        emit_range(source, frames, 13.2, 14.1)
        wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 14.1 - 1e-9)
        clock.t = 13.9
        late = dispatcher.handle(detection(Tag.DANGEROUS, 13.9, clock))
        assert late is not None and late.decision is not None
        assert late.decision.severity is Tag.NON_DANGEROUS
        assert late.decision.assessment is not None
        assert late.decision.assessment.features["from_episode"] == 1.0

        emit_range(source, frames, 14.2, 20.2)
        wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 20.2 - 1e-9)
        clock.t = 20.0
        executed = len(actuator.executed)
        inactive = dispatcher.handle(detection(Tag.DANGEROUS, 20.0, clock))
        assert inactive is not None and inactive.decision is not None
        assert inactive.decision.verdict is Verdict.REJECTED
        assert not inactive.decision.actuated
        assert len(actuator.executed) == executed
    finally:
        dispatcher.stop()
        verifier.stop()


def test_follow_ups_after_a_long_contact_use_its_end_time(tmp_path: Path) -> None:
    """AC-011: a contact longer than the activity window stays recent until its end."""
    clock = FakeClock(9.0)
    config = _config(tmp_path, idle_timeout_s=100.0, queue_max=64)
    times = np.arange(9.0, 23.0, fx.DT, dtype=np.float64)
    frames = fx.frames_from_positions(
        times,
        _trajectory(
            (9.6, 17.6),
            EYE,
            HandShape.FIST,
            [(9.6, 1.0), (9.8, 0.05), (17.3, 0.05), (17.6, 1.2)],
            times,
        ),
    )
    contact = detect_contacts(frames, fx.LAYOUT, config.analysis)[0]
    end = contact.t_start + contact.duration_s
    window = config.analysis.activity_window_s
    assert end - contact.t_start > window

    verifier, source, _, _ = make_verifier(
        tmp_path, config=config, clock=clock, estimator=PoseBook(frames)
    )
    slot: LatestSlot[Detection] = LatestSlot()
    actuator = RecordingActuator("meta_AA", ActuatorKind.METAMOTION)
    sink = MemorySink()
    gui = RecordingGui()
    dispatcher = Dispatcher(
        slot,
        verifier,
        [actuator],
        sink,
        gui,
        clock,
        cooldown_s=0.2,
        deadline_s=5.0,
    )
    previous = 0.0
    verifier.start()
    dispatcher.start()
    try:
        emitted = emit_range(source, frames, 9.0, 11.6)
        previous = emitted[-1].t
        clock.t = 11.6
        first = dispatcher.handle(detection(Tag.DANGEROUS, 11.4, clock))
        assert first is not None and first.decision is not None
        assert first.decision.verdict is Verdict.CONFIRMED
        assert first.decision.severity is Tag.DANGEROUS

        def emit_until(target: float) -> None:
            nonlocal previous
            while previous < target - 1e-9:
                step = min(previous + 0.6, target)
                emit_range(source, frames, previous + 0.5 * fx.DT, step)
                wait_until(
                    lambda step=step: (verifier.analyzed_until_t or 0.0) >= step - 0.1 - 1e-9
                )
                previous = step

        # A follow-up inside the window of the 7 s contact keeps the latest
        # severity, even though the contact started long before (R-14).
        follow_up_at = end + 1.6
        emit_until(follow_up_at + 0.4)
        clock.t = follow_up_at + 0.2
        follow_up = dispatcher.handle(detection(Tag.DANGEROUS, follow_up_at, clock))
        assert follow_up is not None and follow_up.decision is not None
        assert follow_up.decision.verdict is Verdict.CONFIRMED
        assert follow_up.decision.severity is Tag.DANGEROUS
        assert follow_up.decision.assessment is not None
        assert follow_up.decision.assessment.features["from_episode"] == 1.0

        # Once one activity window passed the end, the episode is inactive.
        late_at = end + window + 1.5
        emit_until(late_at + 0.4)
        clock.t = late_at
        executed = len(actuator.executed)
        late = dispatcher.handle(detection(Tag.DANGEROUS, late_at - 0.2, clock))
        assert late is not None and late.decision is not None
        assert late.decision.verdict is Verdict.REJECTED
        assert not late.decision.actuated
        assert len(actuator.executed) == executed
    finally:
        dispatcher.stop()
        verifier.stop()


# ---------------------------------------------------------------------------
# Stub regression (AC-013): the stub path has no episode side effects
# ---------------------------------------------------------------------------


def test_stub_verifier_still_returns_undecided() -> None:
    from cpsa.adapters.verifier_stub import StubVerifier

    assessment = StubVerifier().verify(
        Detection(Tag.DANGEROUS, datetime.now(timezone.utc), 0.0, ())
    )

    assert isinstance(assessment, VisionAssessment)
    assert assessment.verdict is Verdict.UNDECIDED


# ---------------------------------------------------------------------------
# F2: live vision observer hooks
# ---------------------------------------------------------------------------


class _RecordingObserver:
    """Record the observer call order on one replayed episode."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def on_episode(self, episode_id: str, t_start: float, active: bool) -> None:
        self.calls.append(("episode", episode_id, t_start, active))

    def on_tracked(self, tracked: TrackedFrame) -> None:
        self.calls.append(("tracked", int(tracked.seq)))

    def on_report(self, report: EpisodeReport) -> None:
        self.calls.append(("report", report))


class _RaisingObserver:
    """Every callback fails: the verifier must log it and swallow it."""

    def on_episode(self, episode_id: str, t_start: float, active: bool) -> None:
        raise RuntimeError("dashboard down")

    def on_tracked(self, tracked: TrackedFrame) -> None:
        raise RuntimeError("dashboard down")

    def on_report(self, report: EpisodeReport) -> None:
        raise RuntimeError("dashboard down")


def replay_observed_episode(tmp_path: Path, observer: object | None = None) -> EpisodeReport | None:
    """Replay the fixtures until one episode ends, with an optional observer."""
    frames = episode_frames()
    source = ManualVideoSource()
    clock = FakeClock(9.0)
    config = _config(
        tmp_path,
        idle_timeout_s=0.2,
        analysis=replace(load(SHIPPED).vision.analysis, activity_window_s=0.5),
    )
    verifier = VisionVerifier(
        source,
        PoseBook(frames),
        None,
        config,
        clock,
        fx.LAYOUT,
        log_dir=tmp_path / "logs",
        observer=observer,  # type: ignore[arg-type]
    )
    verifier.start()
    try:
        emit_range(source, frames, 9.0, 10.4)
        verifier.observe(detection(Tag.DANGEROUS, 10.2, clock))
        wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 10.4 - 1e-9)
        clock.t = 30.0
        wait_until(lambda: verifier.last_report is not None)
        return verifier.last_report
    finally:
        verifier.stop()


def test_observer_receives_episode_start_tracked_frames_end_and_report(tmp_path: Path) -> None:
    observer = _RecordingObserver()

    report = replay_observed_episode(tmp_path, observer)

    assert report is not None
    kinds = [call[0] for call in observer.calls]
    assert kinds[0] == "episode"
    assert observer.calls[0][3] is True
    assert kinds[-2:] == ["episode", "report"]
    assert observer.calls[-2][3] is False
    assert observer.calls[-1] == ("report", report)
    assert all(kind == "tracked" for kind in kinds[1:-2])
    assert kinds.count("tracked") > 0
    episode_ids = {call[1] for call in observer.calls if call[0] == "episode"}
    assert episode_ids == {report.episode_id}


def test_a_raising_observer_is_logged_and_does_not_change_the_report(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    baseline = replay_observed_episode(tmp_path)
    with caplog.at_level(logging.ERROR):
        observed = replay_observed_episode(tmp_path, _RaisingObserver())

    assert baseline is not None and observed is not None
    assert observed.contact_count == baseline.contact_count
    assert observed.peak_severity is baseline.peak_severity
    assert observed.labels == baseline.labels
    assert "Vision observer failed" in caplog.text
