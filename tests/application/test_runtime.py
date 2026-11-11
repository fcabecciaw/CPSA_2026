import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from cpsa.application.config import load
from cpsa.application.runtime import Runtime, build_runtime, build_vision_runtime
from cpsa.application.vision_verifier import VisionVerifier
from cpsa.domain.model import ActuatorKind, Detection, SampleKind, Side, Tag, Verdict
from tests.application.test_config import SHIPPED
from tests.fakes import (
    FakeClassifier,
    FakeClock,
    FakeExtractor,
    FakeSensorSource,
    MemorySink,
    RecordingActuator,
    RecordingGui,
    ScriptedVerifier,
)


class Parts:
    def __init__(self) -> None:
        self.journal: list[str] = []
        self.clock = FakeClock()
        self.source = FakeSensorSource(journal=self.journal)
        self.actuator = RecordingActuator("meta_AA", ActuatorKind.METAMOTION, self.journal)
        self.sink = MemorySink()
        self.gui = RecordingGui(self.journal)
        self.runtime = build_runtime(
            load(SHIPPED),
            source=self.source,
            extractor=FakeExtractor(),
            classifier=FakeClassifier([Tag.DANGEROUS]),
            verifier=ScriptedVerifier(),
            actuators=[self.actuator],
            sink=self.sink,
            gui=self.gui,
            clock=self.clock,
        )


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_start_and_stop_order() -> None:
    parts = Parts()
    parts.runtime.start()
    parts.runtime.stop()
    assert parts.journal == [
        "gui.start",
        "sensors.start",
        "gui.stop",
        "sensors.stop",
        "actuator:meta_AA.stop",
    ]


def test_samples_flow_from_sensors_to_actuation() -> None:
    parts = Parts()
    parts.runtime.start()
    try:
        emit = parts.source.on_sample
        assert emit is not None
        for _ in range(150 + 2 * 75):  # third window is the first classified one
            for side in (Side.LEFT, Side.RIGHT):
                emit(side, SampleKind.ACC, (0.0, 0.0, 1000.0), parts.clock.t)
                emit(side, SampleKind.GYR, (0.0, 0.0, 0.0), parts.clock.t)
                emit(side, SampleKind.QUAT, (0.0, 0.0, 0.0, 10000.0), parts.clock.t)
        assert wait_until(lambda: len(parts.sink.records) == 1)
    finally:
        parts.runtime.stop()

    record = parts.sink.records[0]
    assert record.tag is Tag.DANGEROUS and record.decision is not None
    assert parts.actuator.executed == [{"duty": 100, "duration": 900}]


def test_failing_stop_does_not_prevent_stopping_the_rest() -> None:
    parts = Parts()

    def broken_stop() -> None:
        raise RuntimeError("gui crashed")

    parts.gui.stop = broken_stop  # type: ignore[method-assign]
    parts.runtime.start()
    parts.runtime.stop()
    assert parts.source.stopped and parts.actuator.stopped


def test_failed_start_stops_started_components() -> None:
    parts = Parts()

    def broken_start(_: object) -> None:
        raise RuntimeError("BLE adapter missing")

    parts.source.start = broken_start  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        parts.runtime.start()
    assert parts.gui.stopped and parts.actuator.stopped


def test_failed_camera_open_keeps_the_runtime_running(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    from cpsa.application.vision_verifier import VisionVerifier
    from tests.application.test_vision_verifier import (
        ManualVideoSource,
        PoseBook,
        _config,
        episode_frames,
    )
    from tests.unit import vision_fixtures as fx

    class FailingVideoSource(ManualVideoSource):
        def start(self, on_frame, on_error=None) -> None:
            raise RuntimeError("camera busy")

    frames = episode_frames()
    clock = FakeClock()
    verifier = VisionVerifier(
        FailingVideoSource(),
        PoseBook(frames),
        None,
        _config(tmp_path, fast_budget_s=0.05),
        clock,
        fx.LAYOUT,
        log_dir=tmp_path / "logs",
    )
    source = FakeSensorSource()
    actuator = RecordingActuator("meta_AA", ActuatorKind.METAMOTION)
    sink = MemorySink()
    runtime = build_runtime(
        load(SHIPPED),
        source=source,
        extractor=FakeExtractor(),
        classifier=FakeClassifier([Tag.DANGEROUS]),
        verifier=verifier,
        actuators=[actuator],
        sink=sink,
        gui=RecordingGui(),
        clock=clock,
    )
    runtime.start()
    try:
        emit = source.on_sample
        assert emit is not None
        for _ in range(150 + 2 * 75):
            for side in (Side.LEFT, Side.RIGHT):
                emit(side, SampleKind.ACC, (0.0, 0.0, 1000.0), clock.t)
                emit(side, SampleKind.GYR, (0.0, 0.0, 0.0), clock.t)
                emit(side, SampleKind.QUAT, (0.0, 0.0, 0.0, 10000.0), clock.t)
        assert wait_until(lambda: len(sink.records) == 1)
        assert (
            verifier.verify(Detection(Tag.DANGEROUS, clock.now(), 10.2, ())).verdict
            is Verdict.UNDECIDED
        )
    finally:
        runtime.stop()

    record = sink.records[0]
    assert record.decision is not None
    assert record.decision.verdict is Verdict.UNDECIDED
    assert record.decision.actuated
    assert actuator.executed
    assert "Vision video source failed to open" in caplog.text


def test_vision_verifier_lifecycle_and_imu_tee(tmp_path) -> None:
    from cpsa.application.vision_verifier import VisionVerifier
    from tests.application.test_vision_verifier import (
        ManualVideoSource,
        PoseBook,
        _config,
        episode_frames,
    )
    from tests.unit import vision_fixtures as fx

    config = load(SHIPPED)
    frames = episode_frames()
    video = ManualVideoSource()
    clock = FakeClock()
    verifier = VisionVerifier(
        video,
        PoseBook(frames),
        None,
        _config(tmp_path),
        clock,
        fx.LAYOUT,
        log_dir=tmp_path / "logs",
    )
    source = FakeSensorSource()
    runtime = build_runtime(
        config,
        source=source,
        extractor=FakeExtractor(),
        classifier=FakeClassifier([Tag.DANGEROUS]),
        verifier=verifier,
        actuators=[RecordingActuator("meta_AA", ActuatorKind.METAMOTION)],
        sink=MemorySink(),
        gui=RecordingGui(),
        clock=clock,
    )
    runtime.start()
    try:
        assert video.start_calls == 1
        assert source.on_sample is not None
        source.on_sample(Side.LEFT, SampleKind.ACC, (1000.0, 0.0, 0.0), 1.0)
        source.on_sample(Side.LEFT, SampleKind.GYR, (0.0, 0.0, 0.0), 1.0)
        source.on_sample(Side.RIGHT, SampleKind.ACC, (0.0, 1000.0, 0.0), 1.0)
        assert verifier.imu_sample_count(Side.LEFT) == 1
        assert verifier.imu_sample_count(Side.RIGHT) == 1
        assert verifier.analyzed_until_t is None
    finally:
        runtime.stop()

    assert video.stop_calls == 1
    with pytest.raises(RuntimeError):
        verifier.verify(Detection(Tag.DANGEROUS, clock.now(), 10.2, ()))


def test_imu_status_needs_a_fresh_sample_from_both_sides() -> None:
    parts = Parts()
    parts.runtime.start()
    try:
        emit = parts.source.on_sample
        assert emit is not None
        assert parts.runtime.imu_status(parts.clock.t) == "lost"

        emit(Side.LEFT, SampleKind.ACC, (0.0, 0.0, 1000.0), parts.clock.t)
        assert parts.runtime.imu_status(parts.clock.t) == "lost"

        emit(Side.RIGHT, SampleKind.ACC, (0.0, 0.0, 1000.0), parts.clock.t)
        assert parts.runtime.imu_status(parts.clock.t) == "connected"

        # Exactly 1.0 s after the last sample is still "within the last second".
        parts.clock.advance(1.0)
        assert parts.runtime.imu_status(parts.clock.t) == "connected"

        parts.clock.advance(0.001)
        assert parts.runtime.imu_status(parts.clock.t) == "lost"

        emit(Side.LEFT, SampleKind.ACC, (0.0, 0.0, 1000.0), parts.clock.t)
        assert parts.runtime.imu_status(parts.clock.t) == "lost"

        emit(Side.RIGHT, SampleKind.ACC, (0.0, 0.0, 1000.0), parts.clock.t)
        assert parts.runtime.imu_status(parts.clock.t) == "connected"
    finally:
        parts.runtime.stop()


def test_imu_status_is_lost_before_the_first_sample() -> None:
    parts = Parts()
    assert parts.runtime.imu_status(parts.clock.t) == "lost"


class FakeVisionVerifier:
    """Duck-typed ``VisionVerifier`` for the camera-only runtime (F10)."""

    def __init__(self) -> None:
        self.verifications: list[Detection] = []
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def verify(self, detection: Detection) -> Verdict:
        self.verifications.append(detection)
        return Verdict.CONFIRMED


def vision_parts(
    journal: list[str], actuators: list[RecordingActuator]
) -> tuple[
    tuple[Runtime, Callable[[Detection], None]],
    FakeVisionVerifier,
    MemorySink,
    FakeClock,
]:
    clock = FakeClock()
    verifier = FakeVisionVerifier()
    sink = MemorySink()
    built = build_vision_runtime(
        load(SHIPPED),
        verifier=cast(VisionVerifier, verifier),
        actuators=actuators,
        sink=sink,
        gui=RecordingGui(journal),
        clock=clock,
    )
    return built, verifier, sink, clock


def test_a_vision_detection_reaches_the_actuator_and_the_event_sink() -> None:
    journal: list[str] = []
    actuator = RecordingActuator("meta_AA", ActuatorKind.METAMOTION, journal)
    (runtime, on_detection), verifier, sink, clock = vision_parts(journal, [actuator])

    runtime.start()
    try:
        on_detection(Detection(Tag.DANGEROUS, clock.now(), 10.2, ()))
        assert wait_until(lambda: len(sink.records) == 1)
    finally:
        runtime.stop()

    record = sink.records[0]
    assert record.tag is Tag.DANGEROUS and record.decision is not None
    assert record.decision.verdict is Verdict.CONFIRMED
    assert actuator.executed
    assert verifier.verifications[0].tag is Tag.DANGEROUS
    assert verifier.started and verifier.stopped
    assert journal == ["gui.start", "gui.stop", "actuator:meta_AA.stop"]


def test_a_vision_detection_without_an_actuator_is_still_recorded() -> None:
    journal: list[str] = []
    (runtime, on_detection), _, sink, clock = vision_parts(journal, [])

    runtime.start()
    try:
        on_detection(Detection(Tag.NON_DANGEROUS, clock.now(), 10.2, ()))
        assert wait_until(lambda: len(sink.records) == 1)
    finally:
        runtime.stop()

    record = sink.records[0]
    assert record.decision is not None
    assert record.decision.verdict is Verdict.CONFIRMED
    assert record.decision.actuations == ()
    assert not record.decision.actuated


def test_the_vision_runtime_has_no_imu_source() -> None:
    journal: list[str] = []
    (runtime, _), _, _, clock = vision_parts(journal, [])

    runtime.start()
    try:
        assert runtime.imu_status(clock.t) == "lost"
    finally:
        runtime.stop()


class RecordingPublisher:
    """Record the ``medical.*`` messages a ``WebGui`` publishes."""

    def __init__(self) -> None:
        self.states: list[tuple[str, Mapping[str, object]]] = []
        self.events: list[tuple[str, Mapping[str, object]]] = []

    def state(self, type: str, data: Mapping[str, object]) -> None:
        self.states.append((type, data))

    def event(self, type: str, data: Mapping[str, object]) -> None:
        self.events.append((type, data))

    def payloads(self, message_type: str) -> list[Mapping[str, object]]:
        """Every event payload of ``message_type``, in publication order."""
        return [data for kind, data in self.events if kind == message_type]


def real_vision_wiring(
    tmp_path: Path, actuators: list[RecordingActuator]
) -> tuple[Runtime, Any, list[Any], RecordingPublisher, MemorySink, FakeClock, VisionVerifier]:
    """Wire the real trigger and continuous verifier through ``build_vision_runtime``.

    The F8 callback forwards to the runtime's slot exactly as the medical mode
    does, and ``WebGui`` publishes the decision (AC-023, AC-024).
    """
    from cpsa.adapters.gui_web import WebGui
    from cpsa.vision.trigger import TriggerConfig, VisionTrigger
    from tests.application.test_vision_verifier import (
        ManualVideoSource,
        PoseBook,
        _config,
        episode_frames,
    )
    from tests.unit import vision_fixtures as fx

    frames = episode_frames()
    clock = FakeClock(9.0)
    video = ManualVideoSource()
    feed: list[Callable[[Detection], None]] = []

    def on_detection(detection: Detection) -> None:
        if feed:
            feed[0](detection)

    verifier = VisionVerifier(
        video,
        PoseBook(frames),
        None,
        # The test delivers the whole fixture at once, so the pre-roll must
        # retain the firing window until the dispatcher asks for the assessment;
        # the board delivers frames at the camera rate and keeps the default.
        _config(tmp_path, queue_max=512, preroll_s=10.0),
        clock,
        fx.LAYOUT,
        log_dir=tmp_path / "logs",
        continuous=True,
        trigger=VisionTrigger(TriggerConfig()),
        on_detection=on_detection,
    )
    publisher = RecordingPublisher()
    sink = MemorySink()
    runtime, dispatch = build_vision_runtime(
        load(SHIPPED),
        verifier=verifier,
        actuators=actuators,
        sink=sink,
        gui=WebGui(publisher, fx.LAYOUT, 640, 480, clock, trigger="vision"),
        clock=clock,
    )
    feed.append(dispatch)
    return runtime, video, list(frames), publisher, sink, clock, verifier


def replay_real_contact_episode(
    tmp_path: Path, actuators: list[RecordingActuator], *, end: bool = False
) -> tuple[RecordingPublisher, MemorySink]:
    """Replay the fixture through the real vision pipeline until it decides.

    With ``end`` the fake clock then jumps past ``idle_timeout_s`` so the episode
    ends and the camera-only tag change is published too.
    """
    from tests.application.test_vision_verifier import emit_range

    runtime, video, frames, publisher, sink, clock, verifier = real_vision_wiring(
        tmp_path, actuators
    )
    runtime.start()
    try:
        emit_range(video, frames, 9.0, 16.0)
        assert wait_until(lambda: confirmed_decisions(publisher))
        if end:
            # Analyze the whole replay before ending it, so nothing it still
            # fires is dispatched after the tag change.
            assert wait_until(lambda: (verifier.analyzed_until_t or 0.0) >= 16.0 - 1e-9)
            clock.t = 200.0
            assert wait_until(lambda: bool(cleared_decisions(publisher)))
    finally:
        runtime.stop()
    return publisher, sink


def confirmed_decisions(publisher: RecordingPublisher) -> list[Mapping[str, object]]:
    """The published decisions that carry a verdict (tag-only updates have none)."""
    return [
        payload
        for payload in publisher.payloads("medical.decision")
        if payload["verdict"] is not None
    ]


def cleared_decisions(publisher: RecordingPublisher) -> list[Mapping[str, object]]:
    """The published tag changes that clear the episode tag (FR-029)."""
    return [
        payload
        for payload in publisher.payloads("medical.decision")
        if payload["tag"] == "NON_STEREOTIPY"
    ]


def test_a_real_contacts_fire_becomes_one_medical_decision(tmp_path: Path) -> None:
    actuator = RecordingActuator("meta_AA", ActuatorKind.METAMOTION)

    publisher, sink = replay_real_contact_episode(tmp_path, [actuator], end=True)

    decisions = confirmed_decisions(publisher)
    assert len(decisions) == 1
    assert decisions[0]["verdict"] == "CONFIRMED"
    assert decisions[0]["tag"] in {"NON_DANGEROUS", "DANGEROUS"}
    assert decisions[0]["vision"] is not None
    record = sink.records[0]
    assert record.decision is not None and record.decision.verdict is Verdict.CONFIRMED
    assert record.tag.name == decisions[0]["tag"]
    assert record.decision.actuated
    assert actuator.executed
    # AC-023: the episode end yields exactly one NON_STEREOTIPY tag change.
    clears = cleared_decisions(publisher)
    assert len(clears) == 1
    assert clears[0]["tag_changed"] is True
    assert clears[0]["verdict"] is None


def test_a_real_contacts_fire_without_an_actuator_still_publishes_the_decision(
    tmp_path: Path,
) -> None:
    publisher, sink = replay_real_contact_episode(tmp_path, [])

    decisions = confirmed_decisions(publisher)
    assert len(decisions) == 1
    assert decisions[0]["verdict"] == "CONFIRMED"
    assert decisions[0]["actuations"] == []
    record = sink.records[0]
    assert record.decision is not None
    assert not record.decision.actuated
