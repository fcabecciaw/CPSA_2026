import time
from collections.abc import Callable

import pytest

from cpsa.application.config import load
from cpsa.application.runtime import build_runtime
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
