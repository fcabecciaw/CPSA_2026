import time
from collections.abc import Callable

import pytest

from cpsa.application.config import load
from cpsa.application.runtime import build_runtime
from cpsa.domain.model import ActuatorKind, SampleKind, Side, Tag
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
