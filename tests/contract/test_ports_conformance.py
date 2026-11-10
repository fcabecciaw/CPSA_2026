"""Adapters and test fakes satisfy the port protocols they stand in for."""

import time
from pathlib import Path

import pytest

from cpsa import ports
from cpsa.adapters.clock import SystemClock
from cpsa.adapters.event_log import JsonlEventSink
from cpsa.adapters.gui_null import NullGui
from cpsa.adapters.verifier_stub import StubVerifier
from cpsa.domain.model import DecisionRecord, Detection, Tag, Verdict
from tests import fakes


@pytest.mark.parametrize(
    ("instance", "port"),
    [
        (StubVerifier(), ports.Verifier),
        (NullGui(), ports.Gui),
        (SystemClock(), ports.Clock),
        (JsonlEventSink(Path("unused")), ports.EventSink),
        (fakes.FakeClock(), ports.Clock),
        (fakes.FakeExtractor(), ports.FeatureExtractor),
        (fakes.FakeClassifier(), ports.Classifier),
        (fakes.ScriptedVerifier(), ports.Verifier),
        (fakes.RecordingActuator("meta_A"), ports.Actuator),
        (fakes.FakeSensorSource(), ports.SensorSource),
        (fakes.MemorySink(), ports.EventSink),
        (fakes.RecordingGui(), ports.Gui),
    ],
    ids=lambda value: type(value).__name__,
)
def test_implements_port(instance: object, port: type) -> None:
    assert isinstance(instance, port)


def test_stub_verifier_is_always_undecided() -> None:
    clock = fakes.FakeClock()
    for tag in Tag:
        assessment = StubVerifier().verify(Detection(tag, clock.now(), 0.0, ()))
        assert assessment.verdict is Verdict.UNDECIDED
        assert assessment.severity is None


def test_null_gui_accepts_the_full_lifecycle() -> None:
    gui = NullGui()
    gui.start()
    gui.publish(DecisionRecord(fakes.EPOCH, 0.0, Tag.DANGEROUS, True, None))
    gui.stop()


def test_system_clock_is_monotonic_and_timezone_aware() -> None:
    clock = SystemClock()
    first = clock.monotonic()
    time.sleep(0.001)
    assert clock.monotonic() > first
    assert clock.now().tzinfo is not None
