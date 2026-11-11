"""``MedicalMode`` start/stop against fake board adapters (F4, IF-704)."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

import cpsa.board.medical_mode as medical_mode
from cpsa.adapters.gui_web import WebGui
from cpsa.application.config import Config, load
from cpsa.board.__main__ import BoardAdapters, VisionStartupError
from cpsa.board.medical_mode import STATE_PERIOD_S, MedicalMode
from cpsa.domain.model import Detection, SampleKind, Side, Verdict
from cpsa.ports import SensorSource
from cpsa.supervisor.contract import ModeContext, ModeStartError
from tests.application.test_config import SHIPPED
from tests.fakes import (
    FakeClassifier,
    FakeClock,
    FakeExtractor,
    FakeSensorSource,
    MemorySink,
    RecordingActuator,
)

BOTH = {"CPSA_L2", "CPSA_R2"}


class FakePublisher:
    """Record every state and event; the mode passes plain ``data`` objects."""

    def __init__(self) -> None:
        self.states: list[tuple[str, Mapping[str, Any]]] = []
        self.events: list[tuple[str, Mapping[str, Any]]] = []

    def state(self, type: str, data: Mapping[str, Any]) -> None:
        self.states.append((type, dict(data)))

    def event(self, type: str, data: Mapping[str, Any]) -> None:
        self.events.append((type, dict(data)))

    def payloads(self, message_type: str) -> list[Mapping[str, Any]]:
        return [data for kind, data in self.states if kind == message_type]


class FakeVerifier:
    """A ``Verifier`` with the public camera-health attribute and the stop hook."""

    def __init__(self) -> None:
        self.stopped = False
        self.health_error: BaseException | None = None

    def verify(self, detection: Detection) -> Verdict:
        return Verdict.UNDECIDED

    def stop(self) -> None:
        self.stopped = True


class BlockingRuntime:
    """A ``Runtime`` stand-in whose ``start`` blocks like an opening camera."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.stopped = False

    def start(self) -> None:
        self.started.set()
        self.release.wait(5.0)

    def stop(self) -> None:
        self.stopped = True

    def imu_status(self, now: float) -> str:
        return "lost"


class Harness:
    """Fake board adapters plus the two helpers ``MedicalMode`` calls."""

    def __init__(
        self,
        *,
        missing: frozenset[str] | None = frozenset(),
        config: Config | None = None,
    ) -> None:
        self.config = load(SHIPPED) if config is None else config
        self.journal: list[str] = []
        self.source = FakeSensorSource([BOTH], self.journal)
        self.verifier = FakeVerifier()
        self.actuator = RecordingActuator("meta_AA", journal=self.journal)
        self.acts_started = False
        self.missing = missing
        self.captured: dict[str, Any] = {}
        self.publisher = FakePublisher()
        self.clock = FakeClock()
        self.context_value: ModeContext | None = None
        self.discovery_started = threading.Event()
        self.classifier_started = threading.Event()
        self.classifier_gate: threading.Event | None = None

    def adapters(
        self,
        config: Config,
        *,
        gui: object | None = None,
        observer: object | None = None,
        clock: object | None = None,
    ) -> BoardAdapters:
        self.captured = {"gui": gui, "observer": observer, "clock": clock}
        return BoardAdapters(
            source=self.source,
            load_classifier=self._load_classifier,
            start_actuators=self._start_actuators,
            verifier=self.verifier,
            gui=gui,  # type: ignore[arg-type]
            sink=MemorySink(),
            clock=clock if clock is not None else self.clock,  # type: ignore[arg-type]
        )

    def discover(
        self, source: SensorSource, settings: object, stop_event: threading.Event
    ) -> set[str] | None:
        """Mimic the real discovery: only the supervisor's event cancels it."""
        if self.missing is None:
            self.discovery_started.set()
            if stop_event.wait(10.0):
                return None
            return set()
        return set(self.missing)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(medical_mode, "board_adapters", self.adapters)
        monkeypatch.setattr(medical_mode, "discover_bluecoins", self.discover)

    def context(self) -> ModeContext:
        self.context_value = ModeContext(self.publisher, self.config, threading.Event())
        return self.context_value

    def _load_classifier(self) -> tuple[FakeExtractor, FakeClassifier]:
        self.classifier_started.set()
        gate = self.classifier_gate
        if gate is not None:
            gate.wait(5.0)
        return FakeExtractor(), FakeClassifier()

    def _start_actuators(self) -> list[RecordingActuator]:
        self.acts_started = True
        return [self.actuator]


def fed_samples(harness: Harness) -> None:
    """Emit one fresh sample per wrist, stamped with the monotonic clock."""
    emit = harness.source.on_sample
    assert emit is not None
    for side in (Side.LEFT, Side.RIGHT):
        emit(side, SampleKind.ACC, (0.0, 0.0, 1000.0), time.monotonic())


def wait_for_imu(harness: Harness, status: str, timeout: float = 3.0) -> bool:
    """Keep the wrists fed until a published state reports ``status``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        fed_samples(harness)
        if any(payload["imu"] == status for payload in harness.publisher.payloads("medical.state")):
            return True
        time.sleep(0.05)
    return False


def test_missing_bluecoin_raises_imu_not_found_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(missing=frozenset({"CPSA_L2"}))
    harness.install(monkeypatch)
    mode = MedicalMode(harness.config)

    with pytest.raises(ModeStartError) as info:
        mode.start(harness.context())

    assert info.value.code == "imu_not_found"
    assert "CPSA_L2" in info.value.message
    assert harness.publisher.payloads("medical.state")[0]["imu"] == "searching"
    assert harness.verifier.stopped
    assert not harness.acts_started
    assert mode.health()["errors"] == ["imu_not_found"]
    assert mode.health()["imu"] == "n/a"


def start_in_background(
    mode: MedicalMode, ctx: ModeContext
) -> tuple[threading.Thread, list[BaseException]]:
    """Run ``mode.start`` on a thread and collect what it raised."""
    failures: list[BaseException] = []

    def run() -> None:
        try:
            mode.start(ctx)
        except Exception as exc:  # collected and asserted by the caller
            failures.append(exc)

    thread = threading.Thread(target=run, name="test-medical-start")
    thread.start()
    return thread, failures


def test_stop_during_discovery_returns_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(missing=None)
    harness.install(monkeypatch)
    mode = MedicalMode(harness.config)
    ctx = harness.context()

    starter, failures = start_in_background(mode, ctx)
    try:
        assert harness.discovery_started.wait(3.0)
        ctx.stop_event.set()  # the supervisor sets it on shutdown
        starter.join(3.0)
        assert not starter.is_alive()
    finally:
        mode.stop(1.0)

    assert failures == []
    assert harness.verifier.stopped
    assert not harness.acts_started
    assert [payload["imu"] for payload in harness.publisher.payloads("medical.state")] == [
        "searching"
    ]
    assert mode.health()["imu"] == "n/a"
    assert mode.health()["errors"] == []


def test_stop_during_the_classifier_load_releases_and_returns_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness()
    harness.classifier_gate = threading.Event()
    harness.install(monkeypatch)
    mode = MedicalMode(harness.config)
    ctx = harness.context()

    starter, failures = start_in_background(mode, ctx)
    try:
        assert harness.classifier_started.wait(3.0)
        ctx.stop_event.set()  # the supervisor sets it on shutdown
        harness.classifier_gate.set()
        starter.join(3.0)
        assert not starter.is_alive()
    finally:
        harness.classifier_gate.set()
        mode.stop(1.0)

    assert failures == []
    assert harness.verifier.stopped
    assert not harness.acts_started
    assert mode.health()["imu"] == "n/a"
    assert mode.health()["errors"] == []


def test_stop_during_the_vision_start_releases_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness()
    harness.install(monkeypatch)
    runtime = BlockingRuntime()
    monkeypatch.setattr(medical_mode, "build_runtime", lambda *args, **kwargs: runtime)
    mode = MedicalMode(harness.config)
    ctx = harness.context()

    starter, failures = start_in_background(mode, ctx)
    try:
        assert runtime.started.wait(3.0)
        ctx.stop_event.set()  # the supervisor sets it on shutdown
        runtime.release.set()
        starter.join(3.0)
        assert not starter.is_alive()
    finally:
        runtime.release.set()
        mode.stop(1.0)

    assert failures == []
    assert runtime.stopped
    assert harness.verifier.stopped
    assert mode.health()["imu"] == "n/a"
    assert mode.health()["errors"] == []


def test_start_with_a_stop_already_requested_touches_no_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness()
    harness.install(monkeypatch)
    mode = MedicalMode(harness.config)
    ctx = harness.context()
    ctx.stop_event.set()

    mode.start(ctx)

    assert harness.captured == {}
    assert harness.publisher.states == []
    assert not harness.verifier.stopped
    assert mode.health()["imu"] == "n/a"


def test_successful_start_publishes_searching_then_connected_and_injects_the_gui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness()
    harness.install(monkeypatch)
    mode = MedicalMode(harness.config)

    mode.start(harness.context())
    try:
        assert harness.publisher.payloads("medical.state")[0]["imu"] == "searching"
        assert isinstance(harness.captured["gui"], WebGui)
        assert harness.captured["observer"] is harness.captured["gui"]
        assert harness.acts_started

        assert wait_for_imu(harness, "connected")
    finally:
        assert mode.stop(2.0) is True

    assert harness.source.stopped
    assert harness.actuator.stopped
    assert mode.health()["imu"] == "n/a"


def test_health_reports_the_vision_state_and_the_model_stem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load(SHIPPED)
    enabled = replace(config, vision=replace(config.vision, enabled=True))
    harness = Harness(config=enabled)
    harness.install(monkeypatch)
    mode = MedicalMode(enabled)

    mode.start(harness.context())
    try:
        health = mode.health()
        assert health["camera"] == "ok"
        assert health["model"] == enabled.vision.model.path.stem
        assert health["fps"] is None and health["dpu_ms"] is None
        assert set(health) == {"imu", "camera", "model", "fps", "dpu_ms", "errors"}
    finally:
        mode.stop(2.0)


def test_health_with_vision_disabled_has_no_camera_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load(SHIPPED)
    disabled = replace(config, vision=replace(config.vision, enabled=False))
    harness = Harness(config=disabled)
    harness.install(monkeypatch)
    mode = MedicalMode(disabled)

    assert mode.health()["camera"] is None
    assert mode.health()["model"] is None


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (FileNotFoundError("/dev/video0"), "camera_missing"),
        (RuntimeError("camera busy"), "camera_error"),
    ],
)
def test_a_camera_failure_at_start_raises_and_releases(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, code: str
) -> None:
    config = load(SHIPPED)
    harness = Harness(config=replace(config, vision=replace(config.vision, enabled=True)))
    harness.verifier.health_error = error
    harness.install(monkeypatch)
    mode = MedicalMode(harness.config)

    with pytest.raises(ModeStartError) as info:
        mode.start(harness.context())

    assert info.value.code == code
    assert harness.source.stopped
    assert harness.actuator.stopped
    assert mode.health()["errors"] == [code]
    assert mode.health()["camera"] == code


@pytest.mark.parametrize(
    ("cause", "code"),
    [
        (FileNotFoundError("model.onnx"), "model_missing"),
        (
            RuntimeError("xir and vart Python bindings are required for DpuPoseEstimator"),
            "dpu_error",
        ),
        (ValueError("invalid YUYV size 641x480"), "camera_error"),
    ],
)
def test_vision_startup_errors_map_by_cause(
    monkeypatch: pytest.MonkeyPatch, cause: Exception, code: str
) -> None:
    harness = Harness()

    def failing_adapters(config: Config, **kwargs: object) -> BoardAdapters:
        raise VisionStartupError(str(cause)) from cause

    monkeypatch.setattr(medical_mode, "board_adapters", failing_adapters)
    mode = MedicalMode(harness.config)

    with pytest.raises(ModeStartError) as info:
        mode.start(harness.context())

    assert info.value.code == code
    assert mode.health()["errors"] == [code]


def test_state_period_is_one_second() -> None:
    assert STATE_PERIOD_S == 1.0
