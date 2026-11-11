"""The supervisor switch state machine, status rate and failure behavior (AC-07)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cpsa.adapters.clock import SystemClock
from cpsa.application.config import Config, SupervisorConfig, load
from cpsa.supervisor.contract import ModeContext, ModeStartError
from cpsa.supervisor.supervisor import Supervisor

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "config" / "board.yaml"
TIMEOUT_S = 3.0


class RecordingPublisher:
    """Thread-safe Publisher that records states and events for assertions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self.states: dict[str, dict[str, Any]] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status_history: list[dict[str, Any]] = []

    def state(self, type: str, data: dict[str, Any]) -> None:
        payload = dict(data)
        with self._changed:
            self.states[type] = payload
            if type == "status":
                self.status_history.append(payload)
            self._changed.notify_all()

    def event(self, type: str, data: dict[str, Any]) -> None:
        with self._changed:
            self.events.append((type, dict(data)))
            self._changed.notify_all()

    def wait_until(
        self, predicate: Callable[[RecordingPublisher], bool], timeout: float = TIMEOUT_S
    ) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: predicate(self), timeout)

    def progress_steps(self) -> list[str]:
        with self._lock:
            return [data["step"] for name, data in self.events if name == "mode.progress"]

    def error_codes(self) -> list[str]:
        with self._lock:
            return [data["code"] for name, data in self.events if name == "error"]

    def error_payloads(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(data) for name, data in self.events if name == "error"]

    def clear_events(self) -> None:
        with self._lock:
            self.events.clear()


class GatedPublisher(RecordingPublisher):
    """RecordingPublisher that can hold the switch thread at one progress step."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_step: str | None = None
        self.held = threading.Event()
        self.release = threading.Event()

    def event(self, type: str, data: dict[str, Any]) -> None:
        if type == "mode.progress" and data.get("step") == self.gate_step:
            self.held.set()
            if not self.release.wait(TIMEOUT_S):
                raise AssertionError("the test never released the publisher gate")
        super().event(type, data)


def wait_until(predicate: Callable[[], bool], timeout: float = TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not met in time")
        time.sleep(0.005)


class FakeMedicalMode:
    """A controllable ``Mode`` implementation; no hardware and no threads of its own."""

    name = "medical"

    def __init__(
        self,
        *,
        error: ModeStartError | None = None,
        gate: threading.Event | None = None,
        release: bool = True,
    ) -> None:
        self.error = error
        self.gate = gate
        self.release = release
        self.starts = 0
        self.starts_done = threading.Event()
        self.stopped = threading.Event()
        self.stop_timeout: float | None = None
        self.context: ModeContext | None = None

    def start(self, ctx: ModeContext) -> None:
        self.context = ctx
        self.starts += 1
        if self.gate is not None and not self.gate.wait(TIMEOUT_S):
            raise AssertionError("test gate was never released")
        self.starts_done.set()
        if self.error is not None:
            raise self.error

    def stop(self, timeout_s: float) -> bool:
        self.stop_timeout = timeout_s
        self.stopped.set()
        return self.release

    def health(self) -> dict[str, Any]:
        return {
            "imu": "connected",
            "camera": "ok",
            "model": "fake",
            "fps": 10.0,
            "dpu_ms": 4.0,
            "errors": [],
        }


class BlockingStartMode:
    """A mode whose ``start`` waits on ``ctx.stop_event`` (the F4 cancellation seam)."""

    name = "medical"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.unblocked = threading.Event()
        self.stop_calls: list[float] = []

    def start(self, ctx: ModeContext) -> None:
        self.entered.set()
        ctx.stop_event.wait(TIMEOUT_S)
        self.unblocked.set()

    def stop(self, timeout_s: float) -> bool:
        self.stop_calls.append(timeout_s)
        return True

    def health(self) -> dict[str, Any]:
        return {
            "imu": "n/a",
            "camera": None,
            "model": None,
            "fps": None,
            "dpu_ms": None,
            "errors": [],
        }


def make_config(default_mode: str) -> Config:
    return replace(load(SHIPPED), supervisor=SupervisorConfig(default_mode=default_mode))


def make_supervisor(
    publisher: RecordingPublisher,
    mode: FakeMedicalMode | BlockingStartMode,
    *,
    default_mode: str = "medical",
    status_interval_s: float = 0.01,
) -> Supervisor:
    return Supervisor(
        {"medical": mode},
        publisher,
        make_config(default_mode),
        SystemClock(),
        status_interval_s=status_interval_s,
        switch_timeout_s=1.0,
    )


def test_switch_to_medical_publishes_the_four_steps() -> None:
    publisher = RecordingPublisher()
    mode = FakeMedicalMode()
    supervisor = make_supervisor(publisher, mode)
    try:
        supervisor.start_default()

        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        assert publisher.progress_steps() == ["stopping", "releasing", "starting", "ready"]
        status = supervisor.status()
        assert status["mode"] == "medical"
        assert status["mode_state"] == "running"
        assert status["perception"]["camera"] == "ok"
        assert status["imu"] == "connected"
        assert status["errors"] == []
    finally:
        supervisor.shutdown(1.0)


def test_second_mode_set_during_a_switch_is_busy() -> None:
    publisher = RecordingPublisher()
    gate = threading.Event()
    mode = FakeMedicalMode(gate=gate)
    supervisor = make_supervisor(publisher, mode, default_mode="idle")
    try:
        supervisor.start_default()
        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        publisher.clear_events()

        supervisor.request_mode("medical", 11)
        assert publisher.wait_until(lambda p: "starting" in p.progress_steps())
        supervisor.request_mode("idle", 12)

        assert publisher.wait_until(lambda p: "busy" in p.error_codes())
        assert publisher.error_payloads()[0]["ref_seq"] == 12

        gate.set()
        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        assert supervisor.status()["mode_state"] == "running"
    finally:
        gate.set()
        supervisor.shutdown(1.0)


def test_setting_the_active_mode_returns_ready_immediately() -> None:
    publisher = RecordingPublisher()
    mode = FakeMedicalMode()
    supervisor = make_supervisor(publisher, mode)
    try:
        supervisor.start_default()
        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        publisher.clear_events()

        supervisor.request_mode("medical", 21)

        assert publisher.progress_steps() == ["ready"]
        assert mode.starts == 1
        assert supervisor.status()["mode_state"] == "running"
    finally:
        supervisor.shutdown(1.0)


def test_failed_start_leaves_idle_failed_with_the_error_until_the_next_switch() -> None:
    publisher = RecordingPublisher()
    mode = FakeMedicalMode(error=ModeStartError("imu_not_found", "no BlueCoins found"))
    supervisor = make_supervisor(publisher, mode)
    try:
        supervisor.start_default()

        assert publisher.wait_until(lambda p: "failed" in p.progress_steps())
        assert publisher.progress_steps() == ["stopping", "releasing", "starting", "failed"]
        status = supervisor.status()
        assert status["mode"] == "idle"
        assert status["mode_state"] == "failed"
        assert status["errors"] == ["imu_not_found"]
        assert publisher.wait_until(lambda p: p.states["status"]["errors"] == ["imu_not_found"])

        mode.error = None
        publisher.clear_events()
        supervisor.request_mode("medical", 31)

        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        status = supervisor.status()
        assert (status["mode"], status["mode_state"], status["errors"]) == (
            "medical",
            "running",
            [],
        )
    finally:
        supervisor.shutdown(1.0)


def test_failed_release_marks_the_switch_failed() -> None:
    publisher = RecordingPublisher()
    mode = FakeMedicalMode(release=False)
    supervisor = make_supervisor(publisher, mode)
    try:
        supervisor.start_default()
        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        publisher.clear_events()

        supervisor.request_mode("idle", 41)

        assert publisher.wait_until(lambda p: "failed" in p.progress_steps())
        assert publisher.progress_steps() == ["stopping", "failed"]
        status = supervisor.status()
        assert (status["mode"], status["mode_state"], status["errors"]) == (
            "idle",
            "failed",
            ["internal"],
        )
    finally:
        supervisor.shutdown(1.0)


def test_switch_back_to_idle_releases_the_mode() -> None:
    publisher = RecordingPublisher()
    mode = FakeMedicalMode()
    supervisor = make_supervisor(publisher, mode)
    try:
        supervisor.start_default()
        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        publisher.clear_events()

        supervisor.request_mode("idle", 51)

        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        assert publisher.progress_steps() == ["stopping", "releasing", "starting", "ready"]
        assert mode.stopped.is_set()
        assert mode.stop_timeout == 1.0
        status = supervisor.status()
        assert (status["mode"], status["mode_state"]) == ("idle", "stopped")
    finally:
        supervisor.shutdown(1.0)


def test_status_is_published_periodically() -> None:
    publisher = RecordingPublisher()
    supervisor = make_supervisor(publisher, FakeMedicalMode(), default_mode="idle")
    try:
        supervisor.start_default()

        assert publisher.wait_until(lambda p: len(p.status_history) >= 3)
        for status in publisher.status_history:
            assert set(status) == {
                "mode",
                "mode_state",
                "perception",
                "imu",
                "errors",
                "uptime_s",
            }
            assert status["uptime_s"] >= 0.0
    finally:
        supervisor.shutdown(1.0)


def test_status_is_published_on_change_without_waiting_for_the_tick() -> None:
    publisher = RecordingPublisher()
    supervisor = make_supervisor(
        publisher, FakeMedicalMode(), default_mode="idle", status_interval_s=60.0
    )
    try:
        supervisor.start_default()
        assert publisher.wait_until(lambda p: len(p.status_history) > 0)
        publisher.status_history.clear()

        supervisor.request_mode("medical", 5)

        assert publisher.wait_until(lambda p: p.states["status"].get("mode_state") == "running")
        assert len(publisher.status_history) > 0
    finally:
        supervisor.shutdown(1.0)


def test_unknown_mode_is_a_bad_command() -> None:
    publisher = RecordingPublisher()
    supervisor = make_supervisor(publisher, FakeMedicalMode(), default_mode="idle")
    try:
        supervisor.start_default()
        publisher.clear_events()

        supervisor.request_mode("lab", 61)

        assert publisher.error_codes() == ["bad_command"]
        assert publisher.error_payloads()[0]["ref_seq"] == 61
    finally:
        supervisor.shutdown(1.0)


def test_shutdown_stops_the_active_mode() -> None:
    publisher = RecordingPublisher()
    mode = FakeMedicalMode()
    supervisor = make_supervisor(publisher, mode)
    supervisor.start_default()
    assert publisher.wait_until(lambda p: "ready" in p.progress_steps())

    supervisor.shutdown(1.0)

    assert mode.stopped.is_set()
    assert supervisor.status()["mode"] == "idle"
    assert supervisor.status()["mode_state"] == "stopped"


def test_shutdown_cancels_a_start_that_is_still_blocking() -> None:
    """R-03: the stop event is registered before start(), so shutdown can cancel it."""
    publisher = RecordingPublisher()
    mode = BlockingStartMode()
    supervisor = make_supervisor(publisher, mode)
    supervisor.start_default()
    assert mode.entered.wait(TIMEOUT_S)

    supervisor.shutdown(1.0)

    assert mode.unblocked.is_set()
    assert mode.stop_calls == [1.0]
    assert (supervisor.status()["mode"], supervisor.status()["mode_state"]) == ("idle", "stopped")


def test_shutdown_before_the_start_event_is_registered_skips_the_start() -> None:
    """R-20: shutdown in the window after the release and before the registration.

    The previous stop event is already cleared when ``shutdown`` looks for one,
    so the new stop event is set under the same lock that registers it and the
    mode is never started.
    """
    publisher = GatedPublisher()
    mode = FakeMedicalMode()
    supervisor = make_supervisor(publisher, mode, default_mode="idle")
    try:
        supervisor.start_default()
        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        publisher.clear_events()
        publisher.gate_step = "releasing"

        supervisor.request_mode("medical", 11)
        assert publisher.held.wait(TIMEOUT_S)

        shutdown_thread = threading.Thread(target=supervisor.shutdown, args=(1.0,), daemon=True)
        shutdown_thread.start()
        wait_until(lambda: supervisor._closing)
        # shutdown() has now read _active_stop_event, which is None: this is the
        # window; the switch thread is still held before its registration.
        time.sleep(0.05)
        publisher.release.set()
        shutdown_thread.join(TIMEOUT_S)

        assert not shutdown_thread.is_alive()
        assert mode.starts == 0
        assert mode.context is None
        assert not mode.stopped.is_set()
        assert publisher.progress_steps() == ["stopping", "releasing", "starting"]
        assert (supervisor.status()["mode"], supervisor.status()["mode_state"]) == (
            "idle",
            "stopped",
        )
    finally:
        publisher.release.set()
        supervisor.shutdown(1.0)


def test_a_mode_that_does_not_release_is_kept_for_later_stops() -> None:
    """R-06: a failed release must not make the supervisor forget the mode."""
    publisher = RecordingPublisher()
    mode = FakeMedicalMode(release=False)
    supervisor = make_supervisor(publisher, mode)
    try:
        supervisor.start_default()
        assert publisher.wait_until(lambda p: "ready" in p.progress_steps())
        publisher.clear_events()

        supervisor.request_mode("idle", 41)
        assert publisher.wait_until(lambda p: "failed" in p.progress_steps())

        mode.release = True
        mode.stopped.clear()
        supervisor.shutdown(1.0)

        assert mode.stopped.is_set()
    finally:
        supervisor.shutdown(1.0)


def test_request_mode_is_refused_while_the_supervisor_is_closing() -> None:
    """R-03: no mode may start once shutdown began."""
    publisher = RecordingPublisher()
    mode = FakeMedicalMode()
    supervisor = make_supervisor(publisher, mode)
    supervisor.start_default()
    assert publisher.wait_until(lambda p: "ready" in p.progress_steps())

    supervisor.shutdown(1.0)
    publisher.clear_events()

    supervisor.request_mode("medical", 7)

    assert mode.starts == 1
    assert publisher.progress_steps() == []
    assert publisher.error_codes() == ["internal"]


def test_an_error_without_a_client_seq_omits_ref_seq() -> None:
    """R-02: ref_seq is the client command's optional ``seq``, not the server's."""
    publisher = RecordingPublisher()
    supervisor = make_supervisor(publisher, FakeMedicalMode(), default_mode="idle")
    try:
        supervisor.start_default()
        publisher.clear_events()

        supervisor.request_mode("lab")

        assert publisher.error_payloads() == [
            {"code": "bad_command", "message": "unknown mode 'lab'"}
        ]

        supervisor.request_mode("lab", 9)

        assert publisher.error_payloads()[1]["ref_seq"] == 9
    finally:
        supervisor.shutdown(1.0)


def test_start_default_rejects_an_unregistered_default_mode() -> None:
    publisher = RecordingPublisher()
    supervisor = Supervisor(
        {},
        publisher,
        make_config("medical"),
        SystemClock(),
        status_interval_s=0.01,
    )

    with pytest.raises(ValueError, match="not registered"):
        supervisor.start_default()
