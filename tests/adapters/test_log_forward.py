"""The logging handler that forwards device lines to the dashboard (``log`` events)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from cpsa.adapters.web.log_forward import (
    DEDUPE_WINDOW_S,
    MAX_MESSAGE_CHARS,
    RATE_LIMIT,
    LogForwardHandler,
    install,
    level_for,
    source_for,
)


class FakePublisher:
    """Collect the ``(type, data)`` publications of a handler."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.events: list[tuple[str, Mapping[str, Any]]] = []
        self.error = error

    def event(self, type: str, data: Mapping[str, Any]) -> None:
        if self.error is not None:
            raise self.error
        self.events.append((type, data))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def record(logger_name: str, message: str, *args: object, level: int = logging.INFO):
    return logging.LogRecord(
        logger_name,
        level,
        __file__,
        1,
        message,
        args or None,
        None,
    )


def handler_at(publisher: FakePublisher, clock: FakeClock) -> LogForwardHandler:
    return LogForwardHandler(publisher, clock=clock)


@pytest.mark.parametrize(
    ("logger_name", "message", "expected"),
    [
        (
            "cpsa.adapters.bluecoin.source",
            "[left:CPSA_L1] Connected successfully",
            "bluecoin:CPSA_L1",
        ),
        ("cpsa.adapters.bluecoin.source", "BlueCoin CPSA_R2: right connected", "bluecoin:CPSA_R2"),
        (
            "cpsa.adapters.bluecoin.source",
            "BlueCoin status changed from CONNECTED to IDLE",
            "bluecoin",
        ),
        ("cpsa.adapters.bluecoin.source", "No BlueCoin devices found", "bluecoin"),
        (
            "cpsa.adapters.actuators.metamotion",
            "[MetaMotion AA:BB] Connected successfully",
            "metamotion",
        ),
        ("cpsa.adapters.actuators.metamotion", "No MetaWear devices found", "metamotion"),
        ("cpsa.adapters.actuators.speaker", "[Speaker AA:BB] Disconnected", "speaker"),
        ("cpsa.adapters.actuators.led_strip", "[LedStrip 10.0.0.2] Connected successfully", "led"),
        ("cpsa.adapters.actuators.factory", "Actuator started: 1", "actuator"),
        ("cpsa.adapters.camera.uvc", "camera frame callback failed", "camera"),
        ("linuxpy.device", "control failed", "camera"),
        ("cpsa.supervisor", "Mode medical failed to start", "supervisor"),
        ("cpsa.board.medical", "Medical mode started", "medical"),
        ("cpsa.board", "System running", "board"),
        ("cpsa.vision.core", "episode published", "vision"),
        ("cpsa.something.else", "hello", "else"),
    ],
)
def test_source_for(logger_name: str, message: str, expected: str) -> None:
    assert source_for(logger_name, message) == expected


@pytest.mark.parametrize(
    ("levelno", "expected"),
    [
        (logging.DEBUG, "info"),
        (logging.INFO, "info"),
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
        (logging.CRITICAL, "error"),
    ],
)
def test_level_for(levelno: int, expected: str) -> None:
    assert level_for(levelno) == expected


def test_emit_publishes_the_protocol_payload() -> None:
    publisher = FakePublisher()
    handler = handler_at(publisher, FakeClock())

    handler.emit(record("cpsa.adapters.bluecoin.source", "[left:CPSA_L1] %s", "connected"))

    assert publisher.events == [
        (
            "log",
            {
                "level": "info",
                "source": "bluecoin:CPSA_L1",
                "message": "[left:CPSA_L1] connected",
            },
        )
    ]


def test_debug_records_are_not_forwarded() -> None:
    publisher = FakePublisher()
    handler = handler_at(publisher, FakeClock())

    handler.emit(record("cpsa.adapters.camera.uvc", "noise", level=logging.DEBUG))

    assert publisher.events == []


def test_messages_are_truncated_to_the_schema_limit() -> None:
    publisher = FakePublisher()
    handler = handler_at(publisher, FakeClock())

    handler.emit(record("cpsa.adapters.camera.uvc", "x" * (MAX_MESSAGE_CHARS + 40)))

    ((_, data),) = publisher.events
    assert len(data["message"]) == MAX_MESSAGE_CHARS
    assert data["message"] == "x" * MAX_MESSAGE_CHARS


def test_identical_messages_are_collapsed_within_the_window() -> None:
    publisher = FakePublisher()
    clock = FakeClock()
    handler = handler_at(publisher, clock)
    line = record("cpsa.adapters.bluecoin.source", "[left:CPSA_L1] Connection failed, retrying...")

    handler.emit(line)
    clock.advance(DEDUPE_WINDOW_S - 0.01)
    handler.emit(line)
    assert len(publisher.events) == 1

    clock.advance(0.02)
    handler.emit(line)
    assert len(publisher.events) == 2


def test_dedupe_distinguishes_level_source_and_message() -> None:
    publisher = FakePublisher()
    handler = handler_at(publisher, FakeClock())

    handler.emit(record("cpsa.adapters.camera.uvc", "camera stopped", level=logging.WARNING))
    handler.emit(record("cpsa.adapters.camera.uvc", "camera stopped", level=logging.ERROR))

    assert [data["level"] for _, data in publisher.events] == ["warning", "error"]


def test_dedupe_survives_an_interleaved_other_line() -> None:
    publisher = FakePublisher()
    handler = handler_at(publisher, FakeClock())
    retry = record("cpsa.adapters.bluecoin.source", "[left:CPSA_L1] Connection failed, retrying...")

    handler.emit(retry)
    handler.emit(record("cpsa.adapters.bluecoin.source", "[left:CPSA_L1] Thread started"))
    handler.emit(retry)

    assert [data["message"] for _, data in publisher.events] == [
        "[left:CPSA_L1] Connection failed, retrying...",
        "[left:CPSA_L1] Thread started",
    ]


def test_rate_limit_allows_a_burst_of_twenty_then_drops() -> None:
    publisher = FakePublisher()
    clock = FakeClock()
    handler = handler_at(publisher, clock)

    for index in range(RATE_LIMIT + 5):
        handler.emit(record("cpsa.adapters.camera.uvc", f"frame {index}"))

    assert len(publisher.events) == RATE_LIMIT

    clock.advance(1.0)
    handler.emit(record("cpsa.adapters.camera.uvc", "after one second"))
    assert len(publisher.events) == RATE_LIMIT + 1


def test_publisher_errors_never_reach_the_producer() -> None:
    publisher = FakePublisher(error=RuntimeError("publisher is gone"))
    handler = handler_at(publisher, FakeClock())

    handler.emit(record("cpsa.adapters.camera.uvc", "camera failed"))

    assert publisher.events == []


def test_a_bad_format_argument_does_not_raise() -> None:
    publisher = FakePublisher()
    handler = handler_at(publisher, FakeClock())
    line = logging.LogRecord(
        "cpsa.adapters.camera.uvc", logging.INFO, __file__, 1, "value %d", ("x",), None
    )

    handler.emit(line)

    assert publisher.events[0][1]["message"] == "value %d"


def test_install_attaches_one_handler_per_logger() -> None:
    publisher = FakePublisher()
    logger = logging.getLogger("cpsa-test-log-forward")

    try:
        first = install(publisher, loggers=("cpsa-test-log-forward",))
        second = install(publisher, loggers=("cpsa-test-log-forward",))

        assert first is second
        assert sum(isinstance(item, LogForwardHandler) for item in logger.handlers) == 1
    finally:
        for item in list(logger.handlers):
            if isinstance(item, LogForwardHandler):
                logger.removeHandler(item)
