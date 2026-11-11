"""Forward device and connection log records to the dashboard (``log`` events).

The websocket protocol has a v1-compatible ``log`` event
(``docs/protocol/v1/log.schema.json``) so the operator can see BlueCoin,
MetaMotion, camera, mode and supervisor problems at the booth without an SSH
session. This module adapts :mod:`logging` to the :class:`cpsa.ports.Publisher`
port: one handler is attached to the loggers of the hardware adapters and the
supervisor, and every INFO-and-above record is published as one ``log`` event.

The handler is deliberately forgiving. Producers log from the BLE, camera and
supervisor threads, so ``emit`` never raises and never blocks: the publisher is
non-blocking, and records over the rate limit or repeated within the dedupe
window are dropped instead of queued.

Source mapping (from the logger name and the message):

- ``cpsa.adapters.bluecoin`` -> ``bluecoin`` or ``bluecoin:<device>``, with the
  device parsed from ``[left:CPSA_L1]`` or ``BlueCoin CPSA_L1:`` prefixes;
- ``cpsa.adapters.actuators`` -> ``metamotion``, ``speaker``, ``led`` or
  ``actuator``;
- ``cpsa.adapters.camera`` and ``linuxpy.device`` -> ``camera``;
- ``cpsa.supervisor`` -> ``supervisor``;
- ``cpsa.board.medical``/``cpsa.board.fake`` -> ``medical``, other ``cpsa.board``
  records -> ``board``;
- ``cpsa.vision`` -> ``vision``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

MAX_MESSAGE_CHARS = 300
"""``log.message`` may hold at most this many characters (log.schema.json)."""

RATE_LIMIT = 20
"""At most this many events per second, overall."""

DEDUPE_WINDOW_S = 2.0
"""Identical messages published within this window are collapsed into one."""

FORWARDED_LOGGERS = (
    "cpsa.adapters.bluecoin",
    "cpsa.adapters.actuators",
    "cpsa.adapters.camera",
    "linuxpy.device",
    "cpsa.supervisor",
    "cpsa.board",
    "cpsa.vision",
)
"""Roots whose INFO-and-above records become dashboard log lines."""

_BRACKET_DEVICE = re.compile(r"^\[([^\]\s]+)\]")
_NAMED_DEVICE = re.compile(r"^BlueCoin\s+([^:\s]+)\s*:")


class EventPublisher(Protocol):
    """The part of :class:`cpsa.ports.Publisher` this handler needs."""

    def event(self, type: str, data: Mapping[str, Any]) -> None: ...


def level_for(levelno: int) -> str:
    """Map a logging level to the protocol ``level``; below INFO is ``info``."""
    if levelno >= logging.ERROR:
        return "error"
    if levelno >= logging.WARNING:
        return "warning"
    return "info"


def source_for(logger_name: str, message: str) -> str:
    """Short device or component id for one record."""
    if logger_name.startswith("cpsa.adapters.bluecoin"):
        device = _bluecoin_device(message)
        return f"bluecoin:{device}" if device is not None else "bluecoin"
    if logger_name.startswith("cpsa.adapters.actuators"):
        return _actuator_source(logger_name, message)
    if logger_name.startswith(("cpsa.adapters.camera", "linuxpy.device")):
        return "camera"
    if logger_name.startswith("cpsa.supervisor"):
        return "supervisor"
    if logger_name.startswith(("cpsa.board.medical", "cpsa.board.fake")):
        return "medical"
    if logger_name.startswith("cpsa.board"):
        return "board"
    if logger_name.startswith("cpsa.vision"):
        return "vision"
    return logger_name.rsplit(".", 1)[-1] or "log"


def _bluecoin_device(message: str) -> str | None:
    """The device id from ``[left:CPSA_L1] ...`` or ``BlueCoin CPSA_L1: ...``.

    The thread tags left/right in front of the BLE name; the panel source keeps
    the device name, e.g. ``bluecoin:CPSA_L1`` for both message forms.
    """
    match = _BRACKET_DEVICE.match(message)
    if match is not None:
        device = match.group(1)
        side, separator, name = device.partition(":")
        if separator and side in ("left", "right") and name:
            return name
        return device
    match = _NAMED_DEVICE.match(message)
    if match is not None:
        return match.group(1)
    return None


def _actuator_source(logger_name: str, message: str) -> str:
    text = f"{logger_name} {message}"
    if "metamotion" in text.lower() or "MetaMotion" in text:
        return "metamotion"
    if "speaker" in logger_name or "[Speaker" in message:
        return "speaker"
    if "led" in logger_name or "[LedStrip" in message:
        return "led"
    return "actuator"


class LogForwardHandler(logging.Handler):
    """Publish INFO-and-above records as ``log`` events, never raising or blocking.

    The rate limiter is a token bucket refilled at :data:`RATE_LIMIT` per second,
    so a burst of at most :data:`RATE_LIMIT` records is accepted at once and the
    long-run rate never exceeds the limit. The bucket, the duplicate window and
    the clock are guarded by one lock because records arrive from several
    threads.
    """

    def __init__(
        self,
        publisher: EventPublisher,
        *,
        clock: Callable[[], float] | None = None,
        level: int = logging.INFO,
    ) -> None:
        super().__init__(level)
        self._publisher = publisher
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.Lock()
        self._tokens = float(RATE_LIMIT)
        self._refilled_at = self._clock()
        self._recent: dict[tuple[str, str, str], float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno < self.level:
                return
            message = self._message(record)
            if message == "":
                return
            source = source_for(record.name, message)
            level = level_for(record.levelno)
            if not self._accept(level, source, message):
                return
            self._publisher.event("log", {"level": level, "source": source, "message": message})
        except Exception:  # noqa: BLE001 - a logging handler must never raise
            # Drop the record: the dashboard panel is diagnostic, and raising
            # into a BLE or camera thread could stop the demo.
            pass

    def _message(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - bad %-args must not escape logging
            message = record.msg if isinstance(record.msg, str) else str(record.msg)
        return message[:MAX_MESSAGE_CHARS]

    def _accept(self, level: str, source: str, message: str) -> bool:
        """Reserve one event slot, applying the dedupe window and the token bucket."""
        key = (level, source, message)
        with self._lock:
            now = self._clock()
            last_at = self._recent.get(key)
            if last_at is not None and now - last_at < DEDUPE_WINDOW_S:
                return False
            elapsed = max(0.0, now - self._refilled_at)
            self._tokens = min(float(RATE_LIMIT), self._tokens + elapsed * RATE_LIMIT)
            self._refilled_at = now
            if self._tokens < 1.0:
                return False
            self._tokens -= 1.0
            self._recent[key] = now
            if len(self._recent) > 64:
                self._recent = {
                    item: seen
                    for item, seen in self._recent.items()
                    if now - seen < DEDUPE_WINDOW_S
                }
            return True


def install(
    publisher: EventPublisher,
    *,
    loggers: Iterable[str] = FORWARDED_LOGGERS,
    clock: Callable[[], float] | None = None,
) -> LogForwardHandler:
    """Attach one handler to ``loggers`` and return it.

    Calling this twice with the same loggers is harmless: an already installed
    :class:`LogForwardHandler` is reused instead of duplicating every line.
    """
    handler = LogForwardHandler(publisher, clock=clock)
    targets = [logging.getLogger(name) for name in loggers]
    for target in targets:
        existing = next(
            (item for item in target.handlers if isinstance(item, LogForwardHandler)),
            None,
        )
        if existing is not None:
            return existing
    for target in targets:
        target.addHandler(handler)
    return handler
