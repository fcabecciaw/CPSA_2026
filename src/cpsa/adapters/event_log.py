"""System log and JSON Lines event log, grouped in one folder per day.

Layout: ``<base_path>/<YYYY-MM-DD>/system.log`` and ``.../events.jsonl``.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, TextIO

from cpsa.domain.model import Decision, DecisionRecord

SYSTEM_LOG = "system.log"
EVENT_LOG = "events.jsonl"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def day_folder(base_path: Path, day: date) -> Path:
    return base_path / day.isoformat()


class DailyFileHandler(logging.Handler):
    """Write log records to ``<base>/<day>/system.log``, switching file at midnight."""

    def __init__(self, base_path: Path) -> None:
        super().__init__()
        self._base_path = base_path
        self._day: date | None = None
        self._stream: TextIO | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            day = datetime.fromtimestamp(record.created).date()
            stream = self._stream if day == self._day and self._stream else self._open(day)
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def _open(self, day: date) -> TextIO:
        if self._stream is not None:
            self._stream.close()
        folder = day_folder(self._base_path, day)
        folder.mkdir(parents=True, exist_ok=True)
        self._stream = (folder / SYSTEM_LOG).open("a", encoding="utf-8")
        self._day = day
        return self._stream

    def close(self) -> None:
        self.acquire()
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            self.release()
        super().close()


def configure_logging(base_path: Path, level: str, console: TextIO = sys.stderr) -> None:
    """Send all ``logging`` output at ``level`` and above to the console and system log."""
    formatter = logging.Formatter(LOG_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in (logging.StreamHandler(console), DailyFileHandler(base_path)):
        handler.setFormatter(formatter)
        root.addHandler(handler)


class JsonlEventSink:
    """Append one JSON object per :class:`DecisionRecord` to the day's ``events.jsonl``.

    The day folder is taken from the record's own timestamp. Thread-safe.
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._lock = threading.Lock()

    def write(self, record: DecisionRecord) -> None:
        line = json.dumps(to_json(record))
        folder = day_folder(self._base_path, record.timestamp.date())
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / EVENT_LOG).open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")


def to_json(record: DecisionRecord) -> dict[str, Any]:
    return {
        "timestamp": record.timestamp.isoformat(),
        "window_end_ts": record.window_end_ts,
        "tag": record.tag.name,
        "tag_changed": record.tag_changed,
        "decision": _decision_json(record.decision),
    }


def _decision_json(decision: Decision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "verdict": decision.verdict.value,
        "reason": decision.reason.value,
        "actuated": decision.actuated,
        "actuations": [
            {"actuator_id": a.actuator_id, "params": dict(a.params)} for a in decision.actuations
        ],
    }
