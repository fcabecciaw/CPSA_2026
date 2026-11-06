import json
import logging
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from cpsa.adapters.event_log import JsonlEventSink, configure_logging
from cpsa.domain.model import Actuation, Decision, DecisionRecord, GateReason, Tag, Verdict

STAMP = datetime(2026, 11, 6, 10, 30, tzinfo=timezone(timedelta(hours=1)))


def read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_records_are_appended_as_json_lines_in_the_day_folder(tmp_path: Path) -> None:
    sink = JsonlEventSink(tmp_path)
    decision = Decision(
        Verdict.UNDECIDED,
        GateReason.TIMEOUT,
        True,
        (Actuation("led_10.0.0.2", {"color": (255, 0, 0, 0), "intensity": 100}),),
    )
    sink.write(DecisionRecord(STAMP, 12.5, Tag.DANGEROUS, True, decision))
    sink.write(DecisionRecord(STAMP, 13.25, Tag.NON_STEREOTIPY, True, None))

    assert read_events(tmp_path / "2026-11-06" / "events.jsonl") == [
        {
            "timestamp": "2026-11-06T10:30:00+01:00",
            "window_end_ts": 12.5,
            "tag": "DANGEROUS",
            "tag_changed": True,
            "decision": {
                "verdict": "UNDECIDED",
                "reason": "TIMEOUT",
                "actuated": True,
                "actuations": [
                    {
                        "actuator_id": "led_10.0.0.2",
                        "params": {"color": [255, 0, 0, 0], "intensity": 100},
                    }
                ],
            },
        },
        {
            "timestamp": "2026-11-06T10:30:00+01:00",
            "window_end_ts": 13.25,
            "tag": "NON_STEREOTIPY",
            "tag_changed": True,
            "decision": None,
        },
    ]


def test_records_go_to_the_folder_of_their_own_day(tmp_path: Path) -> None:
    sink = JsonlEventSink(tmp_path)
    sink.write(DecisionRecord(STAMP + timedelta(days=1), 1.0, Tag.DANGEROUS, True, None))
    assert (tmp_path / "2026-11-07" / "events.jsonl").exists()
    assert not (tmp_path / "2026-11-06").exists()


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logger")
def test_system_log_is_written_under_the_day_folder(tmp_path: Path) -> None:
    configure_logging(tmp_path, "INFO")
    logging.getLogger("cpsa.test").info("pipeline started")
    logging.getLogger("cpsa.test").debug("not at this level")

    content = (tmp_path / date.today().isoformat() / "system.log").read_text()
    assert "INFO cpsa.test: pipeline started" in content
    assert "not at this level" not in content
