"""Protocol builders produce schema-valid v1 messages (IF-102..304, F5)."""

from __future__ import annotations

from typing import Any

import pytest

from cpsa.protocol import PROTOCOL_VERSION, system
from tests.contract.protocol_schemas import validate_server_message

WALL = "2026-11-12T08:00:00Z"


def _message(topic: str, data: dict[str, Any]) -> dict[str, Any]:
    return system.envelope(topic, data, seq=1, t=1001.5, wall=WALL)


def test_envelope_has_the_v1_fields() -> None:
    message = _message("status", {"mode": "idle"})

    assert message == {
        "v": PROTOCOL_VERSION,
        "type": "status",
        "seq": 1,
        "t": 1001.5,
        "wall": WALL,
        "data": {"mode": "idle"},
    }


def test_hello_is_schema_valid() -> None:
    data = system.hello(
        "0.1.0",
        ["status", "medical.state"],
        ["idle", "medical"],
        [{"id": "lrelu", "name": "LReLU"}],
    )

    validate_server_message(_message("hello", data))
    assert data["protocol_version"] == 1
    assert data["models"] == [{"id": "lrelu", "name": "LReLU"}]


def test_status_is_schema_valid() -> None:
    data = system.status(
        mode="medical",
        mode_state="running",
        perception={"fps": 9.5, "dpu_ms": 4.2, "camera": "ok", "model": "pose"},
        imu="connected",
        errors=[],
        uptime_s=12.5,
    )

    validate_server_message(_message("status", data))
    assert data["perception"] == {"fps": 9.5, "dpu_ms": 4.2, "camera": "ok", "model": "pose"}


def test_status_fills_not_applicable_perception_values() -> None:
    data = system.status("idle", "stopped", {}, "n/a", [], 0.0)

    validate_server_message(_message("status", data))
    assert data["perception"] == {
        "fps": None,
        "dpu_ms": None,
        "camera": None,
        "model": None,
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (None, None),
        (
            {"code": "imu_not_found", "message": "no BlueCoins"},
            {"code": "imu_not_found", "message": "no BlueCoins"},
        ),
    ],
)
def test_mode_progress_is_schema_valid(error: dict[str, str] | None, expected: Any) -> None:
    data = system.mode_progress("medical", "failed" if error else "ready", error)

    validate_server_message(_message("mode.progress", data))
    assert data["error"] == expected


def test_error_answers_a_command_with_ref_seq() -> None:
    data = system.error("busy", "a switch is in progress", ref_seq=3)

    validate_server_message(_message("error", data))
    assert data == {
        "code": "busy",
        "message": "a switch is in progress",
        "ref_seq": 3,
    }


def test_error_without_ref_seq_is_schema_valid() -> None:
    data = system.error("internal", "boom")

    validate_server_message(_message("error", data))
    assert "ref_seq" not in data


@pytest.mark.parametrize("message_id", [7, "abc", None])
def test_pong_echoes_the_id(message_id: Any) -> None:
    data = system.pong(message_id)

    validate_server_message(_message("pong", data))
    assert data == {"id": message_id}
