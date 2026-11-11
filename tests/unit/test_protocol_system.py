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
    assert data["perception"] == {
        "fps": 9.5,
        "dpu_ms": 4.2,
        "camera": "ok",
        "model": "pose",
        "timing": None,
    }


def test_status_fills_not_applicable_perception_values() -> None:
    data = system.status("idle", "stopped", {}, "n/a", [], 0.0)

    validate_server_message(_message("status", data))
    assert data["perception"] == {
        "fps": None,
        "dpu_ms": None,
        "camera": None,
        "model": None,
        "timing": None,
    }


def test_status_carries_the_performance_snapshot_rounded_to_one_decimal() -> None:
    snapshot = {
        "analyzed_fps": 9.456,
        "camera_fps": 29.96,
        "decode_ms": 1.84,
        "preprocess_ms": 0.94,
        "dpu_ms": 4.24,
        "postprocess_ms": 1.16,
        "pose_ms": 6.51,
        "track_ms": 0.72,
        "latency_ms": 45.33,
        "dropped": 3,
    }

    data = system.status(
        mode="medical",
        mode_state="running",
        perception={"fps": 9.456, "dpu_ms": 4.24, "camera": "ok", "timing": snapshot},
        imu="connected",
        errors=[],
        uptime_s=12.5,
    )

    validate_server_message(_message("status", data))
    assert data["perception"]["fps"] == 9.5
    assert data["perception"]["dpu_ms"] == 4.2
    assert data["perception"]["timing"] == {
        "analyzed_fps": 9.5,
        "camera_fps": 30.0,
        "decode_ms": 1.8,
        "preprocess_ms": 0.9,
        "dpu_ms": 4.2,
        "postprocess_ms": 1.2,
        "pose_ms": 6.5,
        "track_ms": 0.7,
        "latency_ms": 45.3,
        "dropped": 3,
    }


def test_status_timing_keeps_every_key_with_nulls_and_no_drops() -> None:
    data = system.status(
        mode="medical",
        mode_state="running",
        perception={"timing": {"analyzed_fps": None, "dpu_ms": None}},
        imu="connected",
        errors=[],
        uptime_s=1.0,
    )

    validate_server_message(_message("status", data))
    timing = data["perception"]["timing"]
    assert timing == {
        "analyzed_fps": None,
        "camera_fps": None,
        "decode_ms": None,
        "preprocess_ms": None,
        "dpu_ms": None,
        "postprocess_ms": None,
        "pose_ms": None,
        "track_ms": None,
        "latency_ms": None,
        "dropped": 0,
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
