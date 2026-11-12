"""Protocol v1 schemas are valid, every example conforms and the builders match (F-K1)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from cpsa import protocol
from tests.contract.protocol_schemas import (
    CLIENT_TYPES,
    SCHEMA_DIR,
    schema_types,
    validate_client_message,
    validate_server_message,
)

EXAMPLES = sorted((SCHEMA_DIR / "examples").glob("*.json"))
WALL = "2026-11-12T15:00:00.000Z"


@pytest.mark.parametrize("path", sorted(SCHEMA_DIR.glob("*.schema.json")), ids=lambda p: p.name)
def test_schema_is_valid(path) -> None:
    Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_every_type_has_an_example() -> None:
    assert sorted(path.stem for path in EXAMPLES) == schema_types()


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_validates(path) -> None:
    message = json.loads(path.read_text(encoding="utf-8"))
    if message["type"] in CLIENT_TYPES:
        validate_client_message(message)
    else:
        validate_server_message(message)


def _server(message_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return protocol.envelope(message_type, data, 1, 1001.0, WALL)


def test_builder_output_validates() -> None:
    object_ = protocol.ShowcaseObject(
        label="persona", class_id=0, score=0.913, box=(120.5, 48.0, 380.2, 470.0)
    )
    messages = [
        _server("hello", protocol.hello("0.1.0")),
        _server(
            "status",
            protocol.status(
                state="running",
                camera="ok",
                model="yolox_nano",
                fps=59.4,
                dpu_ms=4.7,
                errors=[],
                uptime_s=12.5,
            ),
        ),
        _server(
            "showcase.state",
            protocol.showcase_state(
                "yolox_nano", 640, 480, [object_], fps=59.4, dpu_ms=4.7, total_ms=11.2
            ),
        ),
        _server("error", protocol.error("bad_command", "unknown command")),
        _server("pong", protocol.pong(7)),
    ]

    for message in messages:
        validate_server_message(message)


def test_envelope_rejects_zero_seq() -> None:
    message = json.loads((SCHEMA_DIR / "examples" / "status.json").read_text(encoding="utf-8"))
    message["seq"] = 0
    with pytest.raises(ValidationError):
        validate_server_message(message)


def test_unknown_error_code_is_rejected() -> None:
    message = json.loads((SCHEMA_DIR / "examples" / "error.json").read_text(encoding="utf-8"))
    message["data"]["code"] = "oops"
    with pytest.raises(ValidationError):
        validate_server_message(message)


def test_unknown_status_state_is_rejected() -> None:
    data = protocol.status(
        state="running", camera="ok", model="m", fps=None, dpu_ms=None, errors=[], uptime_s=0.0
    )
    data["state"] = "idle"
    with pytest.raises(ValidationError):
        validate_server_message(_server("status", data))
