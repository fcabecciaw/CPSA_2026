"""Protocol v1 schemas are valid and every example message conforms (FR-506, D3)."""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from tests.contract.protocol_schemas import (
    CLIENT_TYPES,
    SCHEMA_DIR,
    schema_types,
    validate_client_message,
    validate_server_message,
)

EXAMPLES = sorted((SCHEMA_DIR / "examples").glob("*.json"))


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
