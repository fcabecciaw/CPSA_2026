"""Validate protocol v1 messages against the JSON Schemas in ``docs/protocol/v1``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "protocol" / "v1"
CLIENT_TYPES = frozenset({"ping", "mode.set"})


@cache
def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)


def schema_types() -> list[str]:
    """Every message type that has a payload schema."""
    return sorted(
        path.name.removesuffix(".schema.json")
        for path in SCHEMA_DIR.glob("*.schema.json")
        if not path.name.endswith("envelope.schema.json")
    )


def validate_server_message(message: Mapping[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` unless ``message`` is a valid server message."""
    _validator("envelope").validate(message)
    _validator(str(message["type"])).validate(message["data"])


def validate_client_message(message: Mapping[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` unless ``message`` is a valid client command."""
    _validator("client-envelope").validate(message)
    _validator(str(message["type"])).validate(message["data"])
