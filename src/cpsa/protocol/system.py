"""Message builders for the supervisor topics and connection lifecycle (IF-102..304).

Pure code: every function returns the ``data`` payload (or, for :func:`envelope`,
the complete message) as plain JSON-compatible values. The server adapter stamps
the envelope per connection, so nothing here reads the clock or a socket.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cpsa.protocol import PROTOCOL_VERSION


def envelope(
    type: str,
    data: Mapping[str, Any],
    seq: int,
    t: float,
    wall: str,
) -> dict[str, Any]:
    """Wrap ``data`` in the protocol v1 envelope (IF-102)."""
    return {
        "v": PROTOCOL_VERSION,
        "type": type,
        "seq": seq,
        "t": t,
        "wall": wall,
        "data": dict(data),
    }


def hello(
    server_version: str,
    capabilities: Sequence[str],
    modes: Sequence[str],
    models: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """The first message of every connection (IF-201)."""
    return {
        "server_version": server_version,
        "protocol_version": PROTOCOL_VERSION,
        "capabilities": list(capabilities),
        "modes": list(modes),
        "models": [dict(model) for model in models],
    }


def status(
    mode: str,
    mode_state: str,
    perception: Mapping[str, Any],
    imu: str,
    errors: Sequence[str],
    uptime_s: float,
) -> dict[str, Any]:
    """Supervisor ``status`` data (IF-301)."""
    return {
        "mode": mode,
        "mode_state": mode_state,
        "perception": {
            "fps": perception.get("fps"),
            "dpu_ms": perception.get("dpu_ms"),
            "camera": perception.get("camera"),
            "model": perception.get("model"),
        },
        "imu": imu,
        "errors": list(errors),
        "uptime_s": uptime_s,
    }


def mode_progress(mode: str, step: str, error: Mapping[str, str] | None = None) -> dict[str, Any]:
    """One ``mode.progress`` event of a mode switch (IF-302)."""
    return {
        "mode": mode,
        "step": step,
        "error": dict(error) if error is not None else None,
    }


def error(code: str, message: str, ref_seq: int | None = None) -> dict[str, Any]:
    """An ``error`` event; ``ref_seq`` identifies the command it answers (IF-304)."""
    data: dict[str, Any] = {"code": code, "message": message}
    if ref_seq is not None:
        data["ref_seq"] = ref_seq
    return data


def pong(message_id: Any) -> dict[str, Any]:
    """The answer to ``ping``; ``id`` is echoed unchanged (IF-203)."""
    return {"id": message_id}
