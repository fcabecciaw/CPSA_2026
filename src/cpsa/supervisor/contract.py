"""Frozen contract between the supervisor and its modes (interface IF-701..704).

The supervisor runs ``Mode.start`` and ``Mode.stop`` on a worker thread, never on the
server's event loop. A mode reaches perception and devices only through its own
adapters (this slice keeps the camera inside the medical mode's vision verifier).
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, runtime_checkable

from cpsa.application.config import Config
from cpsa.ports import Publisher

ModeName = Literal["idle", "lab", "hologram", "medical", "rps"]
ModeState = Literal["stopped", "starting", "running", "stopping", "failed"]

#: Closed list of error codes (IF-304). Extended only by a protocol addition.
ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "camera_missing",
        "camera_error",
        "model_missing",
        "dpu_error",
        "imu_not_found",
        "asset_invalid",
        "busy",
        "bad_command",
        "internal",
    }
)

#: Keys every ``Mode.health()`` mapping carries. ``None`` means "not applicable or
#: not measured" (for example ``fps`` while perception is not shared, plan D4).
HEALTH_KEYS: Final[tuple[str, ...]] = ("imu", "camera", "model", "fps", "dpu_ms", "errors")


class ModeStartError(Exception):
    """A mode could not start; ``code`` is one of :data:`ERROR_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ModeContext:
    """What a mode receives from the supervisor when it starts."""

    publisher: Publisher
    config: Config
    stop_event: threading.Event


@runtime_checkable
class Mode(Protocol):
    """One demo mode (``medical`` in this slice; ``lab`` later)."""

    name: str

    def start(self, ctx: ModeContext) -> None:
        """Block until the mode runs; raise :class:`ModeStartError` on failure.

        A failed start must release everything it acquired before raising.
        """
        ...

    def stop(self, timeout_s: float) -> bool:
        """Stop and release all devices; ``True`` when everything was released in time."""
        ...

    def health(self) -> Mapping[str, Any]:
        """Current health with the keys in :data:`HEALTH_KEYS`.

        ``imu``: ``"n/a" | "searching" | "connected" | "lost"``;
        ``camera``: ``"ok" | "missing" | "error"`` or ``None``;
        ``model``: model id or ``None``; ``fps``/``dpu_ms``: float or ``None``;
        ``errors``: list of :data:`ERROR_CODES`.
        """
        ...
