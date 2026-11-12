"""The default engine/voice chosen from the dashboard, persisted on disk.

The environment (``CPSA_TTS_DEFAULT_ENGINE``/``CPSA_TTS_DEFAULT_VOICE``) starts
the service; ``PUT /default`` lets the dashboard change it at run time and the
choice survives a restart through ``~/.cache/cpsa-tts/default.json``. The file is
written atomically and never contains the sentence cache keys.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .engines import EngineError, EngineRegistry

log = logging.getLogger("cpsa_tts.defaults")

DEFAULTS_FILE = "default.json"


class DefaultStore:
    """``default.json``: ``{"engine", "voice", "speed"?, "updated_at"}``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser()

    def read(self) -> dict[str, Any] | None:
        """The persisted choice, or None when absent, unreadable or malformed."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            log.warning("ignoring %s: %s", self.path, error)
            return None
        if not isinstance(raw, dict):
            return None
        engine = raw.get("engine")
        voice = raw.get("voice")
        if not isinstance(engine, str) or not isinstance(voice, str):
            return None
        if not engine or not voice:
            return None
        choice: dict[str, Any] = {"engine": engine, "voice": voice}
        speed = raw.get("speed")
        if isinstance(speed, int | float) and not isinstance(speed, bool):
            choice["speed"] = float(speed)
        return choice

    def write(self, engine: str, voice: str, speed: float | None = None) -> Path:
        """Persist the choice atomically: a crash cannot truncate the file."""
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"engine": engine, "voice": voice}
        if speed is not None:
            payload["speed"] = speed
        payload["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        handle, temp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=1)
                file.write("\n")
            os.replace(temp_name, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name)
            raise
        return path


def apply_persisted(registry: EngineRegistry, store: DefaultStore) -> dict[str, Any] | None:
    """Adopt the persisted default when it still names a known engine and voice."""
    choice = store.read()
    if choice is None:
        return None
    try:
        registry.set_defaults(choice["engine"], choice["voice"])
    except EngineError as error:
        log.warning("ignoring %s: %s", store.path, error)
        return None
    if "speed" in choice:
        try:
            registry.set_speed(choice["engine"], choice["speed"])
        except EngineError as error:
            log.warning("ignoring the speed in %s: %s", store.path, error)
            choice.pop("speed")
    return choice
