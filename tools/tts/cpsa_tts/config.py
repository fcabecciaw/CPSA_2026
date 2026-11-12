"""Settings of the local TTS service, read from the environment.

Everything has a working default so the service starts with ``cpsa-tts serve``
and no configuration file. The dashboard talks to ``127.0.0.1:8765``; the two
allowed browser origins are the laptop's own dashboard and the Vite dev server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "cpsa-tts"
DEFAULT_ORIGINS = ("http://10.42.0.83:8080", "http://localhost:5173")
DEFAULT_ENGINE = "kokoro"
DEFAULT_VOICE = "if_sara"
#: A dashboard sentence is <= 12 words (F-E4 #17); 600 characters is generous
#: enough for the longest fact and keeps a single request bounded.
MAX_TEXT_CHARS = 600


def repo_root() -> Path:
    """The repository root, from this file (``tools/tts/cpsa_tts/config.py``)."""
    return Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else default


def _env_origins() -> tuple[str, ...]:
    raw = os.environ.get("CPSA_TTS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_ORIGINS
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration of the service and the CLI tools."""

    host: str = "127.0.0.1"
    port: int = 8765
    cache_dir: Path = DEFAULT_CACHE_DIR
    default_engine: str = DEFAULT_ENGINE
    default_voice: str = DEFAULT_VOICE
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS
    max_text_chars: int = MAX_TEXT_CHARS
    #: Kokoro is fast enough on the CPU and leaves the 6 GB GPU to XTTS.
    kokoro_device: str = "cpu"
    #: ``cuda`` when PyTorch sees the Blackwell GPU, ``cpu`` otherwise.
    xtts_device: str = "auto"
    #: XTTS speaking rate; 1.0 is the model default.
    xtts_speed: float = 1.0
    log_level: str = "INFO"
    samples_dir: Path = field(default_factory=lambda: repo_root() / "build" / "tts-samples")

    @classmethod
    def from_env(cls) -> Settings:
        defaults = Settings()
        try:
            port = int(os.environ.get("CPSA_TTS_PORT", defaults.port))
        except ValueError:
            port = defaults.port
        try:
            speed = float(os.environ.get("CPSA_TTS_XTTS_SPEED", defaults.xtts_speed))
        except ValueError:
            speed = defaults.xtts_speed
        return cls(
            host=os.environ.get("CPSA_TTS_HOST", defaults.host),
            port=port,
            cache_dir=_env_path("CPSA_TTS_CACHE_DIR", defaults.cache_dir),
            default_engine=os.environ.get("CPSA_TTS_ENGINE", defaults.default_engine),
            default_voice=os.environ.get("CPSA_TTS_VOICE", defaults.default_voice),
            allowed_origins=_env_origins(),
            kokoro_device=os.environ.get("CPSA_TTS_KOKORO_DEVICE", defaults.kokoro_device),
            xtts_device=os.environ.get("CPSA_TTS_XTTS_DEVICE", defaults.xtts_device),
            xtts_speed=speed,
            log_level=os.environ.get("CPSA_TTS_LOG_LEVEL", defaults.log_level).upper(),
            samples_dir=_env_path("CPSA_TTS_SAMPLES_DIR", repo_root() / "build" / "tts-samples"),
        )
