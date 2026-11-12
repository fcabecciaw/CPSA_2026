"""Disk cache of synthesized WAV files.

Key: ``sha1(engine, voice, text)`` as frozen by the R2-V task, plus the speaking
rate for the engines that have one (XTTS): the same sentence at another speed is
another WAV. ``text`` is the normalized sentence (:func:`cpsa_tts.text.normalize`),
so the dashboard text and the pregenerated text meet on the same key. The
sentence itself is never written to disk, only its digest, so the cache does not
leak what the dashboard said. Files are written atomically (temp file +
``os.replace``) so a crash cannot leave a truncated WAV in the cache.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def cache_key(engine: str, voice: str, text: str, speed: float | None = None) -> str:
    """The sha1 key of one (engine, voice, text[, speed]), as a hex digest.

    ``speed=None`` (an engine without a rate) keeps the original R2-V key.
    """
    digest = hashlib.sha1()  # noqa: S324 - cache key, not a security boundary
    parts = [engine, voice, text]
    if speed is not None:
        parts.append(f"speed={speed:.2f}")
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class WavCache:
    """A directory of ``<sha1>.wav`` files, PCM 16 bit mono."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.wav"

    def has(self, engine: str, voice: str, text: str, speed: float | None = None) -> bool:
        return self.path(cache_key(engine, voice, text, speed)).is_file()

    def read(self, engine: str, voice: str, text: str, speed: float | None = None) -> bytes | None:
        path = self.path(cache_key(engine, voice, text, speed))
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def write(
        self,
        engine: str,
        voice: str,
        text: str,
        samples: np.ndarray,
        sample_rate: int,
        speed: float | None = None,
    ) -> Path:
        path = self.path(cache_key(engine, voice, text, speed))
        write_wav_atomic(path, samples, sample_rate)
        return path

    def entries(self) -> int:
        try:
            return sum(1 for child in self.directory.iterdir() if child.suffix == ".wav")
        except FileNotFoundError:
            return 0

    def size_bytes(self) -> int:
        total = 0
        try:
            for child in self.directory.iterdir():
                if child.suffix == ".wav":
                    total += child.stat().st_size
        except FileNotFoundError:
            return 0
        return total


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono samples as a WAV file in memory."""
    import io

    buffer = io.BytesIO()
    sf.write(buffer, _pcm16(samples), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_wav_atomic(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write one WAV next to its destination, then rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".wav", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as file:
            sf.write(file, _pcm16(samples), sample_rate, format="WAV", subtype="PCM_16")
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def _pcm16(samples: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] and force the shape soundfile expects for mono."""
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    return np.clip(array, -1.0, 1.0)
