"""Disk cache: key format, atomic writes, counts."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from cpsa_tts.cache import WavCache, cache_key, wav_bytes


def test_key_is_sha1_of_the_triple_and_part_separated() -> None:
    assert cache_key("kokoro", "if_sara", "Ciao") == cache_key("kokoro", "if_sara", "Ciao")
    assert cache_key("kokoro", "if_sara", "Ciao") != cache_key("kokoro", "im_nicola", "Ciao")
    assert cache_key("kokoro", "if_sara", "Ciao") != cache_key("xtts", "if_sara", "Ciao")
    assert cache_key("kokoro", "if_sara", "Ciao") != cache_key("kokoro", "if_sara", "Ciao!")
    assert len(cache_key("kokoro", "if_sara", "Ciao")) == 40


def test_key_cannot_be_confused_by_the_separator() -> None:
    # "a", "b|c" and "a|b", "c" must not collide.
    assert cache_key("e", "b|c", "d") != cache_key("e", "b", "c|d")


def test_write_read_roundtrip_and_no_text_on_disk(tmp_path) -> None:
    cache = WavCache(tmp_path / "wavs")
    samples = np.zeros(2400, dtype=np.float32)
    path = cache.write("kokoro", "if_sara", "Il telefono è sul tavolo.", samples, 24_000)

    assert path.is_file()
    assert path.suffix == ".wav"
    assert cache.has("kokoro", "if_sara", "Il telefono è sul tavolo.")
    payload = cache.read("kokoro", "if_sara", "Il telefono è sul tavolo.")
    assert payload is not None and payload[:4] == b"RIFF"

    # The sentence is only stored as a digest, never in the file name.
    assert "telefono" not in path.name
    assert cache.entries() == 1
    assert cache.size_bytes() >= len(payload)


def test_entries_and_size_ignore_other_files(tmp_path) -> None:
    cache = WavCache(tmp_path)
    (tmp_path / "notes.txt").write_text("not a wav")
    cache.write("kokoro", "if_sara", "uno", np.zeros(240, dtype=np.float32), 24_000)
    assert cache.entries() == 1


def test_missing_entry_is_none(tmp_path) -> None:
    cache = WavCache(tmp_path)
    assert cache.read("kokoro", "if_sara", "mai sintetizzata") is None
    assert cache.has("kokoro", "if_sara", "mai sintetizzata") is False


def test_wav_bytes_is_a_pcm16_mono_wav() -> None:
    payload = wav_bytes(np.full(1200, 0.25, dtype=np.float32), 24_000)
    data, rate = sf.read(io.BytesIO(payload), dtype="float32")
    assert rate == 24_000
    assert data.shape == (1200,)
    assert abs(float(np.max(np.abs(data))) - 0.25) < 0.01


def test_clipping_never_writes_invalid_samples(tmp_path) -> None:
    cache = WavCache(tmp_path)
    samples = np.array([-2.0, 2.0, 0.0], dtype=np.float32)
    path = cache.write("kokoro", "if_sara", "clip", samples, 24_000)
    data, _ = sf.read(path, dtype="float32")
    assert float(np.max(np.abs(data))) <= 1.0
