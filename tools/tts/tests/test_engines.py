"""Engine registry, voice tables and the pure helpers (no model is loaded)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa_tts.cache import WavCache
from cpsa_tts.config import DEFAULT_ORIGINS, Settings, repo_root
from cpsa_tts.engines import (
    CHUNK_GAP_S,
    EngineError,
    EngineRegistry,
    KokoroEngine,
    UnknownEngineError,
    UnknownVoiceError,
    Voice,
    XttsEngine,
    _as_float32,
    _join,
)


def test_kokoro_exposes_the_two_italian_voices() -> None:
    engine = KokoroEngine()
    assert [voice.id for voice in engine.voices] == ["if_sara", "im_nicola"]
    assert all(voice.lang == "it" for voice in engine.voices)
    assert engine.licence == "Apache-2.0"
    assert engine.loaded is False


def test_xtts_offers_four_curated_italian_speakers() -> None:
    engine = XttsEngine()
    assert [voice.id for voice in engine.voices] == [
        "ana_florence",
        "sofia_hellen",
        "gilberto_mathias",
        "damien_black",
    ]
    assert "non-commercial" in engine.licence
    # The bundled model is multilingual; the service only speaks Italian.
    assert engine.language == "it"


def test_xtts_unknown_voice_is_rejected_before_loading() -> None:
    engine = XttsEngine()
    with pytest.raises(UnknownVoiceError):
        engine.synthesize("Ciao", "qualcuno")


def test_xtts_device_auto_picks_cuda_when_present(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert XttsEngine(device="auto")._resolve_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert XttsEngine(device="auto")._resolve_device() == "cpu"
    assert XttsEngine(device="cpu")._resolve_device() == "cpu"


def test_registry_resolves_defaults_and_validates() -> None:
    registry = EngineRegistry(default_engine="kokoro", default_voice="im_nicola")
    engine, voice = registry.resolve(None, None)
    assert engine.id == "kokoro"
    assert voice.id == "im_nicola"

    engine, voice = registry.resolve("xtts", None)
    assert engine.id == "xtts"
    assert voice.id == "ana_florence"

    engine, voice = registry.resolve("xtts", "damien_black")
    assert voice.id == "damien_black"

    with pytest.raises(UnknownEngineError):
        registry.resolve("bark", None)
    with pytest.raises(UnknownVoiceError):
        registry.resolve("kokoro", "af_heart")


def test_registry_rejects_a_default_voice_of_another_engine() -> None:
    with pytest.raises(UnknownVoiceError):
        EngineRegistry(default_engine="kokoro", default_voice="ana_florence")


def test_registry_info_is_cheap_and_never_loads() -> None:
    registry = EngineRegistry()
    rows = {row["id"]: row for row in registry.info()}
    assert set(rows) == {"kokoro", "xtts", "edge"}
    assert rows["kokoro"]["loaded"] is False
    assert rows["kokoro"]["available"] is True
    assert rows["xtts"]["available"] is True
    assert rows["edge"]["available"] is True
    assert rows["edge"]["device"] == "online"
    assert rows["kokoro"]["device"] == "cpu"


def test_join_inserts_a_short_pause_between_chunks() -> None:
    joined = _join([np.ones(240, dtype=np.float32), np.ones(240, dtype=np.float32)], 24_000)
    gap = int(24_000 * CHUNK_GAP_S)
    assert joined.shape == (240 * 2 + gap,)
    assert float(np.max(np.abs(joined[240 : 240 + gap]))) == 0.0


def test_as_float32_accepts_torch_tensors() -> None:
    import torch

    array = _as_float32(torch.tensor([[0.5, -0.5]], dtype=torch.float32))
    assert array.dtype == np.float32
    assert array.shape == (2,)


def test_settings_read_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CPSA_TTS_PORT", "9000")
    monkeypatch.setenv("CPSA_TTS_DEFAULT_ENGINE", "xtts")
    monkeypatch.setenv("CPSA_TTS_DEFAULT_VOICE", "sofia_hellen")
    monkeypatch.setenv("CPSA_TTS_ALLOWED_ORIGINS", "http://localhost:3000, http://x.test")
    monkeypatch.setenv("CPSA_TTS_CACHE_DIR", "/tmp/cpsa-tts-test")
    settings = Settings.from_env()
    assert settings.port == 9000
    assert settings.default_engine == "xtts"
    assert settings.default_voice == "sofia_hellen"
    assert settings.allowed_origins == ("http://localhost:3000", "http://x.test")
    assert str(settings.cache_dir) == "/tmp/cpsa-tts-test"
    # The persisted default lives inside the cache directory.
    assert str(settings.defaults_file) == "/tmp/cpsa-tts-test/default.json"


def test_settings_accept_the_legacy_engine_names(monkeypatch) -> None:
    monkeypatch.delenv("CPSA_TTS_DEFAULT_ENGINE", raising=False)
    monkeypatch.delenv("CPSA_TTS_DEFAULT_VOICE", raising=False)
    monkeypatch.setenv("CPSA_TTS_ENGINE", "edge")
    monkeypatch.setenv("CPSA_TTS_VOICE", "it-IT-ElsaNeural")
    settings = Settings.from_env()
    assert settings.default_engine == "edge"
    assert settings.default_voice == "it-IT-ElsaNeural"


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.default_engine == "kokoro"
    assert settings.edge_rate == "+0%"
    assert settings.defaults_file.name == "default.json"
    assert settings.allowed_origins == DEFAULT_ORIGINS
    assert settings.cache_dir.name == "cpsa-tts"
    assert (repo_root() / "web" / "src").is_dir()


def test_engine_error_hierarchy() -> None:
    assert issubclass(UnknownVoiceError, EngineError)
    assert issubclass(UnknownEngineError, EngineError)


def test_registry_wraps_a_load_failure() -> None:
    class Broken:
        id = "broken"
        label = "Broken"
        licence = "Apache-2.0"
        sample_rate = 24_000
        voices = (Voice(id="broken", name="Broken", lang="it", gender="female"),)

        @property
        def device(self) -> str:
            return "cuda"

        @property
        def loaded(self) -> bool:
            return False

        def dependencies_available(self) -> bool:
            return True

        def load(self) -> None:
            raise RuntimeError("CUDA out of memory")

        def synthesize(self, text: str, voice: str) -> np.ndarray:
            raise AssertionError("never reached")

        def unload(self) -> None:
            return None

    registry = EngineRegistry(default_engine="kokoro", default_voice="if_sara")
    registry.register(Broken())
    with pytest.raises(EngineError, match="CUDA out of memory"):
        registry.load("broken")
    rows = {row["id"]: row for row in registry.info()}
    assert rows["broken"]["available"] is False
    assert "CUDA out of memory" in str(rows["broken"]["error"])


def test_cache_shared_by_cli_and_service(tmp_path) -> None:
    cache = WavCache(tmp_path)
    assert cache.directory == tmp_path
