"""The edge-tts engine, with a fake ``edge_tts`` module: no network, no MP3."""

from __future__ import annotations

import importlib.machinery
import io
import sys
import types
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from cpsa_tts.engines import EdgeEngine, EngineError, UnknownVoiceError, _decode_audio

ITALIAN_VOICES = [
    {
        "ShortName": "it-IT-ElsaNeural",
        "Locale": "it-IT",
        "Gender": "Female",
        "FriendlyName": "Microsoft Elsa Online (Natural) - Italian (Italy)",
    },
    {
        "ShortName": "it-IT-IsabellaNeural",
        "Locale": "it-IT",
        "Gender": "Female",
        "FriendlyName": "Microsoft Isabella Online (Natural) - Italian (Italy)",
    },
    {
        "ShortName": "it-IT-GiuseppeMultilingualNeural",
        "Locale": "it-IT",
        "Gender": "Male",
        "FriendlyName": "Microsoft GiuseppeMultilingual Online (Natural)",
    },
    {
        "ShortName": "en-US-AriaNeural",
        "Locale": "en-US",
        "Gender": "Female",
        "FriendlyName": "Microsoft Aria Online (Natural) - English (United States)",
    },
]


def wav(seconds: float = 0.1, rate: int = 24_000, channels: int = 1) -> bytes:
    """The bytes an engine receives after decoding: a WAV of a quiet tone."""
    samples = np.full(int(rate * seconds), 0.2, dtype=np.float32)
    if channels > 1:
        samples = np.stack([samples] * channels, axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class FakeCommunicate:
    """Stands in for ``edge_tts.Communicate``; records what it was asked."""

    instances: list[FakeCommunicate] = []
    payload = wav()
    error: Exception | None = None

    def __init__(self, text: str, voice: str, **options: Any) -> None:
        self.text = text
        self.voice = voice
        self.options = options
        FakeCommunicate.instances.append(self)

    async def stream(self) -> Any:
        if FakeCommunicate.error is not None:
            raise FakeCommunicate.error
        if FakeCommunicate.payload:
            yield {"type": "audio", "data": FakeCommunicate.payload}
        yield {"type": "WordBoundary", "offset": 0}


def fake_edge_module(voices: list[dict[str, Any]] | None = None) -> types.ModuleType:
    module = types.ModuleType("edge_tts")
    module.__spec__ = importlib.machinery.ModuleSpec("edge_tts", loader=None)  # type: ignore[attr-defined]
    calls: list[int] = []

    async def list_voices(**_: Any) -> list[dict[str, Any]]:
        calls.append(1)
        return list(ITALIAN_VOICES if voices is None else voices)

    module.list_voices = list_voices  # type: ignore[attr-defined]
    module.Communicate = FakeCommunicate  # type: ignore[attr-defined]
    module.calls = calls  # type: ignore[attr-defined]
    return module


@pytest.fixture(autouse=True)
def edge_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    FakeCommunicate.instances = []
    FakeCommunicate.payload = wav()
    FakeCommunicate.error = None
    module = fake_edge_module()
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    return module


def test_edge_advertises_the_four_italian_voices_before_any_query() -> None:
    engine = EdgeEngine()
    assert [voice.id for voice in engine.voices] == [
        "it-IT-ElsaNeural",
        "it-IT-IsabellaNeural",
        "it-IT-DiegoNeural",
        "it-IT-GiuseppeMultilingualNeural",
    ]
    assert all(voice.lang == "it-IT" for voice in engine.voices)
    assert engine.device == "online"
    assert engine.loaded is False
    assert "online" in engine.licence


def test_edge_voice_list_is_queried_once_and_filtered_to_italy(edge_module) -> None:
    engine = EdgeEngine()
    voices = engine.refresh_voices()
    assert [voice.id for voice in voices] == [
        "it-IT-ElsaNeural",
        "it-IT-IsabellaNeural",
        "it-IT-GiuseppeMultilingualNeural",
    ]
    assert voices[0].name == "Elsa"
    assert voices[0].gender == "female"
    assert voices[2].name == "Giuseppe Multilingual"
    assert engine.refresh_voices() == voices
    assert len(edge_module.calls) == 1  # type: ignore[attr-defined]


def test_edge_synthesize_returns_24k_mono_float32(edge_module) -> None:
    engine = EdgeEngine(rate="-5%")
    samples = engine.synthesize("Ciao! Che bello vederti.", "it-IT-ElsaNeural")
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert samples.size == pytest.approx(2_400, abs=48)
    assert engine.loaded is True
    call = FakeCommunicate.instances[-1]
    assert call.text == "Ciao! Che bello vederti."
    assert call.voice == "it-IT-ElsaNeural"
    # The configured speaking rate reaches the Edge endpoint.
    assert call.options["rate"] == "-5%"


def test_edge_resamples_and_downmixes_foreign_audio() -> None:
    # The service's contract is 24 kHz mono whatever the endpoint returns.
    samples = _decode_audio(wav(seconds=0.2, rate=48_000, channels=2))
    assert samples.shape == (4_800,)


def test_edge_offline_error_mentions_the_cache(edge_module) -> None:
    FakeCommunicate.error = OSError("Network is unreachable")
    engine = EdgeEngine()
    with pytest.raises(EngineError, match="internet") as error:
        engine.synthesize("Ciao", "it-IT-ElsaNeural")
    assert "cache" in str(error.value)


def test_edge_unknown_voice_is_rejected_before_the_network(edge_module) -> None:
    engine = EdgeEngine()
    with pytest.raises(UnknownVoiceError):
        engine.synthesize("Ciao", "it-IT-NobodyNeural")
    assert FakeCommunicate.instances == []


def test_edge_load_survives_an_offline_voice_query(monkeypatch) -> None:
    module = fake_edge_module()

    async def boom(**_: Any) -> list[dict[str, Any]]:
        raise OSError("no route to host")

    module.list_voices = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    engine = EdgeEngine()
    engine.load()
    assert engine.loaded is True
    # The curated fallback (same order as a live query) is advertised offline.
    assert [voice.id for voice in engine.voices][0] == "it-IT-ElsaNeural"


def test_edge_without_audio_is_an_error(edge_module) -> None:
    FakeCommunicate.payload = b""
    engine = EdgeEngine()
    with pytest.raises(EngineError, match="no audio"):
        engine.synthesize("Ciao", "it-IT-ElsaNeural")


def test_edge_undecodable_payload_is_an_error() -> None:
    with pytest.raises(EngineError, match="undecodable"):
        _decode_audio(b"not audio at all")
