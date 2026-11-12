"""Engines of the local TTS service: Kokoro 82M, XTTS-v2 and Edge neural voices.

Every engine is imported lazily: ``GET /health`` and ``GET /voices`` must answer
in milliseconds even when no model is resident, and a broken XTTS install must
not prevent Kokoro from working. Every sentence is synthesized as float32 mono
at 24 kHz, then written as PCM 16 WAV, so the browser path and the disk cache
stay uniform whichever engine spoke.

Licences (checked 2026-11-12):
- Kokoro-82M: Apache-2.0 (<https://huggingface.co/hexgrad/Kokoro-82M>).
- XTTS-v2: Coqui Public Model License, non-commercial
  (<https://coqui.ai/cpml>). The R2-V task approved its use for this demo.
- Microsoft neural voices: no licence, only the read-aloud terms of service,
  reached through the unofficial ``edge-tts`` client (see README).
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import logging
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

log = logging.getLogger("cpsa_tts.engines")

#: Silence inserted between the sentence chunks of one utterance.
CHUNK_GAP_S = 0.14
SAMPLE_RATE = 24_000


class EngineError(RuntimeError):
    """The requested engine could not synthesize (missing model, OOM, ...)."""


class UnknownEngineError(EngineError):
    """The engine id is not one of the configured engines."""


class UnknownVoiceError(EngineError):
    """The voice id does not belong to the requested engine."""


@dataclass(frozen=True, slots=True)
class Voice:
    """One selectable voice of a local engine."""

    id: str
    name: str
    lang: str
    gender: str
    sample_rate: int = SAMPLE_RATE

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Engine(Protocol):
    """What the service and the CLI need from one TTS engine."""

    id: str
    label: str
    licence: str
    sample_rate: int

    @property
    def voices(self) -> tuple[Voice, ...]: ...

    @property
    def device(self) -> str: ...

    @property
    def loaded(self) -> bool: ...

    def dependencies_available(self) -> bool: ...

    def load(self) -> None: ...

    def synthesize(self, text: str, voice: str) -> np.ndarray: ...

    def unload(self) -> None: ...


def _join(chunks: list[np.ndarray], sample_rate: int) -> np.ndarray:
    """Concatenate chunks with a short pause, the way one utterance sounds."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return np.asarray(chunks[0], dtype=np.float32)
    gap = np.zeros(int(sample_rate * CHUNK_GAP_S), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        if index > 0:
            pieces.append(gap)
        pieces.append(np.asarray(chunk, dtype=np.float32))
    return np.concatenate(pieces)


class KokoroEngine:
    """Kokoro 82M, Apache-2.0, Italian voices ``if_sara`` and ``im_nicola``.

    The model runs on the CPU: a warm sentence takes well under 100 ms on this
    laptop, which leaves the 6 GB GPU entirely to XTTS.
    """

    id = "kokoro"
    label = "Kokoro 82M"
    licence = "Apache-2.0"
    repo_id = "hexgrad/Kokoro-82M"
    sample_rate = SAMPLE_RATE
    voices: tuple[Voice, ...] = (
        Voice(id="if_sara", name="Sara", lang="it", gender="female"),
        Voice(id="im_nicola", name="Nicola", lang="it", gender="male"),
    )

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._pipeline: Any = None
        self._lock = threading.Lock()
        self._loaded_once = False

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def dependencies_available(self) -> bool:
        return importlib.util.find_spec("kokoro") is not None

    def load(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                return
            from kokoro import KPipeline

            log.info("loading Kokoro 82M on %s", self._device)
            # Passing device="cpu" explicitly makes Kokoro ~300x slower on this
            # laptop (measured: 0.06 s vs 17 s warm); let Kokoro keep its default
            # device and only override it for the GPU.
            extra = {} if self._device in {"", "cpu"} else {"device": self._device}
            self._pipeline = KPipeline(lang_code="i", repo_id=self.repo_id, **extra)
            self._loaded_once = True

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        self._validate_voice(voice)
        self.load()
        pipeline = self._pipeline
        assert pipeline is not None  # for type checkers; load() guarantees it
        chunks: list[np.ndarray] = []
        with self._lock:
            for result in pipeline(text, voice=voice, speed=1.0):
                audio = getattr(result, "audio", None)
                if audio is None:
                    continue
                chunks.append(_as_float32(audio))
        if not chunks:
            raise EngineError("Kokoro returned no audio")
        return _join(chunks, self.sample_rate)

    def unload(self) -> None:
        with self._lock:
            self._pipeline = None
        gc.collect()

    def _validate_voice(self, voice: str) -> None:
        if voice not in {item.id for item in self.voices}:
            raise UnknownVoiceError(f"Kokoro has no voice {voice!r}")


class XttsEngine:
    """XTTS-v2 (Coqui), multilingual including Italian, GPU when available.

    The bundled speakers are multilingual; the four offered here were chosen for
    a warm Italian read and are easy to compare with ``cpsa-tts samples``.
    """

    id = "xtts"
    label = "XTTS-v2 (Coqui)"
    licence = "Coqui Public Model License (CPML), non-commercial"
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    language = "it"
    sample_rate = SAMPLE_RATE
    speakers = {
        "ana_florence": ("Ana Florence", "female"),
        "sofia_hellen": ("Sofia Hellen", "female"),
        "gilberto_mathias": ("Gilberto Mathias", "male"),
        "damien_black": ("Damien Black", "male"),
    }
    voices = tuple(
        Voice(id=voice_id, name=name, lang="it", gender=gender)
        for voice_id, (name, gender) in speakers.items()
    )

    def __init__(self, device: str = "auto", speed: float = 1.0) -> None:
        self._requested_device = device
        self._device = "cpu"
        self._speed = speed
        self._tts: Any = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        # Before the first load an "auto" engine has not resolved its device yet;
        # reporting the request avoids importing torch just for /health.
        return self._device if self._tts is not None else self._requested_device

    @property
    def loaded(self) -> bool:
        return self._tts is not None

    def dependencies_available(self) -> bool:
        return importlib.util.find_spec("TTS") is not None

    def _resolve_device(self) -> str:
        if self._requested_device not in {"auto", ""}:
            return self._requested_device
        try:
            import torch

            available = torch.cuda.is_available()
        except Exception:  # pragma: no cover - broken torch install
            return "cpu"
        return "cuda" if available else "cpu"

    def load(self) -> None:
        with self._lock:
            if self._tts is not None:
                return
            from TTS.api import TTS

            self._device = self._resolve_device()
            # The CPML is non-commercial; the R2-V task approved XTTS-v2 for this
            # demo, so the model manager must not stop on its interactive prompt.
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            log.info("loading XTTS-v2 on %s (CPML, non-commercial)", self._device)
            model = TTS(self.model_name, progress_bar=False)
            self._tts = model.to(self._device)
            # First call builds the vocoder's CUDA kernels and caches the speaker latents.
            self._tts.tts(
                text="Pronto.",
                speaker=self.speakers["ana_florence"][0],
                language=self.language,
                speed=self._speed,
            )

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        if voice not in self.speakers:
            raise UnknownVoiceError(f"XTTS has no voice {voice!r}")
        self.load()
        tts = self._tts
        assert tts is not None
        speaker = self.speakers[voice][0]
        with self._lock:
            try:
                import torch

                with torch.inference_mode():
                    audio = tts.tts(
                        text=text,
                        speaker=speaker,
                        language=self.language,
                        speed=self._speed,
                    )
            except Exception as error:  # noqa: BLE001 - reported as a 502
                raise EngineError(f"XTTS synthesis failed: {error}") from error
        samples = _as_float32(audio)
        if samples.size == 0:
            raise EngineError("XTTS returned no audio")
        return samples

    def unload(self) -> None:
        with self._lock:
            self._tts = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - nothing to release
            pass


class EdgeEngine:
    """Microsoft neural voices through ``edge-tts`` (unofficial, online only).

    The client talks to the Edge read-aloud endpoint, so every new sentence needs
    internet; a sentence already in the disk cache keeps working offline. The
    service converts the MP3 it returns to the same 24 kHz mono float32 PCM the
    other engines produce, so the cache and the browser path never see a
    difference.

    The Italian voice list is queried once and kept in memory: ``/voices`` over
    the network would make every health probe wait for Microsoft. Until the
    first query the curated fallback below is advertised.
    """

    id = "edge"
    label = "Microsoft Neural (edge-tts)"
    licence = "Microsoft read-aloud terms, unofficial client, online only"
    language = "it-IT"
    sample_rate = SAMPLE_RATE
    #: Same order as :func:`_voice_order` (women first, then by id), so picking
    #: the first voice gives Elsa whether or not the list was already queried.
    fallback_voices: tuple[Voice, ...] = (
        Voice(id="it-IT-ElsaNeural", name="Elsa", lang="it-IT", gender="female"),
        Voice(id="it-IT-IsabellaNeural", name="Isabella", lang="it-IT", gender="female"),
        Voice(id="it-IT-DiegoNeural", name="Diego", lang="it-IT", gender="male"),
        Voice(
            id="it-IT-GiuseppeMultilingualNeural",
            name="Giuseppe Multilingual",
            lang="it-IT",
            gender="male",
        ),
    )

    def __init__(
        self,
        rate: str = "+0%",
        volume: str = "+0%",
        timeout_s: int = 15,
        voices: tuple[Voice, ...] | None = None,
    ) -> None:
        self._rate = rate
        self._volume = volume
        self._timeout_s = timeout_s
        self._voices: tuple[Voice, ...] | None = voices
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return "online"

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def voices(self) -> tuple[Voice, ...]:
        return self._voices if self._voices is not None else self.fallback_voices

    def dependencies_available(self) -> bool:
        return importlib.util.find_spec("edge_tts") is not None

    def load(self) -> None:
        if importlib.util.find_spec("edge_tts") is None:
            raise EngineError("edge-tts is not installed: uv sync in tools/tts")
        with self._lock:
            self._loaded = True
        # Best effort: offline or a slow network must not fail the load, the
        # fallback voice list already covers the Italian neural voices.
        try:
            self.refresh_voices()
        except EngineError as error:
            log.warning("%s", error)

    def refresh_voices(self) -> tuple[Voice, ...]:
        """Query the Italian voices once and remember them."""
        if self._voices is not None:
            return self._voices
        try:
            raw = asyncio.run(self._list_voices())
        except Exception as error:  # noqa: BLE001 - offline, DNS, HTTP 5xx
            raise EngineError(
                f"edge-tts could not list the voices ({error}); "
                "the engine needs internet, cached sentences keep working offline"
            ) from error
        italian = [_edge_voice(row) for row in raw if _is_italian(row)]
        found = tuple(sorted((voice for voice in italian if voice is not None), key=_voice_order))
        with self._lock:
            self._voices = found or self.fallback_voices
        return self.voices

    async def _list_voices(self) -> list[Any]:
        import edge_tts

        return list(await edge_tts.list_voices())

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        self._validate_voice(voice)
        self.load()
        try:
            payload = asyncio.run(self._download(text, voice))
        except EngineError:
            raise
        except Exception as error:  # noqa: BLE001 - offline, timeout, HTTP error
            raise EngineError(
                f"edge-tts synthesis failed: {error}. The edge engine needs internet; "
                "sentences already in the cache still play offline"
            ) from error
        return _decode_audio(payload)

    async def _download(self, text: str, voice: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=self._rate,
            volume=self._volume,
            receive_timeout=self._timeout_s,
        )
        payload = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                payload += chunk["data"]
        if not payload:
            raise EngineError("edge-tts returned no audio")
        return bytes(payload)

    def unload(self) -> None:
        self._loaded = False

    def _validate_voice(self, voice: str) -> None:
        if voice not in {item.id for item in self.voices}:
            known = ", ".join(item.id for item in self.voices)
            raise UnknownVoiceError(f"edge has no voice {voice!r} (known: {known})")


def _is_italian(row: Mapping[str, object]) -> bool:
    locale = str(row.get("Locale", "")).lower()
    return locale == "it-it"


def _edge_voice(row: Mapping[str, object]) -> Voice | None:
    """One ``edge_tts.list_voices()`` row as a :class:`Voice`, or None."""
    short = str(row.get("ShortName", "")).strip()
    if not short:
        return None
    return Voice(
        id=short,
        name=_friendly_name(short),
        lang=str(row.get("Locale", EdgeEngine.language)),
        gender=str(row.get("Gender", "female")).lower(),
    )


def _friendly_name(short_name: str) -> str:
    """``it-IT-GiuseppeMultilingualNeural`` -> ``Giuseppe Multilingual``."""
    base = short_name.split("-")[-1]
    if base.endswith("Neural"):
        base = base[: -len("Neural")]
    words = re.findall(r"[A-Z][a-z]+", base) or [base]
    return " ".join(words)


def _voice_order(voice: Voice) -> tuple[str, str]:
    return (voice.gender, voice.id)


def _decode_audio(payload: bytes) -> np.ndarray:
    """Decode an edge-tts MP3 payload to float32 mono at the service rate."""
    import io

    import soundfile as sf

    try:
        data, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except Exception as error:  # noqa: BLE001 - a truncated stream, a bad codec
        raise EngineError(f"edge-tts returned undecodable audio: {error}") from error
    mono = data.mean(axis=1) if data.shape[1] > 1 else data.reshape(-1)
    mono = _as_float32(mono)
    if mono.size == 0:
        raise EngineError("edge-tts returned no audio")
    if rate != SAMPLE_RATE:
        mono = _resample(mono, int(rate), SAMPLE_RATE)
    return mono


def _resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear resampling: the engines all end up at 24 kHz without torchaudio."""
    if samples.size == 0 or from_rate == to_rate or from_rate <= 0:
        return samples
    duration = samples.size / from_rate
    target = max(1, int(round(duration * to_rate)))
    source_time = np.arange(samples.size, dtype=np.float64) / from_rate
    target_time = np.arange(target, dtype=np.float64) / to_rate
    return np.interp(target_time, source_time, samples).astype(np.float32)


def _as_float32(audio: object) -> np.ndarray:
    """Convert a torch tensor, a list of floats or an ndarray to float32 mono."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class EngineRegistry:
    """The configured engines, resolved lazily and loaded on first use."""

    def __init__(
        self,
        kokoro_device: str = "cpu",
        xtts_device: str = "auto",
        xtts_speed: float = 1.0,
        default_engine: str = "kokoro",
        default_voice: str = "if_sara",
        edge_rate: str = "+0%",
    ) -> None:
        self._engines: dict[str, Engine] = {
            "kokoro": KokoroEngine(device=kokoro_device),
            "xtts": XttsEngine(device=xtts_device, speed=xtts_speed),
            "edge": EdgeEngine(rate=edge_rate),
        }
        self._errors: dict[str, str] = {}
        self._default_engine = default_engine
        self.set_defaults(default_engine, default_voice)

    @property
    def engines(self) -> dict[str, Engine]:
        return self._engines

    def register(self, engine: Engine) -> None:
        """Add an engine (used by the tests and by future engines)."""
        self._engines[engine.id] = engine

    def get(self, engine_id: str) -> Engine:
        engine = self._engines.get(engine_id)
        if engine is None:
            known = ", ".join(sorted(self._engines))
            raise UnknownEngineError(f"unknown engine {engine_id!r} (known: {known})")
        return engine

    def resolve(self, engine_id: str | None, voice_id: str | None) -> tuple[Engine, Voice]:
        """Pick the engine and voice of one request from the configured defaults."""
        engine = self.get(engine_id) if engine_id else self.default_engine()
        voices = {voice.id: voice for voice in engine.voices}
        if voice_id:
            voice = voices.get(voice_id)
            if voice is None:
                known = ", ".join(sorted(voices))
                raise UnknownVoiceError(
                    f"unknown voice {voice_id!r} for {engine.id} (known: {known})"
                )
            return engine, voice
        if engine.id == self._default_engine:
            return engine, voices[self._default_voice]
        return engine, next(iter(voices.values()))

    def defaults(self) -> dict[str, str]:
        return {"engine": self._default_engine, "voice": self._default_voice}

    def default_engine(self) -> Engine:
        return self._engines[self._default_engine]

    def set_defaults(self, engine_id: str, voice_id: str) -> None:
        """Validate the configured default engine/voice and remember them."""
        engine = self.get(engine_id)
        if voice_id not in {voice.id for voice in engine.voices}:
            known = ", ".join(voice.id for voice in engine.voices)
            raise UnknownVoiceError(f"unknown voice {voice_id!r} for {engine_id} (known: {known})")
        self._default_engine = engine_id
        self._default_voice = voice_id

    def load(self, engine_id: str) -> None:
        engine = self.get(engine_id)
        try:
            engine.load()
        except EngineError as error:
            self._errors[engine_id] = str(error)
            raise
        except Exception as error:  # noqa: BLE001 - OOM and friends become EngineError
            self._errors[engine_id] = str(error)
            raise EngineError(f"{engine_id} failed to load: {error}") from error
        self._errors.pop(engine_id, None)

    def info(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for engine in self._engines.values():
            available = engine.dependencies_available()
            if engine.id in self._errors:
                available = False
            rows.append(
                {
                    "id": engine.id,
                    "label": engine.label,
                    "licence": engine.licence,
                    "device": engine.device,
                    "available": available,
                    "loaded": engine.loaded,
                    "error": self._errors.get(engine.id),
                    "voices": [voice.to_dict() for voice in engine.voices],
                }
            )
        return rows
