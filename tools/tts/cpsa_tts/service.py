"""HTTP service of the local TTS (FastAPI, 127.0.0.1:8765).

Contract used by the dashboard (frozen by R2-V):

- ``GET  /health``  -> readiness, engines, cache and last-request statistics.
- ``GET  /voices``  -> engines, licences, devices and selectable voices.
- ``POST /tts``     -> ``{text, engine?, voice?}`` answered with ``audio/wav``.
- ``PUT  /default`` -> ``{engine, voice}``: the default used when a request omits
  them, persisted in ``<cache dir>/default.json``.
- ``POST /pregenerate`` -> ``{engine?, voice?}`` starts a background fill of the
  fixed dashboard sentences; ``GET /pregenerate`` reports its progress.

Every synthesized sentence is cached on disk under
``~/.cache/cpsa-tts/<sha1(engine, voice, text)>.wav``; a cache hit costs one
file read. Synthesis runs in a worker thread so the event loop stays free.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .cache import WavCache, cache_key, wav_bytes
from .config import Settings
from .defaults import DefaultStore, apply_persisted
from .engines import (
    Engine,
    EngineError,
    EngineRegistry,
    UnknownEngineError,
    UnknownVoiceError,
    Voice,
)
from .pregenerate import JobRunningError, PregenerateJob

log = logging.getLogger("cpsa_tts.service")

#: How many recent synthesis times feed the p50 shown by ``/health``.
RECENT_MS = 50


class TtsRequest(BaseModel):
    """Body of ``POST /tts``; ``engine`` and ``voice`` default server-side."""

    text: str = Field(min_length=1)
    engine: str | None = None
    voice: str | None = None


class DefaultRequest(BaseModel):
    """Body of ``PUT /default``: both fields are required and validated."""

    engine: str = Field(min_length=1)
    voice: str = Field(min_length=1)


class PregenerateRequest(BaseModel):
    """Body of ``POST /pregenerate``; omitted fields follow the defaults."""

    engine: str | None = None
    voice: str | None = None


@dataclass
class Stats:
    """Counters of the running service, cheap enough to expose in ``/health``."""

    started_at: float = field(default_factory=time.monotonic)
    requests: int = 0
    cache_hits: int = 0
    synth_runs: int = 0
    errors: int = 0
    recent_ms: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_MS))
    last: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        recent = sorted(self.recent_ms)
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "synth_runs": self.synth_runs,
            "errors": self.errors,
            "uptime_s": round(time.monotonic() - self.started_at, 1),
            "p50_synth_ms": round(recent[len(recent) // 2], 1) if recent else None,
            "last": self.last,
        }


class _KeyLocks:
    """One async lock per cache key: two identical requests synthesize once."""

    def __init__(self) -> None:
        self._locks: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        import anyio

        lock = self._locks.get(key)
        if lock is None:
            lock = anyio.Lock()
            self._locks[key] = lock
        return lock


def create_app(
    settings: Settings | None = None,
    registry: EngineRegistry | None = None,
    cache: WavCache | None = None,
    defaults_store: DefaultStore | None = None,
    pregenerate: PregenerateJob | None = None,
) -> FastAPI:
    """Build the application; every collaborator can be injected by the tests."""
    settings = settings or Settings.from_env()
    registry = registry or EngineRegistry(
        kokoro_device=settings.kokoro_device,
        xtts_device=settings.xtts_device,
        xtts_speed=settings.xtts_speed,
        default_engine=settings.default_engine,
        default_voice=settings.default_voice,
        edge_rate=settings.edge_rate,
    )
    wav_cache = cache or WavCache(settings.cache_dir)
    store = defaults_store or DefaultStore(settings.defaults_file)
    # A choice made from the dashboard wins over the environment, so a restart
    # keeps the voice the user picked while comparing.
    apply_persisted(registry, store)
    job = pregenerate or PregenerateJob(wav_cache)
    stats = Stats()
    locks = _KeyLocks()

    app = FastAPI(title="cpsa-tts", version=__version__, docs_url=None, redoc_url=None)
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_methods=["GET", "POST", "PUT", "OPTIONS"],
            allow_headers=["Content-Type"],
            expose_headers=[
                "X-TTS-Engine",
                "X-TTS-Voice",
                "X-TTS-Cache",
                "X-TTS-Synth-Ms",
            ],
            max_age=600,
            # The dashboard page lives on the board (http://10.42.0.83:8080) and
            # calls this loopback service: Chrome's Private/Local Network Access
            # preflight carries Access-Control-Request-Private-Network and needs
            # Access-Control-Allow-Private-Network back, or it rejects the call.
            allow_private_network=True,
        )

    @app.exception_handler(UnknownEngineError)
    @app.exception_handler(UnknownVoiceError)
    async def _unknown(_: Request, error: Exception) -> JSONResponse:
        stats.errors += 1
        return JSONResponse({"error": str(error)}, status_code=400)

    @app.exception_handler(EngineError)
    async def _engine_error(_: Request, error: Exception) -> JSONResponse:
        stats.errors += 1
        log.exception("engine error")
        return JSONResponse({"error": str(error)}, status_code=502)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "cpsa-tts",
            "version": __version__,
            "defaults": registry.defaults(),
            "defaults_file": str(store.path),
            "engines": {
                row["id"]: {
                    "available": row["available"],
                    "loaded": row["loaded"],
                    "device": row["device"],
                    "error": row["error"],
                }
                for row in registry.info()
            },
            "cache": {
                "dir": str(wav_cache.directory),
                "entries": wav_cache.entries(),
                "bytes": wav_cache.size_bytes(),
            },
            "pregenerate": _job_summary(job.status()),
            "stats": stats.to_dict(),
        }

    @app.get("/voices")
    async def voices() -> dict[str, Any]:
        return {"defaults": registry.defaults(), "engines": registry.info()}

    @app.put("/default")
    async def set_default(body: DefaultRequest) -> dict[str, Any]:
        """Change the default engine/voice of every request that omits them."""
        registry.set_defaults(body.engine, body.voice)
        path = store.write(body.engine, body.voice)
        log.info("default voice -> %s/%s (%s)", body.engine, body.voice, path)
        return {"defaults": registry.defaults(), "defaults_file": str(path)}

    @app.post("/pregenerate")
    async def start_pregenerate(body: PregenerateRequest | None = None) -> JSONResponse:
        """Fill the cache with the fixed dashboard sentences, in the background."""
        request_body = body or PregenerateRequest()
        engine, voice = registry.resolve(request_body.engine, request_body.voice)
        try:
            status = job.start(engine, voice)
        except JobRunningError as error:
            return JSONResponse({"error": str(error), "job": job.status()}, status_code=409)
        return JSONResponse(status, status_code=202)

    @app.get("/pregenerate")
    async def pregenerate_status() -> dict[str, Any]:
        return job.status()

    @app.post("/tts")
    async def synthesize(body: TtsRequest) -> Response:
        text = body.text.strip()
        if not text:
            stats.errors += 1
            return JSONResponse({"error": "text is empty"}, status_code=400)
        if len(text) > settings.max_text_chars:
            stats.errors += 1
            return JSONResponse(
                {"error": f"text is longer than {settings.max_text_chars} characters"},
                status_code=400,
            )
        engine, voice = registry.resolve(body.engine, body.voice)
        stats.requests += 1
        key = cache_key(engine.id, voice.id, text)

        cached = wav_cache.read(engine.id, voice.id, text)
        if cached is not None:
            stats.cache_hits += 1
            stats.last = _last(engine, voice, text, 0.0, "hit")
            return _wav_response(cached, engine, voice, "hit", 0.0)

        lock = locks.get(key)
        async with lock:
            # Another request may have filled the cache while we waited.
            cached = wav_cache.read(engine.id, voice.id, text)
            if cached is not None:
                stats.cache_hits += 1
                stats.last = _last(engine, voice, text, 0.0, "hit")
                return _wav_response(cached, engine, voice, "hit", 0.0)

            started = time.perf_counter()
            try:
                samples = await _run_synthesis(registry, engine, text, voice)
            except UnknownVoiceError:
                raise
            except EngineError:
                raise
            except Exception as error:  # noqa: BLE001 - a model crash is a 502
                raise EngineError(f"{engine.id} synthesis failed: {error}") from error
            elapsed_ms = (time.perf_counter() - started) * 1_000
            wav_cache.write(engine.id, voice.id, text, samples, engine.sample_rate)
            payload = wav_bytes(samples, engine.sample_rate)

        stats.synth_runs += 1
        stats.recent_ms.append(elapsed_ms)
        stats.last = _last(engine, voice, text, elapsed_ms, "miss")
        log.info(
            "tts engine=%s voice=%s chars=%d audio_s=%.2f synth_ms=%.0f cache=miss device=%s",
            engine.id,
            voice.id,
            len(text),
            len(samples) / engine.sample_rate,
            elapsed_ms,
            engine.device,
        )
        return _wav_response(payload, engine, voice, "miss", elapsed_ms)

    return app


def _job_summary(status: dict[str, Any]) -> dict[str, Any]:
    """The few fields ``/health`` needs to show a fill in progress."""
    return {
        "running": status["running"],
        "engine": status["engine"],
        "voice": status["voice"],
        "done": status["done"],
        "total": status["total"],
    }


def _last(engine: Engine, voice: Voice, text: str, synth_ms: float, cache: str) -> dict[str, Any]:
    return {
        "engine": engine.id,
        "voice": voice.id,
        "chars": len(text),
        "synth_ms": round(synth_ms, 1),
        "cache": cache,
        "device": engine.device,
    }


async def _run_synthesis(registry: EngineRegistry, engine: Engine, text: str, voice: Voice) -> Any:
    """Load the engine (if needed) and synthesize off the event loop."""
    import anyio

    def work() -> Any:
        registry.load(engine.id)
        return engine.synthesize(text, voice.id)

    return await anyio.to_thread.run_sync(work)


def _wav_response(
    payload: bytes, engine: Engine, voice: Voice, cache: str, synth_ms: float
) -> Response:
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "X-TTS-Engine": engine.id,
            "X-TTS-Voice": voice.id,
            "X-TTS-Cache": cache,
            "X-TTS-Synth-Ms": f"{synth_ms:.1f}",
            "Cache-Control": "no-store",
        },
    )
