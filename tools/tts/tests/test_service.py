"""HTTP contract of the local TTS service, with a fake engine (no models)."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cpsa_tts.config import Settings
from cpsa_tts.engines import EngineError, EngineRegistry, Voice
from cpsa_tts.service import create_app


class FakeEngine:
    """Deterministic engine: 0.1 s of a quiet tone, counting its calls."""

    id = "fake"
    label = "Fake engine"
    licence = "Apache-2.0"
    sample_rate = 24_000
    voices = (
        Voice(id="fake_sara", name="Sara", lang="it", gender="female"),
        Voice(id="fake_nicola", name="Nicola", lang="it", gender="male"),
    )

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._loaded = False
        self._lock = threading.Lock()
        self.fail = False
        self.delay_s = 0.0

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def loaded(self) -> bool:
        return self._loaded

    def dependencies_available(self) -> bool:
        return True

    def load(self) -> None:
        self._loaded = True

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        with self._lock:
            self.calls.append((text, voice))
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise EngineError("boom")
        return np.full(2_400, 0.1, dtype=np.float32)

    def unload(self) -> None:
        self._loaded = False


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


def make_app(tmp_path: Path, engine: FakeEngine) -> tuple[TestClient, EngineRegistry]:
    settings = Settings(
        cache_dir=tmp_path / "cache",
        default_engine="fake",
        default_voice="fake_sara",
    )
    registry = EngineRegistry()
    registry.register(engine)
    registry.set_defaults("fake", "fake_sara")
    return TestClient(create_app(settings=settings, registry=registry)), registry


def test_health_reports_defaults_engines_and_cache(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["defaults"] == {"engine": "fake", "voice": "fake_sara"}
    assert body["engines"]["fake"]["available"] is True
    assert body["engines"]["fake"]["loaded"] is False
    assert body["cache"]["dir"] == str(tmp_path / "cache")
    assert body["cache"]["entries"] == 0
    # A health probe must never load a model.
    assert engine.loaded is False


def test_voices_lists_every_voice_and_its_licence(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    body = client.get("/voices").json()
    row = next(item for item in body["engines"] if item["id"] == "fake")
    assert row["licence"] == "Apache-2.0"
    assert [voice["id"] for voice in row["voices"]] == ["fake_sara", "fake_nicola"]
    assert row["voices"][0]["sample_rate"] == 24_000


def test_tts_miss_then_hit_writes_the_disk_cache(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    payload = {"text": "Ciao! Che bello vederti.", "engine": "fake", "voice": "fake_sara"}

    first = client.post("/tts", json=payload)
    assert first.status_code == 200
    assert first.headers["content-type"] == "audio/wav"
    assert first.content[:4] == b"RIFF"
    assert first.headers["x-tts-engine"] == "fake"
    assert first.headers["x-tts-voice"] == "fake_sara"
    assert first.headers["x-tts-cache"] == "miss"
    assert len(engine.calls) == 1
    assert engine.loaded is True

    second = client.post("/tts", json=payload)
    assert second.status_code == 200
    assert second.headers["x-tts-cache"] == "hit"
    assert second.content == first.content
    assert len(engine.calls) == 1

    health = client.get("/health").json()
    assert health["cache"]["entries"] == 1
    assert health["stats"]["requests"] == 2
    assert health["stats"]["cache_hits"] == 1
    assert health["stats"]["last"]["cache"] == "hit"


def test_tts_defaults_to_the_configured_engine_and_voice(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    response = client.post("/tts", json={"text": "Solo il testo."})
    assert response.status_code == 200
    assert response.headers["x-tts-engine"] == "fake"
    assert response.headers["x-tts-voice"] == "fake_sara"
    assert engine.calls == [("Solo il testo.", "fake_sara")]


def test_tts_explicit_voice_and_engine(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    response = client.post("/tts", json={"text": "Ciao", "engine": "fake", "voice": "fake_nicola"})
    assert response.status_code == 200
    assert response.headers["x-tts-voice"] == "fake_nicola"
    assert engine.calls == [("Ciao", "fake_nicola")]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"text": "Ciao", "engine": "nope"}, 400),
        ({"text": "Ciao", "engine": "fake", "voice": "nope"}, 400),
        ({"text": ""}, 422),
        ({"text": "   "}, 400),
    ],
)
def test_bad_requests_are_rejected_without_synthesizing(tmp_path, engine, body, status) -> None:
    client, _ = make_app(tmp_path, engine)
    response = client.post("/tts", json=body)
    assert response.status_code == status
    assert engine.calls == []


def test_text_longer_than_the_cap_is_rejected(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    response = client.post("/tts", json={"text": "a" * 601})
    assert response.status_code == 400
    assert "600" in response.json()["error"]
    assert engine.calls == []


def test_engine_failure_is_a_502(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    engine.fail = True
    response = client.post("/tts", json={"text": "Ciao"})
    assert response.status_code == 502
    assert "boom" in response.json()["error"]
    health = client.get("/health").json()
    assert health["stats"]["errors"] == 1
    assert health["cache"]["entries"] == 0


def test_cors_allows_the_two_dashboard_origins_and_nothing_else(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    for origin in ("http://localhost:5173", "http://10.42.0.83:8080"):
        response = client.options(
            "/tts",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200, origin
        assert response.headers["access-control-allow-origin"] == origin

    response = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_concurrent_identical_requests_synthesize_once(tmp_path, engine) -> None:
    settings = Settings(
        cache_dir=tmp_path / "cache", default_engine="fake", default_voice="fake_sara"
    )
    registry = EngineRegistry()
    registry.register(engine)
    registry.set_defaults("fake", "fake_sara")
    app = create_app(settings=settings, registry=registry)
    engine.delay_s = 0.05

    async def run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://tts") as client:
            return await asyncio.gather(
                *(client.post("/tts", json={"text": "Stessa frase."}) for _ in range(4))
            )

    responses = asyncio.run(run())
    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    assert {response.content for response in responses} == {responses[0].content}
    assert len(engine.calls) == 1
