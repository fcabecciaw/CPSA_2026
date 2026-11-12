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

from cpsa_tts.cache import WavCache
from cpsa_tts.config import Settings
from cpsa_tts.defaults import DefaultStore
from cpsa_tts.engines import EngineError, EngineRegistry, Voice
from cpsa_tts.phrases import Phrase
from cpsa_tts.pregenerate import PregenerateJob
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


ROW_TEXTS = (
    "Ciao! Che bello vederti.",
    "Il polpo pensa con i tentacoli.",
    "Sessanta sguardi in un secondo.",
)


def make_app(
    tmp_path: Path, engine: FakeEngine, rows: tuple[Phrase, ...] | None = None
) -> tuple[TestClient, EngineRegistry]:
    settings = Settings(
        cache_dir=tmp_path / "cache",
        default_engine="fake",
        default_voice="fake_sara",
    )
    registry = EngineRegistry()
    registry.register(engine)
    registry.set_defaults("fake", "fake_sara")
    cache = WavCache(settings.cache_dir)
    job = PregenerateJob(cache, phrases=lambda: list(rows or []))
    return TestClient(
        create_app(settings=settings, registry=registry, cache=cache, pregenerate=job)
    ), registry


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


def test_cors_answers_chrome_private_network_preflight(tmp_path, engine) -> None:
    # The board page (a private IP) calling this loopback service triggers
    # Chrome's Private/Local Network Access preflight.
    client, _ = make_app(tmp_path, engine)
    response = client.options(
        "/tts",
        headers={
            "Origin": "http://10.42.0.83:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-private-network"] == "true"


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


def test_put_default_persists_and_survives_a_restart(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    response = client.put("/default", json={"engine": "fake", "voice": "fake_nicola"})
    assert response.status_code == 200
    assert response.json()["defaults"] == {"engine": "fake", "voice": "fake_nicola"}
    assert response.json()["defaults_file"] == str(tmp_path / "cache" / "default.json")

    store = DefaultStore(tmp_path / "cache" / "default.json")
    assert store.read() == {"engine": "fake", "voice": "fake_nicola"}

    spoken = client.post("/tts", json={"text": "Ciao"})
    assert spoken.status_code == 200
    assert spoken.headers["x-tts-voice"] == "fake_nicola"

    # A restart of the service adopts the file, overriding the environment.
    restarted, _ = make_app(tmp_path, FakeEngine())
    health = restarted.get("/health").json()
    assert health["defaults"] == {"engine": "fake", "voice": "fake_nicola"}
    assert health["defaults_file"].endswith("default.json")


def test_put_default_rejects_unknown_engine_or_voice(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    assert client.put("/default", json={"engine": "nope", "voice": "x"}).status_code == 400
    assert client.put("/default", json={"engine": "fake", "voice": "nope"}).status_code == 400
    assert client.put("/default", json={"engine": "fake"}).status_code == 422
    assert not (tmp_path / "cache" / "default.json").exists()


def test_pregenerate_fills_the_cache_and_reports_progress(tmp_path, engine) -> None:
    rows = tuple(
        Phrase(id=f"rulebook.{index:03d}", text=text, source="rulebook", kind="rule")
        for index, text in enumerate(ROW_TEXTS)
    )
    client, _ = make_app(tmp_path, engine, rows)
    started = client.post("/pregenerate", json={"engine": "fake", "voice": "fake_sara"})
    assert started.status_code == 202
    body = started.json()
    assert body["engine"] == "fake"
    assert body["voice"] == "fake_sara"
    assert body["total"] == len(rows)

    deadline = time.monotonic() + 10
    status = client.get("/pregenerate").json()
    while status["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
        status = client.get("/pregenerate").json()
    assert status["running"] is False
    assert status["done"] == len(rows)
    assert status["synthesized"] == len(rows)
    assert status["finished_at"] is not None

    health = client.get("/health").json()
    assert health["cache"]["entries"] == len(rows)
    assert health["pregenerate"] == {
        "running": False,
        "engine": "fake",
        "voice": "fake_sara",
        "done": len(rows),
        "total": len(rows),
    }


def test_pregenerate_fills_one_job_at_a_time(tmp_path, engine) -> None:
    rows = tuple(
        Phrase(id=f"rulebook.{index:03d}", text=text, source="rulebook", kind="rule")
        for index, text in enumerate(ROW_TEXTS)
    )
    client, _ = make_app(tmp_path, engine, rows)
    engine.delay_s = 0.05
    first = client.post("/pregenerate", json={"engine": "fake", "voice": "fake_sara"})
    assert first.status_code == 202
    second = client.post("/pregenerate", json={"engine": "fake", "voice": "fake_nicola"})
    assert second.status_code == 409
    assert "already running" in second.json()["error"]
    assert second.json()["job"]["voice"] == "fake_sara"


def test_pregenerate_follows_the_defaults_and_rejects_unknown_engines(tmp_path, engine) -> None:
    rows = tuple(
        Phrase(id=f"rulebook.{index:03d}", text=text, source="rulebook", kind="rule")
        for index, text in enumerate(ROW_TEXTS)
    )
    client, _ = make_app(tmp_path, engine, rows)
    assert client.post("/pregenerate", json={"engine": "nope"}).status_code == 400
    idle = client.get("/pregenerate").json()
    assert idle["running"] is False
    assert idle["total"] == 0

    started = client.post("/pregenerate")
    assert started.status_code == 202
    assert started.json()["engine"] == "fake"
    assert started.json()["voice"] == "fake_sara"


def test_cors_allows_the_voice_picker_methods(tmp_path, engine) -> None:
    # ``PUT /default`` and ``POST /pregenerate`` come from the dashboard origin
    # on the board, i.e. through Chrome's private-network preflight.
    client, _ = make_app(tmp_path, engine)
    for method, path in (("PUT", "/default"), ("POST", "/pregenerate")):
        response = client.options(
            path,
            headers={
                "Origin": "http://10.42.0.83:8080",
                "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": "content-type",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert response.status_code == 200, path
        assert response.headers["access-control-allow-origin"] == "http://10.42.0.83:8080"
        assert response.headers["access-control-allow-private-network"] == "true"
        assert method in response.headers["access-control-allow-methods"]


class SpeedEngine(FakeEngine):
    """A fake engine with an adjustable speaking rate, like XTTS."""

    id = "xtts"
    voices = (Voice(id="ana_florence", name="Ana", lang="it", gender="female"),)

    def __init__(self) -> None:
        super().__init__()
        self.speed = 1.15


def make_speed_app(tmp_path: Path, engine: SpeedEngine) -> TestClient:
    settings = Settings(cache_dir=tmp_path / "cache", default_engine="xtts")
    registry = EngineRegistry()
    registry.register(engine)
    registry.set_defaults("xtts", "ana_florence")
    return TestClient(create_app(settings=settings, registry=registry))


def test_tts_speaks_the_normalized_text_and_shares_its_cache(tmp_path) -> None:
    engine = SpeedEngine()
    client = make_speed_app(tmp_path, engine)
    first = client.post("/tts", json={"text": "Sembra un'auto… dico bene?"})
    assert first.status_code == 200
    assert first.headers["x-tts-cache"] == "miss"
    assert engine.calls == [("Sembra un'auto, dico bene?", "ana_florence")]
    # Another spelling of the same spoken sentence is the same cache entry.
    second = client.post("/tts", json={"text": "Sembra un'auto... dico bene?!"})
    assert second.headers["x-tts-cache"] == "hit"
    client.post("/tts", json={"text": "Ciao a tutti! Siete in 3."})
    assert engine.calls[-1] == ("Ciao a tutti! Siete in 3", "ana_florence")


def test_tts_rejects_text_that_normalizes_to_nothing(tmp_path, engine) -> None:
    client, _ = make_app(tmp_path, engine)
    response = client.post("/tts", json={"text": "«…»"})
    assert response.status_code == 400
    assert engine.calls == []


def test_speed_is_part_of_the_cache_key_and_set_by_put_default(tmp_path) -> None:
    engine = SpeedEngine()
    client = make_speed_app(tmp_path, engine)
    assert client.get("/health").json()["defaults"] == {
        "engine": "xtts",
        "voice": "ana_florence",
        "speed": 1.15,
    }
    assert client.post("/tts", json={"text": "Ciao"}).headers["x-tts-cache"] == "miss"
    assert client.post("/tts", json={"text": "Ciao"}).headers["x-tts-cache"] == "hit"

    response = client.put(
        "/default", json={"engine": "xtts", "voice": "ana_florence", "speed": 1.3}
    )
    assert response.status_code == 200
    assert response.json()["defaults"]["speed"] == 1.3
    assert engine.speed == 1.3
    store = DefaultStore(tmp_path / "cache" / "default.json")
    assert store.read() == {"engine": "xtts", "voice": "ana_florence", "speed": 1.3}
    # Same sentence, new speed: a new WAV.
    assert client.post("/tts", json={"text": "Ciao"}).headers["x-tts-cache"] == "miss"
    assert len(engine.calls) == 2

    # A restart adopts the persisted speed.
    restarted_engine = SpeedEngine()
    make_speed_app(tmp_path, restarted_engine)
    assert restarted_engine.speed == 1.3


def test_put_default_rejects_a_bad_speed(tmp_path, engine) -> None:
    speedy = SpeedEngine()
    client = make_speed_app(tmp_path, speedy)
    too_fast = client.put("/default", json={"engine": "xtts", "voice": "ana_florence", "speed": 5})
    assert too_fast.status_code == 400
    assert speedy.speed == 1.15
    assert not (tmp_path / "cache" / "default.json").exists()

    plain, _ = make_app(tmp_path / "plain", engine)
    no_rate = plain.put("/default", json={"engine": "fake", "voice": "fake_sara", "speed": 1.2})
    assert no_rate.status_code == 400
