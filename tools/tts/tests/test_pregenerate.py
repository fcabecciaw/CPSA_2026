"""The background cache fill behind ``POST /pregenerate`` (no HTTP, no model)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from cpsa_tts.cache import WavCache
from cpsa_tts.config import repo_root
from cpsa_tts.engines import EngineError, Voice
from cpsa_tts.phrases import Phrase, collect
from cpsa_tts.pregenerate import JobRunningError, PregenerateJob

ROWS = tuple(
    Phrase(
        id=f"rulebook.{index:03d}", text=f"Frase numero {index}.", source="rulebook", kind="rule"
    )
    for index in range(4)
)


class FakeEngine:
    id = "fake"
    label = "Fake engine"
    licence = "Apache-2.0"
    sample_rate = 24_000
    voices = (Voice(id="fake_sara", name="Sara", lang="it", gender="female"),)

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.delay_s = 0.0
        self.fail_at: int | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def loaded(self) -> bool:
        return True

    def dependencies_available(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        with self._lock:
            self.calls.append(text)
            position = len(self.calls) - 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_at is not None and position == self.fail_at:
            raise EngineError("edge-tts synthesis failed: no internet")
        return np.full(240, 0.1, dtype=np.float32)

    def unload(self) -> None:
        return None


def make_job(tmp_path: Path, engine: FakeEngine, rows=ROWS) -> PregenerateJob:
    return PregenerateJob(WavCache(tmp_path / "cache"), phrases=lambda: list(rows))


def wait_for_end(job: PregenerateJob, timeout_s: float = 10.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = job.status()
        if not status["running"]:
            return status
        time.sleep(0.01)
    raise AssertionError("pregenerate did not finish in time")


def test_job_fills_the_cache_and_reports_progress(tmp_path) -> None:
    engine = FakeEngine()
    job = make_job(tmp_path, engine)
    voice = engine.voices[0]
    started = job.start(engine, voice)
    assert started["running"] is True
    assert started["engine"] == "fake"
    assert started["voice"] == "fake_sara"
    assert started["total"] == len(ROWS)

    status = wait_for_end(job)
    assert status["running"] is False
    assert status["done"] == len(ROWS)
    assert status["synthesized"] == len(ROWS)
    assert status["cached"] == 0
    assert status["failed"] == 0
    assert status["error"] is None
    assert status["elapsed_s"] is not None
    assert status["finished_at"] is not None
    assert WavCache(tmp_path / "cache").entries() == len(ROWS)


def test_second_run_counts_the_cache_instead_of_synthesizing(tmp_path) -> None:
    engine = FakeEngine()
    cache = WavCache(tmp_path / "cache")
    job = PregenerateJob(cache, phrases=lambda: list(ROWS))
    job.start(engine, engine.voices[0])
    wait_for_end(job)

    job.start(engine, engine.voices[0])
    status = wait_for_end(job)
    assert status["synthesized"] == 0
    assert status["cached"] == len(ROWS)
    assert len(engine.calls) == len(ROWS)


def test_one_job_at_a_time(tmp_path) -> None:
    engine = FakeEngine()
    engine.delay_s = 0.2
    job = make_job(tmp_path, engine)
    job.start(engine, engine.voices[0])
    with pytest.raises(JobRunningError, match="already running"):
        job.start(engine, engine.voices[0])
    wait_for_end(job)


def test_a_failing_sentence_stops_the_fill_and_says_why(tmp_path) -> None:
    engine = FakeEngine()
    engine.fail_at = 1
    job = make_job(tmp_path, engine)
    job.start(engine, engine.voices[0])
    status = wait_for_end(job)
    assert status["running"] is False
    assert "no internet" in str(status["error"])
    assert status["done"] == len(ROWS)
    assert status["failed"] == len(ROWS) - 1
    assert WavCache(tmp_path / "cache").entries() == 1


def test_job_without_phrases_uses_the_dashboard_sources(tmp_path) -> None:
    engine = FakeEngine()
    job = PregenerateJob(WavCache(tmp_path / "cache"))
    # The real phrase source is the dashboard's TypeScript files; only two
    # sentences are filled so the test stays fast.
    rows = list(collect(repo_root()).phrases[:2])
    assert rows and all(row.text for row in rows)
    job.start(engine, engine.voices[0], rows=rows)
    status = wait_for_end(job)
    assert status["total"] == 2


def test_job_synthesizes_the_normalized_text_once_per_spoken_sentence(tmp_path) -> None:
    engine = FakeEngine()
    rows = (
        Phrase(id="scene.0000", text="Sembra un'auto… dico bene?", source="scene", kind="x"),
        # Spoken identically: already cached by the line above.
        Phrase(id="scene.0001", text="Sembra un'auto... dico bene?", source="scene", kind="x"),
    )
    job = make_job(tmp_path, engine, rows)
    job.start(engine, engine.voices[0])
    status = wait_for_end(job)
    assert status["synthesized"] == 1
    assert status["cached"] == 1
    assert engine.calls == ["Sembra un'auto, dico bene?"]
