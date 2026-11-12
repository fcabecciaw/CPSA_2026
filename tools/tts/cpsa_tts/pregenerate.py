"""Background fill of the fixed dashboard sentences for one engine/voice.

``POST /pregenerate`` answers immediately and a daemon thread synthesizes the
same sentences the CLI ``pregenerate`` command fills, so the dashboard can offer
a "Prepara frasi" button and watch the progress on ``GET /pregenerate``. One job
at a time: two fills would fight over the same cache and the same model.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from .cache import WavCache
from .config import repo_root
from .engines import Engine, EngineError, Voice, engine_speed
from .phrases import Phrase, collect_for_cache
from .text import normalize

log = logging.getLogger("cpsa_tts.pregenerate")


class JobRunningError(RuntimeError):
    """A pregenerate job is already filling the cache."""


@dataclass
class JobState:
    """Progress of the current (or last) fill, cheap to serialize in ``/health``."""

    running: bool = False
    engine: str | None = None
    voice: str | None = None
    total: int = 0
    done: int = 0
    synthesized: int = 0
    cached: int = 0
    failed: int = 0
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_s: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PregenerateJob:
    """One background fill at a time, reporting progress through a lock."""

    cache: WavCache
    #: Where the fixed sentences come from; ``None`` reads the dashboard source.
    phrases: Callable[[], Sequence[Phrase]] | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._state = JobState()
        self._thread: threading.Thread | None = None

    def status(self) -> dict[str, object]:
        with self._lock:
            return self._state.to_dict()

    def start(
        self,
        engine: Engine,
        voice: Voice,
        rows: Sequence[Phrase] | None = None,
    ) -> dict[str, object]:
        """Start a fill; ``JobRunningError`` when one is already running."""
        with self._lock:
            if self._state.running:
                raise JobRunningError(
                    f"pregenerate already running for {self._state.engine}/{self._state.voice}"
                )
            selected = list(rows) if rows is not None else list(self._phrase_rows())
            self._state = JobState(
                running=True,
                engine=engine.id,
                voice=voice.id,
                total=len(selected),
                started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            thread = threading.Thread(
                target=self._run,
                args=(engine, voice, selected),
                name="cpsa-tts-pregenerate",
                daemon=True,
            )
            self._thread = thread
            # The snapshot of the start, so a tiny fill that finishes at once
            # still answers `running: true` with the fields of the request.
            started = self._state.to_dict()
        thread.start()
        return started

    def wait(self, timeout_s: float = 30.0) -> dict[str, object]:
        """Block until the current fill ends (the CLI and the tests use it)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        return self.status()

    def _phrase_rows(self) -> Sequence[Phrase]:
        if self.phrases is not None:
            return self.phrases()
        return collect_for_cache(repo_root()).phrases

    # ------------------------------------------------------------------ worker

    def _run(self, engine: Engine, voice: Voice, rows: Sequence[Phrase]) -> None:
        began = time.monotonic()
        try:
            engine.load()
            speed = engine_speed(engine)
            for phrase in rows:
                text = normalize(phrase.text, engine.id)
                if not text or self.cache.has(engine.id, voice.id, text, speed):
                    self._bump(cached=True)
                    continue
                try:
                    samples = engine.synthesize(text, voice.id)
                except EngineError as error:
                    # An offline engine would fail on every sentence: stop at the
                    # first one and say why; the cache keeps what already exists.
                    with self._lock:
                        self._state.error = str(error)
                        self._state.failed = self._state.total - self._state.done
                        self._state.done = self._state.total
                    log.warning("pregenerate %s/%s: %s", engine.id, voice.id, error)
                    return
                self.cache.write(engine.id, voice.id, text, samples, engine.sample_rate, speed)
                self._bump(synthesized=True)
        finally:
            with self._lock:
                self._state.running = False
                self._state.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
                self._state.elapsed_s = round(time.monotonic() - began, 1)
                state = self._state.to_dict()
            log.info(
                "pregenerate %s/%s: %s/%s new, %s cached, %s failed in %ss%s",
                engine.id,
                voice.id,
                state["synthesized"],
                state["total"],
                state["cached"],
                state["failed"],
                state["elapsed_s"],
                f" ({state['error']})" if state["error"] else "",
            )

    def _bump(self, *, cached: bool = False, synthesized: bool = False) -> None:
        with self._lock:
            self._state.done += 1
            if cached:
                self._state.cached += 1
            if synthesized:
                self._state.synthesized += 1
