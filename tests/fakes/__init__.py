"""In-memory implementations of every port in ``cpsa.ports``.

Fakes that take a ``journal`` list append ``"<name>.<method>"`` entries to it so
tests can assert the relative order of lifecycle calls across components.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import ActuatorKind, DecisionRecord, Detection, Tag, Verdict
from cpsa.ports import SampleCallback

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeClock:
    """Manually advanced clock; ``now()`` tracks ``monotonic()`` from a fixed epoch."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def monotonic(self) -> float:
        return self.t

    def now(self) -> datetime:
        return EPOCH + timedelta(seconds=self.t)

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeExtractor:
    """Records every window it receives and returns ``features``."""

    def __init__(self, features: NDArray[np.float32] | None = None) -> None:
        self.features = features if features is not None else np.arange(18, dtype=np.float32)
        self.windows: list[NDArray[np.float32]] = []
        self.error: Exception | None = None

    def extract(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        self.windows.append(window.copy())
        if self.error is not None:
            raise self.error
        return self.features.copy()


class FakeClassifier:
    """Returns the scripted tags in order, then repeats the last one."""

    def __init__(self, tags: Iterable[Tag] = (Tag.DANGEROUS,)) -> None:
        self.tags = list(tags)
        self.calls: list[NDArray[np.float32]] = []
        self.error: Exception | None = None

    def classify(self, features: NDArray[np.float32]) -> Tag:
        self.calls.append(features.copy())
        if self.error is not None:
            raise self.error
        return self.tags.pop(0) if len(self.tags) > 1 else self.tags[0]


class ScriptedVerifier:
    """Returns ``verdict``, or delegates to ``behavior`` (which may block or raise)."""

    def __init__(
        self,
        verdict: Verdict = Verdict.UNDECIDED,
        behavior: Callable[[Detection], Verdict] | None = None,
    ) -> None:
        self.verdict = verdict
        self.behavior = behavior
        self.calls: list[Detection] = []

    def verify(self, detection: Detection) -> Verdict:
        self.calls.append(detection)
        if self.behavior is not None:
            return self.behavior(detection)
        return self.verdict


class RecordingActuator:
    def __init__(
        self,
        actuator_id: str,
        kind: ActuatorKind = ActuatorKind.METAMOTION,
        journal: list[str] | None = None,
    ) -> None:
        self.id = actuator_id
        self.kind = kind
        self.executed: list[Mapping[str, object]] = []
        self.stopped = False
        self.journal = journal

    def execute(self, params: Mapping[str, object]) -> None:
        self.executed.append(dict(params))

    def stop(self) -> None:
        self.stopped = True
        if self.journal is not None:
            self.journal.append(f"actuator:{self.id}.stop")


class FakeSensorSource:
    """``discover`` returns the scripted name sets in order, then repeats the last."""

    def __init__(
        self, discoveries: Iterable[set[str]] = (), journal: list[str] | None = None
    ) -> None:
        self.discoveries = list(discoveries) or [set()]
        self.discover_calls = 0
        self.on_sample: SampleCallback | None = None
        self.stopped = False
        self.journal = journal

    def discover(self, timeout_s: float) -> set[str]:
        self.discover_calls += 1
        if len(self.discoveries) > 1:
            return self.discoveries.pop(0)
        return set(self.discoveries[0])

    def start(self, on_sample: SampleCallback) -> None:
        self.on_sample = on_sample
        if self.journal is not None:
            self.journal.append("sensors.start")

    def stop(self) -> None:
        self.stopped = True
        if self.journal is not None:
            self.journal.append("sensors.stop")


class MemorySink:
    def __init__(self) -> None:
        self.records: list[DecisionRecord] = []
        self._lock = threading.Lock()

    def write(self, record: DecisionRecord) -> None:
        with self._lock:
            self.records.append(record)


class RecordingGui:
    def __init__(self, journal: list[str] | None = None) -> None:
        self.records: list[DecisionRecord] = []
        self.started = False
        self.stopped = False
        self.journal = journal

    def start(self) -> None:
        self.started = True
        if self.journal is not None:
            self.journal.append("gui.start")

    def publish(self, record: DecisionRecord) -> None:
        self.records.append(record)

    def stop(self) -> None:
        self.stopped = True
        if self.journal is not None:
            self.journal.append("gui.stop")
