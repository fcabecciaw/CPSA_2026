"""Contracts between the application layer and its adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import (
    ActuatorKind,
    DecisionRecord,
    Detection,
    SampleKind,
    Side,
    Tag,
    Verdict,
)

SampleCallback = Callable[[Side, SampleKind, tuple[float, ...], float], None]
"""Receives ``(side, kind, values, monotonic_ts)`` for every IMU sample."""


@runtime_checkable
class FeatureExtractor(Protocol):
    def extract(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        """Map a preprocessed ``(150, 20)`` window to an ``(18,)`` feature vector.

        Column order: R acc xyz, R gyr xyz, R quat xyzw, L acc xyz, L gyr xyz, L quat xyzw.
        """
        ...


@runtime_checkable
class Classifier(Protocol):
    def classify(self, features: NDArray[np.float32]) -> Tag:
        """Classify an ``(18,)`` feature vector; raise ``ClassificationError`` on unknown output."""
        ...


@runtime_checkable
class Verifier(Protocol):
    def verify(self, detection: Detection) -> Verdict:
        """Return a verdict. May block or raise; the caller enforces deadline and fail-open."""
        ...


@runtime_checkable
class Actuator(Protocol):
    id: str
    kind: ActuatorKind

    def execute(self, params: Mapping[str, object]) -> None:
        """Request an actuation. Never raises for a disconnected device: logs and returns."""
        ...

    def stop(self) -> None: ...


@runtime_checkable
class SensorSource(Protocol):
    def discover(self, timeout_s: float) -> set[str]:
        """Scan for devices and return the names found."""
        ...

    def start(self, on_sample: SampleCallback) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class EventSink(Protocol):
    def write(self, record: DecisionRecord) -> None: ...


@runtime_checkable
class Gui(Protocol):
    def start(self) -> None: ...

    def publish(self, record: DecisionRecord) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class Clock(Protocol):
    def monotonic(self) -> float: ...

    def now(self) -> datetime: ...
