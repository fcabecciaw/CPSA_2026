"""Contracts between the application layer and its adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

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
    VisionAssessment,
)
from cpsa.vision.types import EpisodeReport, PersonPose, TrackedFrame, VideoFrame

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
    def verify(self, detection: Detection) -> VisionAssessment | Verdict:
        """Return an assessment (or a bare verdict, treated as having no severity).

        May block or raise; the caller enforces deadline and fail-open.
        """
        ...


@runtime_checkable
class DetectionObserver(Protocol):
    def observe(self, detection: Detection) -> None:
        """Receive every stereotypy detection, including those inside the cooldown.

        Must return quickly and never raise into the caller. A verifier may also
        implement this protocol to extend its episode on detections that are not
        sent to ``verify``.
        """
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


@dataclass(frozen=True)
class RgbdFrame:
    """One paired RGB/depth frame; both timestamps use ``CLOCK_MONOTONIC`` seconds."""

    seq: int
    rgb: NDArray[np.uint8]  # (height, width, 3), RGB order
    depth: NDArray[np.uint16]  # (height, width), raw Z16, 0 = invalid
    ts_rgb: float
    ts_depth: float


@runtime_checkable
class FrameSource(Protocol):
    def start(self, on_frame: Callable[[RgbdFrame], None]) -> None:
        """Open the device and deliver paired frames from one delivery thread."""
        ...

    def stop(self) -> None: ...


@runtime_checkable
class VideoSource(Protocol):
    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Open the camera and deliver frames, in order, from one delivery thread.

        A failure after ``start`` returns (device lost, read error) is reported
        once through ``on_error`` and stops delivery; it is never raised on the
        delivery thread.
        """
        ...

    def stop(self) -> None: ...


@runtime_checkable
class PoseEstimator(Protocol):
    def estimate(self, rgb: NDArray[np.uint8]) -> Sequence[PersonPose]:
        """Return the persons in an ``(H, W, 3)`` RGB image, coordinates in its pixels.

        Raises on a model or accelerator failure.
        """
        ...


@runtime_checkable
class EvidenceSink(Protocol):
    def write_episode(
        self,
        episode_id: str,
        frames: Sequence[tuple[NDArray[np.uint8], TrackedFrame | None]],
        fps: float,
        keyframe: int,
    ) -> None:
        """Save a clip with a skeleton overlay and the keyframe ``frames[keyframe]``."""
        ...

    def purge(self, now: datetime) -> None:
        """Remove evidence older than the configured retention."""
        ...


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


@runtime_checkable
class Publisher(Protocol):
    """Pushes protocol messages to every connected display client.

    Callers pass only the message ``type`` and its ``data`` payload; the server adds
    the envelope (``v``, ``seq``, ``t``, ``wall``) per connection (interface IF-102).
    Both methods are thread-safe and never block the caller.
    """

    def state(self, type: str, data: Mapping[str, Any]) -> None:
        """Replace the state of topic ``type``; slow clients get only the latest one."""
        ...

    def event(self, type: str, data: Mapping[str, Any]) -> None:
        """Queue a discrete event; events are delivered in order."""
        ...


@runtime_checkable
class VisionObserver(Protocol):
    """Receives live vision results, for example to drive the medical dashboard.

    Methods are called from the vision verifier's worker thread. Implementations must
    return quickly and must not raise; the verifier logs and swallows any exception,
    so an observer never changes a verdict.
    """

    def on_episode(self, episode_id: str, t_start: float, active: bool) -> None:
        """An episode started (``active=True``) or ended (``active=False``)."""
        ...

    def on_tracked(self, tracked: TrackedFrame) -> None:
        """One analyzed, tracked frame of the active episode."""
        ...

    def on_report(self, report: EpisodeReport) -> None:
        """The final report of an episode that just ended."""
        ...
