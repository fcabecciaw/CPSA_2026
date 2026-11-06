"""Domain types shared by every layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum


class Tag(IntEnum):
    """Stereotypy class produced by the IMU classifier."""

    NO_CLASS = 0
    NON_DANGEROUS = 1
    DANGEROUS = 2
    NON_STEREOTIPY = 3

    @property
    def is_stereotypy(self) -> bool:
        return self in (Tag.NON_DANGEROUS, Tag.DANGEROUS)


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


class SampleKind(Enum):
    ACC = "acc"
    GYR = "gyr"
    QUAT = "quat"


class Verdict(Enum):
    """Outcome of the vision verifier for one detection."""

    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    UNDECIDED = "UNDECIDED"


class GateReason(Enum):
    """Why the verification gate allowed or blocked an actuation."""

    CONFIRMED = "CONFIRMED"
    UNDECIDED = "UNDECIDED"
    TIMEOUT = "TIMEOUT"
    VERIFIER_ERROR = "VERIFIER_ERROR"
    REJECTED = "REJECTED"


class ActuatorKind(Enum):
    METAMOTION = "metamotion"
    SPEAKER = "speaker"
    LED_STRIP = "led_strip"


# Actuator ids are "<prefix><device address>"; the prefix identifies the kind.
ACTUATOR_ID_PREFIX: Mapping[ActuatorKind, str] = {
    ActuatorKind.METAMOTION: "meta_",
    ActuatorKind.SPEAKER: "speaker_",
    ActuatorKind.LED_STRIP: "led_",
}


def actuator_kind_of(actuator_id: str) -> ActuatorKind | None:
    """Return the kind encoded in an actuator id, or ``None`` for an unknown prefix."""
    for kind, prefix in ACTUATOR_ID_PREFIX.items():
        if actuator_id.startswith(prefix):
            return kind
    return None


class ClassificationError(Exception):
    """The classifier produced an output that is not a known :class:`Tag`."""


@dataclass(frozen=True)
class Detection:
    """One classified IMU window."""

    tag: Tag
    timestamp: datetime
    window_end_ts: float
    features: tuple[float, ...]


@dataclass(frozen=True)
class Actuation:
    """A command for one actuator."""

    actuator_id: str
    params: Mapping[str, object]


@dataclass(frozen=True)
class Decision:
    """Result of evaluating one detection through the verification gate."""

    verdict: Verdict
    reason: GateReason
    actuated: bool
    actuations: tuple[Actuation, ...]


@dataclass(frozen=True)
class DecisionRecord:
    """Event-log entry: a tag change, an actuation decision, or both."""

    timestamp: datetime
    window_end_ts: float
    tag: Tag
    tag_changed: bool
    decision: Decision | None
