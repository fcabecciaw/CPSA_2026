"""Domain types shared by every layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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


class FaceRegion(Enum):
    """Face region touched by a hand, mapped from the face-anchor keypoints."""

    EYES = "eyes"
    MOUTH = "mouth"
    NOSE_CHEEK = "nose_cheek"
    FOREHEAD_TEMPLE = "forehead_temple"
    EAR = "ear"
    HEAD_TOP = "head_top"


class HandShape(Enum):
    """Hand configuration at a hand-to-face contact."""

    FIST = "fist"
    POINT_PINCH = "point_pinch"
    OPEN_PALM = "open_palm"
    UNKNOWN = "unknown"


class StereotypyLabel(Enum):
    """Stereotypy type observed by the vision pipeline during an episode."""

    HAND_TO_FACE = "hand_to_face"
    ROCKING = "rocking"
    HAND_FLAPPING = "hand_flapping"


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
class VisionAssessment:
    """Result of the vision verifier for one detection.

    ``severity`` is the vision grade (``NON_DANGEROUS`` or ``DANGEROUS``) and is
    meaningful only with ``Verdict.CONFIRMED``. ``features`` holds the stage-1
    numbers behind the verdict, keyed by feature name. ``episode_id`` links the
    decision to the vision episode report and evidence.
    """

    verdict: Verdict
    severity: Tag | None = None
    features: Mapping[str, float] = field(default_factory=dict)
    episode_id: str | None = None


@dataclass(frozen=True)
class Decision:
    """Result of evaluating one detection through the verification gate.

    ``severity`` is the tag that drove the actuation policy (the vision severity
    for a confirmed assessment, else the IMU tag); ``None`` when nothing actuated.
    """

    verdict: Verdict
    reason: GateReason
    actuated: bool
    actuations: tuple[Actuation, ...]
    severity: Tag | None = None
    assessment: VisionAssessment | None = None


@dataclass(frozen=True)
class DecisionRecord:
    """Event-log entry: a tag change, an actuation decision, or both."""

    timestamp: datetime
    window_end_ts: float
    tag: Tag
    tag_changed: bool
    decision: Decision | None


@dataclass(frozen=True)
class ContactEvent:
    """One hand-to-face contact detected by the IMU-vision fusion."""

    t: float
    side: Side
    region: FaceRegion
    min_distance_m: float
    approach_speed_mps: float
    impact_peak_g: float | None
    occluder: bool
