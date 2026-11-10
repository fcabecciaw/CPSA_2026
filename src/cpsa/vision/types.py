"""Shared data contracts of the 2D vision pipeline.

These types are the frozen interfaces between pose estimation, tracking,
feature extraction, the episode analysis and the adapters. Units:

* image coordinates are pixels, ``x`` right and ``y`` down;
* ``t`` is ``CLOCK_MONOTONIC`` seconds (the same clock as the IMU detections);
* lengths ending in ``_fw`` are face-widths, speeds ``_fwps`` face-widths per
  second, accelerations ``_fwps2`` face-widths per second squared.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import FaceRegion, HandShape, Side, StereotypyLabel, Tag


class FramingState(Enum):
    """How the selected subject is framed in one frame (FR-005)."""

    FULL_BODY = "full_body"
    HALF_BODY = "half_body"
    FACE_NOT_VISIBLE = "face_not_visible"
    HANDS_NOT_VISIBLE = "hands_not_visible"
    NO_PERSON = "no_person"


class KeypointStatus(IntEnum):
    """Per-keypoint tracker status (FR-007, FR-008)."""

    MEASURED = 0
    PREDICTED = 1
    LOST = 2


@dataclass(frozen=True)
class KeypointLayout:
    """Ordered keypoint set produced by a pose model.

    ``face`` lists every face anchor index; ``face_regions`` maps each face
    anchor to the face region it represents. ``left_hand`` and ``right_hand``
    hold the 21 hand keypoints in COCO-WholeBody order (0 wrist, 1-4 thumb,
    5-8 index, 9-12 middle, 13-16 ring, 17-20 little finger), or are empty when
    the model has no hand keypoints. ``edges`` is the skeleton used for drawing.
    """

    name: str
    names: tuple[str, ...]
    face: tuple[int, ...]
    face_regions: tuple[tuple[int, FaceRegion], ...]
    left_hand: tuple[int, ...]
    right_hand: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """Return the index of a keypoint name; raise ``KeyError`` if absent."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(name) from None

    def has(self, name: str) -> bool:
        return name in self.names

    def hand(self, side: Side) -> tuple[int, ...]:
        return self.left_hand if side is Side.LEFT else self.right_hand


@dataclass(frozen=True)
class PersonPose:
    """One detected person.

    ``box`` is ``(x1, y1, x2, y2)`` in pixels; ``xy`` is ``(K, 2)`` pixels and
    ``conf`` ``(K,)`` in ``[0, 1]``, both in the layout order.
    """

    box: NDArray[np.float64]
    score: float
    xy: NDArray[np.float64]
    conf: NDArray[np.float64]


@dataclass(frozen=True)
class PoseFrame:
    """Pose-model output for one camera frame."""

    seq: int
    t: float
    width: int
    height: int
    persons: tuple[PersonPose, ...]


@dataclass(frozen=True)
class VideoFrame:
    """One camera frame. ``decode`` returns ``(height, width, 3)`` RGB uint8.

    Decoding may be costly (for example YUYV conversion), so buffers keep the
    raw frame and only frames that are analyzed or saved are decoded.
    """

    seq: int
    t: float
    width: int
    height: int
    decode: Callable[[], NDArray[np.uint8]]


@dataclass(frozen=True)
class TrackedFrame:
    """Tracker output for the selected subject in one frame.

    Arrays are ``(K, 2)`` pixels (``xy``), pixels/s (``vel``), pixels/s^2
    (``acc``) and their one-sigma uncertainties (``xy_std``, ``vel_std``,
    ``acc_std``);
    ``status`` is ``(K,)`` int8 holding :class:`KeypointStatus` values and
    ``conf`` ``(K,)`` the last measured confidence (0 when never measured).
    ``face_width_px`` is ``None`` when the face width cannot be measured;
    ``foreshortening`` maps each side to observed/reference forearm length.
    """

    seq: int
    t: float
    framing: FramingState
    box: NDArray[np.float64] | None
    xy: NDArray[np.float64]
    vel: NDArray[np.float64]
    acc: NDArray[np.float64]
    xy_std: NDArray[np.float64]
    vel_std: NDArray[np.float64]
    acc_std: NDArray[np.float64]
    status: NDArray[np.int8]
    conf: NDArray[np.float64]
    face_width_px: float | None
    foreshortening: Mapping[Side, float | None]


@dataclass(frozen=True)
class HandFaceSample:
    """Hand-to-face kinematics for one hand in one frame (FR-012, FR-013)."""

    t: float
    side: Side
    distance_fw: float | None
    closing_speed_fwps: float | None
    closing_accel_fwps2: float | None
    approach_angle_deg: float | None
    time_to_contact_s: float | None
    occlusion_fraction: float
    nearest_region: FaceRegion | None


@dataclass(frozen=True)
class Contact:
    """One hand-to-face contact (FR-014)."""

    side: Side
    region: FaceRegion
    t_start: float
    duration_s: float
    min_distance_fw: float
    peak_closing_speed_fwps: float | None
    decel_fwps2: float | None
    hand_shape: HandShape
    head_recoil_fw: float | None
    head_recoil_speed_fwps: float | None


@dataclass(frozen=True)
class EpisodeReport:
    """Stage-2 description of a vision episode (FR-018, FR-019, FR-023).

    ``active`` is ``False`` once no hand-to-face or stereotypy activity has been
    seen for the configured activity window.
    """

    episode_id: str
    t_start: float
    t_end: float | None
    active: bool
    contacts: tuple[Contact, ...]
    contact_count: int
    repetition_hz: float | None
    periodicity: float | None
    interval_cv: float | None
    speed_trend_fwps_per_s: float | None
    peak_severity: Tag | None
    labels: frozenset[StereotypyLabel]
    clock_offset_s: float | None
    imu_consistency: float | None
