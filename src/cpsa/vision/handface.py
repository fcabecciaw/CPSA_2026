"""Hand-to-face kinematics in face-width units (FR-012, FR-013).

Every length is a multiple of the tracked face width and every speed is a
face-width per second, so the features do not change when the subject sits
closer to or farther from the camera (AC-006).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import FaceRegion, Side
from cpsa.vision.types import HandFaceSample, KeypointLayout, KeypointStatus, TrackedFrame

_EPS = 1e-9
_LOST = int(KeypointStatus.LOST)


@dataclass(frozen=True)
class FeatureConfig:
    """Thresholds and tolerances of the hand-to-face features.

    ``hand_conf`` accepts a tracked keypoint as part of the hand point,
    ``face_conf`` accepts it as a face anchor, and ``occlusion_conf`` marks a
    face anchor as occluded when its last confidence is below it.
    ``forehead_margin_fw`` is how far above the brow line a hand point must be
    to count as a forehead or temple touch.
    """

    hand_conf: float = 0.3
    face_conf: float = 0.3
    occlusion_conf: float = 0.3
    forehead_margin_fw: float = 0.1


def _visible(
    frame: TrackedFrame,
    indices: NDArray[np.int64],
    conf_thr: float,
) -> NDArray[np.bool_]:
    """Return which of ``indices`` are tracked, confident and finite."""
    if indices.size == 0:
        empty: NDArray[np.bool_] = np.zeros(0, dtype=np.bool_)
        return empty
    status = np.asarray(frame.status, dtype=np.int8)[indices]
    conf = np.asarray(frame.conf, dtype=np.float64)[indices]
    xy = np.asarray(frame.xy, dtype=np.float64)[indices]
    finite = np.asarray(np.all(np.isfinite(xy), axis=1), dtype=np.bool_)
    result: NDArray[np.bool_] = (status != _LOST) & (conf >= conf_thr) & finite
    return result


def _wrist_index(layout: KeypointLayout, side: Side) -> int | None:
    name = f"{side.value}_wrist"
    return layout.index(name) if layout.has(name) else None


def _brow_line(
    frame: TrackedFrame,
    layout: KeypointLayout,
    cfg: FeatureConfig,
) -> float | None:
    """Return the mean ``y`` of the two brow anchors when both are visible."""
    names = ("face_brow_right", "face_brow_left")
    if not all(layout.has(name) for name in names):
        return None
    indices = np.asarray([layout.index(name) for name in names], dtype=np.int64)
    if not bool(np.all(_visible(frame, indices, cfg.face_conf))):
        return None
    return float(np.mean(np.asarray(frame.xy, dtype=np.float64)[indices, 1]))


def _hand_point(
    frame: TrackedFrame,
    layout: KeypointLayout,
    side: Side,
    cfg: FeatureConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    """Return the hand position, velocity and acceleration.

    The position is the centroid of the confident hand keypoints, or the wrist
    when no hand keypoint is confident (FR-013). ``None`` means the hand is not
    visible at all.
    """
    hand = np.asarray(layout.hand(side), dtype=np.int64)
    xy = np.asarray(frame.xy, dtype=np.float64)
    vel = np.asarray(frame.vel, dtype=np.float64)
    acc = np.asarray(frame.acc, dtype=np.float64)
    if hand.size:
        visible = _visible(frame, hand, cfg.hand_conf)
        if bool(np.any(visible)):
            selected = hand[visible]
            return (
                np.mean(xy[selected], axis=0),
                np.mean(vel[selected], axis=0),
                np.mean(acc[selected], axis=0),
            )
    wrist = _wrist_index(layout, side)
    if wrist is None:
        return None
    if not bool(_visible(frame, np.asarray([wrist], dtype=np.int64), cfg.hand_conf)[0]):
        return None
    return xy[wrist], vel[wrist], acc[wrist]


def _region_of(
    layout: KeypointLayout,
    anchor: int,
    regions: Mapping[int, FaceRegion],
) -> FaceRegion | None:
    """Map one anchor index to its face region, if the layout declares one."""
    return regions.get(anchor)


def hand_face_samples(
    frame: TrackedFrame,
    layout: KeypointLayout,
    cfg: FeatureConfig,
) -> tuple[HandFaceSample, ...]:
    """Compute one :class:`HandFaceSample` per visible hand (FR-013).

    The hand point is the centroid of the confident hand keypoints, else the
    wrist. The face reference point is the centroid of the confident face
    anchors; the distance is measured to the *nearest* confident anchor, and the
    closing speed and acceleration are the hand velocity and acceleration
    projected on the hand-to-face direction (positive while the hand closes).
    The approach angle is the angle between the hand velocity and that direction,
    and the time to contact is ``distance / closing speed`` while closing.

    ``occlusion_fraction`` is the share of layout face anchors that are not
    ``MEASURED`` or sit below ``cfg.occlusion_conf``. The nearest region is the
    region of the nearest anchor; a hand point more than
    ``cfg.forehead_margin_fw`` above the brow line is reported as
    ``FOREHEAD_TEMPLE`` instead.

    Fields that need a face width (``distance_fw``, both kinematics, the time to
    contact) are ``None`` when ``frame.face_width_px`` is unknown; the angle and
    the region do not need a scale and stay available. The tuple is ordered
    left, right.
    """
    xy = np.asarray(frame.xy, dtype=np.float64)
    face = np.asarray(layout.face, dtype=np.int64)
    regions = dict(layout.face_regions)
    face_visible = _visible(frame, face, cfg.face_conf)
    face_points = xy[face[face_visible]]
    face_centroid = np.mean(face_points, axis=0) if face_points.size else None

    if face.size:
        occluded = ~_visible(frame, face, cfg.occlusion_conf)
        occlusion = float(np.count_nonzero(occluded)) / float(face.size)
    else:
        occlusion = 1.0

    brow_y = _brow_line(frame, layout, cfg)
    width = frame.face_width_px
    face_width = float(width) if width is not None and np.isfinite(width) and width > 0.0 else None

    samples: list[HandFaceSample] = []
    for side in (Side.LEFT, Side.RIGHT):
        hand = _hand_point(frame, layout, side, cfg)
        if hand is None:
            continue
        point, velocity, acceleration = hand
        if face_centroid is None:
            samples.append(
                HandFaceSample(
                    t=frame.t,
                    side=side,
                    distance_fw=None,
                    closing_speed_fwps=None,
                    closing_accel_fwps2=None,
                    approach_angle_deg=None,
                    time_to_contact_s=None,
                    occlusion_fraction=occlusion,
                    nearest_region=None,
                )
            )
            continue

        offset = face_centroid - point
        norm = float(np.linalg.norm(offset))
        unit = offset / norm if norm > _EPS else np.zeros(2, dtype=np.float64)

        nearest = int(np.argmin(np.linalg.norm(face_points - point, axis=1)))
        nearest_index = int(face[face_visible][nearest])
        region = _region_of(layout, nearest_index, regions)
        if (
            region is not None
            and face_width is not None
            and brow_y is not None
            and point[1] < brow_y - cfg.forehead_margin_fw * face_width
        ):
            region = FaceRegion.FOREHEAD_TEMPLE

        speed_norm = float(np.linalg.norm(velocity))
        if speed_norm > _EPS and norm > _EPS:
            cosine = float(np.dot(velocity, unit)) / speed_norm
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        else:
            angle = None

        if face_width is None:
            distance = None
            closing = None
            closing_accel = None
            time_to_contact = None
        else:
            offsets: NDArray[np.float64] = np.asarray(
                np.linalg.norm(face_points - point, axis=1), dtype=np.float64
            )
            distance = float(np.min(offsets)) / face_width
            closing = float(np.dot(velocity, unit)) / face_width
            closing_accel = float(np.dot(acceleration, unit)) / face_width
            time_to_contact = distance / closing if closing > _EPS else None

        samples.append(
            HandFaceSample(
                t=frame.t,
                side=side,
                distance_fw=distance,
                closing_speed_fwps=closing,
                closing_accel_fwps2=closing_accel,
                approach_angle_deg=angle,
                time_to_contact_s=time_to_contact,
                occlusion_fraction=occlusion,
                nearest_region=region,
            )
        )
    return tuple(samples)
