"""Hand-to-face contact detection in 2D face-width units (FR-014, AC-007)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import HandShape, Side
from cpsa.vision.handface import FeatureConfig, hand_face_samples
from cpsa.vision.handshape import classify_hand_shape
from cpsa.vision.types import Contact, HandFaceSample, KeypointLayout, KeypointStatus, TrackedFrame

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from cpsa.vision.assessment import AnalysisConfig

_EPS = 1e-9
_LOST = int(KeypointStatus.LOST)
_SHAPES = (HandShape.FIST, HandShape.POINT_PINCH, HandShape.OPEN_PALM, HandShape.UNKNOWN)


@dataclass(frozen=True)
class ContactConfig2D:
    """Contact hysteresis, duration and recoil thresholds (AC-007).

    A contact starts when the hand is within ``contact_enter_fw`` of the face or
    the face is occluded (``occlusion_contact``) with the hand within
    ``occlusion_near_fw``, and continues while the hand stays within the larger
    ``contact_exit_fw``. Shorter than ``min_duration_s`` touches are discarded.
    ``approach_window_s`` sets the peak-closing-speed lookback and
    ``recoil_window_s`` the post-onset head-displacement window.
    """

    contact_enter_fw: float = 0.15
    contact_exit_fw: float = 0.25
    occlusion_contact: float = 0.6
    occlusion_near_fw: float = 0.5
    min_duration_s: float = 0.1
    approach_window_s: float = 0.3
    recoil_window_s: float = 0.3


def _in_contact(sample: HandFaceSample, cfg: ContactConfig2D) -> bool:
    """Return whether a sample starts a contact."""
    if sample.distance_fw is None:
        return False
    if sample.distance_fw <= cfg.contact_enter_fw:
        return True
    return (
        sample.occlusion_fraction >= cfg.occlusion_contact
        and sample.distance_fw <= cfg.occlusion_near_fw
    )


def _holds_contact(sample: HandFaceSample, cfg: ContactConfig2D) -> bool:
    """Return whether a sample keeps an open contact alive (hysteresis)."""
    if sample.distance_fw is None:
        return False
    if sample.distance_fw <= cfg.contact_exit_fw:
        return True
    return (
        sample.occlusion_fraction >= cfg.occlusion_contact
        and sample.distance_fw <= cfg.occlusion_near_fw
    )


def _peak_closing(
    samples: Sequence[HandFaceSample],
    t_start: float,
    window_s: float,
) -> float | None:
    """Return the peak closing speed in the approach window before onset."""
    values = [
        sample.closing_speed_fwps
        for sample in samples
        if sample.closing_speed_fwps is not None and t_start - window_s <= sample.t <= t_start
    ]
    if not values:
        return None
    return max(0.0, max(float(value) for value in values))


def _deceleration(
    samples: Sequence[HandFaceSample],
    t_start: float,
    t_end: float,
    window_s: float,
) -> float | None:
    """Return the closing-speed drop over the onset frames, in fw/s^2.

    The reference peak is the fastest closing sample in the approach window
    before onset; the drop is measured to the slowest closing sample with a
    later timestamp inside the contact. Missing or non-positive drops give
    ``None``.
    """
    approach = [
        sample
        for sample in samples
        if sample.closing_speed_fwps is not None and t_start - window_s <= sample.t <= t_start
    ]
    if not approach:
        return None
    peak_sample = max(approach, key=lambda sample: sample.closing_speed_fwps or 0.0)
    peak = float(peak_sample.closing_speed_fwps or 0.0)
    if peak <= 0.0:
        return None
    after = [
        sample
        for sample in samples
        if sample.closing_speed_fwps is not None and peak_sample.t < sample.t <= t_end
    ]
    if not after:
        return None
    slowest = min(after, key=lambda sample: sample.closing_speed_fwps or 0.0)
    drop = peak - float(slowest.closing_speed_fwps or 0.0)
    elapsed = float(slowest.t) - float(peak_sample.t)
    if drop <= 0.0 or elapsed <= 0.0:
        return None
    return drop / elapsed


def _majority_shape(
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    side: Side,
    t_start: float,
    t_end: float,
    cfg: AnalysisConfig,
) -> HandShape:
    """Return the most frequent hand shape over the contact frames."""
    hand = np.asarray(layout.hand(side), dtype=np.int64)
    counts = dict.fromkeys(_SHAPES, 0)
    if hand.size == 21:
        for frame in frames:
            if not t_start <= frame.t <= t_end:
                continue
            points = np.asarray(frame.xy, dtype=np.float64)[hand]
            confidence = np.asarray(frame.conf, dtype=np.float64)[hand]
            shape = classify_hand_shape(points, confidence, cfg.hand_shape)
            counts[shape] += 1
    return max(_SHAPES, key=lambda shape: (counts[shape], -_SHAPES.index(shape)))


def _face_centroids(
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    cfg: FeatureConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(t, centroid, face_width_px)`` for frames with a visible face."""
    face = np.asarray(layout.face, dtype=np.int64)
    times: list[float] = []
    centroids: list[NDArray[np.float64]] = []
    widths: list[float] = []
    for frame in frames:
        width = frame.face_width_px
        if width is None or not np.isfinite(width) or width <= 0.0:
            continue
        status = np.asarray(frame.status, dtype=np.int8)[face]
        conf = np.asarray(frame.conf, dtype=np.float64)[face]
        xy = np.asarray(frame.xy, dtype=np.float64)[face]
        visible = (status != _LOST) & (conf >= cfg.face_conf) & np.all(np.isfinite(xy), axis=1)
        if not bool(np.any(visible)):
            continue
        times.append(float(frame.t))
        centroids.append(np.mean(xy[visible], axis=0))
        widths.append(float(width))
    if not times:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(centroids, dtype=np.float64),
        np.asarray(widths, dtype=np.float64),
    )


def _head_recoil(
    centroids: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    t_start: float,
    window_s: float,
) -> tuple[float | None, float | None]:
    """Return the largest head displacement and speed after onset, in fw units."""
    times, points, widths = centroids
    if times.size == 0 or t_start not in times:
        return None, None
    onset = int(np.flatnonzero(times == t_start)[0])
    selected = np.flatnonzero((times >= t_start) & (times <= t_start + window_s))
    if selected.size == 0:
        return None, None
    reference = points[onset]
    width = float(widths[onset])
    displacement = np.linalg.norm(points[selected] - reference, axis=1) / width
    recoil = float(np.max(displacement))
    speed: float | None = None
    if selected.size >= 2:
        deltas = np.linalg.norm(np.diff(points[selected], axis=0), axis=1)
        steps = np.diff(times[selected])
        valid = steps > _EPS
        if bool(np.any(valid)):
            speed = float(np.max(deltas[valid] / steps[valid])) / width
    return recoil, speed


def _make_contact(
    side: Side,
    contact_samples: Sequence[HandFaceSample],
    all_samples: Sequence[HandFaceSample],
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    centroids: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    cfg: AnalysisConfig,
) -> Contact | None:
    """Build one :class:`Contact` from its frames, or ``None`` when unusable."""
    t_start = float(contact_samples[0].t)
    t_end = float(contact_samples[-1].t)
    distances = [sample for sample in contact_samples if sample.distance_fw is not None]
    if not distances:
        return None
    closest = min(distances, key=lambda sample: sample.distance_fw or 0.0)
    region = closest.nearest_region
    if region is None:
        region = next(
            (
                sample.nearest_region
                for sample in contact_samples
                if sample.nearest_region is not None
            ),
            None,
        )
    if region is None:
        return None
    recoil, recoil_speed = _head_recoil(centroids, t_start, cfg.contacts.recoil_window_s)
    min_distance = closest.distance_fw
    if min_distance is None:  # pragma: no cover - guarded by the distances filter
        return None
    return Contact(
        side=side,
        region=region,
        t_start=t_start,
        duration_s=t_end - t_start,
        min_distance_fw=float(min_distance),
        peak_closing_speed_fwps=_peak_closing(all_samples, t_start, cfg.contacts.approach_window_s),
        decel_fwps2=_deceleration(all_samples, t_start, t_end, cfg.contacts.approach_window_s),
        hand_shape=_majority_shape(frames, layout, side, t_start, t_end, cfg),
        head_recoil_fw=recoil,
        head_recoil_speed_fwps=recoil_speed,
    )


def detect_contacts(
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
) -> list[Contact]:
    """Detect hand-to-face contacts with distance hysteresis (FR-014).

    Each side is evaluated independently on the hand-to-face samples of
    :func:`cpsa.vision.handface.hand_face_samples`. A contact is a maximal run
    of frames inside the hysteresis band (or of an occluded face with the hand
    near it) that lasts at least ``cfg.contacts.min_duration_s``. The contact
    reports the nearest region at its closest frame, the smallest distance, the
    peak closing speed of the approach window, the closing-speed drop at onset,
    the majority hand shape and the largest post-onset head displacement. The
    result is sorted by start time, then side and region.
    """
    contact_cfg = cfg.contacts
    if contact_cfg.contact_enter_fw <= 0.0:
        raise ValueError("contact_enter_fw must be positive")
    if contact_cfg.contact_exit_fw < contact_cfg.contact_enter_fw:
        raise ValueError("contact_exit_fw must not be smaller than contact_enter_fw")
    if not 0.0 <= contact_cfg.occlusion_contact <= 1.0:
        raise ValueError("occlusion_contact must be in [0, 1]")
    if contact_cfg.min_duration_s < 0.0:
        raise ValueError("min_duration_s must be non-negative")
    if contact_cfg.approach_window_s < 0.0 or contact_cfg.recoil_window_s < 0.0:
        raise ValueError("windows must be non-negative")

    ordered = list(frames)
    if len(ordered) >= 2:
        deltas = np.diff([frame.t for frame in ordered])
        if bool(np.any(deltas <= 0.0)):
            raise ValueError("frame timestamps must be strictly increasing")
    if not ordered:
        return []

    by_side: dict[Side, list[HandFaceSample]] = {Side.LEFT: [], Side.RIGHT: []}
    for frame in ordered:
        for sample in hand_face_samples(frame, layout, cfg.features):
            by_side[sample.side].append(sample)
    centroids = _face_centroids(ordered, layout, cfg.features)

    contacts: list[Contact] = []
    for side in (Side.LEFT, Side.RIGHT):
        samples = by_side[side]
        start: int | None = None
        for index, sample in enumerate(samples):
            if start is None:
                if _in_contact(sample, contact_cfg):
                    start = index
                continue
            if _holds_contact(sample, contact_cfg):
                continue
            contact = _make_contact(
                side,
                samples[start:index],
                samples,
                ordered,
                layout,
                centroids,
                cfg,
            )
            if contact is not None and contact.duration_s + _EPS >= contact_cfg.min_duration_s:
                contacts.append(contact)
            start = None
        if start is not None:
            contact = _make_contact(
                side,
                samples[start:],
                samples,
                ordered,
                layout,
                centroids,
                cfg,
            )
            if contact is not None and contact.duration_s + _EPS >= contact_cfg.min_duration_s:
                contacts.append(contact)

    contacts.sort(key=lambda contact: (contact.t_start, contact.side.value, contact.region.value))
    return contacts
