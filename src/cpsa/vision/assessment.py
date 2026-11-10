"""Stage-1 verdict and stage-2 episode report (FR-015..FR-019, FR-023).

``assess_fast`` is the pure stage-1 decision used by the verifier: it consumes
tracked frames and returns a :class:`VisionAssessment` with a verdict, a
severity whenever the verdict is ``CONFIRMED`` (FR-016, P-02) and the feature
set that drove the decision. ``build_report`` is the stage-2 episode report.

Feature codes used in ``VisionAssessment.features``:

* ``region_code``: 0 = none, 1 = eyes, 2 = mouth, 3 = nose/cheek,
  4 = forehead/temple, 5 = ear, 6 = head top.
* ``hand_shape_code``: 0 = unknown, 1 = fist, 2 = point/pinch, 3 = open palm.

A feature that is unavailable (for example the deceleration of an approach that
never touched the face) is reported as ``-1.0``; ``coverage_ok`` is 1 or 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import (
    FaceRegion,
    HandShape,
    Side,
    StereotypyLabel,
    Tag,
    Verdict,
    VisionAssessment,
)
from cpsa.vision.contacts2d import ContactConfig2D, detect_contacts
from cpsa.vision.fusion import IMUSignal, estimate_clock_offset
from cpsa.vision.handface import FeatureConfig, hand_face_samples
from cpsa.vision.handshape import HandShapeConfig, classify_hand_shape
from cpsa.vision.oscillation import (
    OscillationConfig,
    detect_flapping,
    detect_rocking,
    periodicity,
)
from cpsa.vision.types import (
    Contact,
    EpisodeReport,
    FramingState,
    HandFaceSample,
    KeypointLayout,
    KeypointStatus,
    TrackedFrame,
)

_EPS = 1e-9
_REGION_CODES: Mapping[FaceRegion, float] = {
    FaceRegion.EYES: 1.0,
    FaceRegion.MOUTH: 2.0,
    FaceRegion.NOSE_CHEEK: 3.0,
    FaceRegion.FOREHEAD_TEMPLE: 4.0,
    FaceRegion.EAR: 5.0,
    FaceRegion.HEAD_TOP: 6.0,
}
_SHAPE_CODES: Mapping[HandShape, float] = {
    HandShape.UNKNOWN: 0.0,
    HandShape.FIST: 1.0,
    HandShape.POINT_PINCH: 2.0,
    HandShape.OPEN_PALM: 3.0,
}
_SEVERITY_RANK: Mapping[Tag, int] = {Tag.NON_DANGEROUS: 1, Tag.DANGEROUS: 2}
_LOST = int(KeypointStatus.LOST)


@dataclass(frozen=True)
class SeverityConfig:
    """Vision severity rules (FR-016).

    A contact is dangerous when its region is in ``dangerous_regions``; or its
    hand shape is in ``dangerous_shapes`` while the peak closing speed reaches
    ``shape_speed_fwps``; or the peak closing speed reaches
    ``strike_speed_fwps``; or the deceleration reaches ``strike_decel_fwps2``;
    or the head recoil reaches ``recoil_fw``.
    """

    dangerous_regions: frozenset[FaceRegion] = frozenset({FaceRegion.EYES})
    dangerous_shapes: frozenset[HandShape] = frozenset({HandShape.FIST, HandShape.POINT_PINCH})
    shape_speed_fwps: float = 1.0
    strike_speed_fwps: float = 2.0
    strike_decel_fwps2: float = 4.0
    recoil_fw: float = 0.05


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration of the features, contacts, labels and stage-1 verdict.

    ``after`` thresholds: ``approach_fw`` and ``approach_speed_fwps`` define the
    hand-to-face approach that confirms an episode without a contact;
    ``min_framed_fraction`` is the share of framed window frames required for a
    ``REJECTED`` verdict; ``max_gap_s`` is the largest allowed hole in the
    analyzed window (P-03); ``activity_window_s`` is how long an episode stays
    active after its last hand-to-face or stereotypy event; and
    ``repetition_band_hz`` bounds the contact-repetition search on the distance
    signal.
    """

    features: FeatureConfig = field(default_factory=FeatureConfig)
    contacts: ContactConfig2D = field(default_factory=ContactConfig2D)
    hand_shape: HandShapeConfig = field(default_factory=HandShapeConfig)
    oscillation: OscillationConfig = field(default_factory=OscillationConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    approach_fw: float = 0.4
    approach_speed_fwps: float = 1.0
    min_framed_fraction: float = 0.6
    max_gap_s: float = 0.2
    activity_window_s: float = 3.0
    repetition_band_hz: tuple[float, float] = (0.2, 5.0)


def severity_of(contact: Contact, cfg: AnalysisConfig) -> Tag:
    """Grade one contact as ``DANGEROUS`` or ``NON_DANGEROUS`` (FR-016, AC-009)."""
    rules = cfg.severity
    if contact.region in rules.dangerous_regions:
        return Tag.DANGEROUS
    peak = contact.peak_closing_speed_fwps
    if (
        contact.hand_shape in rules.dangerous_shapes
        and peak is not None
        and peak >= rules.shape_speed_fwps
    ):
        return Tag.DANGEROUS
    if peak is not None and peak >= rules.strike_speed_fwps:
        return Tag.DANGEROUS
    if contact.decel_fwps2 is not None and contact.decel_fwps2 >= rules.strike_decel_fwps2:
        return Tag.DANGEROUS
    if contact.head_recoil_fw is not None and contact.head_recoil_fw >= rules.recoil_fw:
        return Tag.DANGEROUS
    return Tag.NON_DANGEROUS


def _distance_key(sample: HandFaceSample) -> float:
    """Sort key that orders samples by distance, unmeasured samples last."""
    distance = sample.distance_fw
    return float("inf") if distance is None else float(distance)


def severity_of_approach(
    samples: Sequence[HandFaceSample],
    hand_shape: HandShape,
    cfg: AnalysisConfig,
) -> Tag:
    """Grade a confirmed approach that has no contact yet (FR-016, P-02).

    The applicable rules are the nearest region, the hand shape with its speed
    threshold and the peak closing speed. Deceleration and head recoil are
    missing before contact and count as not met.
    """
    rules = cfg.severity
    measured = [sample for sample in samples if sample.distance_fw is not None]
    region = min(measured, key=_distance_key).nearest_region if measured else None
    speeds = [
        float(sample.closing_speed_fwps)
        for sample in samples
        if sample.closing_speed_fwps is not None
    ]
    peak = max(speeds) if speeds else 0.0
    if region is not None and region in rules.dangerous_regions:
        return Tag.DANGEROUS
    if hand_shape in rules.dangerous_shapes and peak >= rules.shape_speed_fwps:
        return Tag.DANGEROUS
    if peak >= rules.strike_speed_fwps:
        return Tag.DANGEROUS
    return Tag.NON_DANGEROUS


def _confident(
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


def _framed(frame: TrackedFrame, layout: KeypointLayout, cfg: AnalysisConfig) -> bool:
    """Return whether the subject is framed with the face and both wrists visible."""
    if frame.framing not in (FramingState.HALF_BODY, FramingState.FULL_BODY):
        return False
    if frame.face_width_px is None or not np.isfinite(frame.face_width_px):
        return False
    face = np.asarray(layout.face, dtype=np.int64)
    visible_face = int(np.count_nonzero(_confident(frame, face, cfg.features.face_conf)))
    if visible_face < 2:
        return False
    for side in (Side.LEFT, Side.RIGHT):
        name = f"{side.value}_wrist"
        if not layout.has(name):
            return False
        wrist = np.asarray([layout.index(name)], dtype=np.int64)
        if not bool(_confident(frame, wrist, cfg.features.hand_conf)[0]):
            return False
    return True


def _framed_fraction(
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
) -> float:
    """Return the share of window frames that are framed with visible hands."""
    if not frames:
        return 0.0
    framed = sum(1 for frame in frames if _framed(frame, layout, cfg))
    return float(framed) / float(len(frames))


def _coverage(
    frames: Sequence[TrackedFrame],
    t_from: float,
    t_to: float,
    max_gap_s: float,
    source_healthy: bool,
) -> tuple[bool, float]:
    """Return the P-03 coverage gate and the last analyzed timestamp.

    Coverage requires a first frame at or before ``t_from + max_gap_s``, a last
    frame at or after ``t_to - max_gap_s`` and no gap above ``max_gap_s`` between
    consecutive frames inside the window. An unhealthy source always fails the
    gate, because neither freshness nor delivery can be trusted.
    """
    if not frames:
        return False, float(t_from)
    times = np.asarray([frame.t for frame in frames], dtype=np.float64)
    end = float(times[-1])
    if not source_healthy:
        return False, end
    start_ok = bool(times[0] <= t_from + max_gap_s)
    end_ok = bool(times[-1] >= t_to - max_gap_s)
    inside = times[(times >= t_from) & (times <= t_to)]
    # Insert the window bounds so an uncovered start or end is a gap too, not
    # only a hole between two analyzed frames (P-03).
    chain = np.concatenate(([t_from], inside, [t_to]))
    gaps_ok = bool(np.all(np.diff(chain) <= max_gap_s))
    return bool(start_ok and end_ok and gaps_ok), end


def _by_side(samples: Sequence[HandFaceSample]) -> dict[Side, list[HandFaceSample]]:
    grouped: dict[Side, list[HandFaceSample]] = {Side.LEFT: [], Side.RIGHT: []}
    for sample in samples:
        grouped[sample.side].append(sample)
    return grouped


def _closest(samples: Sequence[HandFaceSample]) -> HandFaceSample | None:
    measured = [sample for sample in samples if sample.distance_fw is not None]
    if not measured:
        return None
    return min(measured, key=_distance_key)


def _hand_shape_at(
    frame: TrackedFrame | None,
    layout: KeypointLayout,
    side: Side,
    cfg: AnalysisConfig,
) -> HandShape:
    """Classify the hand of one frame, ``UNKNOWN`` when the hand is unavailable."""
    if frame is None:
        return HandShape.UNKNOWN
    hand = np.asarray(layout.hand(side), dtype=np.int64)
    if hand.size != 21:
        return HandShape.UNKNOWN
    return classify_hand_shape(frame.xy[hand], frame.conf[hand], cfg.hand_shape)


def _set_geometry(
    features: dict[str, float],
    samples: Sequence[HandFaceSample],
    frames_by_t: Mapping[float, TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
) -> None:
    """Fill the region and hand-shape codes from the closest sample."""
    closest = _closest(samples)
    if closest is None:
        return
    region = closest.nearest_region
    features["region_code"] = _REGION_CODES.get(region, 0.0) if region is not None else 0.0
    shape = _hand_shape_at(frames_by_t.get(closest.t), layout, closest.side, cfg)
    features["hand_shape_code"] = _SHAPE_CODES[shape]


def _base_features(
    coverage_ok: bool,
    coverage_end_t: float,
    framed_fraction: float,
    occlusion_max: float,
    min_distance: float | None,
    peak_closing: float | None,
) -> dict[str, float]:
    return {
        "min_distance_fw": -1.0 if min_distance is None else float(min_distance),
        "peak_closing_speed_fwps": -1.0 if peak_closing is None else float(peak_closing),
        "decel_fwps2": -1.0,
        "occlusion_max": float(occlusion_max),
        "head_recoil_fw": -1.0,
        "region_code": 0.0,
        "hand_shape_code": 0.0,
        "framed_fraction": float(framed_fraction),
        "coverage_end_t": float(coverage_end_t),
        "coverage_ok": 1.0 if coverage_ok else 0.0,
    }


def assess_fast(
    frames: Sequence[TrackedFrame],
    t_from: float,
    t_to: float,
    layout: KeypointLayout,
    cfg: AnalysisConfig,
    *,
    source_healthy: bool = True,
) -> VisionAssessment:
    """Return the stage-1 assessment for ``[t_from, t_to]`` (FR-015..FR-017).

    ``source_healthy`` carries the P-03 health signal: the caller sets it when
    no ``on_error`` arrived and the last captured frame is fresh enough. The
    verdict is ``UNDECIDED`` when the analyzed frames do not cover the window
    within ``cfg.max_gap_s`` or when the source is unhealthy. A contact that
    starts inside the window, or else a hand-to-face approach within
    ``cfg.approach_fw`` at or above ``cfg.approach_speed_fwps``, gives
    ``CONFIRMED``; its severity is always set: the maximum over the contacts or
    ``severity_of_approach`` for an approach-only window (P-02). ``REJECTED``
    needs the framing condition of FR-015 over the window and no hand inside
    ``cfg.contacts.contact_exit_fw``; everything else is ``UNDECIDED``.

    ``episode_id`` stays ``None``: the episode runtime fills it.
    """
    if not np.isfinite(t_from) or not np.isfinite(t_to) or t_to < t_from:
        raise ValueError(f"invalid assessment window [{t_from}, {t_to}]")
    if not np.isfinite(cfg.max_gap_s) or cfg.max_gap_s <= 0.0:
        raise ValueError("cfg.max_gap_s must be positive")
    if not 0.0 <= cfg.min_framed_fraction <= 1.0:
        raise ValueError("cfg.min_framed_fraction must be in [0, 1]")

    ordered = sorted(frames, key=lambda frame: frame.t)
    frames_by_t = {frame.t: frame for frame in ordered}
    coverage_ok, coverage_end_t = _coverage(ordered, t_from, t_to, cfg.max_gap_s, source_healthy)
    window = [frame for frame in ordered if t_from <= frame.t <= t_to]
    samples = [
        sample for frame in window for sample in hand_face_samples(frame, layout, cfg.features)
    ]
    distances = [float(sample.distance_fw) for sample in samples if sample.distance_fw is not None]
    speeds = [
        float(sample.closing_speed_fwps)
        for sample in samples
        if sample.closing_speed_fwps is not None
    ]
    min_distance = min(distances) if distances else None
    peak_closing = max(0.0, max(speeds)) if speeds else None
    occlusion_max = max((sample.occlusion_fraction for sample in samples), default=0.0)
    features = _base_features(
        coverage_ok,
        coverage_end_t,
        _framed_fraction(window, layout, cfg),
        occlusion_max,
        min_distance,
        peak_closing,
    )
    if not coverage_ok:
        _set_geometry(features, samples, frames_by_t, layout, cfg)
        return VisionAssessment(Verdict.UNDECIDED, None, features)

    contacts = detect_contacts(ordered, layout, cfg) if ordered else []
    window_contacts = [contact for contact in contacts if t_from <= contact.t_start <= t_to]
    if window_contacts:
        ranked = max(
            ((severity_of(contact, cfg), contact) for contact in window_contacts),
            key=lambda item: (_SEVERITY_RANK[item[0]], -item[1].t_start),
        )
        severity, chosen = ranked
        features["min_distance_fw"] = float(chosen.min_distance_fw)
        features["peak_closing_speed_fwps"] = (
            -1.0
            if chosen.peak_closing_speed_fwps is None
            else float(chosen.peak_closing_speed_fwps)
        )
        features["decel_fwps2"] = -1.0 if chosen.decel_fwps2 is None else float(chosen.decel_fwps2)
        features["head_recoil_fw"] = (
            -1.0 if chosen.head_recoil_fw is None else float(chosen.head_recoil_fw)
        )
        features["region_code"] = _REGION_CODES.get(chosen.region, 0.0)
        features["hand_shape_code"] = _SHAPE_CODES[chosen.hand_shape]
        return VisionAssessment(Verdict.CONFIRMED, severity, features)

    approaches: list[tuple[Side, list[HandFaceSample], list[HandFaceSample]]] = []
    for side, side_samples in _by_side(samples).items():
        qualifying = [
            sample
            for sample in side_samples
            if sample.distance_fw is not None
            and sample.distance_fw <= cfg.approach_fw
            and (sample.closing_speed_fwps or 0.0) >= cfg.approach_speed_fwps
        ]
        if qualifying:
            approaches.append((side, side_samples, qualifying))
    if approaches:
        ranked_approaches = []
        for side, side_samples, qualifying in approaches:
            closest = min(qualifying, key=_distance_key)
            shape = _hand_shape_at(frames_by_t.get(closest.t), layout, side, cfg)
            severity = severity_of_approach(side_samples, shape, cfg)
            ranked_approaches.append((severity, closest, shape))
        severity, closest, shape = max(
            ranked_approaches,
            key=lambda item: (_SEVERITY_RANK[item[0]], -item[1].t),
        )
        closest_distance = closest.distance_fw
        features["min_distance_fw"] = -1.0 if closest_distance is None else float(closest_distance)
        features["region_code"] = (
            _REGION_CODES.get(closest.nearest_region, 0.0)
            if closest.nearest_region is not None
            else 0.0
        )
        features["hand_shape_code"] = _SHAPE_CODES[shape]
        return VisionAssessment(Verdict.CONFIRMED, severity, features)

    _set_geometry(features, samples, frames_by_t, layout, cfg)
    framed = features["framed_fraction"]
    if (
        framed >= cfg.min_framed_fraction
        and min_distance is not None
        and min_distance > cfg.contacts.contact_exit_fw
    ):
        return VisionAssessment(Verdict.REJECTED, None, features)
    return VisionAssessment(Verdict.UNDECIDED, None, features)


def _distance_signal(
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the per-frame minimum hand-to-face distance signal (in fw)."""
    times: list[float] = []
    values: list[float] = []
    for frame in frames:
        distances = [
            float(sample.distance_fw)
            for sample in hand_face_samples(frame, layout, cfg.features)
            if sample.distance_fw is not None
        ]
        if not distances:
            continue
        times.append(float(frame.t))
        values.append(min(distances))
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )


def _interval_statistics(contacts: Sequence[Contact]) -> tuple[float | None, float | None]:
    """Return the contact repetition frequency and the interval CV."""
    starts = np.asarray([contact.t_start for contact in contacts], dtype=np.float64)
    if starts.size < 2:
        return None, None
    intervals = np.diff(starts)
    mean_interval = float(np.mean(intervals))
    repetition = 1.0 / mean_interval if mean_interval > _EPS else None
    if intervals.size < 2:
        return repetition, None
    spread = float(np.std(intervals, ddof=1))
    coefficient = spread / mean_interval if mean_interval > _EPS else None
    return repetition, coefficient


def _speed_trend(contacts: Sequence[Contact]) -> float | None:
    """Return the linear slope of the contact peak closing speeds."""
    points = [
        (contact.t_start, float(contact.peak_closing_speed_fwps))
        for contact in contacts
        if contact.peak_closing_speed_fwps is not None
    ]
    if len(points) < 2:
        return None
    times = np.asarray([point[0] for point in points], dtype=np.float64)
    speeds = np.asarray([point[1] for point in points], dtype=np.float64)
    centered = times - float(np.mean(times))
    variance = float(centered @ centered)
    if variance <= _EPS:
        return None
    return float(centered @ (speeds - float(np.mean(speeds))) / variance)


def _active(
    frames: Sequence[TrackedFrame],
    contacts: Sequence[Contact],
    labels: frozenset[StereotypyLabel],
    now: float,
    layout: KeypointLayout,
    cfg: AnalysisConfig,
) -> bool:
    """Return whether hand-to-face or stereotypy activity happened recently."""
    cutoff = now - cfg.activity_window_s
    if any(contact.t_start + contact.duration_s >= cutoff for contact in contacts):
        return True
    if not labels.intersection({StereotypyLabel.ROCKING, StereotypyLabel.HAND_FLAPPING}):
        return False
    recent = [frame for frame in frames if frame.t >= cutoff]
    return bool(recent) and (
        detect_rocking(recent, layout, cfg) or detect_flapping(recent, layout, cfg)
    )


def _wrist_acceleration(
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    side: Side,
    cfg: AnalysisConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return camera wrist acceleration magnitude in fw/s^2 for one side."""
    name = f"{side.value}_wrist"
    if not layout.has(name):
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    wrist = np.asarray([layout.index(name)], dtype=np.int64)
    times: list[float] = []
    values: list[float] = []
    for frame in frames:
        width = frame.face_width_px
        if width is None or not np.isfinite(width) or width <= 0.0:
            continue
        if not bool(_confident(frame, wrist, cfg.features.hand_conf)[0]):
            continue
        acceleration = np.asarray(frame.acc, dtype=np.float64)[wrist[0]]
        if not bool(np.all(np.isfinite(acceleration))):
            continue
        times.append(float(frame.t))
        values.append(float(np.linalg.norm(acceleration)) / float(width))
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )


def _clock(
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
    imu: IMUSignal,
) -> tuple[float | None, float | None]:
    """Return the IMU-to-camera clock offset and consistency (FR-023)."""
    best: tuple[float, float] | None = None
    for side in (Side.LEFT, Side.RIGHT):
        times, values = _wrist_acceleration(frames, layout, side, cfg)
        if times.size < 3:
            continue
        try:
            offset, correlation = estimate_clock_offset(imu.t_s, imu.acc_norm_g, times, values)
        except ValueError:
            continue
        if not (np.isfinite(offset) and np.isfinite(correlation)):
            continue
        if best is None or correlation > best[1]:
            best = (float(offset), float(correlation))
    return (None, None) if best is None else best


def build_report(
    episode_id: str,
    t_start: float,
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
    now: float,
    imu: IMUSignal | None = None,
) -> EpisodeReport:
    """Build the stage-2 episode report (FR-018, FR-019, FR-023).

    Contacts come from :func:`cpsa.vision.contacts2d.detect_contacts` over the
    episode frames. ``repetition_hz`` is the contact repetition
    (``1 / mean interval``), falling back to the periodicity of the distance
    signal when there are fewer than two contacts; ``periodicity`` is the
    strength of that distance oscillation and ``interval_cv`` the coefficient of
    variation of the intervals (``None`` below three contacts).
    ``speed_trend_fwps_per_s`` is the least-squares slope of the contact peak
    speeds. ``peak_severity`` comes from the contacts only: the rocking and
    flapping labels never change severity (FR-019). ``active`` is true while a
    contact or an oscillation happened within ``cfg.activity_window_s`` of
    ``now``. When ``imu`` is given, the clock offset and the wrist-motion
    correlation come from :func:`cpsa.vision.fusion.estimate_clock_offset`.
    """
    if not np.isfinite(t_start):
        raise ValueError("t_start must be finite")
    if not np.isfinite(now):
        raise ValueError("now must be finite")
    ordered = sorted((frame for frame in frames if frame.t >= t_start), key=lambda frame: frame.t)
    contacts = tuple(detect_contacts(ordered, layout, cfg)) if ordered else ()
    repetition, interval_cv = _interval_statistics(contacts)
    signal_t, signal_values = _distance_signal(ordered, layout, cfg)
    periodicity_strength: float | None = None
    if signal_t.size >= 4:
        frequency, strength = periodicity(signal_t, signal_values, cfg.repetition_band_hz)
        periodicity_strength = strength if frequency is not None else 0.0
        if repetition is None and frequency is not None:
            repetition = frequency

    labels: set[StereotypyLabel] = set()
    if contacts:
        labels.add(StereotypyLabel.HAND_TO_FACE)
    if detect_rocking(ordered, layout, cfg):
        labels.add(StereotypyLabel.ROCKING)
    if detect_flapping(ordered, layout, cfg):
        labels.add(StereotypyLabel.HAND_FLAPPING)
    frozen_labels = frozenset(labels)
    peak_severity = (
        max(
            (severity_of(contact, cfg) for contact in contacts), key=lambda tag: _SEVERITY_RANK[tag]
        )
        if contacts
        else None
    )
    offset, consistency = (None, None)
    if imu is not None:
        offset, consistency = _clock(ordered, layout, cfg, imu)
    return EpisodeReport(
        episode_id=episode_id,
        t_start=float(t_start),
        t_end=float(ordered[-1].t) if ordered else None,
        active=_active(ordered, contacts, frozen_labels, float(now), layout, cfg),
        contacts=contacts,
        contact_count=len(contacts),
        repetition_hz=repetition,
        periodicity=periodicity_strength,
        interval_cv=interval_cv,
        speed_trend_fwps_per_s=_speed_trend(contacts),
        peak_severity=peak_severity,
        labels=frozen_labels,
        clock_offset_s=offset,
        imu_consistency=consistency,
    )
