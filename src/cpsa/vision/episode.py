"""Episode state, deterministic frame selection and the synchronous analyzer.

``EpisodeAnalyzer`` is the pure core of the 2D pipeline (FR-002, FR-003,
FR-015..FR-020): it consumes pose frames in timestamp order, tracks one subject
and answers the stage-1 verdict and the stage-2 report from frame timestamps
only. Live capture and replay drive the same analyzer, which is what makes
FR-025 possible.

``select_frames`` is the deterministic analyzed-frame rule of P-06: the first
frame at or after every grid point ``t0 + k / analysis_fps``. It depends on
timestamps only, so the same recorded frames select the same grid in live and
replay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import numpy as np

from cpsa.domain.model import FaceRegion, HandShape, Side, Tag, Verdict, VisionAssessment
from cpsa.vision.assessment import (
    AnalysisConfig,
    assess_fast,
    build_report,
    severity_of,
    severity_of_approach,
)
from cpsa.vision.contacts2d import (
    _SHAPES,
    _face_centroids,
    _head_recoil,
    _holds_contact,
    _in_contact,
)
from cpsa.vision.fusion import IMUSignal
from cpsa.vision.handface import hand_face_samples
from cpsa.vision.handshape import classify_hand_shape
from cpsa.vision.oscillation import detect_flapping, detect_rocking
from cpsa.vision.tracker2d import KeypointTracker2D, Tracker2DConfig
from cpsa.vision.types import (
    Contact,
    EpisodeReport,
    HandFaceSample,
    KeypointLayout,
    PoseFrame,
    TrackedFrame,
)

__all__ = ["EpisodeAnalyzer", "select_frames"]

_EPS = 1e-9


class _Timed(Protocol):
    @property
    def t(self) -> float: ...


class _Tracker(Protocol):
    def update(self, frame: PoseFrame) -> TrackedFrame: ...

    def reset(self) -> None: ...


TimedT = TypeVar("TimedT", bound=_Timed)


def select_frames(
    frames: Sequence[TimedT],
    analysis_fps: float,
    t0: float | None = None,
) -> list[TimedT]:
    """Return the analyzed frames of one episode (P-06).

    For every grid point ``t0 + k / analysis_fps`` keeps the first frame whose
    timestamp is at or after it; a frame is kept at most once. ``t0`` defaults
    to the first frame's timestamp. Frames are sorted by timestamp, so the
    result is deterministic and depends on timestamps only.
    """
    if not np.isfinite(analysis_fps) or analysis_fps <= 0.0:
        raise ValueError("analysis_fps must be a finite positive number")
    ordered = sorted(frames, key=lambda frame: frame.t)
    if not ordered:
        return []
    start = float(ordered[0].t) if t0 is None else float(t0)
    if not np.isfinite(start):
        raise ValueError("t0 must be finite")
    selected: list[TimedT] = []
    step = 0
    for frame in ordered:
        if frame.t + _EPS < start + step / analysis_fps:
            continue
        selected.append(frame)
        step = max(int(np.floor((frame.t - start) * analysis_fps + _EPS)) + 1, 0)
    return selected


@dataclass
class _ContactRun:
    """Incremental state of one hand-to-face contact run (FR-014, FR-021).

    :func:`~cpsa.vision.contacts2d.detect_contacts` re-reads the whole run on
    every frame, which makes grading a held contact quadratic in its length.
    This state accumulates the same quantities from each new frame instead: the
    closest sample, the deceleration pair, the hand-shape votes and the bounded
    recoil window. The resulting contact is therefore identical to the one
    ``detect_contacts`` builds for the same frames, at a cost per frame that
    does not grow with the contact.

    An ended run stays pending until frames up to ``t_start +
    recoil_window_s`` have been pushed, so a late head recoil still upgrades
    its finalized grade. A frame without a sample of this side does not end the
    run: :func:`~cpsa.vision.contacts2d.detect_contacts` grades per-side sample
    runs and bridges such a gap, so the incremental state does the same.
    """

    side: Side
    start_index: int
    t_start: float
    t_end: float
    counted_index: int
    recoil_end_index: int
    holds: bool = True
    min_distance_fw: float | None = None
    closest_region: FaceRegion | None = None
    fallback_region: FaceRegion | None = None
    peak_closing_speed_fwps: float | None = None
    decel_peak_t: float | None = None
    decel_peak_speed_fwps: float = 0.0
    decel_slowest_t: float | None = None
    decel_slowest_speed_fwps: float | None = None
    shape_counts: dict[HandShape, int] = field(default_factory=lambda: dict.fromkeys(_SHAPES, 0))
    peak: tuple[int, float] | None = None

    def observe_frame(self, frame: TrackedFrame, index: int, recoil_window_s: float) -> None:
        """Extend the bounded recoil window to one pushed frame."""
        if float(frame.t) <= self.t_start + recoil_window_s + _EPS:
            self.recoil_end_index = index

    def observe_approach(
        self,
        frames: Sequence[TrackedFrame],
        layout: KeypointLayout,
        cfg: AnalysisConfig,
    ) -> None:
        """Cache the approach-window peak used by the closing speed and deceleration."""
        lookback = self.t_start - cfg.contacts.approach_window_s
        first = self.start_index
        while first > 0 and float(frames[first - 1].t) + _EPS >= lookback:
            first -= 1
        samples: list[HandFaceSample] = []
        for position in range(first, self.start_index + 1):
            samples.extend(
                sample
                for sample in hand_face_samples(frames[position], layout, cfg.features)
                if sample.side is self.side
            )
        speeds = [
            float(sample.closing_speed_fwps)
            for sample in samples
            if sample.closing_speed_fwps is not None
        ]
        if speeds:
            self.peak_closing_speed_fwps = max(0.0, max(speeds))
        if samples:
            peak = max(samples, key=lambda sample: sample.closing_speed_fwps or 0.0)
            self.decel_peak_t = float(peak.t)
            self.decel_peak_speed_fwps = float(peak.closing_speed_fwps or 0.0)

    def observe_contact(
        self,
        index: int,
        sample: HandFaceSample,
        frames: Sequence[TrackedFrame],
        layout: KeypointLayout,
        cfg: AnalysisConfig,
    ) -> None:
        """Fold one contact sample and its frame into the run state."""
        self.t_end = float(sample.t)
        self._count_shape(frames, index, layout, cfg)
        distance = sample.distance_fw
        if distance is not None:
            if self.min_distance_fw is None or distance < self.min_distance_fw:
                self.min_distance_fw = distance
                self.closest_region = sample.nearest_region
            if self.fallback_region is None and sample.nearest_region is not None:
                self.fallback_region = sample.nearest_region
        closing = sample.closing_speed_fwps
        if (
            closing is not None
            and self.decel_peak_t is not None
            and self.t_end > self.decel_peak_t
            and (self.decel_slowest_speed_fwps is None or closing < self.decel_slowest_speed_fwps)
        ):
            self.decel_slowest_speed_fwps = closing
            self.decel_slowest_t = self.t_end

    def contact(
        self,
        frames: Sequence[TrackedFrame],
        layout: KeypointLayout,
        cfg: AnalysisConfig,
    ) -> Contact | None:
        """Build the contact the whole run describes, or ``None`` while unusable."""
        if self.min_distance_fw is None:
            return None
        region = self.closest_region if self.closest_region is not None else self.fallback_region
        if region is None:
            return None
        centroids = _face_centroids(
            frames[self.start_index : self.recoil_end_index + 1], layout, cfg.features
        )
        recoil, recoil_speed = _head_recoil(centroids, self.t_start, cfg.contacts.recoil_window_s)
        return Contact(
            side=self.side,
            region=region,
            t_start=self.t_start,
            duration_s=self.t_end - self.t_start,
            min_distance_fw=float(self.min_distance_fw),
            peak_closing_speed_fwps=self.peak_closing_speed_fwps,
            decel_fwps2=self._deceleration(),
            hand_shape=self._hand_shape(),
            head_recoil_fw=recoil,
            head_recoil_speed_fwps=recoil_speed,
        )

    def _count_shape(
        self,
        frames: Sequence[TrackedFrame],
        index: int,
        layout: KeypointLayout,
        cfg: AnalysisConfig,
    ) -> None:
        hand = np.asarray(layout.hand(self.side), dtype=np.int64)
        for position in range(self.counted_index + 1, index + 1):
            frame = frames[position]
            if hand.size == 21:
                shape = classify_hand_shape(
                    np.asarray(frame.xy, dtype=np.float64)[hand],
                    np.asarray(frame.conf, dtype=np.float64)[hand],
                    cfg.hand_shape,
                )
            else:
                shape = HandShape.UNKNOWN
            self.shape_counts[shape] += 1
        self.counted_index = index

    def _hand_shape(self) -> HandShape:
        """Return the majority shape with the tie-break of ``detect_contacts``."""
        return max(
            _SHAPES,
            key=lambda shape: (self.shape_counts[shape], -_SHAPES.index(shape)),
        )

    def _deceleration(self) -> float | None:
        if self.decel_peak_t is None or self.decel_peak_speed_fwps <= 0.0:
            return None
        if self.decel_slowest_t is None or self.decel_slowest_speed_fwps is None:
            return None
        drop = self.decel_peak_speed_fwps - self.decel_slowest_speed_fwps
        elapsed = self.decel_slowest_t - self.decel_peak_t
        if drop <= 0.0 or elapsed <= 0.0:
            return None
        return drop / elapsed


class EpisodeAnalyzer:
    """Track one subject through one episode and answer stage 1 and stage 2.

    ``start`` resets the Kalman states but keeps the session-long skeleton
    references (FR-009, P-09); ``end`` freezes the episode. ``assess`` applies
    the P-04 episode-grade state on top of :func:`~cpsa.vision.assessment.assess_fast`:

    1. coverage or source health not met -> ``UNDECIDED``;
    2. a contact or qualifying approach in the window -> ``CONFIRMED`` with the
       window maximum severity;
    3. a latest event that ended in the window -> ``CONFIRMED`` with that
       severity (``features.from_episode = 1``);
    4. recent rocking or flapping only -> ``UNDECIDED``
       (``features.activity_only = 1``);
    5. otherwise the stage-1 result, which can be ``REJECTED``.

    ``latest_grade`` is maintained by :meth:`push`, not by :meth:`assess`: every
    frame that carries a contact or qualifying approach records that event's own
    severity. Recency uses the event end (a contact's last held frame or an
    approach's last frame), the same reference as the stage-2 report activity, so
    a follow-up within ``activity_window_s`` of the end stays confirmed even when
    the contact started long before. A held contact records the current frame as
    its end, so it counts as active now. Contact runs are accumulated
    incrementally, and an ended run keeps being regraded until its recoil window
    is complete, so a short contact with a late head recoil still ends with the
    grade ``detect_contacts`` reports for those frames. This keeps assessments
    free of retained state (so a timed-out assessment cannot leak an untraced
    mutation) and makes audit replay reproduce the grade from the pushed frames
    alone.
    """

    def __init__(
        self,
        layout: KeypointLayout,
        tracker_cfg: Tracker2DConfig,
        analysis_cfg: AnalysisConfig,
        *,
        tracker: _Tracker | None = None,
    ) -> None:
        self._layout = layout
        self._cfg = analysis_cfg
        self._tracker: _Tracker = (
            tracker if tracker is not None else KeypointTracker2D(tracker_cfg, layout)
        )
        self._episode_id = ""
        self._t_start = 0.0
        self._frames: list[TrackedFrame] = []
        self._last_t: float | None = None
        self._latest_grade: tuple[float, Tag] | None = None
        self._contact_runs: list[_ContactRun] = []
        self._active_contacts: dict[Side, _ContactRun] = {}
        self._contact_peak: tuple[int, float] | None = None
        self._started = False
        self._ended = False

    # -- lifecycle --------------------------------------------------------

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def t_start(self) -> float:
        return self._t_start

    @property
    def frames(self) -> tuple[TrackedFrame, ...]:
        return tuple(self._frames)

    @property
    def latest_grade(self) -> tuple[float, Tag] | None:
        """End time and severity of the latest hand-to-face event in the episode."""
        return self._latest_grade

    @property
    def contact_peak(self) -> tuple[int, float] | None:
        """Severity rank and start time of the highest-severity contact so far.

        The rank is 1 for ``NON_DANGEROUS`` and 2 for ``DANGEROUS``. Ties keep
        the later contact, matching the peak rule of the episode report. A live
        or pending run contributes its current (or finalized) grade, so a late
        recoil upgrades the peak instead of leaving a stale rank.
        """
        peaks = [run.peak for run in self._contact_runs if run.peak is not None]
        if self._contact_peak is not None:
            peaks.append(self._contact_peak)
        return max(peaks) if peaks else None

    @property
    def skeleton_state(self) -> dict[str, object] | None:
        """Serializable session-level skeleton state, or ``None`` without one."""
        tracker = self._tracker
        if isinstance(tracker, KeypointTracker2D):
            return tracker.skeleton.state()
        return None

    def restore_skeleton_state(self, state: Mapping[str, object]) -> None:
        """Restore session-level skeleton references captured in an earlier episode."""
        tracker = self._tracker
        if isinstance(tracker, KeypointTracker2D):
            tracker.skeleton.restore(state)

    @property
    def started(self) -> bool:
        return self._started and not self._ended

    def start(self, episode_id: str, t: float) -> None:
        """Begin a new episode; keeps the session-long skeleton references."""
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        if not np.isfinite(t):
            raise ValueError("episode start time must be finite")
        self._tracker.reset()
        self._episode_id = episode_id
        self._t_start = float(t)
        self._frames = []
        self._last_t = None
        self._latest_grade = None
        self._contact_runs.clear()
        self._active_contacts.clear()
        self._contact_peak = None
        self._started = True
        self._ended = False

    def end(self) -> None:
        """Freeze the episode; later pushes and assessments raise."""
        if self._started and not self._ended:
            for run in tuple(self._contact_runs):
                self._finalize_contact(run)
            self._contact_runs.clear()
            self._active_contacts.clear()
            self._ended = True

    def push(self, pose_frame: PoseFrame) -> TrackedFrame:
        """Track and append one pose frame. Frames must be in timestamp order."""
        if not self._started or self._ended:
            raise RuntimeError("episode is not active")
        if self._last_t is not None and pose_frame.t < self._last_t:
            raise ValueError("pose frames must arrive in timestamp order")
        tracked = self._tracker.update(pose_frame)
        self._frames.append(tracked)
        self._last_t = float(pose_frame.t)
        self._update_events(tracked)
        return tracked

    def _update_events(self, frame: TrackedFrame) -> None:
        """Record the latest hand-to-face event carried by one pushed frame (P-04)."""
        index = len(self._frames) - 1
        samples = hand_face_samples(frame, self._layout, self._cfg.features)
        by_side: dict[Side, HandFaceSample | None] = {Side.LEFT: None, Side.RIGHT: None}
        for found in samples:
            by_side[found.side] = found

        recoil_window = self._cfg.contacts.recoil_window_s
        for run in tuple(self._contact_runs):
            run.observe_frame(frame, index, recoil_window)

        for side in (Side.LEFT, Side.RIGHT):
            sample = by_side[side]
            active = self._active_contacts.get(side)
            if active is not None:
                if sample is None:
                    # ``detect_contacts`` grades per-side sample runs, so a
                    # frame without a sample of this side bridges the contact
                    # instead of ending it. The run keeps its old end and stays
                    # pending until a sample that does not hold arrives (R-15).
                    continue
                if _holds_contact(sample, self._cfg.contacts):
                    active.observe_contact(index, sample, self._frames, self._layout, self._cfg)
                    continue
                active.holds = False
                del self._active_contacts[side]
            if sample is None:
                continue
            if _in_contact(sample, self._cfg.contacts):
                self._start_contact(side, index, sample)
                continue
            distance = sample.distance_fw
            if (
                distance is not None
                and distance <= self._cfg.approach_fw
                and (sample.closing_speed_fwps or 0.0) >= self._cfg.approach_speed_fwps
            ):
                self._grade_approach(frame, side, sample.t)

        self._settle_contacts(frame)

    def _start_contact(self, side: Side, index: int, sample: HandFaceSample) -> None:
        """Begin one contact run at the pushed frame ``index``."""
        run = _ContactRun(
            side=side,
            start_index=index,
            t_start=float(sample.t),
            t_end=float(sample.t),
            counted_index=index - 1,
            recoil_end_index=index,
        )
        run.observe_approach(self._frames, self._layout, self._cfg)
        run.observe_contact(index, sample, self._frames, self._layout, self._cfg)
        self._contact_runs.append(run)
        self._active_contacts[side] = run

    def _settle_contacts(self, frame: TrackedFrame) -> None:
        """Regrade runs over their recoil tail and finalize the complete ones (FR-021).

        Every pending run is regraded on every pushed frame, including one that
        lost its side sample: ``detect_contacts`` bridges a missing sample, so
        the run may still continue and its recoil tail must keep growing until
        the window closes (R-10, R-15). Only a run that can no longer continue is
        finalized.
        """
        window = self._cfg.contacts.recoil_window_s
        for run in tuple(self._contact_runs):
            self._record_contact(run)
            if not run.holds and float(frame.t) + _EPS >= run.t_start + window:
                self._finalize_contact(run)

    def _record_contact(self, run: _ContactRun) -> None:
        """Grade one contact run from its accumulated state (FR-016, FR-021)."""
        contact = run.contact(self._frames, self._layout, self._cfg)
        if contact is None or contact.duration_s + _EPS < self._cfg.contacts.min_duration_s:
            return
        severity = severity_of(contact, self._cfg)
        # Recency uses the event end, matching the report activity (R-14).
        self._record_grade(contact.t_start + contact.duration_s, severity)
        run.peak = (1 if severity is Tag.NON_DANGEROUS else 2, contact.t_start)

    def _finalize_contact(self, run: _ContactRun) -> None:
        """Freeze one ended contact once its recoil tail is complete."""
        self._record_contact(run)
        self._contact_runs = [other for other in self._contact_runs if other is not run]
        if run.peak is not None and (self._contact_peak is None or run.peak > self._contact_peak):
            self._contact_peak = run.peak

    def _grade_approach(self, frame: TrackedFrame, side: Side, event_t: float) -> None:
        """Record one qualifying approach with its own severity (P-02)."""
        cutoff = event_t - self._cfg.activity_window_s
        side_samples = [
            sample
            for recent in self._frames
            if recent.t >= cutoff
            for sample in hand_face_samples(recent, self._layout, self._cfg.features)
            if sample.side is side
        ]
        hand = np.asarray(self._layout.hand(side), dtype=np.int64)
        if hand.size == 21:
            shape = classify_hand_shape(frame.xy[hand], frame.conf[hand], self._cfg.hand_shape)
        else:
            shape = HandShape.UNKNOWN
        # ``event_t`` is the current (last) frame of the approach: the recency
        # reference is the event end, as for contacts (R-14).
        self._record_grade(event_t, severity_of_approach(side_samples, shape, self._cfg))

    def _record_grade(self, t_end: float, severity: Tag) -> None:
        """Record one event's end time and severity; a later end wins (R-14)."""
        if self._latest_grade is None or t_end >= self._latest_grade[0]:
            self._latest_grade = (float(t_end), severity)

    # -- stage 1 ----------------------------------------------------------

    def assess(
        self,
        t_from: float,
        t_to: float,
        now: float | None = None,
        *,
        source_healthy: bool = True,
    ) -> VisionAssessment:
        """Return the stage-1 assessment for ``[t_from, t_to]`` at ``now``.

        The assessment is read-only: episode state, including ``latest_grade``,
        changes only when :meth:`push` records an event, so replaying the pushed
        frames reproduces it without replaying assessments.
        """
        if not self._started or self._ended:
            raise RuntimeError("episode is not active")
        if now is None:
            now = self._frames[-1].t if self._frames else self._t_start
        if not np.isfinite(now):
            raise ValueError("now must be finite")

        base = assess_fast(
            self._frames,
            t_from,
            t_to,
            self._layout,
            self._cfg,
            source_healthy=source_healthy,
        )
        features = dict(base.features)
        if base.verdict is Verdict.CONFIRMED:
            return VisionAssessment(Verdict.CONFIRMED, base.severity, features, self._episode_id)

        if not bool(features.get("coverage_ok", 0.0)):
            return VisionAssessment(Verdict.UNDECIDED, None, features, self._episode_id)

        recent = self._recent_grade(now)
        if recent is not None:
            features["from_episode"] = 1.0
            return VisionAssessment(Verdict.CONFIRMED, recent, features, self._episode_id)

        if self._activity_only(now):
            features["activity_only"] = 1.0
            return VisionAssessment(Verdict.UNDECIDED, None, features, self._episode_id)

        return VisionAssessment(base.verdict, None, features, self._episode_id)

    def _recent_grade(self, now: float) -> Tag | None:
        """Return the latest grade when its event ended inside the activity window.

        The recency reference is the event end, the same cutoff used by
        :func:`~cpsa.vision.assessment.build_report` activity, so ``assess`` and
        ``report.active`` cannot disagree (R-14). A held contact records the
        current frame as its end, so it counts as active now.
        """
        grade = self._latest_grade
        if grade is None or grade[0] < now - self._cfg.activity_window_s:
            return None
        return grade[1]

    def _activity_only(self, now: float) -> bool:
        """Return whether recent activity is rocking or flapping only (FR-019)."""
        cutoff = now - self._cfg.activity_window_s
        recent = [frame for frame in self._frames if frame.t >= cutoff]
        if not recent:
            return False
        return detect_rocking(recent, self._layout, self._cfg) or detect_flapping(
            recent, self._layout, self._cfg
        )

    # -- stage 2 ----------------------------------------------------------

    def active(self, now: float) -> bool:
        """Return whether the episode saw hand-to-face or stereotypy activity recently."""
        if not self._started or self._ended:
            return False
        if self._recent_grade(now) is not None:
            return True
        return self._build_report(now).active

    def report(self, now: float, imu: IMUSignal | None = None) -> EpisodeReport:
        """Return the stage-2 episode report (FR-018, FR-019, FR-023)."""
        if not self._started:
            raise RuntimeError("episode has not started")
        return self._build_report(now, imu)

    def _build_report(self, now: float, imu: IMUSignal | None = None) -> EpisodeReport:
        return build_report(
            self._episode_id,
            self._t_start,
            self._frames,
            self._layout,
            self._cfg,
            now,
            imu,
        )
