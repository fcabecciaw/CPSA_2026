"""Threaded episode runtime for the 2D vision pipeline (FR-001..FR-003, FR-024, FR-025).

The capture callback never blocks: it appends raw frames to a bounded pre-roll
ring and, while an episode is active, offers them to a bounded analysis queue
that drops the oldest frame when the analyzer falls behind (P-05).

``VisionVerifier`` is the application-layer shell around the pure
:class:`~cpsa.vision.episode.EpisodeAnalyzer`:

* one analysis thread owns the analyzer and pushes decoded, estimated frames;
* ``verify`` never touches the analyzer: it posts a request and waits on a
  condition until the result is posted or the stage-1 budget expires;
* ``observe`` starts or extends the episode without waiting (P-04);
* ``start`` and ``stop`` manage the source and the threads; a stalled estimator
  is abandoned after the join timeout with a warning;
* every pushed frame and every assessment is recorded in an
  :class:`EpisodeTrace`, which audit-mode replay re-executes exactly (P-06).

In continuous mode (``continuous=True``, FR-027..FR-029) the same analysis
thread also runs without an external detection. While no episode is active,
frames on the ``analysis_fps`` grid are estimated and tracked by a dedicated
watch :class:`~cpsa.vision.episode.EpisodeAnalyzer`, and a
:class:`~cpsa.vision.trigger.VisionTrigger` turns its contact events and the
rocking/flapping state into detections for an injected ``on_detection``
callback. During an episode the trigger is fed from the episode analyzer for
follow-ups, and the end of the episode forwards the trigger's ``clear`` fire.

Trace JSON Lines layout, written at episode end::

    {"type": "episode", "episode_id": ..., "t_start": ..., "layout": ...,
     "analysis_fps": ..., "report_now": ...}
    {"type": "frame", "seq": ..., "t": ...}
    {"type": "assessment", "index": ..., "t_from": ..., "t_to": ...,
     "now": ..., "timeout": false, "source_healthy": true, "error": false}
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from cpsa.application.config import VisionConfig
from cpsa.domain.model import (
    Detection,
    SampleKind,
    Side,
    Tag,
    Verdict,
    VisionAssessment,
)
from cpsa.ports import Clock, EvidenceSink, PoseEstimator, VideoSource, VisionObserver
from cpsa.vision.assessment import AnalysisConfig, severity_of
from cpsa.vision.episode import EpisodeAnalyzer
from cpsa.vision.fusion import IMUSignal, estimate_clock_offset
from cpsa.vision.oscillation import detect_flapping, detect_rocking
from cpsa.vision.trigger import TriggerFire, VisionTrigger
from cpsa.vision.types import (
    EpisodeReport,
    KeypointLayout,
    KeypointStatus,
    PoseFrame,
    TrackedFrame,
    VideoFrame,
)

__all__ = [
    "EpisodeTrace",
    "TraceAssessment",
    "TraceFrame",
    "VisionVerifier",
    "episode_report_json",
    "with_imu",
]

logger = logging.getLogger("cpsa.vision")

_EPS = 1e-9
_SEVERITY_RANK: Mapping[Tag, int] = {Tag.NON_DANGEROUS: 1, Tag.DANGEROUS: 2}
_WATCH_MAX_S = 300.0  # restart the watch analyzer at least every 5 min (memory bound)


# ---------------------------------------------------------------------------
# Episode trace (P-06)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceFrame:
    """One frame actually pushed into the analyzer, after selection and drops."""

    seq: int
    t: float


@dataclass(frozen=True)
class TraceAssessment:
    """One stage-1 assessment, replayed by audit mode.

    ``index`` is the number of pushed frames at which the result was computed.
    ``timeout`` and ``error`` mark the two deterministic failure outcomes: the
    stage-1 budget expired and the estimator failed. Audit replay reproduces
    them without re-running inference.
    """

    t_from: float
    t_to: float
    now: float
    index: int
    timeout: bool = False
    source_healthy: bool = True
    error: bool = False


@dataclass(frozen=True)
class EpisodeTrace:
    """The complete replay record of one episode.

    ``skeleton`` is the session-level :class:`~cpsa.vision.skeleton.SkeletonModel`
    snapshot as it was when the episode started. Audit replay restores it, so the
    first episode of a session and every later episode reproduce the same
    session-long bone references.
    """

    episode_id: str
    t_start: float
    layout: str
    analysis_fps: float
    report_now: float
    frames: tuple[TraceFrame, ...]
    assessments: tuple[TraceAssessment, ...]
    skeleton: Mapping[str, Any] | None = None

    def lines(self) -> list[str]:
        """Return the JSON Lines representation of the trace."""
        header = {
            "type": "episode",
            "episode_id": self.episode_id,
            "t_start": self.t_start,
            "layout": self.layout,
            "analysis_fps": self.analysis_fps,
            "report_now": self.report_now,
            "skeleton": self.skeleton,
        }
        lines = [json.dumps(header, sort_keys=True)]
        for index, frame in enumerate(self.frames):
            lines.append(
                json.dumps(
                    {"type": "frame", "index": index, "seq": frame.seq, "t": frame.t},
                    sort_keys=True,
                )
            )
        for record in self.assessments:
            lines.append(
                json.dumps(
                    {
                        "type": "assessment",
                        "index": record.index,
                        "t_from": record.t_from,
                        "t_to": record.t_to,
                        "now": record.now,
                        "timeout": record.timeout,
                        "source_healthy": record.source_healthy,
                        "error": record.error,
                    },
                    sort_keys=True,
                )
            )
        return lines

    def write(self, path: Path) -> None:
        """Write the trace atomically to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(self.lines()) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path) -> EpisodeTrace:
        """Read and validate a trace written by :meth:`write`."""
        return parse_trace(path.read_text(encoding="utf-8").splitlines())


def parse_trace(lines: Iterable[str]) -> EpisodeTrace:
    """Parse the JSON Lines of an episode trace; raise ``ValueError`` if invalid."""
    header: dict[str, Any] | None = None
    frames: list[TraceFrame] = []
    assessments: list[TraceAssessment] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"trace line {number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"trace line {number}: expected an object")
        kind = row.get("type")
        if kind == "episode":
            header = row
        elif kind == "frame":
            frames.append(TraceFrame(seq=int(row["seq"]), t=float(row["t"])))
        elif kind == "assessment":
            assessments.append(
                TraceAssessment(
                    t_from=float(row["t_from"]),
                    t_to=float(row["t_to"]),
                    now=float(row["now"]),
                    index=int(row["index"]),
                    timeout=bool(row.get("timeout", False)),
                    source_healthy=bool(row.get("source_healthy", True)),
                    error=bool(row.get("error", False)),
                )
            )
        else:
            raise ValueError(f"trace line {number}: unknown type {kind!r}")
    if header is None:
        raise ValueError("trace has no episode header")
    skeleton = header.get("skeleton")
    if skeleton is not None and not isinstance(skeleton, dict):
        raise ValueError("trace skeleton must be an object or null")
    return EpisodeTrace(
        episode_id=str(header["episode_id"]),
        t_start=float(header["t_start"]),
        layout=str(header["layout"]),
        analysis_fps=float(header["analysis_fps"]),
        report_now=float(header["report_now"]),
        frames=tuple(frames),
        assessments=tuple(assessments),
        skeleton=skeleton,
    )


def episode_report_json(report: EpisodeReport, trace_path: Path | None = None) -> dict[str, Any]:
    """Return the JSON record written for one episode report."""
    return {
        "event": "episode_report",
        "episode_id": report.episode_id,
        "t_start": report.t_start,
        "t_end": report.t_end,
        "active": report.active,
        "contact_count": report.contact_count,
        "repetition_hz": report.repetition_hz,
        "periodicity": report.periodicity,
        "interval_cv": report.interval_cv,
        "speed_trend_fwps_per_s": report.speed_trend_fwps_per_s,
        "peak_severity": None if report.peak_severity is None else report.peak_severity.name,
        "labels": sorted(label.value for label in report.labels),
        "clock_offset_s": report.clock_offset_s,
        "imu_consistency": report.imu_consistency,
        "trace": None if trace_path is None else str(trace_path),
        "contacts": [
            {
                "side": contact.side.value,
                "region": contact.region.value,
                "t_start": contact.t_start,
                "duration_s": contact.duration_s,
                "hand_shape": contact.hand_shape.value,
            }
            for contact in report.contacts
        ],
    }


# ---------------------------------------------------------------------------
# Analysis queue items and requests
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _FrameItem:
    frame: VideoFrame


@dataclass(eq=False)
class _ObserveItem:
    detection: Detection


@dataclass(eq=False)
class _VerifyItem:
    detection: Detection


@dataclass(eq=False)
class _HealthItem:
    pass


_Item: TypeAlias = _FrameItem | _ObserveItem | _VerifyItem | _HealthItem


@dataclass(eq=False)
class _Request:
    detection: Detection
    t_from: float
    t_to: float
    now: float
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    result: VisionAssessment | None = None
    abandoned: bool = False


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class VisionVerifier:
    """Two-stage vision verifier with threaded capture and analysis (P-05).

    Args:
        source: camera adapter delivering raw frames.
        estimator: pose adapter; called only while an episode is active.
        evidence: clip/keyframe sink, or ``None`` to disable evidence writes.
        cfg: validated vision configuration.
        clock: monotonic clock shared with the dispatcher and IMU stream.
        layout: keypoint layout of the pose model.
        log_dir: log base path; episode reports and (without evidence) traces go
            under ``<log_dir>/<day>/vision/``.
        observer: live vision results for a display adapter; called from the
            analysis thread. An exception is logged and swallowed, so it never
            changes a verdict (F2).
        continuous: run pose inference and tracking while idle, and start
            episodes from the camera alone (FR-027..FR-029, plan F8). Off by
            default, which keeps the IMU-triggered behavior unchanged (AC-026).
        trigger: the camera-only trigger; required with ``continuous`` and
            ignored without it, so the IMU-triggered path is unchanged.
        on_detection: receives the detections the trigger fires, from the
            analysis thread; required with ``continuous``. It must not block, and
            an exception is logged and swallowed.
    """

    def __init__(
        self,
        source: VideoSource,
        estimator: PoseEstimator,
        evidence: EvidenceSink | None,
        cfg: VisionConfig,
        clock: Clock,
        layout: KeypointLayout,
        *,
        log_dir: Path | None = None,
        observer: VisionObserver | None = None,
        continuous: bool = False,
        trigger: VisionTrigger | None = None,
        on_detection: Callable[[Detection], None] | None = None,
    ) -> None:
        if continuous and (trigger is None or on_detection is None):
            raise ValueError("continuous vision verification requires a trigger and on_detection")
        self._source = source
        self._estimator = estimator
        self._evidence = evidence
        self._cfg = cfg
        self._clock = clock
        self._layout = layout
        self._log_dir = log_dir
        self._observer = observer
        self._analyzer = EpisodeAnalyzer(layout, cfg.tracker, cfg.analysis)

        self._continuous = continuous
        self._trigger = trigger if continuous else None
        self._on_detection = on_detection
        self._watch: EpisodeAnalyzer | None = None
        self._watch_counter = 0
        self._watch_started_t = 0.0
        self._watch_grid_t = 0.0
        self._watch_contacts: tuple[Tag, ...] = ()
        self._watch_error: BaseException | None = None
        self._episode_contacts: tuple[Tag, ...] = ()
        self._trigger_t: float | None = None
        self._osc_recent: deque[TrackedFrame] = deque()

        self._cond = threading.Condition()
        self._items: deque[_Item] = deque()
        self._frame_items = 0
        self._pre_roll: deque[VideoFrame] = deque()
        self._dropped = 0
        self._drop_logged = False
        self._last_capture_t: float | None = None
        self._health_error: BaseException | None = None

        self._requests: list[_Request] = []
        self._episode_active = False
        self._want_active = False
        self._episode_id: str | None = None
        self._pending_episode_id: str | None = None
        self._episode_t_start = 0.0
        self._episode_counter = 0
        self._last_detection_mono: float | None = None
        self._episode_error: BaseException | None = None
        self._grid_t = 0.0
        self._snapshot_index = 0
        self._snapshot_until: float | None = None
        self._last_pushed_seq: int | None = None
        self._trace_frames: list[TraceFrame] = []
        self._trace_assessments: list[TraceAssessment] = []
        self._retained: deque[tuple[VideoFrame, TrackedFrame]] = deque()
        self._peak_event: tuple[int, float] | None = None
        self._peak_window: list[tuple[VideoFrame, TrackedFrame]] = []
        self._episode_skeleton: Mapping[str, Any] | None = None
        self._last_report: EpisodeReport | None = None
        self._last_trace: EpisodeTrace | None = None

        self._imu: dict[Side, deque[tuple[float, float]]] = {}
        self._imu_lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._lifecycle = threading.Lock()

    # -- public state (also used by tests) --------------------------------

    @property
    def episode_id(self) -> str | None:
        """Id of the active or most recent episode."""
        with self._cond:
            return self._episode_id

    @property
    def preroll_len(self) -> int:
        """Number of raw frames currently held in the bounded pre-roll."""
        with self._cond:
            return len(self._pre_roll)

    @property
    def last_report(self) -> EpisodeReport | None:
        """Report of the most recently finished episode."""
        with self._cond:
            return self._last_report

    @property
    def last_trace(self) -> EpisodeTrace | None:
        """Trace of the most recently finished episode."""
        with self._cond:
            return self._last_trace

    @property
    def analyzed_until_t(self) -> float | None:
        """Camera timestamp of the last frame pushed into the analyzer (P-05)."""
        with self._cond:
            return self._snapshot_until

    @property
    def health_error(self) -> BaseException | None:
        """Error that made the video source unhealthy, or ``None``.

        ``start`` records a camera that cannot be opened and ``_on_source_error``
        records a stream failure; both leave the verifier fail-open (``verify``
        returns ``UNDECIDED``) instead of raising. A ``Mode`` uses this read-only
        signal to report the failure it would otherwise not see (R-05).
        """
        with self._cond:
            return self._health_error

    def imu_sample_count(self, side: Side) -> int:
        """Number of buffered acceleration samples for one side (P-07)."""
        with self._imu_lock:
            buffer = self._imu.get(side)
            return 0 if buffer is None else len(buffer)

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Start the analysis thread and open the camera source.

        A camera that cannot be opened marks the source unhealthy and is logged;
        the rest of the runtime (sensors, dispatcher) keeps running and ``verify``
        then fails open with ``UNDECIDED``. Only construction errors (a missing
        model or vision runtime) are fatal and raise.
        """
        with self._lifecycle:
            if self._started:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="cpsa-vision-analysis", daemon=True
            )
            self._started = True
            self._thread.start()
            try:
                self._source.start(self._on_frame, self._on_source_error)
            except Exception as exc:
                with self._cond:
                    first = self._health_error is None
                    self._health_error = exc
                    self._cond.notify_all()
                if first:
                    logger.error(
                        "Vision video source failed to open; continuing without frames",
                        exc_info=exc,
                    )

    def stop(self) -> None:
        """Close the source and join the analysis thread with a bounded wait (P-05)."""
        with self._lifecycle:
            if not self._started:
                return
            self._stop.set()
            with self._cond:
                self._cond.notify_all()
            try:
                self._source.stop()
            except Exception:
                logger.exception("Failed to stop the vision video source")
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning("Vision analysis thread did not stop within 2 s; abandoning it")
            self._thread = None
            self._started = False

    # -- capture-side callbacks -------------------------------------------

    def _on_frame(self, frame: VideoFrame) -> None:
        """Capture callback: keep the pre-roll and offer active-episode frames."""
        with self._cond:
            self._pre_roll.append(frame)
            while self._pre_roll and frame.t - self._pre_roll[0].t > self._cfg.preroll_s:
                self._pre_roll.popleft()
            self._last_capture_t = frame.t
            want = self._want_active
            continuous = self._continuous
        if want or continuous:
            self._offer_frame(frame)

    def _offer_frame(self, frame: VideoFrame) -> None:
        """Append a frame to the bounded analysis queue, dropping the oldest."""
        with self._cond:
            if not self._want_active and not self._continuous:
                return
            if self._frame_items >= self._cfg.queue_max:
                for index, item in enumerate(self._items):
                    if isinstance(item, _FrameItem):
                        del self._items[index]
                        self._frame_items -= 1
                        self._dropped += 1
                        if not self._drop_logged:
                            self._drop_logged = True
                            logger.warning(
                                "Vision analysis queue full (%d frames); dropping the oldest frame",
                                self._cfg.queue_max,
                            )
                        break
            self._items.append(_FrameItem(frame))
            self._frame_items += 1
            self._cond.notify_all()

    def _on_source_error(self, error: Exception) -> None:
        """Capture callback: mark the source unhealthy and wake the analyzer."""
        with self._cond:
            first = self._health_error is None
            self._health_error = error
            self._items.append(_HealthItem())
            self._cond.notify_all()
        if first:
            logger.error("Vision video source failed", exc_info=error)

    # -- IMU feed (P-07) --------------------------------------------------

    def on_imu_sample(
        self, side: Side, kind: SampleKind, values: Sequence[float], ts: float
    ) -> None:
        """Keep a bounded IMU acceleration-magnitude buffer for the episode report.

        Raw BlueCoin acceleration is in mg; the report needs g (FR-023).
        """
        if kind is not SampleKind.ACC:
            return
        try:
            numbers = [float(value) for value in values[:3]]
        except (TypeError, ValueError):
            return
        if len(numbers) != 3 or not all(np.isfinite(numbers)):
            return
        magnitude = float(np.sqrt(sum(value * value for value in numbers))) / 1000.0
        sample_ts = float(ts)
        if not np.isfinite(sample_ts):
            return
        with self._imu_lock:
            buffer = self._imu.setdefault(side, deque())
            buffer.append((sample_ts, magnitude))
            horizon = sample_ts - self._cfg.imu_buffer_s
            while buffer and buffer[0][0] < horizon:
                buffer.popleft()

    # -- verifier port ----------------------------------------------------

    def verify(self, detection: Detection) -> VisionAssessment:
        """Verify one detection within ``fast_budget_s`` (FR-015, P-05)."""
        with self._cond:
            if not self._started:
                raise RuntimeError("VisionVerifier.start() was not called")
            episode_id = self._ensure_episode_id_locked(detection)
            self._want_active = True
            self._items.append(_VerifyItem(detection))
            now = self._clock.monotonic()
            request = _Request(
                detection=detection,
                t_from=detection.window_end_ts - self._cfg.lookback_s,
                t_to=detection.window_end_ts + self._cfg.fast_after_s,
                now=now,
                deadline=now + self._cfg.fast_budget_s,
            )
            self._requests.append(request)
            self._cond.notify_all()
        real_limit = time.monotonic() + self._cfg.fast_budget_s + 1.0

        while True:
            with self._cond:
                if request.done.is_set() and request.result is not None:
                    return request.result
                remaining = min(request.deadline - self._clock.monotonic(), 0.02)
            if remaining <= 0.0 or time.monotonic() > real_limit:
                break
            with self._cond:
                if not request.done.is_set():
                    self._cond.wait(timeout=remaining)

        with self._cond:
            healthy = self._healthy_locked(self._clock.monotonic())
            index = self._snapshot_index
            request.abandoned = True
            if request in self._requests:
                self._requests.remove(request)
            self._trace_assessments.append(
                TraceAssessment(
                    t_from=request.t_from,
                    t_to=request.t_to,
                    now=request.now,
                    index=index,
                    timeout=True,
                    source_healthy=healthy,
                )
            )
            self._cond.notify_all()
        logger.warning(
            "Vision verification of %s timed out after %.3f s",
            detection.tag.name,
            self._cfg.fast_budget_s,
        )
        return VisionAssessment(
            Verdict.UNDECIDED,
            None,
            {"timeout": 1.0},
            episode_id,
        )

    def observe(self, detection: Detection) -> None:
        """Start or extend the episode for every stereotypy detection (P-04).

        The dispatcher delivers observations from a bounded worker queue, so a
        detection can arrive after the episode it belongs to has already ended.
        Such a stale window must not open a spurious episode: when its
        ``window_end_ts`` is older than ``idle_timeout_s`` the observation is
        dropped instead.
        """
        age = self._clock.monotonic() - float(detection.window_end_ts)
        if age > self._cfg.idle_timeout_s:
            logger.debug(
                "Ignoring stale vision observation (window ended %.3f s ago, limit %.3f s)",
                age,
                self._cfg.idle_timeout_s,
            )
            return
        with self._cond:
            if not self._started:
                return
            self._ensure_episode_id_locked(detection)
            self._want_active = True
            self._items.append(_ObserveItem(detection))
            self._cond.notify_all()

    # -- analysis thread --------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._next_item()
            if item is None:
                self._maybe_end_episode()
                continue
            try:
                if isinstance(item, _FrameItem):
                    if self._continuous and not self._episode_active:
                        self._watch_frame(item.frame)
                    else:
                        self._analyze_frame(item.frame)
                elif isinstance(item, _ObserveItem):
                    self._start_or_extend(item.detection)
                elif isinstance(item, _VerifyItem):
                    self._start_or_extend(item.detection)
                    self._serve_requests()
                else:
                    self._serve_requests(force=True)
            except Exception:
                logger.exception("Vision analysis failed for one item")
            self._maybe_end_episode()
        try:
            if self._episode_active:
                self._finish_episode(self._clock.monotonic())
        except Exception:
            logger.exception("Failed to finish the vision episode on shutdown")

    def _next_item(self) -> _Item | None:
        with self._cond:
            if not self._items:
                self._cond.wait(timeout=0.02)
            if self._items:
                item = self._items.popleft()
                if isinstance(item, _FrameItem):
                    self._frame_items -= 1
                return item
        return None

    def _ensure_episode_id_locked(self, detection: Detection) -> str:
        """Return the id of the active or pending episode; allocate it once."""
        if self._episode_active and self._episode_id is not None:
            return self._episode_id
        if self._pending_episode_id is None:
            self._episode_counter += 1
            stamp = detection.timestamp.strftime("%Y%m%dT%H%M%S")
            self._pending_episode_id = f"{stamp}-{self._episode_counter:04d}"
        return self._pending_episode_id

    def _start_or_extend(self, detection: Detection) -> None:
        now = self._clock.monotonic()
        with self._cond:
            self._last_detection_mono = now
            if self._episode_active:
                return
            episode_id = self._ensure_episode_id_locked(detection)
            self._pending_episode_id = None
            self._episode_id = episode_id
            self._episode_t_start = detection.window_end_ts - self._cfg.lookback_s
            self._episode_active = True
            self._want_active = True
            self._grid_t = self._episode_t_start
            self._snapshot_index = 0
            self._snapshot_until = None
            self._last_pushed_seq = None
            self._episode_error = None
            self._drop_logged = False
            self._dropped = 0
            self._trace_frames = []
            self._trace_assessments = []
            self._retained.clear()
            self._peak_event = None
            self._peak_window = []
            self._episode_skeleton = self._analyzer.skeleton_state
            snapshot = [frame for frame in self._pre_roll if frame.t >= self._episode_t_start]
            if self._trigger is not None:
                self._episode_contacts = ()
                # The watch path already fed the frames up to the firing one;
                # replayed pre-roll frames must not feed the trigger again.
                self._trigger_t = float(detection.window_end_ts)
        self._analyzer.start(episode_id, self._episode_t_start)
        logger.info("Vision episode %s started at %.3f", episode_id, self._episode_t_start)
        self._notify_episode(episode_id, self._episode_t_start, True)
        for frame in snapshot:
            if self._stop.is_set():
                break
            self._analyze_frame(frame)

    def _analyze_frame(self, frame: VideoFrame) -> None:
        with self._cond:
            if not self._episode_active or self._episode_error is not None:
                return
            if self._last_pushed_seq is not None and frame.seq <= self._last_pushed_seq:
                return
            if self._snapshot_until is not None and frame.t <= self._snapshot_until:
                return
            if frame.t + _EPS < self._grid_t:
                return
            t_start = self._episode_t_start
        try:
            rgb = frame.decode()
            persons = self._estimator.estimate(rgb)
        except Exception as exc:
            self._record_episode_error(exc)
            return
        pose = PoseFrame(frame.seq, frame.t, frame.width, frame.height, tuple(persons))
        tracked = self._analyzer.push(pose)
        with self._cond:
            self._grid_t = (
                t_start
                + (int(np.floor((frame.t - t_start) * self._cfg.analysis_fps + _EPS)) + 1)
                / self._cfg.analysis_fps
            )
            self._last_pushed_seq = frame.seq
            self._snapshot_index += 1
            self._snapshot_until = frame.t
            self._trace_frames.append(TraceFrame(frame.seq, frame.t))
            self._retained.append((frame, tracked))
            retain_s = (
                self._cfg.evidence.clip_s
                + self._cfg.analysis.activity_window_s
                + self._cfg.idle_timeout_s
                + 1.0
            )
            while self._retained and frame.t - self._retained[0][0].t > retain_s:
                self._retained.popleft()
            self._update_peak_window(frame, tracked)
        self._feed_episode_trigger(frame, tracked)
        self._notify_tracked(tracked)
        self._serve_requests()

    def _update_peak_window(self, frame: VideoFrame, tracked: TrackedFrame) -> None:
        """Keep the clip window around the highest-severity contact (FR-021).

        The rolling ``_retained`` buffer is bounded by the idle and activity
        windows, so a peak that happens early in a long episode would be evicted
        before the evidence is written. The peak window is collected separately
        and never pruned by age.
        """
        if self._evidence is None or not self._cfg.evidence.enabled:
            return
        peak = self._analyzer.contact_peak
        if peak is None:
            return
        half = 0.5 * self._cfg.evidence.clip_s
        previous = self._peak_event
        if previous is None or peak[1] != previous[1]:
            self._peak_event = peak
            self._peak_window = [
                item for item in self._retained if abs(item[0].t - peak[1]) <= half + _EPS
            ]
            return
        self._peak_event = peak
        if abs(frame.t - peak[1]) <= half + _EPS:
            self._peak_window.append((frame, tracked))

    def _record_episode_error(self, error: BaseException) -> None:
        with self._cond:
            first = self._episode_error is None
            self._episode_error = error
        if first:
            logger.error(
                "Pose estimation failed in vision episode %s", self._episode_id, exc_info=error
            )
        self._serve_requests(force=True)

    def _serve_requests(self, *, force: bool = False) -> None:
        """Post assessments for requests whose window is covered (analysis thread only).

        Source health is evaluated at assessment completion with the current
        monotonic time and the value actually used is recorded in the trace, so
        audit replay reproduces the same verdict even when the freshness gate
        fails only later in the wait.
        """
        with self._cond:
            if not self._episode_active:
                return
            until = self._snapshot_until
            index = self._snapshot_index
            error = self._episode_error
            ready = [
                request
                for request in self._requests
                if not request.done.is_set()
                and not request.abandoned
                and (
                    force
                    or error is not None
                    or (until is not None and request.t_to <= until + _EPS)
                )
            ]
        computed: list[tuple[_Request, VisionAssessment, int, bool, bool]] = []
        for request in ready:
            completed_at = self._clock.monotonic()
            if completed_at > request.deadline:
                continue
            healthy = self._source_healthy(completed_at)
            if error is not None:
                result = VisionAssessment(
                    Verdict.UNDECIDED,
                    None,
                    {"error": 1.0},
                    self._episode_id,
                )
                computed.append((request, result, index, healthy, True))
                continue
            result = self._analyzer.assess(
                request.t_from,
                request.t_to,
                request.now,
                source_healthy=healthy,
            )
            computed.append((request, result, index, healthy, False))
        with self._cond:
            for request, result, at_index, healthy, failed in computed:
                if request.done.is_set() or request.abandoned:
                    continue
                request.result = result
                request.done.set()
                self._trace_assessments.append(
                    TraceAssessment(
                        t_from=request.t_from,
                        t_to=request.t_to,
                        now=request.now,
                        index=at_index,
                        timeout=False,
                        source_healthy=healthy,
                        error=failed,
                    )
                )
                if request in self._requests:
                    self._requests.remove(request)
            self._cond.notify_all()

    def _healthy_locked(self, now: float) -> bool:
        if self._health_error is not None:
            return False
        last = self._last_capture_t
        return last is not None and now - last <= self._cfg.stale_after_s

    def _source_healthy(self, now: float) -> bool:
        with self._cond:
            return self._healthy_locked(now)

    def _maybe_end_episode(self) -> None:
        now = self._clock.monotonic()
        with self._cond:
            if not self._episode_active or self._requests:
                return
            last_detection = self._last_detection_mono
        if last_detection is not None and now - last_detection < self._cfg.idle_timeout_s:
            return
        if self._analyzer.active(now):
            return
        self._finish_episode(now)

    def _finish_episode(self, now: float) -> None:
        with self._cond:
            if not self._episode_active:
                return
            episode_id = self._episode_id
            t_start = self._episode_t_start
            trace_frames = tuple(self._trace_frames)
            trace_assessments = tuple(self._trace_assessments)
            self._episode_active = False
            self._want_active = False
            self._last_detection_mono = None
            self._pending_episode_id = None
        if episode_id is None:
            return
        try:
            report = self._with_imu(self._analyzer.report(now))
            trace = EpisodeTrace(
                episode_id=episode_id,
                t_start=t_start,
                layout=self._layout.name,
                analysis_fps=self._cfg.analysis_fps,
                report_now=now,
                frames=trace_frames,
                assessments=trace_assessments,
                skeleton=self._episode_skeleton,
            )
            trace_path = self._write_trace(trace)
            self._write_report(report, trace_path)
            self._notify_episode(episode_id, t_start, False)
            self._notify_report(report)
            self._write_evidence(report)
            if self._evidence is not None and self._cfg.evidence.enabled:
                try:
                    self._evidence.purge(self._clock.now())
                except Exception:
                    logger.exception("Vision evidence purge failed")
            self._analyzer.end()
        finally:
            # A failing report or output step must not lose the clear fire that
            # resets the dashboard tag and starts the refractory period.
            self._finish_trigger(now)
        with self._cond:
            self._last_report = report
            self._last_trace = trace
            self._peak_event = None
            self._peak_window = []
            self._episode_skeleton = None
        logger.info("Vision episode %s ended at %.3f", episode_id, now)

    # -- continuous watch (FR-027..FR-029, plan F8) -----------------------

    def _watch_frame(self, frame: VideoFrame) -> None:
        """Analyze one idle frame on the watch grid and poll the trigger (FR-027).

        The watch is the camera-only replacement for an external detection: pose
        inference runs on the ``analysis_fps`` grid, a dedicated analyzer tracks
        the subject, and the trigger turns its contact and oscillation state into
        detections. A frame the estimator or the tracker rejects is logged and
        skipped; the watch continues with the next frame.
        """
        trigger = self._trigger
        if trigger is None or frame.t + _EPS < self._watch_grid_t:
            return
        try:
            rgb = frame.decode()
            persons = self._estimator.estimate(rgb)
        except Exception as exc:
            self._record_watch_error(exc)
            return
        if self._watch_error is not None:
            logger.info("Vision watch path recovered after: %s", self._watch_error)
            self._watch_error = None
        watch = self._watch_analyzer(frame.t)
        pose = PoseFrame(frame.seq, frame.t, frame.width, frame.height, tuple(persons))
        try:
            tracked = watch.push(pose)
        except Exception:
            logger.exception("Vision watch frame %d could not be tracked", frame.seq)
            return
        self._watch_grid_t = (
            self._watch_started_t
            + (int(np.floor((frame.t - self._watch_started_t) * self._cfg.analysis_fps + _EPS)) + 1)
            / self._cfg.analysis_fps
        )
        self._remember_oscillation_frame(tracked)
        self._watch_contacts = self._feed_trigger(
            trigger, watch.contact_events, self._watch_contacts, frame.t
        )
        self._emit_fire(trigger.poll(frame.t, episode_active=False))

    def _watch_analyzer(self, t: float) -> EpisodeAnalyzer:
        """Return the watch analyzer, restarting it after an episode or every 5 min."""
        watch = self._watch
        if watch is None or t - self._watch_started_t > _WATCH_MAX_S:
            self._watch_counter += 1
            watch = EpisodeAnalyzer(self._layout, self._cfg.tracker, self._cfg.analysis)
            watch.start(f"watch-{self._watch_counter}", t)
            self._watch = watch
            self._watch_started_t = t
            self._watch_grid_t = t
            self._watch_contacts = ()
        return watch

    def _feed_episode_trigger(self, frame: VideoFrame, tracked: TrackedFrame) -> None:
        """Feed the episode analyzer's events to the trigger for follow-ups (FR-020)."""
        trigger = self._trigger
        if trigger is None:
            return
        self._remember_oscillation_frame(tracked)
        last = self._trigger_t
        if last is not None and frame.t <= last + _EPS:
            return  # pre-roll replay, already fed by the watch path
        self._episode_contacts = self._feed_trigger(
            trigger, self._analyzer.contact_events, self._episode_contacts, frame.t
        )
        self._trigger_t = frame.t
        self._emit_fire(trigger.poll(frame.t, episode_active=True))

    def _feed_trigger(
        self,
        trigger: VisionTrigger,
        events: Sequence[tuple[float, Tag]],
        fed: tuple[Tag, ...],
        t: float,
    ) -> tuple[Tag, ...]:
        """Feed new contact events, grade upgrades, and the oscillation state at ``t``.

        Returns the grades now fed. The analyzer keeps one entry per contact in
        start order and upgrades its grade in place over the recoil tail, so one
        trigger entry can change grade after it was first fed. Every new contact is
        fed once and an upgraded one is re-fed, which the trigger needs to report
        the final grade; it counts each contact once whatever it is fed. Upgrades
        of contacts that already left ``contact_window_s`` are skipped: they can no
        longer take part in a start fire.
        """
        horizon = t - trigger.config.contact_window_s
        for index, (event_t, severity) in enumerate(events):
            if index < len(fed):
                if fed[index] is severity:
                    continue
                if event_t < horizon - _EPS:
                    continue
            trigger.contact(event_t, severity)
        trigger.oscillation(t, self._oscillation_active(t))
        return tuple(severity for _, severity in events)

    def _remember_oscillation_frame(self, tracked: TrackedFrame) -> None:
        """Keep the tracked frames of the last ``oscillation_min_s`` for FR-019."""
        trigger = self._trigger
        if trigger is None:
            return
        if self._osc_recent and tracked.t <= self._osc_recent[-1].t:
            return  # replayed pre-roll frame, already in the window
        self._osc_recent.append(tracked)
        horizon = tracked.t - trigger.config.oscillation_min_s
        while self._osc_recent and self._osc_recent[0].t < horizon - _EPS:
            self._osc_recent.popleft()

    def _oscillation_active(self, t: float) -> bool:
        """Whether rocking or flapping is present in the last ``oscillation_min_s``."""
        trigger = self._trigger
        if trigger is None:
            return False
        cutoff = t - trigger.config.oscillation_min_s
        recent = [frame for frame in self._osc_recent if frame.t >= cutoff]
        if not recent:
            return False
        cfg = self._cfg.analysis
        return detect_rocking(recent, self._layout, cfg) or detect_flapping(
            recent, self._layout, cfg
        )

    def _finish_trigger(self, now: float) -> None:
        """Forward the trigger's ``clear`` fire and restart the watch (FR-029)."""
        trigger = self._trigger
        if trigger is None:
            return
        self._emit_fire(trigger.episode_ended(now))
        self._osc_recent.clear()
        self._watch = None
        self._watch_contacts = ()
        self._episode_contacts = ()
        self._trigger_t = None

    def _emit_fire(self, fire: TriggerFire | None) -> None:
        """Turn one trigger decision into a detection for ``on_detection`` (F8)."""
        if fire is None:
            return
        callback = self._on_detection
        if callback is None:
            return
        detection = Detection(
            tag=fire.severity,
            timestamp=self._clock.now(),
            window_end_ts=fire.t,
            features=(),
        )
        try:
            callback(detection)
        except Exception:
            logger.exception("Vision trigger callback failed for %s", fire.reason)

    def _record_watch_error(self, error: BaseException) -> None:
        """Log a watch-path inference failure once; the watch keeps running (FR-032)."""
        first = self._watch_error is None
        self._watch_error = error
        if first:
            logger.error("Pose estimation failed in the vision watch path", exc_info=error)

    # -- outputs ----------------------------------------------------------

    def _notify_episode(self, episode_id: str, t_start: float, active: bool) -> None:
        observer = self._observer
        if observer is None:
            return
        try:
            observer.on_episode(episode_id, t_start, active)
        except Exception:
            logger.exception("Vision observer failed in on_episode")

    def _notify_tracked(self, tracked: TrackedFrame) -> None:
        observer = self._observer
        if observer is None:
            return
        try:
            observer.on_tracked(tracked)
        except Exception:
            logger.exception("Vision observer failed in on_tracked")

    def _notify_report(self, report: EpisodeReport) -> None:
        observer = self._observer
        if observer is None:
            return
        try:
            observer.on_report(report)
        except Exception:
            logger.exception("Vision observer failed in on_report")

    def _trace_path(self, episode_id: str, day: datetime) -> Path | None:
        folder = day.strftime("%Y-%m-%d")
        if self._cfg.evidence.enabled:
            return self._cfg.evidence.base_path / folder / episode_id / "trace.jsonl"
        if self._log_dir is not None:
            return self._log_dir / folder / "vision" / episode_id / "trace.jsonl"
        return None

    def _write_trace(self, trace: EpisodeTrace) -> Path | None:
        path = self._trace_path(trace.episode_id, self._clock.now())
        if path is None:
            logger.warning(
                "No evidence base or log dir; episode %s trace not written", trace.episode_id
            )
            return None
        try:
            trace.write(path)
        except OSError:
            logger.exception("Failed to write the vision trace for %s", trace.episode_id)
            return None
        return path

    def _write_report(self, report: EpisodeReport, trace_path: Path | None) -> None:
        payload = episode_report_json(report, trace_path)
        logger.info("episode_report %s", json.dumps(payload, sort_keys=True))
        if self._log_dir is None:
            return
        folder = self._clock.now().strftime("%Y-%m-%d")
        path = self._log_dir / folder / "vision" / "reports.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            logger.exception("Failed to append the vision episode report")

    def _write_evidence(self, report: EpisodeReport) -> None:
        sink = self._evidence
        if sink is None or not self._cfg.evidence.enabled:
            return
        with self._cond:
            retained = list(self._retained)
            peak_event = self._peak_event
            peak_window = list(self._peak_window)
        if not retained:
            return
        peak_t = self._peak_time(report, retained)
        half = 0.5 * self._cfg.evidence.clip_s
        window: list[tuple[VideoFrame, TrackedFrame]] = []
        if peak_event is not None and abs(peak_event[1] - peak_t) <= _EPS:
            window = [item for item in peak_window if peak_t - half <= item[0].t <= peak_t + half]
        if not window:
            window = [item for item in retained if peak_t - half <= item[0].t <= peak_t + half]
        if not window:
            logger.warning(
                "No retained frames cover the peak window at %.3f s of episode %s; "
                "writing the nearest available frame instead",
                peak_t,
                report.episode_id,
            )
            window = [min(retained, key=lambda item: abs(item[0].t - peak_t))]
        images: list[tuple[NDArray[np.uint8], TrackedFrame | None]] = []
        times: list[float] = []
        for frame, tracked in window:
            try:
                images.append((frame.decode(), tracked))
            except Exception:
                logger.exception("Failed to decode evidence frame %d", frame.seq)
                continue
            times.append(frame.t)
        if not images:
            logger.warning("No evidence frames available for episode %s", report.episode_id)
            return
        keyframe = min(range(len(images)), key=lambda index: abs(times[index] - peak_t))
        try:
            sink.write_episode(report.episode_id, images, self._cfg.analysis_fps, keyframe)
        except Exception:
            logger.exception("Failed to write evidence for episode %s", report.episode_id)

    def _peak_time(
        self,
        report: EpisodeReport,
        retained: Sequence[tuple[VideoFrame, TrackedFrame]],
    ) -> float:
        if not report.contacts:
            return retained[-1][0].t
        contact = max(
            report.contacts,
            key=lambda item: (
                _SEVERITY_RANK[severity_of(item, self._cfg.analysis)],
                item.t_start,
            ),
        )
        return contact.t_start

    # -- IMU report fields (P-07) -----------------------------------------

    def _with_imu(self, report: EpisodeReport) -> EpisodeReport:
        signals: dict[Side, IMUSignal] = {}
        for side in (Side.LEFT, Side.RIGHT):
            signal = self._imu_signal(side)
            if signal is not None:
                signals[side] = signal
        with self._cond:
            frames = self._analyzer.frames
        return with_imu(report, frames, self._layout, self._cfg.analysis, signals)

    def _imu_signal(self, side: Side) -> IMUSignal | None:
        with self._imu_lock:
            buffer = self._imu.get(side)
            samples = list(buffer) if buffer is not None else []
        if len(samples) < 3:
            return None
        times = np.asarray([sample[0] for sample in samples], dtype=np.float64)
        values = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        return IMUSignal(t_s=times, acc_norm_g=values)


def wrist_acceleration(
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
    side: Side,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return camera wrist-acceleration magnitude in face-widths/s^2 for one side."""
    name = f"{side.value}_wrist"
    if not layout.has(name):
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    wrist = layout.index(name)
    times: list[float] = []
    values: list[float] = []
    for frame in frames:
        width = frame.face_width_px
        if width is None or not np.isfinite(width) or width <= 0.0:
            continue
        if int(frame.status[wrist]) == int(KeypointStatus.LOST):
            continue
        if float(frame.conf[wrist]) < cfg.features.hand_conf:
            continue
        acceleration = np.asarray(frame.acc[wrist], dtype=np.float64)
        if not bool(np.all(np.isfinite(acceleration))):
            continue
        times.append(float(frame.t))
        values.append(float(np.linalg.norm(acceleration)) / float(width))
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def with_imu(
    report: EpisodeReport,
    frames: Sequence[TrackedFrame],
    layout: KeypointLayout,
    cfg: AnalysisConfig,
    imu: Mapping[Side, IMUSignal],
) -> EpisodeReport:
    """Attach the clock offset and consistency of the best-correlated side (FR-023).

    Each side's IMU acceleration magnitude is compared with that side's camera
    wrist acceleration; the side with the higher correlation wins. The offset
    must be added to IMU timestamps to place them on the camera clock.
    """
    best: tuple[float, float] | None = None
    for side in (Side.LEFT, Side.RIGHT):
        signal = imu.get(side)
        if signal is None:
            continue
        times, values = wrist_acceleration(frames, layout, cfg, side)
        if times.size < 3:
            continue
        try:
            offset, correlation = estimate_clock_offset(
                signal.t_s, signal.acc_norm_g, times, values
            )
        except ValueError:
            continue
        if not (np.isfinite(offset) and np.isfinite(correlation)):
            continue
        if best is None or correlation > best[1]:
            best = (float(offset), float(correlation))
    if best is None:
        return report
    return replace(report, clock_offset_s=best[0], imu_consistency=best[1])
