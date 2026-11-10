"""``cpsa-replay-vision``: replay recorded RGB frames through the episode analyzer.

Two modes (P-06, revised):

* **audit** (``--trace trace.jsonl``) re-executes exactly the frames a live
  episode pushed, at the same push index, with the same request bounds, ``now``,
  ``source_healthy`` and failure flags. Its verdicts, severities, features and
  report are identical to the live run (FR-025, AC-014).
* **ideal** (default) applies :func:`~cpsa.vision.episode.select_frames` to all
  session frames with no timing effects, for development and threshold tuning.
  No equivalence claim is made for this mode.

Detections come from ``--detections`` (JSON Lines) or, when absent, from the
session's trial annotations. IMU signals come from the session ``imu.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cpsa.adapters.recording.reader import Session, SessionFormatError
from cpsa.application.config import Config, ConfigError, VisionConfig, load
from cpsa.application.vision_verifier import (
    EpisodeTrace,
    episode_report_json,
    with_imu,
)
from cpsa.domain.model import Detection, Side, Tag, Verdict, VisionAssessment
from cpsa.ports import PoseEstimator
from cpsa.vision.episode import EpisodeAnalyzer, select_frames
from cpsa.vision.fusion import IMUSignal
from cpsa.vision.layouts import get_layout
from cpsa.vision.types import EpisodeReport, KeypointLayout, PoseFrame, VideoFrame

logger = logging.getLogger("cpsa.replay_vision")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STARTUP = 2


@dataclass(frozen=True)
class ReplayResult:
    """Assessments and final report of one replayed episode."""

    assessments: tuple[VisionAssessment, ...]
    report: EpisodeReport | None

    def to_json_lines(self) -> list[str]:
        """Return one JSON line per assessment plus the episode report."""
        lines = [
            json.dumps(
                {
                    "type": "assessment",
                    "episode_id": assessment.episode_id,
                    "verdict": assessment.verdict.value,
                    "severity": (None if assessment.severity is None else assessment.severity.name),
                    "features": {
                        name: float(value) for name, value in sorted(assessment.features.items())
                    },
                },
                sort_keys=True,
            )
            for assessment in self.assessments
        ]
        if self.report is not None:
            lines.append(json.dumps(episode_report_json(self.report), sort_keys=True))
        return lines


def _push(
    analyzer: EpisodeAnalyzer,
    estimator: PoseEstimator,
    frame: VideoFrame,
) -> None:
    rgb = frame.decode()
    persons = estimator.estimate(rgb)
    analyzer.push(PoseFrame(frame.seq, frame.t, frame.width, frame.height, tuple(persons)))


def replay_audit(
    frames: Sequence[VideoFrame],
    estimator: PoseEstimator,
    cfg: VisionConfig,
    layout: KeypointLayout,
    trace: EpisodeTrace,
    *,
    imu: Mapping[Side, IMUSignal] | None = None,
) -> ReplayResult:
    """Re-execute a live episode from its trace and return the same outputs (P-06)."""
    by_seq = {frame.seq: frame for frame in frames}
    ordered = sorted(trace.assessments, key=lambda record: record.index)
    analyzer = EpisodeAnalyzer(layout, cfg.tracker, cfg.analysis)
    if trace.skeleton is not None:
        analyzer.restore_skeleton_state(trace.skeleton)
    analyzer.start(trace.episode_id, trace.t_start)
    pushed = 0
    assessments: list[VisionAssessment] = []
    for record in ordered:
        if record.index > len(trace.frames):
            raise ValueError(
                f"assessment index {record.index} exceeds {len(trace.frames)} traced frames"
            )
        if record.index < pushed:
            raise ValueError("assessment indices are not monotonic")
        while pushed < record.index:
            pushed_frame = trace.frames[pushed]
            frame = by_seq.get(pushed_frame.seq)
            if frame is None:
                raise ValueError(f"traced frame seq {pushed_frame.seq} is not in the input")
            _push(analyzer, estimator, frame)
            pushed += 1
        if record.timeout:
            result = VisionAssessment(
                Verdict.UNDECIDED,
                None,
                {"timeout": 1.0},
                trace.episode_id,
            )
        elif record.error:
            result = VisionAssessment(
                Verdict.UNDECIDED,
                None,
                {"error": 1.0},
                trace.episode_id,
            )
        else:
            result = analyzer.assess(
                record.t_from,
                record.t_to,
                record.now,
                source_healthy=record.source_healthy,
            )
        assessments.append(result)
    while pushed < len(trace.frames):
        pushed_frame = trace.frames[pushed]
        frame = by_seq.get(pushed_frame.seq)
        if frame is None:
            raise ValueError(f"traced frame seq {pushed_frame.seq} is not in the input")
        _push(analyzer, estimator, frame)
        pushed += 1
    report = analyzer.report(trace.report_now)
    if imu:
        report = with_imu(report, analyzer.frames, layout, cfg.analysis, imu)
    return ReplayResult(tuple(assessments), report)


def replay_ideal(
    frames: Sequence[VideoFrame],
    estimator: PoseEstimator,
    cfg: VisionConfig,
    layout: KeypointLayout,
    detections: Sequence[Detection],
    *,
    imu: Mapping[Side, IMUSignal] | None = None,
) -> ReplayResult:
    """Replay all selected frames without timing effects (development mode)."""
    if not detections:
        return ReplayResult((), None)
    ordered_frames = sorted(frames, key=lambda frame: frame.t)
    analyzer = EpisodeAnalyzer(layout, cfg.tracker, cfg.analysis)
    assessments: list[VisionAssessment] = []
    selected: list[VideoFrame] = []
    index = 0
    started = False
    counter = 0
    ordered_detections = sorted(detections, key=lambda item: item.window_end_ts)
    for detection in ordered_detections:
        if not started:
            counter += 1
            stamp = detection.timestamp.strftime("%Y%m%dT%H%M%S")
            t_start = detection.window_end_ts - cfg.lookback_s
            analyzer.start(f"{stamp}-{counter:04d}", t_start)
            selected = select_frames(ordered_frames, cfg.analysis_fps, t0=t_start)
            index = 0
            started = True
        t_to = detection.window_end_ts + cfg.fast_after_s
        while index < len(selected) and selected[index].t <= t_to:
            _push(analyzer, estimator, selected[index])
            index += 1
        assessments.append(
            analyzer.assess(detection.window_end_ts - cfg.lookback_s, t_to, now=t_to)
        )
    while index < len(selected):
        _push(analyzer, estimator, selected[index])
        index += 1
    last_frame = selected[-1].t if selected else ordered_detections[-1].window_end_ts
    now = max(
        last_frame,
        ordered_detections[-1].window_end_ts + cfg.idle_timeout_s + cfg.analysis.activity_window_s,
    )
    report = analyzer.report(now)
    if imu:
        report = with_imu(report, analyzer.frames, layout, cfg.analysis, imu)
    return ReplayResult(tuple(assessments), report)


# ---------------------------------------------------------------------------
# Session adapters
# ---------------------------------------------------------------------------


def video_frames_from_session(session: Session) -> list[VideoFrame]:
    """Wrap the session's RGB JPEGs as lazy :class:`VideoFrame` objects."""
    width, height = _session_frame_size(session)
    frames: list[VideoFrame] = []
    for index in range(len(session.frames)):
        seq = int(session.frames.seq[index])
        timestamp = float(session.frames.ts_rgb[index])
        frames.append(
            VideoFrame(
                seq=seq,
                t=timestamp,
                width=width,
                height=height,
                decode=_rgb_loader(session, index),
            )
        )
    return frames


def imu_signals_from_session(session: Session) -> dict[Side, IMUSignal]:
    """Build per-side acceleration-magnitude signals in g from ``imu.csv``."""
    signals: dict[Side, IMUSignal] = {}
    rows = session.imu
    if rows.size == 0:
        return signals
    for side in (Side.LEFT, Side.RIGHT):
        selection = (rows["side"] == side.value) & (rows["kind"] == "acc")
        times = np.asarray(rows["ts"][selection], dtype=np.float64)
        values = np.asarray(
            np.sqrt(
                rows["v0"][selection] ** 2 + rows["v1"][selection] ** 2 + rows["v2"][selection] ** 2
            ),
            dtype=np.float64,
        )
        order = np.argsort(times, kind="stable")
        times = times[order]
        values = values[order] / 1000.0
        finite = np.isfinite(times) & np.isfinite(values)
        times, values = times[finite], values[finite]
        if times.size < 3 or np.any(np.diff(times) <= 0.0):
            continue
        signals[side] = IMUSignal(t_s=times, acc_norm_g=values)
    return signals


def load_detections(path: Path) -> tuple[Detection, ...]:
    """Read a JSON Lines detection file (``window_end_ts``, optional ``tag``)."""
    detections: list[Detection] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict) or "window_end_ts" not in row:
            raise ValueError(f"{path}: line {number}: missing window_end_ts")
        tag = Tag[str(row.get("tag", Tag.DANGEROUS.name))]
        timestamp = _detection_timestamp(row.get("timestamp"))
        detections.append(
            Detection(
                tag=tag,
                timestamp=timestamp,
                window_end_ts=float(row["window_end_ts"]),
                features=(),
            )
        )
    return tuple(detections)


def detections_from_trials(session: Session) -> tuple[Detection, ...]:
    """Use the session's trial cues as detections for the ideal replay."""
    return tuple(
        Detection(
            tag=Tag.DANGEROUS,
            timestamp=session.created,
            window_end_ts=float(trial.cue_ts),
            features=(),
        )
        for trial in session.trials
        if trial.status == "ok"
    )


def _session_frame_size(session: Session) -> tuple[int, int]:
    camera = session.meta.get("camera")
    if isinstance(camera, Mapping):
        width = camera.get("width")
        height = camera.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height
    return 640, 480


def _rgb_loader(session: Session, index: int) -> Callable[[], NDArray[np.uint8]]:
    def load() -> NDArray[np.uint8]:
        return session.rgb(index)

    return load


def _detection_timestamp(value: object) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpsa-replay-vision",
        description="Replay a recorded session through the 2D vision episode analyzer.",
    )
    parser.add_argument("session_dir", type=Path, help="session directory (schema v1)")
    parser.add_argument("--config", type=Path, default=Path("config/board.yaml"))
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--onnx", type=Path, help="ONNX pose model (CPU stand-in)")
    model.add_argument("--xmodel", type=Path, help="compiled DPU pose model")
    parser.add_argument("--detections", type=Path, help="JSON Lines detections file")
    parser.add_argument(
        "--trace",
        type=Path,
        help="audit mode: re-execute this live episode trace exactly",
    )
    parser.add_argument("--out", type=Path, help="write JSON Lines here instead of stdout")
    return parser


def _estimator(config: Config, args: argparse.Namespace, layout: KeypointLayout) -> PoseEstimator:
    vision = config.vision
    if args.xmodel is not None:
        from cpsa.adapters.pose.dpu import DpuPoseEstimator

        return DpuPoseEstimator(args.xmodel, layout, vision.model.conf_thr, vision.model.iou_thr)
    path = args.onnx if args.onnx is not None else vision.model.path
    from cpsa.adapters.pose.onnx import OnnxPoseEstimator

    return OnnxPoseEstimator(
        path,
        layout,
        (vision.model.input_height, vision.model.input_width),
        vision.model.conf_thr,
        vision.model.iou_thr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load(args.config)
    except ConfigError as exc:
        print(f"cpsa-replay-vision: {exc}", file=sys.stderr)
        return EXIT_STARTUP
    try:
        session = Session.open(args.session_dir)
    except (SessionFormatError, OSError) as exc:
        print(f"cpsa-replay-vision: {exc}", file=sys.stderr)
        return EXIT_STARTUP
    layout = get_layout(config.vision.layout)
    try:
        estimator = _estimator(config, args, layout)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"cpsa-replay-vision: {exc}", file=sys.stderr)
        return EXIT_STARTUP

    frames = video_frames_from_session(session)
    imu = imu_signals_from_session(session)
    try:
        if args.trace is not None:
            trace = EpisodeTrace.read(args.trace)
            result = replay_audit(frames, estimator, config.vision, layout, trace, imu=imu)
        else:
            detections = (
                load_detections(args.detections)
                if args.detections is not None
                else detections_from_trials(session)
            )
            result = replay_ideal(frames, estimator, config.vision, layout, detections, imu=imu)
    except (ValueError, OSError) as exc:
        print(f"cpsa-replay-vision: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        logger.exception("Vision replay failed")
        return EXIT_ERROR

    lines = result.to_json_lines()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        for line in lines:
            print(line)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
