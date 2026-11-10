"""Recorded, labeled sessions acceptance (P-08, AC-007, AC-008).

This is a ``target`` test: it needs the labeled session tree produced on the
board, whose location is given by ``CPSA_LABELED_SESSIONS``. When the variable is
unset (CI, developer machines) the test skips with an explicit reason. The tree
contains session directories (schema v1) and a ``labels.jsonl`` file with one
record per analyzed window::

    {"session": "20260101-000000_P01", "window_end_ts": 12.4,
     "verdict": "CONFIRMED", "severity": "DANGEROUS",
     "side": "right", "region": "eyes", "hand_shape": "fist"}

Labeled verdicts remain blocked until the sessions are recorded and labeled by
the user; the synthetic tests cover the same logic meanwhile.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cpsa.adapters.pose.onnx import OnnxPoseEstimator
from cpsa.adapters.recording.reader import Session
from cpsa.application.config import load
from cpsa.board.vision_replay import (
    imu_signals_from_session,
    replay_ideal,
    video_frames_from_session,
)
from cpsa.domain.model import Detection, FaceRegion, HandShape, Side, Tag, Verdict
from cpsa.vision.assessment import severity_of
from cpsa.vision.layouts import get_layout

pytestmark = pytest.mark.target

ENV = "CPSA_LABELED_SESSIONS"
REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "config" / "board.yaml"
LABEL_FIELDS = (
    "session",
    "window_end_ts",
    "verdict",
    "severity",
    "side",
    "region",
    "hand_shape",
)


def labels_path() -> Path:
    root = os.environ.get(ENV)
    if not root:
        pytest.skip(f"{ENV} is not set; recorded sessions are not available")
    path = Path(root) / "labels.jsonl"
    if not path.is_file():
        pytest.skip(f"{path} is missing")
    return path


def read_labels(path: Path) -> list[dict[str, object]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path}: line {number}: invalid JSON: {exc.msg}") from exc
        assert isinstance(record, dict), f"{path}: line {number}: expected an object"
        for key in LABEL_FIELDS:
            assert key in record, f"{path}: line {number}: missing {key}"
        records.append(record)
    return records


def test_labeled_sessions_match_their_labels() -> None:
    path = labels_path()
    labels = read_labels(path)
    assert labels, f"{path} has no labels"

    config = load(SHIPPED)
    layout = get_layout(config.vision.layout)
    try:
        estimator = OnnxPoseEstimator(
            config.vision.model.path,
            layout,
            (config.vision.model.input_height, config.vision.model.input_width),
            conf_thr=config.vision.model.conf_thr,
            iou_thr=config.vision.model.iou_thr,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"pose model unavailable: {exc}")

    root = path.parent
    for record in labels:
        session = Session.open(root / str(record["session"]))
        frames = video_frames_from_session(session)
        window_end = float(record["window_end_ts"])
        detection = Detection(
            tag=Tag.DANGEROUS,
            timestamp=session.created,
            window_end_ts=window_end,
            features=(),
        )
        result = replay_ideal(
            frames,
            estimator,
            config.vision,
            layout,
            [detection],
            imu=imu_signals_from_session(session),
        )

        assert result.assessments, f"{record['session']}: no assessment"
        assessment = result.assessments[-1]
        expected_verdict = Verdict[str(record["verdict"])]
        expected_severity = Tag[str(record["severity"])]
        expected_side = Side(str(record["side"]))
        expected_region = FaceRegion(str(record["region"]))
        expected_shape = HandShape[str(record["hand_shape"])]
        assert assessment.verdict is expected_verdict, (
            f"{record['session']} window {window_end}: "
            f"verdict {assessment.verdict.value}, expected {expected_verdict.value}"
        )
        assert assessment.features.get("coverage_ok") == 1.0, (
            f"{record['session']} window {window_end}: "
            f"coverage_ok {assessment.features.get('coverage_ok')}"
        )

        assert result.report is not None, f"{record['session']}: no episode report"
        window_start = window_end - config.vision.lookback_s
        window_stop = window_end + config.vision.fast_after_s
        contacts = [
            contact
            for contact in result.report.contacts
            if window_start <= contact.t_start <= window_stop
        ]
        if expected_verdict is Verdict.CONFIRMED:
            assert contacts, (
                f"{record['session']} window {window_end}: no contact in "
                f"[{window_start:.3f}, {window_stop:.3f}], got "
                f"{[(contact.t_start, contact.region.value) for contact in result.report.contacts]}"
            )
            matching = [
                contact
                for contact in contacts
                if contact.side is expected_side
                and contact.region is expected_region
                and contact.hand_shape is expected_shape
            ]
            assert matching, (
                f"{record['session']} window {window_end}: no {expected_side.value} "
                f"{expected_region.value} {expected_shape.value} contact, got "
                f"{[(c.side.value, c.region.value, c.hand_shape.value) for c in contacts]}"
            )
            assert assessment.severity is expected_severity, (
                f"{record['session']} window {window_end}: severity "
                f"{assessment.severity}, expected {expected_severity.name}"
            )
            contact = matching[0]
            assert severity_of(contact, config.vision.analysis) is expected_severity, (
                f"{record['session']} window {window_end}: contact severity "
                f"{severity_of(contact, config.vision.analysis).name}, "
                f"expected {expected_severity.name}"
            )
        elif expected_verdict is Verdict.REJECTED:
            assert not contacts, (
                f"{record['session']} window {window_end}: REJECTED with contacts in "
                f"[{window_start:.3f}, {window_stop:.3f}], got "
                f"{[(contact.t_start, contact.region.value) for contact in contacts]}"
            )
