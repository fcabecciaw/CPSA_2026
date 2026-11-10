"""AC-04: replaying ``imu.csv`` matches calling the stages directly.

The native feature extractor keeps process-global state, so each replay runs in a
fresh spawned interpreter, exactly like the golden classifier test.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.target,
    pytest.mark.skipif(platform.machine() != "aarch64", reason="native libraries are aarch64-only"),
]

SESSION_ID = "20260101-120000_P01"
ROW_COUNT = 400


def write_session(root: Path) -> Path:
    path = root / SESSION_ID
    (path / "rgb").mkdir(parents=True)
    (path / "depth").mkdir()
    (path / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": SESSION_ID,
                "created": "2026-01-01T12:00:00+01:00",
                "participant_id": "P01",
                "imu": {
                    "enabled": True,
                    "left": "CPSA_L2",
                    "right": "CPSA_R2",
                    "max_skew_ms": 30,
                    "stale_ms": 50,
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "frames.csv").write_text("seq,trial_id,ts_rgb,ts_depth,rgb,depth\n", encoding="utf-8")
    (path / "trials.jsonl").write_text("", encoding="utf-8")

    lines = ["ts,side,kind,v0,v1,v2,v3"]
    for index in range(ROW_COUNT):
        ts = 1000.0 + index * 0.02
        phase = index * 0.15
        acc = (1000.0 + 80.0 * math.sin(phase), 200.0 * math.cos(phase), 300.0)
        gyr = (10.0 * math.sin(phase), 20.0, 30.0 * math.cos(phase))
        quat = (0.0, 0.0, math.sin(phase / 2) * 10000.0, math.cos(phase / 2) * 10000.0)
        for side in ("left", "right"):
            lines.append(f"{ts},{side},acc,{acc[0]:.6f},{acc[1]:.6f},{acc[2]:.6f},")
            lines.append(f"{ts},{side},gyr,{gyr[0]:.6f},{gyr[1]:.6f},{gyr[2]:.6f},")
            lines.append(
                f"{ts},{side},quat,{quat[0]:.6f},{quat[1]:.6f},{quat[2]:.6f},{quat[3]:.6f}"
            )
    (path / "imu.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _Clock:
    def __init__(self, base: datetime, first_ts: float) -> None:
        self._base = base
        self._first_ts = first_ts
        self._ts = first_ts

    def set(self, ts: float) -> None:
        self._ts = ts

    def monotonic(self) -> float:
        return self._ts

    def now(self) -> datetime:
        return self._base + timedelta(seconds=self._ts - self._first_ts)


def _direct_replay(session_path: str) -> list[tuple[float, str]]:
    """Call the live pipeline stages directly, without the replay helper."""
    from cpsa.adapters.native.classifier import NativeClassifier, NativeFeatureExtractor
    from cpsa.adapters.recording.reader import Session
    from cpsa.application.classification import ClassificationStage
    from cpsa.application.slot import LatestSlot
    from cpsa.domain.model import Detection, SampleKind, Side
    from cpsa.imu.sync import IMUSynchronizer
    from cpsa.imu.window import Windower

    session = Session.open(session_path)
    rows = session.imu
    order = np.argsort(rows["ts"], kind="stable")
    first_ts = float(rows["ts"][order[0]])
    clock = _Clock(session.created, first_ts)
    detections: list[Detection] = []

    class RecordingSlot(LatestSlot[Detection]):
        def put(self, item: Detection) -> None:
            detections.append(item)

    extractor = NativeFeatureExtractor()
    classifier = NativeClassifier()
    stage = ClassificationStage(extractor, classifier, RecordingSlot(), clock)
    windower = Windower(stage.on_window)
    synchronizer = IMUSynchronizer(
        session.sync_max_skew_s, session.sync_stale_s, windower.add, clock
    )
    for index in order.tolist():
        row = rows[index]
        ts = float(row["ts"])
        clock.set(ts)
        kind = SampleKind(str(row["kind"]))
        count = 4 if kind is SampleKind.QUAT else 3
        synchronizer.update(
            Side(str(row["side"])),
            kind,
            tuple(float(row[f"v{number}"]) for number in range(count)),
            ts,
        )
    return [(float(item.window_end_ts), item.tag.name) for item in detections]


def _production_replay(session_path: str, out_path: str) -> list[tuple[float, str]]:
    from cpsa.adapters.recording.reader import Session
    from cpsa.board.replay import native_stages, replay_session

    session = Session.open(session_path)
    extractor, classifier = native_stages()
    tags = replay_session(session, Path(out_path), extractor=extractor, classifier=classifier)
    return [(tag.window_end_ts, tag.tag) for tag in tags]


def test_replay_session_matches_direct_stage_calls(tmp_path: Path) -> None:
    session_path = write_session(tmp_path)
    out_path = tmp_path / "tags.jsonl"
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        expected = pool.submit(_direct_replay, str(session_path)).result()
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        actual = pool.submit(_production_replay, str(session_path), str(out_path)).result()

    assert actual == expected
    assert actual
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == len(actual)
