"""IMU replay plumbing with fake extractor/classifier (no native libraries)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpsa.adapters.recording.reader import Session
from cpsa.board.replay import main, replay_session
from tests.fakes import FakeClassifier, FakeExtractor

SESSION_ID = "20260101-120000_P01"


def write_session(root: Path, row_count: int = 375) -> Path:
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
    for index in range(row_count):
        ts = 1000.0 + index * 0.02
        for side in ("left", "right"):
            lines.append(f"{ts},{side},acc,1000,200,300,")
            lines.append(f"{ts},{side},gyr,10,20,30,")
            lines.append(f"{ts},{side},quat,0,0,0,10000")
    (path / "imu.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_replay_plumbing_produces_one_tag_per_classified_window(tmp_path: Path) -> None:
    session_path = write_session(tmp_path)
    session = Session.open(session_path)
    extractor = FakeExtractor()
    classifier = FakeClassifier()
    out = tmp_path / "tags.jsonl"

    tags = replay_session(session, out, extractor=extractor, classifier=classifier)

    assert len(tags) == 2
    assert [tag.tag for tag in tags] == ["DANGEROUS", "DANGEROUS"]
    assert tags[0].window_end_ts == pytest.approx(1000.0 + 299 * 0.02)
    assert tags[1].window_end_ts == pytest.approx(1000.0 + 374 * 0.02)
    for tag in tags:
        assert tag.timestamp.startswith("2026-01-01T12:00:")
    assert len(extractor.windows) == 3  # window #2 is only used for calibration
    assert [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] == [
        tag.to_json() for tag in tags
    ]


def test_replay_cli_refuses_a_non_aarch64_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session_path = write_session(tmp_path, row_count=1)
    monkeypatch.setattr("cpsa.board.replay.platform.machine", lambda: "x86_64")

    code = main([str(session_path), "--out", str(tmp_path / "tags.jsonl")])

    assert code == 2
    assert "aarch64" in capsys.readouterr().err


def test_replay_cli_reports_a_missing_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(tmp_path / "missing"), "--out", str(tmp_path / "tags.jsonl")])
    assert code == 2
    assert "not a session" in capsys.readouterr().err
