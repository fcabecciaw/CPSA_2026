"""Schema-v1 writer -> reader round trip with fake camera and IMU sources."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cpsa.adapters.recording.reader import Session, SessionFormatError
from cpsa.adapters.recording.writer import (
    CameraInfo,
    ImuInfo,
    SessionWriter,
    TrialRecord,
    XuValues,
)
from cpsa.domain.model import SampleKind, Side
from cpsa.ports import RgbdFrame
from cpsa.recorder.protocol import load as load_protocol
from cpsa.recorder.protocol import plan_trials
from tests.fakes import FakeClock, FakeFrameSource, FakeSensorSource

CAMERA = CameraInfo(
    rgb_device="/dev/video0",
    depth_device="/dev/video2",
    width=640,
    height=480,
    fps=30.0,
    rgb_exposure=300,
    xu=XuValues(laser_power=16, accuracy=2, filter=6, confidence=1),
)
IMU = ImuInfo(
    enabled=True,
    left="CPSA_L2",
    right="CPSA_R2",
    max_skew_ms=30.0,
    stale_ms=50.0,
)
DIGEST = "a" * 64


def make_protocol_path(tmp_path: Path) -> Path:
    path = tmp_path / "protocol.yaml"
    path.write_text(
        """
seed: 11
distances_cm: [55]
pre_s: 0.5
post_s: 0.5
rest_s: 0.0
cue_lead_s: 0.0
classes:
  - {name: fist_temple, category: dangerous, hand: both, target: temple, reps: 1, duration_s: 1.0}
""",
        encoding="utf-8",
    )
    return path


def make_frame(seq: int, ts: float) -> RgbdFrame:
    rgb = np.empty((6, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = (seq * 7) % 256
    rgb[:, :, 1] = (seq * 11) % 256
    rgb[:, :, 2] = (seq * 13) % 256
    depth = (np.arange(48, dtype=np.uint16).reshape(6, 8) + seq * 10).astype(np.uint16)
    return RgbdFrame(seq=seq, rgb=rgb, depth=depth, ts_rgb=ts, ts_depth=ts + 0.001)


def new_writer(
    root: Path,
    clock: FakeClock,
    *,
    imu: ImuInfo = IMU,
    queue_size: int = 90,
    calibration: Path | None = None,
) -> SessionWriter:
    return SessionWriter(
        root=root,
        session_id="20260101-000000_P01",
        participant_id="P01",
        camera=CAMERA,
        imu=imu,
        protocol_sha256=DIGEST,
        protocol_seed=11,
        clock=clock,
        calibration_path=calibration,
        pre_s=0.5,
        post_s=0.5,
        queue_size=queue_size,
    )


def feed_imu(sensor: FakeSensorSource, start: float, end: float, rate_hz: float) -> int:
    values = {
        SampleKind.ACC: (1000.0, 200.0, 300.0),
        SampleKind.GYR: (10.0, 20.0, 30.0),
        SampleKind.QUAT: (0.0, 0.0, 0.0, 10000.0),
    }
    count = int(round((end - start) * rate_hz)) + 1
    for index in range(count):
        ts = start + index / rate_hz
        for side in (Side.LEFT, Side.RIGHT):
            for kind, sample in values.items():
                assert sensor.on_sample is not None
                sensor.on_sample(side, kind, sample, ts)
    return count


def test_writer_reader_roundtrip_with_trial_windows_and_imu(tmp_path: Path) -> None:
    clock = FakeClock(100.0)
    protocol = load_protocol(make_protocol_path(tmp_path))
    trials = plan_trials(protocol)
    assert len(trials) == 2

    writer = new_writer(tmp_path / "sessions", clock)
    frames_source = FakeFrameSource()
    sensor_source = FakeSensorSource()
    frames_source.start(writer.write_frame)
    sensor_source.start(writer.write_sample)
    sensor_count = feed_imu(sensor_source, 95.0, 115.0, 50.0)

    seq = 0
    ts = 95.0
    cues = [100.0, 110.0]
    emitted: list[tuple[int, float]] = []
    for index, (trial, cue) in enumerate(zip(trials, cues, strict=True)):
        while ts <= cue + 1e-9:
            assert frames_source.on_frame is not None
            frames_source.on_frame(make_frame(seq, ts))
            emitted.append((seq, ts))
            ts = round(ts + 0.1, 6)
            seq += 1
        writer.begin_trial(
            trial.trial_id,
            cue - protocol.pre_s,
            cue + trial.duration_s + protocol.post_s,
        )
        while ts <= cue + trial.duration_s + protocol.post_s + 1e-9:
            assert frames_source.on_frame is not None
            frames_source.on_frame(make_frame(seq, ts))
            emitted.append((seq, ts))
            ts = round(ts + 0.1, 6)
            seq += 1
        writer.record_trial(
            TrialRecord(
                trial_id=trial.trial_id,
                class_name=trial.class_name,
                hand=trial.hand,
                target=trial.target,
                cue_ts=cue,
                start_ts=cue,
                end_ts=cue + trial.duration_s,
                status="bad" if index == 0 else "ok",
                note="operator marked bad" if index == 0 else "",
                distance_cm=trial.distance_cm,
            )
        )

    # Frames after the last window must not be written.
    while ts <= 115.0:
        assert frames_source.on_frame is not None
        frames_source.on_frame(make_frame(seq, ts))
        emitted.append((seq, ts))
        ts = round(ts + 0.1, 6)
        seq += 1

    expected_written = [
        (frame_seq, trial.trial_id)
        for trial, cue in zip(trials, cues, strict=True)
        for frame_seq, frame_ts in emitted
        if cue - protocol.pre_s <= frame_ts <= cue + trial.duration_s + protocol.post_s
    ]
    counters = writer.finalize(pairs_dropped=2)
    frames_source.stop()
    sensor_source.stop()

    assert counters.frames_written == len(expected_written) == 42
    assert counters.frames_dropped_queue == 0
    assert counters.pairs_dropped == 2
    assert counters.trials == {"ok": 1, "bad": 1}
    assert counters.imu_samples["left"][SampleKind.ACC.value] == sensor_count
    assert counters.imu_rate_hz["left"][SampleKind.ACC.value] == pytest.approx(50.0, rel=1e-3)

    session = Session.open(tmp_path / "sessions" / "20260101-000000_P01")
    assert session.schema_version == 1
    assert session.session_id == "20260101-000000_P01"
    assert session.participant_id == "P01"
    assert session.imu_enabled is True
    assert session.imu_left == "CPSA_L2"
    assert session.sync_max_skew_s == pytest.approx(0.030)
    assert session.calibration_path is None
    assert len(session.frames) == 42
    assert session.frames.seq.tolist() == [item[0] for item in expected_written]
    assert session.frames.trial_id.tolist() == [item[1] for item in expected_written]
    assert np.all(np.diff(session.frames.ts_rgb) > 0)
    for index in range(len(session.frames)):
        data = session.load_frame(index)
        assert data.rgb.shape == (6, 8, 3)
        assert data.rgb.dtype == np.uint8
        assert data.depth.shape == (6, 8)
        assert data.depth.dtype == np.uint16
        source = make_frame(session.frames.seq[index], session.frames.ts_rgb[index])
        assert np.array_equal(data.depth, source.depth)
        assert np.abs(data.rgb.astype(int) - source.rgb.astype(int)).mean() < 8

    assert len(session.trials) == 2
    assert [trial.trial_id for trial in session.trials] == [1, 2]
    assert [trial.status for trial in session.trials] == ["bad", "ok"]
    assert all(trial.distance_cm == 55 for trial in session.trials)

    assert session.imu.size == sensor_count * 6
    assert set(session.imu["side"].tolist()) == {"left", "right"}
    assert set(session.imu["kind"].tolist()) == {"acc", "gyr", "quat"}
    quaternions = session.imu[session.imu["kind"] == "quat"]
    assert np.all(quaternions["v3"] == 10000.0)
    assert np.isnan(session.imu[session.imu["kind"] == "acc"]["v3"]).all()

    meta = json.loads((session.path / "session.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert meta["protocol"] == {"sha256": DIGEST, "seed": 11}
    assert meta["counters"]["frames_written"] == 42
    assert meta["host"]["hostname"]
    assert "platform" in meta["host"]


def test_queue_overflow_drops_frames_and_counts_them(tmp_path: Path) -> None:
    clock = FakeClock(100.0)
    writer = new_writer(tmp_path / "sessions", clock, queue_size=1)
    writer.begin_trial(1, 0.0, 10.0)
    writer.write_frame(make_frame(0, 1.0))
    writer.write_frame(make_frame(1, 2.0))

    counters = writer.finalize()

    assert counters.frames_written == 1
    assert counters.frames_dropped_queue == 1
    session = Session.open(tmp_path / "sessions" / "20260101-000000_P01")
    assert len(session.frames) == 1
    assert session.frames.seq.tolist() == [0]


def test_no_imu_writes_header_only_and_records_disabled(tmp_path: Path) -> None:
    clock = FakeClock(100.0)
    writer = new_writer(
        tmp_path / "sessions",
        clock,
        imu=ImuInfo(enabled=False, left=None, right=None),
    )
    counters = writer.finalize()

    session = Session.open(tmp_path / "sessions" / "20260101-000000_P01")
    assert counters.imu_samples == {}
    assert counters.imu_rate_hz == {}
    assert session.imu_enabled is False
    assert session.imu_left is None
    assert session.imu_right is None
    assert session.imu.size == 0
    assert (session.path / "imu.csv").read_text(encoding="utf-8") == "ts,side,kind,v0,v1,v2,v3\n"


def test_calibration_is_copied_and_hashed(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"schema_version": 1}\n', encoding="utf-8")
    clock = FakeClock(100.0)
    writer = new_writer(tmp_path / "sessions", clock, calibration=calibration)
    writer.finalize()

    session = Session.open(tmp_path / "sessions" / "20260101-000000_P01")
    assert session.calibration_path is not None
    assert session.calibration_path.read_bytes() == calibration.read_bytes()
    meta = json.loads((session.path / "session.json").read_text(encoding="utf-8"))
    assert meta["calibration"]["file"] == "calibration.json"
    assert len(meta["calibration"]["sha256"]) == 64


def test_finalize_is_idempotent(tmp_path: Path) -> None:
    clock = FakeClock(100.0)
    writer = new_writer(tmp_path / "sessions", clock)
    first = writer.finalize()
    second = writer.finalize()
    assert first == second
    assert Session.open(tmp_path / "sessions" / "20260101-000000_P01").frames.seq.size == 0


def test_reader_errors_name_the_file_and_key(tmp_path: Path) -> None:
    clock = FakeClock(100.0)
    writer = new_writer(tmp_path / "sessions", clock)
    writer.finalize()
    session_path = tmp_path / "sessions" / "20260101-000000_P01"

    with pytest.raises(SessionFormatError, match="schema_version"):
        meta = json.loads((session_path / "session.json").read_text(encoding="utf-8"))
        meta["schema_version"] = 2
        (session_path / "session.json").write_text(json.dumps(meta), encoding="utf-8")
        Session.open(session_path)

    meta["schema_version"] = 1
    (session_path / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    (session_path / "frames.csv").write_text("seq,trial_id\n", encoding="utf-8")
    with pytest.raises(SessionFormatError, match="frames.csv"):
        Session.open(session_path)
