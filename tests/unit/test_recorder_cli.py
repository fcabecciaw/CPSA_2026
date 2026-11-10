"""Recorder CLI helpers: participant validation, `--no-imu`, session ids."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from cpsa.application.config import load
from cpsa.recorder.__main__ import (
    build_parser,
    camera_info,
    create_sensor_source,
    imu_info,
    pairs_dropped,
    session_id_for,
    validate_participant,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "config" / "board.yaml"


def test_session_id_matches_the_schema() -> None:
    created = datetime(2026, 11, 6, 10, 30, 5)
    assert session_id_for(created, "P01") == "20261106-103005_P01"


@pytest.mark.parametrize("participant", ["P01", "anon-01", "A_B_9", "x" * 32])
def test_valid_participant_ids_are_accepted(participant: str) -> None:
    assert validate_participant(participant) == participant


@pytest.mark.parametrize("participant", ["", "P 01", "P01/../x", "x" * 33, "P01\n"])
def test_invalid_participant_ids_are_rejected(participant: str) -> None:
    with pytest.raises(ValueError, match="participant id"):
        validate_participant(participant)


def test_no_imu_avoids_constructing_the_bluecoin_source() -> None:
    config = load(SHIPPED)
    assert create_sensor_source(config, no_imu=True) is None
    info = imu_info(config, no_imu=True)
    assert info.enabled is False
    assert info.left is None
    assert info.right is None


def test_camera_and_imu_info_copy_the_configuration() -> None:
    config = load(SHIPPED)
    camera = camera_info(config)
    assert camera.rgb_device == config.camera.rgb_device
    assert camera.xu.laser_power == config.camera.xu.laser_power
    info = imu_info(config, no_imu=False)
    assert info.enabled is True
    assert info.left == config.bluecoins.left
    assert info.max_skew_ms == config.sync.max_skew_ms


def test_record_arguments_parse() -> None:
    args = build_parser().parse_args(
        [
            "record",
            "--config",
            "config/board.yaml",
            "--protocol",
            "config/protocol.yaml",
            "--participant",
            "P01",
            "--out",
            "/tmp/sessions",
            "--no-imu",
            "--queue",
            "12",
        ]
    )
    assert args.command == "record"
    assert args.no_imu is True
    assert args.queue == 12
    assert args.participant == "P01"


class _Stats:
    dropped_rgb = 3
    dropped_depth = 4


class _CameraWithStats:
    stats = _Stats()


class _CameraWithoutStats:
    pass


def test_pairs_dropped_uses_the_optional_adapter_counters() -> None:
    assert pairs_dropped(_CameraWithStats()) == 7  # type: ignore[arg-type]
    assert pairs_dropped(_CameraWithoutStats()) is None  # type: ignore[arg-type]
