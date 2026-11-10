"""FileEvidenceSink contract: clip, keyframe and retention (FR-021, NFR-005)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from cpsa.adapters.evidence import FileEvidenceSink
from tests.unit import vision_fixtures as fx


def _frame(index: int) -> np.ndarray:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 0] = (index * 40) % 256
    image[:, :, 1] = 128
    image[:, :, 2] = 200
    return image


def _tracked():
    return fx.make_frame(0, 1.0, fx.neutral_xy())


def test_write_episode_creates_keyframe_and_clip(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    sink = FileEvidenceSink(tmp_path / "evidence", 7, fx.LAYOUT)
    frames = [(_frame(index), _tracked()) for index in range(3)]

    sink.write_episode("20260101T000000-0001", frames, 10.0, keyframe=1)

    folder = sink.episode_folder("20260101T000000-0001")
    assert folder == tmp_path / "evidence" / date.today().isoformat() / "20260101T000000-0001"
    keyframe = folder / "keyframe.jpg"
    assert keyframe.is_file() and keyframe.stat().st_size > 0
    clip = folder / "clip.mp4"
    if not clip.is_file():
        clip = folder / "clip.avi"
    assert clip.is_file() and clip.stat().st_size > 0


def test_keyframe_scales_with_the_requested_index(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    import cv2

    sink = FileEvidenceSink(tmp_path / "evidence", 7, None)
    dark = (_frame(0), None)
    bright = (np.full((48, 64, 3), 255, dtype=np.uint8), None)

    sink.write_episode("dark", [dark, bright], 10.0, keyframe=0)
    sink.write_episode("bright", [dark, bright], 10.0, keyframe=1)

    first = cv2.imread(str(sink.episode_folder("dark") / "keyframe.jpg"))
    second = cv2.imread(str(sink.episode_folder("bright") / "keyframe.jpg"))
    assert first is not None and second is not None
    assert first.mean() < second.mean()


def test_write_episode_without_a_layout_still_draws_keypoints(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    sink = FileEvidenceSink(tmp_path / "evidence", 0, None)

    sink.write_episode("ep", [(_frame(0), _tracked())], 10.0, 0)

    assert (sink.episode_folder("ep") / "keyframe.jpg").is_file()


def test_purge_removes_day_folders_older_than_the_retention(tmp_path: Path) -> None:
    base = tmp_path / "evidence"
    today = date.today()
    old = base / (today - timedelta(days=9)).isoformat() / "ep"
    yesterday = base / (today - timedelta(days=1)).isoformat() / "ep"
    keep = base / today.isoformat() / "ep"
    other = base / "not-a-day" / "ep"
    for folder in (old, yesterday, keep, other):
        folder.mkdir(parents=True)
        (folder / "clip.mp4").write_bytes(b"x")

    FileEvidenceSink(base, 7).purge(datetime.now().astimezone())

    assert not old.exists()
    assert yesterday.exists() and keep.exists() and other.exists()

    FileEvidenceSink(base, 0).purge(datetime.now().astimezone())
    assert not yesterday.exists()
    assert keep.exists() and other.exists()


def test_purge_of_a_missing_base_is_a_no_op(tmp_path: Path) -> None:
    FileEvidenceSink(tmp_path / "missing", 1).purge(datetime.now().astimezone())


@pytest.mark.parametrize("episode_id", ["", "a/b", "a\\b"])
def test_invalid_episode_id_is_rejected(tmp_path: Path, episode_id: str) -> None:
    sink = FileEvidenceSink(tmp_path / "evidence", 1)

    with pytest.raises(ValueError):
        sink.write_episode(episode_id, [(_frame(0), None)], 10.0, 0)


def test_invalid_frames_fps_and_keyframe_are_rejected(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    sink = FileEvidenceSink(tmp_path / "evidence", 1)
    image = _frame(0)

    with pytest.raises(ValueError):
        sink.write_episode("ep", [], 10.0, 0)
    with pytest.raises(ValueError):
        sink.write_episode("ep", [(image, None)], 0.0, 0)
    with pytest.raises(ValueError):
        sink.write_episode("ep", [(image, None)], 10.0, 1)
    with pytest.raises(ValueError):
        sink.write_episode("ep", [(image, None), (image[:10], None)], 10.0, 0)


def test_negative_retention_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FileEvidenceSink(tmp_path, -1)
