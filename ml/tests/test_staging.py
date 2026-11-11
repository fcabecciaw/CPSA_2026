"""Tests for the staging helpers that run on Kaggle (no Kaggle, no network)."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Any

import pytest

from cpsa_ml.datasets import staging
from cpsa_ml.datasets.staging import load_manifest, stage_archive, stage_dataset


def _dataset(root: Path) -> Path:
    """Build a miniature ``halfbust62`` tree and return its root."""

    for folder in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / folder).mkdir(parents=True)
    (root / "data.yaml").write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    for name in ("a.jpg", "b.jpg"):
        (root / "images/train" / name).write_bytes(name.encode())
        label = root / "labels/train" / f"{Path(name).stem}.txt"
        label.write_text("0 0 0 0 0\n", encoding="utf-8")
    (root / "images/val" / "c.jpg").write_bytes(b"c")
    (root / "labels/val" / "c.txt").write_text("0 0 0 0 0\n", encoding="utf-8")
    return root


def _files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != staging.MARKER
    )


def test_listing_never_stats_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``stage_dataset`` lists with ``os.walk``, so no per-file ``stat`` is issued."""

    source = _dataset(tmp_path / "source")
    target = tmp_path / "target"
    listing = {"active": False}
    real_is_file = Path.is_file
    real_path_stat = Path.stat
    real_os_stat = os.stat

    def guarded_is_file(self: Path) -> bool:
        if listing["active"]:
            raise AssertionError("Path.is_file called while listing")
        return real_is_file(self)

    def guarded_path_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if listing["active"]:
            raise AssertionError("Path.stat called while listing")
        return real_path_stat(self, *args, **kwargs)

    def guarded_os_stat(*args: object, **kwargs: object) -> os.stat_result:
        if listing["active"]:
            raise AssertionError("os.stat called while listing")
        return real_os_stat(*args, **kwargs)

    real_list = staging._list_files

    def guarded_list(root: Path) -> list[str]:
        listing["active"] = True
        try:
            return real_list(root)
        finally:
            listing["active"] = False

    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    monkeypatch.setattr(Path, "stat", guarded_path_stat)
    monkeypatch.setattr(os, "stat", guarded_os_stat)
    monkeypatch.setattr(staging, "_list_files", guarded_list)

    staged = stage_dataset(source, target)

    assert listing["active"] is False
    assert staged == target
    assert _files(target) == _files(source)
    assert (target / staging.MARKER).is_file()


def test_stage_dataset_copies_with_plain_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copying never goes through ``shutil.copyfile``, which stats both ends."""

    source = _dataset(tmp_path / "source")

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("shutil.copyfile stats the source and the target twice")

    monkeypatch.setattr(staging.shutil, "copyfile", fail)

    target = stage_dataset(source, tmp_path / "target")

    assert (target / "images/train/a.jpg").read_bytes() == b"a.jpg"
    assert (target / "labels/val/c.txt").read_text(encoding="utf-8") == "0 0 0 0 0\n"


def test_stage_dataset_copies_the_included_paths_and_data_yaml(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "source")
    target = tmp_path / "target"

    stage_dataset(
        source,
        target,
        include=["images/train/a.jpg", "labels/train/a.txt"],
    )

    assert _files(target) == ["data.yaml", "images/train/a.jpg", "labels/train/a.txt"]
    assert (target / "images/train/a.jpg").read_bytes() == b"a.jpg"


def test_stage_dataset_rejects_a_missing_included_path(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "source")

    with pytest.raises(FileNotFoundError) as excinfo:
        stage_dataset(
            source,
            tmp_path / "target",
            include=["images/train/a.jpg", "images/train/zzz.jpg"],
        )

    message = str(excinfo.value)
    assert "1 included paths are missing" in message
    assert "images/train/zzz.jpg" in message
    assert not (tmp_path / "target").exists()


def test_stage_dataset_reuses_a_finished_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _dataset(tmp_path / "source")
    target = stage_dataset(source, tmp_path / "target")
    (source / "images/train/a.jpg").unlink()

    assert stage_dataset(source, target) == target
    assert "reusing the local copy" in capsys.readouterr().out
    assert (target / "images/train/a.jpg").is_file()


@pytest.mark.parametrize("wrapped", [True, False], ids=["top-level-folder", "flat"])
def test_stage_archive_returns_the_folder_that_holds_data_yaml(
    tmp_path: Path, wrapped: bool
) -> None:
    build = _dataset(tmp_path / "build")
    archive = tmp_path / "hagrid62.tar"
    with tarfile.open(archive, "w") as tar:
        if wrapped:
            tar.add(build, arcname="hagrid62")
        else:
            for path in sorted(build.rglob("*")):
                tar.add(path, arcname=path.relative_to(build).as_posix())

    root = stage_archive(archive, tmp_path / "extracted")

    assert root.name == ("hagrid62" if wrapped else "extracted")
    assert (root / "data.yaml").is_file()
    assert (root / "images/train/a.jpg").read_bytes() == b"a.jpg"
    assert (root / staging.MARKER).is_file()

    # The marker is the reuse contract: the extraction must not need the archive.
    archive.unlink()
    assert stage_archive(archive, tmp_path / "extracted") == root


def test_stage_archive_rejects_a_traversal_member(tmp_path: Path) -> None:
    """A member that escapes the target folder fails instead of being written."""

    archive = tmp_path / "traversal.tar"
    with tarfile.open(archive, "w") as tar:
        payload = b"escape"
        member = tarfile.TarInfo("../x")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))

    with pytest.raises(tarfile.TarError):
        stage_archive(archive, tmp_path / "extracted")

    assert not (tmp_path / "x").exists()
    assert not (tmp_path / "extracted" / staging.MARKER).exists()


def test_stage_archive_opens_the_archive_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headers are scanned while extracting, not by a second full pass."""

    build = _dataset(tmp_path / "build")
    archive = tmp_path / "hagrid62.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(build, arcname="hagrid62")
    opens: list[tuple[object, ...]] = []
    real_open = tarfile.open

    def counting_open(*args: Any, **kwargs: Any) -> tarfile.TarFile:
        opens.append(args)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(staging.tarfile, "open", counting_open)

    root = stage_archive(archive, tmp_path / "extracted")

    assert len(opens) == 1
    assert (root / "images/train/a.jpg").read_bytes() == b"a.jpg"


def test_stage_archive_refuses_without_the_pep706_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``tarfile.data_filter`` the archive is not extracted unchecked."""

    build = _dataset(tmp_path / "build")
    archive = tmp_path / "hagrid62.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(build, arcname="hagrid62")
    monkeypatch.delattr(staging.tarfile, "data_filter")

    with pytest.raises(RuntimeError, match="tarfile.data_filter"):
        stage_archive(archive, tmp_path / "extracted")

    assert not (tmp_path / "extracted" / "hagrid62").exists()


def test_stage_archive_rejects_an_archive_without_a_dataset(tmp_path: Path) -> None:
    archive = tmp_path / "empty.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(_dataset(tmp_path / "build") / "images", arcname="images")

    with pytest.raises(FileNotFoundError, match="holds no data.yaml"):
        stage_archive(archive, tmp_path / "extracted")


def test_load_manifest_returns_sorted_paths_with_data_yaml() -> None:
    entries = load_manifest("halfbust62_h256")

    assert entries == sorted(entries)
    assert "data.yaml" in entries
    assert all(not path.startswith("/") for path in entries)
    assert "images/train" in "".join(entries)
    assert any(path.startswith("labels/val/") for path in entries)


def test_load_manifest_rejects_an_unknown_manifest() -> None:
    with pytest.raises(FileNotFoundError, match="no staging manifest named 'missing'"):
        load_manifest("missing")
