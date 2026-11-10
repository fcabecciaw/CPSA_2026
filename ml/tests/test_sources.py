"""Tests for the COCO image/annotation source locators (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from cpsa_ml.datasets.sources import (
    WHOLEBODY_DESCRIPTION,
    WHOLEBODY_FILENAMES,
    WHOLEBODY_GDRIVE_IDS,
    find_coco_root,
    wholebody_annotations,
)


def _write_annotations(path: Path, description: str = WHOLEBODY_DESCRIPTION) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "info": {"description": description, "year": 2017},
                "images": [{"id": 1}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _no_download(file_id: str, destination: Path) -> None:  # pragma: no cover - asserted
    raise AssertionError(f"unexpected download of {file_id} to {destination}")


def test_find_coco_root_returns_the_directory_with_both_image_trees(tmp_path: Path) -> None:
    attached = tmp_path / "input" / "coco-2017-dataset.1" / "coco2017"
    (attached / "train2017").mkdir(parents=True)
    (attached / "val2017").mkdir()
    partial = tmp_path / "input" / "partial" / "val2017"
    partial.mkdir(parents=True)

    assert find_coco_root(tmp_path / "input") == attached


def test_find_coco_root_requires_the_image_trees(tmp_path: Path) -> None:
    (tmp_path / "input" / "something" / "val2017").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="train2017"):
        find_coco_root(tmp_path / "input")
    with pytest.raises(FileNotFoundError, match="train2017"):
        find_coco_root(tmp_path / "missing")


def test_wholebody_annotations_prefers_attached_files(tmp_path: Path) -> None:
    attached = {
        split: _write_annotations(tmp_path / "input" / "wholebody" / name)
        for split, name in WHOLEBODY_FILENAMES.items()
    }

    annotations = wholebody_annotations(
        tmp_path / "input", tmp_path / "cache", download=_no_download
    )

    assert annotations == attached
    assert not (tmp_path / "cache").exists()


def test_wholebody_annotations_downloads_only_missing_splits(tmp_path: Path) -> None:
    attached_train = _write_annotations(
        tmp_path / "input" / "wholebody" / WHOLEBODY_FILENAMES["train"]
    )
    calls: list[tuple[str, Path]] = []

    def download(file_id: str, destination: Path) -> None:
        calls.append((file_id, destination))
        _write_annotations(destination)

    annotations = wholebody_annotations(tmp_path / "input", tmp_path / "cache", download=download)

    assert annotations["train"] == attached_train
    assert annotations["val"] == tmp_path / "cache" / WHOLEBODY_FILENAMES["val"]
    assert calls == [(WHOLEBODY_GDRIVE_IDS["val"], annotations["val"])]
    assert annotations["val"].is_file()


def test_wholebody_annotations_reuses_the_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def download(file_id: str, destination: Path) -> None:
        calls.append(file_id)
        _write_annotations(destination)

    first = wholebody_annotations(tmp_path / "input", tmp_path / "cache", download=download)
    second = wholebody_annotations(tmp_path / "input", tmp_path / "cache", download=download)

    assert first == second
    assert calls == [WHOLEBODY_GDRIVE_IDS["train"], WHOLEBODY_GDRIVE_IDS["val"]]


def test_wholebody_annotations_validates_the_description(tmp_path: Path) -> None:
    _write_annotations(
        tmp_path / "input" / "wholebody" / WHOLEBODY_FILENAMES["train"], description="COCO"
    )
    _write_annotations(tmp_path / "input" / "wholebody" / WHOLEBODY_FILENAMES["val"])

    with pytest.raises(ValueError, match="not a COCO-WholeBody annotation"):
        wholebody_annotations(tmp_path / "input", tmp_path / "cache", download=_no_download)


def test_wholebody_annotations_validates_downloaded_files(tmp_path: Path) -> None:
    def download(file_id: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('{"info": {"description": "COCO"}}', encoding="utf-8")

    with pytest.raises(ValueError, match="not a COCO-WholeBody annotation"):
        wholebody_annotations(tmp_path / "input", tmp_path / "cache", download=download)


def test_wholebody_annotations_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "input" / "wholebody" / WHOLEBODY_FILENAMES["train"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        wholebody_annotations(tmp_path / "input", tmp_path / "cache", download=_no_download)


def test_wholebody_annotations_reads_only_the_prefix_of_a_large_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cpsa_ml.datasets import sources

    monkeypatch.setattr(sources, "_INFO_PREFIX_BYTES", 64)
    path = tmp_path / "input" / "wholebody" / WHOLEBODY_FILENAMES["train"]
    path.parent.mkdir(parents=True)
    # Truncated JSON: the prefix holds `info`, the rest is unreadable filler.
    path.write_text(
        '{"info": {"description": "COCO-WholeBody"}, "images": [',
        encoding="utf-8",
    )
    (tmp_path / "input" / "wholebody" / WHOLEBODY_FILENAMES["val"]).write_text(
        '{"info": {"description": "COCO-WholeBody"}, "images": []}', encoding="utf-8"
    )

    annotations = wholebody_annotations(
        tmp_path / "input", tmp_path / "cache", download=_no_download
    )
    assert annotations["train"] == path


def test_download_with_gdown_uses_the_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cpsa_ml.datasets import sources

    fake = ModuleType("gdown")
    calls: list[tuple[str, str]] = []

    def download(*, id: str, output: str, quiet: bool) -> str:
        calls.append((id, output))
        Path(output).write_text(
            json.dumps({"info": {"description": WHOLEBODY_DESCRIPTION}}), encoding="utf-8"
        )
        return output

    fake.download = download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gdown", fake)

    annotations = sources.wholebody_annotations(tmp_path / "input", tmp_path / "cache")

    assert calls == [
        (WHOLEBODY_GDRIVE_IDS["train"], str(annotations["train"])),
        (WHOLEBODY_GDRIVE_IDS["val"], str(annotations["val"])),
    ]
    assert annotations["train"].parent == tmp_path / "cache"


def test_download_with_gdown_explains_the_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cpsa_ml.datasets import sources

    monkeypatch.setitem(sys.modules, "gdown", None)

    with pytest.raises(RuntimeError, match=r"cpsa-ml\[sources\]"):
        sources.wholebody_annotations(tmp_path / "input", tmp_path / "cache")
