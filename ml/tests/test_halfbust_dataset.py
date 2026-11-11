"""Tests for the COCO-WholeBody -> YOLO-pose half-bust builder."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cpsa_ml.datasets.halfbust import BuildConfig, build_halfbust_dataset, main
from cpsa_ml.schema import HALFBUST_62


def _build(
    tmp_path: Path,
    synthetic_coco: tuple[Path, Path],
    *,
    schema: str = "halfbust62",
    split: str = "train",
    occlusion_p: float = 0.0,
    max_images: int | None = 3,
    seed: int = 0,
) -> tuple[Path, object]:
    coco_root, wholebody_json = synthetic_coco
    out = tmp_path / f"out-{schema}-{split}"
    report = build_halfbust_dataset(
        BuildConfig(
            coco_root=coco_root,
            wholebody_json=wholebody_json,
            split=split,
            schema=schema,
            out=out,
            max_images=max_images,
            occlusion_p=occlusion_p,
            seed=seed,
        )
    )
    return out, report


def test_builds_halfbust_dataset(tmp_path: Path, synthetic_coco: tuple[Path, Path]) -> None:
    out, report = _build(tmp_path, synthetic_coco, occlusion_p=1.0)
    assert report.images_written == 4
    assert report.labels_written == 4
    assert report.annotations_skipped == 0
    assert report.occlusions >= 1
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 4
    assert len(list((out / "labels" / "train").glob("*.txt"))) == 4

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["kpt_shape"] == [62, 3]
    assert data["flip_idx"] == list(HALFBUST_62.flip_idx)
    assert data["kpt_oks_sigmas"] == list(HALFBUST_62.oks_sigmas)
    assert data["names"] == {0: "person"}
    assert data["train"] == "images/train"
    assert data["val"] == "images/train"  # val not built yet
    assert "path" not in data  # portable across /kaggle/input mounts

    schema = yaml.safe_load((out / "schema.json").read_text())
    assert schema["name"] == "halfbust62"
    assert len(schema["flip_idx"]) == 62


def test_labels_have_normalized_keypoints(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    out, _ = _build(tmp_path, synthetic_coco)
    lines = [
        line
        for path in (out / "labels" / "train").glob("*.txt")
        for line in path.read_text().splitlines()
    ]
    assert len(lines) == 4
    expected_tokens = 5 + 62 * 3
    for line in lines:
        tokens = line.split()
        assert len(tokens) == expected_tokens
        assert tokens[0] == "0"
        assert float(tokens[1]) == 0.5
        assert float(tokens[2]) == 0.5
        assert float(tokens[3]) == 1.0
        assert float(tokens[4]) == 1.0
        for index in range(62):
            x, y, visibility = tokens[5 + index * 3 : 5 + index * 3 + 3]
            assert visibility in {"0", "1"}
            if visibility == "0":
                assert float(x) == 0.0
                assert float(y) == 0.0
            else:
                assert 0.0 <= float(x) <= 1.0
                assert 0.0 <= float(y) <= 1.0


def test_keypoint_outside_crop_is_dropped(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    out, _ = _build(tmp_path, synthetic_coco)
    label = next((out / "labels" / "train").glob("*_a4.txt")).read_text().split()
    # Schema position 9 is left_wrist; the fixture puts it outside the crop.
    visibility_index = 5 + 9 * 3 + 2
    assert label[visibility_index] == "0"
    assert label[visibility_index - 2] == "0.0"
    assert label[visibility_index - 1] == "0.0"


def test_coco17_schema_drops_below_hip_keypoints(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    out, _ = _build(tmp_path, synthetic_coco, schema="coco17")
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["kpt_shape"] == [17, 3]
    assert len(data["flip_idx"]) == 17
    label = next((out / "labels" / "train").glob("*_a1.txt")).read_text().split()
    assert len(label) == 5 + 17 * 3
    left_hip = 5 + 11 * 3 + 2
    left_knee = 5 + 13 * 3 + 2
    assert label[left_hip] == "1"
    assert label[left_knee] == "0"


def test_build_train_then_val_updates_yaml(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    out, _ = _build(tmp_path, synthetic_coco)
    build_halfbust_dataset(
        BuildConfig(
            coco_root=synthetic_coco[0],
            wholebody_json=synthetic_coco[1],
            split="val",
            schema="halfbust62",
            out=out,
            max_images=3,
        )
    )
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    stats = yaml.safe_load((out / "stats.json").read_text())
    assert set(stats["splits"]) == {"train", "val"}


def test_seed_makes_occlusion_deterministic(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    first, _ = _build(tmp_path / "first", synthetic_coco, occlusion_p=1.0, seed=7)
    second, _ = _build(tmp_path / "second", synthetic_coco, occlusion_p=1.0, seed=7)
    first_labels = {
        path.name: path.read_text() for path in (first / "labels" / "train").glob("*.txt")
    }
    second_labels = {
        path.name: path.read_text() for path in (second / "labels" / "train").glob("*.txt")
    }
    assert first_labels == second_labels


def test_invalid_split_rejected(tmp_path: Path, synthetic_coco: tuple[Path, Path]) -> None:
    with pytest.raises(ValueError, match="split"):
        BuildConfig(
            coco_root=synthetic_coco[0],
            wholebody_json=synthetic_coco[1],
            split="test",
            out=tmp_path / "out",
        )


def test_cli_builds_dataset(
    tmp_path: Path, synthetic_coco: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cli"
    code = main(
        [
            "--coco-root",
            str(synthetic_coco[0]),
            "--wholebody-json",
            str(synthetic_coco[1]),
            "--split",
            "train",
            "--schema",
            "coco17",
            "--out",
            str(out),
            "--max-images",
            "1",
            "--occlusion-p",
            "0.5",
            "--seed",
            "1",
        ]
    )
    assert code == 0
    assert (out / "data.yaml").is_file()
    assert "images_written" in capsys.readouterr().out
