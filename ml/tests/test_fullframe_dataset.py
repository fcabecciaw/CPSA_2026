"""Tests for the COCO-WholeBody -> YOLO-pose full-frame builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from cpsa_ml.datasets import fullframe
from cpsa_ml.datasets.fullframe import (
    FULL_FRACTION,
    FullFrameConfig,
    build_fullframe_dataset,
    full_frame_target_count,
    half_train_count,
    main,
    render_labeled_preview,
    resolve_target_count,
)
from cpsa_ml.schema import HALFBUST_62

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def _build(
    tmp_path: Path,
    synthetic_coco: tuple[Path, Path],
    *,
    split: str = "train",
    max_images: int | None = None,
    target_count: int | None = None,
    seed: int = 0,
    min_visible: int = 3,
    name: str = "out",
) -> tuple[Path, object]:
    coco_root, wholebody_json = synthetic_coco
    out = tmp_path / name
    report = build_fullframe_dataset(
        FullFrameConfig(
            coco_root=coco_root,
            wholebody_json=wholebody_json,
            split=split,
            out=out,
            max_images=max_images,
            target_count=target_count,
            seed=seed,
            min_visible=min_visible,
        )
    )
    return out, report


def _rewrite_json(source: Path, target: Path, mutate) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _blank_person(annotation: dict) -> None:
    """Make a person unlabelable: no keypoint is visible."""

    for key, count in (
        ("keypoints", 17),
        ("foot_kpts", 6),
        ("face_kpts", 68),
        ("lefthand_kpts", 21),
        ("righthand_kpts", 21),
    ):
        annotation[key] = [0.0] * (count * 3)
    for flag in ("foot_valid", "face_valid", "lefthand_valid", "righthand_valid"):
        annotation[flag] = False


def test_writes_the_original_image_and_the_coco_person_box(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    coco_root, _ = synthetic_coco
    out, report = _build(tmp_path, synthetic_coco)

    assert report.images_written == 3
    # Image 1 has two persons, the other two images one each.
    assert report.labels_written == 4
    assert report.persons_labeled == 4
    assert report.qualifying_images == 3
    assert report.target_count is None

    for written in (out / "images" / "train").glob("*.jpg"):
        source = coco_root / "train2017" / written.name.split("_", 1)[1]
        assert written.read_bytes() == source.read_bytes(), "full frames must be untouched copies"

    line = next((out / "labels" / "train").glob("*000000000001.txt")).read_text().split()
    # The fixture bbox is [230, 110, 180, 380] on a 640x480 image; the bottom
    # 10 px fall outside the frame and are clipped.
    assert line[0] == "0"
    assert [float(value) for value in line[1:5]] == pytest.approx(
        [0.5, 295 / 480, 180 / 640, 370 / 480]
    )
    # Schema position 1 is left_eye at (310, 145): normalized to the image.
    assert [float(value) for value in line[8:10]] == pytest.approx(
        [310 / IMAGE_WIDTH, 145 / IMAGE_HEIGHT], abs=1e-5
    )
    assert line[10] == "1"
    # Schema position 9 (left_wrist) is the fixture's (260, 300).
    assert [float(value) for value in line[5 + 9 * 3 : 5 + 9 * 3 + 2]] == pytest.approx(
        [260 / IMAGE_WIDTH, 300 / IMAGE_HEIGHT], abs=1e-5
    )
    # Two persons in image 1 -> two label lines.
    lines = (out / "labels" / "train" / "000000000001_000000000001.txt").read_text()
    assert len(lines.splitlines()) == 2


def test_label_uses_the_true_box_not_the_keypoint_extent(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    out, _ = _build(tmp_path, synthetic_coco)
    values = next((out / "labels" / "train").glob("*000000000003.txt")).read_text().split()
    box = [float(value) for value in values[1:5]]
    # Annotations can extend past the frame (the fixture's third image does);
    # the label box is clipped to the image and stays the true person extent.
    assert box[2] == pytest.approx(400 / IMAGE_WIDTH)
    assert box[3] == pytest.approx(370 / IMAGE_HEIGHT)
    # Knees and ankles are outside HALFBUST_62, so the person box is taller
    # than the labelled keypoint extent -- that is the point of this dataset.
    keypoint_rows = [
        (float(values[5 + index * 3]), float(values[6 + index * 3]))
        for index in range(62)
        if values[7 + index * 3] == "1"
    ]
    labelled_height = max(y for _, y in keypoint_rows) - min(y for _, y in keypoint_rows)
    assert box[3] > labelled_height


def test_person_without_visible_keypoints_gets_no_line(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    _, json_path = synthetic_coco

    def mutate(payload: dict) -> None:
        _blank_person(payload["annotations"][1])  # second person of image 1

    annotated = _rewrite_json(json_path, tmp_path / "one_blank.json", mutate)
    out = tmp_path / "blank"
    report = build_fullframe_dataset(
        FullFrameConfig(
            coco_root=synthetic_coco[0],
            wholebody_json=annotated,
            split="train",
            out=out,
        )
    )

    assert report.images_written == 3
    assert report.persons_labeled == 3
    assert report.persons_unlabeled == 1
    lines = (out / "labels" / "train").glob("*.txt")
    assert sum(len(path.read_text().splitlines()) for path in lines) == 3


def test_image_whose_only_person_is_unlabelable_is_skipped(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    _, json_path = synthetic_coco

    def mutate(payload: dict) -> None:
        _blank_person(payload["annotations"][2])  # the only person of image 2

    annotated = _rewrite_json(json_path, tmp_path / "one_empty.json", mutate)
    out = tmp_path / "empty"
    report = build_fullframe_dataset(
        FullFrameConfig(
            coco_root=synthetic_coco[0],
            wholebody_json=annotated,
            split="train",
            out=out,
        )
    )

    assert report.images_written == 2
    assert report.images_skipped_no_label == 1
    assert not list((out / "labels" / "train").glob("*000000000002*"))


def test_crowd_images_are_skipped_entirely(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    """A crowd region hides visible people that a label line cannot ignore."""

    _, json_path = synthetic_coco

    def mutate(payload: dict) -> None:
        payload["annotations"][2]["iscrowd"] = 1

    annotated = _rewrite_json(json_path, tmp_path / "crowd.json", mutate)
    out = tmp_path / "crowd"
    report = build_fullframe_dataset(
        FullFrameConfig(
            coco_root=synthetic_coco[0],
            wholebody_json=annotated,
            split="train",
            out=out,
        )
    )

    assert report.crowd_annotations == 1
    assert report.images_skipped_crowd == 1
    assert report.images_written == 2
    assert not list((out / "images" / "train").glob("*000000000002*"))


def test_min_visible_threshold_is_configurable(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    _, report = _build(tmp_path, synthetic_coco, min_visible=7)
    # Body 0-11 gives 12 points but the fixture also has face and hand points,
    # so only an impossible threshold drops everyone.
    assert report.persons_labeled == 4

    _, strict = _build(tmp_path, synthetic_coco, min_visible=63, name="strict")
    assert strict.persons_labeled == 0
    assert strict.images_written == 0


def test_sampling_is_deterministic_and_seeded(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    first, first_report = _build(tmp_path, synthetic_coco, target_count=1, seed=5, name="first")
    second, _ = _build(tmp_path, synthetic_coco, target_count=1, seed=5, name="second")
    assert first_report.images_written == 1
    assert first_report.qualifying_images == 3
    assert [path.name for path in (first / "labels" / "train").glob("*.txt")] == [
        path.name for path in (second / "labels" / "train").glob("*.txt")
    ]

    maxed, maxed_report = _build(tmp_path, synthetic_coco, max_images=2, name="maxed")
    assert maxed_report.images_written == 2
    assert len(list((maxed / "images" / "train").glob("*.jpg"))) == 2


def test_val_split_writes_every_qualifying_image(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    out, report = _build(tmp_path, synthetic_coco, split="val", target_count=1)

    assert report.target_count == 1
    assert report.images_written == 3
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["val"] == "images/val"
    assert data["train"] == "images/val"  # train not built in this root
    assert data["kpt_shape"] == [62, 3]
    assert data["flip_idx"] == list(HALFBUST_62.flip_idx)
    assert data["kpt_oks_sigmas"] == list(HALFBUST_62.oks_sigmas)
    stats = yaml.safe_load((out / "stats.json").read_text())
    assert stats["splits"]["val"]["images_written"] == 3


def test_train_then_val_updates_the_yaml(tmp_path: Path, synthetic_coco: tuple[Path, Path]) -> None:
    out, _ = _build(tmp_path, synthetic_coco)
    build_fullframe_dataset(
        FullFrameConfig(
            coco_root=synthetic_coco[0],
            wholebody_json=synthetic_coco[1],
            split="val",
            out=out,
        )
    )
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    stats = yaml.safe_load((out / "stats.json").read_text())
    assert set(stats["splits"]) == {"train", "val"}


@pytest.mark.parametrize(
    ("n_half", "fraction", "expected"),
    [
        (0, 0.25, 0),
        (100, 0.25, 33),
        (300, 0.25, 100),
        (100, 0.5, 100),
        (90, 0.25, 30),
    ],
)
def test_full_frame_target_count(n_half: int, fraction: float, expected: int) -> None:
    assert full_frame_target_count(n_half, fraction) == expected


def test_full_frame_target_count_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="n_half"):
        full_frame_target_count(-1)
    for fraction in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            full_frame_target_count(10, fraction)


def test_resolve_target_count_takes_the_smallest_cap(tmp_path: Path) -> None:
    base = {
        "coco_root": tmp_path,
        "wholebody_json": tmp_path / "wb.json",
        "split": "train",
        "out": tmp_path / "out",
    }
    assert resolve_target_count(FullFrameConfig(**base)) is None
    assert resolve_target_count(FullFrameConfig(**base, target_count=50)) == 50
    assert (
        resolve_target_count(FullFrameConfig(**base, target_count=50, half_train_count=100)) == 33
    )
    assert (
        resolve_target_count(
            FullFrameConfig(**base, target_count=40, max_images=10, half_train_count=100)
        )
        == 10
    )


def test_config_rejects_invalid_values(tmp_path: Path) -> None:
    base = {
        "coco_root": tmp_path,
        "wholebody_json": tmp_path / "wb.json",
        "split": "test",
        "out": tmp_path / "out",
    }
    with pytest.raises(ValueError, match="split"):
        FullFrameConfig(**base)
    base["split"] = "train"
    for overrides, message in (
        ({"fraction": 0.0}, "fraction"),
        ({"target_count": -1}, "target_count"),
        ({"max_images": -2}, "max_images"),
        ({"min_visible": -1}, "min_visible"),
        ({"jpeg_quality": 0}, "jpeg_quality"),
    ):
        with pytest.raises(ValueError, match=message):
            FullFrameConfig(**base, **overrides)


def test_half_train_count_reads_the_stats_file(tmp_path: Path) -> None:
    root = tmp_path / "halfbust62"
    root.mkdir()
    (root / "stats.json").write_text(
        json.dumps({"splits": {"train": {"images_written": 21_345}}}), encoding="utf-8"
    )
    assert half_train_count(root) == 21_345
    assert half_train_count(root / "stats.json") == 21_345
    assert full_frame_target_count(half_train_count(root)) == 7115

    with pytest.raises(FileNotFoundError, match="stats.json"):
        half_train_count(tmp_path / "absent")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "stats.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="splits.train"):
        half_train_count(broken)


def test_half_train_count_drives_the_builder_count(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    root = tmp_path / "halfbust62"
    root.mkdir()
    (root / "stats.json").write_text(
        json.dumps({"splits": {"train": {"images_written": 9}}}), encoding="utf-8"
    )
    out = tmp_path / "mixed"
    report = build_fullframe_dataset(
        FullFrameConfig(
            coco_root=synthetic_coco[0],
            wholebody_json=synthetic_coco[1],
            split="train",
            out=out,
            half_train_count=half_train_count(root),
        )
    )
    # 9 half images at f = 0.25 means three full frames, and the fixture has
    # exactly three qualifying images.
    assert report.target_count == 3
    assert report.images_written == 3
    assert report.fraction == FULL_FRACTION


def test_render_labeled_preview_writes_a_jpeg(
    tmp_path: Path, synthetic_coco: tuple[Path, Path]
) -> None:
    from PIL import Image

    out, _ = _build(tmp_path, synthetic_coco)
    preview = render_labeled_preview(
        out / "images" / "train",
        out / "labels" / "train",
        tmp_path / "preview.jpg",
        count=2,
    )
    assert preview.is_file()
    with Image.open(preview) as image:
        assert image.format == "JPEG"
        assert image.width == IMAGE_WIDTH
        assert image.height == 2 * IMAGE_HEIGHT + 8

    with pytest.raises(ValueError, match="no label files"):
        render_labeled_preview(tmp_path / "missing", tmp_path / "missing", tmp_path / "x.jpg")
    with pytest.raises(ValueError, match="count"):
        render_labeled_preview(
            out / "images" / "train", out / "labels" / "train", tmp_path / "y", count=0
        )


def test_cli_builds_and_renders(
    tmp_path: Path,
    synthetic_coco: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "cli"
    code = main(
        [
            "--coco-root",
            str(synthetic_coco[0]),
            "--wholebody-json",
            str(synthetic_coco[1]),
            "--split",
            "val",
            "--out",
            str(out),
            "--max-images",
            "2",
            "--preview",
            str(tmp_path / "preview.jpg"),
        ]
    )
    assert code == 0
    stdout = capsys.readouterr().out
    assert "images_written" in stdout
    assert "preview:" in stdout
    assert (out / "data.yaml").is_file()
    assert len(list((out / "images" / "val").glob("*.jpg"))) == 3


def test_module_exports_the_shared_helpers() -> None:
    """The builder must not reimplement the half-bust mapping."""

    assert fullframe.map_keypoints is not None
    assert fullframe.format_label is not None
    assert fullframe.wholebody_keypoints is not None
