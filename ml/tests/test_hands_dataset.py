"""Tests for the hand-crop dataset builder (synthetic sources, no downloads)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from cpsa_ml.datasets.hands import (
    SOURCE_LICENSES,
    HagridSource,
    HandBuildConfig,
    HandSample,
    SplitStats,
    UltralyticsHandSource,
    build_hand_dataset,
    crop_hand_sample,
    hand_crop_window,
    iter_hagrid_samples,
    iter_ultralytics_samples,
    write_hand_crops,
)
from cpsa_ml.schema import HALFBUST_LEFT_HAND_INDICES, HAND_21, get_schema

IMAGE_SIZE = 160


def hand_points(cx: float, cy: float, scale: float = 1.0) -> np.ndarray:
    """Return a MediaPipe-ordered ``(21, 3)`` hand: wrist below, fingers above."""

    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = (cx, cy, 1.0)
    for finger in range(5):
        x = cx + (finger - 2) * 6.0 * scale
        for joint in range(4):
            index = 1 + finger * 4 + joint
            points[index] = (x, cy - (18.0 + joint * 12.0) * scale, 1.0)
    return points


def draw_hand(path: Path, points: np.ndarray, size: int = IMAGE_SIZE) -> None:
    image = Image.new("RGB", (size, size), (40, 44, 52))
    draw = ImageDraw.Draw(image)
    for x, y, visibility in points:
        if visibility > 0:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(240, 220, 190))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)


def yolo_line(points: np.ndarray, size: int = IMAGE_SIZE, visibility: float = 2.0) -> str:
    xs, ys = points[:, 0] / size, points[:, 1] / size
    values = [0, 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()), xs.max() - xs.min()]
    values.append(ys.max() - ys.min())
    tokens = [f"{value:.6f}" for value in values]
    for x, y, visible in points:
        mark = visibility if visible > 0 else 0.0
        tokens.extend((f"{x / size:.6f}", f"{y / size:.6f}", f"{mark:g}"))
    return " ".join(tokens)


def make_ultralytics_tree(root: Path, train: int = 3, val: int = 1) -> Path:
    for split, count in (("train", train), ("val", val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"IMG_{split}_{index:03d}"
            points = hand_points(70.0 + index, 120.0)
            draw_hand(root / "images" / split / f"{stem}.jpg", points)
            (root / "labels" / split / f"{stem}.txt").write_text(
                yolo_line(points) + "\n", encoding="utf-8"
            )
    return root


def make_hagrid_tree(
    root: Path,
    gesture: str = "peace",
    hands: int = 1,
    offset_box: bool = False,
) -> Path:
    images = root / "images" / f"train_val_{gesture}"
    points = hand_points(90.0, 120.0)
    draw_hand(images / "00000000-0000-0000-0000-000000000001.jpg", points)

    width, height = IMAGE_SIZE, IMAGE_SIZE
    normalized = [[float(x) / width, float(y) / height] for x, y, _ in points]
    x1 = float(points[:, 0].min()) / width
    y1 = float(points[:, 1].min()) / height
    box_w = float(points[:, 0].max() - points[:, 0].min()) / width
    box_h = float(points[:, 1].max() - points[:, 1].min()) / height
    if offset_box:
        # The official gesture box misses the top finger joints: the builder has
        # to union it with the landmarks.
        y1 += 0.5 * box_h
        box_h *= 0.5
    entry = {
        "bboxes": [[x1, y1, box_w, box_h] for _ in range(hands)],
        "labels": [gesture for _ in range(hands)],
        "hand_landmarks": [normalized for _ in range(hands)],
        "user_id": "synthetic",
        "meta": {"age": [30.0], "gender": ["female"], "race": ["Asian"]},
    }
    annotations = root / "annotations" / "train" / f"{gesture}.json"
    annotations.parent.mkdir(parents=True, exist_ok=True)
    annotations.write_text(
        json.dumps({"00000000-0000-0000-0000-000000000001": entry}), encoding="utf-8"
    )
    return root


def test_hand_schema_is_the_left_hand_wholebody_block() -> None:
    assert isinstance(HAND_21, type(get_schema("hand21")))
    assert get_schema("hand21") is HAND_21
    assert HAND_21.wholebody_index == HALFBUST_LEFT_HAND_INDICES
    assert HAND_21.kpt_shape == (21, 3)
    assert HAND_21.flip_idx == tuple(range(21))  # a mirrored crop keeps the joints
    assert len(HAND_21.names) == 21


def test_hand_crop_window_is_square_and_inside_the_image() -> None:
    config = HandBuildConfig(out=Path("/tmp/unused"))
    rng = random.Random(7)
    for _ in range(100):
        x1 = rng.uniform(0, 120)
        y1 = rng.uniform(0, 90)
        box = (x1, y1, x1 + rng.uniform(5, 40), y1 + rng.uniform(5, 40))
        window = hand_crop_window(box, (160, 120), rng, config)
        left, top, right, bottom = window
        assert 0 <= left < right <= 160
        assert 0 <= top < bottom <= 120
        assert abs((right - left) - (bottom - top)) <= 1
        assert left <= 0.5 * (box[0] + box[2]) <= right
        assert top <= 0.5 * (box[1] + box[3]) <= bottom


def test_hand_crop_window_is_deterministic() -> None:
    config = HandBuildConfig(out=Path("/tmp/unused"))
    box = (40.0, 30.0, 90.0, 70.0)
    first = hand_crop_window(box, (160, 160), random.Random(3), config)
    second = hand_crop_window(box, (160, 160), random.Random(3), config)
    assert first == second


def test_crop_hand_sample_keeps_the_hand_box_inside_the_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "hand.jpg"
    points = hand_points(80.0, 130.0, scale=0.8)
    draw_hand(image_path, points)
    sample = HandSample(
        image_path=image_path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        box=(60.0, 20.0, 100.0, 130.0),
        keypoints=points,
        source="test",
        origin_split="train",
    )
    image = Image.open(image_path).convert("RGB")
    cropped = crop_hand_sample(image, sample, random.Random(0), HandBuildConfig(out=tmp_path))
    assert cropped is not None
    crop, line = cropped
    assert crop.size[0] > 2 and crop.size[1] > 2
    tokens = line.split()
    assert len(tokens) == 5 + 21 * 3
    assert tokens[0] == "0"
    cx, cy, box_w, box_h = (float(value) for value in tokens[1:5])
    assert cx - box_w / 2 >= 0.0 and cx + box_w / 2 <= 1.0
    assert cy - box_h / 2 >= 0.0 and cy + box_h / 2 <= 1.0
    keypoints = np.asarray([float(value) for value in tokens[5:]], dtype=np.float32).reshape(21, 3)
    assert int(np.count_nonzero(keypoints[:, 2] > 0)) == 21
    assert keypoints[:, 0].max() <= 1.0 and keypoints[:, 1].max() <= 1.0


def test_crop_hand_sample_drops_keypoints_outside_the_window(tmp_path: Path) -> None:
    image_path = tmp_path / "hand.jpg"
    points = hand_points(80.0, 130.0, scale=0.8)
    draw_hand(image_path, points)
    # Five joints are far outside the source image (and so outside any crop).
    points[-5:, :2] += np.asarray([400.0, 0.0], dtype=np.float32)
    sample = HandSample(
        image_path=image_path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        box=(60.0, 20.0, 100.0, 130.0),
        keypoints=points,
        source="test",
        origin_split="train",
    )
    image = Image.open(image_path).convert("RGB")
    config = HandBuildConfig(out=tmp_path, min_visible=5)
    cropped = crop_hand_sample(image, sample, random.Random(0), config)
    assert cropped is not None
    _, line = cropped
    keypoints = np.asarray([float(value) for value in line.split()[5:]], dtype=np.float32).reshape(
        21, 3
    )
    assert int(np.count_nonzero(keypoints[:, 2] > 0)) == 16
    assert np.all(keypoints[-5:, :] == 0.0)


def test_crop_hand_sample_rejects_too_few_visible_points(tmp_path: Path) -> None:
    image_path = tmp_path / "hand.jpg"
    points = hand_points(80.0, 130.0, scale=0.8)
    points[11:, 2] = 0.0  # only 11 labelled joints
    draw_hand(image_path, points)
    sample = HandSample(
        image_path=image_path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        box=(60.0, 20.0, 100.0, 130.0),
        keypoints=points,
        source="test",
        origin_split="train",
    )
    image = Image.open(image_path).convert("RGB")
    config = HandBuildConfig(out=tmp_path, min_visible=21)
    assert crop_hand_sample(image, sample, random.Random(0), config) is None


def test_build_from_the_ultralytics_tree(tmp_path: Path) -> None:
    root = make_ultralytics_tree(tmp_path / "hand-keypoints")
    out = tmp_path / "hand21"
    report = build_hand_dataset(
        HandBuildConfig(out=out, imgsz=128, val_ratio=0.0),
        ultralytics=UltralyticsHandSource(root=root),
    )
    assert report.splits["train"].images_written == 3
    assert report.splits["val"].images_written == 1
    assert report.to_dict()["totals"]["images"] == 4

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "hand"}
    assert data["kpt_shape"] == [21, 3]
    assert data["flip_idx"] == list(range(21))
    assert data["kpt_oks_sigmas"] == list(HAND_21.oks_sigmas)
    assert data["train"] == "images/train"

    schema = json.loads((out / "schema.json").read_text())
    assert schema["name"] == "hand21"

    stats = json.loads((out / "stats.json").read_text())
    assert stats["class_names"] == ["hand"]
    assert stats["sources"]["ultralytics-hand-keypoints"]["license"].startswith("CC BY-NC-SA")
    assert "license_warning" in stats
    assert stats["splits"]["train"]["handedness"] == {"unknown": 3}

    crops = sorted((out / "images" / "train").glob("*.jpg"))
    assert len(crops) == 3
    with Image.open(crops[0]) as crop:
        assert crop.size == (128, 128)
    label = (out / "labels" / "train" / f"{crops[0].stem}.txt").read_text().split()
    assert len(label) == 5 + 21 * 3


def test_build_from_the_hagrid_tree_reads_the_official_landmarks(tmp_path: Path) -> None:
    root = make_hagrid_tree(tmp_path / "hagrid", hands=2, offset_box=True)
    out = tmp_path / "hand21"
    report = build_hand_dataset(
        HandBuildConfig(out=out, val_ratio=0.0),
        hagrid=HagridSource(images=root / "images", annotations=root / "annotations"),
    )
    assert report.splits["train"].images_written == 2  # two hands in one frame
    assert report.splits["train"].labels == {"peace": 2}
    assert report.splits["train"].handedness == {"unknown": 2}

    stats = json.loads((out / "stats.json").read_text())
    assert stats["sources"]["hagrid"]["license"].startswith("CC BY-SA")
    assert "license_warning" not in stats

    for label_path in (out / "labels" / "train").glob("*.txt"):
        tokens = label_path.read_text().split()
        keypoints = np.asarray([float(value) for value in tokens[5:]], dtype=np.float32).reshape(
            21, 3
        )
        # The crop box is the union of the gesture box and the landmarks, so all
        # 21 points stay inside the crop.
        assert int(np.count_nonzero(keypoints[:, 2] > 0)) == 21


def test_hagrid_iterator_skips_hands_without_landmarks(tmp_path: Path) -> None:
    root = make_hagrid_tree(tmp_path / "hagrid")
    annotation_path = root / "annotations" / "train" / "peace.json"
    payload = json.loads(annotation_path.read_text())
    entry = next(iter(payload.values()))
    entry["hand_landmarks"] = [[]]
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    source = HagridSource(images=root / "images", annotations=root / "annotations")
    assert list(iter_hagrid_samples(source)) == []


def test_ultralytics_iterator_skips_short_label_lines(tmp_path: Path) -> None:
    root = make_ultralytics_tree(tmp_path / "hand-keypoints", train=1, val=0)
    label = root / "labels" / "train" / "IMG_train_000.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    source = UltralyticsHandSource(root=root, splits=("train",))
    assert list(iter_ultralytics_samples(source)) == []


def test_val_ratio_holds_out_training_images(tmp_path: Path) -> None:
    root = make_ultralytics_tree(tmp_path / "hand-keypoints", train=40, val=0)
    out = tmp_path / "hand21"
    report = build_hand_dataset(
        HandBuildConfig(out=out, val_ratio=0.25, seed=1),
        ultralytics=UltralyticsHandSource(root=root),
    )
    val = report.splits["val"].images_written
    assert 3 <= val <= 17
    assert report.splits["train"].images_written + val == 40


def test_resume_skips_existing_crops(tmp_path: Path) -> None:
    root = make_ultralytics_tree(tmp_path / "hand-keypoints")
    config = HandBuildConfig(out=tmp_path / "hand21", val_ratio=0.0)
    first = build_hand_dataset(config, ultralytics=UltralyticsHandSource(root=root))
    crops = sorted((config.out / "images" / "train").glob("*.jpg"))
    second = build_hand_dataset(config, ultralytics=UltralyticsHandSource(root=root))
    assert second.splits["train"].images_written == first.splits["train"].images_written
    assert second.splits["train"].skipped["existing"] == 3
    assert sorted((config.out / "images" / "train").glob("*.jpg")) == crops


def test_write_hand_crops_records_the_handedness(tmp_path: Path) -> None:
    image_path = tmp_path / "hand.jpg"
    points = hand_points(80.0, 130.0, scale=0.8)
    draw_hand(image_path, points)
    config = HandBuildConfig(out=tmp_path / "hand21")
    samples = [
        HandSample(
            image_path=image_path,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            box=(60.0, 20.0, 100.0, 130.0),
            keypoints=points,
            source="recorded",
            origin_split="train",
            handedness=handedness,
        )
        for handedness in ("left", "right")
    ]
    stats = write_hand_crops(config, "train", samples)
    assert isinstance(stats, SplitStats)
    assert stats.images_written == 2
    assert stats.handedness == {"left": 1, "right": 1}
    assert SplitStats.from_dict(stats.to_dict()).to_dict() == stats.to_dict()
    assert all(
        path.name.startswith("rec_") for path in (config.out / "images" / "train").glob("*.jpg")
    )


def test_min_visible_skips_crops_with_few_labelled_joints(tmp_path: Path) -> None:
    root = make_ultralytics_tree(tmp_path / "hand-keypoints", train=2, val=0)
    for label in (root / "labels" / "train").glob("*.txt"):
        tokens = label.read_text().split()
        for index in range(5, 21):  # keep five labelled joints, drop the rest
            tokens[5 + 3 * index : 8 + 3 * index] = ["0.0", "0.0", "0"]
        label.write_text(" ".join(tokens) + "\n", encoding="utf-8")
    out = tmp_path / "hand21"
    report = build_hand_dataset(
        HandBuildConfig(out=out, val_ratio=0.0, min_visible=6),
        ultralytics=UltralyticsHandSource(root=root),
    )
    assert report.splits["train"].images_written == 0
    assert report.splits["train"].skipped["too_few_keypoints"] == 2


def test_build_requires_a_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one source"):
        build_hand_dataset(HandBuildConfig(out=tmp_path / "hand21"))


def test_missing_source_tree_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        build_hand_dataset(
            HandBuildConfig(out=tmp_path / "hand21"),
            ultralytics=UltralyticsHandSource(root=tmp_path / "missing"),
        )


def test_source_licenses_are_recorded() -> None:
    assert SOURCE_LICENSES["hagrid"]["license"].startswith("CC BY-SA")
    assert "NON-COMMERCIAL" in SOURCE_LICENSES["ultralytics-hand-keypoints"]["license"]
