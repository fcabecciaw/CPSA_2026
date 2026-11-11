"""Tests for the full-frame IR-look hand21 dataset builder (synthetic sources)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from cpsa_ml.datasets.handframe import (
    BLUR_P,
    CLAHE_CLIP,
    CLAHE_TILES,
    CONTRAST_RANGE,
    GAMMA_RANGE,
    NOISE_SIGMA_RANGE,
    HandFrameConfig,
    apply_clahe,
    build_handframe_dataset,
    frame_label,
    frame_window,
    ir_look,
    render_label_contact_sheet,
)
from cpsa_ml.datasets.hands import (
    SOURCE_LICENSES,
    HagridSource,
    HandSample,
    UltralyticsHandSource,
)

IMAGE_SIZE = 200
OUT_HW = (256, 320)


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
    # A bright rectangle around the palm keeps the IR-look transform non-flat.
    palm = points[0]
    draw.rectangle((palm[0] - 18, palm[1] - 40, palm[0] + 18, palm[1] - 5), fill=(200, 190, 170))
    for x, y, visibility in points:
        if visibility > 0:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(250, 245, 235))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)


def yolo_line(points: np.ndarray, size: int = IMAGE_SIZE) -> str:
    xs, ys = points[:, 0] / size, points[:, 1] / size
    values = [0, 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()), xs.max() - xs.min()]
    values.append(ys.max() - ys.min())
    tokens = [f"{value:.6f}" for value in values]
    for x, y, visible in points:
        tokens.extend((f"{x / size:.6f}", f"{y / size:.6f}", f"{1.0 if visible > 0 else 0.0:g}"))
    return " ".join(tokens)


def make_hagrid_tree(
    root: Path,
    images: int = 2,
    hands_per_image: int = 1,
    invalid_hand: bool = False,
    gesture: str = "peace",
) -> Path:
    """Write ``images`` HaGRID images of ``hands_per_image`` hands plus annotations."""

    directory = root / "images" / f"train_val_{gesture}"
    entries: dict[str, dict] = {}
    for index in range(images):
        image_id = f"00000000-0000-0000-0000-{index:012d}"
        centres = [(IMAGE_SIZE * 0.35, IMAGE_SIZE * 0.8)]
        if hands_per_image > 1:
            centres.append((IMAGE_SIZE * 0.7, IMAGE_SIZE * 0.8))
        points = [hand_points(cx, cy) for cx, cy in centres]
        draw_hand(directory / f"{image_id}.jpg", np.concatenate(points)[:21])
        entry: dict[str, object] = {
            "bboxes": [],
            "labels": [gesture for _ in points],
            "hand_landmarks": [
                [[float(x) / IMAGE_SIZE, float(y) / IMAGE_SIZE] for x, y, _ in hand]
                for hand in points
            ],
            "user_id": "synthetic",
        }
        boxes = []
        for hand in points:
            boxes.append(
                [
                    float(hand[:, 0].min()) / IMAGE_SIZE,
                    float(hand[:, 1].min()) / IMAGE_SIZE,
                    float(np.ptp(hand[:, 0])) / IMAGE_SIZE,
                    float(np.ptp(hand[:, 1])) / IMAGE_SIZE,
                ]
            )
        entry["bboxes"] = boxes
        if invalid_hand:
            # A second raw box whose landmarks are unusable: the readers skip it,
            # so the entry holds a hand that could never be labelled.
            entry["bboxes"] = boxes + [boxes[0]]
            entry["labels"] = [gesture] * (len(points) + 1)
            entry["hand_landmarks"] = list(entry["hand_landmarks"]) + [
                [[0.5, 0.5] for _ in range(20)]
            ]
        entries[image_id] = entry

    annotations = root / "annotations" / "train" / f"{gesture}.json"
    annotations.parent.mkdir(parents=True, exist_ok=True)
    annotations.write_text(json.dumps(entries), encoding="utf-8")
    return root


def make_ultralytics_tree(root: Path, train: int = 2, val: int = 1) -> Path:
    for split, count in (("train", train), ("val", val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"IMG_{split}_{index:03d}"
            points = hand_points(60.0 + 10.0 * index, 160.0)
            draw_hand(root / "images" / split / f"{stem}.jpg", points)
            (root / "labels" / split / f"{stem}.txt").write_text(
                yolo_line(points) + "\n", encoding="utf-8"
            )
    return root


def build(
    tmp_path: Path,
    *,
    hagrid: Path | None = None,
    ultralytics: Path | None = None,
    name: str = "out",
    **config: object,
) -> tuple[dict, HandFrameConfig]:
    frame_config = HandFrameConfig(out=tmp_path / name, **config)  # type: ignore[arg-type]
    payload = build_handframe_dataset(
        frame_config,
        hagrid=HagridSource(images=hagrid / "images", annotations=hagrid / "annotations")
        if hagrid is not None
        else None,
        ultralytics=UltralyticsHandSource(root=ultralytics) if ultralytics is not None else None,
    )
    return payload, frame_config


def label_lines(out: Path, split: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for path in sorted((out / "labels" / split).glob("*.txt")):
        rows.extend(line.split() for line in path.read_text(encoding="utf-8").splitlines())
    return rows


def test_clahe_constants_match_the_runtime_ir_preprocessing() -> None:
    # Duplicated in cpsa.vision.irdepth (src/): the training look must be the
    # deployed look, so both copies are pinned to the same numbers.
    assert CLAHE_CLIP == 3.0
    assert CLAHE_TILES == (8, 8)


def test_ir_look_ranges_match_the_plan() -> None:
    assert GAMMA_RANGE == (0.6, 1.6)
    assert CONTRAST_RANGE == (0.6, 1.4)
    assert NOISE_SIGMA_RANGE == (0.0, 8.0)
    assert BLUR_P == 0.3


def test_apply_clahe_normalizes_then_applies_the_runtime_clahe() -> None:
    # A flat frame is min-max normalized to zeros; CLAHE then spreads that
    # single histogram bin to one constant value.  Comparing against the direct
    # OpenCV call pins the parameters and the normalization step.
    flat = np.full((32, 32), 7, dtype=np.uint8)
    reference = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES).apply(
        np.zeros_like(flat)
    )
    looked = apply_clahe(flat)
    assert np.array_equal(looked, reference)
    assert len(np.unique(looked)) == 1


def test_apply_clahe_stretches_low_contrast_frames() -> None:
    low_contrast = np.tile(np.arange(20, 40, dtype=np.uint8), (32, 1))
    stretched = apply_clahe(low_contrast)
    assert stretched.dtype == np.uint8
    assert stretched.shape == low_contrast.shape
    assert int(stretched.max()) > int(low_contrast.max())
    # The CLAHE parameters are the runtime ones, not a local default.
    reference = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        ((low_contrast.astype(np.float32) - 20.0) * (255.0 / 19.0)).astype(np.uint8)
    )
    assert np.array_equal(stretched, reference)


def test_ir_look_is_deterministic_per_seed() -> None:
    gray = np.tile(np.linspace(10, 240, 64, dtype=np.uint8), (48, 1))
    first = ir_look(gray, random.Random("7:seed"), clahe_p=0.8)
    again = ir_look(gray, random.Random("7:seed"), clahe_p=0.8)
    other = ir_look(gray, random.Random("8:seed"), clahe_p=0.8)
    assert first.dtype == np.uint8
    assert first.shape == gray.shape
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_ir_look_applies_clahe_last_as_the_runtime_does() -> None:
    gray = np.tile(np.linspace(20, 200, 32, dtype=np.uint8), (32, 1))
    plain = ir_look(gray, random.Random("pipeline"), clahe_p=0.0)
    with_clahe = ir_look(gray, random.Random("pipeline"), clahe_p=1.0)
    assert np.array_equal(with_clahe, apply_clahe(plain))


def test_frame_window_keeps_the_5_4_aspect_and_covers_every_hand() -> None:
    boxes = [(100.0, 100.0, 200.0, 200.0), (300.0, 120.0, 380.0, 190.0)]
    window = frame_window(boxes, (640, 480), random.Random(0), HandFrameConfig(out=Path(".")))
    x1, y1, x2, y2 = window
    assert x1 >= 0 and y1 >= 0 and x2 <= 640 and y2 <= 480
    assert x1 <= 100.0 and x1 <= 300.0 and x2 >= 200.0 and x2 >= 380.0
    assert y1 <= 100.0 and y1 <= 120.0 and y2 >= 200.0 and y2 >= 190.0
    assert abs((x2 - x1) / (y2 - y1) - 1.25) < 0.02
    # The largest (100 px) hand is a sane share of the window width.
    assert 0.2 <= 100.0 / (x2 - x1) <= 0.7


def test_frame_window_grows_to_cover_hands_that_are_far_apart() -> None:
    boxes = [(10.0, 100.0, 60.0, 150.0), (570.0, 100.0, 620.0, 150.0)]
    window = frame_window(boxes, (640, 480), random.Random(0), HandFrameConfig(out=Path(".")))
    x1, y1, x2, y2 = window
    assert x1 <= 10.0 and x2 >= 620.0
    # Covering the union beats the requested hand fraction: the hand is small.
    assert 50.0 / (x2 - x1) < 0.25


def test_frame_window_clamps_inside_a_small_image() -> None:
    window = frame_window(
        [(2.0, 2.0, 20.0, 20.0)], (24, 24), random.Random(0), HandFrameConfig(out=Path("."))
    )
    x1, y1, x2, y2 = window
    assert (x1, y1) == (0, 0)
    assert x2 <= 24 and y2 <= 24


def test_frame_label_drops_hands_whose_centre_is_outside_the_window() -> None:
    window = (0, 0, 100, 50)
    inside = HandSample(
        image_path=Path("a.jpg"),
        image_size=(200, 200),
        box=(10.0, 30.0, 60.0, 70.0),
        keypoints=hand_points(35.0, 60.0),
        source="hagrid",
        origin_split="train",
    )
    outside = HandSample(
        image_path=Path("a.jpg"),
        image_size=(200, 200),
        box=(120.0, 30.0, 170.0, 70.0),
        keypoints=hand_points(145.0, 60.0),
        source="hagrid",
        origin_split="train",
    )
    line = frame_label(inside, window)
    assert line is not None
    assert frame_label(outside, window) is None
    tokens = line.split()
    assert len(tokens) == 5 + 21 * 3
    assert tokens[0] == "0"
    values = np.asarray([float(value) for value in tokens[1:]], dtype=np.float64)
    assert np.all(values >= 0.0) and np.all(values <= 1.0)
    keypoints = values[4:].reshape(21, 3)
    # The fingertips (y 6-42) are inside the window; the wrist (y 60) is not.
    assert keypoints[0, 2] == 0.0
    assert np.count_nonzero(keypoints[:, 2]) == 20


def test_each_source_image_becomes_one_gray_3_channel_frame_of_imgsz_hw(tmp_path: Path) -> None:
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=3)
    payload, config = build(tmp_path, hagrid=hagrid, val_ratio=0.0)

    train = sorted((config.out / "images" / "train").glob("*.png"))
    assert len(train) == 3
    assert payload["totals"]["images"] == 3
    for path in train:
        with Image.open(path) as image:
            assert image.size == (OUT_HW[1], OUT_HW[0])
            array = np.asarray(image.convert("RGB"))
        assert array.shape == (OUT_HW[0], OUT_HW[1], 3)
        assert np.array_equal(array[:, :, 0], array[:, :, 1])
        assert np.array_equal(array[:, :, 1], array[:, :, 2])


def test_every_hand_inside_the_window_is_labelled(tmp_path: Path) -> None:
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=1, hands_per_image=2)
    payload, config = build(tmp_path, hagrid=hagrid, val_ratio=0.0)
    lines = label_lines(config.out, "train")
    assert len(lines) == 2
    assert payload["splits"]["train"]["labels_written"] == 2
    assert payload["splits"]["train"]["hands_seen"] == 2
    assert payload["splits"]["train"]["hands_outside_window"] == 0
    for tokens in lines:
        assert len(tokens) == 5 + 21 * 3
        values = np.asarray([float(value) for value in tokens[1:]], dtype=np.float64)
        center_x, center_y, width, height = values[:4]
        assert 0.0 < width <= 1.0 and 0.0 < height <= 1.0
        assert center_x - width / 2 >= 0.0 and center_x + width / 2 <= 1.0
        assert center_y - height / 2 >= 0.0 and center_y + height / 2 <= 1.0
        assert np.count_nonzero(values[4:].reshape(21, 3)[:, 2]) >= 5


def test_an_image_with_an_unlabelled_hand_is_skipped(tmp_path: Path) -> None:
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=2, invalid_hand=True)
    payload, config = build(tmp_path, hagrid=hagrid, val_ratio=0.0)
    assert list((config.out / "images" / "train").glob("*.png")) == []
    assert payload["totals"]["images"] == 0
    skipped = payload["splits"]["train"]["skip_reasons"]
    assert skipped.get("unlabelled_hand") == 2


def test_the_build_is_byte_identical_for_the_same_seed(tmp_path: Path) -> None:
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=3, hands_per_image=2)
    first, first_config = build(tmp_path, hagrid=hagrid, name="seed0")
    second, second_config = build(tmp_path, hagrid=hagrid, name="seed0-again")
    assert first["totals"] == second["totals"]
    first_files = sorted((first_config.out / "images" / "train").glob("*.png"))
    second_files = sorted((second_config.out / "images" / "train").glob("*.png"))
    assert [path.name for path in first_files] == [path.name for path in second_files]
    for left, right in zip(first_files, second_files, strict=True):
        assert left.read_bytes() == right.read_bytes()

    _, other_config = build(tmp_path, hagrid=hagrid, name="seed1", seed=1)
    other_files = sorted((other_config.out / "images" / "train").glob("*.png"))
    assert [path.name for path in other_files] == [path.name for path in first_files]
    assert any(
        left.read_bytes() != right.read_bytes()
        for left, right in zip(first_files, other_files, strict=True)
    )


def test_val_ratio_holds_out_a_deterministic_share_of_train_images(tmp_path: Path) -> None:
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=12)
    first, _ = build(tmp_path, hagrid=hagrid, name="val-a", val_ratio=0.5)
    second, _ = build(tmp_path, hagrid=hagrid, name="val-b", val_ratio=0.5)
    assert first["splits"]["val"]["images_written"] > 0
    assert first["splits"]["train"]["images_written"] > 0
    assert first["splits"]["val"]["images_written"] == second["splits"]["val"]["images_written"]
    assert (
        first["splits"]["train"]["images_written"] + first["splits"]["val"]["images_written"] == 12
    )


def test_ultralytics_source_maps_the_source_splits(tmp_path: Path) -> None:
    tree = make_ultralytics_tree(tmp_path / "hand-keypoints", train=2, val=1)
    payload, config = build(tmp_path, ultralytics=tree, val_ratio=0.0)
    assert payload["splits"]["train"]["images_written"] == 2
    assert payload["splits"]["val"]["images_written"] == 1
    assert list((config.out / "images" / "train").glob("*.png"))
    assert list((config.out / "images" / "val").glob("*.png"))
    assert payload["splits"]["train"]["sources"] == {"ultralytics-hand-keypoints": 2}
    assert len(label_lines(config.out, "train")) == 2


def test_max_images_caps_each_split(tmp_path: Path) -> None:
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=5)
    payload, config = build(tmp_path, hagrid=hagrid, max_images=2, val_ratio=0.0)
    assert len(list((config.out / "images" / "train").glob("*.png"))) == 2
    assert payload["splits"]["train"]["skip_reasons"]["cap_reached"] == 3


def test_dataset_metadata_is_hand21_with_the_source_licences(tmp_path: Path) -> None:
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=1)
    tree = make_ultralytics_tree(tmp_path / "hand-keypoints")
    payload, config = build(tmp_path, hagrid=hagrid, ultralytics=tree)

    data = yaml.safe_load((config.out / "data.yaml").read_text(encoding="utf-8"))
    assert data["kpt_shape"] == [21, 3]
    assert data["flip_idx"] == list(range(21))
    assert data["names"] == {0: "hand"}
    assert data["train"] == "images/train" and data["val"] == "images/val"

    schema = json.loads((config.out / "schema.json").read_text(encoding="utf-8"))
    assert schema["name"] == "hand21"
    assert len(schema["names"]) == 21

    stats = json.loads((config.out / "stats.json").read_text(encoding="utf-8"))
    assert stats["imgsz_hw"] == [256, 320]
    assert stats["window"]["hand_frac"] == [0.25, 0.6]
    assert stats["look"]["clahe_clip"] == CLAHE_CLIP
    assert stats["look"]["clahe_tiles"] == list(CLAHE_TILES)
    assert stats["look"]["grayscale"] is True
    assert stats["licenses"]["hagrid"] == SOURCE_LICENSES["hagrid"]["license"]
    assert (
        stats["licenses"]["ultralytics-hand-keypoints"]
        == SOURCE_LICENSES["ultralytics-hand-keypoints"]["license"]
    )
    assert "CC BY-NC-SA" in stats["license_warning"]
    assert stats["totals"] == payload["totals"]


def test_contact_sheet_renders_a_labelled_grid(tmp_path: Path) -> None:
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=4)
    _, config = build(tmp_path, hagrid=hagrid, val_ratio=0.0)
    sheet = render_label_contact_sheet(
        config.out / "images" / "train",
        config.out / "labels" / "train",
        tmp_path / "sheet.png",
        count=4,
        columns=2,
        cell_width=160,
    )
    with Image.open(sheet) as image:
        assert image.format == "PNG"
        assert image.width == 2 * 160 + 6
        assert image.height > 0


def test_names_and_split_do_not_depend_on_the_root_path(tmp_path: Path) -> None:
    # A local run and the Kaggle notebook extract the archives somewhere else;
    # the same images must still give the same file names, split and bytes.
    first = make_hagrid_tree(tmp_path / "a" / "hagrid", images=6)
    second = make_hagrid_tree(tmp_path / "b" / "deeper" / "hagrid", images=6)
    left, left_config = build(tmp_path, hagrid=first, name="left", val_ratio=0.5)
    right, right_config = build(tmp_path, hagrid=second, name="right", val_ratio=0.5)
    assert left["splits"] == right["splits"]
    for split in ("train", "val"):
        left_files = sorted((left_config.out / "images" / split).glob("*.png"))
        right_files = sorted((right_config.out / "images" / split).glob("*.png"))
        assert [path.name for path in left_files] == [path.name for path in right_files]
        for one, other in zip(left_files, right_files, strict=True):
            assert one.read_bytes() == other.read_bytes()


def test_resume_skips_frames_that_already_exist(tmp_path: Path) -> None:
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=2)
    _, config = build(tmp_path, hagrid=hagrid, name="resume")
    again, _ = build(tmp_path, hagrid=hagrid, name="resume")
    assert again["totals"]["images"] == 2
    assert again["splits"]["train"]["skip_reasons"]["existing"] == 2


@pytest.mark.parametrize(
    "config",
    [
        {"imgsz_hw": (250, 320)},
        {"hand_frac": (0.0, 0.5)},
        {"hand_frac": (0.6, 0.25)},
        {"clahe_p": 1.5},
        {"val_ratio": 1.0},
        {"max_images": 0},
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, config: dict) -> None:
    with pytest.raises(ValueError):
        HandFrameConfig(out=tmp_path, **config)


def test_build_needs_a_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one source"):
        build_handframe_dataset(HandFrameConfig(out=tmp_path))


def test_small_images_are_still_resized_to_imgsz_hw(tmp_path: Path) -> None:
    # 160 px HaGRID-384p frames are narrower than the requested window; the
    # builder clamps the window to the image and resizes it anyway.
    hagrid = make_hagrid_tree(tmp_path / "hagrid", images=1)
    _, config = build(tmp_path, hagrid=hagrid, val_ratio=0.0)
    for path in (config.out / "images" / "train").glob("*.png"):
        with Image.open(path) as image:
            assert image.size == (320, 256)
