"""Tests for the training-list curation (no Kaggle, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from cpsa_ml.datasets.curation import (
    CurationConfig,
    ExtraDataset,
    curate_train_list,
    image_height,
    keypoint_groups,
    visible_keypoint_flags,
)

N_KEYPOINTS = 62


def _label(*, hands: tuple[int, int] = (0, 0), face: int = 0) -> str:
    """One YOLO-pose label line with the requested visible hand/face keypoints."""

    values = [0.0, 0.5, 0.5, 1.0, 1.0]
    for index in range(N_KEYPOINTS):
        visible = index < 11  # body points are always visible on these crops
        if 11 <= index < 20:
            visible = index - 11 < face
        elif index >= 20:
            hand, offset = (0, index - 20) if index < 41 else (1, index - 41)
            visible = offset < hands[hand]
        values += [0.5, 0.5, float(visible)]
    return " ".join(f"{value:g}" for value in values)


def _crop(
    root: Path,
    split: str,
    stem: str,
    height: int,
    *,
    hands: tuple[int, int] = (0, 0),
    face: int = 0,
    width: int = 64,
) -> Path:
    """Write one JPEG crop and its label under ``root``."""

    image = root / "images" / split / f"{stem}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height)).save(image, format="JPEG")
    label = root / "labels" / split / f"{stem}.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(_label(hands=hands, face=face) + "\n", encoding="utf-8")
    return image


def _dataset(root: Path, images: dict[str, Path]) -> Path:
    """Create the split skeleton of a dataset around already written images."""

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for path in images.values():
        assert path.is_file()
    return root


def _config(root: Path, tmp_path: Path, **overrides: object) -> CurationConfig:
    values: dict[str, object] = {"root": root, "out_dir": tmp_path / "work"}
    values.update(overrides)
    return CurationConfig(**values)


# ---------------------------------------------------------------------------
# Keypoint groups and label parsing
# ---------------------------------------------------------------------------


def test_keypoint_groups_follow_the_schema() -> None:
    groups = keypoint_groups("halfbust62")
    assert len(groups.face) == 9
    assert len(groups.left_hand) == len(groups.right_hand) == 21
    assert set(groups.face).isdisjoint(groups.left_hand)
    assert set(groups.left_hand).isdisjoint(groups.right_hand)


def test_keypoint_groups_reject_a_schema_without_hands() -> None:
    with pytest.raises(ValueError, match="no face/hand"):
        keypoint_groups("coco17")


def test_visible_keypoint_flags_needs_one_full_hand(tmp_path: Path) -> None:
    groups = keypoint_groups("halfbust62")
    label = tmp_path / "label.txt"

    def flags(*, hands: tuple[int, int] = (0, 0), face: int = 0) -> tuple[bool, bool]:
        label.write_text(_label(hands=hands, face=face) + "\n", encoding="utf-8")
        return visible_keypoint_flags(label, groups, n_keypoints=N_KEYPOINTS)

    assert flags() == (False, False)
    assert flags(hands=(5, 5)) == (False, False)  # 5 + 5 is not one full hand
    assert flags(hands=(10, 0)) == (True, False)
    assert flags(hands=(0, 10)) == (True, False)
    assert flags(hands=(9, 9)) == (False, False)
    assert flags(face=4) == (False, False)
    assert flags(face=5) == (False, True)


def test_visible_keypoint_flags_treat_missing_and_empty_labels_as_empty(tmp_path: Path) -> None:
    groups = keypoint_groups("halfbust62")
    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    assert visible_keypoint_flags(empty, groups, n_keypoints=N_KEYPOINTS) == (False, False)
    missing = tmp_path / "missing.txt"
    assert visible_keypoint_flags(missing, groups, n_keypoints=N_KEYPOINTS) == (False, False)


def test_visible_keypoint_flags_reject_a_malformed_label(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text("0 0.5 0.5 1 1 0.5 0.5 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 191 values"):
        visible_keypoint_flags(label, keypoint_groups(), n_keypoints=N_KEYPOINTS)


def test_image_height_reads_the_jpeg_header(tmp_path: Path) -> None:
    image = tmp_path / "crop.jpg"
    Image.new("RGB", (86, 115)).save(image, format="JPEG")
    assert image_height(image) == 115


def test_image_height_rejects_a_non_image(tmp_path: Path) -> None:
    junk = tmp_path / "crop.jpg"
    junk.write_bytes(b"not a jpeg")
    with pytest.raises(ValueError, match="cannot read the image header"):
        image_height(junk)


# ---------------------------------------------------------------------------
# Curated train list
# ---------------------------------------------------------------------------


def test_curate_train_list_keeps_tall_crops_and_repeats_hands(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    hand = _crop(root, "train", "a_hand", 200, hands=(10, 0))
    face = _crop(root, "train", "b_face", 200, face=5)
    plain = _crop(root, "train", "c_plain", 200)
    _crop(root, "train", "d_short", 127, hands=(10, 0))
    _dataset(root, {"a": hand, "b": face, "c": plain})

    result = curate_train_list(_config(root, tmp_path))

    assert result.train_list.read_text(encoding="utf-8").splitlines() == [
        str(hand),
        str(hand),
        str(face),
        str(plain),
    ]
    summary = result.summary
    assert summary["train_images_total"] == 4
    assert summary["train_images_kept"] == 3
    assert summary["train_entries_before_repeat"] == 3
    assert summary["train_entries"] == 4
    assert summary["train_images_repeated"] == 1
    assert summary["hand_images_total"] == 2
    assert summary["hand_images_kept"] == 1
    assert summary["hand_share_all"] == 0.5
    assert summary["hand_share_kept"] == 0.3333
    assert summary["face_images_total"] == 1
    assert summary["face_images_kept"] == 1
    assert summary["face_share_all"] == 0.25
    assert summary["face_share_kept"] == 0.3333
    assert summary["labels_missing"] == 0
    assert json.loads(result.summary_path.read_text(encoding="utf-8")) == summary


def test_curate_train_list_repeats_face_crops_when_enabled(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    both = _crop(root, "train", "a_both", 200, hands=(10, 0), face=5)
    face = _crop(root, "train", "b_face", 200, face=5)
    _dataset(root, {"a": both, "b": face})

    result = curate_train_list(_config(root, tmp_path, hand_repeat=2, face_repeat=3))

    # The face factor is the larger one and applies exactly once: no multiplicative boost.
    assert result.train_list.read_text(encoding="utf-8").splitlines() == [
        str(both),
        str(both),
        str(both),
        str(face),
        str(face),
        str(face),
    ]
    assert result.summary["train_entries"] == 6
    assert result.summary["face_share_kept"] == 1.0


def test_curate_train_list_can_switch_oversampling_off(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    hand = _crop(root, "train", "a_hand", 200, hands=(10, 0))
    _dataset(root, {"a": hand})

    result = curate_train_list(_config(root, tmp_path, hand_repeat=1, face_repeat=1))

    assert result.train_list.read_text(encoding="utf-8").splitlines() == [str(hand)]


def test_curate_train_list_writes_the_size_filtered_val_list(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    train = _crop(root, "train", "a_train", 200)
    tall = _crop(root, "val", "a_tall", 160)
    short = _crop(root, "val", "b_short", 159)
    _dataset(root, {"train": train, "tall": tall, "short": short})

    result = curate_train_list(_config(root, tmp_path))

    assert result.val_h160_list.read_text(encoding="utf-8").splitlines() == [str(tall)]
    assert result.summary["val_images_total"] == 2
    assert result.summary["val_images_h160"] == 1
    assert result.summary["val_min_height"] == 160


def test_curate_train_list_counts_missing_labels(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    image = root / "images" / "train" / "a_unlabelled.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 200)).save(image, format="JPEG")
    _dataset(root, {"a": image})

    result = curate_train_list(_config(root, tmp_path))

    assert result.summary["labels_missing"] == 1
    assert result.summary["hand_images_kept"] == 0
    assert result.train_list.read_text(encoding="utf-8").splitlines() == [str(image)]


# ---------------------------------------------------------------------------
# Override labels (Task P pseudo-labels)
# ---------------------------------------------------------------------------


def test_curate_train_list_mirrors_the_override_labels(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    image = _crop(root, "train", "a_crop", 200)
    _dataset(root, {"a": image})
    gt_label = root / "labels" / "train" / "a_crop.txt"
    gt_text = gt_label.read_text(encoding="utf-8")
    override = tmp_path / "halfbust-pl"
    (override / "labels" / "train").mkdir(parents=True)
    (override / "labels" / "train" / "a_crop.txt").write_text(
        _label(hands=(21, 21), face=9) + "\n", encoding="utf-8"
    )

    result = curate_train_list(_config(root, tmp_path, labels_root=override))

    mirror = tmp_path / "work" / "curated_train_mirror"
    mirrored_image = mirror / "images" / "train" / "a_crop.jpg"
    assert (mirror / "images" / "train").is_symlink()
    assert (mirror / "images" / "train").resolve() == (root / "images" / "train").resolve()
    mirrored_label = mirror / "labels" / "train" / "a_crop.txt"
    assert mirrored_label.is_symlink()
    assert mirrored_label.resolve() == (override / "labels" / "train" / "a_crop.txt").resolve()
    # The override label decides the repetition, so the two-entry repeat proves it is used.
    assert result.train_list.read_text(encoding="utf-8").splitlines() == [
        str(mirrored_image),
        str(mirrored_image),
    ]
    assert result.summary["labels_root"] == str(override.resolve())
    assert result.summary["mirror"] == str(mirror.resolve())
    # The read-only dataset is untouched.
    assert gt_label.read_text(encoding="utf-8") == gt_text


def test_curate_train_list_rejects_missing_override_labels(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    _crop(root, "train", "a_crop", 200)
    _dataset(root, {"a": root / "images" / "train" / "a_crop.jpg"})
    override = tmp_path / "halfbust-pl"
    (override / "labels" / "train").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="override labels are missing"):
        curate_train_list(_config(root, tmp_path, labels_root=override))


def test_curate_train_list_requires_the_override_label_tree(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    _crop(root, "train", "a_crop", 200)
    _dataset(root, {"a": root / "images" / "train" / "a_crop.jpg"})

    with pytest.raises(FileNotFoundError, match="override labels not found"):
        curate_train_list(_config(root, tmp_path, labels_root=tmp_path / "absent"))


# ---------------------------------------------------------------------------
# Extra datasets
# ---------------------------------------------------------------------------


def test_curate_train_list_mixes_extra_datasets(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    base = _crop(root, "train", "a_base", 200)
    _dataset(root, {"a": base})
    extra = tmp_path / "hagrid"
    first = _crop(extra, "train", "g_one", 200)
    second = _crop(extra, "train", "g_two", 200)

    result = curate_train_list(_config(root, tmp_path, extra=(ExtraDataset(extra, 3),)))

    assert result.train_list.read_text(encoding="utf-8").splitlines() == [
        str(base),
        str(first),
        str(first),
        str(first),
        str(second),
        str(second),
        str(second),
    ]
    assert result.summary["extra"] == [
        {
            "root": str(extra.resolve()),
            "repeat": 3,
            "train_images": 2,
            "train_entries": 6,
        }
    ]
    assert result.summary["extra_entries"] == 6
    assert result.summary["train_entries_total"] == 7


def test_curate_train_list_rejects_an_empty_extra_dataset(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    _crop(root, "train", "a_base", 200)
    _dataset(root, {"a": root / "images" / "train" / "a_base.jpg"})
    extra = tmp_path / "hagrid"
    (extra / "images" / "train").mkdir(parents=True)
    (extra / "labels" / "train").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="no images under"):
        curate_train_list(_config(root, tmp_path, extra=(ExtraDataset(extra),)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_curation_config_normalizes_paths_and_extra_entries(tmp_path: Path) -> None:
    config = CurationConfig(
        root=str(tmp_path / "halfbust"),
        out_dir=str(tmp_path / "work"),
        labels_root=str(tmp_path / "labels"),
        extra=[(str(tmp_path / "hagrid"), 2)],  # type: ignore[list-item]
    )
    assert config.root == tmp_path / "halfbust"
    assert config.out_dir == tmp_path / "work"
    assert config.labels_root == tmp_path / "labels"
    assert config.extra == (ExtraDataset(tmp_path / "hagrid", 2),)
    assert config.min_height == 128
    assert config.hand_repeat == 2
    assert config.face_repeat == 1
    assert config.val_min_height == 160
    assert config.schema == "halfbust62"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_height": -1}, "min_height"),
        ({"val_min_height": -1}, "val_min_height"),
        ({"hand_repeat": 0}, "hand_repeat"),
        ({"face_repeat": 0}, "face_repeat"),
        ({"extra": ((Path("root"), 0),)}, "repeat"),
    ],
)
def test_curation_config_rejects_invalid_values(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path / "halfbust", tmp_path, **overrides)


def test_curate_train_list_requires_the_train_split(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="no images under"):
        curate_train_list(_config(root, tmp_path))


def test_curate_train_list_requires_a_kept_crop(tmp_path: Path) -> None:
    root = tmp_path / "halfbust"
    _crop(root, "train", "a_short", 100)
    _dataset(root, {"a": root / "images" / "train" / "a_short.jpg"})

    with pytest.raises(ValueError, match="is >= 128px tall"):
        curate_train_list(_config(root, tmp_path))
