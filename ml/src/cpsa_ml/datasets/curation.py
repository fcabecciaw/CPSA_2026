"""Curate the Ultralytics training list of the half-bust pose dataset.

The demo sees a seated user at 0.45-0.7 m, so hands and face carry most of the
signal, but the built crops are small (median 86x115 px) and most of them show
no hand at all.  This module turns the built dataset into an explicit training
list:

* keep only crops at least ``min_height`` pixels tall (default 128), so hand and
  face keypoints survive the letterbox to the training size;
* repeat crops that show a hand with at least :data:`HAND_MIN_VISIBLE` (10)
  visible keypoints ``hand_repeat`` times (default 2), and optionally crops with
  at least :data:`FACE_MIN_VISIBLE` (5) visible face keypoints ``face_repeat``
  times (default 1 = off).  A crop that qualifies for both is repeated
  ``max(hand_repeat, face_repeat)`` times, so one crop never gets two
  multiplicative boosts;
* mix in extra ``halfbust62`` datasets (for example HaGRID gesture frames
  labelled by a teacher) ``repeat`` times each;
* report the composition before and after in a JSON summary next to the list;
* emit the size-filtered val list (``h >= 160``) that is reported as the
  demo-like metric next to the full GT val split;
* mirror the images into the output directory when an override label root is
  given (pseudo-labels from a parallel ``labels/`` tree), because Ultralytics
  derives the label path from the image path.

Ultralytics accepts a ``.txt`` file with one absolute image path per line as a
``train`` entry and resolves labels by replacing the last ``/images/`` path
component, so a list may repeat the same path and a mirrored tree can override
the labels without touching the read-only dataset.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cpsa_ml.schema import KeypointSchema, get_schema

__all__ = [
    "FACE_MIN_VISIBLE",
    "HAND_MIN_VISIBLE",
    "VAL_MIN_HEIGHT",
    "CurationConfig",
    "CurationResult",
    "ExtraDataset",
    "curate_train_list",
    "image_height",
    "keypoint_groups",
    "visible_keypoint_flags",
]

LOGGER = logging.getLogger(__name__)

#: A hand counts as usable when at least this many of its 21 keypoints are visible.
HAND_MIN_VISIBLE = 10

#: A face counts as usable when at least this many of its 9 keypoints are visible.
FACE_MIN_VISIBLE = 5

#: Default height of the size-filtered val list, close to the seated-user demo.
VAL_MIN_HEIGHT = 160

#: Image suffixes Ultralytics accepts for a pose dataset.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

#: Names of the artefacts written next to each other in ``out_dir``.
TRAIN_LIST_NAME = "curated_train.txt"
VAL_H160_LIST_NAME = "curated_val_h160.txt"
SUMMARY_NAME = "curated_train_summary.json"
MIRROR_NAME = "curated_train_mirror"


@dataclass(frozen=True, slots=True)
class ExtraDataset:
    """Another ``halfbust62``-layout dataset mixed into the training list.

    Args:
        root: Dataset root with ``images/{train,val}`` and ``labels/{train,val}``.
        repeat: How many times each of its train images is listed (``>= 1``).
    """

    root: Path
    repeat: int = 1

    def __post_init__(self) -> None:
        repeat = int(self.repeat)
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "repeat", repeat)
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")


@dataclass(frozen=True, slots=True)
class CurationConfig:
    """What to keep, repeat and where to write the curated lists.

    Args:
        root: ``halfbust62`` dataset root (must contain ``images/train``).
        out_dir: Directory for the train list, the size-filtered val list, the
            JSON summary and the mirror tree; ``/kaggle/working`` on Kaggle.
        min_height: Smallest crop height kept, in pixels.
        hand_repeat: Repeat factor for crops with a hand of ``>= 10`` visible
            keypoints (``1`` disables the oversampling).
        face_repeat: Repeat factor for crops with ``>= 5`` visible face
            keypoints (``1`` disables it).
        val_min_height: Smallest height of the size-filtered val list.
        labels_root: Dataset root holding an alternative ``labels/train`` tree
            (for example Task P pseudo-labels); ``None`` keeps the GT labels.
        schema: Keypoint schema name both datasets were built with.
        extra: Additional datasets whose train images join the list.
    """

    root: Path
    out_dir: Path
    min_height: int = 128
    hand_repeat: int = 2
    face_repeat: int = 1
    val_min_height: int = VAL_MIN_HEIGHT
    labels_root: Path | None = None
    schema: str = "halfbust62"
    extra: tuple[ExtraDataset, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "out_dir", Path(self.out_dir))
        if self.labels_root is not None:
            object.__setattr__(self, "labels_root", Path(self.labels_root))
        extra = tuple(self.extra)
        object.__setattr__(
            self,
            "extra",
            tuple(
                item if isinstance(item, ExtraDataset) else ExtraDataset(*item) for item in extra
            ),
        )
        if self.min_height < 0:
            raise ValueError(f"min_height must be non-negative, got {self.min_height}")
        if self.val_min_height < 0:
            raise ValueError(f"val_min_height must be non-negative, got {self.val_min_height}")
        if int(self.hand_repeat) < 1:
            raise ValueError(f"hand_repeat must be >= 1, got {self.hand_repeat}")
        if int(self.face_repeat) < 1:
            raise ValueError(f"face_repeat must be >= 1, got {self.face_repeat}")


@dataclass(frozen=True, slots=True)
class CurationResult:
    """The written lists plus the composition summary of the curated list."""

    train_list: Path
    val_h160_list: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class KeypointGroups:
    """Schema keypoints the curation scores: one face group and two hands."""

    face: tuple[int, ...]
    left_hand: tuple[int, ...]
    right_hand: tuple[int, ...]


def keypoint_groups(schema: str | KeypointSchema = "halfbust62") -> KeypointGroups:
    """Return the face and per-hand keypoint indices of a schema.

    Args:
        schema: Schema name or schema object.

    Returns:
        The indices grouped by :class:`KeypointGroups`.

    Raises:
        ValueError: If the schema has no face or no hand keypoints, because the
            curation would then silently over-sample nothing.
    """

    resolved = get_schema(schema)
    groups = KeypointGroups(
        face=tuple(i for i, name in enumerate(resolved.names) if name.startswith("face_")),
        left_hand=tuple(
            i for i, name in enumerate(resolved.names) if name.startswith("left_hand_")
        ),
        right_hand=tuple(
            i for i, name in enumerate(resolved.names) if name.startswith("right_hand_")
        ),
    )
    if not (groups.face and groups.left_hand and groups.right_hand):
        raise ValueError(f"schema {resolved.name!r} has no face/hand keypoint group to curate on")
    return groups


def image_height(path: Path | str) -> int:
    """Return the pixel height of an image, reading only its header.

    ``PIL.Image.open`` is lazy: it parses the JPEG header and never decodes the
    pixels, which keeps the pass over 104k crops in the seconds range.

    Raises:
        ValueError: If the file cannot be read as an image.
    """

    try:
        with Image.open(path) as image:
            return int(image.height)
    except OSError as exc:
        raise ValueError(f"cannot read the image header of {path}: {exc}") from exc


def _label_visibility(label_path: Path, n_keypoints: int) -> NDArray[np.bool_]:
    """Visibility matrix ``(instances, n_keypoints)`` of one YOLO-pose label file."""

    width = 5 + 3 * n_keypoints
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return np.zeros((0, n_keypoints), dtype=bool)
    if not text:
        return np.zeros((0, n_keypoints), dtype=bool)
    values = np.asarray(text.split(), dtype=np.float32)
    if values.size % width:
        raise ValueError(
            f"{label_path}: expected {width} values per row (5 + 3 * {n_keypoints}), "
            f"got {values.size}"
        )
    rows = values.reshape(-1, width)
    return rows[:, 7:width:3] > 0


def visible_keypoint_flags(
    label_path: Path | str,
    groups: KeypointGroups,
    *,
    n_keypoints: int,
    hand_min_visible: int = HAND_MIN_VISIBLE,
    face_min_visible: int = FACE_MIN_VISIBLE,
) -> tuple[bool, bool]:
    """``(has_hand, has_face)`` of one label file.

    A crop ``has_hand`` when at least one of its instances shows a single hand
    with ``hand_min_visible`` visible keypoints, and ``has_face`` when at least
    one instance shows ``face_min_visible`` visible face keypoints.  A missing
    or empty label file yields ``(False, False)``: it carries no evidence of a
    hand or face either.
    """

    visibility = _label_visibility(Path(label_path), n_keypoints)
    if not visibility.size:
        return False, False
    left = visibility[:, list(groups.left_hand)].sum(axis=1)
    right = visibility[:, list(groups.right_hand)].sum(axis=1)
    face = visibility[:, list(groups.face)].sum(axis=1)
    has_hand = bool(((left >= hand_min_visible) | (right >= hand_min_visible)).any())
    has_face = bool((face >= face_min_visible).any())
    return has_hand, has_face


def _image_files(directory: Path) -> list[Path]:
    """Sorted image files of a split directory; empty when it does not exist."""

    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _mirror_train_images(
    images: Sequence[Path], labels_dir: Path, mirror: Path
) -> dict[Path, Path]:
    """Mirror images plus override labels so Ultralytics resolves the override.

    Ultralytics replaces the last ``/images/`` component of an image path with
    ``/labels/`` to find the label file, so the override only takes effect when
    the images are reachable through a mirrored tree.  The image split directory
    is one symlink (not 60k file links); each override label is a file symlink.

    Returns:
        ``{source image: mirrored image path}`` for every image.

    Raises:
        FileNotFoundError: If an override label is missing or the label tree is
            empty, so a partial pseudo-label set cannot silently fall back to GT.
    """

    split_images = mirror / "images" / "train"
    split_labels = mirror / "labels" / "train"
    split_images.parent.mkdir(parents=True, exist_ok=True)
    if split_labels.is_symlink():
        split_labels.unlink()
    elif split_labels.is_dir():
        shutil.rmtree(split_labels)
    split_labels.mkdir(parents=True, exist_ok=True)
    if split_images.is_symlink() or split_images.exists():
        split_images.unlink()
    os.symlink(images[0].parent.resolve(), split_images, target_is_directory=True)

    mirrored: dict[Path, Path] = {}
    missing: list[str] = []
    for image in images:
        source_label = labels_dir / f"{image.stem}.txt"
        if not source_label.is_file():
            missing.append(str(source_label))
            continue
        os.symlink(source_label.resolve(), split_labels / source_label.name)
        mirrored[image] = split_images / image.name
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} override labels are missing under {labels_dir} "
            f"(first: {missing[:3]}); the curated train set must not mix GT and override labels"
        )
    return mirrored


def curate_train_list(config: CurationConfig) -> CurationResult:
    """Write the curated train list, the size-filtered val list and the summary.

    Args:
        config: Dataset roots, repeat factors and output directory.

    Returns:
        The written paths and the JSON-ready summary.

    Raises:
        FileNotFoundError: If the train split, an extra dataset or an override
            label tree is missing.
        ValueError: If no crop survives the filter or a label file is corrupt.
    """

    started = time.perf_counter()
    schema = get_schema(config.schema)
    groups = keypoint_groups(schema)
    root = config.root.resolve()
    out_dir = config.out_dir.resolve()
    labels_dir = (
        (config.labels_root / "labels" / "train").resolve()
        if config.labels_root is not None
        else root / "labels" / "train"
    )
    if config.labels_root is not None and not labels_dir.is_dir():
        raise FileNotFoundError(f"override labels not found: {labels_dir}")
    train_images = _image_files(root / "images" / "train")
    if not train_images:
        raise FileNotFoundError(f"no images under {root / 'images' / 'train'}")
    val_images = _image_files(root / "images" / "val")

    kept: list[Path] = []
    multiplicity: dict[Path, int] = {}
    hand_all = face_all = hand_kept = face_kept = missing_labels = 0
    for image in train_images:
        label_path = labels_dir / f"{image.stem}.txt"
        has_hand, has_face = visible_keypoint_flags(
            label_path, groups, n_keypoints=len(schema.names)
        )
        missing_labels += int(not label_path.is_file())
        hand_all += int(has_hand)
        face_all += int(has_face)
        if image_height(image) < config.min_height:
            continue
        kept.append(image)
        hand_kept += int(has_hand)
        face_kept += int(has_face)
        repeat = 1
        if has_hand:
            repeat = max(repeat, int(config.hand_repeat))
        if has_face:
            repeat = max(repeat, int(config.face_repeat))
        multiplicity[image] = repeat
    if not kept:
        raise ValueError(
            f"no crop under {root / 'images' / 'train'} is >= {config.min_height}px tall"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    mirrored = (
        _mirror_train_images(kept, labels_dir, out_dir / MIRROR_NAME)
        if config.labels_root is not None
        else {}
    )
    entries: list[str] = []
    for image in kept:
        target = mirrored.get(image, image)
        entries.extend([str(target)] * multiplicity[image])

    extra_summary: list[dict[str, Any]] = []
    extra_entries = 0
    for extra in config.extra:
        extra_root = extra.root.resolve()
        extra_train = _image_files(extra_root / "images" / "train")
        if not extra_train:
            raise FileNotFoundError(f"no images under {extra_root / 'images' / 'train'}")
        if not (extra_root / "labels" / "train").is_dir():
            raise FileNotFoundError(f"labels not found: {extra_root / 'labels' / 'train'}")
        for image in extra_train:
            entries.extend([str(image)] * extra.repeat)
        extra_entries += len(extra_train) * extra.repeat
        extra_summary.append(
            {
                "root": str(extra_root),
                "repeat": extra.repeat,
                "train_images": len(extra_train),
                "train_entries": len(extra_train) * extra.repeat,
            }
        )

    val_h160 = [image for image in val_images if image_height(image) >= config.val_min_height]
    train_list = out_dir / TRAIN_LIST_NAME
    train_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
    val_h160_list = out_dir / VAL_H160_LIST_NAME
    val_h160_list.write_text("\n".join(str(path) for path in val_h160) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "dataset_root": str(root.resolve()),
        "labels_root": str(config.labels_root.resolve()) if config.labels_root else None,
        "mirror": str(out_dir / MIRROR_NAME) if mirrored else None,
        "schema": schema.name,
        "min_height": int(config.min_height),
        "hand_repeat": int(config.hand_repeat),
        "face_repeat": int(config.face_repeat),
        "val_min_height": int(config.val_min_height),
        "hand_min_visible": HAND_MIN_VISIBLE,
        "face_min_visible": FACE_MIN_VISIBLE,
        "train_images_total": len(train_images),
        "train_images_kept": len(kept),
        "train_entries_before_repeat": len(kept),
        "train_entries": sum(multiplicity.values()),
        "train_images_repeated": sum(1 for count in multiplicity.values() if count > 1),
        "hand_images_total": hand_all,
        "hand_images_kept": hand_kept,
        "hand_share_all": round(hand_all / len(train_images), 4),
        "hand_share_kept": round(hand_kept / len(kept), 4),
        "face_images_total": face_all,
        "face_images_kept": face_kept,
        "face_share_all": round(face_all / len(train_images), 4),
        "face_share_kept": round(face_kept / len(kept), 4),
        "labels_missing": missing_labels,
        "val_images_total": len(val_images),
        "val_images_h160": len(val_h160),
        "extra": extra_summary,
        "extra_entries": extra_entries,
        "train_entries_total": len(entries),
        "train_list": str(train_list.resolve()),
        "val_h160_list": str(val_h160_list.resolve()),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    summary_path = out_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"curated train list: {len(train_images)} -> {len(kept)} crops "
        f"(h >= {config.min_height}px) -> {len(entries)} entries "
        f"(hand x{config.hand_repeat}, face x{config.face_repeat}); "
        f"hands {summary['hand_share_all']:.0%} -> {summary['hand_share_kept']:.0%}, "
        f"faces {summary['face_share_all']:.0%} -> {summary['face_share_kept']:.0%}",
        flush=True,
    )
    print(
        f"val: {len(val_images)} crops, {len(val_h160)} at h >= {config.val_min_height}px; "
        f"wrote {train_list} and {val_h160_list}",
        flush=True,
    )
    return CurationResult(
        train_list=train_list,
        val_h160_list=val_h160_list,
        summary_path=summary_path,
        summary=summary,
    )
