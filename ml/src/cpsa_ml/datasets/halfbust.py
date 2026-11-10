"""Build a YOLO-pose half-bust dataset from COCO-WholeBody annotations.

The builder reads the COCO-WholeBody JSON format (COCO 2017 images plus
``coco_wholebody_*_v1.0.json``) and writes one cropped image per person.  The
crop starts above the head and ends at the hip line (or at an aspect-ratio
limited box) so the framing matches a seated subject filmed at 0.45-0.7 m.

Labels are Ultralytics YOLO-pose text files::

    class cx cy w h x1 y1 v1 x2 y2 v2 ...

Coordinates are normalized to the crop.  Keypoints outside the crop get
``v=0`` and ``x=y=0``.

The mapping, label formatting and dataset-writing helpers are public because
:mod:`cpsa_ml.datasets.fullframe` reuses them unchanged for whole images.

Example::

    python -m cpsa_ml.datasets.halfbust \
        --coco-root data/coco \
        --wholebody-json annotations/coco_wholebody_train_v1.0.json \
        --split train --schema halfbust62 --out data/halfbust62 \
        --occlusion-p 0.3 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from cpsa_ml.schema import KeypointSchema, get_schema

__all__ = [
    "BuildConfig",
    "BuildReport",
    "build_halfbust_dataset",
    "format_label",
    "load_image",
    "main",
    "map_keypoints",
    "resolve_split_dir",
    "update_stats",
    "wholebody_keypoints",
    "write_dataset_yaml",
]

LOGGER = logging.getLogger(__name__)

HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, eyes, ears
SHOULDER_KEYPOINTS = (5, 6)
HIP_KEYPOINTS = (11, 12)

#: Normalized ``cx cy w h`` of a label written for a whole-crop person.
FULL_CROP_BOX = (0.5, 0.5, 1.0, 1.0)

LEFT_HAND_KEYPOINTS = tuple(range(91, 112))
RIGHT_HAND_KEYPOINTS = tuple(range(112, 133))
FACE_KEYPOINTS = tuple(range(23, 91))


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for :func:`build_halfbust_dataset`."""

    coco_root: Path
    wholebody_json: Path
    split: str
    out: Path
    schema: KeypointSchema | str = "halfbust62"
    max_images: int | None = None
    occlusion_p: float = 0.0
    seed: int = 0
    margin: float = 0.18
    aspect: float = 0.75
    min_visible: int = 3
    jpeg_quality: int = 95
    max_hand_crops: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "coco_root", Path(self.coco_root))
        object.__setattr__(self, "wholebody_json", Path(self.wholebody_json))
        object.__setattr__(self, "out", Path(self.out))
        object.__setattr__(self, "schema", get_schema(self.schema))
        if self.split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {self.split!r}")
        if not 0.0 <= self.occlusion_p <= 1.0:
            raise ValueError("occlusion_p must be in [0, 1]")
        if not 0.0 < self.aspect <= 2.0:
            raise ValueError("aspect must be in (0, 2]")
        if self.max_images is not None and self.max_images <= 0:
            raise ValueError("max_images must be positive")
        if self.min_visible < 0:
            raise ValueError("min_visible must be non-negative")


@dataclass
class BuildReport:
    """Counters returned by :func:`build_halfbust_dataset`."""

    schema: str
    split: str
    out: Path
    source_images: int = 0
    images_written: int = 0
    labels_written: int = 0
    annotations_used: int = 0
    annotations_skipped: int = 0
    occlusions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "split": self.split,
            "out": str(self.out),
            "source_images": self.source_images,
            "images_written": self.images_written,
            "labels_written": self.labels_written,
            "annotations_used": self.annotations_used,
            "annotations_skipped": self.annotations_skipped,
            "occlusions": self.occlusions,
        }


@dataclass
class _HandCrop:
    image: Image.Image
    annotation_id: int


def _as_keypoints(
    value: Any, count: int, field_name: str, annotation_id: Any
) -> NDArray[np.float32]:
    if value is None:
        return np.zeros((count, 3), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.size != count * 3:
        raise ValueError(
            f"annotation {annotation_id}: {field_name} has {array.size} values, "
            f"expected {count * 3}"
        )
    return array.reshape(count, 3)


def wholebody_keypoints(annotation: Mapping[str, Any]) -> NDArray[np.float32]:
    """Assemble the 133 COCO-WholeBody keypoints for one annotation."""

    annotation_id = annotation.get("id", "?")
    body = _as_keypoints(annotation.get("keypoints"), 17, "keypoints", annotation_id)
    foot = _as_keypoints(annotation.get("foot_kpts"), 6, "foot_kpts", annotation_id)
    face = _as_keypoints(annotation.get("face_kpts"), 68, "face_kpts", annotation_id)
    left = _as_keypoints(annotation.get("lefthand_kpts"), 21, "lefthand_kpts", annotation_id)
    right = _as_keypoints(annotation.get("righthand_kpts"), 21, "righthand_kpts", annotation_id)

    out = np.zeros((133, 3), dtype=np.float32)
    out[0:17] = body
    out[17:23] = foot
    out[23:91] = face
    out[91:112] = left
    out[112:133] = right

    for flag, slc in (
        ("foot_valid", slice(17, 23)),
        ("face_valid", slice(23, 91)),
        ("lefthand_valid", slice(91, 112)),
        ("righthand_valid", slice(112, 133)),
    ):
        if not bool(annotation.get(flag, True)):
            out[slc] = 0.0
    return out


def _has_head_and_shoulders(keypoints: NDArray[np.float32]) -> bool:
    head = int(np.count_nonzero(keypoints[list(HEAD_KEYPOINTS), 2] > 0))
    shoulders = int(np.count_nonzero(keypoints[list(SHOULDER_KEYPOINTS), 2] > 0))
    return head >= 2 and shoulders >= 2


def _clip_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    if box_w > width:
        box_w = float(width)
    if box_h > height:
        box_h = float(height)
    x1 = min(max(0.0, x1), width - box_w)
    y1 = min(max(0.0, y1), height - box_h)
    return x1, y1, x1 + box_w, y1 + box_h


def _head_box(
    keypoints: NDArray[np.float32], image_w: int, image_h: int, margin: float
) -> tuple[float, float, float, float] | None:
    """Estimate a box around the whole head (including hair)."""

    face = keypoints[list(FACE_KEYPOINTS)]
    face = face[face[:, 2] > 0]
    if len(face) >= 5:
        xs, ys = face[:, 0], face[:, 1]
        face_w = max(float(xs.max() - xs.min()), 4.0)
        face_h = max(float(ys.max() - ys.min()), 4.0)
        return _clip_box(
            float(xs.min()) - margin * face_w,
            float(ys.min()) - 0.55 * face_h,
            float(xs.max()) + margin * face_w,
            float(ys.max()) + 0.20 * face_h,
            image_w,
            image_h,
        )

    head = keypoints[list(HEAD_KEYPOINTS)]
    head = head[head[:, 2] > 0]
    if len(head) < 2:
        return None

    xs, ys = head[:, 0], head[:, 1]
    span_w = max(float(xs.max() - xs.min()), 4.0)
    span_h = max(float(ys.max() - ys.min()), 4.0)
    # Eyes and ears span less than a full head; add a conservative margin.
    head_w = max(span_w * 1.6, span_h * 1.2)
    head_h = max(span_h * 2.4, head_w * 1.15)
    cx = 0.5 * float(xs.min() + xs.max())
    cy = 0.5 * float(ys.min() + ys.max())
    return _clip_box(
        cx - head_w / 2,
        cy - 0.85 * head_h,
        cx + head_w / 2,
        cy + 0.30 * head_h,
        image_w,
        image_h,
    )


def _half_bust_box(
    keypoints: NDArray[np.float32],
    head: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    aspect: float,
) -> tuple[int, int, int, int]:
    """Crop from the head top to the hips, constrained to ``aspect``."""

    shoulders = keypoints[list(SHOULDER_KEYPOINTS)]
    shoulders = shoulders[shoulders[:, 2] > 0]
    if len(shoulders) != 2:
        raise ValueError("head+shoulder eligibility should have guaranteed two shoulders")

    shoulder_w = float(abs(shoulders[0, 0] - shoulders[1, 0]))
    shoulder_x = float(np.mean(shoulders[:, 0]))
    shoulder_y = float(np.mean(shoulders[:, 1]))
    if shoulder_w < 1.0:
        shoulder_w = max((head[2] - head[0]) * 1.5, image_w * 0.1)

    hips = keypoints[list(HIP_KEYPOINTS)]
    hips = hips[hips[:, 2] > 0]
    # Fallback: shoulder-to-hip is ~1.15 shoulder widths.
    hip_y = float(np.mean(hips[:, 1])) if len(hips) == 2 else shoulder_y + 1.15 * shoulder_w

    head_w = head[2] - head[0]
    head_h = head[3] - head[1]
    top = head[1]
    bottom = max(hip_y, head[3] + 0.20 * head_h, top + 2.5 * head_h)
    centre_x = 0.5 * (0.5 * (head[0] + head[2]) + shoulder_x)

    natural_w = max(1.8 * shoulder_w, 2.2 * head_w)
    height = bottom - top
    width = max(natural_w, aspect * height)
    height = width / aspect

    x1 = centre_x - width / 2
    x2 = centre_x + width / 2
    y1 = top
    y2 = top + height

    # Shift before clipping so the head margin survives when possible.
    if y2 > image_h:
        y1 -= y2 - image_h
        y2 = float(image_h)
    if x2 > image_w:
        x1 -= x2 - image_w
        x2 = float(image_w)
    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0

    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, image_w, image_h)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def map_keypoints(
    keypoints: NDArray[np.float32],
    schema: KeypointSchema,
    box: tuple[int, int, int, int],
) -> NDArray[np.float32]:
    """Project the 133 keypoints into ``schema`` order, normalized to ``box``.

    Keypoints outside the box (or without visibility) are written as
    ``x = y = visibility = 0``.  A full-frame caller passes the whole image as
    ``box`` so nothing is dropped.
    """

    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    out = np.zeros((schema.num_keypoints, 3), dtype=np.float32)
    for out_i, wholebody_i in enumerate(schema.wholebody_index):
        x, y, visibility = keypoints[wholebody_i]
        if visibility <= 0 or not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < x1 - 0.5 or x > x2 + 0.5 or y < y1 - 0.5 or y > y2 + 0.5:
            continue
        out[out_i, 0] = float(np.clip((x - x1) / width, 0.0, 1.0))
        out[out_i, 1] = float(np.clip((y - y1) / height, 0.0, 1.0))
        out[out_i, 2] = 1.0
    return out


def format_label(
    class_id: int,
    keypoints: NDArray[np.float32],
    box: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 1.0),
) -> str:
    """One YOLO-pose label line; ``box`` is normalized ``cx cy w h``.

    The half-bust builder always crops one person to the whole image, so it
    keeps the default box.  A full-frame caller passes the normalized COCO
    person box so the box matches the true person extent.
    """

    values = [str(class_id)]
    if box == FULL_CROP_BOX:
        # Keep the half-bust label text byte-identical to earlier releases.
        values.extend(("0.5", "0.5", "1.0", "1.0"))
    else:
        values.extend(f"{value:.6f}" for value in box)
    for x, y, visibility in keypoints:
        if visibility > 0:
            values.extend((f"{x:.6f}", f"{y:.6f}", "1"))
        else:
            values.extend(("0.0", "0.0", "0"))
    return " ".join(values)


def _face_box(keypoints: NDArray[np.float32]) -> tuple[float, float, float, float] | None:
    face = keypoints[list(FACE_KEYPOINTS)]
    face = face[face[:, 2] > 0]
    if len(face) < 3:
        head = keypoints[list(HEAD_KEYPOINTS)]
        face = head[head[:, 2] > 0]
    if len(face) < 3:
        return None
    xs, ys = face[:, 0], face[:, 1]
    pad_x = max(2.0, 0.10 * float(xs.max() - xs.min()))
    pad_y = max(2.0, 0.10 * float(ys.max() - ys.min()))
    return (
        float(xs.min()) - pad_x,
        float(ys.min()) - pad_y,
        float(xs.max()) + pad_x,
        float(ys.max()) + pad_y,
    )


def load_image(path: Path) -> Image.Image | None:
    """Read ``path`` as an RGB image, or ``None`` when it cannot be decoded."""

    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        LOGGER.warning("cannot read image %s: %s", path, exc)
        return None


def _crop_hand(
    image: Image.Image,
    keypoints: NDArray[np.float32],
    indices: Sequence[int],
    margin: float = 0.25,
) -> Image.Image | None:
    hand = keypoints[list(indices)]
    hand = hand[hand[:, 2] > 0]
    if len(hand) < 3:
        return None
    xs, ys = hand[:, 0], hand[:, 1]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    if width < 4.0 or height < 4.0:
        return None
    x1 = int(max(0, xs.min() - margin * width))
    y1 = int(max(0, ys.min() - margin * height))
    x2 = int(min(image.width, xs.max() + margin * width))
    y2 = int(min(image.height, ys.max() + margin * height))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = image.crop((x1, y1, x2, y2)).convert("RGBA")
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    radius = max(3.0, 0.20 * min(crop.width, crop.height))
    for x, y, _ in hand:
        cx, cy = x - x1, y - y1
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    crop.putalpha(mask)
    return crop


def _build_hand_pool(
    data: Mapping[str, Any],
    image_paths: Mapping[int, Path],
    rng: random.Random,
    max_crops: int,
) -> list[_HandCrop]:
    grouped: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in image_paths:
            continue
        if bool(annotation.get("lefthand_valid", True)):
            grouped[image_id].append((int(annotation["id"]), LEFT_HAND_KEYPOINTS))
        if bool(annotation.get("righthand_valid", True)):
            grouped[image_id].append((int(annotation["id"]), RIGHT_HAND_KEYPOINTS))

    if not grouped:
        return []

    annotations_by_image: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for annotation in data.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])][int(annotation["id"])] = annotation

    image_ids = sorted(grouped)
    rng.shuffle(image_ids)
    pool: list[_HandCrop] = []
    for image_id in image_ids:
        image = load_image(image_paths[image_id])
        if image is None:
            continue
        for annotation_id, indices in grouped[image_id]:
            annotation = annotations_by_image[image_id][annotation_id]
            hand = _crop_hand(image, wholebody_keypoints(annotation), indices)
            if hand is not None:
                pool.append(_HandCrop(image=hand, annotation_id=annotation_id))
                if len(pool) >= max_crops:
                    return pool
    return pool


def _paste_hand_over_face(
    crop: Image.Image,
    keypoints: NDArray[np.float32],
    source_box: tuple[int, int, int, int],
    pool: Sequence[_HandCrop],
    rng: random.Random,
    target_annotation_id: int,
) -> bool:
    face = _face_box(keypoints)
    candidates = [hand for hand in pool if hand.annotation_id != target_annotation_id]
    if face is None or not candidates:
        return False
    x1, y1, x2, y2 = source_box
    face_x1, face_y1, face_x2, face_y2 = face
    face_x1 = max(0.0, face_x1 - x1)
    face_y1 = max(0.0, face_y1 - y1)
    face_x2 = min(float(crop.width), face_x2 - x1)
    face_y2 = min(float(crop.height), face_y2 - y1)
    face_w = face_x2 - face_x1
    face_h = face_y2 - face_y1
    if face_w < 8 or face_h < 8:
        return False

    hand = rng.choice(candidates).image
    target_w = max(12.0, 1.25 * face_w)
    target_h = max(12.0, 1.70 * face_h)
    scale = min(target_w / hand.width, target_h / hand.height)
    new_size = (max(1, int(round(hand.width * scale))), max(1, int(round(hand.height * scale))))
    hand = hand.resize(new_size, Image.Resampling.LANCZOS)

    jitter_x = rng.uniform(-0.08, 0.08) * face_w
    jitter_y = rng.uniform(-0.08, 0.08) * face_h
    paste_x = int(round(face_x1 + (face_w - hand.width) / 2 + jitter_x))
    paste_y = int(round(face_y1 + (face_h - hand.height) / 2 + jitter_y))
    paste_x = max(0, min(crop.width - hand.width, paste_x))
    paste_y = max(0, min(crop.height - hand.height, paste_y))
    crop.paste(hand, (paste_x, paste_y), hand)
    return True


def resolve_split_dir(coco_root: Path, split: str) -> Path:
    """Return the COCO image directory of ``split`` (``*2017`` or plain)."""

    for candidate in (coco_root / f"{split}2017", coco_root / split, coco_root):
        if candidate.is_dir():
            return candidate
    return coco_root


def write_dataset_yaml(out: Path, schema: KeypointSchema, split: str) -> None:
    """Write ``data.yaml`` with relative split paths (portable on Kaggle)."""

    train_exists = (out / "images" / "train").is_dir()
    val_exists = (out / "images" / "val").is_dir()
    current = f"images/{split}"
    # Deliberately omit an absolute ``path``: Ultralytics then resolves the
    # split paths relative to the YAML file, so the dataset still works after
    # it is published and re-attached at a different /kaggle/input mount.
    data = {
        "train": "images/train" if train_exists else current,
        "val": "images/val" if val_exists else current,
        "kpt_shape": list(schema.kpt_shape),
        "flip_idx": list(schema.flip_idx),
        "names": {0: "person"},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def update_stats(out: Path, report: BuildReport) -> None:
    """Merge one split's counters into ``stats.json``."""

    path = out / "stats.json"
    stats: dict[str, Any] = {}
    if path.is_file():
        try:
            stats = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats = {}
    stats["schema"] = report.schema
    stats.setdefault("splits", {})[report.split] = report.to_dict()
    path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_halfbust_dataset(config: BuildConfig) -> BuildReport:
    """Convert COCO-WholeBody annotations into a YOLO-pose dataset.

    Args:
        config: Paths and augmentation settings.

    Returns:
        Counters describing what was written.

    Raises:
        ValueError: If the JSON is malformed or contains no usable images.
    """

    schema = get_schema(config.schema)
    try:
        data = json.loads(config.wholebody_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read COCO-WholeBody JSON {config.wholebody_json}: {exc}") from exc

    images = data.get("images")
    annotations = data.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO JSON must contain 'images' and 'annotations' lists")
    if not images:
        raise ValueError("COCO JSON contains no images")

    split_dir = resolve_split_dir(config.coco_root, config.split)
    image_paths: dict[int, Path] = {}
    image_records: dict[int, Mapping[str, Any]] = {}
    for image in images:
        image_id = int(image["id"])
        path = split_dir / str(image["file_name"])
        if not path.is_file():
            alternate = config.coco_root / str(image["file_name"])
            if alternate.is_file():
                path = alternate
            else:
                LOGGER.warning("missing image %s", path)
                continue
        image_paths[image_id] = path
        image_records[image_id] = image

    if not image_paths:
        raise ValueError(f"no COCO images found under {config.coco_root}")

    annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)):
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    rng = random.Random(config.seed)
    hand_pool: list[_HandCrop] = []
    if config.occlusion_p > 0:
        hand_pool = _build_hand_pool(data, image_paths, rng, config.max_hand_crops)
        if not hand_pool:
            LOGGER.warning("occlusion_p > 0 but no hand crops were found")

    image_dir = config.out / "images" / config.split
    label_dir = config.out / "labels" / config.split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    report = BuildReport(schema=schema.name, split=config.split, out=config.out)
    ordered_ids = sorted(
        image_records, key=lambda value: (image_records[value]["file_name"], value)
    )
    for image_id in ordered_ids:
        if config.max_images is not None and report.source_images >= config.max_images:
            break
        report.source_images += 1
        image_annotations = annotations_by_image.get(image_id, [])
        if not image_annotations:
            continue
        image = load_image(image_paths[image_id])
        if image is None:
            continue

        for annotation in image_annotations:
            keypoints = wholebody_keypoints(annotation)
            if not _has_head_and_shoulders(keypoints):
                report.annotations_skipped += 1
                continue
            head = _head_box(keypoints, image.width, image.height, config.margin)
            if head is None:
                report.annotations_skipped += 1
                continue
            box = _half_bust_box(keypoints, head, image.width, image.height, config.aspect)
            crop = image.crop(box)
            mapped = map_keypoints(keypoints, schema, box)
            if int(np.count_nonzero(mapped[:, 2] > 0)) < config.min_visible:
                report.annotations_skipped += 1
                continue

            if (
                config.occlusion_p > 0
                and hand_pool
                and rng.random() < config.occlusion_p
                and _paste_hand_over_face(
                    crop, keypoints, box, hand_pool, rng, int(annotation["id"])
                )
            ):
                report.occlusions += 1

            stem = Path(str(image_records[image_id]["file_name"])).stem.replace(" ", "_")
            output_stem = f"{image_id:012d}_{stem}_a{int(annotation['id'])}"
            crop.save(
                image_dir / f"{output_stem}.jpg",
                format="JPEG",
                quality=config.jpeg_quality,
                subsampling=0,
            )
            (label_dir / f"{output_stem}.txt").write_text(
                format_label(0, mapped) + "\n", encoding="utf-8"
            )
            report.images_written += 1
            report.labels_written += 1
            report.annotations_used += 1

    write_dataset_yaml(config.out, schema, config.split)
    (config.out / "schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    update_stats(config.out, report)
    if report.images_written == 0:
        LOGGER.warning("no images were written; check the JSON split and visibility flags")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True, help="COCO 2017 root")
    parser.add_argument("--wholebody-json", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--schema", choices=("halfbust62", "coco17"), default="halfbust62")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--occlusion-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.18)
    parser.add_argument("--min-visible", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = build_halfbust_dataset(
        BuildConfig(
            coco_root=args.coco_root,
            wholebody_json=args.wholebody_json,
            split=args.split,
            schema=args.schema,
            out=args.out,
            max_images=args.max_images,
            occlusion_p=args.occlusion_p,
            seed=args.seed,
            margin=args.margin,
            min_visible=args.min_visible,
            jpeg_quality=args.jpeg_quality,
        )
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
