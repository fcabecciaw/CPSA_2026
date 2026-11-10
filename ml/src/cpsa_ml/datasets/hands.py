"""Build a YOLO-pose hand-crop dataset for the stage-2 hand landmark model.

The stage-1 body model finds the person; the stage-2 hand model only sees a
square crop around one hand and returns 21 MediaPipe landmarks.  Every source
sample is therefore stored as a hand box plus 21 landmarks in image
coordinates, and this module re-crops it: the box is enlarged 1.6-2.2x, the
centre and the size are jittered, the window is kept square, cut out and
resized to ``imgsz`` (192 px by default).  Labels are Ultralytics YOLO-pose text
files::

    class cx cy w h x1 y1 v1 x2 y2 v2 ...

``class`` is always 0 (``hand``) and the box is the hand box inside the crop.
Left and right hands share the class; the handedness is only counted in
``stats.json``.  ``flip_idx`` is the identity because a mirrored crop keeps each
landmark on the same finger joint.  Keypoint order is MediaPipe: wrist, then
thumb, index, middle, ring and pinky with four joints each.

Sources
-------
``hagrid``
    HaGRIDv2 ``annotations_with_landmarks`` JSON files.  Each entry holds
    per-hand normalized ``bboxes`` (top-left ``x y w h``) and 21 normalized
    MediaPipe ``hand_landmarks``.  The crop box is the union of both, because
    the gesture box does not always contain every fingertip.
``ultralytics-hand-keypoints``
    The Ultralytics hand-keypoints dataset (YOLO-pose labels, 21 keypoints,
    full-frame boxes).  Also re-cropped, so both sources share one geometry.

Splits follow the source split: ``train`` images become training crops,
``val``/``test`` images become validation crops.  A deterministic
``val_ratio`` share of the training images is additionally held out so the
validation split is never empty when a source only ships training images.

Example::

    python -m cpsa_ml.datasets.hands \\
        --hagrid-images data/hagrid/hagrid_30k \\
        --hagrid-annotations data/hagrid/annotations \\
        --ultralytics-root data/hand-keypoints \\
        --out data/hand21
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from cpsa_ml.datasets.halfbust import format_label, load_image
from cpsa_ml.schema import KeypointSchema, get_schema

__all__ = [
    "SOURCE_LICENSES",
    "SPLITS",
    "HagridSource",
    "HandBuildConfig",
    "HandBuildReport",
    "HandSample",
    "SplitStats",
    "UltralyticsHandSource",
    "build_hand_dataset",
    "crop_hand_sample",
    "hand_crop_window",
    "iter_hagrid_samples",
    "iter_ultralytics_samples",
    "main",
    "write_hand_crops",
    "write_hand_dataset_yaml",
]

LOGGER = logging.getLogger(__name__)

SPLITS = ("train", "val")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
KEYPOINTS = 21
#: Smallest accepted source box (pixels); smaller hands crop to mush.
MIN_SOURCE_BOX = 8.0
#: Directory-name prefixes of the HaGRID 384p sample (`train_val_peace`).
HAGRID_SPLIT_PREFIXES = ("train_val_", "train_", "val_", "test_")

HAGRID = "hagrid"
ULTRALYTICS = "ultralytics-hand-keypoints"

#: Provenance recorded in ``stats.json`` and repeated in ``ml/README.md``.
SOURCE_LICENSES: dict[str, dict[str, str]] = {
    HAGRID: {
        "name": "HaGRIDv2 (annotations_with_landmarks + 384p sample)",
        "license": "CC BY-SA 4.0 variant (share-alike, attribution)",
        "url": "https://github.com/hukenovs/hagrid",
    },
    ULTRALYTICS: {
        "name": "Ultralytics hand-keypoints",
        "license": "CC BY-NC-SA 4.0 (NON-COMMERCIAL; no commercial redistribution)",
        "url": "https://docs.ultralytics.com/datasets/pose/hand-keypoints/",
    },
}


def _image_files(directory: Path) -> list[Path]:
    """Return the decodable image paths of ``directory`` sorted by name."""

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _image_size(path: Path) -> tuple[int, int] | None:
    """Read a JPEG/PNG header and return ``(width, height)``."""

    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError) as exc:
        LOGGER.warning("cannot read image header %s: %s", path, exc)
        return None


def _find_image(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class HandSample:
    """One hand: a bounding box and 21 landmarks in source-image pixels.

    Args:
        image_path: Source image.
        image_size: Source image ``(width, height)``.
        box: Hand box ``(x1, y1, x2, y2)`` in pixels.
        keypoints: ``(21, 3)`` array of ``(x, y, visibility)`` in pixels.
        source: Source name used for statistics.
        origin_split: Split of the source (``train``/``val``/``test``).
        label: Source label (HaGRID gesture) for statistics only.
        handedness: ``"left"``/``"right"`` when the source knows it.
    """

    image_path: Path
    image_size: tuple[int, int]
    box: tuple[float, float, float, float]
    keypoints: NDArray[np.float32]
    source: str
    origin_split: str
    label: str = ""
    handedness: str = ""

    def __post_init__(self) -> None:
        array = np.asarray(self.keypoints, dtype=np.float32)
        if array.shape != (KEYPOINTS, 3):
            raise ValueError(f"expected {KEYPOINTS} keypoints, got {array.shape}")
        object.__setattr__(self, "keypoints", array)
        object.__setattr__(self, "image_path", Path(self.image_path))

    @property
    def key(self) -> str:
        """Stable identity of this hand, independent of iteration order."""

        x1, y1, x2, y2 = (round(value, 2) for value in self.box)
        return f"{self.source}|{self.image_path.name}|{x1},{y1},{x2},{y2}"


@dataclass(frozen=True)
class HagridSource:
    """HaGRIDv2 images plus the ``annotations_with_landmarks`` JSON tree.

    Attributes:
        images: Directory whose sub-directories hold the images of one gesture
            (``hagrid_30k`` of the 384p sample).  A directory of images directly
            inside it is also accepted.
        annotations: ``annotations`` directory with ``train``/``val``/``test``
            sub-directories, or a single annotation JSON.
        splits: Annotation splits to read; missing ones are skipped.
    """

    images: Path
    annotations: Path
    splits: tuple[str, ...] = ("train", "val", "test")

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", Path(self.images))
        object.__setattr__(self, "annotations", Path(self.annotations))


@dataclass(frozen=True)
class UltralyticsHandSource:
    """The Ultralytics hand-keypoints tree (``images/<split>``, ``labels/<split>``)."""

    root: Path
    splits: tuple[str, ...] = ("train", "val")

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))


@dataclass(frozen=True)
class HandBuildConfig:
    """Crop geometry, output paths and limits.

    Attributes:
        out: Dataset root; ``images/<split>``, ``labels/<split>``, ``data.yaml``,
            ``schema.json`` and ``stats.json`` are written below it.
        schema: Keypoint schema; must be the 21-keypoint hand schema.
        imgsz: Square output crop size.
        expand: Uniform range multiplying ``max(width, height)`` of the hand.
        shift: Random centre offset as a fraction of the crop side.
        scale_jitter: Relative random variation of the crop side.
        min_visible: Minimum landmarks inside the crop for a usable label.
        val_ratio: Deterministic share of training images held out for validation.
        max_images: Maximum crops written per split (``None`` for no limit).
        jpeg_quality: JPEG quality of the written crops.
        seed: Seed of the deterministic per-sample jitter.
        overwrite: Re-crop samples whose files already exist.
    """

    out: Path
    schema: str | KeypointSchema = "hand21"
    imgsz: int = 192
    expand: tuple[float, float] = (1.6, 2.2)
    shift: float = 0.06
    scale_jitter: float = 0.10
    min_visible: int = 5
    val_ratio: float = 0.1
    max_images: int | None = None
    jpeg_quality: int = 95
    seed: int = 0
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "out", Path(self.out))
        object.__setattr__(self, "schema", get_schema(self.schema))
        low, high = self.expand
        if low < 1.0 or high < low:
            raise ValueError("expand must be a non-decreasing range starting at >= 1.0")
        if self.imgsz < 32:
            raise ValueError("imgsz must be at least 32")
        if not 0.0 <= self.shift < 0.5:
            raise ValueError("shift must be in [0, 0.5)")
        if not 0.0 <= self.scale_jitter < 0.9:
            raise ValueError("scale_jitter must be in [0, 0.9)")
        if self.min_visible < 1 or self.min_visible > KEYPOINTS:
            raise ValueError(f"min_visible must be in [1, {KEYPOINTS}]")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.max_images is not None and self.max_images <= 0:
            raise ValueError("max_images must be positive")


@dataclass
class SplitStats:
    """Per-split counters written to ``stats.json``."""

    source_images: int = 0
    hands_seen: int = 0
    images_written: int = 0
    labels_written: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    handedness: Counter[str] = field(default_factory=Counter)
    sources: Counter[str] = field(default_factory=Counter)
    labels: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_images": self.source_images,
            "hands_seen": self.hands_seen,
            "images_written": self.images_written,
            "labels_written": self.labels_written,
            "skip_reasons": dict(sorted(self.skipped.items())),
            "handedness": dict(sorted(self.handedness.items())),
            "sources": dict(sorted(self.sources.items())),
            "labels": dict(sorted(self.labels.items())),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SplitStats:
        """Restore counters persisted in ``stats.json`` (resumable runs)."""

        return cls(
            source_images=int(payload.get("source_images", 0)),
            hands_seen=int(payload.get("hands_seen", 0)),
            images_written=int(payload.get("images_written", 0)),
            labels_written=int(payload.get("labels_written", 0)),
            skipped=Counter(payload.get("skip_reasons", {})),
            handedness=Counter(payload.get("handedness", {})),
            sources=Counter(payload.get("sources", {})),
            labels=Counter(payload.get("labels", {})),
        )


@dataclass
class HandBuildReport:
    """Result of :func:`build_hand_dataset`."""

    schema: str
    out: Path
    imgsz: int
    splits: dict[str, SplitStats]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "out": str(self.out),
            "imgsz": self.imgsz,
            "splits": {split: stats.to_dict() for split, stats in self.splits.items()},
            "totals": {
                "images": sum(stats.images_written for stats in self.splits.values()),
                "labels": sum(stats.labels_written for stats in self.splits.values()),
            },
        }


def hand_crop_window(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    rng: random.Random,
    config: HandBuildConfig,
) -> tuple[int, int, int, int]:
    """Return the square, jittered crop window ``(x1, y1, x2, y2)`` in pixels.

    The window is clamped inside the image, so no padding is ever introduced;
    a hand near the border loses context instead of gaining black pixels.
    """

    width, height = image_size
    x1, y1, x2, y2 = box
    span = max(x2 - x1, y2 - y1, 1.0)
    side = span * rng.uniform(*config.expand)
    side *= 1.0 + rng.uniform(-config.scale_jitter, config.scale_jitter)
    side = min(max(side, 2.0), float(min(width, height)))
    centre_x = 0.5 * (x1 + x2) + rng.uniform(-config.shift, config.shift) * side
    centre_y = 0.5 * (y1 + y2) + rng.uniform(-config.shift, config.shift) * side

    left = min(max(centre_x - side / 2, 0.0), width - side)
    top = min(max(centre_y - side / 2, 0.0), height - side)
    left_i = int(round(left))
    top_i = int(round(top))
    right_i = min(width, max(left_i + 2, int(round(left + side))))
    bottom_i = min(height, max(top_i + 2, int(round(top + side))))
    return left_i, top_i, right_i, bottom_i


def crop_hand_sample(
    image: Image.Image,
    sample: HandSample,
    rng: random.Random,
    config: HandBuildConfig,
) -> tuple[Image.Image, str] | None:
    """Crop one hand and return ``(crop, label_line)``, or ``None`` to skip.

    ``None`` means the crop holds fewer than ``config.min_visible`` landmarks or
    the hand box does not overlap the window at all.
    """

    width, height = image.size
    window = hand_crop_window(sample.box, (width, height), rng, config)
    window_x1, window_y1, window_x2, window_y2 = window
    crop_width = window_x2 - window_x1
    crop_height = window_y2 - window_y1
    if crop_width < 2 or crop_height < 2:
        return None

    keypoints = np.zeros((KEYPOINTS, 3), dtype=np.float32)
    for index, (x, y, visibility) in enumerate(sample.keypoints):
        if visibility <= 0 or not (np.isfinite(x) and np.isfinite(y)):
            continue
        if not (window_x1 - 0.5 <= x <= window_x2 + 0.5):
            continue
        if not (window_y1 - 0.5 <= y <= window_y2 + 0.5):
            continue
        keypoints[index] = (
            float(np.clip((x - window_x1) / crop_width, 0.0, 1.0)),
            float(np.clip((y - window_y1) / crop_height, 0.0, 1.0)),
            1.0,
        )
    if int(np.count_nonzero(keypoints[:, 2] > 0)) < config.min_visible:
        return None

    box_x1, box_y1, box_x2, box_y2 = sample.box
    left = float(np.clip((box_x1 - window_x1) / crop_width, 0.0, 1.0))
    right = float(np.clip((box_x2 - window_x1) / crop_width, 0.0, 1.0))
    top = float(np.clip((box_y1 - window_y1) / crop_height, 0.0, 1.0))
    bottom = float(np.clip((box_y2 - window_y1) / crop_height, 0.0, 1.0))
    if right - left <= 0.0 or bottom - top <= 0.0:
        return None

    crop = image.crop(window)
    box = (0.5 * (left + right), 0.5 * (top + bottom), right - left, bottom - top)
    return crop, format_label(0, keypoints, box)


def _output_stem(sample: HandSample) -> str:
    digest = hashlib.sha1(sample.key.encode("utf-8")).hexdigest()[:10]
    prefix = {"hagrid": "hagrid", "ultralytics-hand-keypoints": "ultra", "recorded": "rec"}.get(
        sample.source, "hand"
    )
    return f"{prefix}_{sample.image_path.stem[:32]}_{digest}"


def _write_hand_crop(
    config: HandBuildConfig,
    split: str,
    sample: HandSample,
    stats: SplitStats,
) -> bool:
    """Write one crop; return ``True`` when the sample was counted as written."""

    image_dir = config.out / "images" / split
    label_dir = config.out / "labels" / split
    stem = _output_stem(sample)
    label_path = label_dir / f"{stem}.txt"
    image_path = image_dir / f"{stem}.jpg"
    if label_path.is_file() and image_path.is_file() and not config.overwrite:
        stats.images_written += 1
        stats.labels_written += 1
        stats.skipped["existing"] += 1
        stats.sources[sample.source] += 1
        stats.handedness[sample.handedness or "unknown"] += 1
        if sample.label:
            stats.labels[sample.label] += 1
        return True

    image = load_image(sample.image_path)
    if image is None:
        stats.skipped["unreadable"] += 1
        return False
    if image.size != sample.image_size:
        stats.skipped["size_mismatch"] += 1
        return False

    rng = random.Random(f"{config.seed}:{sample.key}")
    cropped = crop_hand_sample(image, sample, rng, config)
    if cropped is None:
        stats.skipped["too_few_keypoints"] += 1
        return False
    crop, label = cropped

    crop.resize((config.imgsz, config.imgsz), Image.Resampling.LANCZOS).save(
        image_path, format="JPEG", quality=config.jpeg_quality, subsampling=0
    )
    label_path.write_text(label + "\n", encoding="utf-8")
    stats.images_written += 1
    stats.labels_written += 1
    stats.sources[sample.source] += 1
    stats.handedness[sample.handedness or "unknown"] += 1
    if sample.label:
        stats.labels[sample.label] += 1
    return True


def write_hand_crops(
    config: HandBuildConfig,
    split: str,
    samples: Iterator[HandSample] | Sequence[HandSample],
    stats: SplitStats | None = None,
) -> SplitStats:
    """Crop and write ``samples`` into ``split``; return updated counters.

    Reused by :mod:`cpsa_ml.pseudolabel` so recorded frames and public datasets
    share one crop geometry.  Samples whose files already exist are skipped, so
    an interrupted build can simply be restarted.
    """

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    (config.out / "images" / split).mkdir(parents=True, exist_ok=True)
    (config.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    counters = stats if stats is not None else SplitStats()
    for sample in samples:
        counters.hands_seen += 1
        _write_hand_crop(config, split, sample, counters)
    return counters


def _gesture_name(directory_name: str) -> str:
    name = directory_name
    for prefix in HAGRID_SPLIT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _hagrid_annotation_files(annotations: Path, split: str, gesture: str | None) -> list[Path]:
    if annotations.is_file():
        return [annotations]
    split_dir = annotations / split
    if not split_dir.is_dir():
        return []
    if gesture is None:
        return sorted(split_dir.glob("*.json"))
    preferred = split_dir / f"{gesture}.json"
    if preferred.is_file():
        return [preferred]
    return []


def _load_hagrid_entries(
    annotations: Path, splits: Sequence[str], gesture: str | None, needed: set[str]
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Load only the annotations of ``needed`` image ids, keyed by image id."""

    entries: dict[str, tuple[str, dict[str, Any]]] = {}
    for split in splits:
        for path in _hagrid_annotation_files(annotations, split, gesture):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read HaGRID annotations {path}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"HaGRID annotations {path} must be a JSON object")
            for image_id in needed:
                if image_id in data and image_id not in entries:
                    entries[image_id] = (split, data[image_id])
    return entries


def _hagrid_hands(
    entry: dict[str, Any], image_size: tuple[int, int]
) -> Iterator[tuple[tuple[float, float, float, float], NDArray[np.float32], str]]:
    """Yield ``(pixel box, keypoints, label)`` for the usable hands of one entry."""

    width, height = image_size
    boxes = entry.get("bboxes") or []
    landmarks = entry.get("hand_landmarks") or []
    labels = entry.get("labels") or []
    for index, (box, hand) in enumerate(zip(boxes, landmarks, strict=False)):
        points = np.asarray(hand, dtype=np.float32)
        if points.shape != (KEYPOINTS, 2) or not np.isfinite(points).all():
            continue
        values = np.asarray(box, dtype=np.float32)
        if values.shape != (4,) or not np.isfinite(values).all():
            continue
        if float(values[2]) <= 0.0 or float(values[3]) <= 0.0:
            continue

        # HaGRID boxes are normalized top-left ``x y w h``; the union with the
        # landmark extent keeps every fingertip, which those boxes can miss.
        x1 = float(min(values[0], points[:, 0].min())) * width
        y1 = float(min(values[1], points[:, 1].min())) * height
        x2 = float(max(values[0] + values[2], points[:, 0].max())) * width
        y2 = float(max(values[1] + values[3], points[:, 1].max())) * height
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 < MIN_SOURCE_BOX or y2 - y1 < MIN_SOURCE_BOX:
            continue

        keypoints = np.zeros((KEYPOINTS, 3), dtype=np.float32)
        keypoints[:, 0] = points[:, 0] * width
        keypoints[:, 1] = points[:, 1] * height
        keypoints[:, 2] = 1.0
        label = str(labels[index]) if index < len(labels) else ""
        yield (x1, y1, x2, y2), keypoints, label


def iter_hagrid_samples(source: HagridSource) -> Iterator[HandSample]:
    """Yield one :class:`HandSample` per annotated hand of the HaGRID images."""

    images_root = Path(source.images)
    if not images_root.is_dir():
        raise ValueError(f"HaGRID image directory not found: {images_root}")
    annotations = Path(source.annotations)
    if not annotations.exists():
        raise ValueError(f"HaGRID annotations not found: {annotations}")

    gesture_dirs = sorted(path for path in images_root.iterdir() if path.is_dir())
    directories: list[tuple[Path, str | None]] = (
        [(path, _gesture_name(path.name)) for path in gesture_dirs]
        if gesture_dirs
        else [(images_root, None)]
    )
    for directory, gesture in directories:
        files = _image_files(directory)
        if not files:
            continue
        needed = {path.stem for path in files}
        entries = _load_hagrid_entries(annotations, source.splits, gesture, needed)
        if not entries:
            LOGGER.warning(
                "no HaGRID annotations matched %d images in %s (gesture %r)",
                len(files),
                directory,
                gesture,
            )
            continue
        for path in files:
            found = entries.get(path.stem)
            if found is None:
                continue
            origin_split, entry = found
            image_size = _image_size(path)
            if image_size is None:
                continue
            for box, keypoints, label in _hagrid_hands(entry, image_size):
                yield HandSample(
                    image_path=path,
                    image_size=image_size,
                    box=box,
                    keypoints=keypoints,
                    source=HAGRID,
                    origin_split=origin_split,
                    label=label,
                )


def _yolo_samples(
    image_path: Path, image_size: tuple[int, int], text: str, origin_split: str
) -> Iterator[HandSample]:
    width, height = image_size
    for line in text.splitlines():
        parts = line.split()
        expected = 5 + KEYPOINTS * 3
        if len(parts) < expected:
            continue
        try:
            cx, cy, box_w, box_h = (float(value) for value in parts[1:5])
            values = np.asarray([float(value) for value in parts[5:expected]], dtype=np.float32)
        except ValueError:
            continue
        keypoints = values.reshape(KEYPOINTS, 3)
        keypoints[:, 0] *= width
        keypoints[:, 1] *= height
        keypoints[:, 2] = (keypoints[:, 2] > 0).astype(np.float32)
        box = (
            (cx - box_w / 2) * width,
            (cy - box_h / 2) * height,
            (cx + box_w / 2) * width,
            (cy + box_h / 2) * height,
        )
        yield HandSample(
            image_path=image_path,
            image_size=image_size,
            box=box,
            keypoints=keypoints,
            source=ULTRALYTICS,
            origin_split=origin_split,
        )


def iter_ultralytics_samples(source: UltralyticsHandSource) -> Iterator[HandSample]:
    """Yield one :class:`HandSample` per labelled hand of hand-keypoints."""

    root = Path(source.root)
    for split in source.splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not label_dir.is_dir():
            raise ValueError(f"hand-keypoints labels not found: {label_dir}")
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = _find_image(image_dir, label_path.stem)
            if image_path is None:
                LOGGER.warning("missing image for label %s", label_path)
                continue
            image_size = _image_size(image_path)
            if image_size is None:
                continue
            text = label_path.read_text(encoding="utf-8", errors="replace")
            yield from _yolo_samples(image_path, image_size, text, split)


def _target_split(sample: HandSample, val_ratio: float) -> str:
    """Map the source split to ours, holding out a hash share of training data."""

    if sample.origin_split != "train" or val_ratio <= 0.0:
        return "val" if sample.origin_split in {"val", "test"} else "train"
    digest = hashlib.sha1(f"{sample.source}|{sample.image_path}".encode()).digest()
    share = int.from_bytes(digest[:4], "big") / 2**32
    return "val" if share < val_ratio else "train"


def _at_cap(config: HandBuildConfig, stats: SplitStats) -> bool:
    return config.max_images is not None and stats.images_written >= config.max_images


def write_hand_dataset_yaml(out: Path, schema: KeypointSchema) -> None:
    """Write ``data.yaml`` for the single ``hand`` class with relative paths."""

    data = {
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": list(schema.kpt_shape),
        "flip_idx": list(schema.flip_idx),
        "names": {0: "hand"},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _stats_payload(
    config: HandBuildConfig, schema: KeypointSchema, splits: dict[str, SplitStats]
) -> dict[str, Any]:
    used = sorted({name for stats in splits.values() for name in stats.sources})
    payload: dict[str, Any] = {
        "schema": schema.name,
        "kpt_shape": list(schema.kpt_shape),
        "class_names": ["hand"],
        "imgsz": config.imgsz,
        "crop": {
            "expand": list(config.expand),
            "shift": config.shift,
            "scale_jitter": config.scale_jitter,
            "min_visible": config.min_visible,
            "jitter_seed": config.seed,
        },
        "sources": {name: dict(SOURCE_LICENSES[name]) for name in used},
        "licenses": {name: SOURCE_LICENSES[name]["license"] for name in used},
        "splits": {split: stats.to_dict() for split, stats in splits.items()},
        "totals": {
            "images": sum(stats.images_written for stats in splits.values()),
            "labels": sum(stats.labels_written for stats in splits.values()),
        },
    }
    if ULTRALYTICS in used:
        payload["license_warning"] = (
            "The Ultralytics hand-keypoints source is CC BY-NC-SA 4.0: this dataset is "
            "NOT cleared for commercial use or redistribution."
        )
    return payload


def build_hand_dataset(
    config: HandBuildConfig,
    *,
    hagrid: HagridSource | None = None,
    ultralytics: UltralyticsHandSource | None = None,
) -> HandBuildReport:
    """Build the hand-crop dataset from the given sources.

    Args:
        config: Output paths and crop geometry.
        hagrid: HaGRID images plus landmark annotations.
        ultralytics: Ultralytics hand-keypoints tree.

    Returns:
        Per-split counters.  ``stats.json`` holds the same numbers plus licenses.

    Raises:
        ValueError: If no source is given, the schema is not the hand schema, or
            a source tree is missing.
    """

    schema = get_schema(config.schema)
    if schema.num_keypoints != KEYPOINTS:
        raise ValueError(f"the hand dataset needs {KEYPOINTS} keypoints, got {schema.name}")
    streams: list[tuple[str, Iterator[HandSample]]] = []
    if hagrid is not None:
        streams.append((HAGRID, iter_hagrid_samples(hagrid)))
    if ultralytics is not None:
        streams.append((ULTRALYTICS, iter_ultralytics_samples(ultralytics)))
    if not streams:
        raise ValueError("give at least one source: hagrid=... and/or ultralytics=...")

    for split in SPLITS:
        (config.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    splits = {split: SplitStats() for split in SPLITS}

    finished = False
    for _, samples in streams:
        if finished:
            break
        seen_images: set[Path] = set()
        for sample in samples:
            if finished:
                break
            target = _target_split(sample, config.val_ratio)
            stats = splits[target]
            if sample.image_path not in seen_images:
                seen_images.add(sample.image_path)
                stats.source_images += 1
            if _at_cap(config, stats):
                stats.skipped["cap_reached"] += 1
                finished = all(_at_cap(config, value) for value in splits.values())
                continue
            stats.hands_seen += 1
            _write_hand_crop(config, target, sample, stats)
            finished = all(_at_cap(config, value) for value in splits.values())

    write_hand_dataset_yaml(config.out, schema)
    (config.out / "schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    payload = _stats_payload(config, schema, splits)
    (config.out / "stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = HandBuildReport(schema=schema.name, out=config.out, imgsz=config.imgsz, splits=splits)
    for split, stats in splits.items():
        LOGGER.info(
            "%s: %d crops from %d images (%d skipped)",
            split,
            stats.images_written,
            stats.source_images,
            sum(stats.skipped.values()),
        )
        if stats.images_written == 0:
            LOGGER.warning("%s split is empty; check the source paths and splits", split)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hagrid-images", type=Path, default=None)
    parser.add_argument("--hagrid-annotations", type=Path, default=None)
    parser.add_argument("--hagrid-splits", default="train,val,test")
    parser.add_argument("--ultralytics-root", type=Path, default=None)
    parser.add_argument("--ultralytics-splits", default="train,val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=192)
    parser.add_argument("--expand", default="1.6,2.2")
    parser.add_argument("--shift", type=float, default=0.06)
    parser.add_argument("--scale-jitter", type=float, default=0.10)
    parser.add_argument("--min-visible", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.hagrid_images is None and args.ultralytics_root is None:
        raise SystemExit("give --hagrid-images/--hagrid-annotations or --ultralytics-root")
    low, _, high = args.expand.partition(",")
    report = build_hand_dataset(
        HandBuildConfig(
            out=args.out,
            imgsz=args.imgsz,
            expand=(float(low), float(high or low)),
            shift=args.shift,
            scale_jitter=args.scale_jitter,
            min_visible=args.min_visible,
            val_ratio=args.val_ratio,
            max_images=args.max_images,
            jpeg_quality=args.jpeg_quality,
            seed=args.seed,
            overwrite=args.overwrite,
        ),
        hagrid=(
            HagridSource(
                images=args.hagrid_images,
                annotations=args.hagrid_annotations or args.hagrid_images,
                splits=tuple(part for part in args.hagrid_splits.split(",") if part),
            )
            if args.hagrid_images
            else None
        ),
        ultralytics=(
            UltralyticsHandSource(
                root=args.ultralytics_root,
                splits=tuple(part for part in args.ultralytics_splits.split(",") if part),
            )
            if args.ultralytics_root
            else None
        ),
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
