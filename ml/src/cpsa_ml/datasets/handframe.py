"""Build a full-frame IR-look hand21 dataset for the top-down hand model.

The crop builder (:mod:`cpsa_ml.datasets.hands`) teaches the hand landmarks on
a tight, square crop around one hand.  The deployed hand model is different: it
sees the **whole 640x480 IR frame** (``256x320`` after resizing) and must find
every hand itself, like the body model finds the person.  This builder turns the
same sources into that dataset: one window per source image, chosen around all
the hands of the image, resized to ``imgsz_hw`` (``256x320`` by default), and
written as a grayscale, IR-looking frame.

Window
------
The window has a 5:4 aspect (the IR frame is 4:3 and the model is exported at
256x320, so 5:4 is close to the sensor and a clean multiple of 32).  Its width
is chosen so that the largest hand spans ``hand_frac`` (25-60%) of it, matching
a hand seen from above at 50-70 cm, and then grown -- never shrunk -- until it
covers the union of every hand in the image.  Cropping hands away would leave
unlabelled hands in the frame, which teaches the model to ignore them, so the
union always wins over the requested hand fraction.  The window is clamped
inside the image (no padding) and resized to ``imgsz_hw``.

Labels are Ultralytics YOLO-pose text files::

    class cx cy w h x1 y1 v1 x2 y2 v2 ...

``class`` is always 0 (``hand``), one line per hand whose box centre falls
inside the window; keypoints outside the window keep ``v = 0``.  Images whose
source annotation contains a hand that never reaches a sample (HaGRID box
without valid landmarks, unparsable YOLO line) are skipped entirely, so the
frame never shows an unlabelled hand.

IR look
-------
Every written frame gets the same treatment, in this order, seeded per image so
a rebuild is byte-identical:

1. grayscale, later replicated to three equal channels (the model keeps the
   3-channel interface of the rest of the pipeline);
2. gamma 0.6-1.6 and contrast 0.6-1.4;
3. Gaussian noise, sigma 0-8;
4. Gaussian blur with probability 0.3;
5. min-max normalize plus ``cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))``
   with probability ``clahe_p`` -- the same transform the runtime applies to the
   INZI IR image (``cpsa.vision.irdepth``).

The frames are written as PNG, so the three channels stay byte-equal and the
training/validation split never picks up JPEG chroma noise on a gray image.

Example::

    python -m cpsa_ml.datasets.handframe \\
        --hagrid-images data/hagrid/hagrid_30k \\
        --hagrid-annotations data/hagrid/annotations \\
        --ultralytics-root data/hand-keypoints \\
        --out data/hand-topdown
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from cpsa_ml.datasets.halfbust import format_label, load_image
from cpsa_ml.datasets.hands import (
    HAGRID,
    KEYPOINTS,
    SOURCE_LICENSES,
    SPLITS,
    ULTRALYTICS,
    HagridSource,
    HandSample,
    UltralyticsHandSource,
    _hagrid_annotation_files,
    iter_hagrid_samples,
    iter_ultralytics_samples,
    write_hand_dataset_yaml,
)
from cpsa_ml.eval import read_yolo_pose_labels
from cpsa_ml.schema import KeypointSchema, get_schema

__all__ = [
    "BLUR_P",
    "CLAHE_CLIP",
    "CLAHE_TILES",
    "CONTRAST_RANGE",
    "GAMMA_RANGE",
    "NOISE_SIGMA_RANGE",
    "WINDOW_ASPECT",
    "FrameStats",
    "HandFrameConfig",
    "apply_clahe",
    "build_handframe_dataset",
    "frame_label",
    "frame_window",
    "ir_look",
    "main",
    "render_label_contact_sheet",
]

LOGGER = logging.getLogger(__name__)

#: CLAHE of the runtime IR preprocessing (``cpsa.vision.irdepth``).  Duplicated
#: here because ``src/`` and ``ml/`` ship separately, and the training look must
#: match the deployed one exactly.
CLAHE_CLIP = 3.0
CLAHE_TILES = (8, 8)

#: Window aspect as ``width / height`` (5:4).
WINDOW_ASPECT = 1.25

#: IR-look ranges, in the order :func:`ir_look` applies them.
GAMMA_RANGE = (0.6, 1.6)
CONTRAST_RANGE = (0.6, 1.4)
NOISE_SIGMA_RANGE = (0.0, 8.0)
BLUR_P = 0.3
_BLUR_KSIZE = (3, 3)

#: Prefix of the output file name, per source (kept short: Kaggle paths).
_SOURCE_PREFIX = {HAGRID: "hagrid", ULTRALYTICS: "ultra"}


@dataclass(frozen=True)
class HandFrameConfig:
    """Window geometry, IR look and output paths of the full-frame dataset.

    Args:
        out: Dataset root; ``images/<split>``, ``labels/<split>``, ``data.yaml``,
            ``schema.json`` and ``stats.json`` are written below it.
        imgsz_hw: Written frame size as ``(height, width)``; ``(256, 320)``
            matches the exported ONNX input.
        hand_frac: Range of the largest hand's share of the window width.
        clahe_p: Probability of the runtime CLAHE on a written frame.
        seed: Seed of the per-image window and IR-look draws.
        max_images: Maximum frames written per split (``None`` for no limit).
        val_ratio: Deterministic share of the ``train`` source images held out.
        overwrite: Rewrite frames whose files already exist.
    """

    out: Path
    imgsz_hw: tuple[int, int] = (256, 320)
    hand_frac: tuple[float, float] = (0.25, 0.6)
    clahe_p: float = 0.8
    seed: int = 0
    max_images: int | None = None
    val_ratio: float = 0.05
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "out", Path(self.out))
        height, width = (int(value) for value in self.imgsz_hw)
        if height < 32 or width < 32:
            raise ValueError("imgsz_hw must be at least 32x32")
        if height % 32 or width % 32:
            raise ValueError("imgsz_hw must be a multiple of 32 for the YOLO strides")
        object.__setattr__(self, "imgsz_hw", (height, width))
        low, high = self.hand_frac
        if not 0.0 < low <= high < 1.0:
            raise ValueError("hand_frac must be an increasing range inside (0, 1)")
        if not 0.0 <= self.clahe_p <= 1.0:
            raise ValueError("clahe_p must be in [0, 1]")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.max_images is not None and self.max_images <= 0:
            raise ValueError("max_images must be positive")

    @property
    def out_size(self) -> tuple[int, int]:
        """Written ``(width, height)``."""

        return self.imgsz_hw[1], self.imgsz_hw[0]


@dataclass
class FrameStats:
    """Per-split counters written to ``stats.json``."""

    source_images: int = 0
    images_written: int = 0
    hands_seen: int = 0
    labels_written: int = 0
    hands_outside_window: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    sources: Counter[str] = field(default_factory=Counter)
    labels: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_images": self.source_images,
            "images_written": self.images_written,
            "hands_seen": self.hands_seen,
            "labels_written": self.labels_written,
            "hands_outside_window": self.hands_outside_window,
            "skip_reasons": dict(sorted(self.skipped.items())),
            "sources": dict(sorted(self.sources.items())),
            "labels": dict(sorted(self.labels.items())),
        }


def _minmax_normalize(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Stretch ``gray`` to the full 0-255 range; a flat image becomes zeros."""

    values = np.asarray(gray, dtype=np.float32)
    low = float(values.min()) if values.size else 0.0
    high = float(values.max()) if values.size else 0.0
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - low) * (255.0 / (high - low))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def apply_clahe(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply the runtime IR CLAHE to a grayscale image.

    Min-max normalization first (flat image -> zeros), then
    ``cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)``, exactly
    like ``cpsa.vision.irdepth.ir_to_input``.
    """

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
    return clahe.apply(_minmax_normalize(gray))


def ir_look(
    gray: NDArray[np.uint8],
    rng: random.Random,
    *,
    clahe_p: float = 0.8,
) -> NDArray[np.uint8]:
    """Turn one grayscale frame into an IR-looking one, in the documented order.

    Args:
        gray: ``(H, W)`` uint8 grayscale frame.
        rng: Seeded RNG; every draw (gamma, contrast, noise, blur, CLAHE) comes
            from it, so the same seed and frame always give the same output.
        clahe_p: Probability of applying :func:`apply_clahe`.

    Returns:
        ``(H, W)`` uint8 frame; replicate it to three channels to write it.
    """

    image = np.asarray(gray, dtype=np.float32) / 255.0
    gamma = rng.uniform(*GAMMA_RANGE)
    contrast = rng.uniform(*CONTRAST_RANGE)
    sigma = rng.uniform(*NOISE_SIGMA_RANGE)
    blur = rng.random() < BLUR_P
    use_clahe = rng.random() < clahe_p

    image = np.clip(np.power(np.clip(image, 0.0, 1.0), gamma), 0.0, 1.0)
    image = np.clip((image - 0.5) * contrast + 0.5, 0.0, 1.0)
    if sigma > 0.0:
        noise_rng = np.random.default_rng(rng.getrandbits(64))
        image = np.clip(image + noise_rng.normal(0.0, sigma / 255.0, image.shape), 0.0, 1.0)
    if blur:
        image = cv2.GaussianBlur(image, _BLUR_KSIZE, 0)
    looked = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if use_clahe:
        looked = apply_clahe(looked)
    return looked


def frame_window(
    boxes: Sequence[tuple[float, float, float, float]],
    image_size: tuple[int, int],
    rng: random.Random,
    config: HandFrameConfig,
) -> tuple[int, int, int, int]:
    """Return the ``(x1, y1, x2, y2)`` window of one source image, in pixels.

    The window has :data:`WINDOW_ASPECT` unless the image is too small, is sized
    so the largest hand spans ``hand_frac`` of its width, and is grown until it
    covers every box.  It is then centred on the boxes and clamped inside the
    image, so no padding is ever introduced.
    """

    if not boxes:
        raise ValueError("at least one hand box is needed to choose the window")
    width, height = image_size
    if width < 2 or height < 2:
        raise ValueError(f"image size must be at least 2x2, got {width}x{height}")

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    span = max(
        max(box[2] - box[0] for box in boxes),
        max(box[3] - box[1] for box in boxes),
        1.0,
    )

    fraction = rng.uniform(*config.hand_frac)
    window_w = max(span / fraction, right - left, (bottom - top) * WINDOW_ASPECT)
    window_h = window_w / WINDOW_ASPECT
    # Clamp per axis: the union is already inside the image, so the clamped
    # window still covers it (only the aspect can change, for tiny images).
    window_w = min(window_w, float(width))
    window_h = min(window_h, float(height))

    body_w = max(2, int(round(window_w)))
    body_h = max(2, int(round(window_h)))
    centre_x = 0.5 * (left + right)
    centre_y = 0.5 * (top + bottom)
    x1 = int(round(min(max(centre_x - body_w / 2.0, 0.0), width - body_w)))
    y1 = int(round(min(max(centre_y - body_h / 2.0, 0.0), height - body_h)))
    return x1, y1, x1 + body_w, y1 + body_h


def frame_label(sample: HandSample, window: tuple[int, int, int, int]) -> str | None:
    """Return the YOLO-pose line of one hand in ``window``, or ``None`` to drop it.

    A hand is labelled when its box centre is inside the window; keypoints
    outside the window are written with ``v = 0``.
    """

    window_x1, window_y1, window_x2, window_y2 = (float(value) for value in window)
    crop_width = window_x2 - window_x1
    crop_height = window_y2 - window_y1
    if crop_width < 2.0 or crop_height < 2.0:
        return None

    x1, y1, x2, y2 = sample.box
    centre_x = 0.5 * (x1 + x2)
    centre_y = 0.5 * (y1 + y2)
    if not (window_x1 <= centre_x <= window_x2 and window_y1 <= centre_y <= window_y2):
        return None

    keypoints = np.zeros((KEYPOINTS, 3), dtype=np.float32)
    for index, (x, y, visibility) in enumerate(sample.keypoints):
        if visibility <= 0 or not (np.isfinite(x) and np.isfinite(y)):
            continue
        u = (x - window_x1) / crop_width
        v = (y - window_y1) / crop_height
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            continue
        keypoints[index] = (u, v, 1.0)

    left = float(np.clip((x1 - window_x1) / crop_width, 0.0, 1.0))
    right = float(np.clip((x2 - window_x1) / crop_width, 0.0, 1.0))
    top = float(np.clip((y1 - window_y1) / crop_height, 0.0, 1.0))
    bottom = float(np.clip((y2 - window_y1) / crop_height, 0.0, 1.0))
    if right - left <= 0.0 or bottom - top <= 0.0:
        return None
    box = (0.5 * (left + right), 0.5 * (top + bottom), right - left, bottom - top)
    return format_label(0, keypoints, box)


def _image_key(source: str, image_path: Path) -> str:
    """Stable, machine-independent identity of one source image.

    The full path differs between a local run and the Kaggle notebook, so the
    file name (a HaGRID UUID or a dataset image id) is what the output names,
    the split hash and the per-image seed are derived from.
    """

    return f"{source}|{image_path.name}"


def _output_stem(source: str, image_path: Path) -> str:
    digest = hashlib.sha1(_image_key(source, image_path).encode()).hexdigest()[:10]
    prefix = _SOURCE_PREFIX.get(source, "hand")
    return f"{prefix}_{image_path.stem[:32]}_{digest}"


def _target_split(source: str, image_path: Path, origin_split: str, val_ratio: float) -> str:
    """Map the source split to ours, holding out a hash share of training data."""

    if origin_split == "train" and val_ratio > 0.0:
        digest = hashlib.sha1(_image_key(source, image_path).encode()).digest()
        share = int.from_bytes(digest[:4], "big") / 2**32
        if share < val_ratio:
            return "val"
    return "train" if origin_split == "train" else "val"


def _group_samples(samples: Iterator[HandSample]) -> dict[Path, list[HandSample]]:
    """Group the hands of one source by image, in a deterministic order."""

    grouped: dict[Path, list[HandSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.image_path].append(sample)
    return {path: grouped[path] for path in sorted(grouped)}


def _hagrid_box_counts(source: HagridSource, image_paths: Sequence[Path]) -> dict[Path, int]:
    """Return the raw ``bboxes`` count of the requested HaGRID images.

    ``hands._hagrid_hands`` silently drops a box without usable landmarks, so the
    builder needs the raw count to notice that an image holds a hand it could
    not label.  The annotation file selection mirrors ``iter_hagrid_samples``.
    """

    by_stem = {path.stem: path for path in image_paths}
    counts: dict[Path, int] = {}
    seen: set[Path] = set()
    for split in source.splits:
        for path in _hagrid_annotation_files(Path(source.annotations), split, None):
            if path in seen:
                continue
            seen.add(path)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read HaGRID annotations {path}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"HaGRID annotations {path} must be a JSON object")
            for stem, image_path in by_stem.items():
                if image_path in counts or stem not in data:
                    continue
                entry = data[stem]
                boxes = entry.get("bboxes") if isinstance(entry, dict) else None
                counts[image_path] = len(boxes) if isinstance(boxes, list) else 0
            if len(counts) == len(by_stem):
                return counts
    return counts


def _ultralytics_line_counts(
    source: UltralyticsHandSource, image_paths: Sequence[Path]
) -> dict[Path, int]:
    """Return the raw non-empty label-line count of the requested images."""

    counts: dict[Path, int] = {}
    root = Path(source.root)
    for image_path in image_paths:
        label_path = root / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        try:
            text = label_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        counts[image_path] = sum(1 for line in text.splitlines() if line.strip())
    return counts


def _write_frame(
    config: HandFrameConfig,
    split: str,
    source: str,
    image_path: Path,
    samples: Sequence[HandSample],
    stats: FrameStats,
) -> None:
    """Write one IR-look frame and its labels; update ``stats`` in place."""

    stem = _output_stem(source, image_path)
    image_dir = config.out / "images" / split
    label_dir = config.out / "labels" / split
    frame_path = image_dir / f"{stem}.png"
    label_path = label_dir / f"{stem}.txt"
    if frame_path.is_file() and label_path.is_file() and not config.overwrite:
        stats.images_written += 1
        stats.labels_written += len(label_path.read_text(encoding="utf-8").splitlines())
        stats.skipped["existing"] += 1
        stats.sources[source] += 1
        return

    image = load_image(image_path)
    if image is None:
        stats.skipped["unreadable"] += 1
        return
    if image.size != samples[0].image_size:
        stats.skipped["size_mismatch"] += 1
        return

    seed_key = f"{config.seed}:{_image_key(source, image_path)}"
    rng = random.Random(seed_key)
    window = frame_window([sample.box for sample in samples], image.size, rng, config)

    lines: list[str] = []
    written_labels: Counter[str] = Counter()
    for sample in samples:
        line = frame_label(sample, window)
        if line is None:
            stats.hands_outside_window += 1
            continue
        lines.append(line)
        if sample.label:
            written_labels[sample.label] += 1
    if not lines:
        stats.skipped["no_hands"] += 1
        return

    crop = image.crop(window).resize(config.out_size, Image.Resampling.LANCZOS)
    gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    looked = ir_look(gray, rng, clahe_p=config.clahe_p)
    frame = np.repeat(looked[:, :, None], 3, axis=2)
    Image.fromarray(frame).save(frame_path, format="PNG")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats.images_written += 1
    stats.labels_written += len(lines)
    stats.sources[source] += 1
    stats.labels.update(written_labels)


def _at_cap(config: HandFrameConfig, stats: FrameStats) -> bool:
    return config.max_images is not None and stats.images_written >= config.max_images


def _stats_payload(
    config: HandFrameConfig,
    schema: KeypointSchema,
    splits: Mapping[str, FrameStats],
) -> dict[str, Any]:
    """Assemble the ``stats.json`` payload, including the source licences."""

    used = sorted({name for stats in splits.values() for name in stats.sources})
    payload: dict[str, Any] = {
        "schema": schema.name,
        "kpt_shape": list(schema.kpt_shape),
        "class_names": ["hand"],
        "imgsz_hw": list(config.imgsz_hw),
        "window": {
            "aspect": WINDOW_ASPECT,
            "hand_frac": list(config.hand_frac),
            "clahe_p": config.clahe_p,
            "seed": config.seed,
        },
        "look": {
            "gamma": list(GAMMA_RANGE),
            "contrast": list(CONTRAST_RANGE),
            "noise_sigma": list(NOISE_SIGMA_RANGE),
            "blur_p": BLUR_P,
            "clahe_clip": CLAHE_CLIP,
            "clahe_tiles": list(CLAHE_TILES),
            "grayscale": True,
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


def build_handframe_dataset(
    config: HandFrameConfig,
    *,
    hagrid: HagridSource | None = None,
    ultralytics: UltralyticsHandSource | None = None,
) -> dict[str, Any]:
    """Build the full-frame IR-look hand21 dataset from the given sources.

    One window (``imgsz_hw``, grayscale, IR look) is written per source image.
    An image is skipped when its annotation holds a hand that the source readers
    cannot turn into a label, so the model never learns to ignore a hand.

    Args:
        config: Output paths, window geometry and IR-look probability.
        hagrid: HaGRID images plus landmark annotations.
        ultralytics: Ultralytics hand-keypoints tree.

    Returns:
        The ``stats.json`` payload (per-split counters, look settings, licences).

    Raises:
        ValueError: If no source is given or a source tree is missing.
    """

    schema = get_schema("hand21")
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
    splits = {split: FrameStats() for split in SPLITS}

    for source, samples in streams:
        grouped = _group_samples(samples)
        if not grouped:
            LOGGER.warning("no %s samples found; check the source paths and splits", source)
            continue
        paths = list(grouped)
        expected: dict[Path, int] = {}
        if source == HAGRID and hagrid is not None:
            expected = _hagrid_box_counts(hagrid, paths)
        elif source == ULTRALYTICS and ultralytics is not None:
            expected = _ultralytics_line_counts(ultralytics, paths)

        for image_path in paths:
            hands = grouped[image_path]
            origin_split = hands[0].origin_split
            target = _target_split(source, image_path, origin_split, config.val_ratio)
            stats = splits[target]
            stats.source_images += 1
            stats.hands_seen += len(hands)

            raw = expected.get(image_path)
            if raw is not None and raw > len(hands):
                # A hand exists in the source annotation but could not be
                # labelled: keeping the image would teach "hand = background".
                stats.skipped["unlabelled_hand"] += 1
                continue
            if _at_cap(config, stats):
                stats.skipped["cap_reached"] += 1
                continue
            _write_frame(config, target, source, image_path, hands, stats)

    write_hand_dataset_yaml(config.out, schema)
    (config.out / "schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    payload = _stats_payload(config, schema, splits)
    (config.out / "stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for split, stats in splits.items():
        LOGGER.info(
            "%s: %d frames, %d hands (%d skipped)",
            split,
            stats.images_written,
            stats.labels_written,
            sum(stats.skipped.values()),
        )
        if stats.images_written == 0:
            LOGGER.warning("%s split is empty; check the source paths and splits", split)
    return payload


def render_label_contact_sheet(
    image_dir: Path,
    label_dir: Path,
    out: Path,
    *,
    count: int = 8,
    columns: int = 4,
    cell_width: int = 320,
    gutter: int = 6,
) -> Path:
    """Draw labelled frames into one grid image for a quick visual check.

    Args:
        image_dir: ``images/<split>`` directory.
        label_dir: ``labels/<split>`` directory of the same split.
        out: Image path to write; the format follows the suffix (PNG/JPEG).
        count: Number of frames, in sorted stem order.
        columns: Grid columns.
        cell_width: Each cell is scaled to this width.
        gutter: Pixels between cells.

    Returns:
        ``out``.

    Raises:
        ValueError: If the arguments are invalid or no frame could be read.
    """

    if count <= 0 or columns <= 0:
        raise ValueError("count and columns must be positive")
    labels = sorted(Path(label_dir).glob("*.txt"))[:count]
    if not labels:
        raise ValueError(f"no label files under {label_dir}")

    cells: list[Image.Image] = []
    for label_path in labels:
        image_path = Path(image_dir) / f"{label_path.stem}.png"
        image = load_image(image_path)
        if image is None:
            continue
        targets = read_yolo_pose_labels(
            label_path.read_text(encoding="utf-8"), image.width, image.height, KEYPOINTS
        )
        draw = ImageDraw.Draw(image)
        for target in targets:
            x1, y1, x2, y2 = (float(value) for value in target.box)
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
            for x, y, visibility in target.keypoints:
                if visibility <= 0:
                    continue
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))
        scale = cell_width / image.width
        cells.append(
            image.resize(
                (cell_width, max(1, int(round(image.height * scale)))),
                Image.Resampling.LANCZOS,
            )
        )
    if not cells:
        raise ValueError(f"no readable frames for the labels under {label_dir}")

    rows = (len(cells) + columns - 1) // columns
    width = columns * cell_width + (columns - 1) * gutter
    height = rows * max(cell.height for cell in cells) + (rows - 1) * gutter
    canvas = Image.new("RGB", (width, height), (24, 24, 28))
    for index, cell in enumerate(cells):
        column = index % columns
        row = index // columns
        canvas.paste(cell, (column * (cell_width + gutter), row * (cell.height + gutter)))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".jpg", ".jpeg"}:
        canvas.save(out, format="JPEG", quality=92)
    else:
        canvas.save(out, format="PNG")
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hagrid-images", type=Path, default=None)
    parser.add_argument("--hagrid-annotations", type=Path, default=None)
    parser.add_argument("--hagrid-splits", default="train,val,test")
    parser.add_argument("--ultralytics-root", type=Path, default=None)
    parser.add_argument("--ultralytics-splits", default="train,val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz-hw", default="256,320")
    parser.add_argument("--hand-frac", default="0.25,0.6")
    parser.add_argument("--clahe-p", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.hagrid_images is None and args.ultralytics_root is None:
        raise SystemExit("give --hagrid-images/--hagrid-annotations or --ultralytics-root")
    height, _, width = args.imgsz_hw.partition(",")
    low, _, high = args.hand_frac.partition(",")
    payload = build_handframe_dataset(
        HandFrameConfig(
            out=args.out,
            imgsz_hw=(int(height), int(width or height)),
            hand_frac=(float(low), float(high or low)),
            clahe_p=args.clahe_p,
            seed=args.seed,
            max_images=args.max_images,
            val_ratio=args.val_ratio,
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
    print(json.dumps(payload["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
