"""Build a full-frame YOLO-pose dataset from COCO-WholeBody annotations.

The half-bust builder (:mod:`cpsa_ml.datasets.halfbust`) produces crops that
match a seated subject, but a model trained only on those crops has never seen
a standing person and its predicted box hugs the keypoint extent instead of the
true person extent.  The runtime uses that box height (divided by the face
width) to decide ``full_body`` vs ``half_body`` framing
(``cpsa.vision.subject``), so the box matters.  This builder therefore writes
the *original* COCO image unchanged and labels every usable person with the
true COCO person box::

    class cx cy w h x1 y1 v1 x2 y2 v2 ...

Keypoints use the same ``HALFBUST_62`` mapping and ``flip_idx`` as the
half-bust dataset because the label mapping and the dataset writers are
imported from :mod:`cpsa_ml.datasets.halfbust` -- there is one implementation.

Mixing
------
Notebook 02 writes this set next to the half-bust set from notebook 01, and
training concatenates both ``images/train`` directories.  The full-frame share
is ``f = 0.25`` by default, so ``n_full = round(f / (1 - f) * n_half)``: 25% of
the *mixed* training images are full frames.  That keeps enough whole-body
views for the box to reach the feet while half-body crops keep dominating the
scale the deployed model sees.

Unlabeled people
----------------
Ultralytics' own COCO converter (``ultralytics.data.converter.convert_coco``)
skips ``iscrowd`` annotations and annotations whose keypoints carry no visible
point, and keeps the image with whatever labels remain.  This builder does the
same for a person that fails :attr:`FullFrameConfig.min_visible` (or whose COCO
box is degenerate): no label line is written and the person is counted in
``persons_unlabeled``.  People in a **crowd region** are different: a crowd
annotation covers many visible people that a YOLO label cannot express as
"ignore", so keeping the image would teach the model that those people are
background.  An image that contains a crowd annotation is therefore skipped
entirely (``images_skipped_crowd``).  An image with no labelable person is not
written either, because a negative sample is not what this dataset is for.

No hand-over-face occlusion is applied in this mode: the full-frame set exists
to teach the true person extent, so its pixels stay untouched.

Example::

    python -m cpsa_ml.datasets.fullframe \\
        --coco-root data/coco \\
        --wholebody-json annotations/coco_wholebody_train_v1.0.json \\
        --split train --schema halfbust62 --out data/fullframe62 \\
        --half-train-root data/halfbust62 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cpsa_ml.datasets.halfbust import (
    format_label,
    load_image,
    map_keypoints,
    resolve_split_dir,
    update_stats,
    wholebody_keypoints,
    write_dataset_yaml,
)
from cpsa_ml.eval import HALFBUST62_N_KPT, read_yolo_pose_labels
from cpsa_ml.schema import KeypointSchema, get_schema

__all__ = [
    "FULL_FRACTION",
    "FullFrameConfig",
    "FullFrameReport",
    "build_fullframe_dataset",
    "full_frame_target_count",
    "half_train_count",
    "main",
    "render_labeled_preview",
    "resolve_target_count",
]

LOGGER = logging.getLogger(__name__)

#: Default share of full-frame images in the mixed training set.
FULL_FRACTION = 0.25

_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})


@dataclass(frozen=True)
class FullFrameConfig:
    """Configuration for :func:`build_fullframe_dataset`.

    Args:
        coco_root: COCO 2017 root holding ``train2017``/``val2017``.
        wholebody_json: ``coco_wholebody_<split>_v1.0.json`` of the same split.
        split: ``train`` or ``val``.
        out: Dataset root; ``images/<split>`` and ``labels/<split>`` are written.
        schema: Keypoint schema written to the labels (default ``halfbust62``).
        max_images: Write at most this many qualifying images (train only).
        target_count: Write at most this many qualifying images (train only).
        fraction: Full-frame share ``f`` used with ``half_train_count``.
        half_train_count: ``n_half`` of the half-bust train split; when set the
            cap becomes :func:`full_frame_target_count`.
        seed: Seed of the train-split image sample.
        min_visible: Schema keypoints a person must show to get a label line.
        jpeg_quality: Quality used only when the source is not already a JPEG.
    """

    coco_root: Path
    wholebody_json: Path
    split: str
    out: Path
    schema: KeypointSchema | str = "halfbust62"
    max_images: int | None = None
    target_count: int | None = None
    fraction: float = FULL_FRACTION
    half_train_count: int | None = None
    seed: int = 0
    min_visible: int = 3
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        object.__setattr__(self, "coco_root", Path(self.coco_root))
        object.__setattr__(self, "wholebody_json", Path(self.wholebody_json))
        object.__setattr__(self, "out", Path(self.out))
        object.__setattr__(self, "schema", get_schema(self.schema))
        if self.split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {self.split!r}")
        if not 0.0 < self.fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {self.fraction}")
        for name in ("max_images", "target_count", "half_train_count"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.min_visible < 0:
            raise ValueError("min_visible must be non-negative")
        if not 0 < self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in (0, 100]")


@dataclass
class FullFrameReport:
    """Counters returned by :func:`build_fullframe_dataset`."""

    schema: str
    split: str
    out: Path
    source_images: int = 0
    qualifying_images: int = 0
    images_written: int = 0
    labels_written: int = 0
    persons_labeled: int = 0
    persons_unlabeled: int = 0
    crowd_annotations: int = 0
    images_skipped_crowd: int = 0
    images_skipped_no_label: int = 0
    images_skipped_missing: int = 0
    target_count: int | None = None
    fraction: float = FULL_FRACTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "split": self.split,
            "out": str(self.out),
            "source_images": self.source_images,
            "qualifying_images": self.qualifying_images,
            "images_written": self.images_written,
            "labels_written": self.labels_written,
            "persons_labeled": self.persons_labeled,
            "persons_unlabeled": self.persons_unlabeled,
            "crowd_annotations": self.crowd_annotations,
            "images_skipped_crowd": self.images_skipped_crowd,
            "images_skipped_no_label": self.images_skipped_no_label,
            "images_skipped_missing": self.images_skipped_missing,
            "target_count": self.target_count,
            "fraction": self.fraction,
        }


def full_frame_target_count(n_half: int, fraction: float = FULL_FRACTION) -> int:
    """Number of full frames for a mixed set: ``round(f / (1 - f) * n_half)``.

    Args:
        n_half: Number of half-body *training* images.
        fraction: Desired full-frame share of the mixed training set, ``f``.

    Returns:
        The number of full-frame training images to write.

    Raises:
        ValueError: If ``n_half`` is negative or ``fraction`` is not in (0, 1).
    """

    if n_half < 0:
        raise ValueError(f"n_half must be non-negative, got {n_half}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    return int(round(fraction / (1.0 - fraction) * n_half))


def half_train_count(half_root: Path) -> int:
    """Read ``splits.train.images_written`` from a half-bust ``stats.json``.

    Args:
        half_root: Half-bust dataset root, or its ``stats.json`` path.

    Returns:
        The number of half-body training images.

    Raises:
        FileNotFoundError: If ``stats.json`` is missing.
        ValueError: If it has no train-split count.
    """

    path = Path(half_root)
    if path.is_dir():
        path = path / "stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"half-bust stats.json not found: {path}")
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
        count = int(stats["splits"]["train"]["images_written"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} has no splits.train.images_written counter: {exc}") from exc
    if count < 0:
        raise ValueError(f"{path}: negative train image count")
    return count


def resolve_target_count(config: FullFrameConfig) -> int | None:
    """Effective train-split image cap (``None`` means "write every one").

    ``max_images`` and ``target_count`` are both absolute caps; when
    ``half_train_count`` is set its :func:`full_frame_target_count` joins them
    and the smallest value wins.
    """

    caps = [value for value in (config.max_images, config.target_count) if value is not None]
    if config.half_train_count is not None:
        caps.append(full_frame_target_count(config.half_train_count, config.fraction))
    return min(caps) if caps else None


def _normalized_box(bbox: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Convert a COCO ``[x, y, w, h]`` box to normalized ``cx cy w h``, clipped to the image."""

    try:
        x, y, box_w, box_h = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in (x, y, box_w, box_h)):
        return None
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + box_w)
    y2 = min(float(height), y + box_h)
    if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
        return None
    return (
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def _label_for_person(
    annotation: Mapping[str, Any],
    schema: KeypointSchema,
    width: int,
    height: int,
    min_visible: int,
) -> str | None:
    """Return the label line of one person, or ``None`` when it is unusable."""

    box = _normalized_box(annotation.get("bbox"), width, height)
    if box is None:
        return None
    keypoints = wholebody_keypoints(annotation)
    mapped = map_keypoints(keypoints, schema, (0, 0, width, height))
    if int(np.count_nonzero(mapped[:, 2] > 0)) < min_visible:
        return None
    return format_label(0, mapped, box)


def _image_size(image: Mapping[str, Any]) -> tuple[int, int] | None:
    try:
        width = int(image["width"])
        height = int(image["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    """Read only the size of an image, or ``None`` when it cannot be opened."""

    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        LOGGER.warning("cannot read image %s: %s", path, exc)
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _write_image(source: Path, target: Path, jpeg_quality: int) -> None:
    """Copy the original frame; re-encode only when it is not already a JPEG."""

    if source.suffix.lower() in _JPEG_SUFFIXES:
        shutil.copyfile(source, target)
        return
    image = load_image(source)
    if image is None:
        raise OSError(f"cannot decode {source}")
    image.save(target, format="JPEG", quality=jpeg_quality, subsampling=0)


def _select_images(
    qualifying: Sequence[tuple[int, Mapping[str, Any]]],
    config: FullFrameConfig,
    target_count: int | None,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Deterministically sample the train split; the val split keeps every image."""

    if config.split == "val" or target_count is None or target_count >= len(qualifying):
        return list(qualifying)
    rng = random.Random(config.seed)
    chosen = sorted(rng.sample(range(len(qualifying)), target_count))
    return [qualifying[index] for index in chosen]


def build_fullframe_dataset(config: FullFrameConfig) -> FullFrameReport:
    """Convert COCO-WholeBody annotations into a full-frame YOLO-pose dataset.

    Args:
        config: Paths, sampling and label settings.

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

    persons_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    crowd_by_image: dict[int, int] = defaultdict(int)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if int(annotation.get("iscrowd", 0)):
            crowd_by_image[image_id] += 1
        else:
            persons_by_image[image_id].append(annotation)

    image_dir = config.out / "images" / config.split
    label_dir = config.out / "labels" / config.split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    report = FullFrameReport(
        schema=schema.name,
        split=config.split,
        out=config.out,
        target_count=resolve_target_count(config),
        fraction=config.fraction,
    )
    ordered_ids = sorted(
        image_records, key=lambda value: (str(image_records[value]["file_name"]), value)
    )
    qualifying: list[tuple[int, list[str]]] = []
    for image_id in ordered_ids:
        report.source_images += 1
        crowd = crowd_by_image.get(image_id, 0)
        if crowd:
            report.crowd_annotations += crowd
            report.images_skipped_crowd += 1
            continue
        size = _image_size(image_records[image_id]) or _image_dimensions(image_paths[image_id])
        if size is None:
            report.images_skipped_missing += 1
            continue
        width, height = size

        lines: list[str] = []
        for annotation in persons_by_image.get(image_id, []):
            line = _label_for_person(annotation, schema, width, height, config.min_visible)
            if line is None:
                report.persons_unlabeled += 1
                continue
            lines.append(line)
        if not lines:
            report.images_skipped_no_label += 1
            continue
        report.qualifying_images += 1
        qualifying.append((image_id, lines))

    selected = _select_images(qualifying, config, report.target_count)
    for image_id, lines in selected:
        source = image_paths[image_id]
        stem = Path(str(image_records[image_id]["file_name"])).stem.replace(" ", "_")
        output_stem = f"{image_id:012d}_{stem}"
        try:
            _write_image(source, image_dir / f"{output_stem}.jpg", config.jpeg_quality)
        except OSError as exc:
            LOGGER.warning("cannot copy image %s: %s", source, exc)
            report.images_skipped_missing += 1
            continue
        (label_dir / f"{output_stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        report.images_written += 1
        report.labels_written += len(lines)
        report.persons_labeled += len(lines)

    write_dataset_yaml(config.out, schema, config.split)
    (config.out / "schema.json").write_text(
        json.dumps(schema.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    update_stats(config.out, report)  # type: ignore[arg-type]
    if report.images_written == 0:
        LOGGER.warning("no images were written; check the JSON split and visibility flags")
    return report


def render_labeled_preview(
    image_dir: Path,
    label_dir: Path,
    out: Path,
    *,
    count: int = 2,
    n_kpt: int = HALFBUST62_N_KPT,
    max_width: int = 960,
    jpeg_quality: int = 92,
) -> Path:
    """Stack ``count`` labelled frames (GT boxes + keypoints) into one JPEG.

    Args:
        image_dir: ``images/<split>`` directory.
        label_dir: ``labels/<split>`` directory of the same split.
        out: JPEG path to write.
        count: Number of frames, in sorted stem order.
        n_kpt: Keypoints per label line.
        max_width: Resize each frame to at most this width before stacking.
        jpeg_quality: JPEG quality of the composite.

    Returns:
        ``out``.

    Raises:
        ValueError: If no label file or no readable frame is found.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    labels = sorted(Path(label_dir).glob("*.txt"))[:count]
    if not labels:
        raise ValueError(f"no label files under {label_dir}")
    frames: list[Image.Image] = []
    for label_path in labels:
        image_path = Path(image_dir) / f"{label_path.stem}.jpg"
        image = load_image(image_path)
        if image is None:
            continue
        targets = read_yolo_pose_labels(
            label_path.read_text(encoding="utf-8"), image.width, image.height, n_kpt
        )
        draw = ImageDraw.Draw(image)
        for target in targets:
            x1, y1, x2, y2 = (float(value) for value in target.box)
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
            for x, y, visibility in target.keypoints:
                if visibility <= 0:
                    continue
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 40, 40))
        if image.width > max_width:
            scale = max_width / image.width
            image = image.resize(
                (max_width, max(1, int(round(image.height * scale)))),
                Image.Resampling.LANCZOS,
            )
        frames.append(image)
    if not frames:
        raise ValueError(f"no readable frames for the labels under {label_dir}")
    gutter = 8
    width = max(frame.width for frame in frames)
    height = sum(frame.height for frame in frames) + gutter * (len(frames) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    top = 0
    for frame in frames:
        canvas.paste(frame, (0, top))
        top += frame.height + gutter
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="JPEG", quality=jpeg_quality)
    return Path(out)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True, help="COCO 2017 root")
    parser.add_argument("--wholebody-json", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--schema", choices=("halfbust62", "coco17"), default="halfbust62")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=FULL_FRACTION)
    parser.add_argument(
        "--half-train-root",
        type=Path,
        default=None,
        help="half-bust dataset root; its stats.json sets the full-frame count",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-visible", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--preview", type=Path, default=None, help="write a GT preview JPEG")
    parser.add_argument("--preview-count", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    half_count = half_train_count(args.half_train_root) if args.half_train_root else None
    report = build_fullframe_dataset(
        FullFrameConfig(
            coco_root=args.coco_root,
            wholebody_json=args.wholebody_json,
            split=args.split,
            schema=args.schema,
            out=args.out,
            max_images=args.max_images,
            target_count=args.target_count,
            fraction=args.fraction,
            half_train_count=half_count,
            seed=args.seed,
            min_visible=args.min_visible,
            jpeg_quality=args.jpeg_quality,
        )
    )
    if args.preview is not None:
        preview = render_labeled_preview(
            args.out / "images" / args.split,
            args.out / "labels" / args.split,
            args.preview,
            count=args.preview_count,
        )
        print(f"preview: {preview}")
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
