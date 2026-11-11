"""Auto-label top-down F200 IR frames into a ``hand21`` pose dataset.

The deployed hand model sees the whole IR frame after the runtime preprocessing
(min-max normalisation, CLAHE ``3.0`` on an ``(8, 8)`` grid, three equal
channels).  This module feeds the same preprocessed frames to an existing
``hand21`` checkpoint, keeps only frames whose labels can be trusted, and writes
a YOLO-pose dataset that :mod:`cpsa_ml.finetune_hand` can train on::

    python -m cpsa_ml.datasets.irpseudo \\
        --frames build/ir_capture_20261112 --weights build/models/hand_topdown/best.pt \\
        --out build/ir_pseudo --conf 0.5 --kpt-conf 0.5 --val-ratio 0.15

A frame is kept only when **every** detected hand passes ``--conf`` and has at
least :data:`MIN_CONFIDENT_KEYPOINTS` keypoints above ``--kpt-conf``.  A frame
with one shaky hand is dropped whole, so the dataset never teaches the model to
ignore a hand that was left unlabelled.  Keypoints below ``--kpt-conf`` are
written with ``v = 0``.

Besides the dataset (``images/<split>``, ``labels/<split>``, ``data.yaml``) the
run writes ``stats.json`` (kept/dropped counts, thresholds, sources) and
``contact_sheet.png`` so a human can review the labels before training.
``--mix-handframe`` copies a seeded subset of an existing full-frame dataset's
train split into the new train split to limit forgetting.

The recordings contain faces: keep the frames and the derived datasets local;
do not upload them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cpsa_ml.datasets.halfbust import format_label
from cpsa_ml.datasets.handframe import (
    CLAHE_CLIP,
    CLAHE_TILES,
    apply_clahe,
    render_label_contact_sheet,
)
from cpsa_ml.datasets.hands import write_hand_dataset_yaml
from cpsa_ml.schema import get_schema

__all__ = [
    "CHANNELS",
    "KEYPOINTS",
    "MIN_CONFIDENT_KEYPOINTS",
    "PREDICT_CONF_FLOOR",
    "PREDICT_IMGSZ",
    "HandPrediction",
    "IrPseudoConfig",
    "YoloPosePredictor",
    "build_dataset",
    "iter_ir_frames",
    "label_frame",
    "main",
    "preprocess_ir",
    "split_for",
]

LOGGER = logging.getLogger(__name__)

KEYPOINTS = 21
"""Keypoints of the ``hand21`` schema."""

CHANNELS = 3
"""Channels of the runtime model input (a gray image replicated three times)."""

MIN_CONFIDENT_KEYPOINTS = 15
"""Keypoints a hand needs above ``kpt_conf`` to be labelled at all."""

FILLED_VISIBILITY = 1.0
"""Visibility of a labelled keypoint (``0`` for one below ``kpt_conf``)."""

PREDICT_IMGSZ = 320
"""Square inference size matching the training size of the hand model."""

PREDICT_CONF_FLOOR = 0.05
"""Lowest detection confidence handed to the per-frame gates.

Ultralytics' default ``conf=0.25`` would silently remove weak hands *before*
the "drop the whole frame" rule runs, leaving those hands unlabelled.
"""

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})

SPLITS = ("train", "val")


@dataclass(frozen=True)
class HandPrediction:
    """One predicted hand in source-image pixels.

    Attributes:
        box: ``(x1, y1, x2, y2)`` in pixels.
        conf: Detection confidence.
        keypoints: ``(21, 3)`` array of ``x, y, confidence`` in pixels.
    """

    box: tuple[float, float, float, float]
    conf: float
    keypoints: NDArray[np.float32]

    def __post_init__(self) -> None:
        points = np.asarray(self.keypoints, dtype=np.float32)
        if points.shape != (KEYPOINTS, 3):
            raise ValueError(
                f"keypoints must be ({KEYPOINTS}, 3), got {tuple(points.shape)}; "
                "the weights must be a hand21 pose checkpoint"
            )
        object.__setattr__(self, "keypoints", points)


class Predictor(Protocol):
    """Anything that labels one RGB, ``(H, W, 3)`` uint8 image."""

    def __call__(self, image: NDArray[np.uint8]) -> Sequence[HandPrediction]: ...


PredictorFactory = Callable[[Path, str], Predictor]


@dataclass(frozen=True)
class IrPseudoConfig:
    """Inputs, gates and output paths of one auto-label run.

    Args:
        out: Dataset root; ``images/<split>``, ``labels/<split>``,
            ``data.yaml``, ``stats.json`` and ``contact_sheet.png`` are written
            below it.
        frames: Directory of raw IR frames (``.png``/``.jpg``).
        frames_npy: ``N x H x W`` uint8 stack of raw IR frames instead.
        weights: Checkpoint used for the labels; recorded in ``stats.json``.
        conf: Minimum detection confidence of every hand in a kept frame.
        kpt_conf: Minimum keypoint confidence; below it a keypoint gets ``v=0``.
        min_keypoints: Keypoints above ``kpt_conf`` needed per hand.
        val_ratio: Seeded share of the kept frames held out for validation.
        seed: Split seed and the seed of the ``--mix-handframe`` draw.
        mix_handframe: Existing full-frame ``hand21`` dataset to mix in.
        mix_ratio: Mixed train frames as a multiple of the kept frames.
        overwrite: Allow writing into an existing output dataset.
    """

    out: Path
    frames: Path | None = None
    frames_npy: Path | None = None
    weights: Path | None = None
    conf: float = 0.5
    kpt_conf: float = 0.5
    min_keypoints: int = MIN_CONFIDENT_KEYPOINTS
    val_ratio: float = 0.15
    seed: int = 0
    mix_handframe: Path | None = None
    mix_ratio: float = 1.0
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "out", Path(self.out))
        for name in ("frames", "frames_npy", "weights", "mix_handframe"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if (self.frames is None) == (self.frames_npy is None):
            raise ValueError("give exactly one of frames and frames_npy")
        if not 0.0 < self.conf <= 1.0:
            raise ValueError("conf must be in (0, 1]")
        if not 0.0 <= self.kpt_conf <= 1.0:
            raise ValueError("kpt_conf must be in [0, 1]")
        if not 0 < self.min_keypoints <= KEYPOINTS:
            raise ValueError(f"min_keypoints must be in [1, {KEYPOINTS}]")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.mix_ratio < 0.0:
            raise ValueError("mix_ratio must be non-negative")


@dataclass
class _Counters:
    """Per-run and per-split counters written to ``stats.json``."""

    seen: int = 0
    kept: int = 0
    unreadable: int = 0
    drop_reasons: Counter[str] = field(default_factory=Counter)
    labeled_keypoints: int = 0
    low_confidence_keypoints: int = 0
    splits: dict[str, dict[str, int]] = field(
        default_factory=lambda: {split: {"frames": 0, "hands": 0} for split in SPLITS}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "dropped": sum(self.drop_reasons.values()),
            "unreadable": self.unreadable,
            "drop_reasons": dict(sorted(self.drop_reasons.items())),
            "keypoints": {
                "labeled": self.labeled_keypoints,
                "below_kpt_conf": self.low_confidence_keypoints,
            },
            "splits": {split: dict(values) for split, values in self.splits.items()},
        }


def split_for(name: str, *, seed: int, val_ratio: float) -> str:
    """Assign a stable split from a seeded hash of the frame name."""

    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2.0**32
    return "val" if bucket < val_ratio else "train"


def preprocess_ir(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply the runtime IR preprocessing to one ``(H, W)`` gray frame.

    Min-max normalisation, then ``cv2.createCLAHE(3.0, (8, 8))`` (the constants
    of :mod:`cpsa_ml.datasets.handframe`), then three equal channels, exactly
    like ``cpsa.adapters.camera.irdepth.ir_to_input`` on the board.

    Raises:
        ValueError: If ``gray`` is not a 2-D uint8 array.
    """

    image = np.asarray(gray)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"frame must be 2-D uint8, got {image.shape} {image.dtype}")
    enhanced = apply_clahe(image)
    return np.repeat(enhanced[:, :, np.newaxis], CHANNELS, axis=2)


def _read_gray(path: Path) -> NDArray[np.uint8] | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else np.ascontiguousarray(image)


def iter_ir_frames(
    frames: Path | None, frames_npy: Path | None
) -> Iterator[tuple[str, NDArray[np.uint8] | None]]:
    """Yield ``(name, gray frame)`` from a frame directory or an ``.npy`` stack.

    ``gray`` is ``None`` when an image file cannot be decoded.
    """

    if frames is not None:
        paths = sorted(
            path
            for path in Path(frames).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in paths:
            yield path.stem, _read_gray(path)
        return

    if frames_npy is None:  # pragma: no cover - IrPseudoConfig enforces one source
        raise ValueError("give frames or frames_npy")
    stack = np.load(frames_npy)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3 or stack.dtype != np.uint8:
        raise ValueError(f"{frames_npy} must be N x H x W uint8, got {stack.shape} {stack.dtype}")
    width = max(5, len(str(len(stack) - 1)))
    for index in range(len(stack)):
        yield f"ir_{index:0{width}d}", np.ascontiguousarray(stack[index])


def _hand_label(
    hand: HandPrediction,
    *,
    width: int,
    height: int,
    kpt_conf: float,
    min_keypoints: int,
) -> tuple[str | None, str, int, int]:
    """Turn one prediction into a label line.

    Returns:
        ``(line, reason, labeled_keypoints, low_confidence_keypoints)``; ``line``
        is ``None`` and ``reason`` names the gate that rejected the hand.
    """

    x1, y1, x2, y2 = hand.box
    if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
        return None, "invalid_box", 0, 0
    x1 = float(np.clip(x1, 0.0, width))
    x2 = float(np.clip(x2, 0.0, width))
    y1 = float(np.clip(y1, 0.0, height))
    y2 = float(np.clip(y2, 0.0, height))
    if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
        return None, "invalid_box", 0, 0

    points = np.zeros((KEYPOINTS, 3), dtype=np.float32)
    low = 0
    for index, (x, y, confidence) in enumerate(hand.keypoints):
        if confidence >= kpt_conf:
            points[index] = (
                float(np.clip(x / width, 0.0, 1.0)),
                float(np.clip(y / height, 0.0, 1.0)),
                FILLED_VISIBILITY,
            )
        else:
            low += 1
    labeled = KEYPOINTS - low
    if labeled < min_keypoints:
        return None, "low_keypoints", labeled, low

    box = (
        0.5 * (x1 + x2) / width,
        0.5 * (y1 + y2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )
    return format_label(0, points, box), "", labeled, low


def label_frame(
    predictions: Sequence[HandPrediction],
    *,
    width: int,
    height: int,
    conf: float,
    kpt_conf: float,
    min_keypoints: int = MIN_CONFIDENT_KEYPOINTS,
) -> tuple[list[str], str, int, int]:
    """Return ``(label lines, drop reason, labeled keypoints, low-confidence ones)``.

    A frame is kept only when every detected hand passes ``conf`` and has at
    least ``min_keypoints`` keypoints above ``kpt_conf``.  The first failing
    hand decides the reason; a kept frame has one line per hand and the keypoint
    counts cover all of its hands.
    """

    if not predictions:
        return [], "no_hand", 0, 0
    if any(hand.conf < conf for hand in predictions):
        return [], "low_conf_hand", 0, 0
    lines: list[str] = []
    labeled = 0
    low = 0
    for hand in predictions:
        line, reason, hand_labeled, hand_low = _hand_label(
            hand, width=width, height=height, kpt_conf=kpt_conf, min_keypoints=min_keypoints
        )
        if line is None:
            return [], reason, 0, 0
        lines.append(line)
        labeled += hand_labeled
        low += hand_low
    return lines, "", labeled, low


class YoloPosePredictor:
    """Ultralytics ``YOLO.predict`` wrapped as a :class:`Predictor`."""

    def __init__(
        self,
        weights: Path,
        *,
        imgsz: int = PREDICT_IMGSZ,
        device: str = "cpu",
        conf_floor: float = PREDICT_CONF_FLOOR,
    ) -> None:
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"weights not found: {self.weights}")
        self.imgsz = int(imgsz)
        self.device = device
        self.conf_floor = float(conf_floor)
        self._yolo = YOLO(str(self.weights), verbose=False)

    def __call__(self, image: NDArray[np.uint8]) -> list[HandPrediction]:
        results = self._yolo.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf_floor,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or len(boxes) == 0:
            return []
        if keypoints is None:
            raise ValueError(
                "the model returned boxes without keypoints; the weights must be a pose checkpoint"
            )
        xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float64)
        confidences = np.asarray(boxes.conf.cpu().numpy(), dtype=np.float64)
        points = np.asarray(keypoints.data.cpu().numpy(), dtype=np.float32)
        if points.ndim != 3 or points.shape[1:] != (KEYPOINTS, 3):
            raise ValueError(
                f"the model predicts keypoints of shape {points.shape}; hand21 needs "
                f"(N, {KEYPOINTS}, 3)"
            )
        return [
            HandPrediction(
                box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                conf=float(confidence),
                keypoints=points[index],
            )
            for index, (row, confidence) in enumerate(zip(xyxy, confidences, strict=True))
        ]


def _prepare_out(out: Path, overwrite: bool) -> None:
    """Create ``out`` or clear a previous dataset below it."""

    known = (out / "images", out / "labels")
    existing = [path for path in known if path.is_dir() and any(path.iterdir())]
    exists = bool(existing) or (out / "data.yaml").is_file()
    if exists and not overwrite:
        raise FileExistsError(
            f"{out} already holds a dataset; pass --overwrite or choose a fresh directory"
        )
    if overwrite:
        for path in known:
            if path.is_dir():
                shutil.rmtree(path)
        for name in ("data.yaml", "stats.json", "contact_sheet.png"):
            (out / name).unlink(missing_ok=True)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)


def _unique_stem(name: str, used: set[str]) -> str:
    """Return ``name``, suffixed if two sources share a stem."""

    stem = name
    index = 1
    while stem in used:
        stem = f"{name}_{index}"
        index += 1
    used.add(stem)
    return stem


def _move_one_frame_to_val(
    out: Path, written: list[tuple[str, str]], counters: _Counters, val_ratio: float
) -> None:
    """Keep a usable validation split when the hash put every frame in train.

    ``val_ratio == 0`` is respected: a caller who asks for a train-only dataset
    gets one.
    """

    if val_ratio <= 0.0 or counters.splits["val"]["frames"] > 0:
        return
    if counters.splits["train"]["frames"] < 2:
        return
    split, stem = written.pop()
    for kind, suffix in (("images", ".png"), ("labels", ".txt")):
        source = out / kind / split / f"{stem}{suffix}"
        target = out / kind / "val" / f"{stem}{suffix}"
        if source.is_file():
            source.replace(target)
    counters.splits[split]["frames"] -= 1
    counters.splits["val"]["frames"] += 1
    LOGGER.warning("no frame hashed into val; moved %s to val", stem)


def _mix_handframe(config: IrPseudoConfig, kept: int, counters: _Counters) -> dict[str, Any]:
    """Copy a seeded subset of an existing train split into ``images/train``."""

    info: dict[str, Any] = {
        "root": None if config.mix_handframe is None else str(config.mix_handframe),
        "available": 0,
        "requested": 0,
        "copied": 0,
    }
    if config.mix_handframe is None:
        return info

    root = config.mix_handframe
    images_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    candidates = (
        sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and (labels_dir / f"{path.stem}.txt").is_file()
        )
        if images_dir.is_dir()
        else []
    )
    info["available"] = len(candidates)
    requested = int(round(kept * config.mix_ratio))
    info["requested"] = requested
    if requested <= 0 or not candidates:
        if requested > 0:
            LOGGER.warning("no labelled train frames under %s; nothing mixed in", images_dir)
        return info

    rng = random.Random(config.seed)
    chosen = rng.sample(candidates, min(requested, len(candidates)))
    for image in chosen:
        stem = f"mix_{image.stem}"
        # PNG keeps the dataset single-format, so the contact sheet always finds
        # the mixed frames next to the auto-labelled ones.
        with Image.open(image) as source:
            source.convert("RGB").save(config.out / "images" / "train" / f"{stem}.png")
        text = (labels_dir / f"{image.stem}.txt").read_text(encoding="utf-8")
        (config.out / "labels" / "train" / f"{stem}.txt").write_text(text, encoding="utf-8")
        counters.splits["train"]["frames"] += 1
        counters.splits["train"]["hands"] += sum(1 for line in text.splitlines() if line.strip())
        info["copied"] += 1
    LOGGER.info("mixed %d frames from %s", info["copied"], root)
    return info


def _contact_sheet(out: Path, counters: _Counters) -> str | None:
    """Render labelled frames of the largest non-empty split."""

    split = max(SPLITS, key=lambda name: counters.splits[name]["frames"])
    if counters.splits[split]["frames"] == 0:
        return None
    count = min(16, counters.splits[split]["frames"])
    path = render_label_contact_sheet(
        out / "images" / split,
        out / "labels" / split,
        out / "contact_sheet.png",
        count=count,
    )
    LOGGER.info("review the labels in %s", path)
    return path.relative_to(out).as_posix()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_dataset(config: IrPseudoConfig, predictor: Predictor) -> dict[str, Any]:
    """Auto-label the frames, write the dataset, and return the ``stats.json`` payload."""

    _prepare_out(config.out, config.overwrite)
    counters = _Counters()
    written: list[tuple[str, str]] = []
    used_stems: set[str] = set()

    for name, gray in iter_ir_frames(config.frames, config.frames_npy):
        counters.seen += 1
        if gray is None:
            counters.unreadable += 1
            counters.drop_reasons["unreadable"] += 1
            continue
        image = preprocess_ir(gray)
        height, width = image.shape[:2]
        lines, reason, labeled, low = label_frame(
            predictor(image),
            width=width,
            height=height,
            conf=config.conf,
            kpt_conf=config.kpt_conf,
            min_keypoints=config.min_keypoints,
        )
        if reason:
            counters.drop_reasons[reason] += 1
            continue
        counters.labeled_keypoints += labeled
        counters.low_confidence_keypoints += low

        stem = _unique_stem(name, used_stems)
        split = split_for(stem, seed=config.seed, val_ratio=config.val_ratio)
        Image.fromarray(image).save(config.out / "images" / split / f"{stem}.png", format="PNG")
        (config.out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        counters.kept += 1
        counters.splits[split]["frames"] += 1
        counters.splits[split]["hands"] += len(lines)
        written.append((split, stem))

    _move_one_frame_to_val(config.out, written, counters, config.val_ratio)
    mix = _mix_handframe(config, len(written), counters)
    write_hand_dataset_yaml(config.out, get_schema("hand21"))
    contact_sheet = _contact_sheet(config.out, counters)

    payload: dict[str, Any] = {
        "schema": "hand21",
        "kpt_shape": list(get_schema("hand21").kpt_shape),
        "preprocessing": {
            "clahe_clip": CLAHE_CLIP,
            "clahe_tiles": list(CLAHE_TILES),
            "channels": CHANNELS,
        },
        "source": {
            "frames": None if config.frames is None else str(config.frames),
            "frames_npy": None if config.frames_npy is None else str(config.frames_npy),
            "weights": None if config.weights is None else str(config.weights),
        },
        "thresholds": {
            "conf": config.conf,
            "kpt_conf": config.kpt_conf,
            "min_keypoints": config.min_keypoints,
        },
        "val_ratio": config.val_ratio,
        "seed": config.seed,
        "mix": mix,
        "contact_sheet": contact_sheet,
        "frames": counters.to_dict(),
        "license_warning": (
            "Board recordings may contain faces: keep this dataset and the frames local."
        ),
    }
    _write_json_atomic(config.out / "stats.json", payload)
    LOGGER.info(
        "kept %d of %d frames (%d dropped): %s",
        counters.kept,
        counters.seen,
        counters.seen - counters.kept,
        dict(sorted(counters.drop_reasons.items())),
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=None, help="directory of raw IR frames")
    parser.add_argument("--frames-npy", type=Path, default=None, help="N x H x W uint8 IR stack")
    parser.add_argument("--weights", type=Path, required=True, help="hand21 checkpoint (best.pt)")
    parser.add_argument("--out", type=Path, required=True, help="output dataset root")
    parser.add_argument("--conf", type=float, default=0.5, help="per-hand detection threshold")
    parser.add_argument("--kpt-conf", type=float, default=0.5, help="per-keypoint threshold")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="held-out share of frames")
    parser.add_argument("--seed", type=int, default=0, help="split and mix seed")
    parser.add_argument("--mix-handframe", type=Path, default=None, help="dataset to mix in")
    parser.add_argument(
        "--mix-ratio", type=float, default=1.0, help="mixed train frames per kept frame"
    )
    parser.add_argument("--device", default="cpu", help='"cpu" or a CUDA index for predict')
    parser.add_argument(
        "--overwrite", action="store_true", help="replace a dataset already under --out"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    predictor_factory: PredictorFactory | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = IrPseudoConfig(
        out=args.out,
        frames=args.frames,
        frames_npy=args.frames_npy,
        weights=args.weights,
        conf=args.conf,
        kpt_conf=args.kpt_conf,
        val_ratio=args.val_ratio,
        seed=args.seed,
        mix_handframe=args.mix_handframe,
        mix_ratio=args.mix_ratio,
        overwrite=args.overwrite,
    )
    factory: PredictorFactory = (
        predictor_factory
        if predictor_factory is not None
        else (lambda weights, device: YoloPosePredictor(weights, device=device))
    )
    payload = build_dataset(config, factory(args.weights, args.device))
    frames = payload["frames"]
    if frames["kept"] == 0:
        print(
            f"no frame passed the gates ({frames['drop_reasons']}); lower --conf/--kpt-conf "
            "or capture more frames",
            file=sys.stderr,
        )
        return 1
    print(
        f"kept {frames['kept']} of {frames['seen']} frames; "
        f"train {frames['splits']['train']['frames']}, val {frames['splits']['val']['frames']}"
    )
    print(f"wrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
