#!/usr/bin/env python3
"""Score one variant's INT8 and FP32 heads with the same metric code (host).

``tools/vitis/dump_quant_outputs.py`` runs in the container and writes
``quant_outputs.npz``: the ``vai_q_pytorch`` test-mode head tensors converted to
the int8 values the DPU produces, with ``fix_point`` and the letterbox
parameters of every image.  This script picks the dump up on the host and

1. decodes the int8 heads with :class:`cpsa.vision.pose_decode.QuantizedRawHeadDecoder`
   (the runtime fast path), mapping boxes and keypoints back with
   :func:`cpsa.vision.pose_decode.unletterbox`;
2. optionally rebuilds the same deploy graph in float from ``best.pt`` and
   decodes its heads with :func:`cpsa.vision.pose_decode.decode_raw_heads`;
3. scores both with :func:`cpsa_ml.eval.evaluate` against the YOLO-pose labels,
   so FP32 and INT8 differ only in the tensors, never in the metric code;
4. writes ``metrics_int8.json`` and ``metrics_fp32_host.json`` plus a
   ``comparison.json``/``comparison.md`` table against a baseline report.

The variant study uses the table to accept a faster activation/input pairing
only when its INT8 wrist, hand and face PCK@0.2fw stay within two points of the
baseline (see ``ml/README.md``); ``decision`` in ``comparison.json`` evaluates
that rule for the configured groups.

Example::

    PYTHONPATH=src python tools/vitis/eval_quant.py \\
        --dump build/variant_study/hswish_480x640/quant/quant_outputs.npz \\
        --weights build/variant_study/hswish_480x640/best.pt \\
        --data /data/halfbust62 --test-split val \\
        --baseline build/variant_study/baseline/quant/metrics_int8.json \\
        --out build/variant_study/hswish_480x640/eval

Half-body and full-frame accuracy side by side
----------------------------------------------
One dump can hold both val splits: ``dump_quant_outputs.py`` tags each frame
with the group of its ``--test-images NAME=DIR`` source (the two datasets share
COCO image stems, so file names alone cannot tell them apart).  Repeat
``--eval-split NAME=IMAGES:LABELS`` to score every group separately and write
``comparison_splits.md``/``comparison_splits.json`` next to the usual reports::

    PYTHONPATH=src python tools/vitis/eval_quant.py \\
        --dump quant/quant_outputs.npz --weights best.pt \\
        --eval-split half=/data/halfbust62/images/val:/data/halfbust62/labels/val \\
        --eval-split full=/data/fullframe62/images/val:/data/fullframe62/labels/val \\
        --out eval

The first split keeps the plain ``metrics_int8.json``/``comparison.json``
names, so the variant-study decision rule and its baseline keep working.  The
``framing.*`` rows show how many people each split really shows as full body,
using the runtime threshold.

Runs on the host (Python 3.10); torch is imported only for the FP32 path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parents[1]
for _search_path in (ROOT / "src", ROOT / "ml/src", TOOLS_DIR):
    if str(_search_path) not in sys.path:
        sys.path.insert(0, str(_search_path))

from cpsa_ml.eval import (  # noqa: E402
    HALFBUST62_GROUPS,
    EvalConfig,
    ImageSample,
    PersonPrediction,
    PersonTarget,
    evaluate,
    read_yolo_pose_labels,
)

from cpsa.vision.layouts import HAND21  # noqa: E402
from cpsa.vision.pose_decode import (  # noqa: E402
    QuantizedRawHeadDecoder,
    decode_raw_heads,
    letterbox,
    unletterbox,
)
from cpsa.vision.types import PersonPose  # noqa: E402

DEFAULT_DEPLOY_MODULE = ROOT / "ml/src/cpsa_ml/deploy.py"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

#: Groups whose INT8 PCK@0.2fw must stay close to the baseline (proposal rule).
DEFAULT_DECISION_GROUPS: tuple[str, ...] = ("wrists", "left_hand", "right_hand", "face")
DECISION_PCK = "0.20"
DECISION_MARGIN = 0.02


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantDump:
    """Contents of ``quant_outputs.npz``, arrays aligned to ``strides``."""

    path: Path
    names: tuple[str, ...]
    groups: tuple[str, ...]
    image_widths: NDArray[np.int64]
    image_heights: NDArray[np.int64]
    scales: NDArray[np.float64]
    pad_x: NDArray[np.float64]
    pad_y: NDArray[np.float64]
    strides: tuple[int, ...]
    fix_points: tuple[int, ...]
    input_size: tuple[int, int]
    n_kpt: int
    reg_max: int
    num_classes: int
    channel_layout: dict[str, int]
    heads: tuple[NDArray[np.int8], ...]

    @property
    def shapes(self) -> tuple[tuple[int, int, int], ...]:
        """Per-stride ``(H, W, C)`` NHWC shape of the dumped heads."""

        return tuple(tuple(int(size) for size in head.shape[1:]) for head in self.heads)

    def __len__(self) -> int:
        return len(self.names)

    def subset(self, indices: Sequence[int]) -> QuantDump:
        """A copy holding only ``indices`` (in that order)."""

        order = np.asarray(list(indices), dtype=np.intp)
        if order.size == 0:
            raise ValueError("indices must not be empty")
        return QuantDump(
            path=self.path,
            names=tuple(self.names[int(index)] for index in order),
            groups=tuple(self.groups[int(index)] for index in order),
            image_widths=self.image_widths[order],
            image_heights=self.image_heights[order],
            scales=self.scales[order],
            pad_x=self.pad_x[order],
            pad_y=self.pad_y[order],
            strides=self.strides,
            fix_points=self.fix_points,
            input_size=self.input_size,
            n_kpt=self.n_kpt,
            reg_max=self.reg_max,
            num_classes=self.num_classes,
            channel_layout=self.channel_layout,
            heads=tuple(head[order] for head in self.heads),
        )


def _require(data: dict[str, NDArray[np.generic]], key: str, path: Path) -> NDArray[np.generic]:
    if key not in data:
        raise SystemExit(f"{path} has no '{key}' array; re-run dump_quant_outputs.py")
    return data[key]


def _scalar(data: dict[str, NDArray[np.generic]], key: str, path: Path) -> int:
    array = np.asarray(_require(data, key, path)).reshape(-1)
    if array.size != 1:
        raise SystemExit(f"{path}: '{key}' must be a scalar, got shape {array.shape}")
    return int(array[0])


def load_dump(path: Path, limit: int | None = None) -> QuantDump:
    """Read and validate a ``quant_outputs.npz`` (optionally only ``limit`` frames)."""

    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"quant dump not found: {source}")
    with np.load(source, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}

    names = tuple(str(name) for name in np.asarray(_require(data, "names", source)).tolist())
    groups_raw = data.get("groups")
    if groups_raw is None:
        groups = tuple("" for _ in names)
    else:
        groups = tuple(str(group) for group in np.asarray(groups_raw).tolist())
        if len(groups) < len(names):
            raise SystemExit(f"{source}: 'groups' has fewer entries than 'names'")
    strides = tuple(int(value) for value in np.atleast_1d(_require(data, "strides", source)))
    fix_points = tuple(int(value) for value in np.atleast_1d(_require(data, "fix_points", source)))
    if len(strides) != len(fix_points):
        raise SystemExit(f"{source}: {len(strides)} strides but {len(fix_points)} fix points")

    input_size_raw = np.atleast_1d(_require(data, "input_size", source))
    if input_size_raw.size != 2:
        raise SystemExit(f"{source}: input_size must have two values")
    input_size = (int(input_size_raw[0]), int(input_size_raw[1]))

    layout_raw = np.atleast_1d(_require(data, "channel_layout", source))
    if layout_raw.size != 3:
        raise SystemExit(f"{source}: channel_layout must have box/class/keypoint")
    channel_layout = {
        "box": int(layout_raw[0]),
        "class": int(layout_raw[1]),
        "keypoint": int(layout_raw[2]),
    }
    n_kpt = _scalar(data, "n_kpt", source)
    reg_max = _scalar(data, "reg_max", source)
    num_classes = _scalar(data, "num_classes", source)
    if channel_layout["class"] != num_classes:
        raise SystemExit(
            f"{source}: class channels {channel_layout['class']} != num_classes {num_classes}"
        )
    expected_channels = channel_layout["box"] + channel_layout["class"] + 3 * n_kpt
    if channel_layout["box"] != 4 * reg_max:
        raise SystemExit(
            f"{source}: box channels {channel_layout['box']} != 4 * reg_max {4 * reg_max}"
        )
    if channel_layout["keypoint"] != 3 * n_kpt:
        raise SystemExit(
            f"{source}: keypoint channels {channel_layout['keypoint']} != 3 * n_kpt {3 * n_kpt}"
        )

    heads: list[NDArray[np.int8]] = []
    for stride in strides:
        key = f"int8_head_stride{stride}"
        head = np.asarray(_require(data, key, source))
        if head.dtype != np.int8:
            raise SystemExit(f"{source}: '{key}' is {head.dtype}, expected int8")
        if head.ndim != 4:
            raise SystemExit(f"{source}: '{key}' must be (N, H, W, C), got {head.shape}")
        if head.shape[0] != len(names):
            raise SystemExit(f"{source}: '{key}' has {head.shape[0]} frames but {len(names)} names")
        height, width, channels = (int(size) for size in head.shape[1:])
        if channels != expected_channels:
            raise SystemExit(
                f"{source}: '{key}' has {channels} channels, expected {expected_channels}"
            )
        if input_size[0] % height or input_size[1] % width:
            raise SystemExit(f"{source}: '{key}' {height}x{width} does not divide {input_size}")
        stride_y, stride_x = input_size[0] // height, input_size[1] // width
        if stride_y != stride or stride_x != stride:
            raise SystemExit(
                f"{source}: '{key}' implies stride {(stride_y, stride_x)}, expected {stride}"
            )
        heads.append(head)

    count = len(names) if limit is None else min(int(limit), len(names))
    if count <= 0:
        raise SystemExit(f"{source}: no frames to evaluate")
    for key in ("image_widths", "image_heights", "scales", "pad_x", "pad_y"):
        if len(np.atleast_1d(_require(data, key, source))) < count:
            raise SystemExit(f"{source}: '{key}' has fewer than {count} entries")

    return QuantDump(
        path=source,
        names=names[:count],
        groups=groups[:count],
        image_widths=np.asarray(data["image_widths"][:count], dtype=np.int64),
        image_heights=np.asarray(data["image_heights"][:count], dtype=np.int64),
        scales=np.asarray(data["scales"][:count], dtype=np.float64),
        pad_x=np.asarray(data["pad_x"][:count], dtype=np.float64),
        pad_y=np.asarray(data["pad_y"][:count], dtype=np.float64),
        strides=strides,
        fix_points=fix_points,
        input_size=input_size,
        n_kpt=n_kpt,
        reg_max=reg_max,
        num_classes=num_classes,
        channel_layout=channel_layout,
        heads=tuple(head[:count] for head in heads),
    )


# ---------------------------------------------------------------------------
# Ground truth and predictions
# ---------------------------------------------------------------------------


def read_targets(
    labels_dir: Path,
    name: str,
    width: int,
    height: int,
    n_kpt: int,
) -> tuple[PersonTarget, ...]:
    """Read ``<labels_dir>/<name>.txt`` as pixel-space targets (empty file: none)."""

    label_path = Path(labels_dir) / f"{name}.txt"
    if not label_path.is_file():
        raise SystemExit(f"missing label file: {label_path}")
    return read_yolo_pose_labels(
        label_path.read_text(encoding="utf-8"), width=width, height=height, n_kpt=n_kpt
    )


def to_prediction(person: PersonPose) -> PersonPrediction:
    """Convert a decoder :class:`cpsa.vision.types.PersonPose` to an eval prediction."""

    xy = np.asarray(person.xy, dtype=np.float64)
    conf = np.asarray(person.conf, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"person.xy must be (K, 2), got {xy.shape}")
    if conf.shape != (xy.shape[0],):
        raise ValueError(f"person.conf must be ({xy.shape[0]},), got {conf.shape}")
    keypoints = np.concatenate([xy, conf[:, None]], axis=1)
    return PersonPrediction(
        box=np.asarray(person.box, dtype=np.float64),
        score=float(person.score),
        keypoints=keypoints,
    )


def build_samples(
    frames: Sequence[Sequence[PersonPose]],
    names: Sequence[str],
    labels_dir: Path,
    widths: Sequence[int],
    heights: Sequence[int],
    n_kpt: int,
) -> list[ImageSample]:
    """Pair decoded frames with their ground truth, in the dump's order."""

    if not (len(frames) == len(names) == len(widths) == len(heights)):
        raise ValueError(
            f"got {len(frames)} frames, {len(names)} names, {len(widths)} widths and "
            f"{len(heights)} heights"
        )
    samples: list[ImageSample] = []
    for index, name in enumerate(names):
        samples.append(
            ImageSample(
                name=name,
                predictions=tuple(to_prediction(person) for person in frames[index]),
                targets=read_targets(
                    labels_dir, name, int(widths[index]), int(heights[index]), n_kpt
                ),
            )
        )
    return samples


def evaluate_samples(
    samples: Sequence[ImageSample],
    config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Thin wrapper around :func:`cpsa_ml.eval.evaluate` (single metric source)."""

    return evaluate(samples, config)


def evaluate_int8_subset(
    dump: QuantDump,
    indices: Sequence[int],
    labels_dir: Path,
    config: EvalConfig,
    conf_thr: float,
    iou_thr: float,
) -> tuple[dict[str, Any], float]:
    """Decode and score the selected frames of one dump.

    Returns:
        ``(report, decode_seconds)``.
    """

    subset = dump.subset(indices)
    decoder = int8_decoder(subset)
    started = time.perf_counter()
    frames = decode_int8(subset, decoder, conf_thr, iou_thr)
    seconds = time.perf_counter() - started
    samples = build_samples(
        frames,
        subset.names,
        labels_dir,
        subset.image_widths,
        subset.image_heights,
        subset.n_kpt,
    )
    return evaluate_samples(samples, config), seconds


def evaluate_fp32_subset(
    weights: Path,
    dump: QuantDump,
    indices: Sequence[int],
    images_dir: Path,
    labels_dir: Path,
    config: EvalConfig,
    deploy_module_path: Path = DEFAULT_DEPLOY_MODULE,
    conf_thr: float = 0.25,
    iou_thr: float = 0.7,
    progress_every: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the float deploy graph and score the selected frames."""

    subset = dump.subset(indices)
    images = resolve_test_images(subset, images_dir)
    frames, timing = predict_fp32(
        weights,
        subset,
        images,
        deploy_module_path=deploy_module_path,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        progress_every=progress_every,
    )
    samples = build_samples(
        frames,
        subset.names,
        labels_dir,
        subset.image_widths,
        subset.image_heights,
        subset.n_kpt,
    )
    return evaluate_samples(samples, config), timing


# ---------------------------------------------------------------------------
# INT8 path (numpy only)
# ---------------------------------------------------------------------------


def int8_decoder(dump: QuantDump) -> QuantizedRawHeadDecoder:
    """Runtime fast-path decoder configured for the dump's shapes and fix points."""

    return QuantizedRawHeadDecoder(
        shapes=dump.shapes,
        strides=dump.strides,
        fix_points=dump.fix_points,
        n_kpt=dump.n_kpt,
        reg_max=dump.reg_max,
    )


def decode_int8(
    dump: QuantDump,
    decoder: QuantizedRawHeadDecoder | None = None,
    conf_thr: float = 0.25,
    iou_thr: float = 0.7,
) -> list[list[PersonPose]]:
    """Decode every dumped frame with the int8 fast path and unletterbox it."""

    active = decoder if decoder is not None else int8_decoder(dump)
    frames: list[list[PersonPose]] = []
    for index in range(len(dump)):
        heads = [head[index] for head in dump.heads]
        persons = active.decode(heads, conf_thr, iou_thr)
        frames.append(
            unletterbox(
                persons,
                float(dump.scales[index]),
                (float(dump.pad_x[index]), float(dump.pad_y[index])),
            )
        )
    return frames


# ---------------------------------------------------------------------------
# FP32 path (torch, host)
# ---------------------------------------------------------------------------


def load_deploy_module(path: Path) -> ModuleType:
    """Load ``cpsa_ml/deploy.py`` by path (it also runs under other Python versions)."""

    spec = importlib.util.spec_from_file_location("cpsa_ml_deploy", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load deploy module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_test_images(dump: QuantDump, images_dir: Path) -> list[Path]:
    """Map the dump's image names back to files under ``images_dir``."""

    directory = Path(images_dir)
    if not directory.is_dir():
        raise SystemExit(f"test image directory not found: {directory}")
    by_stem = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing = [name for name in dump.names if name not in by_stem]
    if missing:
        raise SystemExit(
            f"{directory} has no image for {len(missing)} dumped name(s): {', '.join(missing[:5])}"
        )
    return [by_stem[name] for name in dump.names]


def predict_fp32(
    weights: Path,
    dump: QuantDump,
    images: Sequence[Path],
    deploy_module_path: Path = DEFAULT_DEPLOY_MODULE,
    conf_thr: float = 0.25,
    iou_thr: float = 0.7,
    progress_every: int = 0,
) -> tuple[list[list[PersonPose]], dict[str, Any]]:
    """Run the float deploy graph and decode its heads like the runtime does."""

    import dump_quant_outputs as dq
    import torch

    module = load_deploy_module(Path(deploy_module_path))
    deploy, checkpoint_info = dq.load_deploy_from_weights(module, Path(weights), dump.input_size)
    if tuple(int(value) for value in deploy.spec()["strides"]) != dump.strides:
        raise SystemExit(
            f"deploy strides {deploy.spec()['strides']} do not match the dump's {dump.strides}"
        )

    frames: list[list[PersonPose]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, image_path in enumerate(images):
            image = dq.load_image(Path(image_path))
            canvas, scale, (pad_x, pad_y) = letterbox(image, dump.input_size)
            tensor = torch.from_numpy(dq.to_model_tensor(canvas)[None, ...])
            channels = sum(dump.channel_layout.values())
            heads = [dq.to_nhwc(output.numpy(), channels) for output in deploy(tensor)]
            shapes = tuple(tuple(int(size) for size in head.shape) for head in heads)
            if shapes != dump.shapes:
                raise SystemExit(
                    f"deploy heads {shapes} do not match the dumped heads {dump.shapes}"
                )
            persons = decode_raw_heads(
                heads, dump.strides, dump.n_kpt, conf_thr, iou_thr, dump.reg_max
            )
            frames.append(unletterbox(persons, scale, (pad_x, pad_y)))
            if progress_every and (index + 1) % progress_every == 0:
                print(f"[fp32] {index + 1}/{len(images)} frames", flush=True)

    seconds = time.perf_counter() - started
    timing = {
        "images": len(frames),
        "seconds": seconds,
        "milliseconds_per_image": 1000.0 * seconds / len(frames) if frames else None,
        "checkpoint": checkpoint_info,
    }
    return frames, timing


# ---------------------------------------------------------------------------
# Hand21 metrics (one hand, 21 keypoints)
# ---------------------------------------------------------------------------

#: A predicted keypoint counts as correct within this fraction of the hand box.
HAND21_PCK_THRESHOLD = 0.2
#: Box IoU at which a prediction is matched to a ground-truth hand.
HAND21_MATCH_IOU = 0.5
HAND21_N_KPT = len(HAND21.names)


def box_iou_matrix(
    boxes_a: NDArray[np.float64],
    boxes_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pairwise IoU of ``(N, 4)`` and ``(M, 4)`` ``x1, y1, x2, y2`` boxes."""

    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(right - left, 0.0, None) * np.clip(bottom - top, 0.0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0.0, intersection / np.where(union > 0.0, union, 1.0), 0.0)


def match_hands(
    targets: Sequence[PersonTarget],
    predictions: Sequence[PersonPrediction],
    iou_thr: float = HAND21_MATCH_IOU,
) -> list[tuple[int, int]]:
    """Greedy one-to-one ``(target, prediction)`` matches above ``iou_thr``.

    Candidate pairs are taken in descending IoU (ties keep the target then
    prediction order), so the matching is deterministic and a prediction is
    never scored against two ground-truth hands.
    """

    if not targets or not predictions:
        return []
    overlaps = box_iou_matrix(
        np.stack([target.box for target in targets]),
        np.stack([prediction.box for prediction in predictions]),
    )
    order = np.argsort(-overlaps, axis=None, kind="stable")
    used_targets: set[int] = set()
    used_predictions: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for flat_index in order:
        overlap = float(overlaps.flat[int(flat_index)])
        if overlap < iou_thr:
            break
        target_index, prediction_index = divmod(int(flat_index), len(predictions))
        if target_index in used_targets or prediction_index in used_predictions:
            continue
        used_targets.add(target_index)
        used_predictions.add(prediction_index)
        pairs.append((target_index, prediction_index))
    return pairs


def hand21_metrics(
    samples: Sequence[ImageSample],
    *,
    n_kpt: int = HAND21_N_KPT,
    threshold: float = HAND21_PCK_THRESHOLD,
    match_iou: float = HAND21_MATCH_IOU,
    kpt_conf_thr: float = 0.0,
) -> dict[str, Any]:
    """PCK@``threshold`` and detection recall for single-hand 21-keypoint dumps.

    PCK normalises the keypoint error by the largest side of the matched
    ground-truth hand box, which is the hand analogue of the half-bust
    ``PCK@0.2fw`` (there is no face to measure a hand against).  Only matched
    hands contribute keypoints; :func:`match_hands` links a prediction to at
    most one target at ``match_iou`` so the detection recall
    (``matched / ground-truth hands``) and the pose accuracy stay separate.
    """

    names = (
        tuple(HAND21.names)
        if n_kpt == len(HAND21.names)
        else tuple(f"kpt_{index}" for index in range(n_kpt))
    )
    visible = np.zeros(n_kpt, dtype=np.int64)
    correct = np.zeros(n_kpt, dtype=np.int64)
    ground_truth = 0
    predicted = 0
    matched = 0
    frames_with_detection = 0

    for sample in samples:
        ground_truth += len(sample.targets)
        predicted += len(sample.predictions)
        if sample.predictions:
            frames_with_detection += 1
        for target_index, prediction_index in match_hands(
            sample.targets, sample.predictions, match_iou
        ):
            target = sample.targets[target_index]
            prediction = sample.predictions[prediction_index]
            matched += 1
            box = np.asarray(target.box, dtype=np.float64)
            normaliser = float(max(box[2] - box[0], box[3] - box[1]))
            if not normaliser > 0.0:
                continue
            truth = target.keypoints[:n_kpt]
            guess = prediction.keypoints[:n_kpt]
            for index in range(n_kpt):
                if truth[index, 2] <= 0.0 or not np.isfinite(truth[index, :2]).all():
                    continue
                visible[index] += 1
                if guess[index, 2] < kpt_conf_thr or not np.isfinite(guess[index, :2]).all():
                    continue
                error = float(np.linalg.norm(guess[index, :2] - truth[index, :2]))
                if error <= threshold * normaliser:
                    correct[index] += 1

    per_keypoint = {
        name: {
            "pck": float(correct[index] / visible[index]) if visible[index] else None,
            "visible": int(visible[index]),
            "correct": int(correct[index]),
        }
        for index, name in enumerate(names)
    }
    total_visible = int(visible.sum())
    return {
        "schema": "hand21",
        "frames": len(samples),
        "frames_with_detection": frames_with_detection,
        "threshold": float(threshold),
        "normaliser": "gt_box_max_side",
        "match_iou": float(match_iou),
        "kpt_conf_thr": float(kpt_conf_thr),
        "pck": {
            "threshold": float(threshold),
            "mean": float(correct.sum() / total_visible) if total_visible else None,
            "visible": total_visible,
            "per_keypoint": per_keypoint,
        },
        "detection": {
            "gt": ground_truth,
            "predicted": predicted,
            "matched": matched,
            "recall": float(matched / ground_truth) if ground_truth else None,
        },
    }


def hand21_rows(
    int8: dict[str, Any],
    fp32_host: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Comparison rows: detection recall, mean PCK and one row per keypoint."""

    def value(report: dict[str, Any] | None, path: Sequence[str]) -> float | None:
        return None if report is None else metric_value(report, path)

    rows = [
        {
            "metric": f"detection.recall_iou{HAND21_MATCH_IOU}",
            "int8": value(int8, ("detection", "recall")),
            "fp32_host": value(fp32_host, ("detection", "recall")),
        },
        {
            "metric": f"pck@{HAND21_PCK_THRESHOLD:.1f}.mean",
            "int8": value(int8, ("pck", "mean")),
            "fp32_host": value(fp32_host, ("pck", "mean")),
        },
    ]
    for name in int8["pck"]["per_keypoint"]:
        rows.append(
            {
                "metric": f"pck@{HAND21_PCK_THRESHOLD:.1f}.{name}",
                "int8": value(int8, ("pck", "per_keypoint", name, "pck")),
                "fp32_host": value(fp32_host, ("pck", "per_keypoint", name, "pck")),
            }
        )
    for row in rows:
        row["delta_int8_fp32"] = delta(row["int8"], row["fp32_host"])
    return rows


def render_hand21_markdown(
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    frames: int,
    int8_label: str = "INT8",
    fp32_label: str = "FP32 host",
) -> str:
    """Markdown table for one hand21 split."""

    lines = [
        f"Hand21 `{split}`: {frames} frames, {int8_label} vs {fp32_label}",
        "",
        f"| metric | {int8_label} | {fp32_label} | Δ {int8_label}−{fp32_label} |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {_format_value(row['int8'])} | "
            f"{_format_value(row['fp32_host'])} | {_format_delta(row['delta_int8_fp32'])} |"
        )
    return "\n".join(lines) + "\n"


def group_detection_summary(
    dump: QuantDump,
    conf_thr: float,
    iou_thr: float,
) -> dict[str, dict[str, int]]:
    """Per dumped group: frames, frames with a detection, detections."""

    groups = dump.groups if any(dump.groups) else tuple("" for _ in range(len(dump)))
    summary: dict[str, dict[str, int]] = {}
    for group in sorted(set(groups)):
        indices = [index for index, name in enumerate(groups) if name == group]
        subset = dump.subset(indices)
        frames = decode_int8(subset, int8_decoder(subset), conf_thr, iou_thr)
        summary[group or "all"] = {
            "frames": len(indices),
            "frames_with_detection": sum(1 for persons in frames if persons),
            "detections": sum(len(persons) for persons in frames),
        }
    return summary


def run_hand21_evaluation(args: argparse.Namespace, dump: QuantDump) -> int:
    """Score a hand21 dump: PCK@0.2 per keypoint, detection recall, INT8 vs FP32."""

    if args.eval_split:
        specs = parse_eval_splits(args.eval_split)
        selections = {spec.name: split_indices(dump, spec) for spec in specs}
    else:
        images_dir, labels_dir = resolve_dirs(args)
        name = args.test_split or "test"
        spec = EvalSplit(name=name, images=images_dir, labels=labels_dir)
        specs = [spec]
        selections = {
            name: [
                index
                for index, frame_name in enumerate(dump.names)
                if (labels_dir / f"{frame_name}.txt").is_file()
            ]
        }
    for spec in specs:
        if not selections[spec.name]:
            raise SystemExit(
                f"no dumped frame matches hand21 split {spec.name!r}; check --labels/--eval-split"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    score_fp32 = args.weights is not None and not args.skip_fp32
    results: dict[str, Any] = {}
    tables: list[str] = []
    for spec in specs:
        selected = selections[spec.name]
        subset = dump.subset(selected)
        int8_metrics = hand21_metrics(
            build_samples(
                decode_int8(subset, int8_decoder(subset), args.conf_thr, args.iou_thr),
                subset.names,
                spec.labels,
                subset.image_widths,
                subset.image_heights,
                subset.n_kpt,
            ),
            n_kpt=dump.n_kpt,
            kpt_conf_thr=args.kpt_conf_thr,
        )
        entry: dict[str, Any] = {
            "frames": len(selected),
            "images": str(spec.images),
            "labels": str(spec.labels),
            "int8": int8_metrics,
        }
        if score_fp32:
            fp32_frames, timing = predict_fp32(
                args.weights,
                subset,
                resolve_test_images(subset, spec.images),
                deploy_module_path=args.deploy_module,
                conf_thr=args.conf_thr,
                iou_thr=args.iou_thr,
                progress_every=args.progress_every,
            )
            entry["fp32_host"] = hand21_metrics(
                build_samples(
                    fp32_frames,
                    subset.names,
                    spec.labels,
                    subset.image_widths,
                    subset.image_heights,
                    subset.n_kpt,
                ),
                n_kpt=dump.n_kpt,
                kpt_conf_thr=args.kpt_conf_thr,
            )
            entry["fp32_timing"] = timing
        rows = hand21_rows(int8_metrics, entry.get("fp32_host"))
        entry["rows"] = rows
        results[spec.name] = entry
        table = render_hand21_markdown(
            rows,
            split=spec.name,
            frames=len(selected),
            int8_label="INT8",
            fp32_label="FP32 host" if score_fp32 else "n/a",
        )
        tables.append(table)
        meta = dict(_dump_meta(args, dump, spec.images, spec.labels, frames=len(selected)))
        meta["split"] = spec.name
        _write_json(
            args.out / f"metrics_int8_hand21_{_split_slug(spec.name)}.json",
            {"meta": meta, "metrics": int8_metrics},
        )
        if "fp32_host" in entry:
            _write_json(
                args.out / f"metrics_fp32_host_hand21_{_split_slug(spec.name)}.json",
                {"meta": meta, "metrics": entry["fp32_host"]},
            )
        print(
            f"[hand21 {spec.name}] {len(selected)} frames: "
            f"detection recall {_format_value(int8_metrics['detection']['recall'])}, "
            f"PCK@{HAND21_PCK_THRESHOLD:.1f} {_format_value(int8_metrics['pck']['mean'])}",
            flush=True,
        )

    primary = specs[0].name
    groups = group_detection_summary(dump, args.conf_thr, args.iou_thr)
    summary = {
        "schema": "hand21",
        "variant": args.variant,
        "inputs": {
            "dump": str(dump.path),
            "weights": str(args.weights) if args.weights is not None else None,
            "splits": {
                spec.name: {"images": str(spec.images), "labels": str(spec.labels)}
                for spec in specs
            },
        },
        "primary_split": primary,
        "splits": results,
        "groups": groups,
    }
    _write_json(args.out / "comparison_hand21.json", summary)
    report = "\n".join(tables)
    report += (
        "\nDumped groups: "
        + ", ".join(
            f"{name} {entry['frames_with_detection']}/{entry['frames']} frames with a detection"
            for name, entry in groups.items()
        )
        + "\n"
    )
    (args.out / "comparison_hand21.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    return 0


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def report_of(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap ``{"meta": ..., "metrics": {...}}``; bare reports pass through."""

    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, dict) else payload


def metric_value(report: dict[str, Any], path: Sequence[str]) -> float | None:
    """Follow ``path`` into a report; return ``None`` for absent/non-finite values."""

    current: Any = report
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    value = float(current)
    return value if math.isfinite(value) else None


def delta(variant: float | None, reference: float | None) -> float | None:
    """``variant - reference`` where both are known, else ``None``."""

    if variant is None or reference is None:
        return None
    return variant - reference


def _metric_specs() -> list[tuple[str, tuple[str, ...]]]:
    specs: list[tuple[str, tuple[str, ...]]] = [
        ("detection.ap50", ("detection", "ap50")),
        ("pose.map50-95", ("pose", "map50_95")),
        ("pose.map50", ("pose", "map50")),
    ]
    for group, _ in HALFBUST62_GROUPS:
        specs.append(
            (f"{group}.pck_all@{DECISION_PCK}fw", ("groups", group, "pck_all", DECISION_PCK))
        )
    for group, _ in HALFBUST62_GROUPS:
        specs.append((f"{group}.pck@{DECISION_PCK}fw", ("groups", group, "pck", DECISION_PCK)))
        specs.append((f"{group}.missing_rate", ("groups", group, "missing_rate")))
    specs.extend(
        [
            ("hand_centroid.mean_fw", ("centroids", "hand", "all", "mean_fw")),
            ("face_centroid.mean_fw", ("centroids", "face", "mean_fw")),
            ("person_missing_rate", ("counts", "person_missing_rate")),
            ("framing.gt_full_body_fraction", ("framing", "gt_box", "full_body_fraction")),
            ("framing.gt_half_body_fraction", ("framing", "gt_box", "half_body_fraction")),
            (
                "framing.pred_full_body_fraction",
                ("framing", "predicted_box", "full_body_fraction"),
            ),
            (
                "framing.pred_half_body_fraction",
                ("framing", "predicted_box", "half_body_fraction"),
            ),
            (
                "framing.pred_unclassified_fraction",
                ("framing", "predicted_box", "unclassified_fraction"),
            ),
        ]
    )
    return specs


def comparison_rows(
    int8: dict[str, Any],
    fp32_host: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """One row per reported metric, with INT8 deltas against both references."""

    rows: list[dict[str, Any]] = []
    for label, path in _metric_specs():
        int8_value = metric_value(int8, path)
        fp32_value = metric_value(fp32_host, path) if fp32_host is not None else None
        baseline_value = metric_value(baseline, path) if baseline is not None else None
        rows.append(
            {
                "metric": label,
                "path": list(path),
                "int8": int8_value,
                "fp32_host": fp32_value,
                "baseline": baseline_value,
                "delta_int8_fp32": delta(int8_value, fp32_value),
                "delta_int8_baseline": delta(int8_value, baseline_value),
            }
        )
    return rows


def decision(
    int8: dict[str, Any],
    baseline: dict[str, Any] | None,
    groups: Sequence[str] = DEFAULT_DECISION_GROUPS,
    pck: str = DECISION_PCK,
    margin: float = DECISION_MARGIN,
) -> dict[str, Any]:
    """Evaluate the proposal rule: INT8 PCK must not drop by more than ``margin``.

    ``status`` is ``"pass"`` when every group is within the margin, ``"fail"``
    when at least one group is below it and ``"unknown"`` when a value (or the
    whole baseline) is missing, so absent evidence never reads as a pass.  A
    baseline PCK of zero counts as missing evidence: a variant cannot be
    "within two points" of an accuracy floor that does not exist yet.
    """

    details: dict[str, Any] = {}
    failed: list[str] = []
    unknown: list[str] = []
    for group in groups:
        path = ("groups", group, "pck_all", pck)
        int8_value = metric_value(int8, path)
        baseline_value = metric_value(baseline, path) if baseline is not None else None
        difference = delta(int8_value, baseline_value)
        if difference is None or baseline_value == 0.0:
            ok: bool | None = None
            unknown.append(group)
        else:
            ok = difference >= -margin
            if not ok:
                failed.append(group)
        details[group] = {
            "pck_all": pck,
            "int8": int8_value,
            "baseline": baseline_value,
            "delta": difference,
            "ok": ok,
        }
    if unknown:
        status = "unknown"
    elif failed:
        status = "fail"
    else:
        status = "pass"
    return {
        "groups": list(groups),
        "pck_all": pck,
        "margin": margin,
        "status": status,
        "passed": None if status == "unknown" else status == "pass",
        "failed_groups": failed,
        "unknown_groups": unknown,
        "details": details,
    }


def _format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def render_markdown(rows: Sequence[dict[str, Any]], labels: dict[str, str]) -> str:
    """Markdown table of the comparison rows (``labels`` name each reference)."""

    int8_label = labels.get("int8", "INT8")
    fp32_label = labels.get("fp32_host", "FP32 host")
    baseline_label = labels.get("baseline", "baseline")
    lines = [
        f"| metric | {int8_label} | {fp32_label} | {baseline_label} | "
        f"Δ {int8_label}−{fp32_label} | Δ {int8_label}−{baseline_label} |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {_format_value(row['int8'])} | "
            f"{_format_value(row['fp32_host'])} | {_format_value(row['baseline'])} | "
            f"{_format_delta(row['delta_int8_fp32'])} | "
            f"{_format_delta(row['delta_int8_baseline'])} |"
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class EvalSplit:
    """One named subset of a dump: ``NAME=IMAGES:LABELS``."""

    name: str
    images: Path
    labels: Path


def parse_eval_split(spec: str) -> EvalSplit:
    """Parse ``NAME=IMAGES:LABELS`` into an :class:`EvalSplit`.

    Raises:
        SystemExit: If the spec does not have exactly that shape.
    """

    text = str(spec)
    name, separator, rest = text.partition("=")
    images, separator2, labels = rest.partition(":")
    if not (separator and separator2 and name and images and labels):
        raise SystemExit(f"--eval-split must be NAME=IMAGES:LABELS, got {text!r}")
    return EvalSplit(name=name, images=Path(images), labels=Path(labels))


def parse_eval_splits(specs: Sequence[str]) -> list[EvalSplit]:
    """Parse repeatable ``--eval-split`` values and reject duplicate names."""

    splits = [parse_eval_split(spec) for spec in specs]
    names = [split.name for split in splits]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate --eval-split names: {names}")
    return splits


def split_indices(dump: QuantDump, split: EvalSplit) -> list[int]:
    """Dump indices that belong to ``split``.

    A dump written with grouped ``--test-images`` (``NAME=DIR``) carries a
    ``groups`` array, which is authoritative even when two splits contain the
    same image stem.  Older dumps fall back to "this split has a label file for
    the frame".
    """

    if any(dump.groups):
        return [index for index, group in enumerate(dump.groups) if group == split.name]
    return [
        index
        for index, name in enumerate(dump.names)
        if (Path(split.labels) / f"{name}.txt").is_file()
    ]


def _split_slug(name: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_" for character in name
    )


def render_split_markdown(
    rows_by_split: dict[str, Sequence[dict[str, Any]]],
    labels: dict[str, str],
    splits: Sequence[str],
) -> str:
    """Side-by-side table with one INT8 (and optional FP32) column per split."""

    if not splits:
        raise ValueError("splits must not be empty")
    first = rows_by_split[splits[0]]
    int8_label = labels.get("int8", "INT8")
    fp32_label = labels.get("fp32_host", "FP32 host")
    baseline_label = labels.get("baseline", "baseline")
    has_fp32 = any(row["fp32_host"] is not None for name in splits for row in rows_by_split[name])
    has_baseline = any(row["baseline"] is not None for row in first)

    header = ["metric"]
    if has_baseline:
        header.append(baseline_label)
    for name in splits:
        header.append(f"{name} {int8_label}")
        if has_fp32:
            header.append(f"{name} {fp32_label}")
        if has_baseline:
            header.append(f"Δ {name}−{baseline_label}")
    lines = [
        "| " + " | ".join(header) + " |",
        "| --- " + "| ---: " * (len(header) - 1) + "|",
    ]
    indexed = {name: {row["metric"]: row for row in rows_by_split[name]} for name in splits}
    for row in first:
        cells = [row["metric"]]
        if has_baseline:
            cells.append(_format_value(row["baseline"]))
        for name in splits:
            match = indexed[name][row["metric"]]
            cells.append(_format_value(match["int8"]))
            if has_fp32:
                cells.append(_format_value(match["fp32_host"]))
            if has_baseline:
                cells.append(_format_delta(match["delta_int8_baseline"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True, help="quant_outputs.npz")
    parser.add_argument(
        "--weights", type=Path, default=None, help="best.pt for the FP32 host reference"
    )
    parser.add_argument("--deploy-module", type=Path, default=DEFAULT_DEPLOY_MODULE)
    parser.add_argument("--data", type=Path, default=None, help="half-bust dataset root")
    parser.add_argument("--test-split", default="val")
    parser.add_argument("--test-images", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None, help="label directory")
    parser.add_argument(
        "--eval-split",
        action="append",
        default=[],
        metavar="NAME=IMAGES:LABELS",
        help=(
            "score a named subset of the dump separately (repeat for a side-by-side table); "
            "membership comes from the dump's groups written by --test-images NAME=DIR"
        ),
    )
    parser.add_argument("--baseline", type=Path, default=None, help="baseline metrics JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default=None, help="variant name for the reports")
    parser.add_argument("--conf-thr", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.7)
    parser.add_argument("--kpt-conf-thr", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N frames")
    parser.add_argument("--skip-fp32", action="store_true", help="skip the FP32 host reference")
    parser.add_argument("--progress-every", type=int, default=0)
    return parser.parse_args(argv)


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return ``(images_dir, labels_dir)`` for the dumped frames."""

    if args.labels is not None:
        labels = args.labels
    elif args.data is not None:
        labels = Path(args.data) / "labels" / args.test_split
    else:
        raise SystemExit("pass --labels or --data to locate the ground-truth labels")

    if args.test_images is not None:
        images = args.test_images
    elif args.data is not None:
        images = Path(args.data) / "images" / args.test_split
    else:
        raise SystemExit("pass --test-images or --data to locate the evaluated images")
    return Path(images), Path(labels)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _dump_meta(
    args: argparse.Namespace,
    dump: QuantDump,
    images_dir: Path | None,
    labels_dir: Path | None,
    frames: int | None = None,
) -> dict[str, Any]:
    """Common ``meta`` block recorded next to every metrics report."""

    return {
        "variant": args.variant,
        "dump": str(dump.path),
        "weights": str(args.weights) if args.weights is not None else None,
        "images": str(images_dir) if images_dir is not None else None,
        "labels": str(labels_dir) if labels_dir is not None else None,
        "frames": len(dump) if frames is None else int(frames),
        "input_size": list(dump.input_size),
        "strides": list(dump.strides),
        "fix_points": list(dump.fix_points),
        "n_kpt": dump.n_kpt,
        "reg_max": dump.reg_max,
        "num_classes": dump.num_classes,
        "channel_layout": dump.channel_layout,
        "conf_thr": args.conf_thr,
        "iou_thr": args.iou_thr,
    }


def _load_baseline(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    if path is None:
        return None, "n/a"
    baseline_path = Path(path)
    if not baseline_path.is_file():
        raise SystemExit(f"baseline metrics not found: {baseline_path}")
    report = report_of(json.loads(baseline_path.read_text(encoding="utf-8")))
    return report, baseline_path.parent.name or baseline_path.stem


def run_split_evaluation(args: argparse.Namespace, dump: QuantDump, config: EvalConfig) -> int:
    """Evaluate several named dump subsets and write a side-by-side report."""

    specs = parse_eval_splits(args.eval_split)
    indices: dict[str, list[int]] = {}
    covered: set[int] = set()
    for spec in specs:
        selected = split_indices(dump, spec)
        if not selected:
            raise SystemExit(
                f"no dumped frame matches --eval-split {spec.name!r}; dump with "
                "--test-images NAME=DIR using the same name"
            )
        indices[spec.name] = selected
        covered.update(selected)
    if len(covered) != len(dump):
        print(
            f"warning: {len(dump) - len(covered)} dumped frame(s) match no --eval-split",
            flush=True,
        )

    baseline_report, baseline_label = _load_baseline(args.baseline)
    score_fp32 = args.weights is not None and not args.skip_fp32
    if baseline_report is None and score_fp32:
        baseline_label = "fp32_host"

    results: dict[str, Any] = {}
    metas: dict[str, dict[str, Any]] = {}
    timings: dict[str, dict[str, Any]] = {}
    for spec in specs:
        selected = indices[spec.name]
        meta = _dump_meta(args, dump, spec.images, spec.labels, frames=len(selected))
        meta["split"] = spec.name
        int8_report, seconds = evaluate_int8_subset(
            dump, selected, spec.labels, config, args.conf_thr, args.iou_thr
        )
        entry: dict[str, Any] = {
            "frames": len(selected),
            "images": str(spec.images),
            "labels": str(spec.labels),
            "int8": int8_report,
        }
        per_split_timing: dict[str, Any] = {
            "int8_decode_seconds": seconds,
            "int8_decode_milliseconds_per_image": 1000.0 * seconds / len(selected),
        }
        print(
            f"[{spec.name}] {len(selected)} frames: AP50 {int8_report['detection']['ap50']:.4f}, "
            f"pose mAP50-95 {int8_report['pose']['map50_95']:.4f}",
            flush=True,
        )
        if score_fp32:
            fp32_report, fp32_timing = evaluate_fp32_subset(
                args.weights,
                dump,
                selected,
                spec.images,
                spec.labels,
                config,
                deploy_module_path=args.deploy_module,
                conf_thr=args.conf_thr,
                iou_thr=args.iou_thr,
                progress_every=args.progress_every,
            )
            entry["fp32_host"] = fp32_report
            per_split_timing["fp32_seconds"] = fp32_timing["seconds"]
            per_split_timing["fp32_milliseconds_per_image"] = fp32_timing["milliseconds_per_image"]
            meta["fp32_checkpoint"] = fp32_timing["checkpoint"]
            _write_json(
                args.out / f"metrics_fp32_host_{_split_slug(spec.name)}.json",
                {"meta": dict(meta), "metrics": fp32_report},
            )
            print(
                f"[{spec.name}] FP32 host: AP50 {fp32_report['detection']['ap50']:.4f}, "
                f"pose mAP50-95 {fp32_report['pose']['map50_95']:.4f}",
                flush=True,
            )
        meta["timings"] = dict(per_split_timing)
        _write_json(
            args.out / f"metrics_int8_{_split_slug(spec.name)}.json",
            {"meta": dict(meta), "metrics": int8_report},
        )
        results[spec.name] = entry
        metas[spec.name] = meta
        timings[spec.name] = per_split_timing

    names = [spec.name for spec in specs]
    primary = names[0]
    _write_json(
        args.out / "metrics_int8.json",
        {"meta": dict(metas[primary]), "metrics": results[primary]["int8"]},
    )
    if "fp32_host" in results[primary]:
        _write_json(
            args.out / "metrics_fp32_host.json",
            {"meta": dict(metas[primary]), "metrics": results[primary]["fp32_host"]},
        )

    labels = {
        "int8": "INT8",
        "fp32_host": "FP32 host" if score_fp32 else "n/a",
        "baseline": baseline_label,
    }
    rows_by_split = {
        name: comparison_rows(
            results[name]["int8"],
            results[name].get("fp32_host"),
            baseline_report if baseline_report is not None else results[name].get("fp32_host"),
        )
        for name in names
    }
    decision_baseline = (
        baseline_report if baseline_report is not None else results[primary].get("fp32_host")
    )
    verdict = decision(results[primary]["int8"], decision_baseline)
    summary = {
        "variant": args.variant,
        "inputs": {
            "dump": str(dump.path),
            "weights": str(args.weights) if args.weights is not None else None,
            "baseline": str(args.baseline) if args.baseline is not None else None,
            "splits": {
                name: {"images": results[name]["images"], "labels": results[name]["labels"]}
                for name in names
            },
        },
        "splits": results,
        "timings": timings,
        "decision": verdict,
        "rows": rows_by_split,
    }
    _write_json(args.out / "comparison_splits.json", summary)

    table = render_split_markdown(rows_by_split, labels, names)
    (args.out / "comparison_splits.md").write_text(table, encoding="utf-8")
    print(table, flush=True)

    primary_rows = rows_by_split[primary]
    _write_json(
        args.out / "comparison.json",
        {
            "variant": args.variant,
            "split": primary,
            "splits": names,
            "inputs": summary["inputs"],
            "frames": results[primary]["frames"],
            "timings": timings[primary],
            "decision": verdict,
            "rows": primary_rows,
        },
    )
    (args.out / "comparison.md").write_text(render_markdown(primary_rows, labels), encoding="utf-8")
    print(
        f"decision ({primary}): {verdict['status']} (groups {', '.join(verdict['groups'])}, "
        f"PCK@{verdict['pck_all']}fw margin {verdict['margin']:.2f})",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    dump = load_dump(args.dump, limit=args.limit)
    if dump.n_kpt == HAND21_N_KPT:
        return run_hand21_evaluation(args, dump)
    if dump.n_kpt != EvalConfig().n_kpt:
        raise SystemExit(
            f"the dump has {dump.n_kpt} keypoints; the half-bust metrics need {EvalConfig().n_kpt}"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    config = EvalConfig(kpt_conf_thr=args.kpt_conf_thr)
    if args.eval_split:
        return run_split_evaluation(args, dump, config)

    images_dir, labels_dir = resolve_dirs(args)
    meta = _dump_meta(args, dump, images_dir, labels_dir)

    # -- INT8 --------------------------------------------------------------
    started = time.perf_counter()
    int8_frames = decode_int8(dump, int8_decoder(dump), args.conf_thr, args.iou_thr)
    int8_seconds = time.perf_counter() - started
    int8_samples = build_samples(
        int8_frames, dump.names, labels_dir, dump.image_widths, dump.image_heights, dump.n_kpt
    )
    int8_report = evaluate_samples(int8_samples, config)
    meta["timings"] = {
        "int8_decode_seconds": int8_seconds,
        "int8_decode_milliseconds_per_image": 1000.0 * int8_seconds / len(dump),
    }

    _write_json(args.out / "metrics_int8.json", {"meta": dict(meta), "metrics": int8_report})
    print(
        f"[int8] {len(dump)} frames: AP50 {int8_report['detection']['ap50']:.4f}, "
        f"pose mAP50-95 {int8_report['pose']['map50_95']:.4f}",
        flush=True,
    )

    # -- FP32 host ---------------------------------------------------------
    fp32_report: dict[str, Any] | None = None
    if args.skip_fp32:
        if args.baseline is None:
            raise SystemExit("--skip-fp32 needs --baseline for the comparison table")
        print("[fp32] skipped (--skip-fp32)", flush=True)
    elif args.weights is None:
        if args.baseline is None:
            raise SystemExit("pass --weights (FP32 reference) or --baseline")
        print("[fp32] skipped (no --weights)", flush=True)
    else:
        images = resolve_test_images(dump, images_dir)
        fp32_frames, fp32_timing = predict_fp32(
            args.weights,
            dump,
            images,
            deploy_module_path=args.deploy_module,
            conf_thr=args.conf_thr,
            iou_thr=args.iou_thr,
            progress_every=args.progress_every,
        )
        fp32_samples = build_samples(
            fp32_frames, dump.names, labels_dir, dump.image_widths, dump.image_heights, dump.n_kpt
        )
        fp32_report = evaluate_samples(fp32_samples, config)
        meta["timings"]["fp32_seconds"] = fp32_timing["seconds"]
        meta["timings"]["fp32_milliseconds_per_image"] = fp32_timing["milliseconds_per_image"]
        meta["fp32_checkpoint"] = fp32_timing["checkpoint"]
        _write_json(
            args.out / "metrics_fp32_host.json", {"meta": dict(meta), "metrics": fp32_report}
        )
        print(
            f"[fp32] {len(dump)} frames: AP50 {fp32_report['detection']['ap50']:.4f}, "
            f"pose mAP50-95 {fp32_report['pose']['map50_95']:.4f}",
            flush=True,
        )

    # -- comparison --------------------------------------------------------
    baseline_report, baseline_label = _load_baseline(args.baseline)
    if baseline_report is None and fp32_report is not None:
        baseline_report = fp32_report
        baseline_label = "fp32_host"

    rows = comparison_rows(int8_report, fp32_report, baseline_report)
    labels = {
        "int8": "INT8",
        "fp32_host": "FP32 host" if fp32_report is not None else "n/a",
        "baseline": baseline_label,
    }
    verdict = decision(int8_report, baseline_report)
    summary = {
        "variant": args.variant,
        "inputs": {
            "dump": str(dump.path),
            "weights": str(args.weights) if args.weights is not None else None,
            "images": str(images_dir),
            "labels": str(labels_dir),
            "baseline": str(args.baseline) if args.baseline is not None else "fp32_host",
        },
        "frames": len(dump),
        "timings": meta["timings"],
        "decision": verdict,
        "rows": rows,
    }
    _write_json(args.out / "comparison.json", summary)
    table = render_markdown(rows, labels)
    (args.out / "comparison.md").write_text(table, encoding="utf-8")
    print(table, flush=True)
    print(
        f"decision: {verdict['status']} (groups {', '.join(verdict['groups'])}, "
        f"PCK@{verdict['pck_all']}fw margin {verdict['margin']:.2f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
