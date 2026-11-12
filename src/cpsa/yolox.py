"""YOLOX-nano on the DPU: preprocessing, decoding, NMS and COCO labels (F-S1).

The Vitis AI model-zoo ``yolox_nano`` graph is COCO-80 and stops before
post-processing. The zoo preprocessing is raw-pixel BGR (``mean 0, scale 1``):
bilinear letterbox to ``416x416`` with pad value ``114``. The compiled graph
takes ``[1, 416, 416, 3]`` xint8 NHWC with ``fix_point -2``
(``round(pixel * 2 ** -2)``), reproduced by a 256-entry LUT, and returns three
raw heads ``(1, 52, 52, 85)``, ``(1, 26, 26, 85)``, ``(1, 13, 13, 85)`` xint8
with ``fix_point 3``. The per-cell channels are
``[tx, ty, tw, th, objectness, 80 class logits]``; upstream ``YOLOXHead``
applies ``sigmoid`` to channels ``4:`` and then the grid/stride decode, so the
decoder runs with ``logits=True``.

:class:`YoloxDetector` splits one frame into ``submit`` (preprocess and start
the DPU) and ``wait`` (block, decode, map to source pixels), which is what lets
the pipeline overlap the DPU with the CPU: two buffer sets are in flight at
most, so frame N is on the DPU while frame N-1 is decoded and drawn.

The decode is multi-label: a grid cell is not one class, it yields one candidate
per class whose ``objectness * class`` score passes that class's threshold
(at most :data:`MAX_CLASSES_PER_CELL` per cell), then the class-aware NMS
removes the duplicates. A bottle held in front of a person therefore stays
detectable in the person's own cells instead of losing them to the ``person``
class, and :func:`decode_yolox` accepts the per-class thresholds the showcase
resolves from ``model.class_conf`` and ``model.object_conf``.

``model.zoom`` centre-crops the frame before the letterbox: the crop is a view,
so the same 416x416 input sees the objects ``zoom`` times larger with no extra
DPU work or frame copy, and :func:`zoom_crop` plus the ``origin`` of
:func:`boxes_to_source` map the decoded boxes back to full-frame pixels.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from cpsa.dpu import (
    DpuRunner,
    OutputTensor,
    dequantize_fix_point,
    input_quantization_lut_scaled,
)

__all__ = [
    "COCO80_EN",
    "COCO80_IT",
    "Detection",
    "MAX_CLASSES_PER_CELL",
    "MAX_DETECTIONS",
    "OBJECT_CLASSES",
    "PERSON_CLASS",
    "PERSON_CLASSES",
    "Pass",
    "QuantizedYoloxDecoder",
    "YOLOX_INPUT_SCALE",
    "YOLOX_INPUT_SIZE",
    "YOLOX_PAD_VALUE",
    "YOLOX_STRIDES",
    "YoloxDetector",
    "ZoomCrop",
    "boxes_to_source",
    "class_thresholds",
    "decode_yolox",
    "detections_from_heads",
    "label_for",
    "labels_for",
    "prepare_input",
    "prepare_input_into",
    "zoom_crop",
]

#: Model input ``(height, width)`` of the zoo YOLOX-nano graph.
YOLOX_INPUT_SIZE = (416, 416)

#: Letterbox fill value of the zoo preprocessing (grey, as in upstream YOLOX).
YOLOX_PAD_VALUE = 114

#: Network strides of the three YOLOX heads, smallest feature map last.
YOLOX_STRIDES = (8, 16, 32)

#: ``max(0, 1)`` normalisation of the zoo prototxt: raw 0..255 pixels.
YOLOX_INPUT_SCALE = 1.0

#: Boxes kept after non-maximum suppression, as in the zoo post-processing.
MAX_DETECTIONS = 100

#: Candidates one grid cell may yield: a bottle in front of a person shares the
#: person's cells, and one class per cell would let ``person`` swallow the bottle.
MAX_CLASSES_PER_CELL = 3

_NMS_MATRIX_CANDIDATES = 512

_NMS_CANDIDATE_CAP = 300
"""Candidates with the best scores kept before NMS.

A real frame yields a few dozen candidates, so the cap only trips on a
pathological head (a busy texture read as hundreds of objects); it bounds the
pairwise pass there without changing the small frames the showcase sees.
"""
"""Candidate count up to which the NMS computes its overlaps as one matrix."""

_MIN_CHANNELS = 6

#: Channel layout of a head cell: ``[tx, ty, tw, th, objectness, classes...]``.
_OBJECTNESS = 4
_CLASS_START = 5
_BOX_CHANNELS = 4


@dataclass(frozen=True)
class Detection:
    """One detected object in source-image pixels.

    Attributes:
        box: ``(x1, y1, x2, y2)`` in source pixels, clipped to the image.
        score: Objectness times the probability of the class it was decoded as.
        class_id: COCO-80 index, where ``0`` is ``person``.
    """

    box: tuple[float, float, float, float]
    score: float
    class_id: int


def labels_for(language: str) -> tuple[str, ...]:
    """Return the COCO-80 label table of ``language`` (``it`` or ``en``).

    Raises:
        ValueError: If ``language`` is neither ``it`` nor ``en``.
    """

    if language == "it":
        return COCO80_IT
    if language == "en":
        return COCO80_EN
    raise ValueError(f"unknown label language {language!r}, expected 'it' or 'en'")


def label_for(class_id: int, labels: Sequence[str]) -> str:
    """Return the label of a COCO class; an unknown index keeps its number."""
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class {class_id}"


def class_thresholds(
    conf_thr: float,
    *,
    class_conf: Mapping[str, float] | None = None,
    object_conf: float | None = None,
    labels: Sequence[str] | None = None,
) -> dict[int, float]:
    """Resolve the configuration's class thresholds into COCO-80 class ids.

    ``class_conf`` names classes by their canonical COCO-80 English name
    (``{"person": 0.40}``); every class it omits takes ``object_conf``, and
    ``conf_thr`` is the fallback of both. The showcase keeps ``person`` strict
    and the objects held in front of it loose, which is what lets a bottle
    inside the person's cells survive the decode.

    Args:
        conf_thr: Class-independent default threshold.
        class_conf: Thresholds by COCO-80 English name.
        object_conf: Threshold of every class ``class_conf`` does not name;
            ``None`` keeps ``conf_thr``.
        labels: Label table that maps a name to its class id; defaults to the
            English COCO names.

    Returns:
        ``{class_id: threshold}`` of every class of ``labels``.

    Raises:
        ValueError: If a name is not in ``labels`` or a threshold is outside
            ``[0, 1]``.
    """

    if not 0.0 <= conf_thr <= 1.0:
        raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
    default = conf_thr if object_conf is None else float(object_conf)
    if not 0.0 <= default <= 1.0:
        raise ValueError(f"object_conf must be in [0, 1], got {default}")
    table = COCO80_EN if labels is None else labels
    thresholds = {class_id: default for class_id in range(len(table))}
    for name, raw_threshold in (class_conf or {}).items():
        if name not in table:
            raise ValueError(f"unknown COCO class {name!r} in class_conf")
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"class_conf[{name!r}] must be in [0, 1], got {threshold}")
        thresholds[table.index(name)] = threshold
    return thresholds


# --------------------------------------------------------------------------- #
# Head decode
# --------------------------------------------------------------------------- #


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logistic function without spurious overflow warnings."""

    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-values))


def _iou(box: NDArray[np.float64], boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Intersection over union of one ``xyxy`` box against many."""

    width = np.maximum(0.0, np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]))
    height = np.maximum(0.0, np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]))
    intersection = width * height
    union = (
        (box[2] - box[0]) * (box[3] - box[1])
        + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        - intersection
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0.0, intersection / union, 0.0)


def _pairwise_iou(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Intersection over union of every ``xyxy`` box against every box."""

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    width = np.clip(
        np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :]), 0.0, None
    )
    height = np.clip(
        np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :]), 0.0, None
    )
    intersection = width * height
    area = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    union = area[:, None] + area[None, :] - intersection
    overlap = np.zeros(intersection.shape, dtype=np.float64)
    np.divide(intersection, union, out=overlap, where=union > 0.0)
    return overlap


def _class_aware_nms(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    class_ids: NDArray[np.intp],
    iou_thr: float,
) -> NDArray[np.intp]:
    """Greedy per-class suppression: overlapping boxes of different classes both survive.

    Suppression never crosses classes, so each class is passed on its own: the
    survivor set is exactly the one of the class-aware pass over the frame (the
    score order inside a class is unchanged) and the pairwise overlap stays
    ``sum(n_c ** 2)`` instead of ``n ** 2``. A class cannot contribute more than
    :data:`MAX_DETECTIONS` survivors to the capped output, so its pass stops
    there. Within one matrix-sized class the overlaps are computed once and the
    pass only flips booleans; beyond :data:`_NMS_MATRIX_CANDIDATES` a per-row
    scan is used instead.
    """

    if scores.size <= 1:
        return np.argsort(-scores, kind="stable").astype(np.intp)
    keep: list[int] = []
    for class_id in np.unique(class_ids):
        members = np.flatnonzero(class_ids == class_id)
        order = members[np.argsort(-scores[members], kind="stable")]
        remaining = np.ones(order.size, dtype=np.bool_)
        kept = 0
        if order.size <= _NMS_MATRIX_CANDIDATES:
            overlap = _pairwise_iou(boxes[order])
            for position in range(order.size):
                if not remaining[position]:
                    continue
                keep.append(int(order[position]))
                kept += 1
                remaining[position] = False
                remaining[overlap[position] > iou_thr] = False
                if kept >= MAX_DETECTIONS:
                    break
            continue
        for position in range(order.size):
            if not remaining[position]:
                continue
            keep.append(int(order[position]))
            kept += 1
            remaining[position] = False
            rest = np.flatnonzero(remaining)
            overlap = _iou(boxes[order[position]], boxes[order[rest]])
            remaining[rest[overlap > iou_thr]] = False
            if kept >= MAX_DETECTIONS or rest.size == 0:
                break
    return np.asarray(keep, dtype=np.intp)


def _finalize(
    boxes: list[NDArray[np.float64]],
    scores: list[NDArray[np.float64]],
    classes: list[NDArray[np.intp]],
    iou_thr: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Concatenate the per-head candidates, suppress duplicates and cap the count."""

    if not boxes:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.intp),
        )
    candidates = np.concatenate(boxes, axis=0)
    candidate_scores = np.concatenate(scores, axis=0)
    candidate_classes = np.concatenate(classes, axis=0)
    if candidate_scores.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.intp),
        )
    if candidate_scores.size > _NMS_CANDIDATE_CAP:
        best = np.argsort(-candidate_scores, kind="stable")[:_NMS_CANDIDATE_CAP]
        candidates = candidates[best]
        candidate_scores = candidate_scores[best]
        candidate_classes = candidate_classes[best]

    keep = _class_aware_nms(candidates, candidate_scores, candidate_classes, iou_thr)
    keep = keep[np.argsort(-candidate_scores[keep], kind="stable")][:MAX_DETECTIONS]
    return candidates[keep], candidate_scores[keep], candidate_classes[keep]


def _class_thresholds(
    conf_thr: float,
    class_thr: Mapping[int, float] | None,
    classes: int,
) -> NDArray[np.float64]:
    """Return the ``objectness * class`` threshold of every class of one head.

    ``conf_thr`` is the class-independent default; ``class_thr`` overrides single
    COCO classes, which is how the showcase gives ``person`` a stricter threshold
    than the objects in front of it.
    """

    thresholds = np.full(classes, conf_thr, dtype=np.float64)
    if not class_thr:
        return thresholds
    for raw_class_id, raw_threshold in class_thr.items():
        class_id = int(raw_class_id)
        if not 0 <= class_id < classes:
            raise ValueError(
                f"class threshold for class {class_id}, but the head has {classes} classes"
            )
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"class threshold for class {class_id} must be in [0, 1], got {threshold}"
            )
        thresholds[class_id] = threshold
    return thresholds


def _cell_candidates(
    scores: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    usable: NDArray[np.bool_],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Select the ``(cell, class)`` candidates of one head's score matrix.

    ``scores`` is ``(cells, classes)`` objectness times class probability,
    ``thresholds`` the per-class minimum and ``usable`` the cells whose box is
    valid. A cell yields one candidate per class above its own threshold, capped
    to the :data:`MAX_CLASSES_PER_CELL` best classes, so a bottle that overlaps a
    person is never lost to the winning ``person`` class. Candidates come back in
    cell order, best class first within a cell, which the flat reference path and
    the int8 path share so their output order is identical.
    """

    qualified = (scores > thresholds) & usable[:, None]
    counts = qualified.sum(axis=1)
    rows = np.flatnonzero(counts)
    if rows.size == 0:
        return (
            np.zeros(0, dtype=np.intp),
            np.zeros(0, dtype=np.intp),
            np.zeros(0, dtype=np.float64),
        )
    cell_qualified = qualified[rows]
    if np.max(counts[rows]) <= 1:
        # Most cells of a real frame hold a single class above its threshold.
        cell_index, class_ids = np.nonzero(cell_qualified)
        cells = rows[cell_index]
        return cells, class_ids, scores[cells, class_ids]
    cell_scores = np.where(cell_qualified, scores[rows], -np.inf)
    # Stable sort: equal scores keep the lower class id, on both paths.
    class_ids = np.argsort(-cell_scores, axis=1, kind="stable")[:, :MAX_CLASSES_PER_CELL]
    row_index = np.arange(rows.size, dtype=np.intp)[:, None]
    kept = cell_qualified[row_index, class_ids].ravel()
    return (
        np.repeat(rows, kept.reshape(rows.size, -1).sum(axis=1)),
        class_ids.ravel()[kept],
        cell_scores[row_index, class_ids].ravel()[kept],
    )


def _decode_head(
    head: NDArray[np.float64],
    stride: int,
    *,
    logits: bool,
    thresholds: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Decode one stride's ``(H, W, 85)`` head into candidate ``xyxy`` boxes."""

    regression = head[..., :_BOX_CHANNELS]
    objectness = head[..., _OBJECTNESS]
    class_scores = head[..., _CLASS_START:]
    if logits:
        objectness = _sigmoid(objectness)
        class_scores = _sigmoid(class_scores)
    scores = objectness[..., None] * class_scores

    height, width = head.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    center_x = (regression[..., 0] + grid_x) * stride
    center_y = (regression[..., 1] + grid_y) * stride
    with np.errstate(over="ignore"):
        box_width = np.exp(regression[..., 2]) * stride
        box_height = np.exp(regression[..., 3]) * stride
    boxes = np.stack(
        (
            center_x - box_width / 2.0,
            center_y - box_height / 2.0,
            center_x + box_width / 2.0,
            center_y + box_height / 2.0,
        ),
        axis=-1,
    )

    usable = (
        np.isfinite(boxes).all(axis=-1)
        & (boxes[..., 2] > boxes[..., 0])
        & (boxes[..., 3] > boxes[..., 1])
    )
    cells, class_ids, kept_scores = _cell_candidates(
        scores.reshape(height * width, -1), thresholds, usable.reshape(-1)
    )
    return boxes.reshape(-1, _BOX_CHANNELS)[cells], kept_scores, class_ids


def decode_yolox(
    heads: Sequence[NDArray[np.floating[Any]]],
    strides: Sequence[int],
    *,
    conf_thr: float,
    iou_thr: float,
    logits: bool,
    class_thr: Mapping[int, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Decode YOLOX heads into boxes in model-input pixels.

    A grid cell is decoded in multi-label mode: every class whose
    ``objectness * class`` score passes its own threshold yields a candidate (at
    most :data:`MAX_CLASSES_PER_CELL` per cell), so a bottle seen inside a
    person's cells survives alongside the person it overlaps.

    Args:
        heads: One ``(H_i, W_i, 5 + n_classes)`` tensor per stride, in
            ``strides`` order.
        strides: Network stride of each head (smallest first, e.g. 8, 16, 32).
        conf_thr: Default minimum ``objectness * class`` score in ``[0, 1]``.
        iou_thr: NMS IoU threshold in ``[0, 1]``; suppression is class aware.
        logits: ``True`` when the head channels hold raw logits (the zoo
            graphs), ``False`` when they were already activated.
        class_thr: Minimum ``objectness * class`` score of single COCO classes,
            overriding ``conf_thr``; the classes it omits keep ``conf_thr``.

    Returns:
        ``(boxes, scores, class_ids)`` where ``boxes`` is ``(N, 4)`` ``xyxy`` in
        input pixels, ``scores`` is ``(N,)`` and ``class_ids`` is ``(N,)``
        ``intp``; at most :data:`MAX_DETECTIONS` survive, score-descending.

    Raises:
        ValueError: If the head/stride counts differ, a head is not
            ``(H, W, C >= 6)``, the channel counts differ, a stride is not
            positive, a threshold is outside ``[0, 1]``, or ``class_thr`` names
            a class the heads do not have.
    """

    if not 0.0 <= conf_thr <= 1.0:
        raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
    if not 0.0 <= iou_thr <= 1.0:
        raise ValueError(f"iou_thr must be in [0, 1], got {iou_thr}")
    if len(heads) != len(strides):
        raise ValueError(f"got {len(heads)} heads but {len(strides)} strides")
    if not heads:
        raise ValueError("at least one head is required")

    channels = None
    boxes: list[NDArray[np.float64]] = []
    scores: list[NDArray[np.float64]] = []
    classes: list[NDArray[np.intp]] = []
    for index, (head, stride) in enumerate(zip(heads, strides, strict=True)):
        array = np.asarray(head, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(f"head {index} must be (H, W, C), got shape {array.shape}")
        if array.shape[2] < _MIN_CHANNELS:
            raise ValueError(
                f"head {index} needs at least {_MIN_CHANNELS} channels, got {array.shape[2]}"
            )
        if channels is None:
            channels = array.shape[2]
        elif array.shape[2] != channels:
            raise ValueError(f"head {index} has {array.shape[2]} channels, expected {channels}")
        if stride <= 0:
            raise ValueError(f"stride {index} must be positive, got {stride}")
        thresholds = _class_thresholds(conf_thr, class_thr, array.shape[2] - _CLASS_START)
        decoded = _decode_head(array, int(stride), logits=logits, thresholds=thresholds)
        boxes.append(decoded[0])
        scores.append(decoded[1])
        classes.append(decoded[2])

    return _finalize(boxes, scores, classes, iou_thr)


def _threshold_ints(thresholds: NDArray[np.float64], fix_point: int) -> NDArray[np.int8]:
    """Return the smallest int8 logit each threshold can accept, one per class.

    With ``q`` the raw logit, the reference path scores a candidate with
    ``sigmoid(q * 2 ** -fix_point)`` and keeps ``score > threshold``. Since
    sigmoid is monotonic, that implies ``q > logit(threshold) * 2 ** fix_point``,
    whose integer equivalent is ``floor(threshold) + 1``. One less is returned,
    clipped to the int8 range, so float rounding at a quantization step can never
    drop a candidate that the reference path keeps; the exact float comparison
    removes the extra one and a threshold of ``1.0`` keeps nothing.
    """

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logits = np.log(thresholds) - np.log1p(-thresholds)
        scaled = logits * 2.0**fix_point
        gates = np.where(
            np.isfinite(scaled),
            np.floor(scaled) - 1.0,
            np.where(scaled > 0.0, 128.0, -128.0),
        )
    return np.clip(gates, -128.0, 127.0).astype(np.int8)


@dataclass(frozen=True)
class _HeadLayout:
    """Precomputed state of one int8 head for :class:`QuantizedYoloxDecoder`."""

    height: int
    width: int
    channels: int
    stride: int
    fix_point: int
    dequant_factor: float
    grid_x: NDArray[np.float64]
    grid_y: NDArray[np.float64]
    keep: NDArray[np.bool_]


class QuantizedYoloxDecoder:
    """Threshold-first decoder for the DPU's quantized YOLOX heads.

    Behaviourally identical to dequantizing each head and calling
    :func:`decode_yolox` with ``logits=True``, but the objectness and class
    logits are thresholded in the int8 domain first. ``score > threshold`` with
    ``score = sigmoid(objectness) * sigmoid(class)`` implies that both logits
    already exceed their integer gate, so a cell whose int8 objectness is below
    the loosest class gate, or whose class logits are all below their own gates,
    can never qualify; those cells are dropped with two vectorised scans of the
    raw buffers. Sigmoid, grid decode, box decode and dequantization then run
    only on the survivors, which are decoded in multi-label mode like the float
    reference (one candidate per class above its threshold, at most
    :data:`MAX_CLASSES_PER_CELL` per cell).

    Grids and scratch buffers are precomputed in the constructor; the scratch
    buffers make one instance non-reentrant, so call :meth:`decode` from one
    thread at a time.

    Args:
        shapes: Per-output ``(H, W, C)`` NHWC shapes, in the order the DPU
            returns the tensors.
        strides: Feature-map stride of each output (8, 16, 32).
        fix_points: Output ``fix_point`` of each tensor; values are dequantized
            with ``value * 2 ** -fix_point``.

    Raises:
        ValueError: If shapes, strides or fix points do not line up, or a head
            has fewer than six channels.
    """

    def __init__(
        self,
        shapes: Sequence[tuple[int, int, int]],
        strides: Sequence[int],
        fix_points: Sequence[int],
    ) -> None:
        if not shapes:
            raise ValueError("shapes must not be empty")
        if len(shapes) != len(strides) or len(shapes) != len(fix_points):
            raise ValueError(
                f"got {len(shapes)} shapes, {len(strides)} strides and {len(fix_points)} fix points"
            )

        layouts: list[_HeadLayout] = []
        for index, (shape, raw_stride, fix_point) in enumerate(
            zip(shapes, strides, fix_points, strict=True)
        ):
            if len(shape) != 3:
                raise ValueError(f"shapes[{index}] must be (H, W, C), got {shape}")
            height, width, channels = (int(value) for value in shape)
            if height <= 0 or width <= 0:
                raise ValueError(f"shapes[{index}] must be positive, got {shape}")
            if channels < _MIN_CHANNELS:
                raise ValueError(
                    f"shapes[{index}] needs at least {_MIN_CHANNELS} channels, got {channels}"
                )
            stride = int(raw_stride)
            if stride <= 0:
                raise ValueError(f"strides[{index}] must be positive, got {stride}")
            cells = height * width
            layouts.append(
                _HeadLayout(
                    height=height,
                    width=width,
                    channels=channels,
                    stride=stride,
                    fix_point=int(fix_point),
                    dequant_factor=2.0 ** -int(fix_point),
                    # Same order as ``np.meshgrid(arange(width), arange(height))``
                    # flattened in C order, without building the 2-D grids.
                    grid_x=np.tile(np.arange(width, dtype=np.float64), height),
                    grid_y=np.repeat(np.arange(height, dtype=np.float64), width),
                    keep=np.empty(cells, dtype=np.bool_),
                )
            )

        self._layouts = tuple(layouts)
        self._order = tuple(
            sorted(range(len(layouts)), key=lambda position: layouts[position].stride)
        )
        # Per-class thresholds and their int8 gates are a property of the
        # configuration, not of the frame: build them once, not 50 times a second.
        self._threshold_cache: dict[
            tuple[float, tuple[tuple[int, float], ...], int, int],
            tuple[NDArray[np.float64], NDArray[np.int8]],
        ] = {}
        self.last_candidate_count = 0

    def _thresholds(
        self,
        layout: _HeadLayout,
        conf_thr: float,
        class_thr: Mapping[int, float] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
        """Return the float thresholds of a head and their int8 gates, cached."""
        overrides = tuple(sorted(class_thr.items())) if class_thr else ()
        key = (conf_thr, overrides, layout.channels, layout.fix_point)
        cached = self._threshold_cache.get(key)
        if cached is None:
            thresholds = _class_thresholds(conf_thr, class_thr, layout.channels - _CLASS_START)
            cached = (thresholds, _threshold_ints(thresholds, layout.fix_point))
            self._threshold_cache[key] = cached
        return cached

    def decode(
        self,
        outputs: Sequence[NDArray[np.int8]],
        conf_thr: float,
        iou_thr: float,
        class_thr: Mapping[int, float] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
        """Decode raw int8 heads, thresholding the logits before activation.

        Args:
            outputs: Per-output int8 ``(H, W, C)`` (or ``(1, H, W, C)``) tensors
                in constructor order.
            conf_thr: Default minimum ``objectness * class`` score; keep scores
                strictly above it.
            iou_thr: IoU threshold of the class-aware NMS.
            class_thr: Minimum ``objectness * class`` score of single COCO
                classes, overriding ``conf_thr``; the classes it omits keep
                ``conf_thr``.

        Returns:
            ``(boxes, scores, class_ids)`` exactly as :func:`decode_yolox` would
            return them for the dequantized heads.

        Raises:
            ValueError: If a threshold is outside ``[0, 1]``, ``class_thr``
                names a class the heads do not have, an output is not int8, or
                its shape does not match the constructor's.
        """

        if not 0.0 <= conf_thr <= 1.0:
            raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
        if not 0.0 <= iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in [0, 1], got {iou_thr}")
        if len(outputs) != len(self._layouts):
            raise ValueError(
                f"got {len(outputs)} outputs but this decoder expects {len(self._layouts)}"
            )

        boxes: list[NDArray[np.float64]] = []
        scores: list[NDArray[np.float64]] = []
        classes: list[NDArray[np.intp]] = []
        candidates = 0

        # Ascending stride reproduces the anchor order of decode_yolox.
        for position in self._order:
            layout = self._layouts[position]
            raw = np.asarray(outputs[position])
            if raw.dtype != np.int8:
                raise ValueError(f"outputs[{position}] must be int8, got {raw.dtype}")
            if raw.ndim == 4 and raw.shape[0] == 1:
                raw = raw[0]
            if raw.shape != (layout.height, layout.width, layout.channels):
                raise ValueError(
                    f"outputs[{position}] must be "
                    f"{(layout.height, layout.width, layout.channels)}, got {raw.shape}"
                )

            # Objectness first (one column), then the class logits of its
            # survivors only: a background cell costs a single byte compare,
            # not an 80-channel scan. A candidate needs the loosest class gate
            # to pass, so the objectness gate is the minimum of the class gates.
            thresholds, gates = self._thresholds(layout, conf_thr, class_thr)
            flat = raw.reshape(layout.height * layout.width, layout.channels)
            np.greater_equal(flat[:, _OBJECTNESS], np.min(gates), out=layout.keep)
            index = np.flatnonzero(layout.keep)
            if index.size == 0:
                continue
            index = index[np.any(flat[index, _CLASS_START:] >= gates, axis=1)]
            if index.size == 0:
                continue

            factor = layout.dequant_factor
            regression = flat[index, :_BOX_CHANNELS].astype(np.float64) * factor
            objectness = _sigmoid(flat[index, _OBJECTNESS].astype(np.float64) * factor)
            class_scores = _sigmoid(flat[index, _CLASS_START:].astype(np.float64) * factor)
            cell_scores = objectness[:, None] * class_scores

            center_x = (regression[:, 0] + layout.grid_x[index]) * layout.stride
            center_y = (regression[:, 1] + layout.grid_y[index]) * layout.stride
            with np.errstate(over="ignore"):
                box_width = np.exp(regression[:, 2]) * layout.stride
                box_height = np.exp(regression[:, 3]) * layout.stride
            box = np.stack(
                (
                    center_x - box_width / 2.0,
                    center_y - box_height / 2.0,
                    center_x + box_width / 2.0,
                    center_y + box_height / 2.0,
                ),
                axis=-1,
            )

            usable = (
                np.isfinite(box).all(axis=-1) & (box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])
            )
            cells, class_ids, kept_scores = _cell_candidates(cell_scores, thresholds, usable)
            if cells.size == 0:
                continue
            candidates += int(cells.size)
            boxes.append(box[cells])
            scores.append(kept_scores)
            classes.append(class_ids)

        self.last_candidate_count = candidates
        return _finalize(boxes, scores, classes, iou_thr)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def _validate_frame(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return ``rgb`` after checking the shape and dtype the graph needs."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {image.dtype}")
    return image


@dataclass(frozen=True)
class ZoomCrop:
    """The part of the frame the detector sees, in full-frame pixels."""

    left: int
    top: int
    width: int
    height: int

    @property
    def origin(self) -> tuple[float, float]:
        """The crop's top-left corner, for :func:`boxes_to_source`."""
        return (float(self.left), float(self.top))

    @property
    def size(self) -> tuple[int, int]:
        """The crop size as ``(height, width)``, for :func:`boxes_to_source`."""
        return (self.height, self.width)


def zoom_crop(
    width: int,
    height: int,
    zoom: float,
    centre: tuple[float, float] = (0.5, 0.5),
) -> ZoomCrop:
    """Return the region of a ``width x height`` frame that ``zoom`` shows.

    ``zoom`` 1.0 is the whole frame; 1.3 feeds the central 77% of it into the
    same 416x416 input, so every object reaches the network 1.3 times larger and
    a small object the whole frame shrank below a stride cell becomes visible
    again. ``centre`` is the crop's centre as a fraction of the frame and is
    clamped so the crop always stays inside it.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        zoom: Linear magnification of the crop, at least 1.0.
        centre: Crop centre as ``(x, y)`` fractions of the frame.

    Returns:
        The crop, in full-frame pixels.

    Raises:
        ValueError: If the frame is empty, ``zoom`` is below 1.0 or the centre
            is not a pair of fractions.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"frame must be positive, got {width}x{height}")
    if not math.isfinite(zoom) or zoom < 1.0:
        raise ValueError(f"zoom must be at least 1.0, got {zoom}")
    centre_x, centre_y = float(centre[0]), float(centre[1])
    if not (0.0 <= centre_x <= 1.0 and 0.0 <= centre_y <= 1.0):
        raise ValueError(f"centre must be fractions in [0, 1], got {centre}")
    crop_width = max(1, min(width, round(width / zoom)))
    crop_height = max(1, min(height, round(height / zoom)))
    left = round(centre_x * width - crop_width / 2.0)
    top = round(centre_y * height - crop_height / 2.0)
    return ZoomCrop(
        left=min(max(left, 0), width - crop_width),
        top=min(max(top, 0), height - crop_height),
        width=crop_width,
        height=crop_height,
    )


def letterbox_geometry(
    height: int,
    width: int,
    out_height: int,
    out_width: int,
) -> tuple[int, int, float, int, int]:
    """Return ``(resized_height, resized_width, scale, top, left)``.

    The rounding matches upstream YOLOX's letterbox, so the OpenCV path, the
    reference checker and the NumPy geometry agree.
    """

    scale = min(out_height / height, out_width / width)
    resized_height = max(1, min(out_height, round(height * scale)))
    resized_width = max(1, min(out_width, round(width * scale)))
    return (
        resized_height,
        resized_width,
        float(scale),
        (out_height - resized_height) // 2,
        (out_width - resized_width) // 2,
    )


def prepare_input(
    rgb: NDArray[np.uint8],
    size: tuple[int, int] = YOLOX_INPUT_SIZE,
) -> tuple[NDArray[np.uint8], float, tuple[float, float]]:
    """Convert one RGB frame into the graph's BGR letterboxed input.

    Args:
        rgb: ``(H, W, 3)`` RGB uint8 frame.
        size: Target ``(height, width)``.

    Returns:
        ``(bgr, scale, (pad_x, pad_y))``: the contiguous letterboxed BGR image
        and the mapping :func:`boxes_to_source` inverts.

    Raises:
        ValueError: If the frame or ``size`` is unusable.
    """

    image = _validate_frame(rgb)
    out_height, out_width = int(size[0]), int(size[1])
    if out_height <= 0 or out_width <= 0:
        raise ValueError(f"size must be positive, got {size}")
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("rgb must not be empty")
    if (height, width) == (out_height, out_width):
        same = np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), dtype=np.uint8)
        return np.ascontiguousarray(same), 1.0, (0.0, 0.0)
    resized_height, resized_width, scale, top, left = letterbox_geometry(
        height, width, out_height, out_width
    )
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    bgr = np.asarray(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), dtype=np.uint8)
    canvas = np.asarray(
        cv2.copyMakeBorder(
            bgr,
            top,
            out_height - resized_height - top,
            left,
            out_width - resized_width - left,
            cv2.BORDER_CONSTANT,
            value=(YOLOX_PAD_VALUE, YOLOX_PAD_VALUE, YOLOX_PAD_VALUE),
        ),
        dtype=np.uint8,
    )
    return canvas, scale, (float(left), float(top))


def _quantize_into(
    image: NDArray[np.uint8],
    lut: NDArray[np.int8],
    out: NDArray[np.int8],
) -> None:
    """Apply the quantization table with OpenCV's byte LUT (a pure remapping)."""

    cv2.LUT(image, np.ascontiguousarray(lut.view(np.uint8)), dst=out.view(np.uint8))


def prepare_input_into(
    rgb: NDArray[np.uint8],
    size: tuple[int, int],
    lut: NDArray[np.int8],
    out: NDArray[np.int8],
    *,
    fill_pad: bool = True,
) -> tuple[float, tuple[float, float]]:
    """Letterbox, swap and quantize one frame straight into a DPU input view.

    The resized frame is written into the middle of ``out`` (the ``(H, W, 3)``
    view of the DPU input tensor) and only that region is quantized by the LUT,
    so no full-size uint8 canvas is allocated or copied per frame.

    Args:
        rgb: ``(H, W, 3)`` RGB uint8 frame.
        size: Target ``(height, width)``, the model input size.
        lut: 256-entry int8 quantization table, as built by
            :func:`cpsa.dpu.input_quantization_lut_scaled`.
        out: ``(height, width, 3)`` C-contiguous int8 buffer to fill.
        fill_pad: Write the quantized pad value into the border rows. Pass
            ``False`` when a previous call with the same source size already
            filled them (the border never changes).

    Returns:
        ``(scale, (pad_x, pad_y))``, the mapping :func:`boxes_to_source` inverts.

    Raises:
        ValueError: If the frame, the target size or ``out`` do not match.
    """

    image = _validate_frame(rgb)
    out_height, out_width = int(size[0]), int(size[1])
    if out.shape != (out_height, out_width, 3) or out.dtype != np.int8:
        raise ValueError(
            f"out must be a {(out_height, out_width, 3)} int8 array, got {out.shape} {out.dtype}"
        )
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("out must be C-contiguous")

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("rgb must not be empty")
    if (height, width) == (out_height, out_width):
        _quantize_into(np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), dtype=np.uint8), lut, out)
        return 1.0, (0.0, 0.0)
    resized_height, resized_width, scale, top, left = letterbox_geometry(
        height, width, out_height, out_width
    )
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    bgr = np.asarray(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), dtype=np.uint8)
    if left == 0 and resized_width == out_width:
        # A 4:3 frame letterboxes to the full width of the square input, so the
        # destination is one contiguous row block of the DPU buffer.
        if fill_pad:
            pad_quantized = lut[YOLOX_PAD_VALUE]
            out[:top].fill(pad_quantized)
            out[top + resized_height :].fill(pad_quantized)
        _quantize_into(bgr, lut, out[top : top + resized_height])
    else:
        canvas = np.full((out_height, out_width, 3), YOLOX_PAD_VALUE, dtype=np.uint8)
        canvas[top : top + resized_height, left : left + resized_width] = bgr
        _quantize_into(canvas, lut, out)
    return scale, (float(left), float(top))


def boxes_to_source(
    boxes: NDArray[np.floating[Any]],
    scale: float,
    pad: tuple[float, float],
    size: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    """Map ``xyxy`` boxes from letterboxed input pixels to clipped source pixels.

    Args:
        boxes: ``(N, 4)`` ``xyxy`` boxes in model-input pixels.
        scale: Uniform resize factor returned by :func:`prepare_input`.
        pad: ``(pad_x, pad_y)`` returned by :func:`prepare_input`.
        size: Size ``(height, width)`` of the region the model saw; a zoomed
            detector passes the crop size here.
        origin: Top-left corner of that region in the full frame, ``(0, 0)``
            when the model saw the whole frame.

    Returns:
        ``(N, 4)`` boxes in full-frame pixels: clipped to the region the model
        saw (so a box that ran into the letterbox pad stops at the crop edge)
        and then shifted to the full frame.

    Raises:
        ValueError: If ``scale`` is not positive.
    """

    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    array: NDArray[np.float64] = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    pad_x, pad_y = float(pad[0]), float(pad[1])
    array[:, [0, 2]] = (array[:, [0, 2]] - pad_x) / scale
    array[:, [1, 3]] = (array[:, [1, 3]] - pad_y) / scale
    height, width = size
    array[:, [0, 2]] = np.clip(array[:, [0, 2]], 0.0, float(width))
    array[:, [1, 3]] = np.clip(array[:, [1, 3]], 0.0, float(height))
    array[:, [0, 2]] += float(origin[0])
    array[:, [1, 3]] += float(origin[1])
    return array


def detections_from_heads(
    heads: Sequence[NDArray[np.floating[Any]]],
    strides: Sequence[int],
    *,
    conf_thr: float,
    iou_thr: float,
    scale: float,
    pad: tuple[float, float],
    source_hw: tuple[int, int],
    logits: bool = True,
    fix_points: Sequence[int] | None = None,
    class_thr: Mapping[int, float] | None = None,
) -> list[Detection]:
    """Decode YOLOX heads into source-pixel detections.

    This is the detector's CPU side without the DPU: :meth:`YoloxDetector.wait`
    calls it with the dequantized graph outputs, while
    ``tools/vitis/yolox_reference.py`` reuses it with the package model's
    reference heads.

    Args:
        heads: ``(H_i, W_i, 85)`` head tensors in ``strides`` order.
        strides: Network stride of each head.
        conf_thr: Minimum ``objectness * class`` score.
        iou_thr: Class-aware NMS IoU threshold.
        scale: Uniform resize factor from :func:`prepare_input`.
        pad: ``(pad_x, pad_y)`` from :func:`prepare_input`.
        source_hw: Source frame ``(height, width)``.
        logits: ``True`` for raw logits (the zoo graphs).
        fix_points: When given, each head is round-tripped through int8 with
            that ``fix_point`` first, reproducing the DPU's output precision.
        class_thr: Per-class scores overriding ``conf_thr`` (see :func:`decode_yolox`).

    Returns:
        Detections in source pixels, score-descending, degenerate boxes dropped.
    """

    if fix_points is not None:
        if len(fix_points) != len(heads):
            raise ValueError(f"got {len(heads)} heads but {len(fix_points)} fix points")
        staged: list[NDArray[np.floating[Any]]] = []
        for head, fix_point in zip(heads, fix_points, strict=True):
            scaled = np.asarray(head, dtype=np.float64) * (2.0 ** int(fix_point))
            quantized = np.clip(np.rint(scaled), -128.0, 127.0).astype(np.int8)
            staged.append(dequantize_fix_point(quantized, int(fix_point)))
        heads = staged

    boxes, scores, class_ids = decode_yolox(
        heads,
        strides,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        logits=logits,
        class_thr=class_thr,
    )
    return _detections_from_decoded(
        boxes, scores, class_ids, scale=scale, pad=pad, source_hw=source_hw
    )


def _detections_from_decoded(
    boxes: NDArray[np.floating[Any]],
    scores: NDArray[np.floating[Any]],
    class_ids: NDArray[np.integer[Any]],
    *,
    scale: float,
    pad: tuple[float, float],
    source_hw: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
) -> list[Detection]:
    """Map decoded input-pixel boxes to full-frame pixels, score-descending."""

    if scores.size == 0:
        return []
    mapped = boxes_to_source(boxes, scale, pad, source_hw, origin)
    detections = [
        Detection(
            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            score=float(score),
            class_id=int(class_id),
        )
        for box, score, class_id in zip(mapped, scores, class_ids, strict=True)
        if box[2] > box[0] and box[3] > box[1]
    ]
    detections.sort(key=lambda detection: detection.score, reverse=True)
    return detections


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #


_THRESHOLD_CACHE_MAX = 32
"""Threshold tables kept per decoder before a live calibration clears them."""


class YoloxDetector:
    """Run one compiled YOLOX ``.xmodel`` graph on the KV260 DPU.

    Args:
        xmodel_path: Path to the compiled ``.xmodel``.
        conf_thr: Default minimum ``objectness * class`` score for a detection.
        iou_thr: Class-aware NMS IoU threshold.
        class_thr: Minimum ``objectness * class`` score of single COCO classes,
            overriding ``conf_thr``: the showcase keeps a stricter ``person`` and
            a looser threshold for the objects held in front of it.
        zoom: Centre-crop magnification before the letterbox (``1.0`` is the
            whole frame, ``1.3`` the central 77%). Objects reach the network
            ``zoom`` times larger at no extra DPU cost.
        zoom_center: Crop centre as ``(x, y)`` fractions of the frame.

    Raises:
        RuntimeError: If the ``xir``/``vart`` Python bindings are unavailable.
        FileNotFoundError: If ``xmodel_path`` does not exist.
        ValueError: If a threshold is out of range or names a class the graph
            does not have, the graph has no DPU subgraph, or the tensor layout is
            unexpected.
    """

    def __init__(
        self,
        xmodel_path: str | Path,
        *,
        conf_thr: float = 0.35,
        iou_thr: float = 0.45,
        class_thr: Mapping[int, float] | None = None,
        zoom: float = 1.0,
        zoom_center: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        if not 0.0 <= conf_thr <= 1.0:
            raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
        if not 0.0 <= iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in [0, 1], got {iou_thr}")
        self._conf_thr = float(conf_thr)
        self._iou_thr = float(iou_thr)
        self._class_thr = dict(class_thr or {})
        self._zoom = 1.0
        self._zoom_center = (0.5, 0.5)
        self.set_zoom(zoom, zoom_center)

        self._runner = DpuRunner(xmodel_path)
        self._outputs: tuple[OutputTensor, ...] = self._runner.outputs
        strides = sorted(tensor.stride for tensor in self._outputs)
        if strides != list(YOLOX_STRIDES):
            raise ValueError(f"expected the YOLOX strides {list(YOLOX_STRIDES)}, got {strides}")
        # Fail at boot, not on the first frame, when a per-class threshold names
        # a class the graph does not have.
        _class_thresholds(self._conf_thr, self._class_thr, self._outputs[0].shape[2] - _CLASS_START)

        self._lut = input_quantization_lut_scaled(self._runner.input_fix_point, YOLOX_INPUT_SCALE)
        self._input_hw = self._runner.input_hw

        # Two slots so the loop can submit frame N while decoding frame N-1:
        # a DPU job owns its input and output buffers until wait() returns.
        self._slots = [self._slot() for _ in range(2)]
        self._next_slot = 0
        self._token = 0
        self._pending: dict[int, _Job] = {}
        self._decoder = QuantizedYoloxDecoder(
            [tensor.shape for tensor in self._outputs],
            [tensor.stride for tensor in self._outputs],
            [tensor.fix_point for tensor in self._outputs],
        )
        self._stage_ms: dict[str, float] = {"preprocess": 0.0, "dpu": 0.0, "decode": 0.0}

    def _slot(self) -> _Slot:
        """Allocate one input/output buffer set, with its pad border pre-quantized."""

        buffer = np.empty([1, self._input_hw[0], self._input_hw[1], 3], dtype=np.int8)
        view = buffer.reshape((*self._input_hw, 3))
        view.fill(self._lut[YOLOX_PAD_VALUE])
        return _Slot(
            input_buffer=buffer,
            input_view=view,
            output_buffers=[
                np.empty([1, *tensor.shape], dtype=np.int8) for tensor in self._outputs
            ],
        )

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input ``(height, width)``."""
        return self._input_hw

    @property
    def strides(self) -> tuple[int, ...]:
        """Network strides of the output heads, ascending."""
        return tuple(sorted(tensor.stride for tensor in self._outputs))

    @property
    def stage_ms(self) -> dict[str, float]:
        """Per-stage timings of the last :meth:`submit`/:meth:`wait`, in milliseconds."""

        return dict(self._stage_ms)

    @property
    def pending_jobs(self) -> int:
        """DPU jobs submitted and not yet waited for."""
        return len(self._pending)

    @property
    def last_candidate_count(self) -> int:
        """Candidates that survived the int8 prefilter in the last :meth:`wait`."""
        return self._decoder.last_candidate_count

    @property
    def zoom(self) -> tuple[float, tuple[float, float]]:
        """The centre crop ``(zoom, (x, y))`` the next :meth:`submit` uses."""
        return self._zoom, self._zoom_center

    def set_zoom(self, zoom: float, zoom_center: tuple[float, float] = (0.5, 0.5)) -> None:
        """Change the centre crop from the next :meth:`submit` on (live calibration).

        Only two numbers change: the crop is computed per frame and a slot
        whose letterbox geometry differs repaints its padding, so there is no
        restart and no model reload. A job already in flight keeps the crop it
        was submitted with, so its boxes still map back to the right pixels.

        Raises:
            ValueError: If ``zoom`` is below 1.0 or the centre is not a pair of
                fractions.
        """

        if not math.isfinite(zoom) or zoom < 1.0:
            raise ValueError(f"zoom must be at least 1.0, got {zoom}")
        if len(zoom_center) != 2 or not all(
            math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in zoom_center
        ):
            raise ValueError(f"zoom_center must be fractions in [0, 1], got {zoom_center}")
        self._zoom = float(zoom)
        self._zoom_center = (float(zoom_center[0]), float(zoom_center[1]))

    @property
    def class_thresholds(self) -> dict[int, float]:
        """The per-class thresholds the next decode uses, by COCO-80 class id."""
        return dict(self._class_thr)

    def set_class_thresholds(self, class_thr: Mapping[int, float]) -> None:
        """Replace the per-class thresholds from the next decode on (live calibration).

        No reload: the decoder keys its float thresholds and int8 prefilter
        gates by the threshold table, so the next frame builds (and caches) the
        new ones. A job already in flight decodes with the new table too.

        Raises:
            ValueError: If a threshold is outside ``[0, 1]`` or names a class the
                graph does not have; the previous table is kept.
        """

        _class_thresholds(self._conf_thr, class_thr, self._outputs[0].shape[2] - _CLASS_START)
        self._class_thr = dict(class_thr)
        cache = self._decoder._threshold_cache
        if len(cache) > _THRESHOLD_CACHE_MAX:
            # A slider dragged for a while leaves one entry per stop.
            cache.clear()

    def submit(self, rgb: NDArray[np.uint8], pass_: Pass | None = None) -> int:
        """Preprocess ``rgb`` and start the DPU without waiting for it.

        Returns a token for :meth:`wait`. Submitting frame N and then waiting
        for frame N-1 overlaps the DPU with the CPU decode and the draw. Two
        jobs may be in flight (one per slot); a third submit raises.

        Args:
            rgb: The ``(H, W, 3)`` RGB uint8 frame.
            pass_: The pass this frame runs: its crop and the classes
                :meth:`wait` keeps. ``None`` uses the detector's configured
                zoom and reports every class.

        Raises:
            RuntimeError: If both slots are busy.
            ValueError: If ``rgb`` is not an ``(H, W, 3)`` uint8 frame, or the
                pass asks for a crop the frame cannot give.
        """

        image = _validate_frame(rgb)
        if not self._slots:
            raise RuntimeError("the DPU runner is closed")
        if len(self._pending) >= len(self._slots):
            raise RuntimeError("both DPU slots are busy; call wait() before submitting again")
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        schedule = pass_ if pass_ is not None else Pass(self._zoom)
        height, width = int(image.shape[0]), int(image.shape[1])
        crop = zoom_crop(width, height, schedule.zoom, self._zoom_center)
        geometry = crop.size

        started = time.perf_counter()
        fill_pad = slot.pad_geometry != geometry
        # The crop is a view: cv2 resize reads it in place, so a zoomed frame
        # letterboxes into the same buffer without an extra copy.
        scale, pad = prepare_input_into(
            image[crop.top : crop.top + crop.height, crop.left : crop.left + crop.width],
            self._input_hw,
            self._lut,
            slot.input_view,
            fill_pad=fill_pad,
        )
        slot.pad_geometry = geometry
        self._stage_ms["preprocess"] = (time.perf_counter() - started) * 1000.0

        dpu_started = time.perf_counter()
        job_id = self._runner.execute([slot.input_buffer], slot.output_buffers)
        self._token += 1
        token = self._token
        self._pending[token] = _Job(
            slot=slot,
            job_id=job_id,
            dpu_started=dpu_started,
            scale=scale,
            pad=pad,
            source_hw=geometry,
            origin=crop.origin,
            classes=schedule.classes,
        )
        return token

    def wait(self, token: int) -> list[Detection]:
        """Wait for a job from :meth:`submit` and decode it into detections.

        Raises:
            ValueError: If ``token`` is not a pending job.
        """

        job = self._pending.pop(token, None)
        if job is None:
            raise ValueError(f"no pending DPU job with token {token}")
        self._runner.wait(job.job_id)
        self._stage_ms["dpu"] = (time.perf_counter() - job.dpu_started) * 1000.0

        started = time.perf_counter()
        boxes, scores, class_ids = self._decoder.decode(
            job.slot.output_buffers, self._conf_thr, self._iou_thr, self._class_thr
        )
        detections = _detections_from_decoded(
            boxes,
            scores,
            class_ids,
            scale=job.scale,
            pad=job.pad,
            source_hw=job.source_hw,
            origin=job.origin,
        )
        if job.classes is not None:
            detections = [item for item in detections if item.class_id in job.classes]
        self._stage_ms["decode"] = (time.perf_counter() - started) * 1000.0
        return detections

    def detect(self, rgb: NDArray[np.uint8], pass_: Pass | None = None) -> list[Detection]:
        """Run the DPU model on an ``(H, W, 3)`` RGB uint8 frame, through ``pass_``."""
        return self.wait(self.submit(rgb, pass_))

    def close(self) -> None:
        """Release the DPU runner and the two buffer sets (R2-G model swap)."""

        if not self._slots:
            return
        self._pending.clear()
        self._slots = []
        self._runner.close()


@dataclass
class _Slot:
    """One DPU buffer set: the quantized input view and the raw int8 heads."""

    input_buffer: NDArray[np.int8]
    input_view: NDArray[np.int8]
    output_buffers: list[NDArray[np.int8]]
    pad_geometry: tuple[int, int] | None = None
    """Source ``(height, width)`` whose letterbox pad border is already in the buffer."""


@dataclass(frozen=True)
class _Job:
    """One submitted DPU job, enough to decode its outputs in :meth:`YoloxDetector.wait`."""

    slot: _Slot
    job_id: Any
    dpu_started: float
    scale: float
    pad: tuple[float, float]
    source_hw: tuple[int, int]
    origin: tuple[float, float] = (0.0, 0.0)
    classes: frozenset[int] | None = None
    """Classes :meth:`YoloxDetector.wait` keeps; ``None`` keeps every class."""


COCO80_EN: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

COCO80_IT: tuple[str, ...] = (
    "persona",
    "bicicletta",
    "auto",
    "motocicletta",
    "aereo",
    "autobus",
    "treno",
    "camion",
    "barca",
    "semaforo",
    "idrante",
    "stop",
    "parchimetro",
    "panchina",
    "uccello",
    "gatto",
    "cane",
    "cavallo",
    "pecora",
    "mucca",
    "elefante",
    "orso",
    "zebra",
    "giraffa",
    "zaino",
    "ombrello",
    "borsa",
    "cravatta",
    "valigia",
    "frisbee",
    "sci",
    "snowboard",
    "palla",
    "aquilone",
    "mazza da baseball",
    "guantone",
    "skateboard",
    "tavola da surf",
    "racchetta",
    "bottiglia",
    "calice",
    "tazza",
    "forchetta",
    "coltello",
    "cucchiaio",
    "ciotola",
    "banana",
    "mela",
    "panino",
    "arancia",
    "broccolo",
    "carota",
    "hot dog",
    "pizza",
    "ciambella",
    "torta",
    "sedia",
    "divano",
    "pianta",
    "letto",
    "tavolo",
    "water",
    "tv",
    "portatile",
    "mouse",
    "telecomando",
    "tastiera",
    "cellulare",
    "microonde",
    "forno",
    "tostapane",
    "lavandino",
    "frigorifero",
    "libro",
    "orologio",
    "vaso",
    "forbici",
    "orsacchiotto",
    "phon",
    "spazzolino",
)


PERSON_CLASS = 0
"""COCO-80 id of ``person``: the one class a whole-frame pass reports."""

PERSON_CLASSES = frozenset({PERSON_CLASS})
"""The classes of the whole-frame pass: only a person walking in from a side."""

OBJECT_CLASSES = frozenset(range(PERSON_CLASS + 1, len(COCO80_EN)))
"""The classes of the zoomed pass: every held object and every piece of furniture."""


@dataclass(frozen=True)
class Pass:
    """One detector pass: the region the network sees and the classes it reports.

    The showcase alternates two passes because one crop cannot serve both ends
    of the booth: the whole frame finds a person entering from a side, and the
    centre crop (``model.zoom``) makes a small held bottle big enough to stay
    above a stride cell. ``classes`` ``None`` reports every class, which is what
    a detector without the alternating schedule does.

    Attributes:
        zoom: Crop magnification, ``1.0`` for the whole frame.
        classes: The class ids this pass may report, ``None`` for all of them.
    """

    zoom: float
    classes: frozenset[int] | None = None

    @property
    def persons_only(self) -> bool:
        """Whether this pass reports nothing but the person class."""
        return self.classes == PERSON_CLASSES
