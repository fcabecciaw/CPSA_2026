"""YOLOX anchor-free head decoder (F-S1).

The Vitis AI model-zoo YOLOX graphs stop before post-processing: every stride
returns one ``(H, W, 85)`` tensor whose per-cell channels are

```text
[ tx | ty | tw | th | objectness | class scores (80) ]
```

with ``tx``/``ty`` offsets in grid cells, ``tw``/``th`` in log space, and the
objectness/class channels as raw logits (the zoo ``postprocess`` applies
``sigmoid`` to channels ``4:``).  Decoding follows upstream YOLOX:

```text
cx = (tx + grid_x) * stride     cy = (ty + grid_y) * stride
w  = exp(tw) * stride           h  = exp(th) * stride
score = objectness * max(class)
```

Boxes come back as ``xyxy`` in model-input pixels; mapping them to the source
frame and clipping is the adapter's job (:func:`cpsa.adapters.detect.yolox_dpu.
boxes_to_source`).  The module is numpy only, so the decode maths can be tested
and reused without the DPU runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["decode_yolox"]

#: Boxes kept after non-maximum suppression, as in the zoo post-processing.
MAX_DETECTIONS = 100

_MIN_CHANNELS = 6


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


def _class_aware_nms(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    class_ids: NDArray[np.intp],
    iou_thr: float,
) -> NDArray[np.intp]:
    """Greedy per-class suppression: overlapping boxes of different classes both survive."""

    order = np.argsort(-scores, kind="stable").astype(np.intp)
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        if rest.size == 0:
            break
        overlap = _iou(boxes[best], boxes[rest])
        same_class = class_ids[rest] == class_ids[best]
        order = rest[~(same_class & (overlap > iou_thr))]
    return np.asarray(keep, dtype=np.intp)


def _decode_head(
    head: NDArray[np.float64],
    stride: int,
    *,
    logits: bool,
    conf_thr: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Decode one stride's ``(H, W, 85)`` head into candidate ``xyxy`` boxes."""

    regression = head[..., :4]
    objectness = head[..., 4]
    class_scores = head[..., 5:]
    if logits:
        objectness = _sigmoid(objectness)
        class_scores = _sigmoid(class_scores)

    class_ids = np.argmax(class_scores, axis=-1)
    best = np.take_along_axis(class_scores, class_ids[..., None], axis=-1)[..., 0]
    scores = objectness * best

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
        np.isfinite(scores)
        & np.isfinite(boxes).all(axis=-1)
        & (scores > conf_thr)
        & (boxes[..., 2] > boxes[..., 0])
        & (boxes[..., 3] > boxes[..., 1])
    )
    selected = np.flatnonzero(usable).astype(np.intp)
    return (
        boxes.reshape(-1, 4)[selected],
        scores.reshape(-1)[selected],
        class_ids.reshape(-1)[selected].astype(np.intp),
    )


def decode_yolox(
    heads: Sequence[NDArray[np.floating[Any]]],
    strides: Sequence[int],
    *,
    conf_thr: float,
    iou_thr: float,
    logits: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Decode YOLOX heads into boxes in model-input pixels.

    Args:
        heads: One ``(H_i, W_i, 5 + n_classes)`` tensor per stride, in
            ``strides`` order.
        strides: Network stride of each head (smallest first, e.g. 8, 16, 32).
        conf_thr: Minimum ``objectness * class`` score in ``[0, 1]``.
        iou_thr: NMS IoU threshold in ``[0, 1]``; suppression is class aware.
        logits: ``True`` when the head channels hold raw logits (the zoo
            graphs), ``False`` when they were already activated.

    Returns:
        ``(boxes, scores, class_ids)`` where ``boxes`` is ``(N, 4)`` ``xyxy`` in
        input pixels, ``scores`` is ``(N,)`` and ``class_ids`` is ``(N,)``
        ``intp``; at most :data:`MAX_DETECTIONS` survive, score-descending.
        All three are empty when nothing passes ``conf_thr``.

    Raises:
        ValueError: If the head/stride counts differ, a head is not
            ``(H, W, C >= 6)``, the channel counts differ, a stride is not
            positive, or a threshold is outside ``[0, 1]``.
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
        decoded = _decode_head(array, int(stride), logits=logits, conf_thr=conf_thr)
        boxes.append(decoded[0])
        scores.append(decoded[1])
        classes.append(decoded[2])

    candidates = np.concatenate(boxes, axis=0)
    candidate_scores = np.concatenate(scores, axis=0)
    candidate_classes = np.concatenate(classes, axis=0)
    if candidate_scores.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.intp),
        )

    keep = _class_aware_nms(candidates, candidate_scores, candidate_classes, iou_thr)
    keep = keep[np.argsort(-candidate_scores[keep], kind="stable")][:MAX_DETECTIONS]
    return candidates[keep], candidate_scores[keep], candidate_classes[keep]
