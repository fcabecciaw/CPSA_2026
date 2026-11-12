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
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cpsa.vision.pose_decode import _class_threshold_int

__all__ = ["MAX_DETECTIONS", "QuantizedYoloxDecoder", "decode_yolox"]

#: Boxes kept after non-maximum suppression, as in the zoo post-processing.
MAX_DETECTIONS = 100

_MIN_CHANNELS = 6

#: Channel layout of a head cell: ``[tx, ty, tw, th, objectness, classes...]``.
_OBJECTNESS = 4
_CLASS_START = 5
_BOX_CHANNELS = 4


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

    keep = _class_aware_nms(candidates, candidate_scores, candidate_classes, iou_thr)
    keep = keep[np.argsort(-candidate_scores[keep], kind="stable")][:MAX_DETECTIONS]
    return candidates[keep], candidate_scores[keep], candidate_classes[keep]


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

    return _finalize(boxes, scores, classes, iou_thr)


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
    class_max: NDArray[np.int8]


class QuantizedYoloxDecoder:
    """Threshold-first decoder for the DPU's quantized YOLOX heads.

    Behaviourally identical to dequantizing each head and calling
    :func:`decode_yolox` with ``logits=True``, but the objectness and class
    logits are thresholded in the int8 domain first.  ``score > conf_thr`` with
    ``score = sigmoid(objectness) * sigmoid(best_class)`` implies that *both*
    logits already exceed ``logit(conf_thr)``, so a cell whose int8 objectness or
    best class logit is below that integer threshold can never qualify; those
    cells are dropped with one vectorised scan of the raw buffers.  Sigmoid,
    grid decode, box decode and dequantization then run only on the survivors.
    The exact ``score > conf_thr`` comparison still decides, on the dequantized
    values, so no surviving cell is kept that the float path would drop.

    Grids, channel offsets and scratch buffers are precomputed in the
    constructor; the scratch buffers make one instance non-reentrant, so call
    :meth:`decode` from one thread at a time.

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
                    class_max=np.empty(cells, dtype=np.int8),
                )
            )

        self._layouts = tuple(layouts)
        self._order = tuple(
            sorted(range(len(layouts)), key=lambda position: layouts[position].stride)
        )
        self.last_candidate_count = 0

    def decode(
        self,
        outputs: Sequence[NDArray[np.int8]],
        conf_thr: float,
        iou_thr: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
        """Decode raw int8 heads, thresholding the logits before activation.

        Args:
            outputs: Per-output int8 ``(H, W, C)`` (or ``(1, H, W, C)``) tensors
                in constructor order.
            conf_thr: Minimum ``objectness * class`` score; keep scores strictly
                above it.
            iou_thr: IoU threshold of the class-aware NMS.

        Returns:
            ``(boxes, scores, class_ids)`` exactly as :func:`decode_yolox` would
            return them for the dequantized heads.

        Raises:
            ValueError: If a threshold is outside ``[0, 1]``, an output is not
                int8, or its shape does not match the constructor's.
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

            # One step below the exact integer threshold, as in the medical
            # decoder: float rounding at a quantization step can never drop a
            # candidate that the float path keeps, and the exact comparison
            # below removes the extra one.  Objectness first (one column), then
            # the class logits of its survivors only: a background cell costs a
            # single byte compare, not an 80-channel scan.
            threshold = _class_threshold_int(conf_thr, layout.fix_point)
            if threshold > 127:
                continue
            flat = raw.reshape(layout.height * layout.width, layout.channels)
            np.greater_equal(flat[:, _OBJECTNESS], threshold, out=layout.keep)
            index = np.flatnonzero(layout.keep)
            if index.size == 0:
                continue
            class_max = layout.class_max[: index.size]
            np.maximum.reduce(flat[index, _CLASS_START:], axis=1, out=class_max)
            index = index[np.greater_equal(class_max, threshold)]
            count = index.size
            if count == 0:
                continue
            candidates += int(count)

            factor = layout.dequant_factor
            regression = flat[index, :_BOX_CHANNELS].astype(np.float64) * factor
            objectness = _sigmoid(flat[index, _OBJECTNESS].astype(np.float64) * factor)
            class_scores = _sigmoid(flat[index, _CLASS_START:].astype(np.float64) * factor)
            class_ids = np.argmax(class_scores, axis=1)
            best = np.take_along_axis(class_scores, class_ids[:, None], axis=1)[:, 0]
            score = objectness * best

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
                np.isfinite(score)
                & np.isfinite(box).all(axis=-1)
                & (score > conf_thr)
                & (box[:, 2] > box[:, 0])
                & (box[:, 3] > box[:, 1])
            )
            selected = np.flatnonzero(usable)
            boxes.append(box[selected])
            scores.append(score[selected])
            classes.append(class_ids[selected].astype(np.intp))

        self.last_candidate_count = candidates
        return _finalize(boxes, scores, classes, iou_thr)
