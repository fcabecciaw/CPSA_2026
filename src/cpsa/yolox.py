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
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
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
    "MAX_DETECTIONS",
    "QuantizedYoloxDecoder",
    "YOLOX_INPUT_SCALE",
    "YOLOX_INPUT_SIZE",
    "YOLOX_PAD_VALUE",
    "YOLOX_STRIDES",
    "YoloxDetector",
    "boxes_to_source",
    "decode_yolox",
    "detections_from_heads",
    "label_for",
    "labels_for",
    "prepare_input",
    "prepare_input_into",
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
        score: Objectness times the best class probability, in ``0..1``.
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
    objectness = head[..., _OBJECTNESS]
    class_scores = head[..., _CLASS_START:]
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


def _class_threshold_int(conf_thr: float, fix_point: int) -> int:
    """Return the smallest int8 logit that the threshold-first decoder must decode.

    With ``q`` the raw logit, the reference path scores a candidate with
    ``sigmoid(q * 2 ** -fix_point)`` and keeps ``score > conf_thr``. Since
    sigmoid is monotonic that is ``q > logit(conf_thr) * 2 ** fix_point``, whose
    integer equivalent is ``floor(threshold) + 1``. One less is returned, so
    float rounding at a quantization step can never drop a candidate that the
    reference path keeps; the exact float comparison removes the extra one.
    """

    if conf_thr <= 0.0:
        return -128
    if conf_thr >= 1.0:
        return 128
    logit = math.log(conf_thr) - math.log1p(-conf_thr)
    scaled = logit * 2.0**fix_point
    if not math.isfinite(scaled):
        return 128 if scaled > 0.0 else -128
    return max(-128, min(128, math.floor(scaled) - 1))


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
    logits are thresholded in the int8 domain first. ``score > conf_thr`` with
    ``score = sigmoid(objectness) * sigmoid(best_class)`` implies that *both*
    logits already exceed ``logit(conf_thr)``, so a cell whose int8 objectness
    or best class logit is below that integer threshold can never qualify; those
    cells are dropped with one vectorised scan of the raw buffers. Sigmoid, grid
    decode, box decode and dequantization then run only on the survivors.

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

            # Objectness first (one column), then the class logits of its
            # survivors only: a background cell costs a single byte compare,
            # not an 80-channel scan.
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
) -> NDArray[np.float64]:
    """Map ``xyxy`` boxes from letterboxed input pixels to clipped source pixels.

    Args:
        boxes: ``(N, 4)`` ``xyxy`` boxes in model-input pixels.
        scale: Uniform resize factor returned by :func:`prepare_input`.
        pad: ``(pad_x, pad_y)`` returned by :func:`prepare_input`.
        size: Source frame ``(height, width)``.

    Returns:
        ``(N, 4)`` boxes in source pixels, clipped to ``[0, width] x [0, height]``.

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
        heads, strides, conf_thr=conf_thr, iou_thr=iou_thr, logits=logits
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
) -> list[Detection]:
    """Map decoded input-pixel boxes to source pixels, score-descending."""

    if scores.size == 0:
        return []
    mapped = boxes_to_source(boxes, scale, pad, source_hw)
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


class YoloxDetector:
    """Run one compiled YOLOX ``.xmodel`` graph on the KV260 DPU.

    Args:
        xmodel_path: Path to the compiled ``.xmodel``.
        conf_thr: Minimum ``objectness * class`` score for a detection.
        iou_thr: Class-aware NMS IoU threshold.

    Raises:
        RuntimeError: If the ``xir``/``vart`` Python bindings are unavailable.
        FileNotFoundError: If ``xmodel_path`` does not exist.
        ValueError: If a threshold is out of range, the graph has no DPU
            subgraph, or the tensor layout is unexpected.
    """

    def __init__(
        self,
        xmodel_path: str | Path,
        *,
        conf_thr: float = 0.35,
        iou_thr: float = 0.45,
    ) -> None:
        if not 0.0 <= conf_thr <= 1.0:
            raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
        if not 0.0 <= iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in [0, 1], got {iou_thr}")
        self._conf_thr = float(conf_thr)
        self._iou_thr = float(iou_thr)

        self._runner = DpuRunner(xmodel_path)
        self._outputs: tuple[OutputTensor, ...] = self._runner.outputs
        strides = sorted(tensor.stride for tensor in self._outputs)
        if strides != list(YOLOX_STRIDES):
            raise ValueError(f"expected the YOLOX strides {list(YOLOX_STRIDES)}, got {strides}")

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
        """Cells that survived the int8 threshold in the last :meth:`wait`."""
        return self._decoder.last_candidate_count

    def submit(self, rgb: NDArray[np.uint8]) -> int:
        """Preprocess ``rgb`` and start the DPU without waiting for it.

        Returns a token for :meth:`wait`. Submitting frame N and then waiting
        for frame N-1 overlaps the DPU with the CPU decode and the draw. Two
        jobs may be in flight (one per slot); a third submit raises.

        Raises:
            RuntimeError: If both slots are busy.
            ValueError: If ``rgb`` is not an ``(H, W, 3)`` uint8 frame.
        """

        image = _validate_frame(rgb)
        if len(self._pending) >= len(self._slots):
            raise RuntimeError("both DPU slots are busy; call wait() before submitting again")
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        geometry = (int(image.shape[0]), int(image.shape[1]))

        started = time.perf_counter()
        fill_pad = slot.pad_geometry != geometry
        scale, pad = prepare_input_into(
            image,
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
            job.slot.output_buffers, self._conf_thr, self._iou_thr
        )
        detections = _detections_from_decoded(
            boxes,
            scores,
            class_ids,
            scale=job.scale,
            pad=job.pad,
            source_hw=job.source_hw,
        )
        self._stage_ms["decode"] = (time.perf_counter() - started) * 1000.0
        return detections

    def detect(self, rgb: NDArray[np.uint8]) -> list[Detection]:
        """Run the DPU model on an ``(H, W, 3)`` RGB uint8 frame."""
        return self.wait(self.submit(rgb))


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
    job_id: int
    dpu_started: float
    scale: float
    pad: tuple[float, float]
    source_hw: tuple[int, int]


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
