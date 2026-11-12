"""YOLOv5-nano (COCO-80, 640x640) on the DPU: preprocessing, decoding and NMS (Y5).

The Vitis AI model-zoo ``pt_yolov5-nano_coco_640_640_4.6G_3.0`` graph is
COCO-80 YOLOv5n and stops before post-processing. Its prototxt
(``yolov5_nano_pt.prototxt``) sets ``mean 0.0`` and ``scale 0.00392156`` per
channel, i.e. the zoo preprocessing is ``pixel / 255`` over RGB channels, with a
bilinear letterbox to ``640x640`` and pad value ``114``. The compiled graph takes
``[1, 640, 640, 3]`` xint8 NHWC with ``fix_point 6`` and returns three raw heads
``(1, 80, 80, 255)``, ``(1, 40, 40, 255)``, ``(1, 20, 20, 255)`` xint8 with the
DPU's per-head ``fix_point`` (observed 3, 4 and 4). Head ``i`` sits on stride
``640 / H``, so the strides are 8, 16 and 32; the DPU returns them in graph
order, which is not stride order.

A head cell holds three anchor slots ``[tx, ty, tw, th, objectness, 80 classes]``
(``3 * (5 + 80) = 255`` channels): ``slot a`` starts at channel ``85 * a``. The
anchors are the YOLOv5n ones the prototxt passes to the zoo post-processing:

===========  ==============================
stride       anchors ``(w, h)`` in input px
===========  ==============================
8            (10, 13) (16, 30) (33, 23)
16           (30, 61) (62, 45) (59, 119)
32           (116, 90) (156, 198) (373, 326)
===========  ==============================

The zoo post-processing (``vitis::ai::yolov3_post_process`` for
``type: YOLOV5``) keeps every head channel in the logit domain and decodes on the
CPU::

    cx = (sigmoid(tx) * 2 - 0.5 + cell_x) * stride
    cy = (sigmoid(ty) * 2 - 0.5 + cell_y) * stride
    w  = (sigmoid(tw) * 2) ** 2 * anchor_w
    h  = (sigmoid(th) * 2) ** 2 * anchor_h
    score = sigmoid(objectness) * sigmoid(class)

The zoo then runs a per-class NMS over all slots and keeps every class whose
score passes the prototxt's ``conf_threshold``, i.e. it is multi-label. This
module mirrors that decode, with the showcase's per-class thresholds on top, and
adds the int8 prefilter and the two-slot ``submit``/``wait`` pipelining the
YOLOX runtime already uses.

Like :mod:`cpsa.yolox`, the decode is multi-label: a cell can yield one candidate
per class above its own threshold (at most :data:`MAX_CLASSES_PER_CELL` per
anchor slot), so a bottle or a phone held in front of a person is not swallowed
by the ``person`` class that wins the same cells. The shared, model-independent
pieces (the ``Detection`` result type, the COCO-80 labels, the class-threshold
resolution, the letterbox geometry, the score/threshold int8 gates and the
class-aware NMS) come from :mod:`cpsa.yolox` so both detectors decode with one
set of semantics.

This runtime is **not wired into the pipeline or the configuration** yet: it is
an evaluation of YOLOv5n at its native 640 input next to YOLOX-nano at 416.
"""

from __future__ import annotations

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
from cpsa.yolox import (
    COCO80_EN,
    COCO80_IT,
    MAX_CLASSES_PER_CELL,
    MAX_DETECTIONS,
    Detection,
    Pass,
    _cell_candidates,
    _class_thresholds,
    _finalize,
    _sigmoid,
    _threshold_ints,
    _validate_frame,
    boxes_to_source,
    class_thresholds,
    label_for,
    labels_for,
    letterbox_geometry,
)

__all__ = [
    "COCO80_EN",
    "COCO80_IT",
    "Detection",
    "MAX_CLASSES_PER_CELL",
    "MAX_DETECTIONS",
    "QuantizedYolov5Decoder",
    "YOLOV5_ANCHORS",
    "YOLOV5_INPUT_SCALE",
    "YOLOV5_INPUT_SIZE",
    "YOLOV5_PAD_VALUE",
    "YOLOV5_STRIDES",
    "Yolov5Detector",
    "boxes_to_source",
    "class_thresholds",
    "decode_yolov5",
    "detections_from_heads",
    "label_for",
    "labels_for",
    "prepare_input",
    "prepare_input_into",
]

#: Model input ``(height, width)`` of the zoo YOLOv5-nano graph.
YOLOV5_INPUT_SIZE = (640, 640)

#: Letterbox fill value of the zoo preprocessing (grey, 114 in every channel).
YOLOV5_PAD_VALUE = 114

#: Network strides of the three YOLOv5 heads, smallest feature map last.
YOLOV5_STRIDES = (8, 16, 32)

#: ``scale`` of the zoo prototxt: ``pixel * 0.00392156``, i.e. raw ``0..255``
#: pixels normalised to ``0..1`` (unlike YOLOX, which consumes raw pixels).
YOLOV5_INPUT_SCALE = 1.0 / 255.0

#: Anchor ``(width, height)`` pairs of the YOLOv5n head, in input pixels, per
#: stride. They are the ``biases`` of ``yolov5_nano_pt.prototxt`` grouped three
#: per output, exactly as ``vitis::ai::yolov3_post_process`` indexes them.
YOLOV5_ANCHORS: dict[int, tuple[tuple[float, float], ...]] = {
    8: ((10.0, 13.0), (16.0, 30.0), (33.0, 23.0)),
    16: ((30.0, 61.0), (62.0, 45.0), (59.0, 119.0)),
    32: ((116.0, 90.0), (156.0, 198.0), (373.0, 326.0)),
}

#: Anchors of one cell: YOLOv5 head slots are anchor-major, ``85`` channels each.
_ANCHORS_PER_CELL = 3

#: Channel layout of one slot: ``[tx, ty, tw, th, objectness, classes...]``.
_OBJECTNESS = 4
_CLASS_START = 5
_BOX_CHANNELS = 4


def _anchors_for(stride: int) -> NDArray[np.float64]:
    """Return the ``(3, 2)`` anchor table of ``stride``.

    Raises:
        ValueError: If the stride is not one of :data:`YOLOV5_STRIDES`.
    """

    anchors = YOLOV5_ANCHORS.get(int(stride))
    if anchors is None:
        raise ValueError(
            f"no YOLOv5 anchors for stride {stride}; expected one of {sorted(YOLOV5_ANCHORS)}"
        )
    return np.asarray(anchors, dtype=np.float64)


def _head_channels(channels: int, index: int) -> int:
    """Validate one head's channel count and return its class count.

    Raises:
        ValueError: If the channels are not three anchor slots of five box
            channels plus at least one class.
    """

    if channels % _ANCHORS_PER_CELL or channels // _ANCHORS_PER_CELL <= _CLASS_START:
        raise ValueError(
            f"head {index} must hold {_ANCHORS_PER_CELL} anchor slots of "
            f"[tx, ty, tw, th, objectness, classes...], got {channels} channels"
        )
    return channels // _ANCHORS_PER_CELL - _CLASS_START


# --------------------------------------------------------------------------- #
# Head decode
# --------------------------------------------------------------------------- #


def _decode_head(
    head: NDArray[np.float64],
    stride: int,
    *,
    logits: bool,
    thresholds: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Decode one stride's ``(H, W, 255)`` head into candidate ``xyxy`` boxes.

    ``logits=True`` is the zoo graph: all head channels are raw logits and get
    ``sigmoid``. ``logits=False`` means the head was already activated (all 255
    channels are probabilities).
    """

    height, width = head.shape[:2]
    cells = height * width
    classes = head.shape[2] // _ANCHORS_PER_CELL - _CLASS_START
    anchors = _anchors_for(stride)

    slots = head.reshape(cells, _ANCHORS_PER_CELL, -1)
    regression = slots[..., :_BOX_CHANNELS]
    objectness = slots[..., _OBJECTNESS]
    class_scores = slots[..., _CLASS_START:]
    if logits:
        regression = _sigmoid(regression)
        objectness = _sigmoid(objectness)
        class_scores = _sigmoid(class_scores)

    # ``(sigmoid(t) * 2 - 0.5 + cell) * stride`` in centre form, then ``w`` and
    # ``h`` from the anchor: the zoo post-processing, in input pixels.
    grid_x = np.tile(np.arange(width), height)[:, None]
    grid_y = np.repeat(np.arange(height), width)[:, None]
    center_x = (regression[..., 0] * 2.0 - 0.5 + grid_x) * stride
    center_y = (regression[..., 1] * 2.0 - 0.5 + grid_y) * stride
    box_width = (regression[..., 2] * 2.0) ** 2 * anchors[None, :, 0]
    box_height = (regression[..., 3] * 2.0) ** 2 * anchors[None, :, 1]
    boxes = np.stack(
        (
            center_x - box_width / 2.0,
            center_y - box_height / 2.0,
            center_x + box_width / 2.0,
            center_y + box_height / 2.0,
        ),
        axis=-1,
    )

    # One row per (cell, anchor) slot, so a cell yields at most one candidate per
    # class *per anchor*, as in the zoo loop over ``c``.
    scores = (objectness[..., None] * class_scores).reshape(cells * _ANCHORS_PER_CELL, classes)
    usable = (
        np.isfinite(boxes).all(axis=-1)
        & (boxes[..., 2] > boxes[..., 0])
        & (boxes[..., 3] > boxes[..., 1])
    )
    rows, class_ids, kept_scores = _cell_candidates(scores, thresholds, usable.reshape(-1))
    return boxes.reshape(-1, _BOX_CHANNELS)[rows], kept_scores, class_ids


def decode_yolov5(
    heads: Sequence[NDArray[np.floating[Any]]],
    strides: Sequence[int],
    *,
    conf_thr: float,
    iou_thr: float,
    logits: bool,
    class_thr: Mapping[int, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Decode YOLOv5 heads into boxes in model-input pixels.

    A head slot is decoded in multi-label mode: every class whose
    ``objectness * class`` score passes its own threshold yields a candidate (at
    most :data:`MAX_CLASSES_PER_CELL` per slot), so a phone seen inside a
    person's cells survives alongside the person.

    Args:
        heads: One ``(H_i, W_i, 3 * (5 + n_classes))`` tensor per stride, in
            ``strides`` order.
        strides: Network stride of each head (8, 16 or 32).
        conf_thr: Default minimum ``objectness * class`` score in ``[0, 1]``.
        iou_thr: NMS IoU threshold in ``[0, 1]``; suppression is class aware.
        logits: ``True`` when the head channels are raw logits (the zoo graph),
            ``False`` when they were already activated.
        class_thr: Minimum ``objectness * class`` score of single COCO classes,
            overriding ``conf_thr``; the classes it omits keep ``conf_thr``.

    Returns:
        ``(boxes, scores, class_ids)`` where ``boxes`` is ``(N, 4)`` ``xyxy`` in
        input pixels, ``scores`` is ``(N,)`` and ``class_ids`` is ``(N,)``
        ``intp``; at most :data:`MAX_DETECTIONS` survive, score-descending.

    Raises:
        ValueError: If the head/stride counts differ, a head is not
            ``(H, W, C)`` with three anchor slots, the channel counts differ, a
            stride has no anchors, a threshold is outside ``[0, 1]``, or
            ``class_thr`` names a class the heads do not have.
    """

    if not 0.0 <= conf_thr <= 1.0:
        raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
    if not 0.0 <= iou_thr <= 1.0:
        raise ValueError(f"iou_thr must be in [0, 1], got {iou_thr}")
    if len(heads) != len(strides):
        raise ValueError(f"got {len(heads)} heads but {len(strides)} strides")
    if not heads:
        raise ValueError("at least one head is required")

    channels: int | None = None
    boxes: list[NDArray[np.float64]] = []
    scores: list[NDArray[np.float64]] = []
    classes: list[NDArray[np.intp]] = []
    for index, (head, stride) in enumerate(zip(heads, strides, strict=True)):
        array = np.asarray(head, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(f"head {index} must be (H, W, C), got shape {array.shape}")
        _head_channels(array.shape[2], index)
        if channels is None:
            channels = array.shape[2]
        elif array.shape[2] != channels:
            raise ValueError(f"head {index} has {array.shape[2]} channels, expected {channels}")
        if stride <= 0:
            raise ValueError(f"stride {index} must be positive, got {stride}")
        thresholds = _class_thresholds(
            conf_thr, class_thr, array.shape[2] // _ANCHORS_PER_CELL - _CLASS_START
        )
        decoded = _decode_head(array, int(stride), logits=logits, thresholds=thresholds)
        boxes.append(decoded[0])
        scores.append(decoded[1])
        classes.append(decoded[2])

    return _finalize(boxes, scores, classes, iou_thr)


@dataclass(frozen=True)
class _HeadLayout:
    """Precomputed state of one int8 head for :class:`QuantizedYolov5Decoder`."""

    height: int
    width: int
    channels: int
    classes: int
    stride: int
    fix_point: int
    dequant_factor: float
    anchors: NDArray[np.float64]
    grid_x: NDArray[np.float64]
    grid_y: NDArray[np.float64]
    keep: NDArray[np.bool_]


class QuantizedYolov5Decoder:
    """Threshold-first decoder for the DPU's quantized YOLOv5 heads.

    Behaviourally identical to dequantizing each head and calling
    :func:`decode_yolov5` with ``logits=True``, but the objectness and class
    logits are thresholded in the int8 domain first. ``score > threshold`` with
    ``score = sigmoid(objectness) * sigmoid(class)`` implies that both logits
    already exceed their integer gate, so a slot whose int8 objectness is below
    the loosest class gate, or whose class logits are all below their own gates,
    can never qualify; those slots are dropped with two vectorised scans of the
    raw buffers. Sigmoid, anchor/grid decode and dequantization then run only on
    the survivors, which are decoded in multi-label mode like the float
    reference (one candidate per class above its threshold, at most
    :data:`MAX_CLASSES_PER_CELL` per anchor slot).

    Grids and scratch buffers are precomputed in the constructor; the scratch
    buffers make one instance non-reentrant, so call :meth:`decode` from one
    thread at a time.

    Args:
        shapes: Per-output ``(H, W, C)`` NHWC shapes, in the order the DPU
            returns the tensors.
        strides: Feature-map stride of each output (8, 16 or 32).
        fix_points: Output ``fix_point`` of each tensor; values are dequantized
            with ``value * 2 ** -fix_point``.

    Raises:
        ValueError: If shapes, strides or fix points do not line up, a head has
            no usable anchor layout, or a stride has no anchors.
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
            classes = _head_channels(channels, index)
            stride = int(raw_stride)
            if stride <= 0:
                raise ValueError(f"strides[{index}] must be positive, got {stride}")
            cells = height * width
            layouts.append(
                _HeadLayout(
                    height=height,
                    width=width,
                    channels=channels,
                    classes=classes,
                    stride=stride,
                    fix_point=int(fix_point),
                    dequant_factor=2.0 ** -int(fix_point),
                    anchors=_anchors_for(stride),
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
        # configuration, not of the frame: build them once, not many times a second.
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
            thresholds = _class_thresholds(conf_thr, class_thr, layout.classes)
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
                classes, overriding ``conf_thr``.

        Returns:
            ``(boxes, scores, class_ids)`` exactly as :func:`decode_yolov5` would
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

        # Ascending stride reproduces the anchor order of decode_yolov5.
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

            thresholds, gates = self._thresholds(layout, conf_thr, class_thr)
            cells = layout.height * layout.width
            slots = raw.reshape(cells, _ANCHORS_PER_CELL, layout.channels // _ANCHORS_PER_CELL)

            # Objectness first (one value per anchor), then the class logits of
            # its survivors only: a background cell costs a handful of byte
            # compares, not an 80-channel scan. A candidate needs the loosest
            # class gate to pass, so the objectness gate is the minimum gate.
            layout.keep[:] = (slots[:, :, _OBJECTNESS] >= np.min(gates)).any(axis=1)
            index = np.flatnonzero(layout.keep)
            if index.size == 0:
                continue
            survivors = slots[index]
            qualified = np.any(np.any(survivors[:, :, _CLASS_START:] >= gates, axis=2), axis=1)
            if not qualified.any():
                continue
            index = index[qualified]
            survivors = survivors[qualified]

            factor = layout.dequant_factor
            regression = _sigmoid(survivors[:, :, :_BOX_CHANNELS].astype(np.float64) * factor)
            objectness = _sigmoid(survivors[:, :, _OBJECTNESS].astype(np.float64) * factor)
            class_scores = _sigmoid(survivors[:, :, _CLASS_START:].astype(np.float64) * factor)
            cell_scores = (objectness[:, :, None] * class_scores).reshape(
                index.size * _ANCHORS_PER_CELL, layout.classes
            )

            center_x = (
                regression[:, :, 0] * 2.0 - 0.5 + layout.grid_x[index][:, None]
            ) * layout.stride
            center_y = (
                regression[:, :, 1] * 2.0 - 0.5 + layout.grid_y[index][:, None]
            ) * layout.stride
            box_width = (regression[:, :, 2] * 2.0) ** 2 * layout.anchors[None, :, 0]
            box_height = (regression[:, :, 3] * 2.0) ** 2 * layout.anchors[None, :, 1]
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
                np.isfinite(box).all(axis=-1)
                & (box[:, :, 2] > box[:, :, 0])
                & (box[:, :, 3] > box[:, :, 1])
            )
            rows, class_ids, kept_scores = _cell_candidates(
                cell_scores, thresholds, usable.reshape(-1)
            )
            if rows.size == 0:
                continue
            candidates += int(rows.size)
            boxes.append(box.reshape(-1, _BOX_CHANNELS)[rows])
            scores.append(kept_scores)
            classes.append(class_ids)

        self.last_candidate_count = candidates
        return _finalize(boxes, scores, classes, iou_thr)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def prepare_input(
    rgb: NDArray[np.uint8],
    size: tuple[int, int] = YOLOV5_INPUT_SIZE,
) -> tuple[NDArray[np.uint8], float, tuple[float, float]]:
    """Letterbox one RGB frame into the graph's RGB input.

    The zoo prototxt normalises with ``mean 0, scale 1 / 255`` over RGB channels,
    so - unlike the YOLOX pipeline, which swaps to BGR - the channel order is
    kept as it arrives. The ``1 / 255`` factor is applied by the quantization
    LUT, not here (see :func:`prepare_input_into`).

    Args:
        rgb: ``(H, W, 3)`` RGB uint8 frame.
        size: Target ``(height, width)``.

    Returns:
        ``(rgb, scale, (pad_x, pad_y))``: the contiguous letterboxed RGB image
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
        return np.ascontiguousarray(image), 1.0, (0.0, 0.0)
    resized_height, resized_width, scale, top, left = letterbox_geometry(
        height, width, out_height, out_width
    )
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.asarray(
        cv2.copyMakeBorder(
            resized,
            top,
            out_height - resized_height - top,
            left,
            out_width - resized_width - left,
            cv2.BORDER_CONSTANT,
            value=(YOLOV5_PAD_VALUE, YOLOV5_PAD_VALUE, YOLOV5_PAD_VALUE),
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
    """Letterbox, normalise and quantize one frame straight into a DPU input view.

    The resized frame is written into the middle of ``out`` (the ``(H, W, 3)``
    view of the DPU input tensor) and only that region is quantized by the LUT,
    so no full-size uint8 canvas is allocated or copied per frame. The LUT must
    encode the zoo normalisation: ``input_quantization_lut_scaled(fix_point,
    YOLOV5_INPUT_SCALE)``.

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
        _quantize_into(image, lut, out)
        return 1.0, (0.0, 0.0)
    resized_height, resized_width, scale, top, left = letterbox_geometry(
        height, width, out_height, out_width
    )
    pad_quantized = lut[YOLOV5_PAD_VALUE]
    resized = np.asarray(
        cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR),
        dtype=np.uint8,
    )
    if left == 0 and resized_width == out_width:
        # A 4:3 frame letterboxes to the full width of the square input, so the
        # destination is one contiguous row block of the DPU buffer.
        if fill_pad:
            out[:top].fill(pad_quantized)
            out[top + resized_height :].fill(pad_quantized)
        _quantize_into(resized, lut, out[top : top + resized_height])
    else:
        canvas = np.full((out_height, out_width, 3), YOLOV5_PAD_VALUE, dtype=np.uint8)
        canvas[top : top + resized_height, left : left + resized_width] = resized
        _quantize_into(canvas, lut, out)
    return scale, (float(left), float(top))


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
    """Decode YOLOv5 heads into source-pixel detections.

    This is the detector's CPU side without the DPU: :meth:`Yolov5Detector.wait`
    calls it with the dequantized graph outputs, while
    ``tools/vitis/yolov5_reference.py`` reuses it with the reference heads.

    Args:
        heads: ``(H_i, W_i, 3 * (5 + 80))`` head tensors in ``strides`` order.
        strides: Network stride of each head.
        conf_thr: Minimum ``objectness * class`` score.
        iou_thr: Class-aware NMS IoU threshold.
        scale: Uniform resize factor from :func:`prepare_input`.
        pad: ``(pad_x, pad_y)`` from :func:`prepare_input`.
        source_hw: Source frame ``(height, width)``.
        logits: ``True`` for raw logits (the zoo graph).
        fix_points: When given, each head is round-tripped through int8 with
            that ``fix_point`` first, reproducing the DPU's output precision.
        class_thr: Per-class scores overriding ``conf_thr`` (see
            :func:`decode_yolov5`).

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

    boxes, scores, class_ids = decode_yolov5(
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


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #


class Yolov5Detector:
    """Run one compiled YOLOv5 ``.xmodel`` graph on the KV260 DPU.

    Args:
        xmodel_path: Path to the compiled ``.xmodel``.
        conf_thr: Default minimum ``objectness * class`` score for a detection.
        iou_thr: Class-aware NMS IoU threshold.
        class_thr: Minimum ``objectness * class`` score of single COCO classes,
            overriding ``conf_thr``: the showcase keeps a stricter ``person`` and
            a looser threshold for the objects held in front of it.

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
    ) -> None:
        if not 0.0 <= conf_thr <= 1.0:
            raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
        if not 0.0 <= iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in [0, 1], got {iou_thr}")
        self._conf_thr = float(conf_thr)
        self._iou_thr = float(iou_thr)
        self._class_thr = dict(class_thr or {})

        self._runner = DpuRunner(xmodel_path)
        self._outputs: tuple[OutputTensor, ...] = self._runner.outputs
        strides = sorted(tensor.stride for tensor in self._outputs)
        if strides != list(YOLOV5_STRIDES):
            raise ValueError(f"expected the YOLOv5 strides {list(YOLOV5_STRIDES)}, got {strides}")
        # Fail at boot, not on the first frame, when a per-class threshold names
        # a class the graph does not have.
        channels = self._outputs[0].shape[2]
        _class_thresholds(
            self._conf_thr,
            self._class_thr,
            channels // _ANCHORS_PER_CELL - _CLASS_START,
        )

        # ``mean 0, scale 1 / 255`` of the zoo prototxt, folded into the LUT.
        self._lut = input_quantization_lut_scaled(self._runner.input_fix_point, YOLOV5_INPUT_SCALE)
        self._input_hw = self._runner.input_hw

        # Two slots so the loop can submit frame N while decoding frame N-1:
        # a DPU job owns its input and output buffers until wait() returns.
        self._slots = [self._slot() for _ in range(2)]
        self._next_slot = 0
        self._token = 0
        self._pending: dict[int, _Job] = {}
        self._decoder = QuantizedYolov5Decoder(
            [tensor.shape for tensor in self._outputs],
            [tensor.stride for tensor in self._outputs],
            [tensor.fix_point for tensor in self._outputs],
        )
        self._stage_ms: dict[str, float] = {"preprocess": 0.0, "dpu": 0.0, "decode": 0.0}

    def _slot(self) -> _Slot:
        """Allocate one input/output buffer set, with its pad border pre-quantized."""

        buffer = np.empty([1, self._input_hw[0], self._input_hw[1], 3], dtype=np.int8)
        view = buffer.reshape((*self._input_hw, 3))
        view.fill(self._lut[YOLOV5_PAD_VALUE])
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
    def fix_points(self) -> tuple[int, ...]:
        """Output ``fix_point`` of the heads, in stride order."""
        return tuple(
            tensor.fix_point for tensor in sorted(self._outputs, key=lambda item: item.stride)
        )

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

    def submit(self, rgb: NDArray[np.uint8], pass_: Pass | None = None) -> int:
        """Preprocess ``rgb`` and start the DPU without waiting for it.

        ``pass_`` is accepted for the pipeline's detector interface and ignored:
        YOLOv5 always sees the whole frame at 640x640 and reports every class,
        so the YOLOX zoom/alternation schedule does not apply (the pipeline does
        not alternate while YOLOv5 is active).

        Returns a token for :meth:`wait`. Submitting frame N and then waiting
        for frame N-1 overlaps the DPU with the CPU decode and the draw. Two
        jobs may be in flight (one per slot); a third submit raises.

        Raises:
            RuntimeError: If both slots are busy or the runner is closed.
            ValueError: If ``rgb`` is not an ``(H, W, 3)`` uint8 frame.
        """

        image = _validate_frame(rgb)
        if not self._slots:
            raise RuntimeError("the DPU runner is closed")
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
            job.slot.output_buffers, self._conf_thr, self._iou_thr, self._class_thr
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

    def close(self) -> None:
        """Release the DPU runner and the two buffer sets (model swap)."""

        if not self._slots:
            return
        self._pending.clear()
        self._slots = []
        self._runner.close()


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
    """One submitted DPU job, enough to decode its outputs in :meth:`Yolov5Detector.wait`."""

    slot: _Slot
    job_id: Any
    dpu_started: float
    scale: float
    pad: tuple[float, float]
    source_hw: tuple[int, int]
