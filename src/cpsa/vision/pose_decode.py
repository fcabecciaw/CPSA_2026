"""Pure-numpy pose decoding and letterboxing (FR-004, NFR-004).

Two decoders cover the two deployment paths:

* :func:`decode_ultralytics` consumes the decoded YOLOv8-pose output produced by
  the exported graph, shape ``(1, 4 + nc + 3K, N)`` in model-input pixels:
  ``cx, cy, w, h`` (xywh), ``nc`` class probabilities (already sigmoid), then
  ``K`` keypoints as ``x, y, conf`` (conf already sigmoid).
* :func:`decode_raw_heads` consumes the raw head tensors cut before DFL and
  pose decoding, one ``(1, H, W, C)`` NHWC float tensor per stride.  Channels
  are ordered ``[box (4 * reg_max) | class logits (nc) | keypoints (3 * K)]``.

The raw decode matches Ultralytics 8.4.203 ``Pose.kpts_decode`` and
``Detect._get_decode_boxes``.  With ``gx``/``gy`` the grid-cell indices,
``s`` the stride and ``anchor = (gx + 0.5, gy + 0.5)``:

* DFL: ``softmax`` over each side's ``reg_max`` logits, expectation with
  ``0..reg_max-1``, giving ``l, t, r, b`` in grid units;
* box (pixels): ``x1 = (anchor_x - l) * s``, ``y1 = (anchor_y - t) * s``,
  ``x2 = (anchor_x + r) * s``, ``y2 = (anchor_y + b) * s``;
* keypoints (pixels): ``x = (2 * raw_x + gx) * s``,
  ``y = (2 * raw_y + gy) * s``, ``conf = sigmoid(raw_conf)``.

:class:`QuantizedRawHeadDecoder` is the DPU fast path.  It thresholds the raw
int8 class logits before any activation, then runs DFL, sigmoid, grid and
keypoint decode only on the surviving candidates, and precomputes the grids and
scratch buffers.  Its output is identical to dequantizing the tensors and
calling :func:`decode_raw_heads`.

The golden fixtures in ``tests/fixtures/pose/`` pin this convention to the
upstream implementation (P-10).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from cpsa.vision.types import PersonPose

__all__ = [
    "QuantizedRawHeadDecoder",
    "ResizeCallable",
    "decode_raw_heads",
    "decode_ultralytics",
    "letterbox",
    "unletterbox",
]

ResizeCallable = Callable[[NDArray[np.uint8], int, int], NDArray[np.uint8]]
"""``(image, out_height, out_width) -> image`` resize, injectable into :func:`letterbox`."""

_DEFAULT_PAD_VALUE = 114


def _as_candidate_matrix(output: NDArray[np.generic], name: str) -> NDArray[np.float64]:
    """Return ``output`` as a float64 ``(channels, anchors)`` matrix."""

    array = np.asarray(output, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"{name} must have batch size 1, got {array.shape[0]}")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be (1, channels, anchors), got shape {array.shape}")
    if array.shape[1] == 0:
        raise ValueError(f"{name} has no anchors")
    return array


def _xywh_to_xyxy(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    cx, cy, width, height = boxes.T
    return np.stack(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0], axis=1
    )


def _iou(box: NDArray[np.float64], others: NDArray[np.float64]) -> NDArray[np.float64]:
    """IoU of one ``(4,)`` box against ``(n, 4)`` boxes."""

    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area = max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)
    areas = np.clip(others[:, 2] - others[:, 0], 0.0, None) * np.clip(
        others[:, 3] - others[:, 1], 0.0, None
    )
    union = area + areas - inter
    overlap: NDArray[np.float64] = np.divide(
        inter, union, out=np.zeros_like(inter), where=union > 0.0
    )
    return overlap


def _nms(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    classes: NDArray[np.intp],
    iou_thr: float,
) -> NDArray[np.intp]:
    """Class-aware greedy non-maximum suppression."""

    order = np.argsort(-scores, kind="stable").astype(np.intp)
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        if rest.size == 0:
            break
        overlap = _iou(boxes[best], boxes[rest])
        same_class = classes[rest] == classes[best]
        order = rest[~(same_class & (overlap > iou_thr))]
    return np.asarray(keep, dtype=np.intp)


def _select(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    classes: NDArray[np.intp],
    keypoints: NDArray[np.float64],
    conf_thr: float,
    iou_thr: float,
) -> list[PersonPose]:
    """Threshold, NMS and convert candidates to :class:`PersonPose`."""

    finite = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & np.isfinite(keypoints).all(axis=(1, 2))
    )
    usable = (scores > conf_thr) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    candidates = np.flatnonzero(finite & usable).astype(np.intp)
    if candidates.size == 0:
        return []
    keep = candidates[_nms(boxes[candidates], scores[candidates], classes[candidates], iou_thr)]
    return [
        PersonPose(
            box=boxes[index].copy(),
            score=float(scores[index]),
            xy=keypoints[index, :, :2].copy(),
            conf=np.clip(keypoints[index, :, 2], 0.0, 1.0),
        )
        for index in keep
    ]


def decode_ultralytics(
    output: NDArray[np.generic],
    n_kpt: int,
    conf_thr: float,
    iou_thr: float,
) -> list[PersonPose]:
    """Decode an exported Ultralytics pose output ``(1, 4 + nc + 3K, N)``.

    Args:
        output: Decoded predictions in model-input pixels: xywh boxes, class
            probabilities and ``x, y, conf`` keypoints.
        n_kpt: Number of keypoints per person (``K``).
        conf_thr: Minimum class score; keep scores strictly above it.
        iou_thr: IoU threshold for class-aware NMS.

    Returns:
        Persons sorted by descending score.
    """

    if n_kpt < 1:
        raise ValueError(f"n_kpt must be positive, got {n_kpt}")
    matrix = _as_candidate_matrix(output, "output")
    minimum = 4 + 3 * n_kpt + 1
    if matrix.shape[0] < minimum:
        raise ValueError(
            f"output has {matrix.shape[0]} channels; at least {minimum} are needed "
            f"for {n_kpt} keypoints and one class"
        )
    n_class = matrix.shape[0] - 4 - 3 * n_kpt
    boxes = _xywh_to_xyxy(matrix[0:4].T.astype(np.float64))
    class_scores = matrix[4 : 4 + n_class]
    scores = class_scores.max(axis=0)
    classes = class_scores.argmax(axis=0).astype(np.intp)
    keypoints = matrix[4 + n_class :].reshape(n_kpt, 3, -1).transpose(2, 0, 1).astype(np.float64)
    return _select(boxes, scores, classes, keypoints, conf_thr, iou_thr)


def _softmax(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=axis, keepdims=True)


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    # Clipping avoids exp overflow warnings; outside +/-500 the result is 0 or 1 to float64.
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))


def _nhwc(output: NDArray[np.generic], index: int) -> NDArray[np.float64]:
    array = np.asarray(output, dtype=np.float64)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"outputs[{index}] must have batch size 1, got {array.shape[0]}")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"outputs[{index}] must be (1, H, W, C), got shape {array.shape}")
    return array


def decode_raw_heads(
    outputs: Sequence[NDArray[np.generic]],
    strides: Sequence[int],
    n_kpt: int,
    conf_thr: float,
    iou_thr: float,
    reg_max: int = 16,
) -> list[PersonPose]:
    """Decode raw YOLOv8-pose head tensors, one NHWC tensor per stride.

    Args:
        outputs: Per-scale raw tensors shaped ``(1, H, W, C)`` (or ``(H, W, C)``)
            whose channels are ``[box (4 * reg_max) | class logits | 3 * n_kpt]``.
        strides: Feature-map stride for each output, e.g. ``(8, 16, 32)``.
        n_kpt: Number of keypoints per person.
        conf_thr: Minimum class score; keep scores strictly above it.
        iou_thr: IoU threshold for class-aware NMS.
        reg_max: DFL bins per box side.

    Returns:
        Persons sorted by descending score.
    """

    if n_kpt < 1:
        raise ValueError(f"n_kpt must be positive, got {n_kpt}")
    if reg_max < 1:
        raise ValueError(f"reg_max must be positive, got {reg_max}")
    if len(outputs) != len(strides):
        raise ValueError(f"got {len(outputs)} outputs but {len(strides)} strides")
    if len(outputs) == 0:
        raise ValueError("outputs must not be empty")

    boxes: list[NDArray[np.float64]] = []
    scores: list[NDArray[np.float64]] = []
    classes: list[NDArray[np.intp]] = []
    keypoints: list[NDArray[np.float64]] = []
    bins = np.arange(reg_max, dtype=np.float64)

    for index, (output, raw_stride) in enumerate(zip(outputs, strides, strict=True)):
        stride = int(raw_stride)
        if stride <= 0:
            raise ValueError(f"strides[{index}] must be positive, got {stride}")
        raw = _nhwc(output, index)
        height, width, channels = raw.shape
        base = 4 * reg_max + 3 * n_kpt
        if channels <= base:
            raise ValueError(
                f"outputs[{index}] has {channels} channels; at least {base + 1} are needed "
                f"for reg_max={reg_max}, {n_kpt} keypoints and one class"
            )
        n_class = channels - base

        box_logits = raw[..., : 4 * reg_max].reshape(height, width, 4, reg_max)
        distances = (_softmax(box_logits, axis=-1) * bins).sum(axis=-1) * stride
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float64),
            np.arange(width, dtype=np.float64),
            indexing="ij",
        )
        anchor_x = (grid_x + 0.5) * stride
        anchor_y = (grid_y + 0.5) * stride
        left, top, right, bottom = (
            distances[..., 0],
            distances[..., 1],
            distances[..., 2],
            distances[..., 3],
        )
        scale_boxes = np.stack(
            [anchor_x - left, anchor_y - top, anchor_x + right, anchor_y + bottom], axis=-1
        )
        boxes.append(scale_boxes.reshape(-1, 4))

        class_logits = raw[..., 4 * reg_max : 4 * reg_max + n_class].reshape(-1, n_class)
        class_scores = _sigmoid(class_logits)
        scores.append(class_scores.max(axis=1))
        classes.append(class_scores.argmax(axis=1).astype(np.intp))

        raw_keypoints = raw[..., 4 * reg_max + n_class :].reshape(height, width, n_kpt, 3)
        x = (raw_keypoints[..., 0] * 2.0 + grid_x[..., None]) * stride
        y = (raw_keypoints[..., 1] * 2.0 + grid_y[..., None]) * stride
        conf = _sigmoid(raw_keypoints[..., 2])
        keypoints.append(np.stack([x, y, conf], axis=-1).reshape(-1, n_kpt, 3))

    return _select(
        np.concatenate(boxes, axis=0),
        np.concatenate(scores, axis=0),
        np.concatenate(classes, axis=0),
        np.concatenate(keypoints, axis=0),
        conf_thr,
        iou_thr,
    )


def _class_threshold_int(conf_thr: float, fix_point: int) -> int:
    """Return the smallest int8 class logit that the fast path must decode.

    With ``q`` the raw class-logit value, the reference path scores a candidate
    with ``sigmoid(q * 2 ** -fix_point)`` and keeps ``score > conf_thr``.  Since
    sigmoid is monotonic that is ``q > logit(conf_thr) * 2 ** fix_point``, whose
    integer equivalent is ``floor(threshold) + 1``.  This returns one integer
    below it, so float rounding at a quantization step can never drop a
    candidate that the reference path keeps.  The extra candidate is removed by
    the exact float comparison in :func:`_select`, so the decoded persons are
    unchanged.
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
class _RawScale:
    """Precomputed state of one raw-head tensor for the quantized decoder."""

    height: int
    width: int
    channels: int
    stride: int
    fix_point: int
    dequant_factor: float
    n_class: int
    class_start: int
    kpt_start: int
    anchor_x: NDArray[np.float64]
    anchor_y: NDArray[np.float64]
    grid_x: NDArray[np.float64]
    grid_y: NDArray[np.float64]
    keep: NDArray[np.bool_]
    class_max: NDArray[np.int8] | None


def _build_raw_scale(
    height: int,
    width: int,
    channels: int,
    stride: int,
    fix_point: int,
    n_kpt: int,
    reg_max: int,
) -> _RawScale:
    """Build the per-stride grids, slices and scratch buffers."""

    n_class = channels - 4 * reg_max - 3 * n_kpt
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    return _RawScale(
        height=height,
        width=width,
        channels=channels,
        stride=stride,
        fix_point=fix_point,
        dequant_factor=2.0**-fix_point,
        n_class=n_class,
        class_start=4 * reg_max,
        kpt_start=4 * reg_max + n_class,
        anchor_x=((grid_x + 0.5) * stride).reshape(-1),
        anchor_y=((grid_y + 0.5) * stride).reshape(-1),
        grid_x=grid_x.reshape(-1),
        grid_y=grid_y.reshape(-1),
        keep=np.empty(height * width, dtype=np.bool_),
        class_max=np.empty(height * width, dtype=np.int8) if n_class > 1 else None,
    )


class QuantizedRawHeadDecoder:
    """Threshold-first decoder for the DPU's quantized raw pose heads.

    Behaviourally identical to dequantizing each tensor and calling
    :func:`decode_raw_heads`, but the class logits are thresholded in the int8
    domain first, so the DFL softmax, the sigmoids, the grid decode and the
    per-candidate dequantization only run on the survivors.  Grids, channel
    slices, dequantization factors and scratch buffers are precomputed here, so
    decoding a frame does not allocate them again.  The scratch buffers make one
    instance non-reentrant: call :meth:`decode` from a single thread at a time.

    Args:
        shapes: Per-output ``(H, W, C)`` NHWC shapes, in the order in which the
            DPU returns the tensors.
        strides: Feature-map stride for each output.
        fix_points: Output ``fix_point`` for each tensor; values are dequantized
            with ``value * 2 ** -fix_point``, like
            :func:`cpsa.adapters.pose.dpu.dequantize_fix_point`.
        n_kpt: Number of keypoints per person.
        reg_max: DFL bins per box side.

    Raises:
        ValueError: If shapes, strides or fix points do not line up.
    """

    def __init__(
        self,
        shapes: Sequence[tuple[int, int, int]],
        strides: Sequence[int],
        fix_points: Sequence[int],
        n_kpt: int,
        reg_max: int = 16,
    ) -> None:
        if n_kpt < 1:
            raise ValueError(f"n_kpt must be positive, got {n_kpt}")
        if reg_max < 1:
            raise ValueError(f"reg_max must be positive, got {reg_max}")
        if not shapes:
            raise ValueError("shapes must not be empty")
        if len(shapes) != len(strides) or len(shapes) != len(fix_points):
            raise ValueError(
                f"got {len(shapes)} shapes, {len(strides)} strides and {len(fix_points)} fix points"
            )

        base = 4 * reg_max + 3 * n_kpt
        scales: list[_RawScale] = []
        for index, (shape, raw_stride, fix_point) in enumerate(
            zip(shapes, strides, fix_points, strict=True)
        ):
            if len(shape) != 3:
                raise ValueError(f"shapes[{index}] must be (H, W, C), got {shape}")
            height, width, channels = (int(value) for value in shape)
            stride = int(raw_stride)
            if height <= 0 or width <= 0:
                raise ValueError(f"shapes[{index}] must be positive, got {shape}")
            if stride <= 0:
                raise ValueError(f"strides[{index}] must be positive, got {stride}")
            if channels <= base:
                raise ValueError(
                    f"shapes[{index}] has {channels} channels; at least {base + 1} are "
                    f"needed for reg_max={reg_max}, {n_kpt} keypoints and one class"
                )
            scales.append(
                _build_raw_scale(height, width, channels, stride, int(fix_point), n_kpt, reg_max)
            )

        self._scales = tuple(scales)
        self._order = tuple(
            sorted(range(len(scales)), key=lambda position: scales[position].stride)
        )
        self._bins = np.arange(reg_max, dtype=np.float64)
        self._n_kpt = n_kpt
        self._reg_max = reg_max

        self.last_decode_seconds = 0.0
        self.last_select_seconds = 0.0
        self.last_candidate_count = 0

    def decode(
        self,
        outputs: Sequence[NDArray[np.int8]],
        conf_thr: float,
        iou_thr: float,
    ) -> list[PersonPose]:
        """Decode raw int8 heads, thresholding class logits before activation.

        Args:
            outputs: Per-output int8 ``(H, W, C)`` (or ``(1, H, W, C)``) tensors in
                constructor order.
            conf_thr: Minimum class score; keep scores strictly above it.
            iou_thr: IoU threshold for class-aware NMS.

        Returns:
            Persons sorted by descending score, matching
            :func:`decode_raw_heads` on the dequantized tensors.

        Raises:
            ValueError: If an output is not int8 or does not match the
                constructor's shape.
        """

        if len(outputs) != len(self._scales):
            raise ValueError(
                f"got {len(outputs)} outputs but this decoder expects {len(self._scales)}"
            )

        started = time.perf_counter()
        boxes: list[NDArray[np.float64]] = []
        scores: list[NDArray[np.float64]] = []
        classes: list[NDArray[np.intp]] = []
        keypoints: list[NDArray[np.float64]] = []
        candidates = 0
        bins = self._bins
        reg_max = self._reg_max
        n_kpt = self._n_kpt
        box_channels = 4 * reg_max

        # Ascending stride reproduces the anchor order of decode_raw_heads.
        for position in self._order:
            scale = self._scales[position]
            raw = np.asarray(outputs[position])
            if raw.dtype != np.int8:
                raise ValueError(f"outputs[{position}] must be int8, got {raw.dtype}")
            if raw.ndim == 4 and raw.shape[0] == 1:
                raw = raw[0]
            if raw.shape != (scale.height, scale.width, scale.channels):
                raise ValueError(
                    f"outputs[{position}] must be {(scale.height, scale.width, scale.channels)}, "
                    f"got {raw.shape}"
                )

            flat = raw.reshape(scale.height * scale.width, scale.channels)
            threshold = _class_threshold_int(conf_thr, scale.fix_point)
            if scale.n_class == 1:
                np.greater_equal(flat[:, scale.class_start], threshold, out=scale.keep)
            else:
                class_max = scale.class_max
                assert class_max is not None
                np.maximum.reduce(
                    flat[:, scale.class_start : scale.class_start + scale.n_class],
                    axis=1,
                    out=class_max,
                )
                np.greater_equal(class_max, threshold, out=scale.keep)

            index = np.flatnonzero(scale.keep)
            count = index.size
            if count == 0:
                continue
            candidates += int(count)

            factor = scale.dequant_factor
            box = flat[index, :box_channels].astype(np.float64) * factor
            distances = (_softmax(box.reshape(count, 4, reg_max), axis=-1) * bins).sum(
                axis=-1
            ) * scale.stride
            anchor_x = scale.anchor_x[index]
            anchor_y = scale.anchor_y[index]
            boxes.append(
                np.stack(
                    [
                        anchor_x - distances[:, 0],
                        anchor_y - distances[:, 1],
                        anchor_x + distances[:, 2],
                        anchor_y + distances[:, 3],
                    ],
                    axis=-1,
                )
            )

            class_logits = (
                flat[index, scale.class_start : scale.class_start + scale.n_class].astype(
                    np.float64
                )
                * factor
            )
            class_scores = _sigmoid(class_logits)
            scores.append(class_scores.max(axis=1))
            classes.append(class_scores.argmax(axis=1).astype(np.intp))

            kpt = (
                flat[index, scale.kpt_start : scale.kpt_start + 3 * n_kpt].astype(np.float64)
                * factor
            ).reshape(count, n_kpt, 3)
            x = (kpt[..., 0] * 2.0 + scale.grid_x[index][:, None]) * scale.stride
            y = (kpt[..., 1] * 2.0 + scale.grid_y[index][:, None]) * scale.stride
            keypoints.append(np.stack([x, y, _sigmoid(kpt[..., 2])], axis=-1))

        self.last_decode_seconds = time.perf_counter() - started
        self.last_candidate_count = candidates
        if not boxes:
            self.last_select_seconds = 0.0
            return []

        started = time.perf_counter()
        persons = _select(
            np.concatenate(boxes, axis=0),
            np.concatenate(scores, axis=0),
            np.concatenate(classes, axis=0),
            np.concatenate(keypoints, axis=0),
            conf_thr,
            iou_thr,
        )
        self.last_select_seconds = time.perf_counter() - started
        return persons


@dataclass(frozen=True)
class _BilinearTable:
    """Precomputed source indices and weights for one resize geometry."""

    y0: NDArray[np.intp]
    y1: NDArray[np.intp]
    wy: NDArray[np.float32]
    x0: NDArray[np.intp]
    x1: NDArray[np.intp]
    wx: NDArray[np.float32]


@lru_cache(maxsize=8)
def _bilinear_table(height: int, width: int, out_height: int, out_width: int) -> _BilinearTable:
    """Indices and weights of the half-pixel-centred bilinear resize.

    The camera and model sizes are fixed for a run, so the geometry is cached
    and a frame only does the arithmetic.
    """

    y = (np.arange(out_height, dtype=np.float64) + 0.5) * (height / out_height) - 0.5
    x = (np.arange(out_width, dtype=np.float64) + 0.5) * (width / out_width) - 0.5
    y0 = np.clip(np.floor(y).astype(np.intp), 0, height - 1)
    x0 = np.clip(np.floor(x).astype(np.intp), 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    table = _BilinearTable(
        y0=y0,
        y1=y1,
        wy=np.clip(y - y0, 0.0, 1.0).astype(np.float32)[:, None, None],
        x0=x0,
        x1=x1,
        wx=np.clip(x - x0, 0.0, 1.0).astype(np.float32)[None, :, None],
    )
    for array in (table.y0, table.y1, table.wy, table.x0, table.x1, table.wx):
        array.setflags(write=False)
    return table


def _resize_bilinear(
    image: NDArray[np.uint8],
    out_height: int,
    out_width: int,
) -> NDArray[np.uint8]:
    """Resize an ``(H, W, 3)`` uint8 image with bilinear interpolation.

    The vertical pass runs first on whole rows and the horizontal pass then
    gathers within those rows, so both gathers stay cache-friendly; the geometry
    is cached, so a frame only does the arithmetic.  The pose adapters inject
    ``cv2.resize`` with ``INTER_LINEAR`` through :func:`letterbox` (about 70x
    faster per 640x480 -> 512x384 frame on the host measured here); this NumPy
    path is what runs without the ``vision`` extra.
    """

    height, width = image.shape[:2]
    if (height, width) == (out_height, out_width):
        unchanged: NDArray[np.uint8] = image.copy()
        return unchanged
    table = _bilinear_table(height, width, out_height, out_width)
    source = image.astype(np.float32)
    rows = source[table.y0] * (np.float32(1.0) - table.wy) + source[table.y1] * table.wy
    resized = rows[:, table.x0] * (np.float32(1.0) - table.wx) + rows[:, table.x1] * table.wx
    rounded: NDArray[np.uint8] = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
    return rounded


def letterbox(
    rgb: NDArray[np.uint8],
    size: tuple[int, int],
    pad_value: int = _DEFAULT_PAD_VALUE,
    *,
    resize: ResizeCallable | None = None,
) -> tuple[NDArray[np.uint8], float, tuple[float, float]]:
    """Resize with a preserved aspect ratio and pad to ``size``.

    Args:
        rgb: ``(H, W, 3)`` RGB image.
        size: Target ``(height, width)``.  This follows Ultralytics' ``imgsz``
            order.
        pad_value: Fill value for the padding.
        resize: Optional ``(image, out_height, out_width)`` resize callable,
            applied to the scaled frame before padding.  The pose adapters pass
            ``cv2.resize(..., interpolation=cv2.INTER_LINEAR)``, the call
            Ultralytics letterboxes with; when ``None`` the NumPy bilinear
            implementation is used.

    Returns:
        ``(image, scale, (pad_x, pad_y))`` where ``image`` is ``(H, W, 3)``
        uint8, ``scale`` is the uniform resize factor and the padding offsets
        are in pixels.  :func:`unletterbox` inverts the mapping.  When ``rgb``
        already has the requested size and is uint8 it is returned unchanged,
        so the common camera-size-equals-model-size case does not copy.
    """

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got shape {image.shape}")
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("rgb must not be empty")
    out_height, out_width = int(size[0]), int(size[1])
    if out_height <= 0 or out_width <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= pad_value <= 255:
        raise ValueError(f"pad_value must be in [0, 255], got {pad_value}")
    if (height, width) == (out_height, out_width) and image.dtype == np.uint8:
        return image, 1.0, (0.0, 0.0)

    scale = min(out_height / height, out_width / width)
    resized_height = max(1, min(out_height, round(height * scale)))
    resized_width = max(1, min(out_width, round(width * scale)))
    resizer = _resize_bilinear if resize is None else resize
    resized = resizer(image, resized_height, resized_width)
    canvas = np.full((out_height, out_width, 3), pad_value, dtype=np.uint8)
    top = (out_height - resized_height) // 2
    left = (out_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, float(scale), (float(left), float(top))


def unletterbox(
    persons: Sequence[PersonPose],
    scale: float,
    pad: tuple[float, float],
) -> list[PersonPose]:
    """Map boxes and keypoints from letterboxed pixels back to the original image.

    Args:
        persons: Persons in letterboxed model-input pixels.
        scale: Uniform resize factor returned by :func:`letterbox`.
        pad: ``(pad_x, pad_y)`` returned by :func:`letterbox`.
    """

    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    pad_x, pad_y = float(pad[0]), float(pad[1])
    restored: list[PersonPose] = []
    for person in persons:
        box = np.asarray(person.box, dtype=np.float64).copy()
        box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
        xy = np.asarray(person.xy, dtype=np.float64).copy()
        xy[:, 0] = (xy[:, 0] - pad_x) / scale
        xy[:, 1] = (xy[:, 1] - pad_y) / scale
        restored.append(
            PersonPose(
                box=box,
                score=float(person.score),
                xy=xy,
                conf=np.asarray(person.conf, dtype=np.float64).copy(),
            )
        )
    return restored
