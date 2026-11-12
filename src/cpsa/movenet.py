"""MoveNet single-pose on the DPU: preprocessing, decoding and the 17 keypoints (F-E2).

The Vitis AI model-zoo ``movenet`` graph is COCO-17 single-pose and stops at the
four raw heads; every post-processing step stays on the CPU. The zoo
preprocessing is a plain square resize of the whole RGB frame to ``192x192``
(no letterbox, no crop, no channel swap) followed by a scale of ``1/128``.

That scale is not cosmetic. The compiled graph takes ``[1, 192, 192, 3]`` xint8
NHWC with ``fix_point 6``, i.e. the int8 grid covers ``[-2, 2)``; the quantized
package model was calibrated for a factor-2-wide input, so the host maps
``pixel / 128`` into it, which is exactly ``int8 = round(pixel / 2)``. Feeding
raw ``0..255`` pixels saturates every pixel above 2 at ``127`` and the pose
collapses (measured against the package's own float model: 37.9 px mean
keypoint distance raw, 2.7 px at ``/128``); :func:`preprocess` and the
detector's LUT implement ``/128``.

The graph returns four heads at stride 4 (``48x48``), all xint8:

====================  ==========  =============================================
head                  fix_point   contents
====================  ==========  =============================================
keypoint heatmaps     3           ``17`` channels, pre-sigmoid logits
person center         3           ``1`` channel, pre-sigmoid logits
center regression     2           ``34`` channels, ``(dx, dy)`` per keypoint
offsets               7           ``34`` channels, sub-cell offset per keypoint
====================  ==========  =============================================

The compiled graph applies ``sigmoid`` to the heatmap and center heads in a CPU
subgraph, so the runtime applies it too (see :func:`decode`), which is what
makes :func:`decode` a pure function of the int8 head tensors.

There are two decoders with the same result:

* :func:`decode` is the reference: it takes the dequantized ``float64`` heads
  and mirrors the zoo's ``movenetDecode`` line by line. It is the parity oracle
  of the tests, not the path the loop runs.
* :func:`decode_quantized` (and :func:`decode_buffers`) is the A53 path. The
  int8 heads are used directly: the center prior multiplies the raw center
  logits (``sigmoid`` is monotonic, so the argmax is the same), one 256-entry
  table per fix point applies the sigmoid *and* drops the cells below the
  threshold, only the cells that survive are scored -- over all 17 channels at
  once, in ``float32`` -- and the regression/offset fields are read at the
  center and peak cells only. That is where the 14 ms of the board's float64
  decode went. The candidate scan reduces a channel-major copy of the 17
  heatmaps, because a numpy reduction over the ``48x48x17`` C layout runs at a
  byte per loop step and costs more than the rest of the decode; the runner
  builds the int8 views of a DPU slot once, so a frame does not rebuild them.

:class:`DpuMovenet` splits one frame into ``submit`` (preprocess and start the
DPU) and ``wait`` (block, decode, map to source pixels), mirroring
:class:`cpsa.yolox.YoloxDetector`: two buffer sets are in flight at most, so
frame N runs on the DPU while frame N-1 is decoded. :meth:`DpuMovenet.close`
releases the graph and its buffers, which is what a model swap needs (R2-G).

The graph is single-person, and with two people in view a plain image-centre
prior makes it jump between them. :class:`PersonLock` keeps it on one body: the
input of the next frame is a square :class:`Crop` around the confident
keypoints of the previous pose (the official inference recipe, so the other
people are out of the input), and the center-head prior is a Gaussian centred on
the tracked person instead of on the image. The person is given up only when
their pose score stays under the hide threshold for :data:`LEAVE_AFTER_S`, or
when a centre candidate is clearly stronger near the frame centre for
:data:`SWITCH_AFTER_S` -- never frame by frame.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
    "CROP_MARGIN",
    "CROP_SCORE",
    "LEAVE_AFTER_S",
    "LOCK_SIGMA_FRACTION",
    "MIN_CROP_FRACTION",
    "MIN_CROP_KEYPOINTS",
    "STRONGER_FACTOR",
    "SWITCH_AFTER_S",
    "Crop",
    "DpuMovenet",
    "KEYPOINTS",
    "MOVENET_CENTER_BIAS",
    "MOVENET_FEATURE_SIZE",
    "MOVENET_HM_THRESHOLD",
    "MOVENET_INPUT_SCALE",
    "MOVENET_INPUT_SIZE",
    "MOVENET_STRIDE",
    "MovenetHeads",
    "MovenetPose",
    "PersonLock",
    "QuantizedHeads",
    "center_peak",
    "center_prior",
    "crop_region",
    "decode",
    "decode_buffers",
    "decode_quantized",
    "dequantize_heads",
    "preprocess",
    "preprocess_into",
    "quantized_heads",
]

#: Model input ``(height, width)`` of the zoo MoveNet graph.
MOVENET_INPUT_SIZE = (192, 192)

#: Spatial size of the four output heads.
MOVENET_FEATURE_SIZE = 48

#: Network stride of the output heads: ``192 / 48``.
MOVENET_STRIDE = 4

#: Input scale of the zoo preprocessing: the xint8 grid (``fix_point 6``)
#: covers ``[-2, 2)``, so raw ``0..255`` pixels are divided by ``128``.
MOVENET_INPUT_SCALE = 1.0 / 128.0

#: Heatmap values below this are dropped before the keypoint search.
MOVENET_HM_THRESHOLD = 0.1

#: Distance weight added to the center prior and to the regression distance
#: (``1.8`` in the package's ``movenetDecode``/``maxPoint``).
MOVENET_CENTER_BIAS = 1.8

#: A keypoint at or above this score counts towards the crop of the next frame.
CROP_SCORE = 0.3

#: The square input of the next frame is this much wider than the body extent.
CROP_MARGIN = 1.5

#: Fewer confident keypoints than this mean the pose is lost: the crop goes back
#: to the whole frame.
MIN_CROP_KEYPOINTS = 4

#: A crop smaller than this fraction of the frame's short side is degenerate.
MIN_CROP_FRACTION = 0.1

#: Sigma of the lock Gaussian on the center head, as a fraction of the input.
LOCK_SIGMA_FRACTION = 0.15

#: A pose whose score stays below the hide threshold this long has left.
LEAVE_AFTER_S = 0.5

#: A competing centre peak must be this much stronger, for this long, to switch.
STRONGER_FACTOR = 1.5
SWITCH_AFTER_S = 1.0

#: Pose score under which the tracked person counts as gone (``movenet.smoothing.hide_score``).
LOCK_HIDE_SCORE = 0.25

#: COCO-17 keypoint order of the heatmap channels.
KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_NUM_KEYPOINTS = len(KEYPOINTS)
_KEYPOINT_INDEX = np.arange(_NUM_KEYPOINTS, dtype=np.intp)
_OFFSET_PAIRS = np.stack((2 * _KEYPOINT_INDEX, 2 * _KEYPOINT_INDEX + 1), axis=1)
"""The ``(x, y)`` column pair of each keypoint in a 34-field offset cell."""
_CELLS = MOVENET_FEATURE_SIZE * MOVENET_FEATURE_SIZE
_HEATMAP_CHANNELS = _NUM_KEYPOINTS
_CENTER_CHANNELS = 1
_REGRESSION_CHANNELS = _NUM_KEYPOINTS * 2

#: Name fragments the DPU head tensors carry, in the order :func:`decode` wants.
_HEATMAPS = "heatmaps"
_CENTERS = "centers"
_REGS = "regs"
_OFFSETS = "offsets"
_HEAD_NAMES = (_HEATMAPS, _CENTERS, _REGS, _OFFSETS)


@dataclass(frozen=True)
class MovenetPose:
    """One person's COCO-17 keypoints.

    Attributes:
        keypoints: ``(17, 3)`` float32 array of ``(x, y, score)`` in **source
            image pixels**, in :data:`KEYPOINTS` order.
        score: Mean of the 17 keypoint scores, in ``0..1``.
    """

    keypoints: NDArray[np.float32]
    score: float


@dataclass(frozen=True)
class MovenetHeads:
    """The four DPU heads, dequantized to float and shaped ``(C, H, W)``.

    Attributes:
        heatmaps: ``(17, 48, 48)`` keypoint logits (the graph applies
            ``sigmoid`` on the CPU, so these are pre-sigmoid).
        centers: ``(1, 48, 48)`` person-center logits, also pre-sigmoid.
        regs: ``(34, 48, 48)`` center-relative ``(dx, dy)`` per keypoint, in
            feature cells.
        offsets: ``(34, 48, 48)`` sub-cell ``(dx, dy)`` per keypoint.
    """

    heatmaps: NDArray[np.float64]
    centers: NDArray[np.float64]
    regs: NDArray[np.float64]
    offsets: NDArray[np.float64]

    def __post_init__(self) -> None:
        expected = (
            (_HEATMAPS, self.heatmaps, _HEATMAP_CHANNELS),
            (_CENTERS, self.centers, _CENTER_CHANNELS),
            (_REGS, self.regs, _REGRESSION_CHANNELS),
            (_OFFSETS, self.offsets, _REGRESSION_CHANNELS),
        )
        for name, head, channels in expected:
            if head.ndim != 3 or head.shape[0] != channels:
                raise ValueError(f"{name} must be ({channels}, H, W), got {head.shape}")
            if head.shape[1:] != (MOVENET_FEATURE_SIZE, MOVENET_FEATURE_SIZE):
                raise ValueError(
                    f"{name} must be {MOVENET_FEATURE_SIZE}x{MOVENET_FEATURE_SIZE}, "
                    f"got {head.shape[1:]}"
                )


@dataclass(frozen=True)
class QuantizedHeads:
    """The four DPU heads exactly as the graph writes them: int8 ``(48, 48, C)``.

    These are views of the DPU output buffers, not copies: the fast decoder of
    :func:`decode_quantized` reads the quantized values directly.

    Attributes:
        heatmaps: ``(48, 48, 17)`` keypoint logits, xint8.
        centers: ``(48, 48, 1)`` person-center logits, xint8.
        regs: ``(48, 48, 34)`` center-relative ``(dx, dy)`` per keypoint.
        offsets: ``(48, 48, 34)`` sub-cell ``(dx, dy)`` per keypoint.
        heatmaps_fix_point: ``fix_point`` of the heatmap head.
        centers_fix_point: ``fix_point`` of the center head.
        regs_fix_point: ``fix_point`` of the regression head.
        offsets_fix_point: ``fix_point`` of the offset head.
    """

    heatmaps: NDArray[np.int8]
    centers: NDArray[np.int8]
    regs: NDArray[np.int8]
    offsets: NDArray[np.int8]
    heatmaps_fix_point: int
    centers_fix_point: int
    regs_fix_point: int
    offsets_fix_point: int

    def __post_init__(self) -> None:
        expected = (
            (_HEATMAPS, self.heatmaps, _HEATMAP_CHANNELS),
            (_CENTERS, self.centers, _CENTER_CHANNELS),
            (_REGS, self.regs, _REGRESSION_CHANNELS),
            (_OFFSETS, self.offsets, _REGRESSION_CHANNELS),
        )
        for name, head, channels in expected:
            if head.dtype != np.int8:
                raise ValueError(f"{name} must be int8, got {head.dtype}")
            shape = (MOVENET_FEATURE_SIZE, MOVENET_FEATURE_SIZE, channels)
            if head.shape != shape:
                raise ValueError(f"{name} must be {shape}, got {head.shape}")


@functools.lru_cache(maxsize=1)
def center_prior() -> NDArray[np.float64]:
    """Return the ``(48, 48)`` center prior the package multiplies the center head by.

    The zoo's ``lib/data/center_weight_origin.npy`` is exactly
    ``1 / (distance_from_(24, 24) + 1.8)`` (float32 rounding), so the runtime
    rebuilds it instead of loading the file.
    """

    axis = np.arange(MOVENET_FEATURE_SIZE, dtype=np.float64) - 24.0
    distance = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)
    return 1.0 / (distance + MOVENET_CENTER_BIAS)


@dataclass(frozen=True)
class Crop:
    """A square source region resized into the model input.

    Attributes:
        x: Left edge in source pixels.
        y: Top edge in source pixels.
        side: Side of the square, in source pixels.
    """

    x: int
    y: int
    side: int

    def __post_init__(self) -> None:
        if self.side <= 0:
            raise ValueError(f"a crop needs a positive side, got {self.side}")


def crop_region(
    points: NDArray[np.float64],
    frame_size: tuple[int, int],
) -> Crop | None:
    """Return the square crop around ``points``, or ``None`` for the whole frame.

    This is the official MoveNet inference recipe: the next frame is cropped
    around the pose just decoded, with :data:`CROP_MARGIN` of margin, and that
    square is resized into the ``192x192`` input, so the tracked body fills the
    input and the other people in the frame are cut away.

    Args:
        points: The confident keypoint coordinates, ``(N, 2)`` in source pixels.
        frame_size: Source frame ``(height, width)`` in pixels.

    Returns:
        A square crop clamped inside the frame, or ``None`` when the pose is too
        small, too large or too sparse to be worth cropping.
    """

    height, width = int(frame_size[0]), int(frame_size[1])
    if height <= 0 or width <= 0:
        return None
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < MIN_CROP_KEYPOINTS:
        return None
    finite = np.isfinite(points[:, :2]).all(axis=1)
    if not finite.all():
        points = points[finite]
        if len(points) < MIN_CROP_KEYPOINTS:
            return None
    left = float(np.min(points[:, 0]))
    right = float(np.max(points[:, 0]))
    top = float(np.min(points[:, 1]))
    bottom = float(np.max(points[:, 1]))
    side = int(math.ceil(max(right - left, bottom - top) * CROP_MARGIN))
    short_side = min(width, height)
    if side >= short_side or side < MIN_CROP_FRACTION * short_side:
        return None
    x = int(round(0.5 * (left + right) - 0.5 * side))
    y = int(round(0.5 * (top + bottom) - 0.5 * side))
    return Crop(
        x=min(max(x, 0), width - side),
        y=min(max(y, 0), height - side),
        side=side,
    )


def _validate_frame(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return ``rgb`` after checking the shape and dtype the graph needs."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {image.dtype}")
    return image


def preprocess(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Resize one RGB frame to the graph's ``192x192`` input.

    The zoo preprocessing is a plain square resize of the whole frame: no
    letterbox, no padding and no channel swap. The ``/128`` input scale is
    applied by :func:`preprocess_into` through the quantization table.

    Args:
        rgb: ``(H, W, 3)`` RGB uint8 frame.

    Returns:
        ``(192, 192, 3)`` RGB uint8 model input.

    Raises:
        ValueError: If ``rgb`` is not an ``(H, W, 3)`` uint8 frame or is empty.
    """

    image = _validate_frame(rgb)
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("rgb must not be empty")
    if (image.shape[0], image.shape[1]) == MOVENET_INPUT_SIZE:
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(cv2.resize(image, (MOVENET_INPUT_SIZE[1], MOVENET_INPUT_SIZE[0])))


def preprocess_into(
    rgb: NDArray[np.uint8],
    lut: NDArray[np.int8],
    out: NDArray[np.int8],
    crop: Crop | None = None,
) -> None:
    """Resize and quantize one frame (or one crop of it) into a DPU input view.

    Args:
        rgb: ``(H, W, 3)`` RGB uint8 frame.
        lut: 256-entry int8 table, as built by
            :func:`cpsa.dpu.input_quantization_lut_scaled` with
            :data:`MOVENET_INPUT_SCALE`.
        out: ``(192, 192, 3)`` C-contiguous int8 buffer to fill.
        crop: The square source region to feed the model, or ``None`` for the
            whole frame. A crop is what :class:`PersonLock` derives from the
            previous pose; it is smaller than the frame, so it is also cheaper
            to resize.

    Raises:
        ValueError: If the frame or ``out`` do not match the model input, or the
            crop does not fit the frame.
    """

    image = _validate_frame(rgb)
    expected = (*MOVENET_INPUT_SIZE, 3)
    if out.shape != expected or out.dtype != np.int8:
        raise ValueError(f"out must be a {expected} int8 array, got {out.shape} {out.dtype}")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("out must be C-contiguous")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("rgb must not be empty")
    if lut.shape != (256,):
        raise ValueError(f"lut must have 256 entries, got {lut.shape}")
    source = image
    if crop is not None:
        if (
            crop.x < 0
            or crop.y < 0
            or crop.x + crop.side > image.shape[1]
            or crop.y + crop.side > image.shape[0]
        ):
            raise ValueError(
                f"crop {crop} does not fit the {image.shape[1]}x{image.shape[0]} frame"
            )
        source = image[crop.y : crop.y + crop.side, crop.x : crop.x + crop.side]

    # Resize straight into the DPU input view and quantize it in place: no
    # intermediate 192x192 frame is allocated or copied.
    view = out.view(np.uint8)
    cv2.resize(source, (MOVENET_INPUT_SIZE[1], MOVENET_INPUT_SIZE[0]), dst=view)
    cv2.LUT(view, lut.view(np.uint8), dst=view)


def dequantize_heads(
    buffers: Sequence[NDArray[np.int8]],
    tensors: Sequence[OutputTensor],
    order: Sequence[str] = _HEAD_NAMES,
) -> MovenetHeads:
    """Dequantize one set of DPU output buffers into :class:`MovenetHeads`.

    Args:
        buffers: int8 buffers ``(1, 48, 48, C)``, one per output tensor.
        tensors: the graph's output metadata, in the same order as ``buffers``.
        order: which head each entry of ``tensors`` holds, one of
            ``"heatmaps"``, ``"centers"``, ``"regs"``, ``"offsets"``.

    Returns:
        The four heads as ``(C, 48, 48)`` float arrays.

    Raises:
        ValueError: If a buffer does not match its tensor or a head repeats.
    """

    if len(buffers) != len(tensors) or len(tensors) != len(order):
        raise ValueError(
            f"expected {len(order)} buffers and tensors, got {len(buffers)} and {len(tensors)}"
        )
    if sorted(order) != sorted(_HEAD_NAMES):
        raise ValueError(f"order must be a permutation of {list(_HEAD_NAMES)}, got {list(order)}")
    heads: dict[str, NDArray[np.float64]] = {}
    for name, buffer, tensor in zip(order, buffers, tensors, strict=True):
        if tuple(buffer.shape[1:]) != tensor.shape:
            raise ValueError(
                f"{name} buffer shape {tuple(buffer.shape[1:])} does not match "
                f"tensor shape {tensor.shape}"
            )
        heads[name] = np.ascontiguousarray(
            dequantize_fix_point(buffer.reshape(tensor.shape), tensor.fix_point).transpose(2, 0, 1)
        )
    return MovenetHeads(
        heatmaps=heads[_HEATMAPS],
        centers=heads[_CENTERS],
        regs=heads[_REGS],
        offsets=heads[_OFFSETS],
    )


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the logistic function of ``values``, without overflow warnings."""

    return 1.0 / (1.0 + np.exp(-values))


def decode(
    heads: MovenetHeads,
    size: tuple[int, int] = MOVENET_INPUT_SIZE,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    prior: NDArray[np.float64] | NDArray[np.float32] | None = None,
) -> MovenetPose:
    """Decode the four head tensors into one person's keypoints.

    This mirrors the zoo's ``movenetDecode(..., mode='output')`` exactly:

    1. ``sigmoid`` on the heatmaps and the center head, then drop heatmap
       values below :data:`MOVENET_HM_THRESHOLD`;
    2. the person center is the ``argmax`` of ``centers * center_prior()``;
    3. per keypoint, the regression field at the center cell gives a target
       cell; the keypoint heatmap divided by ``distance_to_target + 1.8`` is
       maximised, which is the package's regression-field weighting;
    4. the peak cell plus the offset field (both divided by 48) gives the
       normalised keypoint, and a keypoint whose heatmap value is below the
       threshold keeps a zero score.

    Args:
        heads: The four dequantized heads; heatmaps and centers are logits.
        size: Size ``(height, width)`` of the source region the keypoints are
            mapped to — the whole frame, or the :class:`Crop` the input was
            taken from. The input is a square resize of that region, so the
            normalised coordinates scale by this size.
        origin: Source-pixel origin of that region: the decoded keypoints are
            offset by it, which maps a crop back onto the frame.
        prior: Centre-head prior to use instead of :func:`center_prior`; the
            :class:`PersonLock` passes its Gaussian here.

    Returns:
        The pose in source pixels, in :data:`KEYPOINTS` order.

    Raises:
        ValueError: If ``size`` is not positive.
    """

    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive, got {size}")
    origin_x, origin_y = float(origin[0]), float(origin[1])

    heatmaps = _sigmoid(heads.heatmaps)
    heatmaps = np.where(heatmaps < MOVENET_HM_THRESHOLD, 0.0, heatmaps)
    centers = _sigmoid(heads.centers[0])
    regs = heads.regs
    offsets = heads.offsets

    grid_y, grid_x = np.mgrid[0:MOVENET_FEATURE_SIZE, 0:MOVENET_FEATURE_SIZE]

    weights = center_prior() if prior is None else prior
    center_index = int(np.argmax(centers * weights))
    center_x = center_index % MOVENET_FEATURE_SIZE
    center_y = center_index // MOVENET_FEATURE_SIZE

    keypoints = np.zeros((_NUM_KEYPOINTS, 3), dtype=np.float64)
    for index in range(_NUM_KEYPOINTS):
        # The package casts (value + 0.5) with ``astype(np.int32)``, which
        # truncates towards zero for negative regressions.
        target_x = int(regs[index * 2, center_y, center_x] + 0.5) + center_x
        target_y = int(regs[index * 2 + 1, center_y, center_x] + 0.5) + center_y
        distance = np.sqrt((grid_x - target_x) ** 2 + (grid_y - target_y) ** 2)
        weighted = heatmaps[index] / (distance + MOVENET_CENTER_BIAS)
        peak = int(np.argmax(weighted))
        peak_x = min(max(peak % MOVENET_FEATURE_SIZE, 0), MOVENET_FEATURE_SIZE - 1)
        peak_y = min(max(peak // MOVENET_FEATURE_SIZE, 0), MOVENET_FEATURE_SIZE - 1)
        offset_x = offsets[index * 2, peak_y, peak_x]
        offset_y = offsets[index * 2 + 1, peak_y, peak_x]
        keypoints[index, 0] = (peak_x + offset_x) / MOVENET_FEATURE_SIZE * width + origin_x
        keypoints[index, 1] = (peak_y + offset_y) / MOVENET_FEATURE_SIZE * height + origin_y
        keypoints[index, 2] = heatmaps[index, peak_y, peak_x]

    return MovenetPose(
        keypoints=keypoints.astype(np.float32),
        score=float(np.mean(keypoints[:, 2])),
    )


def quantized_heads(
    buffers: Sequence[NDArray[np.int8]],
    tensors: Sequence[OutputTensor],
    order: Sequence[str] = _HEAD_NAMES,
) -> QuantizedHeads:
    """View one set of DPU output buffers as int8 ``(48, 48, C)`` heads.

    The buffers stay owned by the caller (the runner's slots): every head is a
    view, so a decode costs no dequantization and no transpose.

    Args:
        buffers: int8 buffers ``(1, 48, 48, C)``, one per output tensor.
        tensors: the graph's output metadata, in the same order as ``buffers``.
        order: which head each entry of ``tensors`` holds, one of
            ``"heatmaps"``, ``"centers"``, ``"regs"``, ``"offsets"``.

    Returns:
        The four heads as int8 NHWC views with their fix points.

    Raises:
        ValueError: If a buffer does not match its tensor or a head repeats.
    """

    if len(buffers) != len(tensors) or len(tensors) != len(order):
        raise ValueError(
            f"expected {len(order)} buffers and tensors, got {len(buffers)} and {len(tensors)}"
        )
    if sorted(order) != sorted(_HEAD_NAMES):
        raise ValueError(f"order must be a permutation of {list(_HEAD_NAMES)}, got {list(order)}")
    heads: dict[str, tuple[NDArray[np.int8], int]] = {}
    for name, buffer, tensor in zip(order, buffers, tensors, strict=True):
        if tuple(buffer.shape[1:]) != tensor.shape:
            raise ValueError(
                f"{name} buffer shape {tuple(buffer.shape[1:])} does not match "
                f"tensor shape {tensor.shape}"
            )
        if buffer.dtype != np.int8:
            raise ValueError(f"{name} buffer must be int8, got {buffer.dtype}")
        heads[name] = (buffer.reshape(tensor.shape), int(tensor.fix_point))
    heatmaps, heatmaps_fix_point = heads[_HEATMAPS]
    centers, centers_fix_point = heads[_CENTERS]
    regs, regs_fix_point = heads[_REGS]
    offsets, offsets_fix_point = heads[_OFFSETS]
    return QuantizedHeads(
        heatmaps=heatmaps,
        centers=centers,
        regs=regs,
        offsets=offsets,
        heatmaps_fix_point=heatmaps_fix_point,
        centers_fix_point=centers_fix_point,
        regs_fix_point=regs_fix_point,
        offsets_fix_point=offsets_fix_point,
    )


def center_peak(
    heads: QuantizedHeads,
    prior: NDArray[np.float32] | NDArray[np.float64] | None = None,
) -> tuple[int, int]:
    """Return the winning center cell of ``argmax(sigmoid(centers) * prior)``.

    The sigmoid comes from the fix point's 256-entry table, so the argmax is the
    reference decoder's even under a prior as peaky as the person lock's; the
    raw int8 logit times the prior would not be, because the sigmoid does not
    commute with a non-uniform weight.

    Args:
        heads: The int8 heads, as :func:`quantized_heads` builds them.
        prior: The ``(48, 48)`` center prior, or ``None`` for the image-centre
            one.

    Returns:
        ``(column, row)`` in feature cells.
    """

    weights = _center_prior_f32() if prior is None else prior
    table = _sigmoid_table(heads.centers_fix_point)
    probabilities = table[heads.centers[:, :, 0].view(np.uint8)]
    weighted = probabilities * weights
    flat = int(np.argmax(weighted))
    return flat % MOVENET_FEATURE_SIZE, flat // MOVENET_FEATURE_SIZE


def decode_buffers(
    buffers: Sequence[NDArray[np.int8]],
    tensors: Sequence[OutputTensor],
    *,
    size: tuple[int, int] = MOVENET_INPUT_SIZE,
    order: Sequence[str] = _HEAD_NAMES,
) -> MovenetPose:
    """Decode one set of DPU output buffers directly (see :func:`decode_quantized`)."""

    return decode_quantized(quantized_heads(buffers, tensors, order), size=size)


def decode_quantized(
    heads: QuantizedHeads,
    size: tuple[int, int] = MOVENET_INPUT_SIZE,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    prior: NDArray[np.float32] | NDArray[np.float64] | None = None,
) -> MovenetPose:
    """Decode the int8 heads directly: the A53 path of :func:`decode`.

    The result matches the reference decoder, with the same steps done in the
    integer domain and over all 17 keypoints at once:

    1. ``argmax(sigmoid(centers) * prior)`` is evaluated exactly as the reference
       does, with the fix point's 256-entry sigmoid table in place of the
       float64 ``sigmoid`` (see :func:`center_peak`);
    2. the ``float32`` sigmoid table of the heatmap fix point already zeroes
       every value below :data:`MOVENET_HM_THRESHOLD`, so a cell below the
       table's cutoff cannot win the weighted search and is not scored at all;
    3. the regression and offset fields are dequantized at the center cell and
       at the 17 peak cells only, not over the whole ``48x48x34`` heads.

    Args:
        heads: The four int8 heads, as :func:`quantized_heads` builds them.
        size: Size ``(height, width)`` of the source region the keypoints are
            mapped to, as in :func:`decode`.
        origin: Source-pixel origin of that region, as in :func:`decode`.
        prior: Centre-head prior to use instead of the image-centre one.

    Returns:
        The pose in source pixels, in :data:`KEYPOINTS` order.

    Raises:
        ValueError: If ``size`` is not positive.
    """

    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive, got {size}")
    origin_x, origin_y = float(origin[0]), float(origin[1])

    # The reference decoder multiplies the sigmoid of the centers by the prior,
    # so the int8 path applies the fix point's sigmoid table first.
    center_x, center_y = center_peak(heads, prior)

    # The package casts ``value + 0.5`` to int32, which truncates towards zero;
    # ``reshape(17, 2)`` turns the 34-field regression into an (x, y) pair.
    regression = heads.regs[center_y, center_x].reshape(_NUM_KEYPOINTS, 2)
    regression_scale = np.float32(2.0**-heads.regs_fix_point)
    targets = (regression * regression_scale + np.float32(0.5)).astype(np.int32)
    targets += (center_x, center_y)
    target_x = targets[:, 0]
    target_y = targets[:, 1]

    lut, cutoff = _sigmoid_lut(heads.heatmaps_fix_point)
    # The DPU writes the heatmaps as C-contiguous ``48x48x17``, and a numpy
    # reduction over that 17-byte inner axis runs at about a byte per loop step:
    # on the host that is 48 us for a frame, more than the whole decode. The
    # channel-major copy reduces 48 contiguous bytes at a time instead (10 us
    # with the copy), and only the cells that survive in some channel's peak
    # search are kept: a cell under the cutoff holds zero in the table, so it can
    # never win ``value / (distance + bias)``.
    planes = np.ascontiguousarray(heads.heatmaps.transpose(2, 0, 1))
    cells = np.flatnonzero(planes.max(axis=0).ravel() >= cutoff)
    cells_y, cells_x = np.divmod(cells, MOVENET_FEATURE_SIZE)
    peaks = np.zeros(_NUM_KEYPOINTS, dtype=np.intp)
    scores = np.zeros(_NUM_KEYPOINTS, dtype=np.float32)
    if cells.size:
        values = lut[heads.heatmaps.reshape(_CELLS, _NUM_KEYPOINTS)[cells]]
        # ``float32`` throughout and in place after the distance: the scoring
        # pass allocates the two deltas and their length, nothing else.
        delta_x = np.subtract(cells_x[:, None], target_x[None, :], dtype=np.float32)
        delta_y = np.subtract(cells_y[:, None], target_y[None, :], dtype=np.float32)
        weighted = np.hypot(delta_x, delta_y)
        weighted += np.float32(MOVENET_CENTER_BIAS)
        np.divide(values, weighted, out=weighted)
        best = weighted.argmax(axis=0)
        scores = values[best, _KEYPOINT_INDEX]
        # A keypoint with no cell above the threshold scores zero and peaks on
        # cell (0, 0), exactly as the reference's argmax over an all-zero plane:
        # zeroing its flat cell is what both the offset read and the score need.
        peaks = cells[best]
        peaks *= scores > 0.0

    # The offset field is read at the 17 peak cells only, one (x, y) pair per
    # keypoint, by indexing the 34 columns of each cell with the pair table.
    peak_y, peak_x = np.divmod(peaks, MOVENET_FEATURE_SIZE)
    offsets = heads.offsets[peak_y[:, None], peak_x[:, None], _OFFSET_PAIRS] * np.float32(
        2.0**-heads.offsets_fix_point
    )
    coords = offsets
    coords[:, 0] += peak_x
    coords[:, 1] += peak_y
    np.multiply(
        coords,
        (width / MOVENET_FEATURE_SIZE, height / MOVENET_FEATURE_SIZE),
        out=coords,
    )
    coords[:, 0] += origin_x
    coords[:, 1] += origin_y
    keypoints = np.empty((_NUM_KEYPOINTS, 3), dtype=np.float32)
    keypoints[:, :2] = coords
    keypoints[:, 2] = scores
    return MovenetPose(keypoints=keypoints, score=float(scores.mean()))


@functools.lru_cache(maxsize=1)
def _center_prior_f32() -> NDArray[np.float32]:
    """Return :func:`center_prior` as a read-only ``float32`` table."""

    prior = center_prior().astype(np.float32)
    prior.flags.writeable = False
    return prior


LockPrior = NDArray[np.float32]


def _lock_prior(center_x: float, center_y: float) -> LockPrior:
    """Return the ``(48, 48)`` Gaussian prior centred on a tracked person.

    The image-centre prior of :func:`center_prior` alone lets a stronger person
    anywhere in the input win the centre argmax, which is what makes the
    single-person graph jump when two people are in view. The lock prior falls
    off with :data:`LOCK_SIGMA_FRACTION` of the input instead, so the person
    already being tracked keeps the centre unless it really disappears.

    The Gaussian is separable: two ``exp`` calls over 48 values and one outer
    product cost a few microseconds, where one ``exp`` over the 48x48 grid costs
    several times that on the A53.
    """

    sigma = LOCK_SIGMA_FRACTION * MOVENET_FEATURE_SIZE
    scale = np.float32(1.0 / (2.0 * sigma * sigma))
    axis = np.arange(MOVENET_FEATURE_SIZE, dtype=np.float32)
    horizontal = np.exp(-np.square(axis - np.float32(center_x)) * scale)
    vertical = np.exp(-np.square(axis - np.float32(center_y)) * scale)
    prior: LockPrior = cast(LockPrior, np.multiply.outer(vertical, horizontal))
    return prior


class PersonLock:
    """Keep the single-person decode on the person it is already tracking.

    The lock owns the crop of the next input and the center prior of the next
    decode. :meth:`observe` runs once per decoded pose and decides what the next
    frame sees:

    * while the pose is healthy, the crop follows the confident keypoints and
      the prior follows their centre;
    * a pose under ``hide_score`` for more than :data:`LEAVE_AFTER_S` drops the
      lock, so the next frame is the whole frame with the image-centre prior;
    * a centre candidate that is :data:`STRONGER_FACTOR` stronger than the
      tracked one near the frame centre switches the lock only after
      :data:`SWITCH_AFTER_S`, so the two people can never alternate frame by
      frame.

    Args:
        hide_score: Pose score under which the person counts as gone; the
            pipeline passes ``movenet.smoothing.hide_score``.
    """

    def __init__(self, hide_score: float = LOCK_HIDE_SCORE) -> None:
        self._hide_score = float(hide_score)
        self._crop: Crop | None = None
        self._center: tuple[float, float] | None = None
        self._lost_at: float | None = None
        self._switch_at: float | None = None

    @property
    def crop(self) -> Crop | None:
        """The square source region the next input is taken from, if any."""
        return self._crop

    @property
    def center(self) -> tuple[float, float] | None:
        """The tracked person's centre in source pixels, if tracked."""
        return self._center

    @property
    def locked(self) -> bool:
        """Whether a person is being tracked."""
        return self._center is not None

    def reset(self) -> None:
        """Forget the tracked person: the next frame is the whole frame again."""
        self._crop = None
        self._center = None
        self._lost_at = None
        self._switch_at = None

    def prior(
        self,
        crop: Crop | None,
        source_hw: tuple[int, int],
    ) -> LockPrior:
        """The centre-head prior of the input built from ``crop``.

        Args:
            crop: The region the input is taken from, or ``None`` for the whole
                frame.
            source_hw: Source frame ``(height, width)`` in pixels.

        Returns:
            The Gaussian on the tracked person, or the image-centre prior when
            nothing is tracked.
        """

        height, width = int(source_hw[0]), int(source_hw[1])
        if self._center is None or height <= 0 or width <= 0:
            return _center_prior_f32()
        center_x, center_y = self._center
        if crop is None:
            # The whole frame is resized anisotropically to the square input.
            feature_x = center_x / width * MOVENET_FEATURE_SIZE
            feature_y = center_y / height * MOVENET_FEATURE_SIZE
        else:
            feature_x = (center_x - crop.x) / crop.side * MOVENET_FEATURE_SIZE
            feature_y = (center_y - crop.y) / crop.side * MOVENET_FEATURE_SIZE
        return _lock_prior(feature_x, feature_y)

    def frame_center_prior(
        self,
        crop: Crop | None,
        source_hw: tuple[int, int],
    ) -> LockPrior:
        """The prior of "the person nearest the frame centre", in this input.

        With the whole frame of the graph that is the package's image-centre
        prior; with a crop it is the frame centre mapped into the crop, so the
        two priors the switch hysteresis compares stay in one input space.

        Args:
            crop: The region the input is taken from, or ``None`` for the whole
                frame.
            source_hw: Source frame ``(height, width)`` in pixels.

        Returns:
            The prior of the frame centre, or the image-centre prior for the
            whole frame.
        """

        height, width = int(source_hw[0]), int(source_hw[1])
        if crop is None or height <= 0 or width <= 0:
            return _center_prior_f32()
        feature_x = (0.5 * width - crop.x) / crop.side * MOVENET_FEATURE_SIZE
        feature_y = (0.5 * height - crop.y) / crop.side * MOVENET_FEATURE_SIZE
        return _lock_prior(feature_x, feature_y)

    def observe(
        self,
        pose: MovenetPose,
        now: float,
        source_hw: tuple[int, int],
        *,
        competitor: bool = False,
    ) -> None:
        """Update the lock from the pose just decoded.

        Args:
            pose: The decoded pose, already mapped to source pixels.
            now: Monotonic seconds; the leave and switch timers use it.
            source_hw: Source frame ``(height, width)`` in pixels.
            competitor: Whether a centre candidate near the frame centre is
                clearly stronger than the tracked one this frame.
        """

        keypoints = np.asarray(pose.keypoints, dtype=np.float64)
        confident = keypoints[:, 2] >= CROP_SCORE
        confident &= np.isfinite(keypoints[:, :2]).all(axis=1)
        if float(pose.score) < self._hide_score or int(confident.sum()) < MIN_CROP_KEYPOINTS:
            # Hold the last crop for a moment: a one-frame score dip is the same
            # dropout the keypoint hysteresis rides out.
            if self._lost_at is None:
                self._lost_at = now
            elif now - self._lost_at > LEAVE_AFTER_S:
                self.reset()
            return
        self._lost_at = None

        points = keypoints[confident, :2]
        self._center = (
            0.5 * float(np.min(points[:, 0]) + np.max(points[:, 0])),
            0.5 * float(np.min(points[:, 1]) + np.max(points[:, 1])),
        )
        self._crop = crop_region(points, source_hw)
        if not competitor:
            self._switch_at = None
        elif self._switch_at is None:
            self._switch_at = now
        elif now - self._switch_at > SWITCH_AFTER_S:
            # The other person has held the centre for a full second: let the
            # next full frame pick the person nearest the frame centre.
            self.reset()


@functools.cache
def _sigmoid_table(fix_point: int) -> NDArray[np.float32]:
    """Return a fix point's 256-entry sigmoid table, without the threshold.

    Entry ``u`` is ``sigmoid(signed(u) * 2 ** -fix_point)`` with the int8 byte
    order folded in (``u >= 128`` addresses a negative value), exactly as the
    reference decoder computes it in float64. Unlike :func:`_sigmoid_lut` the
    table keeps the small values: the center head is not thresholded, and its
    sigmoid is what the prior has to be multiplied by.
    """

    signed = np.arange(256, dtype=np.int16)
    signed[signed >= 128] -= 256
    probabilities = _sigmoid(signed.astype(np.float64) * (2.0 ** -int(fix_point)))
    table = probabilities.astype(np.float32)
    table.flags.writeable = False
    return table


@functools.cache
def _sigmoid_lut(fix_point: int) -> tuple[NDArray[np.float32], int]:
    """Return a fix point's 256-entry sigmoid table and its threshold cutoff.

    Entry ``u`` is ``sigmoid(signed(u) * 2 ** -fix_point)`` with the int8 byte
    order folded in (``u >= 128`` addresses a negative value), and every value
    below :data:`MOVENET_HM_THRESHOLD` is stored as zero. One ``take`` on the
    uint8 view of a head therefore applies the sigmoid and the threshold in a
    single pass.

    The cutoff is the smallest int8 value whose table entry is not zero: a
    cell whose raw logit is below it can never win the weighted peak search,
    which is what lets the decoder ignore the background grid.
    """

    table = _sigmoid_table(fix_point)
    probabilities = np.where(table < MOVENET_HM_THRESHOLD, np.float32(0.0), table)
    order = np.concatenate([np.arange(128, 256), np.arange(0, 128)])
    cutoff_index = int(order[np.argmax(probabilities[order] > 0.0)])
    probabilities.flags.writeable = False
    return probabilities, cutoff_index if cutoff_index < 128 else cutoff_index - 256


class DpuMovenet:
    """Run one compiled MoveNet ``.xmodel`` graph on the KV260 DPU.

    Args:
        xmodel_path: Path to the compiled ``.xmodel``.
        hide_score: Pose score under which the tracked person counts as gone;
            the pipeline passes ``movenet.smoothing.hide_score``.

    Raises:
        RuntimeError: If the ``xir``/``vart`` Python bindings are unavailable.
        FileNotFoundError: If ``xmodel_path`` does not exist.
        ValueError: If the graph is not the expected MoveNet graph: one DPU
            subgraph, one ``192x192x3`` int8 input and the four stride-4 heads.
    """

    def __init__(self, xmodel_path: str | Path, *, hide_score: float = LOCK_HIDE_SCORE) -> None:
        self._runner = DpuRunner(xmodel_path, min_channels=1)
        input_hw = self._runner.input_hw
        if input_hw != MOVENET_INPUT_SIZE:
            raise ValueError(f"expected a {MOVENET_INPUT_SIZE} input, got {input_hw}")
        outputs = self._runner.outputs
        if any(tensor.stride != MOVENET_STRIDE for tensor in outputs):
            strides = [tensor.stride for tensor in outputs]
            raise ValueError(f"expected every head at stride {MOVENET_STRIDE}, got {strides}")
        self._order = self._classify(outputs)
        self._outputs = outputs
        self._input_hw = input_hw
        self._lut = input_quantization_lut_scaled(self._runner.input_fix_point, MOVENET_INPUT_SCALE)

        # Two slots so the loop can submit frame N while decoding frame N-1:
        # a DPU job owns its input and output buffers until wait() returns.
        self._slots = [self._slot() for _ in range(2)]
        self._next_slot = 0
        self._token = 0
        self._pending: dict[int, _Job] = {}
        self._lock = PersonLock(hide_score)
        self._stage_ms: dict[str, float] = {
            "preprocess": 0.0,
            "dpu": 0.0,
            "decode": 0.0,
            "lock": 0.0,
        }

    @property
    def lock(self) -> PersonLock:
        """The person lock that keeps the decode on the tracked body."""
        return self._lock

    @staticmethod
    def _classify(outputs: Sequence[OutputTensor]) -> tuple[str, ...]:
        """Return which head each output tensor holds, from the tensor names.

        Raises:
            ValueError: If a head is missing, repeated or has the wrong shape.
        """

        expected = {
            _HEATMAPS: _HEATMAP_CHANNELS,
            _CENTERS: _CENTER_CHANNELS,
            _REGS: _REGRESSION_CHANNELS,
            _OFFSETS: _REGRESSION_CHANNELS,
        }
        if len(outputs) != len(_HEAD_NAMES):
            raise ValueError(f"expected {len(_HEAD_NAMES)} DPU heads, got {len(outputs)}")
        order: dict[int, str] = {}
        for index, tensor in enumerate(outputs):
            name = tensor.name.lower()
            matches = [head for head in _HEAD_NAMES if head in name]
            if len(matches) != 1:
                raise ValueError(f"cannot tell which MoveNet head output {index} is: {tensor.name}")
            head = matches[0]
            if head in order.values():
                raise ValueError(f"two outputs claim the {head} head: {tensor.name}")
            if tensor.shape[2] != expected[head]:
                raise ValueError(
                    f"the {head} head must have {expected[head]} channels, "
                    f"got {tensor.shape[2]} in {tensor.name}"
                )
            order[index] = head
        return tuple(order[index] for index in range(len(outputs)))

    def _slot(self) -> _Slot:
        """Allocate one input/output buffer set and the int8 views of its heads.

        The views are built once per slot, not per frame: a slot's output
        buffers are the same arrays for the life of the runner, so the decoder
        can read the new values without rebuilding (and revalidating) the four
        head views every frame.
        """

        buffer = np.empty([1, *self._input_hw, 3], dtype=np.int8)
        outputs = [np.empty([1, *tensor.shape], dtype=np.int8) for tensor in self._outputs]
        return _Slot(
            input_buffer=buffer,
            input_view=buffer.reshape((*self._input_hw, 3)),
            output_buffers=outputs,
            heads=quantized_heads(outputs, self._outputs, self._order),
        )

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input ``(height, width)``."""
        return self._input_hw

    @property
    def stage_ms(self) -> dict[str, float]:
        """Per-stage timings of the last :meth:`submit`/:meth:`wait`, in milliseconds."""

        return dict(self._stage_ms)

    @property
    def pending_jobs(self) -> int:
        """DPU jobs submitted and not yet waited for."""
        return len(self._pending)

    def submit(self, rgb: NDArray[np.uint8]) -> int:
        """Preprocess ``rgb`` and start the DPU without waiting for it.

        The input is taken from the person lock's crop, with the lock's centre
        prior, so the graph keeps looking at the body it already tracks.

        Returns a token for :meth:`wait`. Submitting frame N and then waiting
        for frame N-1 overlaps the DPU with the CPU decode. Two jobs may be in
        flight (one per slot); a third submit raises.

        Raises:
            RuntimeError: If both slots are busy.
            ValueError: If ``rgb`` is not an ``(H, W, 3)`` uint8 frame.
        """

        image = _validate_frame(rgb)
        if not self._slots:
            raise RuntimeError("the DPU runner is closed")
        if len(self._pending) >= len(self._slots):
            raise RuntimeError("both DPU slots are busy; call wait() before submitting again")
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        source_hw = (int(image.shape[0]), int(image.shape[1]))

        lock_started = time.perf_counter()
        crop = _fitted_crop(self._lock.crop, source_hw)
        prior = self._lock.prior(crop, source_hw)
        self._stage_ms["lock"] = (time.perf_counter() - lock_started) * 1000.0

        started = time.perf_counter()
        preprocess_into(image, self._lut, slot.input_view, crop)
        self._stage_ms["preprocess"] = (time.perf_counter() - started) * 1000.0

        dpu_started = time.perf_counter()
        job_id = self._runner.execute([slot.input_buffer], slot.output_buffers)
        self._token += 1
        token = self._token
        self._pending[token] = _Job(
            slot=slot,
            job_id=job_id,
            dpu_started=dpu_started,
            source_hw=source_hw,
            crop=crop,
            prior=prior,
        )
        return token

    def wait(self, token: int) -> MovenetPose:
        """Wait for a job from :meth:`submit` and decode it into a pose.

        The pose is mapped back onto the source frame, then handed to the
        person lock, which chooses the crop and the prior of the next frame.

        Raises:
            ValueError: If ``token`` is not a pending job.
        """

        job = self._pending.pop(token, None)
        if job is None:
            raise ValueError(f"no pending DPU job with token {token}")
        self._runner.wait(job.job_id)
        self._stage_ms["dpu"] = (time.perf_counter() - job.dpu_started) * 1000.0

        crop = job.crop
        started = time.perf_counter()
        if crop is None:
            pose = decode_quantized(job.slot.heads, size=job.source_hw, prior=job.prior)
        else:
            pose = decode_quantized(
                job.slot.heads,
                size=(crop.side, crop.side),
                origin=(float(crop.x), float(crop.y)),
                prior=job.prior,
            )
        self._stage_ms["decode"] = (time.perf_counter() - started) * 1000.0

        lock_started = time.perf_counter()
        competitor = self._competitor(job.slot.heads, job.prior, crop, job.source_hw)
        self._lock.observe(pose, time.monotonic(), job.source_hw, competitor=competitor)
        self._stage_ms["lock"] += (time.perf_counter() - lock_started) * 1000.0
        return pose

    def _competitor(
        self,
        heads: QuantizedHeads,
        prior: NDArray[np.float32],
        crop: Crop | None,
        source_hw: tuple[int, int],
    ) -> bool:
        """Whether another centre candidate is clearly stronger than the lock.

        The two candidates are the peak of the locked prior and the peak of the
        frame-centre prior (see :meth:`PersonLock.frame_center_prior`), and
        they are weighed by their **probabilities**, not by the prior-weighted
        scores: the priors differ, so only the center head's confidence is
        comparable. The graph is allowed to look away from the tracked person
        only when the person near the frame centre is
        :data:`STRONGER_FACTOR` more confident; :class:`PersonLock` then
        requires that to hold for :data:`SWITCH_AFTER_S`.
        """

        if not self._lock.locked:
            return False
        own_x, own_y = center_peak(heads, prior)
        other_x, other_y = center_peak(heads, self._lock.frame_center_prior(crop, source_hw))
        if (own_x, own_y) == (other_x, other_y):
            return False
        scale = 2.0**-heads.centers_fix_point
        own = float(_sigmoid(heads.centers[own_y, own_x, 0] * scale))
        other = float(_sigmoid(heads.centers[other_y, other_x, 0] * scale))
        return other > STRONGER_FACTOR * own

    def detect(self, rgb: NDArray[np.uint8]) -> MovenetPose:
        """Run the DPU model on an ``(H, W, 3)`` RGB uint8 frame."""
        return self.wait(self.submit(rgb))

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
    heads: QuantizedHeads
    """The int8 head views of :attr:`output_buffers`, built once per slot."""


@dataclass(frozen=True)
class _Job:
    """One submitted DPU job, enough to decode its heads in :meth:`DpuMovenet.wait`."""

    slot: _Slot
    job_id: int
    dpu_started: float
    source_hw: tuple[int, int]
    crop: Crop | None
    """The region the input was taken from: ``None`` means the whole frame."""
    prior: NDArray[np.float32]
    """The centre prior this job's input was decoded with."""


def _fitted_crop(crop: Crop | None, source_hw: tuple[int, int]) -> Crop | None:
    """Return ``crop`` if it fits a frame of ``source_hw``, else ``None``."""

    if crop is None:
        return None
    height, width = int(source_hw[0]), int(source_hw[1])
    if (
        crop.side <= 0
        or crop.x < 0
        or crop.y < 0
        or crop.x + crop.side > width
        or crop.y + crop.side > height
    ):
        return None
    return crop
