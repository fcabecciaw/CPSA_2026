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
"""

from __future__ import annotations

import functools
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
    "QuantizedHeads",
    "center_prior",
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
) -> None:
    """Resize and quantize one frame straight into a DPU input view.

    Args:
        rgb: ``(H, W, 3)`` RGB uint8 frame.
        lut: 256-entry int8 table, as built by
            :func:`cpsa.dpu.input_quantization_lut_scaled` with
            :data:`MOVENET_INPUT_SCALE`.
        out: ``(192, 192, 3)`` C-contiguous int8 buffer to fill.

    Raises:
        ValueError: If the frame or ``out`` do not match the model input.
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

    # Resize straight into the DPU input view and quantize it in place: no
    # intermediate 192x192 frame is allocated or copied.
    view = out.view(np.uint8)
    cv2.resize(image, (MOVENET_INPUT_SIZE[1], MOVENET_INPUT_SIZE[0]), dst=view)
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
        size: Source frame ``(height, width)`` the keypoints are mapped to. The
            input is a square resize of the whole frame, so the normalised
            coordinates scale by this size.

    Returns:
        The pose in source pixels, in :data:`KEYPOINTS` order.

    Raises:
        ValueError: If ``size`` is not positive.
    """

    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive, got {size}")

    heatmaps = _sigmoid(heads.heatmaps)
    heatmaps = np.where(heatmaps < MOVENET_HM_THRESHOLD, 0.0, heatmaps)
    centers = _sigmoid(heads.centers[0])
    regs = heads.regs
    offsets = heads.offsets

    grid_y, grid_x = np.mgrid[0:MOVENET_FEATURE_SIZE, 0:MOVENET_FEATURE_SIZE]

    center_index = int(np.argmax(centers * center_prior()))
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
        keypoints[index, 0] = (peak_x + offset_x) / MOVENET_FEATURE_SIZE * width
        keypoints[index, 1] = (peak_y + offset_y) / MOVENET_FEATURE_SIZE * height
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
) -> MovenetPose:
    """Decode the int8 heads directly: the A53 path of :func:`decode`.

    The result matches the reference decoder, with the same steps done in the
    integer domain and over all 17 keypoints at once:

    1. ``argmax(sigmoid(centers) * prior)`` equals ``argmax(centers * prior)``:
       ``sigmoid`` is monotonic, so only the raw center logits and the
       ``float32`` prior table are needed;
    2. the ``float32`` sigmoid table of the heatmap fix point already zeroes
       every value below :data:`MOVENET_HM_THRESHOLD`, so a cell below the
       table's cutoff cannot win the weighted search and is not scored at all;
    3. the regression and offset fields are dequantized at the center cell and
       at the 17 peak cells only, not over the whole ``48x48x34`` heads.

    Args:
        heads: The four int8 heads, as :func:`quantized_heads` builds them.
        size: Source frame ``(height, width)`` the keypoints are mapped to.

    Returns:
        The pose in source pixels, in :data:`KEYPOINTS` order.

    Raises:
        ValueError: If ``size`` is not positive.
    """

    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive, got {size}")

    # A monotonic transform does not move the argmax, so the raw int8 centers
    # multiplied by the float32 prior pick the same cell as the reference.
    center = int(np.argmax(heads.centers[:, :, 0] * _center_prior_f32()))
    center_x = center % MOVENET_FEATURE_SIZE
    center_y = center // MOVENET_FEATURE_SIZE

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

    signed = np.arange(256, dtype=np.int16)
    signed[signed >= 128] -= 256
    probabilities = _sigmoid(signed.astype(np.float64) * (2.0 ** -int(fix_point)))
    probabilities = np.where(probabilities < MOVENET_HM_THRESHOLD, 0.0, probabilities)
    order = np.concatenate([np.arange(128, 256), np.arange(0, 128)])
    cutoff_index = int(order[np.argmax(probabilities[order] > 0.0)])
    lut = probabilities.astype(np.float32)
    lut.flags.writeable = False
    return lut, cutoff_index if cutoff_index < 128 else cutoff_index - 256


class DpuMovenet:
    """Run one compiled MoveNet ``.xmodel`` graph on the KV260 DPU.

    Args:
        xmodel_path: Path to the compiled ``.xmodel``.

    Raises:
        RuntimeError: If the ``xir``/``vart`` Python bindings are unavailable.
        FileNotFoundError: If ``xmodel_path`` does not exist.
        ValueError: If the graph is not the expected MoveNet graph: one DPU
            subgraph, one ``192x192x3`` int8 input and the four stride-4 heads.
    """

    def __init__(self, xmodel_path: str | Path) -> None:
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
        self._stage_ms: dict[str, float] = {"preprocess": 0.0, "dpu": 0.0, "decode": 0.0}

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

        started = time.perf_counter()
        preprocess_into(image, self._lut, slot.input_view)
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
        )
        return token

    def wait(self, token: int) -> MovenetPose:
        """Wait for a job from :meth:`submit` and decode it into a pose.

        Raises:
            ValueError: If ``token`` is not a pending job.
        """

        job = self._pending.pop(token, None)
        if job is None:
            raise ValueError(f"no pending DPU job with token {token}")
        self._runner.wait(job.job_id)
        self._stage_ms["dpu"] = (time.perf_counter() - job.dpu_started) * 1000.0

        started = time.perf_counter()
        pose = decode_quantized(job.slot.heads, size=job.source_hw)
        self._stage_ms["decode"] = (time.perf_counter() - started) * 1000.0
        return pose

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
