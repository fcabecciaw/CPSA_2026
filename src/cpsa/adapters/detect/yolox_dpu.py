"""Vitis AI DPU YOLOX detector (F-S1).

Runs a compiled model-zoo YOLOX graph (``yolox_nano``, COCO-80, ``416x416``)
on the KV260 DPU and decodes it with :func:`cpsa.vision.yolox_decode.decode_yolox`.

The zoo preprocessing is raw-pixel BGR: ``mean 0, scale 1`` in the model
prototxt, bilinear letterbox to ``416x416`` with pad value ``114``.  The
compiled graph takes ``[1, 416, 416, 3]`` xint8 NHWC with ``fix_point -2``
(``round(pixel * 2 ** -2)``), reproduced by a 256-entry LUT, and returns the
three raw heads ``(1, 52, 52, 85)``, ``(1, 26, 26, 85)``, ``(1, 13, 13, 85)``
xint8 with ``fix_point 3``.  The channels are ``[tx, ty, tw, th, objectness,
80 class logits]``; upstream ``YOLOXHead.postprocess`` applies ``sigmoid`` to
channels ``4:`` and then the grid/stride decode, so the decoder runs with
``logits=True``.

``xir``/``vart`` are imported lazily inside :class:`DpuYoloxDetector`, so the
module's helpers (and the reference checker) import on hosts without the board
runtime.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cpsa.adapters.pose.dpu import (
    dequantize_fix_point,
    input_quantization_lut_scaled,
    quantize_rgb_lut,
)
from cpsa.adapters.pose.resize import LETTERBOX_RESIZE
from cpsa.vision.detection import Detection
from cpsa.vision.pose_decode import letterbox
from cpsa.vision.yolox_decode import QuantizedYoloxDecoder, decode_yolox

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the installed extra
    cv2 = None  # type: ignore[assignment]

__all__ = [
    "DpuYoloxDetector",
    "YOLOX_INPUT_SIZE",
    "YOLOX_PAD_VALUE",
    "YOLOX_STRIDES",
    "boxes_to_source",
    "detections_from_heads",
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


def _validate_frame(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return ``rgb`` after checking the shape and dtype the graph needs."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {image.dtype}")
    return image


def _letterbox_geometry(
    height: int,
    width: int,
    out_height: int,
    out_width: int,
) -> tuple[int, int, float, int, int]:
    """Return ``(resized_height, resized_width, scale, top, left)``.

    The rounding is the one of :func:`cpsa.vision.pose_decode.letterbox`, so the
    OpenCV path and the NumPy path pad identically.
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


def _letterbox_bgr_cv2(
    rgb: NDArray[np.uint8],
    size: tuple[int, int],
) -> tuple[NDArray[np.uint8], float, tuple[float, float]]:
    """OpenCV letterbox (INTER_LINEAR, pad 114) plus the RGB-to-BGR swap.

    Three OpenCV passes instead of the NumPy canvas, the negative-stride channel
    view and the ``ascontiguousarray`` copy: on the KV260 that is most of the
    preprocessing budget.  Pixels are identical to :func:`prepare_input`'s NumPy
    fallback, because ``cvtColor`` for ``RGB2BGR`` is a pure channel swap and the
    pad value is constant.
    """

    assert cv2 is not None
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
    resized_height, resized_width, scale, top, left = _letterbox_geometry(
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


def prepare_input(
    rgb: NDArray[np.uint8],
    size: tuple[int, int] = YOLOX_INPUT_SIZE,
    *,
    resize: Any = LETTERBOX_RESIZE,
) -> tuple[NDArray[np.uint8], float, tuple[float, float]]:
    """Convert one RGB frame into the graph's BGR letterboxed input.

    With OpenCV installed (the ``vision`` extra, always present on the board)
    the resize, the channel swap and the pad border all run in OpenCV; without
    it the NumPy letterbox of :func:`cpsa.vision.pose_decode.letterbox` is used.

    Args:
        rgb: ``(H, W, 3)`` RGB uint8 frame.
        size: Target ``(height, width)``.
        resize: Optional ``(image, out_height, out_width)`` resize callable, as
            in :func:`cpsa.vision.pose_decode.letterbox`; the OpenCV fast path is
            used only for the default (the call Ultralytics letterboxes with).

    Returns:
        ``(bgr, scale, (pad_x, pad_y))``: the contiguous letterboxed BGR image
        and the mapping returned by :func:`~cpsa.vision.pose_decode.letterbox`,
        which :func:`boxes_to_source` inverts.
    """

    if cv2 is not None and resize is LETTERBOX_RESIZE:
        return _letterbox_bgr_cv2(rgb, size)
    image = _validate_frame(rgb)
    letterboxed, scale, pad = letterbox(image, size, YOLOX_PAD_VALUE, resize=resize)
    return np.ascontiguousarray(letterboxed[..., ::-1]), scale, pad


def _quantize_into(
    image: NDArray[np.uint8],
    lut: NDArray[np.int8],
    out: NDArray[np.int8],
) -> None:
    """Apply the quantization table, with OpenCV's byte-LUT when available.

    ``cv2.LUT`` and :func:`cpsa.adapters.pose.dpu.quantize_rgb_lut` are both pure
    byte remappings, so the result is identical; OpenCV's SIMD loop is about four
    times faster per 416x416x3 frame than the ``bytes.translate`` + copy.
    """

    if cv2 is not None:
        cv2.LUT(image, np.ascontiguousarray(lut.view(np.uint8)), dst=out.view(np.uint8))
        return
    quantize_rgb_lut(image, lut, out)


def prepare_input_into(
    rgb: NDArray[np.uint8],
    size: tuple[int, int],
    lut: NDArray[np.int8],
    out: NDArray[np.int8],
    *,
    resize: Any = LETTERBOX_RESIZE,
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
            :func:`cpsa.adapters.pose.dpu.input_quantization_lut_scaled`.
        out: ``(height, width, 3)`` C-contiguous int8 buffer to fill.
        resize: See :func:`prepare_input`.
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
    if cv2 is None or resize is not LETTERBOX_RESIZE:
        bgr, scale, pad = prepare_input(image, size, resize=resize)
        _quantize_into(bgr, lut, out)
        return scale, pad

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("rgb must not be empty")
    if (height, width) == (out_height, out_width):
        _quantize_into(np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), dtype=np.uint8), lut, out)
        return 1.0, (0.0, 0.0)
    resized_height, resized_width, scale, top, left = _letterbox_geometry(
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
    """

    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
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

    This is the detector's CPU side without the DPU: :meth:`DpuYoloxDetector.detect`
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
        Detections in source pixels, score-descending, degenerate boxes
        dropped.
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
        boxes,
        scores,
        class_ids,
        scale=scale,
        pad=pad,
        source_hw=source_hw,
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
            box=(
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            ),
            score=float(score),
            class_id=int(class_id),
        )
        for box, score, class_id in zip(mapped, scores, class_ids, strict=True)
        if box[2] > box[0] and box[3] > box[1]
    ]
    detections.sort(key=lambda detection: detection.score, reverse=True)
    return detections


def _tensor_fix_point(tensor: Any) -> int:
    if not tensor.has_attr("fix_point"):
        return 0
    return int(tensor.get_attr("fix_point"))


def _stride_for(input_hw: tuple[int, int], spatial_hw: tuple[int, int]) -> int:
    input_height, input_width = input_hw
    height, width = spatial_hw
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid output spatial size {spatial_hw}")
    if input_height % height or input_width % width:
        raise ValueError(f"output size {spatial_hw} does not divide the input size {input_hw}")
    stride_y, stride_x = input_height // height, input_width // width
    if stride_y != stride_x:
        raise ValueError(f"non-square stride for output size {spatial_hw} and input {input_hw}")
    return stride_y


class DpuYoloxDetector:
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
        try:
            import vart  # type: ignore[import-not-found]
            import xir  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the board
            raise RuntimeError(
                "xir and vart Python bindings are required for DpuYoloxDetector; "
                "run on the KV260 with the Vitis AI 3.5 runtime"
            ) from exc

        path = Path(xmodel_path)
        if not path.is_file():
            raise FileNotFoundError(f"xmodel not found: {path}")
        if not 0.0 <= conf_thr <= 1.0:
            raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
        if not 0.0 <= iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in [0, 1], got {iou_thr}")

        self._conf_thr = float(conf_thr)
        self._iou_thr = float(iou_thr)

        self._graph = xir.Graph.deserialize(str(path))
        dpu_subgraph = None
        for subgraph in self._graph.get_root_subgraph().toposort_child_subgraph():
            device = subgraph.get_attr("device") if subgraph.has_attr("device") else None
            if device in ("DPU", b"DPU"):
                dpu_subgraph = subgraph
                break
        if dpu_subgraph is None:
            raise ValueError(f"{path} has no DPU subgraph")
        self._runner = vart.Runner.create_runner(dpu_subgraph, "run")

        inputs = self._runner.get_input_tensors()
        outputs = self._runner.get_output_tensors()
        if len(inputs) != 1:
            raise ValueError(f"expected one DPU input tensor, got {len(inputs)}")
        if not outputs:
            raise ValueError("the DPU graph has no output tensors")

        self._input_tensor = inputs[0]
        input_dims = [int(dim) for dim in self._input_tensor.dims]
        if len(input_dims) != 4 or input_dims[0] != 1:
            raise ValueError(f"expected a 4-D batched DPU input, got shape {input_dims}")
        if input_dims[3] != 3:
            raise ValueError(f"expected an NHWC DPU input, got shape {input_dims}")
        self._input_hw = (input_dims[1], input_dims[2])
        self._input_fix_point = _tensor_fix_point(self._input_tensor)
        self._input_lut = input_quantization_lut_scaled(self._input_fix_point, YOLOX_INPUT_SCALE)

        self._output_tensors = outputs
        self._output_fix_points: list[int] = []
        self._output_strides: list[int] = []
        self._output_shapes: list[tuple[int, int, int]] = []
        for index, tensor in enumerate(outputs):
            dims = [int(dim) for dim in tensor.dims]
            if len(dims) != 4 or dims[0] != 1:
                raise ValueError(f"expected NHWC outputs, got shape {dims} at output {index}")
            if dims[3] < 6:
                raise ValueError(f"output {index} has {dims[3]} channels, expected >= 6")
            self._output_strides.append(_stride_for(self._input_hw, (dims[1], dims[2])))
            self._output_fix_points.append(_tensor_fix_point(tensor))
            self._output_shapes.append((dims[1], dims[2], dims[3]))
        self._strides = sorted(self._output_strides)
        if self._strides != list(YOLOX_STRIDES):
            raise ValueError(
                f"expected the YOLOX strides {list(YOLOX_STRIDES)}, got {self._strides}"
            )

        # Two slots so the loop can submit frame N while decoding frame N-1:
        # a DPU job owns its input and output buffers until wait() returns.
        self._slots = [self._slot() for _ in range(2)]
        self._next_slot = 0
        self._token = 0
        self._pending: dict[int, _Job] = {}
        self._decoder = QuantizedYoloxDecoder(
            self._output_shapes, self._output_strides, self._output_fix_points
        )
        self._stage_ms: dict[str, float] = {"preprocess": 0.0, "dpu": 0.0, "decode": 0.0}

    def _slot(self) -> _Slot:
        """Allocate one input/output buffer set, with its pad border pre-quantized."""

        buffer = np.empty([1, self._input_hw[0], self._input_hw[1], 3], dtype=np.int8)
        view = buffer.reshape((*self._input_hw, 3))
        view.fill(self._input_lut[YOLOX_PAD_VALUE])
        return _Slot(
            input_buffer=buffer,
            input_view=view,
            output_buffers=[
                np.empty([int(dim) for dim in tensor.dims], dtype=np.int8)
                for tensor in self._output_tensors
            ],
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_hw

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(self._strides)

    @property
    def stage_ms(self) -> dict[str, float]:
        """Per-stage timings of the last :meth:`detect` steps, in milliseconds."""

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

        Returns a token for :meth:`wait`.  Submitting frame N and then waiting
        for frame N-1 overlaps the DPU with the CPU decode and the draw, which is
        the difference between ~40 ms and ~16 ms of frame budget on the KV260.
        Two jobs may be in flight (one per slot); a third submit raises.

        Raises:
            RuntimeError: If both slots are busy.
            ValueError: If ``rgb`` is not an ``(H, W, 3)`` uint8 frame.
        """

        image = _validate_frame(rgb)
        if len(self._pending) >= len(self._slots):
            raise RuntimeError("both DPU slots are busy; call wait() before submitting again")
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        height, width = int(image.shape[0]), int(image.shape[1])

        started = time.perf_counter()
        geometry = (height, width)
        fill_pad = slot.pad_geometry != geometry
        scale, pad = prepare_input_into(
            image,
            self._input_hw,
            self._input_lut,
            slot.input_view,
            fill_pad=fill_pad,
        )
        slot.pad_geometry = geometry
        self._stage_ms["preprocess"] = (time.perf_counter() - started) * 1000.0

        dpu_started = time.perf_counter()
        job_id = self._runner.execute_async([slot.input_buffer], slot.output_buffers)
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
    """One submitted DPU job, enough to decode its outputs in :meth:`DpuYoloxDetector.wait`."""

    slot: _Slot
    job_id: int
    dpu_started: float
    scale: float
    pad: tuple[float, float]
    source_hw: tuple[int, int]
