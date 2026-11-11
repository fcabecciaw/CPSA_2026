"""Vitis AI DPU pose estimator (NFR-003).

``xir`` and ``vart`` are imported lazily inside :class:`DpuPoseEstimator`, so
the module (and its quantize/dequantize helpers) import on hosts without the
board runtime.

The DPU graph is expected to be cut at the raw YOLOv8-pose head: one or more
``(1, H, W, C)`` NHWC outputs per stride, with channels ordered
``[box (4 * reg_max) | class logits | keypoints (3 * K)]``.  Inputs are
quantized with the tensor's signed ``fix_point`` on normalized ``[0, 1]`` RGB:
``int8 = clip(round(float * 2 ** fix_point), -128, 127)``, reproduced exactly by
a precomputed 256-entry lookup table.  Outputs are decoded by
:class:`cpsa.vision.pose_decode.QuantizedRawHeadDecoder`, which thresholds the
raw int8 class logits before dequantizing or applying softmax/sigmoid; its
result matches :func:`cpsa.vision.pose_decode.decode_raw_heads` on the
dequantized tensors.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cpsa.adapters.pose.resize import LETTERBOX_RESIZE
from cpsa.vision.pose_decode import (
    QuantizedRawHeadDecoder,
    letterbox,
    unletterbox,
)
from cpsa.vision.types import KeypointLayout, PersonPose

logger = logging.getLogger(__name__)

__all__ = [
    "DpuPoseEstimator",
    "dequantize_fix_point",
    "input_quantization_lut",
    "order_outputs_by_stride",
    "quantize_fix_point",
    "quantize_rgb_lut",
]


def quantize_fix_point(values: ArrayLike, fix_point: int) -> NDArray[np.int8]:
    """Quantize floats to int8 with ``round(value * 2 ** fix_point)``.

    Values saturate at ``-128``/``127``.  Non-finite inputs are rejected.
    """

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("cannot quantize non-finite values")
    scaled = np.rint(array * (2.0**fix_point))
    return np.clip(scaled, -128.0, 127.0).astype(np.int8)


def input_quantization_lut(fix_point: int) -> NDArray[np.int8]:
    """Precompute the 256-entry xint8 lookup table for uint8 RGB samples.

    ``lut[value]`` equals what :func:`quantize_fix_point` returns for the
    reference pipeline's ``np.float32(value) / 255.0``, including its float32
    normalisation, rounding and saturation, for every value in ``[0, 255]``.
    """

    normalized = np.arange(256, dtype=np.float32) / np.float32(255.0)
    return quantize_fix_point(normalized, fix_point)


def quantize_rgb_lut(
    rgb: NDArray[np.uint8],
    lut: NDArray[np.int8],
    out: NDArray[np.int8],
) -> NDArray[np.int8]:
    """Map uint8 RGB samples to int8 through a 256-entry lookup table.

    ``out`` must be a C-contiguous int8 buffer with exactly ``rgb.size``
    elements, such as the DPU's input tensor.  The table is applied with one
    C-level byte translation, which reproduces
    ``quantize_fix_point(rgb.astype(np.float32) / 255.0, fix_point)`` exactly
    without any float intermediate over the frame.
    """

    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {rgb.dtype}")
    if lut.shape != (256,) or lut.dtype != np.int8:
        raise ValueError(f"lut must be a 256-element int8 table, got {lut.shape} {lut.dtype}")
    if out.dtype != np.int8 or out.size != rgb.size:
        raise ValueError(f"out must hold {rgb.size} int8 values, got {out.shape} {out.dtype}")
    table = np.ascontiguousarray(lut).view(np.uint8).tobytes()
    np.copyto(out.reshape(-1), np.frombuffer(rgb.tobytes().translate(table), dtype=np.int8))
    return out.reshape(rgb.shape)


def dequantize_fix_point(values: ArrayLike, fix_point: int) -> NDArray[np.float64]:
    """Dequantize int8 values with ``value * 2 ** -fix_point``."""

    return np.asarray(values, dtype=np.float64) * (2.0**-fix_point)


def order_outputs_by_stride(
    outputs: Sequence[NDArray[np.float64]],
    strides: Sequence[int],
) -> list[tuple[int, NDArray[np.float64]]]:
    """Pair outputs with strides and sort by ascending stride.

    DPU output tensors may be returned in any order; the decoder needs the
    smallest stride (largest feature map) first, matching the anchor order of
    the upstream head.
    """

    if len(outputs) != len(strides):
        raise ValueError(f"got {len(outputs)} outputs but {len(strides)} strides")
    pairs = [(int(stride), output) for stride, output in zip(strides, outputs, strict=True)]
    return sorted(pairs, key=lambda pair: pair[0])


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


def _tensor_fix_point(tensor: Any) -> int:
    if not tensor.has_attr("fix_point"):
        return 0
    return int(tensor.get_attr("fix_point"))


class DpuPoseEstimator:
    """Run one compiled ``.xmodel`` pose graph on the KV260 DPU.

    Args:
        xmodel_path: Path to the compiled ``.xmodel``.
        layout: Keypoint layout the model was trained with; ``len(layout)`` is
            the model's keypoint count.
        conf_thr: Minimum class score for a detection.
        iou_thr: NMS IoU threshold.

    Raises:
        RuntimeError: If the ``xir``/``vart`` Python bindings are unavailable.
        ValueError: If the graph has no DPU subgraph or unexpected tensor
            shapes.
    """

    def __init__(
        self,
        xmodel_path: str | Path,
        layout: KeypointLayout,
        conf_thr: float = 0.25,
        iou_thr: float = 0.45,
    ) -> None:
        try:
            import vart  # type: ignore[import-not-found]
            import xir  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the board
            raise RuntimeError(
                "xir and vart Python bindings are required for DpuPoseEstimator; "
                "run on the KV260 with the Vitis AI 3.5 runtime"
            ) from exc

        path = Path(xmodel_path)
        if not path.is_file():
            raise FileNotFoundError(f"xmodel not found: {path}")
        if not 0.0 <= conf_thr <= 1.0:
            raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
        if not 0.0 <= iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in [0, 1], got {iou_thr}")

        self._layout = layout
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
        if len(input_dims) != 4:
            raise ValueError(f"expected a 4-D DPU input, got shape {input_dims}")
        if input_dims[3] == 3:
            _, height, width, _ = input_dims
            self._input_nhwc = True
        elif input_dims[1] == 3:
            _, _, height, width = input_dims
            self._input_nhwc = False
        else:
            raise ValueError(f"DPU input is neither NHWC nor NCHW RGB: {input_dims}")
        self._input_hw = (height, width)
        self._input_fix_point = _tensor_fix_point(self._input_tensor)

        self._output_tensors = outputs
        self._input_buffer = np.empty(input_dims, dtype=np.int8)
        self._input_lut = input_quantization_lut(self._input_fix_point)
        self._input_view = (
            self._input_buffer.reshape((*self._input_hw, 3)) if self._input_nhwc else None
        )
        self._input_scratch = (
            None if self._input_nhwc else np.empty((*self._input_hw, 3), dtype=np.int8)
        )
        self._output_buffers = [
            np.empty([int(dim) for dim in tensor.dims], dtype=np.int8) for tensor in outputs
        ]
        self._strides: list[int] = []
        self._output_fix_points: list[int] = []
        self._output_shapes: list[tuple[int, int, int]] = []
        for index, tensor in enumerate(outputs):
            dims = [int(dim) for dim in tensor.dims]
            if len(dims) != 4 or dims[0] != 1:
                raise ValueError(f"expected NHWC outputs, got shape {dims} at output {index}")
            self._strides.append(_stride_for(self._input_hw, (dims[1], dims[2])))
            self._output_fix_points.append(_tensor_fix_point(tensor))
            self._output_shapes.append((dims[1], dims[2], dims[3]))

        self._decoder = QuantizedRawHeadDecoder(
            self._output_shapes,
            self._strides,
            self._output_fix_points,
            len(layout),
        )
        self._stage_ms: dict[str, float] = {
            "preprocess": 0.0,
            "dpu": 0.0,
            "decode": 0.0,
            "nms": 0.0,
            "unletterbox": 0.0,
        }
        self._last_candidate_count = 0

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_hw

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(self._strides)

    @property
    def stage_ms(self) -> dict[str, float]:
        """Per-stage timings of the last :meth:`estimate`, in milliseconds."""

        return dict(self._stage_ms)

    @property
    def last_candidate_count(self) -> int:
        """Candidates that survived the raw-domain class threshold in the last frame."""

        return self._last_candidate_count

    def estimate(self, rgb: NDArray[np.uint8]) -> list[PersonPose]:
        """Run the DPU model on an ``(H, W, 3)`` RGB uint8 image."""

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {image.dtype}")

        started = time.perf_counter()
        letterboxed, scale, pad = letterbox(image, self._input_hw, resize=LETTERBOX_RESIZE)
        if self._input_nhwc:
            assert self._input_view is not None
            quantize_rgb_lut(letterboxed, self._input_lut, self._input_view)
        else:
            scratch = self._input_scratch
            assert scratch is not None
            quantize_rgb_lut(letterboxed, self._input_lut, scratch)
            np.copyto(self._input_buffer, scratch.transpose(2, 0, 1))
        self._stage_ms["preprocess"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        job_id = self._runner.execute_async([self._input_buffer], self._output_buffers)
        self._runner.wait(job_id)
        self._stage_ms["dpu"] = (time.perf_counter() - started) * 1000.0

        persons = self._decoder.decode(self._output_buffers, self._conf_thr, self._iou_thr)
        self._stage_ms["decode"] = self._decoder.last_decode_seconds * 1000.0
        self._stage_ms["nms"] = self._decoder.last_select_seconds * 1000.0
        self._last_candidate_count = self._decoder.last_candidate_count

        started = time.perf_counter()
        restored = unletterbox(persons, scale, pad)
        self._stage_ms["unletterbox"] = (time.perf_counter() - started) * 1000.0
        return restored
