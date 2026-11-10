"""Vitis AI DPU pose estimator (NFR-003).

``xir`` and ``vart`` are imported lazily inside :class:`DpuPoseEstimator`, so
the module (and its quantize/dequantize helpers) import on hosts without the
board runtime.

The DPU graph is expected to be cut at the raw YOLOv8-pose head: one or more
``(1, H, W, C)`` NHWC outputs per stride, with channels ordered
``[box (4 * reg_max) | class logits | keypoints (3 * K)]``.  Inputs are
quantized with the tensor's signed ``fix_point`` on normalized ``[0, 1]`` RGB:
``int8 = clip(round(float * 2 ** fix_point), -128, 127)``.  Outputs are
dequantized with ``float = int8 * 2 ** -fix_point`` before :func:`decode_raw_heads`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cpsa.vision.pose_decode import decode_raw_heads, letterbox, unletterbox
from cpsa.vision.types import KeypointLayout, PersonPose

logger = logging.getLogger(__name__)

__all__ = [
    "DpuPoseEstimator",
    "dequantize_fix_point",
    "order_outputs_by_stride",
    "quantize_fix_point",
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
        self._output_buffers = [
            np.empty([int(dim) for dim in tensor.dims], dtype=np.int8) for tensor in outputs
        ]
        self._strides: list[int] = []
        self._output_fix_points: list[int] = []
        for index, tensor in enumerate(outputs):
            dims = [int(dim) for dim in tensor.dims]
            if len(dims) != 4 or dims[0] != 1:
                raise ValueError(f"expected NHWC outputs, got shape {dims} at output {index}")
            self._strides.append(_stride_for(self._input_hw, (dims[1], dims[2])))
            self._output_fix_points.append(_tensor_fix_point(tensor))

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_hw

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(self._strides)

    def estimate(self, rgb: NDArray[np.uint8]) -> list[PersonPose]:
        """Run the DPU model on an ``(H, W, 3)`` RGB uint8 image."""

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {image.dtype}")

        letterboxed, scale, pad = letterbox(image, self._input_hw)
        normalized = letterboxed.astype(np.float32) / 255.0
        if self._input_nhwc:
            quantized = quantize_fix_point(normalized, self._input_fix_point)
        else:
            quantized = quantize_fix_point(normalized.transpose(2, 0, 1), self._input_fix_point)
        np.copyto(self._input_buffer, quantized.reshape(self._input_buffer.shape))

        job_id = self._runner.execute_async([self._input_buffer], self._output_buffers)
        self._runner.wait(job_id)

        dequantized = [
            dequantize_fix_point(buffer, fix_point)
            for buffer, fix_point in zip(self._output_buffers, self._output_fix_points, strict=True)
        ]
        ordered = order_outputs_by_stride(dequantized, self._strides)
        persons = decode_raw_heads(
            [output for _, output in ordered],
            [stride for stride, _ in ordered],
            len(self._layout),
            self._conf_thr,
            self._iou_thr,
        )
        return unletterbox(persons, scale, pad)
