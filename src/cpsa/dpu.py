"""The KV260 DPU: one compiled graph, its buffers and its fix-point helpers.

``xir`` and ``vart`` are imported lazily inside :class:`DpuRunner`, so the
module's pure helpers (and the tests that fake the runtime) import on a host
without the Vitis AI bindings.

The quantization convention of the Vitis AI DPU is a signed ``fix_point``:
``int8 = clip(round(float * 2 ** fix_point), -128, 127)`` and
``float = int8 * 2 ** -fix_point``. For ``uint8`` camera samples that map is a
256-entry table, precomputed once per tensor, so a frame costs one
:func:`cv2.LUT` pass instead of a float conversion over every pixel.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DpuRunner",
    "OutputTensor",
    "dequantize_fix_point",
    "input_quantization_lut_scaled",
    "quantize_fix_point",
    "stride_for",
    "tensor_fix_point",
]


def quantize_fix_point(values: ArrayLike, fix_point: int) -> NDArray[np.int8]:
    """Quantize floats to int8 with ``round(value * 2 ** fix_point)``.

    Values saturate at ``-128``/``127``. Non-finite inputs are rejected.
    """

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("cannot quantize non-finite values")
    scaled = np.rint(array * (2.0**fix_point))
    return np.clip(scaled, -128.0, 127.0).astype(np.int8)


def input_quantization_lut_scaled(fix_point: int, value_scale: float) -> NDArray[np.int8]:
    """Precompute the 256-entry xint8 table for ``uint8`` samples into the model domain.

    ``lut[value]`` equals ``quantize_fix_point(np.float32(value) * value_scale,
    fix_point)``: models that consume raw ``0..255`` pixels, such as the YOLOX
    model-zoo detectors, pass ``1.0``.

    Raises:
        ValueError: If ``value_scale`` is negative or not finite.
    """

    if not np.isfinite(value_scale) or value_scale < 0.0:
        raise ValueError(f"value_scale must be finite and non-negative, got {value_scale}")
    samples = np.arange(256, dtype=np.float32) * np.float32(value_scale)
    return quantize_fix_point(samples, fix_point)


def dequantize_fix_point(values: ArrayLike, fix_point: int) -> NDArray[np.float64]:
    """Dequantize int8 values with ``value * 2 ** -fix_point``."""

    return np.asarray(values, dtype=np.float64) * (2.0**-fix_point)


def tensor_fix_point(tensor: Any) -> int:
    """Return the ``fix_point`` attribute of a VART/xir tensor, or ``0``."""

    if not tensor.has_attr("fix_point"):
        return 0
    return int(tensor.get_attr("fix_point"))


def stride_for(input_hw: tuple[int, int], spatial_hw: tuple[int, int]) -> int:
    """Return the network stride of one output feature map.

    Raises:
        ValueError: If the feature map is empty, does not divide the input or
            is not square in stride.
    """

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


@dataclass(frozen=True)
class OutputTensor:
    """One DPU output: its name, ``(H, W, C)`` NHWC shape, ``fix_point`` and stride."""

    name: str
    shape: tuple[int, int, int]
    fix_point: int
    stride: int


class DpuRunner:
    """One DPU subgraph of a compiled ``.xmodel``: metadata, buffers and jobs.

    Args:
        xmodel_path: Path to the compiled ``.xmodel``.
        min_channels: Fewest channels an output head may have. The YOLOX heads
            carry boxes and classes; MoveNet has a single-channel center
            heatmap, so its detector asks for ``1``.

    Raises:
        RuntimeError: If the ``xir``/``vart`` Python bindings are unavailable.
        FileNotFoundError: If ``xmodel_path`` does not exist.
        ValueError: If the graph has no DPU subgraph or the tensor layout is
            not a single batched NHWC input with ``(1, H, W, C)`` outputs.
    """

    def __init__(self, xmodel_path: str | Path, *, min_channels: int = 6) -> None:
        try:
            import vart
            import xir
        except ImportError as exc:  # pragma: no cover - depends on the board
            raise RuntimeError(
                "xir and vart Python bindings are required for the DPU; "
                "run on the KV260 with the Vitis AI 3.5 runtime"
            ) from exc

        path = Path(xmodel_path)
        if not path.is_file():
            raise FileNotFoundError(f"xmodel not found: {path}")

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
        self._input_fix_point = tensor_fix_point(self._input_tensor)

        self._output_tensors = outputs
        output_tensors: list[OutputTensor] = []
        for index, tensor in enumerate(outputs):
            dims = [int(dim) for dim in tensor.dims]
            if len(dims) != 4 or dims[0] != 1:
                raise ValueError(f"expected NHWC outputs, got shape {dims} at output {index}")
            if dims[3] < min_channels:
                raise ValueError(
                    f"output {index} has {dims[3]} channels, expected >= {min_channels}"
                )
            output_tensors.append(
                OutputTensor(
                    name=tensor.name,
                    shape=(dims[1], dims[2], dims[3]),
                    fix_point=tensor_fix_point(tensor),
                    stride=stride_for(self._input_hw, (dims[1], dims[2])),
                )
            )
        self._outputs = tuple(output_tensors)
        self._executions = 0

    def close(self) -> None:
        """Release the VART runner and the xir graph (R2-G model swap).

        VART owns the DPU memory of the graph and frees it when the runner and
        the graph are destroyed, so dropping the references and collecting is
        the release path of the Python bindings.
        """

        self._runner = None
        self._graph = None
        gc.collect()

    @property
    def input_hw(self) -> tuple[int, int]:
        """Model input ``(height, width)``."""
        return self._input_hw

    @property
    def input_fix_point(self) -> int:
        """``fix_point`` of the input tensor."""
        return self._input_fix_point

    @property
    def outputs(self) -> tuple[OutputTensor, ...]:
        """Output tensors in the order the graph returns them."""
        return self._outputs

    def allocate(self) -> list[NDArray[np.int8]]:
        """Allocate one set of output buffers, zeroed."""
        return [np.zeros([1, *tensor.shape], dtype=np.int8) for tensor in self._outputs]

    def execute(self, inputs: list[NDArray[np.int8]], outputs: list[NDArray[np.int8]]) -> Any:
        """Start one asynchronous DPU job and return its opaque job handle.

        The VART Python binding returns a ``(job_id, status)`` tuple; it is passed
        back to :meth:`wait` unchanged, exactly as VART expects.
        """
        self._executions += 1
        if self._runner is None:
            raise RuntimeError("the DPU runner is closed")
        return self._runner.execute_async(inputs, outputs)

    def wait(self, job: Any) -> None:
        """Wait for a job started by :meth:`execute`."""
        if self._runner is None:
            raise RuntimeError("the DPU runner is closed")
        self._runner.wait(job)
