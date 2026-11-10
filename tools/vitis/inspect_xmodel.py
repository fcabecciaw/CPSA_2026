#!/usr/bin/env python3
"""Inspect a quantized or compiled ``.xmodel`` (runs inside the container).

Writes a JSON report with the DPU fingerprint and target, every subgraph and
its device, the operator mix per subgraph, and the accelerator input/output
tensors with dtype and ``fix_point``.  The runtime dequantizes int8 outputs
with ``2 ** -fix_point``, so the fix-points here are the values the board uses.

Two graph levels are reported:

* ``inputs``/``outputs`` are the graph-level tensors (the compiled graph wraps
  the int8 DPU tensors in ``data-fix``/``fix2float`` CPU subgraphs, so the
  outputs here are float32 and carry no fix-point);
* ``dpu_inputs``/``output_order`` are the tensors of the DPU subgraph, i.e. the
  xint8 values the board runner actually produces, ordered by stride.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import xir

EXPECTED_FINGERPRINT = "0x101000016010407"


def _attr(obj: Any, name: str) -> Any:
    if not obj.has_attr(name):
        return None
    value = obj.get_attr(name)
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _tensor(tensor: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": tensor.name,
        "dims": [int(dim) for dim in tensor.dims],
        "dtype": str(tensor.dtype),
    }
    fix_point = _attr(tensor, "fix_point")
    if fix_point is not None:
        entry["fix_point"] = int(fix_point)
    scale = _attr(tensor, "scale")
    if scale is not None:
        entry["scale"] = float(scale)
    return entry


def _device(subgraph: Any) -> str:
    device = _attr(subgraph, "device")
    return str(device) if device is not None else "unknown"


def _layout_and_stride(
    dims: list[int],
    input_hw: tuple[int, int] | None,
) -> tuple[str | None, int | None]:
    """Infer ``(layout, stride)`` of a 4-D output tensor from the input size."""

    if input_hw is None or len(dims) != 4:
        return None, None
    height, width = input_hw
    candidates = (("NHWC", dims[1], dims[2]), ("NCHW", dims[2], dims[3]))
    for layout, rows, cols in candidates:
        if rows and cols and height % rows == 0 and width % cols == 0:
            stride = height // rows
            if stride == width // cols:
                return layout, stride
    return None, None


def _fingerprint(subgraphs: list[Any], graph: Any) -> tuple[str | None, str | None]:
    """Find the DPU fingerprint and where it was read from."""

    for subgraph in subgraphs:
        for key in ("dpu_fingerprint", "fingerprint"):
            value = _attr(subgraph, key)
            if value is not None:
                return hex(int(value)), f"{subgraph.get_name()}:{key}"
    for obj_name, obj in (("root", graph.get_root_subgraph()), ("graph", graph)):
        value = _attr(obj, "fingerprint")
        if value is not None:
            return hex(int(value)), f"{obj_name}:fingerprint"
    return None, None


def inspect(xmodel: Path, input_size: tuple[int, int] | None = None) -> dict[str, Any]:
    graph = xir.Graph.deserialize(str(xmodel))
    root = graph.get_root_subgraph()
    subgraphs = root.toposort_child_subgraph()

    entries: list[dict[str, Any]] = []
    for index, subgraph in enumerate(subgraphs):
        op_types: dict[str, int] = {}
        for op in subgraph.get_ops():
            name = str(op.get_type())
            op_types[name] = op_types.get(name, 0) + 1
        entries.append(
            {
                "index": index,
                "name": subgraph.get_name(),
                "device": _device(subgraph),
                "op_count": int(subgraph.get_op_num()),
                "op_types": dict(sorted(op_types.items())),
                "input_tensors": [_tensor(tensor) for tensor in subgraph.get_input_tensors()],
                "output_tensors": [_tensor(tensor) for tensor in subgraph.get_output_tensors()],
            }
        )

    outputs = [_tensor(tensor) for tensor in root.get_output_tensors() if tensor.ndim > 0]
    inputs = [_tensor(tensor) for tensor in root.get_input_tensors() if tensor.ndim > 0]
    inputs_source = "graph"
    if not inputs and subgraphs:
        # The compiled graph feeds the quantized input from a CPU ``data-fix``
        # subgraph, so the graph-level input list is empty.
        inputs = [
            _tensor(tensor) for tensor in subgraphs[0].get_output_tensors() if tensor.ndim > 0
        ]
        inputs_source = f"subgraph 0 output ({subgraphs[0].get_name()})"

    dpu_subgraphs = [subgraph for subgraph in subgraphs if _device(subgraph) == "DPU"]
    dpu_subgraph = dpu_subgraphs[0] if dpu_subgraphs else None
    dpu_inputs = (
        [_tensor(tensor) for tensor in dpu_subgraph.get_input_tensors() if tensor.ndim > 0]
        if dpu_subgraph is not None
        else []
    )
    quantized_outputs = (
        [_tensor(tensor) for tensor in dpu_subgraph.get_output_tensors() if tensor.ndim > 0]
        if dpu_subgraph is not None
        else outputs
    )

    resolved_input = input_size
    if resolved_input is None and inputs:
        dims = inputs[0]["dims"]
        if len(dims) == 4:
            resolved_input = (dims[1], dims[2]) if dims[3] == 3 else (dims[2], dims[3])

    order = []
    for index, tensor in enumerate(quantized_outputs):
        dims = tensor["dims"]
        layout, stride = _layout_and_stride(dims, resolved_input)
        order.append(
            {
                "index": index,
                "name": tensor["name"],
                "dims": dims,
                "dtype": tensor["dtype"],
                "layout": layout,
                "stride": stride,
                "fix_point": tensor.get("fix_point"),
            }
        )
    known = [item for item in order if item["stride"] is not None]
    order_report = sorted(known, key=lambda item: item["stride"]) or order

    fingerprint, fingerprint_source = _fingerprint(subgraphs, graph)
    target = _attr(dpu_subgraph, "dpu_name") if dpu_subgraph is not None else None
    if target is None:
        target = _attr(root, "target") or _attr(graph, "target")

    return {
        "xmodel": str(xmodel),
        "size_bytes": xmodel.stat().st_size,
        "fingerprint": fingerprint,
        "fingerprint_source": fingerprint_source,
        "fingerprint_matches": (
            isinstance(fingerprint, str) and fingerprint.lower() == EXPECTED_FINGERPRINT.lower()
        ),
        "target": target,
        "subgraph_count": len(entries),
        "subgraphs": entries,
        "inputs": inputs,
        "inputs_source": inputs_source,
        "outputs": outputs,
        "dpu_subgraph": dpu_subgraph.get_name() if dpu_subgraph is not None else None,
        "dpu_inputs": dpu_inputs,
        "output_order": order_report,
        "input_size_hw": list(resolved_input) if resolved_input else None,
        "dpu_subgraph_count": len(dpu_subgraphs),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xmodel", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--input-hw", default=None, help="model input as height,width")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    input_size = None
    if args.input_hw:
        height, width = (int(part) for part in args.input_hw.split(","))
        input_size = (height, width)
    report = inspect(args.xmodel, input_size)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
