"""Export and audit DPU-compatible YOLO-pose ONNX graphs.

The KV260 deployment accepts only conv-style operators in the backbone and
neck: Conv, BatchNorm, ReLU/LeakyReLU, Hard-Swish and Hard-Sigmoid.  This module
exports a fixed ``1x3x480x640`` input at opset 17 and rejects SiLU
(``Sigmoid`` followed by ``Mul``), MatMul, Softmax and attention operators in
the pre-decode graph.  Decode/post-processing operators are reported separately
because Vitis AI may leave them on the CPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx

__all__ = [
    "ONNX_OPSET",
    "OnnxReport",
    "check_onnx",
    "export_onnx",
    "format_report",
    "main",
]

ONNX_OPSET = 17
DEFAULT_IMGSZ = (480, 640)

# Operators that may appear before the keypoint-decode tail.
BACKBONE_ALLOWED = frozenset(
    {
        "Add",
        "AveragePool",
        "BatchNormalization",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Flatten",
        "Gather",
        "GlobalAveragePool",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "LeakyRelu",
        "MaxPool",
        "Mul",
        "Pad",
        "Relu",
        "Reshape",
        "Resize",
        "Shape",
        "Slice",
        "Split",
        "Squeeze",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Upsample",
    }
)

FORBIDDEN_BACKBONE = frozenset(
    {
        "Attention",
        "Einsum",
        "Erf",
        "Gelu",
        "Gemm",
        "LayerNormalization",
        "MatMul",
        "MultiHeadAttention",
        "Sigmoid",
        "Softmax",
        "Tanh",
    }
)


@dataclass(frozen=True)
class OnnxReport:
    """Result of :func:`check_onnx`."""

    path: Path
    input_shape: tuple[int | str, ...] | None
    output_shapes: tuple[tuple[int | str, ...], ...]
    opset: int | None
    backbone_ops: dict[str, int]
    postprocess_ops: dict[str, int]
    violations: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "passed": self.passed,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "opset": self.opset,
            "backbone_ops": self.backbone_ops,
            "postprocess_ops": self.postprocess_ops,
            "violations": list(self.violations),
            "notes": list(self.notes),
        }


def _shape(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...] | None:
    tensor = value_info.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    dims: list[int | str] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return tuple(dims)


def _opset(model: onnx.ModelProto) -> int | None:
    versions = [op.version for op in model.opset_import if op.domain in ("", "ai.onnx")]
    return max(versions) if versions else None


def _is_silu_mul(node: onnx.NodeProto, sigmoid_outputs: set[str]) -> bool:
    return node.op_type == "Mul" and any(name in sigmoid_outputs for name in node.input)


def check_onnx(
    path: str | Path,
    *,
    imgsz: tuple[int, int] = DEFAULT_IMGSZ,
    opset: int = ONNX_OPSET,
) -> OnnxReport:
    """Inspect an exported ONNX graph and report DPU compatibility.

    Args:
        path: ONNX file.
        imgsz: Expected fixed input size as ``(height, width)``.
        opset: Expected opset version; a mismatch is recorded as a note, not a
            violation, so the checker can also audit third-party exports.
    """

    onnx_path = Path(path)
    model = onnx.load(str(onnx_path))
    graph = model.graph

    initializers = {tensor.name for tensor in graph.initializer}
    inputs = [value for value in graph.input if value.name not in initializers]
    input_shape = _shape(inputs[0]) if inputs else None
    output_shapes = tuple(shape for value in graph.output if (shape := _shape(value)) is not None)

    nodes = list(graph.node)
    # The DFL (distribution focal loss) subgraph and the pose decode sit in the
    # detection head but are post-processing from the DPU's point of view.  Its
    # Softmax and 1x1 DFL Conv must not be mistaken for backbone/neck layers.
    decode_indices = {index for index, node in enumerate(nodes) if "/dfl/" in node.name.lower()}
    conv_indices = [
        index
        for index, node in enumerate(nodes)
        if node.op_type == "Conv" and index not in decode_indices
    ]
    last_conv = conv_indices[-1] if conv_indices else len(nodes) - 1
    backbone = [
        node
        for index, node in enumerate(nodes)
        if index <= last_conv and index not in decode_indices
    ]
    postprocess = [
        node for index, node in enumerate(nodes) if index > last_conv or index in decode_indices
    ]

    backbone_ops = dict(sorted(Counter(node.op_type for node in backbone).items()))
    postprocess_ops = dict(sorted(Counter(node.op_type for node in postprocess).items()))

    violations: list[str] = []
    notes: list[str] = []
    expected_shape = (1, 3, int(imgsz[0]), int(imgsz[1]))
    if input_shape != expected_shape:
        violations.append(f"input shape is {input_shape}, expected {expected_shape}")
    actual_opset = _opset(model)
    if actual_opset != opset:
        notes.append(f"opset is {actual_opset}, expected {opset}")

    sigmoid_outputs = {
        output for node in backbone if node.op_type == "Sigmoid" for output in node.output
    }
    for node in backbone:
        if node.op_type in FORBIDDEN_BACKBONE:
            violations.append(f"forbidden backbone operator: {node.op_type}")
        elif node.op_type not in BACKBONE_ALLOWED:
            violations.append(f"operator outside DPU allow-list: {node.op_type}")
        if _is_silu_mul(node, sigmoid_outputs):
            violations.append("SiLU pattern in backbone: Sigmoid output multiplied by x")

    # De-duplicate while preserving order.
    violations = list(dict.fromkeys(violations))
    if "Sigmoid" in postprocess_ops:
        notes.append("Sigmoid appears only after the last Conv: CPU post-processing allowed")
    if "Softmax" in postprocess_ops:
        notes.append("Softmax appears only after the last Conv: CPU post-processing allowed")
    if "MatMul" in postprocess_ops:
        notes.append("MatMul appears only after the last Conv: CPU post-processing allowed")
    if decode_indices:
        notes.append("DFL/pose-decode subgraph reported as CPU post-processing")

    return OnnxReport(
        path=onnx_path,
        input_shape=input_shape,
        output_shapes=output_shapes,
        opset=actual_opset,
        backbone_ops=backbone_ops,
        postprocess_ops=postprocess_ops,
        violations=tuple(violations),
        notes=tuple(notes),
    )


def export_onnx(
    model: Any,
    output: str | Path,
    *,
    imgsz: tuple[int, int] = DEFAULT_IMGSZ,
    opset: int = ONNX_OPSET,
    simplify: bool = False,
    device: str = "cpu",
) -> Path:
    """Export a YOLO checkpoint/YAML to a fixed-input ONNX file.

    Args:
        model: ``ultralytics.YOLO`` instance, checkpoint path, or YAML path.
        output: Destination ``.onnx`` path.
        imgsz: Fixed input size as ``(height, width)``.
        opset: ONNX opset (17 for Vitis AI 3.5).
        simplify: Use onnxslim.  Kept off locally to avoid an extra download.
        device: Export device.
    """

    from ultralytics import YOLO

    output_path = Path(output)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"output must end in .onnx, got {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, YOLO):
        yolo = model
    elif isinstance(model, (str, Path)):
        yolo = YOLO(str(model), verbose=False)
    else:
        raise TypeError("model must be an ultralytics.YOLO, a checkpoint path, or a YAML path")

    exported = Path(
        yolo.export(
            format="onnx",
            imgsz=[int(imgsz[0]), int(imgsz[1])],
            opset=int(opset),
            simplify=bool(simplify),
            dynamic=False,
            device=device,
            batch=1,
        )
    )
    if exported.resolve() != output_path.resolve():
        shutil.move(str(exported), str(output_path))
    return output_path


def format_report(report: OnnxReport) -> str:
    lines = [
        f"ONNX operator check: {'PASS' if report.passed else 'FAIL'}",
        f"  file: {report.path}",
        f"  input: {list(report.input_shape) if report.input_shape else None}",
        f"  opset: {report.opset}",
        f"  backbone+neck ops: {report.backbone_ops}",
        f"  CPU post-processing allowed: {report.postprocess_ops}",
    ]
    if report.notes:
        lines.append("  notes:")
        lines.extend(f"    - {note}" for note in report.notes)
    if report.violations:
        lines.append("  violations:")
        lines.extend(f"    - {violation}" for violation in report.violations)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="ONNX file or YOLO YAML/.pt")
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=None,
        help="where to write the ONNX export",
    )
    parser.add_argument("--check-only", action="store_true", help="do not export; --model is ONNX")
    parser.add_argument(
        "--imgsz", type=int, nargs=2, default=list(DEFAULT_IMGSZ), metavar=("H", "W")
    )
    parser.add_argument("--opset", type=int, default=ONNX_OPSET)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = args.model
    if not args.check_only:
        if args.output is None:
            raise SystemExit("--output is required unless --check-only is used")
        path = export_onnx(path, args.output, imgsz=tuple(args.imgsz), opset=args.opset)
    report = check_onnx(path, imgsz=tuple(args.imgsz), opset=args.opset)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
