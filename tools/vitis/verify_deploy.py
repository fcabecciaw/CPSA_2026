#!/usr/bin/env python3
"""Verify the deploy graph and the compiled outputs against the CPSA decoder.

Runs on the host (``ml`` environment with ``PYTHONPATH=src``) and checks:

1. the float deploy graph, decoded with :func:`cpsa.vision.pose_decode.decode_raw_heads`,
   matches Ultralytics' own decoded predictions on the same input;
2. the quantized test-mode outputs, converted to the board representation
   (NHWC int8 with the ``fix_point`` reported for the compiled ``.xmodel``)
   and passed through :mod:`cpsa.adapters.pose.dpu`, decode through
   ``decode_raw_heads`` with the same candidate count and close values;
3. the DPU output order reported by the compiled graph is handled by
   ``order_outputs_by_stride`` exactly as the runtime needs it.

Candidate values are compared after sorting by box centre (not by score rank):
with untrained weights most cells share almost the same score, so a
``1e-4`` score change permutes the rank order while the geometry is unchanged.

Example::

    cd ml && PYTHONPATH=../src uv run python ../tools/vitis/verify_deploy.py \
        --artifacts ../build/vitis/artifacts --quant ../build/vitis/quant \
        --inspect ../build/vitis/report/inspect_compiled.json \
        --out ../build/vitis/report/decoder_check.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics.nn.tasks import PoseModel

from cpsa.vision.pose_decode import decode_raw_heads
from cpsa.vision.types import PersonPose

TOLERANCE_PX = 0.05
TOLERANCE_SCORE = 1e-4
DECODE_CONF_THR = -1.0
DECODE_IOU_THR = 1.01
MAX_SATURATED_FRACTION = 1e-3
MIN_COSINE = 0.999
MAX_CENTER_DISTANCE_PX = 2.0
MAX_CENTER_DISTANCE_MAX_PX = 4.0


def load_dpu_module(repo_root: Path) -> ModuleType:
    """Load ``cpsa/adapters/pose/dpu.py`` without importing the ONNX adapter."""

    sys.path.insert(0, str(repo_root / "src"))
    path = repo_root / "src/cpsa/adapters/pose/dpu.py"
    spec = importlib.util.spec_from_file_location("cpsa_dpu_verify", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_models(artifacts: Path) -> tuple[torch.nn.Module, PoseModel, dict[str, Any]]:
    config = yaml.safe_load((artifacts / "model.yaml").read_text(encoding="utf-8"))
    pose = PoseModel(
        cfg=config,
        ch=3,
        nc=int(config["nc"]),
        data_kpt_shape=tuple(config["kpt_shape"]),
        verbose=False,
    )
    state = torch.load(str(artifacts / "deploy_state_dict.pt"), map_location="cpu")
    pose.load_state_dict(state, strict=True)
    pose.eval()

    from cpsa_ml.deploy import deploy_from_yolo

    deploy = deploy_from_yolo(pose)
    spec = json.loads((artifacts / "deploy_spec.json").read_text(encoding="utf-8"))
    deploy.input_size = tuple(int(value) for value in spec["input_size"])
    return deploy, pose, spec


def ultralytics_decode(model: PoseModel, images: torch.Tensor, n_kpt: int) -> dict[str, np.ndarray]:
    """Raw Ultralytics decoded predictions, as ``xyxy`` boxes + kpts in pixels."""

    with torch.no_grad():
        output = model(images)
    decoded = output[0] if isinstance(output, tuple) else output
    array = decoded.detach().numpy()[0]  # (4 + nc + 3K, N)
    boxes = array[:4]
    cx, cy, width, height = boxes
    xyxy = np.stack(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0], axis=1
    )
    scores = array[4:5].max(axis=0)
    keypoints = array[5:].reshape(n_kpt, 3, -1).transpose(2, 0, 1)
    return {"boxes": xyxy, "scores": scores, "keypoints": keypoints}


def decoded_arrays(persons: list[PersonPose]) -> dict[str, np.ndarray]:
    boxes = np.stack([person.box for person in persons])
    scores = np.asarray([person.score for person in persons], dtype=np.float64)
    keypoints = np.stack(
        [np.concatenate([person.xy, person.conf[:, None]], axis=1) for person in persons]
    )
    return {"boxes": boxes, "scores": scores, "keypoints": keypoints}


def compare(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    expected_order: np.ndarray,
    actual_order: np.ndarray,
) -> dict[str, float]:
    ordered_expected = {key: value[expected_order] for key, value in expected.items()}
    ordered_actual = {key: value[actual_order] for key, value in actual.items()}
    return {
        "boxes_max_abs_px": float(
            np.max(np.abs(ordered_expected["boxes"] - ordered_actual["boxes"]))
        ),
        "scores_max_abs": float(
            np.max(np.abs(ordered_expected["scores"] - ordered_actual["scores"]))
        ),
        "keypoints_max_abs_px": float(
            np.max(np.abs(ordered_expected["keypoints"] - ordered_actual["keypoints"]))
        ),
        "candidates": int(len(expected_order)),
    }


def to_nhwc(array: np.ndarray, channels: int) -> np.ndarray:
    """Convert a ``(1, C, H, W)`` PyTorch output to ``(1, H, W, C)``."""

    squeezed = np.asarray(array, dtype=np.float64)[0]
    if squeezed.shape[0] == channels:
        return squeezed.transpose(1, 2, 0)[None, ...]
    if squeezed.shape[-1] == channels:
        return squeezed[None, ...]
    raise ValueError(f"cannot find {channels} channels in shape {squeezed.shape}")


def cosine(values: np.ndarray, reference: np.ndarray) -> float:
    flat = values.reshape(-1)
    ref = reference.reshape(-1)
    return float(np.dot(flat, ref) / (np.linalg.norm(flat) * np.linalg.norm(ref)))


def decoded_summary(persons: list[PersonPose]) -> dict[str, Any]:
    if not persons:
        return {"count": 0}
    data = decoded_arrays(persons)
    boxes = data["boxes"]
    centers = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1
    )
    return {
        "count": int(len(persons)),
        "scores_min": float(data["scores"].min()),
        "scores_max": float(data["scores"].max()),
        "scores_mean": float(data["scores"].mean()),
        "centers": centers,
        "boxes": boxes,
        "keypoints": data["keypoints"],
        "scores": data["scores"],
        "finite": bool(np.isfinite(boxes).all() and np.isfinite(data["keypoints"]).all()),
    }


def nearest_distances(source: np.ndarray, reference: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Distance from every ``source`` point to the closest ``reference`` point."""

    distances = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk):
        block = source[start : start + chunk]
        deltas = block[:, None, :] - reference[None, :, :]
        distances[start : start + chunk] = np.linalg.norm(deltas, axis=2).min(axis=1)
    return distances


def set_shift(float_summary: dict[str, Any], board_summary: dict[str, Any]) -> dict[str, Any]:
    """Compare the two candidate sets without relying on an unstable order.

    With untrained weights many cells share the same score, so score rank and
    even centre sorting are ambiguous; the candidate sets are compared with
    nearest-neighbour (Chamfer) distances instead.
    """

    if float_summary["count"] == 0 or board_summary["count"] == 0:
        return {"candidates_compared": 0}
    forward = nearest_distances(float_summary["centers"], board_summary["centers"])
    backward = nearest_distances(board_summary["centers"], float_summary["centers"])
    scores = np.abs(np.sort(float_summary["scores"]) - np.sort(board_summary["scores"]))
    return {
        "candidates_compared": int(min(float_summary["count"], board_summary["count"])),
        "center_distance_mean_px": float(forward.mean()),
        "center_distance_p95_px": float(np.percentile(forward, 95)),
        "center_distance_max_px": float(forward.max()),
        "center_distance_reverse_p95_px": float(np.percentile(backward, 95)),
        "score_delta_mean": float(scores.mean()),
        "score_delta_max": float(scores.max()),
    }


def channel_groups(spec: dict[str, Any]) -> dict[str, slice]:
    layout = spec["channel_layout"]
    box = int(layout["box"])
    classes = int(layout["class"])
    keypoints = int(layout["keypoint"])
    return {
        "box": slice(0, box),
        "class": slice(box, box + classes),
        "keypoint": slice(box + classes, box + classes + keypoints),
    }


def channel_errors(
    float_nhwc: np.ndarray,
    board_nhwc: np.ndarray,
    groups: dict[str, slice],
    n_kpt: int,
    stride: int,
) -> dict[str, float]:
    """Per-group maximum error and the resulting decoded-coordinate bound."""

    errors = {
        name: float(np.abs(float_nhwc[..., group] - board_nhwc[..., group]).max())
        for name, group in groups.items()
    }
    keypoint = board_nhwc[..., groups["keypoint"]].reshape(*board_nhwc.shape[:3], n_kpt, 3)
    float_keypoint = float_nhwc[..., groups["keypoint"]].reshape(*float_nhwc.shape[:3], n_kpt, 3)
    xy_error = float(np.abs(float_keypoint[..., :2] - keypoint[..., :2]).max())
    conf_error = float(np.abs(float_keypoint[..., 2] - keypoint[..., 2]).max())
    return {
        "box_logits_max_delta": errors["box"],
        "class_logits_max_delta": errors["class"],
        "keypoint_values_max_delta": errors["keypoint"],
        "keypoint_xy_max_delta": xy_error,
        "keypoint_conf_max_delta": conf_error,
        "keypoint_xy_bound_px": 2.0 * stride * xy_error,
        "keypoint_conf_bound": 0.25 * conf_error,
        "score_bound": 0.25 * errors["class"],
    }


def box_size_summary(summary: dict[str, Any]) -> dict[str, float]:
    boxes = summary["boxes"]
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    sizes = np.concatenate([widths, heights])
    return {
        "width_mean_px": float(widths.mean()),
        "height_mean_px": float(heights.mean()),
        "size_std_px": float(sizes.std()),
    }


def check_quantized(
    quant_dir: Path,
    inspect: dict[str, Any],
    spec: dict[str, Any],
    dpu: ModuleType,
) -> dict[str, Any]:
    payload = np.load(quant_dir / "test_outputs.npz")
    channels = int(spec["output_channels"])
    n_kpt = int(spec["n_kpt"])
    groups = channel_groups(spec)
    strides_by_name = {
        str(name): int(stride)
        for name, stride in zip(spec["output_names"], spec["strides"], strict=True)
    }
    entries = [entry for entry in inspect.get("output_order", []) if entry.get("stride")]
    by_stride = {int(entry["stride"]): entry for entry in entries}
    report: dict[str, Any] = {
        "compatible": True,
        "checks": {},
        "per_stride": [],
    }

    board_arrays: dict[int, list[np.ndarray]] = {}

    for name, stride in strides_by_name.items():
        entry = by_stride.get(stride)
        if entry is None:
            report["compatible"] = False
            report["checks"][f"stride{stride}_in_xmodel"] = False
            continue
        fix_point = int(entry.get("fix_point") or 0)
        layout = str(entry.get("layout") or "")
        stride_report: dict[str, Any] = {
            "stride": stride,
            "name": name,
            "xmodel_tensor": entry["name"],
            "dims": entry["dims"],
            "layout": layout,
            "fix_point": fix_point,
            "dtype": entry.get("dtype"),
        }
        float_key = f"float_{name}"
        quant_key = f"quant_{name}"
        if float_key not in payload.files or quant_key not in payload.files:
            report["compatible"] = False
            report["checks"][f"stride{stride}_in_npz"] = False
            continue

        float_outputs = payload[float_key]
        quant_outputs = payload[quant_key]
        decoded_float: list[list[PersonPose]] = []
        decoded_board: list[list[PersonPose]] = []
        board_arrays[stride] = []
        numeric: list[dict[str, Any]] = []
        errors: list[dict[str, float]] = []
        floor = 2.0**-fix_point
        for frame_index in range(float_outputs.shape[0]):
            float_nhwc = to_nhwc(float_outputs[frame_index], channels)
            quant_nhwc = to_nhwc(quant_outputs[frame_index], channels)
            board_nhwc = dpu.dequantize_fix_point(
                dpu.quantize_fix_point(quant_nhwc, fix_point), fix_point
            )
            board_arrays[stride].append(board_nhwc)
            numeric.append(
                {
                    "cosine": cosine(board_nhwc, float_nhwc),
                    "max_abs_error": float(np.abs(board_nhwc - float_nhwc).max()),
                    "mean_abs_error": float(np.abs(board_nhwc - float_nhwc).mean()),
                    "saturated_fraction": float(
                        np.mean((quant_nhwc <= -128.0 * floor) | (quant_nhwc >= 127.0 * floor))
                    ),
                    "float_abs_max": float(np.abs(float_nhwc).max()),
                    "board_abs_max": float(np.abs(board_nhwc).max()),
                }
            )
            errors.append(channel_errors(float_nhwc, board_nhwc, groups, n_kpt, stride))
            decoded_float.append(
                decode_raw_heads([float_nhwc], [stride], n_kpt, DECODE_CONF_THR, DECODE_IOU_THR)
            )
            decoded_board.append(
                decode_raw_heads([board_nhwc], [stride], n_kpt, DECODE_CONF_THR, DECODE_IOU_THR)
            )

        float_summaries = [decoded_summary(persons) for persons in decoded_float]
        board_summaries = [decoded_summary(persons) for persons in decoded_board]
        cell_count = int(np.prod(entry["dims"][1:3]))
        shifts = [
            set_shift(float_one, board_one)
            for float_one, board_one in zip(float_summaries, board_summaries, strict=True)
        ]
        counts_match = all(
            float_one["count"] == board_one["count"]
            for float_one, board_one in zip(float_summaries, board_summaries, strict=True)
        )
        finite_ok = all(summary["finite"] for summary in board_summaries)
        cosine_min = float(np.min([item["cosine"] for item in numeric]))
        saturated_max = float(np.max([item["saturated_fraction"] for item in numeric]))
        distance_p95 = float(np.mean([shift["center_distance_p95_px"] for shift in shifts]))
        distance_max = float(np.max([shift["center_distance_max_px"] for shift in shifts]))
        stride_checks = {
            "counts_match": counts_match,
            "count_is_one_per_cell": all(
                summary["count"] == cell_count for summary in board_summaries
            ),
            "finite": finite_ok,
            "no_saturation": saturated_max <= MAX_SATURATED_FRACTION,
            "cosine_ok": cosine_min >= MIN_COSINE,
            "candidate_sets_close": (
                distance_p95 <= MAX_CENTER_DISTANCE_PX
                and distance_max <= MAX_CENTER_DISTANCE_MAX_PX
            ),
        }
        report["checks"].update(
            {f"stride{stride}_{key}": value for key, value in stride_checks.items()}
        )
        report["compatible"] = report["compatible"] and all(stride_checks.values())
        stride_report.update(
            {
                "cells": cell_count,
                "frames": len(float_summaries),
                "numeric": {
                    "cosine_min": cosine_min,
                    "max_abs_error": float(np.max([item["max_abs_error"] for item in numeric])),
                    "mean_abs_error": float(np.mean([item["mean_abs_error"] for item in numeric])),
                    "saturated_fraction_max": saturated_max,
                    "float_abs_max": float(np.max([item["float_abs_max"] for item in numeric])),
                },
                "error_bounds": {
                    key: float(np.max([item[key] for item in errors])) for key in errors[0]
                },
                "decoded": {
                    "count_float": [summary["count"] for summary in float_summaries],
                    "count_board": [summary["count"] for summary in board_summaries],
                    "per_frame": shifts,
                    "box_sizes_float": box_size_summary(float_summaries[0]),
                    "box_sizes_board": box_size_summary(board_summaries[0]),
                },
                "checks": stride_checks,
            }
        )
        report["per_stride"].append(stride_report)

    # The runtime receives DPU tensors in graph order and pairs each with the
    # stride derived from its shape.  Emulate that with the real xmodel order.
    raw_order = sorted(entries, key=lambda entry: entry["index"])
    raw_strides = [int(entry["stride"]) for entry in raw_order]
    expected_order = sorted(raw_strides)
    if len(raw_order) > 1 and len(board_arrays) == len(raw_strides):
        frame_index = 0
        ordered = dpu.order_outputs_by_stride(
            [board_arrays[stride][frame_index] for stride in raw_strides], raw_strides
        )
        order_ok = [stride for stride, _ in ordered] == expected_order
        match_ok = all(
            np.array_equal(array, board_arrays[stride][frame_index]) for stride, array in ordered
        )
        report["order_check"] = {
            "reported_dpu_order": raw_strides,
            "reordered_strides": [stride for stride, _ in ordered],
            "expected_strides": expected_order,
            "order_ok": bool(order_ok),
            "arrays_ok": bool(match_ok),
        }
        report["compatible"] = report["compatible"] and bool(order_ok and match_ok)
        report["checks"]["output_order_by_stride"] = bool(order_ok and match_ok)
    else:
        report["order_check"] = {"skipped": "need the xmodel entries for every stride"}
        report["checks"]["output_order_by_stride"] = False
        report["compatible"] = False

    report["strides"] = sorted(by_stride)
    report["frames"] = int(payload["frames"].shape[0])
    report["all_ok"] = bool(report["compatible"])
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=root / "build/vitis/artifacts")
    parser.add_argument("--quant", type=Path, default=root / "build/vitis/quant")
    parser.add_argument(
        "--inspect", type=Path, default=root / "build/vitis/report/inspect_compiled.json"
    )
    parser.add_argument("--out", type=Path, default=root / "build/vitis/report/decoder_check.json")
    parser.add_argument("--conf-thr", type=float, default=-1.0)
    parser.add_argument("--iou-thr", type=float, default=1.01)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[2]
    dpu = load_dpu_module(repo_root)
    deploy, pose, spec = build_models(args.artifacts)

    frames = np.load(args.artifacts / "deploy_reference.npz")["images"].astype(np.float32)
    images = torch.from_numpy(frames)
    with torch.no_grad():
        deploy_outputs = [tensor.numpy() for tensor in deploy(images)]

    strides = [int(stride) for stride in spec["strides"]]
    our_persons = decode_raw_heads(
        [np.transpose(output, (0, 2, 3, 1)).astype(np.float64) for output in deploy_outputs],
        strides,
        int(spec["n_kpt"]),
        args.conf_thr,
        args.iou_thr,
    )
    expected = ultralytics_decode(pose, images, int(spec["n_kpt"]))
    actual = decoded_arrays(our_persons)
    expected_order = np.argsort(-expected["scores"], kind="stable")
    actual_order = np.argsort(-actual["scores"], kind="stable")
    if expected["scores"].shape[0] != actual["scores"].shape[0]:
        raise SystemExit(
            f"candidate count mismatch: ultralytics {expected['scores'].shape[0]} "
            f"vs decode_raw_heads {actual['scores'].shape[0]}"
        )
    parity = compare(expected, actual, expected_order, actual_order)
    parity["passed"] = (
        parity["boxes_max_abs_px"] <= TOLERANCE_PX
        and parity["scores_max_abs"] <= TOLERANCE_SCORE
        and parity["keypoints_max_abs_px"] <= TOLERANCE_PX
    )

    report: dict[str, Any] = {
        "float_vs_ultralytics": parity,
        "tolerances": {"px": TOLERANCE_PX, "score": TOLERANCE_SCORE},
        "deploy_outputs": [
            {
                "stride": stride,
                "shape": list(output.shape),
                "channels": int(output.shape[1]),
            }
            for stride, output in zip(strides, deploy_outputs, strict=True)
        ],
    }

    inspect: dict[str, Any] = {}
    if args.inspect.is_file():
        inspect = json.loads(args.inspect.read_text(encoding="utf-8"))
        report["xmodel"] = {
            "fingerprint": inspect.get("fingerprint"),
            "fingerprint_matches": inspect.get("fingerprint_matches"),
            "target": inspect.get("target"),
            "dpu_subgraph_count": inspect.get("dpu_subgraph_count"),
            "subgraph_count": inspect.get("subgraph_count"),
            "input": inspect.get("dpu_inputs"),
            "input_size_hw": inspect.get("input_size_hw"),
        }
    else:
        report["xmodel"] = {"missing": str(args.inspect)}
    report["quantized"] = check_quantized(args.quant, inspect, spec, dpu)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    failed = []
    if not parity["passed"]:
        failed.append("deploy graph does not match Ultralytics decoding")
    if not report["quantized"].get("all_ok", False):
        failed.append("quantized outputs are not decoder-compatible")
    if failed:
        raise SystemExit("; ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
