#!/usr/bin/env python3
"""Benchmark DPU/ONNX pose inference latency and throughput (AC-016 support).

Examples::

    uv run --extra vision python scripts/bench_pose.py --onnx model.onnx --layout coco17
    python3 scripts/bench_pose.py --xmodel model.xmodel --layout halfbust62 --frames 200

Without ``--image`` the script uses random uint8 frames, which is enough to
time the model and the decode path; a real image better reflects detector
content but not throughput.  ``--image`` needs Pillow (the ``recorder`` extra).
For the DPU estimator the output also breaks the end-to-end latency into
preprocess/quantize, DPU, dequantize+decode and NMS stages, and reports how many
candidates survived the raw-domain threshold.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from cpsa.adapters.pose.dpu import DpuPoseEstimator
from cpsa.adapters.pose.onnx import OnnxPoseEstimator
from cpsa.vision.layouts import get_layout
from cpsa.vision.types import KeypointLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--onnx", type=Path, help="exported YOLOv8-pose ONNX file")
    model.add_argument("--xmodel", type=Path, help="compiled Vitis AI xmodel")
    parser.add_argument("--layout", default="coco17", help="coco17 or halfbust62")
    parser.add_argument(
        "--input",
        default="480,640",
        help="ONNX model input as height,width (default 480,640)",
    )
    parser.add_argument("--image", type=Path, help="optional RGB image (needs Pillow)")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--conf-thr", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    return parser.parse_args()


def parse_input_size(value: str) -> tuple[int, int]:
    try:
        height, width = (int(part) for part in value.replace("x", ",").split(","))
    except ValueError as exc:
        raise ValueError(f"--input must be height,width, got {value!r}") from exc
    if height <= 0 or width <= 0:
        raise ValueError(f"--input must be positive, got {value!r}")
    return height, width


def build_estimator(args: argparse.Namespace, layout: KeypointLayout) -> Any:
    if args.onnx is not None:
        return OnnxPoseEstimator(
            args.onnx,
            layout,
            input_size=parse_input_size(args.input),
            conf_thr=args.conf_thr,
            iou_thr=args.iou_thr,
        )
    return DpuPoseEstimator(
        args.xmodel,
        layout,
        conf_thr=args.conf_thr,
        iou_thr=args.iou_thr,
    )


def make_frame(args: argparse.Namespace, size: tuple[int, int]) -> np.ndarray:
    if args.image is None:
        rng = np.random.default_rng(args.seed)
        return rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("--image needs Pillow; install the 'recorder' extra") from exc
    with Image.open(args.image) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)


def summarise(times: list[float]) -> dict[str, float]:
    mean = statistics.fmean(times)
    return {
        "mean_ms": mean * 1000.0,
        "median_ms": statistics.median(times) * 1000.0,
        "p95_ms": _p95(times) * 1000.0,
        "fps": 1.0 / mean if mean > 0.0 else float("inf"),
        "frames": float(len(times)),
    }


def summarise_stages(samples: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Summarise the estimator's per-stage millisecond samples."""

    return {
        name: {"mean_ms": statistics.fmean(values), "p95_ms": _p95(values)}
        for name, values in samples.items()
    }


def main() -> int:
    args = parse_args()
    if args.frames <= 0 or args.warmup < 0:
        print("--frames must be positive and --warmup non-negative", file=sys.stderr)
        return 2
    layout = get_layout(args.layout)
    estimator = build_estimator(args, layout)
    frame = make_frame(args, estimator.input_size)

    for _ in range(args.warmup):
        estimator.estimate(frame)

    times: list[float] = []
    detections = 0
    candidates = 0
    stage_samples: dict[str, list[float]] = {}
    for _ in range(args.frames):
        started = time.perf_counter()
        persons = estimator.estimate(frame)
        times.append(time.perf_counter() - started)
        detections += len(persons)
        for name, value in getattr(estimator, "stage_ms", {}).items():
            stage_samples.setdefault(name, []).append(value)
        candidates += getattr(estimator, "last_candidate_count", 0)

    result: dict[str, Any] = summarise(times)
    result["detections_per_frame"] = detections / args.frames
    result["layout"] = args.layout
    result["model"] = str(args.onnx or args.xmodel)
    if stage_samples:
        result["stages"] = summarise_stages(stage_samples)
        result["candidates_per_frame"] = candidates / args.frames
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"{result['model']} ({result['layout']}): "
            f"mean={result['mean_ms']:.2f} ms  median={result['median_ms']:.2f} ms  "
            f"p95={result['p95_ms']:.2f} ms  fps={result['fps']:.1f}  "
            f"detections/frame={result['detections_per_frame']:.2f}"
        )
        for name, stage in result.get("stages", {}).items():
            print(f"  {name:<12s} mean={stage['mean_ms']:7.2f} ms  p95={stage['p95_ms']:7.2f} ms")
        if "candidates_per_frame" in result:
            print(f"  candidates/frame={result['candidates_per_frame']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
