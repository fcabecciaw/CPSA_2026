#!/usr/bin/env python3
"""Micro-benchmark for the F200 RGB/depth conversion paths.

Runs without a camera. On the KV260::

    PYTHONPATH=src python3 scripts/bench_camera_convert.py

The AC-01 target is < 15 ms per 640x480 YUYV frame on the Cortex-A53.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable

import numpy as np

from cpsa.adapters.camera.f200 import yuyv_to_rgb, z16_to_depth

TARGET_MS = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--target-ms", type=float, default=TARGET_MS)
    return parser.parse_args()


def make_frames(width: int, height: int, count: int = 4) -> tuple[list[bytes], list[bytes]]:
    rng = np.random.default_rng(20261106)
    yuyv = [
        rng.integers(0, 256, size=width * height * 2, dtype=np.uint8).tobytes()
        for _ in range(count)
    ]
    z16 = [
        rng.integers(0, 65536, size=width * height, dtype=np.uint16).tobytes() for _ in range(count)
    ]
    return yuyv, z16


def time_calls(
    function: Callable[[bytes, int, int], object],
    frames: list[bytes],
    width: int,
    height: int,
    iterations: int,
) -> list[float]:
    for frame in frames:
        function(frame, width, height)
    times = []
    for index in range(iterations):
        frame = frames[index % len(frames)]
        started = time.perf_counter()
        function(frame, width, height)
        times.append((time.perf_counter() - started) * 1000.0)
    return times


def report(name: str, times: list[float], target_ms: float) -> bool:
    mean = statistics.fmean(times)
    median = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
    print(
        f"{name:10s} mean={mean:6.2f} ms  median={median:6.2f} ms  "
        f"p95={p95:6.2f} ms  min={min(times):6.2f} ms  max={max(times):6.2f} ms  "
        f"({1000.0 / mean:5.1f} fps)"
    )
    return mean < target_ms


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2:
        print("width and height must be positive and width must be even")
        return 2
    if args.iterations <= 0:
        print("--iterations must be positive")
        return 2
    yuyv_frames, z16_frames = make_frames(args.width, args.height)

    print(
        f"python={sys.version.split()[0]} numpy={np.__version__} "
        f"frame={args.width}x{args.height} iterations={args.iterations}"
    )
    rgb_times = time_calls(yuyv_to_rgb, yuyv_frames, args.width, args.height, args.iterations)
    depth_times = time_calls(z16_to_depth, z16_frames, args.width, args.height, args.iterations)
    rgb_ok = report("yuyv->rgb", rgb_times, args.target_ms)
    depth_ok = report("z16->depth", depth_times, args.target_ms)
    print(f"target: {args.target_ms:g} ms per frame")
    if not rgb_ok or not depth_ok:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
