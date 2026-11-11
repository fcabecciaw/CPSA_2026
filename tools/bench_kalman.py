"""Benchmark the batched Kalman kernels: NumPy reference against the C kernel.

Reports the median wall time of one ``step`` call for a full 62-keypoint frame,
which is the budget that matters for the tracker (AC-K5). There is no timing
assertion: the numbers depend on the machine, and the board numbers are taken
on the board.

Usage::

    uv run --no-sync python tools/bench_kalman.py
    uv run --no-sync python tools/bench_kalman.py --lib /path/libcpsa_kalman2d.so
"""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cpsa.adapters.native.kalman import NativeKalmanKernel, load_native_kalman
from cpsa.vision.kalman_kernel import FilterKernel, NumpyKalmanKernel

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_SOURCE = REPO_ROOT / "native" / "kalman2d" / "kalman2d.c"

KEYPOINTS = 62
DT = 1.0 / 30.0
GATE_CHI2 = 9.21
MEASUREMENT_STD_PX = 3.0
JERK_STD_PXPS3 = 1000.0
# Enough samples that the median is stable on a busy host, and a warmup long
# enough to leave the CPU's idle frequency before the first sample is timed.
REPEATS = 1000
WARMUP = 100


def transition(dt: float) -> NDArray[np.float64]:
    """The tracker's constant-acceleration state transition."""
    matrix = np.eye(6, dtype=np.float64)
    matrix[0, 2] = dt
    matrix[0, 4] = 0.5 * dt * dt
    matrix[1, 3] = dt
    matrix[1, 5] = 0.5 * dt * dt
    matrix[2, 4] = dt
    matrix[3, 5] = dt
    return matrix


def process_noise(dt: float) -> NDArray[np.float64]:
    """The tracker's white-jerk process noise."""
    block = np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ],
        dtype=np.float64,
    )
    block *= JERK_STD_PXPS3**2
    covariance = np.zeros((6, 6), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            covariance[2 * row, 2 * column] = block[row, column]
            covariance[2 * row + 1, 2 * column + 1] = block[row, column]
    return covariance


def frame_arrays() -> tuple[NDArray[np.float64], ...]:
    """One full frame of inputs: every keypoint predicted, measured and accepted."""
    rng = np.random.default_rng(20261111)
    transition_matrix = transition(DT)
    noise = process_noise(DT)
    states = np.zeros((KEYPOINTS, 6), dtype=np.float64)
    states[:, 0] = rng.uniform(0.0, 1920.0, KEYPOINTS)
    states[:, 1] = rng.uniform(0.0, 1080.0, KEYPOINTS)
    states[:, 2] = rng.normal(0.0, 200.0, KEYPOINTS)
    states[:, 3] = rng.normal(0.0, 200.0, KEYPOINTS)
    diagonal = np.array([20.0, 20.0, 1000.0, 1000.0, 10000.0, 10000.0]) ** 2
    prior = np.einsum("ki,ij->kij", np.tile(diagonal, (KEYPOINTS, 1)), np.eye(6))
    covariance = transition_matrix @ prior @ transition_matrix.T + noise
    positions = (states @ transition_matrix.T)[:, :2] + rng.normal(
        0.0, MEASUREMENT_STD_PX, (KEYPOINTS, 2)
    )
    predict = np.ones(KEYPOINTS, dtype=np.bool_)
    update = np.ones(KEYPOINTS, dtype=np.bool_)
    variances = np.full(KEYPOINTS, MEASUREMENT_STD_PX**2, dtype=np.float64)
    return transition_matrix, noise, states, covariance, predict, update, positions, variances


def measure(kernel: FilterKernel, arrays: tuple[NDArray[np.float64], ...]) -> list[float]:
    """Return ``REPEATS`` wall times of one ``step`` call, in microseconds."""
    transition_matrix, noise, states, covariance, predict, update, positions, variances = arrays

    def once() -> None:
        kernel.step(
            transition_matrix,
            noise,
            states,
            covariance,
            predict,
            update,
            positions,
            variances,
            GATE_CHI2,
        )

    for _ in range(WARMUP):
        once()
    timings: list[float] = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        once()
        timings.append((time.perf_counter() - started) * 1e6)
    return timings


def report(name: str, timings: list[float]) -> float:
    """Print one row and return its median."""
    median = statistics.median(timings)
    print(
        f"  {name:<7} median {median / 1000.0:7.3f} ms"
        f"  min {min(timings) / 1000.0:7.3f} ms"
        f"  max {max(timings) / 1000.0:7.3f} ms"
        f"  over {len(timings)} steps of {KEYPOINTS} keypoints"
    )
    return median


def build_native(path: Path | None) -> FilterKernel | None:
    """Return the native kernel from ``--lib``, the loader, or a local build."""
    if path is not None:
        return NativeKalmanKernel(path)
    kernel = load_native_kalman()
    if kernel is not None:
        return kernel
    compiler = shutil.which("cc")
    if compiler is None:
        return None
    output = Path(tempfile.mkdtemp(prefix="cpsa-kalman-")) / "libcpsa_kalman2d.so"
    print(f"  building a temporary kernel for this run: {output}")
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c99",
            "-fPIC",
            "-shared",
            "-o",
            str(output),
            str(KERNEL_SOURCE),
            "-lm",
        ],
        check=True,
    )
    return NativeKalmanKernel(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lib", type=Path, default=None, help="path to libcpsa_kalman2d.so (default: loader)"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    arrays = frame_arrays()
    print(
        f"Kalman kernel benchmark: {platform.machine()}, Python {platform.python_version()}, "
        f"dt {DT:.4f} s"
    )
    print("numpy (reference kernel)")
    numpy_median = report("numpy", measure(NumpyKalmanKernel(), arrays))

    native = build_native(arguments.lib)
    if native is None:
        print("native (kernel unavailable: no --lib, no library on disk, no cc)")
        return 0
    print(f"native ({getattr(native, '_path', 'libcpsa_kalman2d.so')})")
    native_median = report("native", measure(native, arrays))
    if native_median > 0.0:
        print(f"speed-up: {numpy_median / native_median:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
