"""Sliding windows over synchronized rows and the preprocessing the native extractor expects.

The window length and hop are fixed by the trained classifier and by the native
feature-extraction ABI (``float[150]`` inputs); they are not configurable.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

WINDOW_SIZE = 150
HOP = 75
ROW_WIDTH = 20

G_TO_MPS2 = 9.80665

# Column indices in raw rows and preprocessed windows.
_RIGHT, _LEFT = 0, 10
_ACC = slice(0, 3)
_QUAT_X, _QUAT_Y, _QUAT_Z, _QUAT_W = 6, 7, 8, 9

WindowCallback = Callable[[NDArray[np.float32], float], None]


def _hamilton_wxyz(
    a_w: NDArray[np.float32],
    a_x: NDArray[np.float32],
    a_y: NDArray[np.float32],
    a_z: NDArray[np.float32],
    b_w: np.float32,
    b_x: np.float32,
    b_y: np.float32,
    b_z: np.float32,
) -> tuple[NDArray[np.float32], ...]:
    # Terms are kept in the baseline order (including zero factors) so float32
    # results stay bit-identical to the reference implementation.
    w = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z
    x = a_w * b_x + a_x * b_w + a_y * b_z - a_z * b_y
    y = a_w * b_y - a_x * b_z + a_y * b_w + a_z * b_x
    z = a_w * b_z + a_x * b_y - a_y * b_x + a_z * b_w
    return w, x, y, z


def _align_quaternion(raw_xyzw: NDArray[np.float32]) -> NDArray[np.float32]:
    """Rotate raw (x10000) xyzw quaternions into the classifier's body frame.

    Applies ``q ⊗ conj(q_align)`` with ``q_align = [w=0, x=√½, y=√½, z=0]`` (swap X/Y,
    flip Z), normalizing before and after; returns raw x10000 xyzw.
    """
    x = raw_xyzw[:, 0].astype(np.float32).copy()
    y = raw_xyzw[:, 1].astype(np.float32).copy()
    z = raw_xyzw[:, 2].astype(np.float32).copy()
    w = raw_xyzw[:, 3].astype(np.float32).copy()

    n = np.sqrt(w * w + x * x + y * y + z * z)
    n[n == 0] = 1.0
    w /= n
    x /= n
    y /= n
    z /= n

    c = np.float32(np.sqrt(0.5))
    w2, x2, y2, z2 = _hamilton_wxyz(w, x, y, z, np.float32(0.0), -c, -c, np.float32(0.0))

    n2 = np.sqrt(w2 * w2 + x2 * x2 + y2 * y2 + z2 * z2)
    n2[n2 == 0] = 1.0
    w2 /= n2
    x2 /= n2
    y2 /= n2
    z2 /= n2

    scale = np.float32(10000.0)
    aligned = np.empty_like(raw_xyzw, dtype=np.float32)
    aligned[:, 0] = (x2 * scale).astype(np.float32)
    aligned[:, 1] = (y2 * scale).astype(np.float32)
    aligned[:, 2] = (z2 * scale).astype(np.float32)
    aligned[:, 3] = (w2 * scale).astype(np.float32)
    return aligned


def preprocess(raw_window: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert a raw ``(N, 20)`` window into the native extractor's input units.

    Accelerations go from mg to m/s², gyroscope rates stay in dps, and quaternions
    are frame-aligned and kept in raw x10000 units. Column order is unchanged.
    """
    raw: NDArray[np.float32] = np.asarray(raw_window, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != ROW_WIDTH:
        raise ValueError(f"expected shape (N, {ROW_WIDTH}), got {raw.shape}")
    out: NDArray[np.float32] = raw.copy()
    for base in (_RIGHT, _LEFT):
        acc = slice(base + _ACC.start, base + _ACC.stop)
        out[:, acc] = raw[:, acc] / 1000.0 * G_TO_MPS2
        quat = slice(base + _QUAT_X, base + _QUAT_W + 1)
        out[:, quat] = _align_quaternion(raw[:, quat])
    return out


class Windower:
    """Group rows into windows of ``WINDOW_SIZE`` rows every ``HOP`` rows.

    The first window is emitted after ``WINDOW_SIZE`` rows, then one every ``HOP``
    rows (the last ``WINDOW_SIZE - HOP`` rows are kept). ``on_window`` receives the
    preprocessed ``(WINDOW_SIZE, 20)`` window and the timestamp of its last row.
    """

    def __init__(self, on_window: WindowCallback) -> None:
        self._on_window = on_window
        self._rows: list[NDArray[np.float32]] = []
        self._ts: list[float] = []
        self._lock = threading.Lock()

    def add(self, row: NDArray[np.float32], ts: float) -> None:
        if np.shape(row) != (ROW_WIDTH,):
            raise ValueError(f"expected a row of {ROW_WIDTH} values, got shape {np.shape(row)}")
        # Rows arrive from both wrist threads. Emitting while holding the lock keeps
        # windows in order and serializes calls into the stateful native extractor.
        with self._lock:
            self._rows.append(np.asarray(row, dtype=np.float32))
            self._ts.append(float(ts))
            if len(self._rows) < WINDOW_SIZE:
                return
            raw_window = np.stack(self._rows[-WINDOW_SIZE:])
            end_ts = self._ts[-1]
            self._rows = self._rows[HOP:]
            self._ts = self._ts[HOP:]
            self._on_window(preprocess(raw_window), end_ts)
