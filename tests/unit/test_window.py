import math

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.imu.window import HOP, WINDOW_SIZE, Windower, preprocess

C = math.sqrt(0.5)


def raw_row(i: int) -> NDArray[np.float32]:
    row = np.zeros(20, dtype=np.float32)
    row[0] = i  # marks the row in R acc x
    row[9] = row[19] = 10000.0  # identity quaternions
    return row


class Windows:
    def __init__(self) -> None:
        self.windows: list[tuple[NDArray[np.float32], float]] = []

    def __call__(self, window: NDArray[np.float32], end_ts: float) -> None:
        self.windows.append((window, end_ts))


def feed(windower: Windower, start: int, count: int) -> None:
    for i in range(start, start + count):
        windower.add(raw_row(i), ts=float(i) / 100)


def first_rows(windows: Windows) -> list[int]:
    # Undo the mg -> m/s^2 conversion of the row marker.
    return [round(float(w[0][0, 0]) / 9.80665 * 1000) for w in windows.windows]


def test_first_window_after_window_size_rows_then_every_hop() -> None:
    windows = Windows()
    windower = Windower(windows)

    feed(windower, 0, WINDOW_SIZE - 1)
    assert windows.windows == []

    feed(windower, WINDOW_SIZE - 1, 1)
    feed(windower, WINDOW_SIZE, HOP - 1)
    assert first_rows(windows) == [0]

    feed(windower, WINDOW_SIZE + HOP - 1, 1 + 2 * HOP)
    assert first_rows(windows) == [0, HOP, 2 * HOP, 3 * HOP]
    assert all(w.shape == (WINDOW_SIZE, 20) for w, _ in windows.windows)


def test_window_end_timestamp_is_last_row_timestamp() -> None:
    windows = Windows()
    windower = Windower(windows)
    feed(windower, 0, WINDOW_SIZE + HOP)
    assert [ts for _, ts in windows.windows] == [
        (WINDOW_SIZE - 1) / 100,
        (WINDOW_SIZE + HOP - 1) / 100,
    ]


def test_rejects_rows_of_wrong_width() -> None:
    with pytest.raises(ValueError):
        Windower(Windows()).add(np.zeros(19, dtype=np.float32), 0.0)


def aligned_xyzw(x: float, y: float, z: float, w: float) -> list[float]:
    """Independent reference: q ⊗ conj([w=0, x=√½, y=√½, z=0]) written out in closed form."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    rw, rx, ry, rz = C * (x + y), C * (z - w), -C * (w + z), C * (y - x)
    n2 = math.sqrt(rw * rw + rx * rx + ry * ry + rz * rz) or 1.0
    return [10000 * v / n2 for v in (rx, ry, rz, rw)]


def window_with(columns: dict[int, float]) -> NDArray[np.float32]:
    raw = np.zeros((WINDOW_SIZE, 20), dtype=np.float32)
    for col, value in columns.items():
        raw[:, col] = value
    return raw


def test_accelerations_are_converted_from_mg_to_mps2_and_gyro_is_unchanged() -> None:
    raw = window_with({0: 1000.0, 1: -500.0, 2: 2000.0, 3: 12.5, 10: 250.0, 15: -400.0})
    out = preprocess(raw)
    np.testing.assert_allclose(out[0, [0, 1, 2, 10]], [9.80665, -4.903325, 19.6133, 2.4516625])
    np.testing.assert_array_equal(out[0, [3, 15]], [12.5, -400.0])
    assert out.dtype == np.float32


@pytest.mark.parametrize(
    "quat",
    [
        (0.0, 0.0, 0.0, 10000.0),
        (1000.0, -2000.0, 3000.0, 9000.0),
        (5000.0, 5000.0, 5000.0, 5000.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_quaternions_are_frame_aligned_on_both_wrists(quat: tuple[float, ...]) -> None:
    raw = window_with({6: quat[0], 7: quat[1], 8: quat[2], 9: quat[3]})
    raw[:, 16:20] = raw[:, 6:10]
    out = preprocess(raw)
    expected = aligned_xyzw(*quat)
    np.testing.assert_allclose(out[0, 6:10], expected, atol=0.01)
    np.testing.assert_allclose(out[0, 16:20], expected, atol=0.01)


def test_identity_quaternion_maps_to_known_frame() -> None:
    out = preprocess(window_with({9: 10000.0}))
    np.testing.assert_allclose(out[0, 6:10], [-7071.0678, -7071.0678, 0.0, 0.0], atol=0.01)


def test_preprocess_does_not_mutate_input() -> None:
    raw = window_with({0: 1000.0, 9: 10000.0})
    before = raw.copy()
    preprocess(raw)
    np.testing.assert_array_equal(raw, before)
