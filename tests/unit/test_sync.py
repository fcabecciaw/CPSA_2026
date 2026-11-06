import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import SampleKind, Side
from cpsa.imu.sync import IMUSynchronizer
from tests.fakes import FakeClock

SKEW_S = 0.030
STALE_S = 0.050


class Rows:
    def __init__(self) -> None:
        self.rows: list[tuple[list[float], float]] = []

    def __call__(self, row: NDArray[np.float32], ts: float) -> None:
        self.rows.append((row.tolist(), ts))


def make(clock: FakeClock) -> tuple[IMUSynchronizer, Rows]:
    rows = Rows()
    return IMUSynchronizer(SKEW_S, STALE_S, rows, clock), rows


def send_triplet(sync: IMUSynchronizer, side: Side, base: float, ts: float) -> None:
    sync.update(side, SampleKind.ACC, (base, base + 1, base + 2), ts)
    sync.update(side, SampleKind.GYR, (base + 3, base + 4, base + 5), ts)
    sync.update(side, SampleKind.QUAT, (base + 6, base + 7, base + 8, base + 9), ts)


def test_aligned_triplets_emit_right_then_left_with_newest_timestamp() -> None:
    clock = FakeClock(start=10.0)
    sync, rows = make(clock)
    send_triplet(sync, Side.LEFT, base=100.0, ts=10.000)
    send_triplet(sync, Side.RIGHT, base=0.0, ts=10.020)

    assert rows.rows == [([float(v) for v in range(10)] + [100.0 + v for v in range(10)], 10.020)]


def test_skew_equal_to_limit_is_combined() -> None:
    clock = FakeClock(start=10.0)
    sync, rows = make(clock)
    send_triplet(sync, Side.LEFT, 0.0, ts=10.0)
    send_triplet(sync, Side.RIGHT, 0.0, ts=10.0 + SKEW_S)
    assert len(rows.rows) == 1


def test_both_triplets_are_cleared_after_emitting() -> None:
    clock = FakeClock(start=10.0)
    sync, rows = make(clock)
    send_triplet(sync, Side.LEFT, 0.0, ts=10.0)
    send_triplet(sync, Side.RIGHT, 0.0, ts=10.0)
    send_triplet(sync, Side.RIGHT, 0.0, ts=10.01)
    assert len(rows.rows) == 1


def test_skew_above_limit_is_not_combined_and_older_side_is_dropped() -> None:
    clock = FakeClock(start=10.0)
    sync, rows = make(clock)
    send_triplet(sync, Side.LEFT, 100.0, ts=10.000)
    send_triplet(sync, Side.RIGHT, 0.0, ts=10.040)
    assert rows.rows == []

    # The left triplet was dropped: a fresh left triplet now pairs with the kept right one.
    send_triplet(sync, Side.LEFT, 200.0, ts=10.045)
    assert len(rows.rows) == 1
    assert rows.rows[0][0][10] == 200.0
    assert rows.rows[0][1] == 10.045


def test_incomplete_triplet_is_not_combined() -> None:
    clock = FakeClock(start=10.0)
    sync, rows = make(clock)
    send_triplet(sync, Side.LEFT, 0.0, ts=10.0)
    sync.update(Side.RIGHT, SampleKind.ACC, (1.0, 2.0, 3.0), 10.0)
    sync.update(Side.RIGHT, SampleKind.GYR, (1.0, 2.0, 3.0), 10.0)
    assert rows.rows == []


def test_stale_triplet_is_discarded_before_alignment() -> None:
    clock = FakeClock(start=10.0)
    sync, rows = make(clock)
    send_triplet(sync, Side.LEFT, 100.0, ts=10.0)

    clock.advance(STALE_S + 0.001)
    # Right samples carry timestamps within skew of the left ones, but the left
    # triplet has been waiting longer than the stale limit.
    send_triplet(sync, Side.RIGHT, 0.0, ts=10.0)
    assert rows.rows == []

    send_triplet(sync, Side.LEFT, 300.0, ts=clock.t)
    send_triplet(sync, Side.RIGHT, 0.0, ts=clock.t)
    assert [r[0][10] for r in rows.rows] == [300.0]


def test_malformed_sample_is_ignored() -> None:
    clock = FakeClock(start=10.0)
    sync, rows = make(clock)
    sync.update(Side.LEFT, SampleKind.QUAT, (1.0, 2.0), 10.0)
    sync.update(Side.LEFT, SampleKind.ACC, ("x", 1.0, 2.0), 10.0)  # type: ignore[arg-type]
    send_triplet(sync, Side.RIGHT, 0.0, ts=10.0)
    sync.update(Side.LEFT, SampleKind.GYR, (1.0, 2.0, 3.0), 10.0)
    assert rows.rows == []
