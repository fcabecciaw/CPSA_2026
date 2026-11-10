"""RGB/depth pairing: latency tolerance, matching, drops, counters, bounds."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

import pytest

from cpsa.vision.pairing import PairedFrame, Pairer, PairingStats

FPS = 30.0
TOLERANCE = 0.5 / FPS


def rgb(value: int = 1) -> bytes:
    return f"rgb-{value}".encode()


def depth(value: int = 2) -> bytes:
    return f"depth-{value}".encode()


def test_in_order_frames_pair_and_increment_sequence() -> None:
    pairer = Pairer(FPS)

    assert pairer.push_rgb(10.0, rgb(7)) is None
    first = pairer.push_depth(10.004, depth(9))

    assert first is not None
    assert first.seq == 0
    assert first.ts_rgb == 10.0
    assert first.ts_depth == 10.004
    assert first.rgb == rgb(7)
    assert first.depth == depth(9)
    assert pairer.pending == (0, 0)

    assert pairer.push_depth(10.5, depth(1)) is None
    second = pairer.push_rgb(10.51, rgb(2))
    assert second is not None
    assert second.seq == 1
    assert pairer.stats() == PairingStats(paired=2, dropped_rgb=0, dropped_depth=0)


def test_depth_can_arrive_first() -> None:
    pairer = Pairer(FPS)

    assert pairer.push_depth(20.0, depth()) is None
    pair = pairer.push_rgb(20.01, rgb())

    assert pair is not None
    assert pair.seq == 0
    assert pair.ts_rgb == 20.01
    assert pair.ts_depth == 20.0


def test_jitter_within_tolerance_pairs_each_frame() -> None:
    pairer = Pairer(FPS)
    pairs: list[PairedFrame] = []

    timestamps = [
        (100.000, 100.004),
        (100.033, 100.029),
        (100.067, 100.071),
        (100.100, 100.096),
        (100.133, 100.130),
    ]
    for index, (ts_rgb, ts_depth) in enumerate(timestamps):
        if index % 2:
            assert pairer.push_depth(ts_depth, depth(index)) is None
            emitted = pairer.push_rgb(ts_rgb, rgb(index))
        else:
            assert pairer.push_rgb(ts_rgb, rgb(index)) is None
            emitted = pairer.push_depth(ts_depth, depth(index))
        assert emitted is not None
        pairs.append(emitted)

    assert [pair.seq for pair in pairs] == list(range(len(timestamps)))
    assert all(abs(pair.ts_rgb - pair.ts_depth) <= TOLERANCE for pair in pairs)
    assert pairer.stats() == PairingStats(len(timestamps), 0, 0)


def test_stream_delayed_by_250ms_pairs_every_frame() -> None:
    pairer = Pairer(FPS)
    count = 60
    lag_frames = 8  # 266 ms, slightly worse than the measured 250 ms
    pairs: list[PairedFrame] = []

    for index in range(count + lag_frames):
        if index < count:
            emitted = pairer.push_depth(index / FPS, depth(index))
            if emitted is not None:
                pairs.append(emitted)
        rgb_index = index - lag_frames
        if 0 <= rgb_index < count:
            emitted = pairer.push_rgb(rgb_index / FPS, rgb(rgb_index))
            if emitted is not None:
                pairs.append(emitted)

    assert len(pairs) == count
    assert [pair.seq for pair in pairs] == list(range(count))
    assert [pair.ts_rgb for pair in pairs] == pytest.approx([index / FPS for index in range(count)])
    assert [pair.ts_depth for pair in pairs] == [pair.ts_rgb for pair in pairs]
    assert [pair.rgb for pair in pairs] == [rgb(index) for index in range(count)]
    assert pairer.stats() == PairingStats(count, 0, 0)
    assert pairer.pending == (0, 0)


@pytest.mark.parametrize("offset", [0.0, TOLERANCE / 2, TOLERANCE, -TOLERANCE / 2, -TOLERANCE])
def test_constant_phase_offset_up_to_tolerance_pairs(offset: float) -> None:
    pairer = Pairer(FPS)

    assert pairer.push_rgb(10.0, rgb()) is None
    pair = pairer.push_depth(10.0 + offset, depth())

    assert pair is not None
    assert pair.ts_rgb == 10.0
    assert pair.ts_depth == 10.0 + offset
    assert pairer.stats() == PairingStats(1, 0, 0)


def test_missing_frames_on_one_stream_are_dropped() -> None:
    pairer = Pairer(FPS)

    pairer.push_rgb(0.0, rgb())
    assert pairer.push_depth(0.0, depth()) is not None

    assert pairer.push_depth(1 / FPS, depth()) is None
    assert pairer.push_depth(2 / FPS, depth()) is None
    pair = pairer.push_rgb(2 / FPS, rgb())
    assert pair is not None

    assert pairer.push_rgb(3 / FPS, rgb()) is None
    assert pairer.push_rgb(4 / FPS, rgb()) is None
    pairer.flush()

    stats = pairer.stats()
    assert stats.paired == 2
    assert stats.dropped_rgb == 2
    assert stats.dropped_depth == 1


def test_out_of_tolerance_frames_are_not_paired() -> None:
    pairer = Pairer(FPS)

    pairer.push_rgb(1.0, rgb())
    assert pairer.push_depth(1.0 + 2 * TOLERANCE, depth()) is None
    assert pairer.stats() == PairingStats(0, 1, 0)

    assert pairer.push_rgb(1.1, rgb()) is None
    assert pairer.stats() == PairingStats(0, 1, 1)


def test_late_frame_older_than_the_waiting_stream_is_dropped() -> None:
    pairer = Pairer(FPS)

    assert pairer.push_depth(10.0, depth()) is None
    assert pairer.push_rgb(9.9, rgb()) is None
    assert pairer.stats() == PairingStats(0, 1, 0)

    pair = pairer.push_rgb(10.001, rgb())
    assert pair is not None
    assert pair.seq == 0
    assert pairer.stats() == PairingStats(1, 1, 0)


def test_flush_counts_all_waiting_streams() -> None:
    pairer = Pairer(FPS)
    pairer.push_rgb(1.0, rgb())
    pairer.push_depth(1.0, depth())
    pairer.push_rgb(2.0, rgb())
    pairer.push_depth(5.0, depth())  # drops the stale rgb and waits for depth

    pairer.flush()
    stats = pairer.stats()
    assert stats.paired == 1
    assert stats.dropped_rgb == 1
    assert stats.dropped_depth == 1
    assert pairer.pending == (0, 0)


def test_horizon_and_max_count_bound_memory() -> None:
    pairer = Pairer(FPS, horizon_s=0.1)
    for index in range(20):
        assert pairer.push_rgb(index / FPS, rgb(index)) is None

    pending_rgb, pending_depth = pairer.pending
    assert pending_depth == 0
    assert 0 < pending_rgb <= 4
    assert pairer.stats().dropped_rgb == 20 - pending_rgb

    bounded = Pairer(FPS, horizon_s=10.0, max_pending=3)
    for index in range(10):
        bounded.push_rgb(index / FPS, rgb(index))
    assert bounded.pending == (3, 0)
    pair = bounded.push_depth(9 / FPS, depth())
    assert pair is not None
    assert pair.rgb == rgb(9)
    assert bounded.pending == (0, 0)
    assert bounded.stats() == PairingStats(1, 9, 0)


def test_tolerance_is_half_a_frame() -> None:
    assert Pairer(30.0).tolerance == pytest.approx(1 / 60)
    assert Pairer(60.0).tolerance == pytest.approx(1 / 120)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True])
def test_invalid_fps_is_rejected(fps: float) -> None:
    with pytest.raises(ValueError):
        Pairer(fps)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon_s": 0},
        {"horizon_s": float("nan")},
        {"horizon_s": True},
        {"max_pending": 0},
        {"max_pending": -1},
        {"max_pending": True},
        {"max_pending": 1.5},
    ],
)
def test_invalid_buffer_bounds_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Pairer(FPS, **kwargs)  # type: ignore[arg-type]


def test_nonfinite_timestamp_is_rejected() -> None:
    pairer = Pairer(FPS)
    with pytest.raises(ValueError):
        pairer.push_rgb(float("nan"), rgb())
    with pytest.raises(ValueError):
        pairer.push_depth(float("inf"), depth())


def test_concurrent_pushes_pair_every_timestamp() -> None:
    pairer = Pairer(FPS)
    count = 300
    barrier = threading.Barrier(2)
    pairs: list[PairedFrame] = []
    lock = threading.Lock()

    def run(push: Callable[[float, bytes], PairedFrame | None], values: list[bytes]) -> None:
        for index, image in enumerate(values):
            pair = push(index / FPS, image)
            if pair is not None:
                with lock:
                    pairs.append(pair)
            barrier.wait(timeout=5)

    rgb_frames = [rgb(index) for index in range(count)]
    depth_frames = [depth(index) for index in range(count)]
    rgb_thread = threading.Thread(target=run, args=(pairer.push_rgb, rgb_frames))
    depth_thread = threading.Thread(target=run, args=(pairer.push_depth, depth_frames))
    rgb_thread.start()
    depth_thread.start()
    rgb_thread.join(timeout=10)
    depth_thread.join(timeout=10)

    assert not rgb_thread.is_alive() and not depth_thread.is_alive()
    assert len(pairs) == count
    assert sorted(pair.seq for pair in pairs) == list(range(count))
    assert pairer.stats() == PairingStats(count, 0, 0)


def test_horizon_must_cover_the_measured_delay() -> None:
    # A deliberately short horizon loses old frames: this is the bounded-memory
    # trade-off, not a matching bug.
    pairer = Pairer(FPS, horizon_s=0.1)
    for index in range(40):
        pairer.push_depth(index / FPS, depth(index))
    pair = pairer.push_rgb(0.0, rgb(0))
    assert pair is None
    assert pairer.stats().dropped_depth > 0
    assert pairer.pending[1] <= math.ceil(0.1 * FPS) + 1
