"""HandPerception: fake source + fake estimator, latest-wins load, stop deadline (F-C)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from cpsa.vision.calibration import DepthModel, Intrinsics
from cpsa.vision.hand3d import HandFrame
from cpsa.vision.hand_perception import TIMING_KEYS, HandPerception
from cpsa.vision.irdepth import IrDepthFrame, ir_to_input
from cpsa.vision.types import PersonPose

WIDTH, HEIGHT = 8, 6
INTRINSICS = Intrinsics(fx=8.0, fy=8.0, cx=4.0, cy=3.0, width=WIDTH, height=HEIGHT)
DEPTH = DepthModel(scale_m=1e-3, offset_m=0.0)
IR = np.arange(WIDTH * HEIGHT, dtype=np.uint8).reshape(HEIGHT, WIDTH) * 4
DEPTH_RAW = np.full((HEIGHT, WIDTH), 500, dtype=np.uint16)


def wait_until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.002)


def person(score: float = 0.9) -> PersonPose:
    xy = np.tile(np.array([[4.0, 3.0]], dtype=np.float64), (21, 1))
    return PersonPose(
        box=np.array([1.0, 1.0, 7.0, 5.0], dtype=np.float64),
        score=score,
        xy=xy,
        conf=np.full(21, 0.8, dtype=np.float64),
    )


class FakeSource:
    """An ``IrDepthSource``-like source the test pushes frames into."""

    def __init__(self) -> None:
        self.running = False
        self.on_frame: Callable[[IrDepthFrame], None] | None = None
        self.stops = 0

    def start(self, on_frame: Callable[[IrDepthFrame], None]) -> None:
        self.on_frame = on_frame
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.stops += 1

    def push(
        self,
        seq: int,
        t: float = 0.0,
        ir: np.ndarray | None = None,
        depth_raw: np.ndarray | None = None,
    ) -> None:
        assert self.on_frame is not None
        self.on_frame(
            IrDepthFrame(
                seq=seq,
                t=t,
                width=WIDTH,
                height=HEIGHT,
                ir=IR if ir is None else ir,
                depth_raw=DEPTH_RAW if depth_raw is None else depth_raw,
                decode_ms=0.5,
            )
        )


class FakeEstimator:
    def __init__(self, persons: list[PersonPose] | None = None) -> None:
        self.persons = [person()] if persons is None else persons
        self.inputs: list[np.ndarray] = []
        self.gate: threading.Event | None = None
        self.started = threading.Event()

    def estimate(self, rgb: np.ndarray) -> list[PersonPose]:
        self.inputs.append(rgb)
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(5.0)
        return list(self.persons)


def test_delivers_hand_frames_built_from_the_ir_model_input() -> None:
    source = FakeSource()
    estimator = FakeEstimator()
    frames: list[HandFrame] = []
    perception = HandPerception(source, estimator, INTRINSICS, DEPTH)
    perception.start(frames.append)
    try:
        source.push(7, t=12.5)
        wait_until(lambda: len(frames) == 1)
    finally:
        assert perception.stop(1.0)

    frame = frames[0]
    assert (frame.seq, frame.t, frame.width, frame.height) == (7, 12.5, WIDTH, HEIGHT)
    assert set(frame.timing) == set(TIMING_KEYS)
    assert all(isinstance(value, float) for value in frame.timing.values())
    assert frame.timing["decode_ms"] == pytest.approx(0.5)

    assert len(frame.hands) == 1
    hand = frame.hands[0]
    assert hand.score == pytest.approx(0.9)
    assert hand.box == (1.0, 1.0, 7.0, 5.0)
    assert hand.kpts2d.shape == (21, 3)
    assert hand.kpts2d.dtype == np.float32
    np.testing.assert_allclose(hand.kpts2d[:, 0], 4.0)
    np.testing.assert_allclose(hand.kpts2d[:, 1], 3.0)
    np.testing.assert_allclose(hand.kpts2d[:, 2], 0.8)
    assert hand.kpts3d.shape == (21, 4)
    np.testing.assert_allclose(hand.kpts3d[:, 2], 0.5, atol=1e-6)
    np.testing.assert_array_equal(hand.kpts3d[:, 3], 1.0)

    assert len(estimator.inputs) == 1
    assert estimator.inputs[0].shape == (HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(estimator.inputs[0], ir_to_input(IR))


def test_latest_frame_wins_while_the_estimator_is_busy() -> None:
    source = FakeSource()
    estimator = FakeEstimator()
    estimator.gate = threading.Event()
    frames: list[HandFrame] = []
    perception = HandPerception(source, estimator, INTRINSICS, DEPTH, max_fps=1000.0)
    perception.start(frames.append)
    try:
        source.push(0)
        wait_until(estimator.started.is_set)
        for seq in range(1, 10):
            source.push(seq)
        time.sleep(0.01)
        estimator.gate.set()
        wait_until(lambda: len(frames) >= 2)
    finally:
        assert perception.stop(1.0)

    assert [frame.seq for frame in frames] == [0, 9]
    assert len(estimator.inputs) == 2


def test_stop_returns_true_and_closes_the_source() -> None:
    source = FakeSource()
    frames: list[HandFrame] = []
    perception = HandPerception(source, FakeEstimator(), INTRINSICS, DEPTH)
    perception.start(frames.append)
    source.push(0)
    wait_until(lambda: len(frames) == 1)

    assert perception.stop(1.0) is True
    assert not perception.running
    assert not source.running
    assert source.stops == 1
    assert perception.stop(1.0) is True


def test_stop_reports_a_loop_that_misses_the_timeout() -> None:
    source = FakeSource()
    estimator = FakeEstimator()
    estimator.gate = threading.Event()
    frames: list[HandFrame] = []
    perception = HandPerception(source, estimator, INTRINSICS, DEPTH)
    perception.start(frames.append)
    source.push(0)
    wait_until(estimator.started.is_set)
    thread = perception._thread
    assert thread is not None

    assert perception.stop(0.05) is False
    estimator.gate.set()
    thread.join(2.0)
    assert not thread.is_alive()
    assert frames == []
    assert not source.running


def test_health_reports_camera_fps_and_dpu_ms() -> None:
    source = FakeSource()
    frames: list[HandFrame] = []
    perception = HandPerception(source, FakeEstimator(), INTRINSICS, DEPTH)
    perception.start(frames.append)
    try:
        source.push(0)
        wait_until(lambda: len(frames) == 1)
        time.sleep(0.04)
        source.push(1)
        wait_until(lambda: len(frames) == 2)

        health = perception.health()
        assert set(health) == {"camera", "fps", "dpu_ms"}
        assert health["camera"] is True
        assert health["fps"] > 0.0
        assert health["dpu_ms"] >= 0.0
    finally:
        assert perception.stop(1.0)

    assert perception.health()["camera"] is False


def test_an_estimator_failure_does_not_stop_the_loop() -> None:
    class FlakyEstimator(FakeEstimator):
        def estimate(self, rgb: np.ndarray) -> list[PersonPose]:
            persons = super().estimate(rgb)
            if len(self.inputs) == 1:
                raise RuntimeError("DPU failure")
            return persons

    source = FakeSource()
    estimator = FlakyEstimator()
    frames: list[HandFrame] = []
    perception = HandPerception(source, estimator, INTRINSICS, DEPTH)
    perception.start(frames.append)
    try:
        source.push(0)
        wait_until(estimator.started.is_set)
        time.sleep(0.04)
        source.push(1)
        wait_until(lambda: len(frames) == 1)
    finally:
        assert perception.stop(1.0)

    assert [frame.seq for frame in frames] == [1]


def test_max_fps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_fps must be positive"):
        HandPerception(FakeSource(), FakeEstimator(), INTRINSICS, DEPTH, max_fps=0.0)
