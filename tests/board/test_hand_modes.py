"""``HologramMode`` and ``RpsMode`` against fake camera and estimator (AC-08)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cpsa.application.config import (
    Config,
    HandsConfig,
    HandsModelConfig,
    HandsRpsConfig,
    load,
)
from cpsa.board.hand_modes import HologramMode, RpsMode
from cpsa.supervisor.contract import HEALTH_KEYS, Mode, ModeContext, ModeStartError
from cpsa.vision.irdepth import IrDepthFrame
from cpsa.vision.types import PersonPose
from tests.application.test_config import SHIPPED
from tests.fakes import FakeClock

WIDTH, HEIGHT = 8, 6


def wait_until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.002)


class FakePublisher:
    """Record every state; the mode passes plain ``data`` objects."""

    def __init__(self) -> None:
        self.states: list[tuple[str, Mapping[str, Any]]] = []
        self._lock = threading.Lock()

    def state(self, type: str, data: Mapping[str, Any]) -> None:
        with self._lock:
            self.states.append((type, dict(data)))

    def event(self, type: str, data: Mapping[str, Any]) -> None:
        raise AssertionError(f"unexpected event {type}")

    def payloads(self, message_type: str) -> list[Mapping[str, Any]]:
        with self._lock:
            return [dict(data) for kind, data in self.states if kind == message_type]

    def last(self, message_type: str) -> Mapping[str, Any] | None:
        payloads = self.payloads(message_type)
        return payloads[-1] if payloads else None


class FakeSource:
    """An ``IrDepthSource``-like source the test pushes INZI frames into."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.on_frame: Callable[[IrDepthFrame], None] | None = None
        self.starts = 0
        self.stops = 0

    def start(self, on_frame: Callable[[IrDepthFrame], None]) -> None:
        self.starts += 1
        if self.error is not None:
            raise self.error
        self.on_frame = on_frame

    def stop(self) -> None:
        self.stops += 1


class FakeEstimator:
    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, rgb: np.ndarray) -> list[PersonPose]:
        self.calls += 1
        xy = np.tile(np.array([[4.0, 3.0]], dtype=np.float64), (21, 1))
        return [
            PersonPose(
                box=np.array([1.0, 1.0, 7.0, 5.0], dtype=np.float64),
                score=0.9,
                xy=xy,
                conf=np.full(21, 0.8, dtype=np.float64),
            )
        ]


class FakeClassifier:
    def __init__(self, name: str = "rock", probability: float = 0.9) -> None:
        self.name = name
        self.probability = probability
        self.paths: list[Path] = []
        self.calls = 0

    def predict(self, kpts2d: np.ndarray) -> tuple[str, float]:
        assert kpts2d.shape == (21, 3)
        self.calls += 1
        return self.name, self.probability


class ScriptedRandom:
    def __init__(self, move: str = "scissors") -> None:
        self.move = move

    def choice(self, seq: Sequence[str]) -> str:
        return self.move


def hands_config(tmp_path: Path, **rps_overrides: object) -> HandsConfig:
    model = tmp_path / "hand21_topdown.xmodel"
    model.write_bytes(b"hand model")
    classifier = tmp_path / "rps_mlp.npz"
    classifier.write_bytes(b"classifier")
    rps = HandsRpsConfig(classifier_path=classifier, **rps_overrides)  # type: ignore[arg-type]
    return HandsConfig(model=HandsModelConfig(path=model), rps=rps)


def config_with(tmp_path: Path, **rps_overrides: object) -> Config:
    return replace(load(SHIPPED), hands=hands_config(tmp_path, **rps_overrides))


def new_ir_frame(seq: int = 1) -> IrDepthFrame:
    return IrDepthFrame(
        seq=seq,
        t=float(seq),
        width=WIDTH,
        height=HEIGHT,
        ir=np.arange(WIDTH * HEIGHT, dtype=np.uint8).reshape(HEIGHT, WIDTH) * 4,
        depth_raw=np.zeros((HEIGHT, WIDTH), dtype=np.uint16),
        decode_ms=0.5,
    )


def push(source: FakeSource, seq: int = 1) -> None:
    assert source.on_frame is not None
    source.on_frame(new_ir_frame(seq))


def context(config: Config, publisher: FakePublisher | None = None) -> ModeContext:
    return ModeContext(
        publisher if publisher is not None else FakePublisher(), config, threading.Event()
    )


def push_until(
    source: FakeSource,
    condition: Callable[[], bool],
    timeout: float = 2.0,
) -> None:
    """Push frames until ``condition`` holds; the analysis loop drops frames when busy."""

    deadline = time.monotonic() + timeout
    seq = 1
    while time.monotonic() < deadline:
        push(source, seq)
        seq += 1
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached in time")


# -- the Mode contract -----------------------------------------------------


def test_both_modes_satisfy_the_mode_protocol(tmp_path: Path) -> None:
    config = config_with(tmp_path)

    assert isinstance(HologramMode(config), Mode)
    assert isinstance(RpsMode(config), Mode)


def test_the_hand_modes_require_the_hands_section() -> None:
    with pytest.raises(ValueError, match="hands"):
        HologramMode(load(SHIPPED))
    with pytest.raises(ValueError, match="hands"):
        RpsMode(load(SHIPPED))


# -- hologram --------------------------------------------------------------


def test_hologram_publishes_frames_and_reports_health(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource()
    estimator = FakeEstimator()
    mode = HologramMode(
        config,
        source_factory=lambda _hands: source,
        estimator_factory=lambda _hands: estimator,
    )
    publisher = FakePublisher()
    ctx = context(config, publisher)

    mode.start(ctx)
    try:
        push_until(source, lambda: bool(publisher.payloads("hologram.hands")))
        payload = publisher.last("hologram.hands")

        assert payload is not None
        assert payload["seq"] == 1
        assert (payload["width"], payload["height"]) == (WIDTH, HEIGHT)
        assert len(payload["hands"]) == 1
        assert len(payload["hands"][0]["kpts2d"]) == 21
        assert estimator.calls >= 1

        health = mode.health()
        assert set(health) == set(HEALTH_KEYS)
        assert health["imu"] == "n/a"
        assert health["model"] == "hand21_topdown"
        assert health["errors"] == []
        wait_until(lambda: mode.health()["camera"] == "ok")
        assert mode.health()["dpu_ms"] is not None
    finally:
        assert mode.stop(1.0) is True

    assert source.stops == 1
    assert mode.health()["camera"] is None

    published = len(publisher.payloads("hologram.hands"))
    push(source, 99)
    time.sleep(0.05)
    assert len(publisher.payloads("hologram.hands")) == published


def test_hologram_reports_a_missing_camera(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource(FileNotFoundError("/dev/video0"))
    mode = HologramMode(
        config,
        source_factory=lambda _hands: source,
        estimator_factory=lambda _hands: FakeEstimator(),
    )
    ctx = context(config)

    with pytest.raises(ModeStartError) as info:
        mode.start(ctx)

    assert info.value.code == "camera_missing"
    assert mode.health()["camera"] == "missing"
    assert mode.health()["errors"] == ["camera_missing"]


def test_hologram_reports_a_missing_model_before_touching_the_camera(tmp_path: Path) -> None:
    config = replace(
        load(SHIPPED),
        hands=HandsConfig(
            model=HandsModelConfig(path=tmp_path / "absent.xmodel"),
            rps=None,
        ),
    )
    source = FakeSource()
    mode = HologramMode(
        config,
        source_factory=lambda _hands: source,
        estimator_factory=lambda _hands: pytest.fail("the estimator must not be built"),
    )

    with pytest.raises(ModeStartError) as info:
        mode.start(context(config))

    assert info.value.code == "model_missing"
    assert source.starts == 0
    assert mode.health()["errors"] == ["model_missing"]


def test_hologram_refuses_a_second_start(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    mode = HologramMode(
        config,
        source_factory=lambda _hands: FakeSource(),
        estimator_factory=lambda _hands: FakeEstimator(),
    )
    ctx = context(config)
    mode.start(ctx)
    try:
        with pytest.raises(ModeStartError) as info:
            mode.start(ctx)
        assert info.value.code == "busy"
    finally:
        assert mode.stop(1.0)


# -- rps -------------------------------------------------------------------


def rps_mode(
    tmp_path: Path,
    *,
    source: FakeSource | None = None,
    classifier: FakeClassifier | None = None,
    clock: FakeClock | None = None,
    **rps_overrides: object,
) -> tuple[RpsMode, FakeSource, FakeClassifier, FakeClock, FakeEstimator]:
    config = config_with(
        tmp_path,
        ready_s=0.02,
        countdown_step_s=0.02,
        shoot_s=0.2,
        result_s=0.5,
        **rps_overrides,
    )
    camera = source if source is not None else FakeSource()
    recognizer = classifier if classifier is not None else FakeClassifier()
    fake_clock = clock if clock is not None else FakeClock(100.0)
    estimator = FakeEstimator()
    mode = RpsMode(
        config,
        source_factory=lambda _hands: camera,
        estimator_factory=lambda _hands: estimator,
        classifier_factory=lambda path: recognizer,
        clock=fake_clock.monotonic,
        rng=ScriptedRandom("scissors"),
    )
    return mode, camera, recognizer, fake_clock, estimator


def test_rps_starts_in_waiting_and_plays_a_round(tmp_path: Path) -> None:
    mode, source, classifier, clock, estimator = rps_mode(tmp_path)
    publisher = FakePublisher()
    ctx = context(config_with(tmp_path), publisher)

    def last() -> Mapping[str, Any]:
        payload = publisher.last("rps.state")
        assert payload is not None
        return payload

    mode.start(ctx)
    try:
        # The initial state is published before the first frame.
        assert last()["phase"] == "waiting"

        push_until(source, lambda: last()["phase"] == "ready")
        clock.advance(0.02)
        push_until(source, lambda: last()["count"] == 3)
        clock.advance(0.02)
        push_until(source, lambda: last()["count"] == 2)
        clock.advance(0.02)
        push_until(source, lambda: last()["count"] == 1)
        clock.advance(0.02)
        push_until(source, lambda: last()["phase"] == "shoot")
        # One analyzed frame inside the shoot window casts a vote.
        processed = estimator.calls
        clock.advance(0.05)
        push_until(source, lambda: estimator.calls > processed)
        clock.advance(0.2)
        push_until(source, lambda: last()["phase"] == "result")

        result = last()
        assert result["player"] == "rock"
        assert result["player_p"] == pytest.approx(1.0)
        assert result["board"] == "scissors"
        assert result["outcome"] == "win"
        assert result["score"] == {"player": 1, "board": 0, "draws": 0}
        assert len(result["hand"]["kpts2d"]) == 21
        assert classifier.calls >= 1

        health = mode.health()
        assert set(health) == set(HEALTH_KEYS)
        assert health["model"] == "hand21_topdown"
        assert health["imu"] == "n/a"
    finally:
        assert mode.stop(1.0) is True

    assert source.stops == 1


def test_rps_reports_a_missing_classifier_before_touching_the_camera(tmp_path: Path) -> None:
    model = tmp_path / "hand21_topdown.xmodel"
    model.write_bytes(b"hand model")
    config = replace(
        load(SHIPPED),
        hands=HandsConfig(
            model=HandsModelConfig(path=model),
            rps=HandsRpsConfig(classifier_path=tmp_path / "absent.npz"),
        ),
    )
    source = FakeSource()
    mode = RpsMode(
        config,
        source_factory=lambda _hands: source,
        estimator_factory=lambda _hands: pytest.fail("the estimator must not be built"),
        classifier_factory=lambda _path: pytest.fail("the classifier must not be loaded"),
        clock=FakeClock(100.0).monotonic,
    )

    with pytest.raises(ModeStartError) as info:
        mode.start(context(config))

    assert info.value.code == "model_missing"
    assert source.starts == 0
    assert mode.health()["errors"] == ["model_missing"]


def test_rps_reports_a_missing_camera(tmp_path: Path) -> None:
    source = FakeSource(FileNotFoundError("/dev/video0"))
    mode, _, _, _, _ = rps_mode(tmp_path, source=source)

    with pytest.raises(ModeStartError) as info:
        mode.start(context(config_with(tmp_path)))

    assert info.value.code == "camera_missing"
    assert mode.health()["camera"] == "missing"


def test_a_corrupt_classifier_is_an_invalid_asset(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    corrupt = RpsMode(
        config,
        source_factory=lambda _hands: FakeSource(),
        estimator_factory=lambda _hands: FakeEstimator(),
        classifier_factory=lambda _path: (_ for _ in ()).throw(ValueError("bad npz")),
        clock=FakeClock(100.0).monotonic,
    )

    with pytest.raises(ModeStartError) as info:
        corrupt.start(context(config))

    assert info.value.code == "asset_invalid"
    assert corrupt.health()["errors"] == ["asset_invalid"]


def test_stop_before_start_is_a_no_op(tmp_path: Path) -> None:
    mode, source, _, _, _ = rps_mode(tmp_path)

    assert mode.stop(0.0) is True
    assert source.starts == 0
    assert mode.health()["camera"] is None


# -- supervisor registration -----------------------------------------------


def test_the_supervisor_registers_both_modes_only_with_a_hands_section(
    tmp_path: Path,
) -> None:
    from cpsa.board.supervisor_main import _load_hand_modes

    assert _load_hand_modes(load(SHIPPED)) == {}

    modes = _load_hand_modes(config_with(tmp_path))

    assert set(modes) == {"hologram", "rps"}
    assert isinstance(modes["hologram"], HologramMode)
    assert isinstance(modes["rps"], RpsMode)
