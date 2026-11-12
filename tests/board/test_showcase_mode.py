"""``ShowcaseMode`` against a fake camera, detector and publisher (AC-S3)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.application.config import (
    Config,
    ShowcaseConfig,
    ShowcaseModelConfig,
    load,
)
from cpsa.board.showcase_mode import (
    ShowcaseMode,
    cap_admits,
    class_color,
    draw_overlay,
    showcase_camera,
)
from cpsa.protocol.showcase import ShowcaseObject
from cpsa.supervisor.contract import HEALTH_KEYS, Mode, ModeContext, ModeStartError
from cpsa.vision.types import VideoFrame
from tests.application.test_config import SHIPPED

WIDTH, HEIGHT = 16, 12

ITALIAN = ("persona", "bicicletta", "auto")


def wait_until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.002)


class FakePublisher:
    """Record every state and the video frames offered to ``/video.mjpeg``."""

    def __init__(self) -> None:
        self.states: list[tuple[str, Mapping[str, Any]]] = []
        self.video: list[tuple[int, NDArray[np.uint8]]] = []
        self._lock = threading.Lock()

    def state(self, type: str, data: Mapping[str, Any]) -> None:
        with self._lock:
            self.states.append((type, dict(data)))

    def event(self, type: str, data: Mapping[str, Any]) -> None:
        raise AssertionError(f"unexpected event {type}")

    def offer_video_frame(self, seq: int, image: NDArray[np.uint8]) -> None:
        with self._lock:
            self.video.append((int(seq), np.array(image, copy=True)))

    def payloads(self, message_type: str) -> list[Mapping[str, Any]]:
        with self._lock:
            return [dict(data) for kind, data in self.states if kind == message_type]

    def last(self, message_type: str) -> Mapping[str, Any] | None:
        payloads = self.payloads(message_type)
        return payloads[-1] if payloads else None

    def frames(self) -> list[tuple[int, NDArray[np.uint8]]]:
        with self._lock:
            return list(self.video)


class FakeDetection:
    """One detection in the F-S1 shape (frozen attributes, tuple box)."""

    def __init__(
        self,
        box: tuple[float, float, float, float],
        score: float,
        class_id: int,
    ) -> None:
        self.box = box
        self.score = score
        self.class_id = class_id


class FakeDetector:
    def __init__(self, detections: list[FakeDetection] | None = None) -> None:
        self.detections = (
            detections
            if detections is not None
            else [FakeDetection((2.0, 2.0, 10.0, 10.0), 0.91, 0)]
        )
        self.paths: list[Path] = []
        self.inputs: list[tuple[int, ...]] = []
        self.stage_ms: dict[str, float] = {"preprocess": 1.5, "dpu": 4.2, "decode": 0.6}

    def detect(self, rgb: NDArray[np.uint8]) -> list[FakeDetection]:
        self.inputs.append(tuple(int(value) for value in rgb.shape))
        return list(self.detections)


class FakePipelinedDetector(FakeDetector):
    """A detector with the DPU ``submit``/``wait`` split of the real adapter."""

    def __init__(self, detections: list[FakeDetection] | None = None) -> None:
        super().__init__(detections)
        self.calls: list[tuple[str, int]] = []
        self._tokens: dict[int, NDArray[np.uint8]] = {}
        self._next = 0

    def submit(self, rgb: NDArray[np.uint8]) -> int:
        self.inputs.append(tuple(int(value) for value in rgb.shape))
        self._next += 1
        self._tokens[self._next] = rgb
        self.calls.append(("submit", self._next))
        return self._next

    def wait(self, token: int) -> list[FakeDetection]:
        self.calls.append(("wait", token))
        self._tokens.pop(token, None)
        return list(self.detections)


class FakeSource:
    """An RGB source the test pushes frames into."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.on_frame: Callable[[VideoFrame], None] | None = None
        self.starts = 0
        self.stops = 0

    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.starts += 1
        if self.error is not None:
            raise self.error
        self.on_frame = on_frame

    def stop(self) -> None:
        self.stops += 1


def showcase_config(tmp_path: Path, **overrides: object) -> ShowcaseConfig:
    model = tmp_path / "yolox_nano.xmodel"
    model.write_bytes(b"detector")
    arguments: dict[str, object] = {
        "default_model": "yolox_nano",
        "models": (ShowcaseModelConfig(name="yolox_nano", kind="yolox", path=model),),
    }
    arguments.update(overrides)
    return ShowcaseConfig(**arguments)  # type: ignore[arg-type]


def config_with(tmp_path: Path, **overrides: object) -> Config:
    return replace(load(SHIPPED), showcase=showcase_config(tmp_path, **overrides))


def frame(seq: int = 1, value: int = 0) -> VideoFrame:
    image = np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)

    def decode() -> NDArray[np.uint8]:
        return image.copy()

    return VideoFrame(seq=seq, t=float(seq), width=WIDTH, height=HEIGHT, decode=decode)


def context(config: Config, publisher: FakePublisher | None = None) -> ModeContext:
    return ModeContext(
        publisher if publisher is not None else FakePublisher(), config, threading.Event()
    )


def mode_with(
    config: Config,
    source: FakeSource,
    detector: FakeDetector | None = None,
) -> ShowcaseMode:
    return ShowcaseMode(
        config,
        source_factory=lambda _camera: source,
        detector_factory=lambda _model: detector if detector is not None else FakeDetector(),
        labels_factory=lambda _language: ITALIAN,
    )


def push_until(
    source: FakeSource,
    condition: Callable[[], bool],
    timeout: float = 2.0,
) -> None:
    """Push frames until ``condition`` holds; the loop drops frames when busy."""

    deadline = time.monotonic() + timeout
    seq = 1
    while time.monotonic() < deadline:
        assert source.on_frame is not None
        source.on_frame(frame(seq))
        seq += 1
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached in time")


def last_objects(publisher: FakePublisher) -> list[Any]:
    payload = publisher.last("showcase.state")
    return [] if payload is None else list(payload["objects"])


def push(source: FakeSource, seq: int = 1) -> None:
    assert source.on_frame is not None
    source.on_frame(frame(seq))


# -- the Mode contract -----------------------------------------------------


def test_the_showcase_mode_satisfies_the_mode_protocol(tmp_path: Path) -> None:
    assert isinstance(ShowcaseMode(config_with(tmp_path)), Mode)


def test_the_showcase_mode_requires_the_showcase_section() -> None:
    with pytest.raises(ValueError, match="showcase"):
        ShowcaseMode(load(SHIPPED))


# -- start failures --------------------------------------------------------


def test_a_missing_model_file_fails_before_the_camera_opens(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    assert config.showcase is not None
    config.showcase.models[0].path.unlink()
    source = FakeSource()
    mode = mode_with(config, source)

    with pytest.raises(ModeStartError) as info:
        mode.start(context(config))

    assert info.value.code == "model_missing"
    assert source.starts == 0
    assert mode.health()["errors"] == ["model_missing"]
    assert mode.health()["camera"] is None


def test_a_detector_that_cannot_load_is_a_dpu_error(tmp_path: Path) -> None:
    config = config_with(tmp_path)

    def broken(_model: ShowcaseModelConfig) -> FakeDetector:
        raise RuntimeError("no DPU")

    mode = ShowcaseMode(
        config,
        source_factory=lambda _camera: FakeSource(),
        detector_factory=broken,
        labels_factory=lambda _language: ITALIAN,
    )

    with pytest.raises(ModeStartError) as info:
        mode.start(context(config))

    assert info.value.code == "dpu_error"
    assert mode.health()["errors"] == ["dpu_error"]


def test_a_missing_rgb_node_fails_with_camera_missing(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource(error=FileNotFoundError("/dev/video0"))
    mode = mode_with(config, source)

    with pytest.raises(ModeStartError) as info:
        mode.start(context(config))

    assert info.value.code == "camera_missing"
    assert source.stops == 1
    health = mode.health()
    assert health["camera"] == "missing"
    assert health["errors"] == ["camera_missing"]


def test_a_camera_that_cannot_be_configured_is_a_camera_error(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource(error=RuntimeError("no camera node matching 'RealSense'"))

    with pytest.raises(ModeStartError) as info:
        mode_with(config, source).start(context(config))

    assert info.value.code == "camera_error"


def test_a_stop_requested_before_start_never_opens_the_camera(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource()
    ctx = context(config)
    ctx.stop_event.set()

    mode_with(config, source).start(ctx)

    assert source.starts == 0


# -- the running mode ------------------------------------------------------


def test_every_analyzed_frame_reaches_the_video_stream(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource()
    detector = FakeDetector()
    publisher = FakePublisher()
    mode = mode_with(config, source, detector)

    mode.start(context(config, publisher))
    try:
        push_until(source, lambda: bool(publisher.frames()))
        # The first pushed frame is also the first processed one: nothing was
        # pending before it, so the loop never has to drop it.
        seq, image = publisher.frames()[0]

        assert seq == 1
        assert detector.inputs, "the detector never saw a frame"
        assert detector.inputs[-1] == (HEIGHT, WIDTH, 3)
        assert image.shape == (HEIGHT, WIDTH, 3)
        # The box of the fake detection is drawn on the copy, not on the frame.
        assert image[2, 2].tolist() == list(class_color(0))
        assert mode.health()["camera"] == "ok"
    finally:
        assert mode.stop(1.0) is True


def test_the_first_state_of_a_start_is_empty_at_the_camera_size(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource()
    publisher = FakePublisher()
    mode = mode_with(config, source)

    mode.start(context(config, publisher))
    try:
        payload = publisher.last("showcase.state")

        assert payload is not None
        assert payload["objects"] == []
        assert payload["counts"] == []
        assert (payload["width"], payload["height"]) == (
            config.vision.camera.width,
            config.vision.camera.height,
        )
        assert payload["fps"] is None
    finally:
        assert mode.stop(1.0) is True


def test_the_detections_reach_the_state_with_their_labels(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource()
    detector = FakeDetector(
        [
            FakeDetection((2.0, 3.0, 12.0, 11.0), 0.82, 1),
            FakeDetection((0.0, 0.0, 8.0, 8.0), 0.95, 0),
        ]
    )
    publisher = FakePublisher()
    mode = mode_with(config, source, detector)

    mode.start(context(config, publisher))
    try:
        push_until(source, lambda: bool(last_objects(publisher)))
        payload = publisher.last("showcase.state")

        assert payload is not None
        assert payload["model"] == "yolox_nano"
        assert (payload["width"], payload["height"]) == (WIDTH, HEIGHT)
        assert [entry["label"] for entry in payload["objects"]] == ["persona", "bicicletta"]
        assert payload["counts"] == [
            {"label": "bicicletta", "count": 1},
            {"label": "persona", "count": 1},
        ]
        assert payload["dpu_ms"] == pytest.approx(4.2)
        assert payload["total_ms"] is not None
    finally:
        assert mode.stop(1.0) is True


def test_the_state_is_throttled_to_ten_hertz(tmp_path: Path) -> None:
    config = config_with(tmp_path, max_fps=200.0)
    source = FakeSource()
    publisher = FakePublisher()
    mode = mode_with(config, source)

    mode.start(context(config, publisher))
    try:
        deadline = time.monotonic() + 0.3
        seq = 1
        while time.monotonic() < deadline:
            push(source, seq)
            seq += 1
            time.sleep(0.002)
        # Ten per second over 0.3 s: four states at most, one is always allowed.
        published = len(publisher.payloads("showcase.state"))
        assert 1 <= published <= 5
        assert len(publisher.frames()) > published
    finally:
        assert mode.stop(1.0) is True


def test_health_reports_the_f4_keys_and_the_model(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource()
    publisher = FakePublisher()
    mode = mode_with(config, source)

    mode.start(context(config, publisher))
    try:
        push_until(source, lambda: (mode.health()["fps"] or 0.0) > 0.0)
        health = mode.health()

        assert set(health) == set(HEALTH_KEYS)
        assert health["imu"] == "n/a"
        assert health["model"] == "yolox_nano"
        assert health["errors"] == []
        assert health["camera"] == "ok"
        assert health["fps"] is not None and health["fps"] > 0.0
        assert health["dpu_ms"] == pytest.approx(4.2)
    finally:
        assert mode.stop(1.0) is True


def test_stop_releases_the_camera_and_the_loop(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource()
    publisher = FakePublisher()
    mode = mode_with(config, source)

    mode.start(context(config, publisher))
    push_until(source, lambda: bool(publisher.frames()))
    assert mode.stop(1.0) is True

    assert source.stops == 1
    health = mode.health()
    assert health["camera"] is None
    assert health["fps"] is None
    assert health["dpu_ms"] is None
    assert health["errors"] == []

    published = len(publisher.payloads("showcase.state"))
    offered = len(publisher.frames())
    push(source, 99)
    time.sleep(0.05)
    assert len(publisher.payloads("showcase.state")) == published
    assert len(publisher.frames()) == offered


def test_the_supervisor_registers_the_mode_only_with_a_showcase_section(
    tmp_path: Path,
) -> None:
    from cpsa.board.supervisor_main import _load_showcase_mode

    assert _load_showcase_mode(load(SHIPPED)) == {}

    modes = _load_showcase_mode(config_with(tmp_path))

    assert set(modes) == {"showcase"}
    assert isinstance(modes["showcase"], ShowcaseMode)


def test_a_second_start_while_running_is_busy(tmp_path: Path) -> None:
    config = config_with(tmp_path)
    source = FakeSource()
    mode = mode_with(config, source)

    mode.start(context(config))
    try:
        with pytest.raises(ModeStartError) as info:
            mode.start(context(config))
        assert info.value.code == "busy"
    finally:
        assert mode.stop(1.0) is True


def test_a_detector_crash_does_not_stop_the_loop(tmp_path: Path) -> None:
    config = config_with(tmp_path)

    class FlakyDetector(FakeDetector):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def detect(self, rgb: NDArray[np.uint8]) -> list[FakeDetection]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("one bad frame")
            return super().detect(rgb)

    source = FakeSource()
    publisher = FakePublisher()
    mode = mode_with(config, source, FlakyDetector())

    mode.start(context(config, publisher))
    try:
        push_until(source, lambda: bool(publisher.frames()))
        assert publisher.last("showcase.state") is not None
    finally:
        assert mode.stop(1.0) is True


# -- the overlay -----------------------------------------------------------


def test_the_overlay_is_a_copy_of_the_frame() -> None:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    objects = [ShowcaseObject("persona", 0, 0.9, (2.0, 2.0, 12.0, 10.0))]

    overlay = draw_overlay(image, objects)

    assert overlay.shape == image.shape
    assert overlay.dtype == image.dtype
    assert not np.shares_memory(overlay, image)
    assert not np.array_equal(overlay, image)
    assert np.array_equal(image, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))


def test_the_overlay_clips_boxes_to_the_frame() -> None:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    objects = [ShowcaseObject("tazza", 41, 0.5, (-40.0, -12.0, 4 * WIDTH, 5 * HEIGHT))]

    overlay = draw_overlay(image, objects)

    assert overlay.shape == (HEIGHT, WIDTH, 3)


def test_each_class_keeps_its_colour() -> None:
    assert class_color(0) == class_color(0)
    assert class_color(0) != class_color(1)
    # The first twelve classes are visually distinct and stay put across runs.
    assert len({class_color(index) for index in range(12)}) == 12


# -- the 60 fps path: pipelining, camera overrides and stage timings (S5) ----


def test_the_pipelined_detector_submits_the_next_frame_before_waiting() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        config = config_with(Path(directory), max_fps=1e6)
        source = FakeSource()
        detector = FakePipelinedDetector()
        publisher = FakePublisher()
        mode = mode_with(config, source, detector)  # type: ignore[arg-type]

        mode.start(context(config, publisher))
        try:
            push(source, 1)
            wait_until(lambda: detector.calls[:1] == [("submit", 1)])
            # The first frame is in flight: nothing has been waited for yet.
            assert ("wait", 1) not in detector.calls
            push(source, 2)
            # Frame 2 is submitted before frame 1 is waited for: the DPU and the
            # CPU work on different frames.
            wait_until(lambda: len(detector.calls) >= 3)
            assert detector.calls[:3] == [
                ("submit", 1),
                ("submit", 2),
                ("wait", 1),
            ]
            wait_until(lambda: any(seq == 1 for seq, _image in publisher.frames()))
        finally:
            assert mode.stop(1.0) is True


def test_a_pipelined_detector_still_offers_every_finished_frame() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        config = config_with(Path(directory), max_fps=1e6)
        source = FakeSource()
        publisher = FakePublisher()
        mode = mode_with(config, source, FakePipelinedDetector())  # type: ignore[arg-type]

        mode.start(context(config, publisher))
        try:
            push_until(source, lambda: len(publisher.frames()) >= 4, timeout=3.0)
            sequences = [seq for seq, _image in publisher.frames()]
            # The pacing cap and the latest-frame slot may drop a frame, never
            # duplicate or reorder one.
            assert sequences[0] == 1
            assert sequences == sorted(set(sequences))
        finally:
            assert mode.stop(1.0) is True


def test_the_showcase_camera_applies_only_the_configured_overrides(tmp_path: Path) -> None:
    base = config_with(tmp_path).vision.camera
    overrides = showcase_config(tmp_path, camera_fps=60.0, exposure=150, gain=320)

    camera = showcase_camera(base, overrides)

    assert (camera.fps, camera.exposure, camera.gain) == (60.0, 150, 320)
    assert (camera.device, camera.width, camera.height) == (
        base.device,
        base.width,
        base.height,
    )


def test_the_showcase_camera_keeps_the_vision_camera_without_overrides(
    tmp_path: Path,
) -> None:
    config = config_with(tmp_path)
    assert config.showcase is not None

    assert showcase_camera(config.vision.camera, config.showcase) is config.vision.camera


def test_the_camera_overrides_reach_only_the_showcase_source(tmp_path: Path) -> None:
    config = config_with(tmp_path, camera_fps=60.0, exposure=150, gain=320)
    seen: list[Any] = []

    def factory(camera: Any) -> FakeSource:
        seen.append(camera)
        return FakeSource()

    mode = ShowcaseMode(
        config,
        source_factory=factory,
        detector_factory=lambda _model: FakeDetector(),
        labels_factory=lambda _language: ITALIAN,
    )
    mode.start(context(config))
    try:
        assert seen
        assert (seen[0].fps, seen[0].exposure, seen[0].gain) == (60.0, 150, 320)
        # The medical pipeline keeps the shipped RGB settings.
        assert config.vision.camera.fps != seen[0].fps or config.vision.camera.exposure != 150
        assert config.vision.camera is not seen[0]
    finally:
        assert mode.stop(1.0) is True


def test_the_stage_timings_are_logged_once_per_period(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    monkeypatch.setattr("cpsa.board.showcase_mode.TIMING_PERIOD_S", 0.0)
    caplog.set_level("INFO", logger="cpsa.board.showcase")

    with tempfile.TemporaryDirectory() as directory:
        config = config_with(Path(directory), max_fps=1e6)
        source = FakeSource()
        publisher = FakePublisher()
        mode = mode_with(config, source, FakePipelinedDetector())  # type: ignore[arg-type]

        mode.start(context(config, publisher))
        try:
            push_until(source, lambda: len(publisher.frames()) >= 2)
        finally:
            assert mode.stop(1.0) is True

    lines = [
        record.getMessage()
        for record in caplog.records
        if "showcase stages over" in record.getMessage()
    ]
    assert lines
    assert "camera=" in lines[-1]
    assert "preprocess=" in lines[-1]
    assert "dpu=" in lines[-1]
    assert "decode=" in lines[-1]
    assert "draw=" in lines[-1]
    assert "total=" in lines[-1]


class FailingOnceDetector(FakePipelinedDetector):
    """A pipeline whose first wait fails: the loop must keep its next token."""

    def __init__(self) -> None:
        super().__init__()
        self.waits = 0

    def wait(self, token: int) -> list[FakeDetection]:
        self.waits += 1
        if self.waits == 1:
            super().wait(token)
            raise RuntimeError("decode blew up")
        return super().wait(token)


def test_a_failed_wait_does_not_wedge_the_pipeline(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import tempfile

    caplog.set_level("ERROR", logger="cpsa.board.showcase")
    with tempfile.TemporaryDirectory() as directory:
        config = config_with(Path(directory), max_fps=1e6)
        source = FakeSource()
        detector = FailingOnceDetector()
        publisher = FakePublisher()
        mode = mode_with(config, source, detector)  # type: ignore[arg-type]

        mode.start(context(config, publisher))
        try:
            push_until(source, lambda: len(publisher.frames()) >= 3, timeout=3.0)
            sequences = [seq for seq, _image in publisher.frames()]
            # Frame 1 is lost with its failed decode; the frames after it arrive.
            assert sequences[0] == 2
            assert sequences == sorted(set(sequences))
        finally:
            assert mode.stop(1.0) is True
    assert detector.waits >= 3


def _admitted(arrivals: list[float], max_fps: float) -> int:
    next_due = 0.0
    count = 0
    for now in arrivals:
        admit, next_due = cap_admits(now, next_due, 1.0 / max_fps)
        count += admit
    return count


def test_the_cap_keeps_every_jittered_frame_at_the_camera_rate() -> None:
    # 60 fps camera with +-1 ms jitter: a strict cadence test dropped about a third.
    jitter = [0.001 if index % 2 else -0.001 for index in range(600)]
    arrivals = [1.0 + index / 60.0 + offset for index, offset in enumerate(jitter)]

    assert _admitted(arrivals, max_fps=60.0) == 600


def test_the_cap_still_halves_a_camera_twice_as_fast() -> None:
    arrivals = [1.0 + index / 60.0 for index in range(600)]

    assert _admitted(arrivals, max_fps=30.0) == 300


def test_a_late_frame_does_not_build_credit_for_a_burst() -> None:
    period = 1.0 / 30.0
    admit, next_due = cap_admits(10.0, 1.0, period)

    assert admit is True
    assert cap_admits(10.0 + period / 2, next_due, period)[0] is False
