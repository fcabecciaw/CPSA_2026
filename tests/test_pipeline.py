"""The detection loop: pipelining, publishing, throttling and camera recovery."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from cpsa.camera import CameraNotFound, VideoFrame
from cpsa.channel import FrameChannel
from cpsa.config import (
    CameraConfig,
    LogConfig,
    ModelConfig,
    ShowcaseConfig,
    TrackerConfig,
    WebConfig,
)
from cpsa.pipeline import Pipeline, StartError, cap_admits
from cpsa.yolox import Detection

WIDTH, HEIGHT = 8, 4


class Sink:
    """Records what the pipeline publishes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.messages: list[tuple[str, dict]] = []

    def put(self, topic: str, data: dict) -> bool:
        with self._lock:
            self.messages.append((topic, dict(data)))
        return True

    def topics(self, topic: str) -> list[dict]:
        with self._lock:
            return [data for name, data in self.messages if name == topic]


class FakeCamera:
    def __init__(self) -> None:
        self.running = False
        self.starts = 0
        self.stops = 0
        self.on_frame: Callable[[VideoFrame], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None
        self._seq = 0

    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.on_frame = on_frame
        self.on_error = on_error
        self.running = True
        self.starts += 1

    def stop(self) -> None:
        self.running = False
        self.stops += 1

    def push(self, value: int = 0) -> None:
        if not self.running or self.on_frame is None:
            raise AssertionError("the camera is not running")
        rgb = np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)
        frame = VideoFrame(
            seq=self._seq, t=time.monotonic(), width=WIDTH, height=HEIGHT, decode=lambda: rgb
        )
        self._seq += 1
        self.on_frame(frame)

    def fail(self, error: Exception) -> None:
        self.running = False
        if self.on_error is not None:
            self.on_error(error)


class FakeDetector:
    def __init__(self, detections: Sequence[Detection] = ()) -> None:
        self.detections = list(detections)
        self.stage_ms = {"preprocess": 1.0, "dpu": 2.0, "decode": 0.5}
        self.calls: list[str] = []
        self._token = 0

    def submit(self, rgb: np.ndarray) -> int:
        self.calls.append("submit")
        self._token += 1
        return self._token

    def wait(self, token: int) -> list[Detection]:
        self.calls.append("wait")
        return list(self.detections)


class SyncDetector:
    """A detector with only ``detect``: the synchronous path."""

    def __init__(self, detections: Sequence[Detection] = ()) -> None:
        self.detections = list(detections)
        self.stage_ms = {"preprocess": 1.0, "dpu": 2.0, "decode": 0.5}
        self.calls = 0

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        self.calls += 1
        return list(self.detections)


PERSON = Detection(box=(1.0, 1.0, 5.0, 3.0), score=0.75, class_id=0)

DEFAULT_TRACKER = TrackerConfig(
    enabled=True,
    iou_match=0.3,
    alpha=0.35,
    fast_alpha=0.8,
    deadband_iou=0.9,
    min_hits=2,
    max_misses=6,
)


def feed(camera: FakeCamera, count: int = 3, *, value: int = 0, delay: float = 0.03) -> None:
    """Push frames slowly enough that the loop takes each one from the slot."""
    for _ in range(count):
        camera.push(value)
        time.sleep(delay)


def config(
    tmp_path: Path,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    tracker: TrackerConfig | None = None,
) -> ShowcaseConfig:
    model = tmp_path / "yolox_nano.xmodel"
    model.write_bytes(b"fake xmodel")
    return ShowcaseConfig(
        camera=CameraConfig(
            device="auto",
            match="RealSense",
            width=width,
            height=height,
            fps=60.0,
            exposure=100,
            gain=300,
            white_balance=4000,
            backlight_compensation=0,
            saturation=110,
            power_line_hz=50,
        ),
        tracker=tracker if tracker is not None else DEFAULT_TRACKER,
        model=ModelConfig(path=model, conf_thr=0.35, iou_thr=0.45),
        labels="it",
        web=WebConfig(
            host="127.0.0.1",
            port=8080,
            video_fps=60.0,
            jpeg_quality=70,
            static_dir=Path("web/dist"),
        ),
        log=LogConfig(level="INFO"),
    )


def build(
    tmp_path: Path,
    *,
    detector: object | None = None,
    camera: FakeCamera | None = None,
    tracker: TrackerConfig | None = None,
) -> tuple[Pipeline, FakeCamera, Sink, FrameChannel]:
    camera = camera if camera is not None else FakeCamera()
    sink = Sink()
    channel = FrameChannel.create(WIDTH, HEIGHT)
    pipeline = Pipeline(
        config(tmp_path, tracker=tracker),
        channel=channel,
        messages=sink,  # type: ignore[arg-type]
        detector_factory=lambda model: detector if detector is not None else FakeDetector(),
        camera_factory=lambda settings: camera,
    )
    return pipeline, camera, sink, channel


def wait_until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.002)


class RunningPipeline:
    """Start a pipeline in a thread and stop it at the end of the test."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.stop = threading.Event()
        self.thread = threading.Thread(target=pipeline.run, args=(self.stop,), daemon=True)

    def __enter__(self) -> RunningPipeline:
        self.pipeline.start()
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop.set()
        self.thread.join(timeout=3.0)
        assert not self.thread.is_alive()


# --------------------------------------------------------------------------- #
# The frame cadence (a8614b1)
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Starting
# --------------------------------------------------------------------------- #


def test_a_missing_model_fails_before_the_camera_opens(tmp_path: Path) -> None:
    camera = FakeCamera()
    pipeline, _, _, channel = build(tmp_path, camera=camera)
    try:
        pipeline._config = ShowcaseConfig(  # type: ignore[misc]
            camera=pipeline._config.camera,
            tracker=pipeline._config.tracker,
            model=ModelConfig(path=tmp_path / "missing.xmodel", conf_thr=0.35, iou_thr=0.45),
            labels="it",
            web=pipeline._config.web,
            log=pipeline._config.log,
        )
        with pytest.raises(StartError) as excinfo:
            pipeline.start()
    finally:
        channel.close()
        channel.unlink()

    assert excinfo.value.code == "model_missing"
    assert camera.starts == 0


def test_a_detector_that_cannot_load_is_a_dpu_error(tmp_path: Path) -> None:
    def factory(model: object) -> FakeDetector:
        raise RuntimeError("no VART")

    sink = Sink()
    channel = FrameChannel.create(WIDTH, HEIGHT)
    camera = FakeCamera()
    pipeline = Pipeline(
        config(tmp_path),
        channel=channel,
        messages=sink,  # type: ignore[arg-type]
        detector_factory=factory,
        camera_factory=lambda settings: camera,
    )
    try:
        with pytest.raises(StartError) as excinfo:
            pipeline.start()
    finally:
        channel.close()
        channel.unlink()

    assert excinfo.value.code == "dpu_error"
    assert camera.starts == 0


@pytest.mark.parametrize(
    ("error", "code"),
    [(CameraNotFound("no node"), "camera_missing"), (RuntimeError("busy"), "camera_error")],
)
def test_a_camera_that_cannot_open_reports_the_failure(
    tmp_path: Path, error: Exception, code: str
) -> None:
    class Broken(FakeCamera):
        def start(self, on_frame, on_error=None) -> None:
            raise error

    sink = Sink()
    channel = FrameChannel.create(WIDTH, HEIGHT)
    pipeline = Pipeline(
        config(tmp_path),
        channel=channel,
        messages=sink,  # type: ignore[arg-type]
        detector_factory=lambda model: FakeDetector(),
        camera_factory=lambda settings: Broken(),
    )
    try:
        with pytest.raises(StartError) as excinfo:
            pipeline.start()
    finally:
        channel.close()
        channel.unlink()

    assert excinfo.value.code == code


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_a_frame_reaches_the_video_and_the_state_with_labels(tmp_path: Path) -> None:
    detector = FakeDetector([PERSON])
    pipeline, camera, sink, channel = build(tmp_path, detector=detector)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            # One frame is in flight while the previous one is decoded, so the
            # first published state follows the second frame.
            feed(camera, value=90)
            wait_until(lambda: sink.topics("showcase.state"))
            state = sink.topics("showcase.state")[0]
            frame = channel.read()
    finally:
        channel.close()
        channel.unlink()

    assert state["model"] == "yolox_nano"
    assert state["width"] == WIDTH and state["height"] == HEIGHT
    assert state["counts"] == [{"label": "persona", "count": 1}]
    assert state["objects"][0]["label"] == "persona"
    assert state["objects"][0]["class_id"] == 0
    assert frame is not None
    seq, image = frame
    assert seq >= 1
    assert not np.all(image == 90)  # the box and the tag were drawn
    assert image[2, 3].tolist() != [90, 90, 90]  # inside the PERSON box


def test_the_tracker_keeps_the_published_box_still(tmp_path: Path) -> None:
    detector = FakeDetector()
    pipeline, camera, sink, channel = build(tmp_path, detector=detector)
    offsets = [-0.05, 0.05, -0.03, 0.03, 0.0, -0.05]
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            for offset in offsets:
                detector.detections = [
                    Detection(box=(1.0 + offset, 1.0, 5.0 + offset, 3.0), score=0.8, class_id=0)
                ]
                camera.push()
                time.sleep(0.06)
            wait_until(lambda: len(sink.topics("showcase.state")) >= 2)
    finally:
        channel.close()
        channel.unlink()

    boxes = {
        tuple(item["box"]) for state in sink.topics("showcase.state") for item in state["objects"]
    }
    # The first published box is whichever jittered frame was born; from then
    # on the tracker must never move it.
    assert len(boxes) == 1


def test_a_disabled_tracker_publishes_the_raw_boxes(tmp_path: Path) -> None:
    detector = FakeDetector()
    disabled = TrackerConfig(
        enabled=False,
        iou_match=0.3,
        alpha=0.35,
        fast_alpha=0.8,
        deadband_iou=0.9,
        min_hits=2,
        max_misses=6,
    )
    pipeline, camera, sink, channel = build(tmp_path, detector=detector, tracker=disabled)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            for offset in (0.0, 0.0, 0.0, 0.75, 0.75, 0.75):
                detector.detections = [
                    Detection(box=(1.0 + offset, 1.0, 5.0 + offset, 3.0), score=0.8, class_id=0)
                ]
                camera.push()
                time.sleep(0.08)
            wait_until(lambda: len(sink.topics("showcase.state")) >= 2)
    finally:
        channel.close()
        channel.unlink()

    boxes = {
        tuple(item["box"]) for state in sink.topics("showcase.state") for item in state["objects"]
    }
    assert len(boxes) >= 2


def test_a_detector_without_submit_runs_the_synchronous_path(tmp_path: Path) -> None:
    detector = SyncDetector([PERSON])
    pipeline, camera, sink, channel = build(tmp_path, detector=detector)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera)
            wait_until(lambda: sink.topics("showcase.state"))
            # States are throttled to 10 Hz; feed over more than one period.
            feed(camera, 6, delay=0.06)
            wait_until(lambda: len(sink.topics("showcase.state")) >= 2)
    finally:
        channel.close()
        channel.unlink()

    assert detector.calls >= 2


def test_the_state_is_throttled_to_ten_hertz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cpsa import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "STATE_PERIOD_S", 10.0)
    pipeline, camera, sink, channel = build(tmp_path)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 5)
            wait_until(lambda: sink.topics("showcase.state"))
            time.sleep(0.1)
    finally:
        channel.close()
        channel.unlink()

    assert len(sink.topics("showcase.state")) == 1


def test_the_next_frame_is_submitted_before_the_previous_is_waited_for(
    tmp_path: Path,
) -> None:
    detector = FakeDetector([PERSON])
    pipeline, camera, sink, channel = build(tmp_path, detector=detector)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 4)
            wait_until(lambda: detector.calls.count("wait") >= 2)
    finally:
        channel.close()
        channel.unlink()

    # The pipeline submits the second frame before it waits for the first one:
    # that overlap is what makes a 60 fps loop possible.
    assert detector.calls[:3] == ["submit", "submit", "wait"]
    assert detector.calls.count("submit") >= detector.calls.count("wait") >= 2


def test_a_detector_failure_does_not_stop_the_loop(tmp_path: Path) -> None:
    class Flaky(FakeDetector):
        def submit(self, rgb: np.ndarray) -> int:
            if not self.calls:
                self.calls.append("submit")
                raise RuntimeError("DPU job failed")
            return super().submit(rgb)

    detector = Flaky([PERSON])
    pipeline, camera, sink, channel = build(tmp_path, detector=detector)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            camera.push()
            wait_until(
                lambda: sink.topics("status") and "dpu_error" in sink.topics("status")[-1]["errors"]
            )
            feed(camera)
            wait_until(lambda: sink.topics("showcase.state"))
    finally:
        channel.close()
        channel.unlink()

    assert sink.topics("showcase.state")


def test_the_camera_error_is_published_and_the_camera_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cpsa import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "CAMERA_RETRY_S", 0.01)
    camera = FakeCamera()
    pipeline, _, sink, channel = build(tmp_path, camera=camera)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera)
            wait_until(lambda: sink.topics("showcase.state"))
            camera.fail(CameraNotFound("unplugged"))

            def recovered() -> bool:
                """A ``status`` with ``camera: ok`` published after the failure."""
                seen_missing = False
                for status in sink.topics("status"):
                    if status["camera"] == "missing":
                        seen_missing = True
                    elif seen_missing and status["camera"] == "ok":
                        return True
                return False

            wait_until(lambda: camera.starts == 2)
            wait_until(recovered)
            statuses = sink.topics("status")
    finally:
        channel.close()
        channel.unlink()

    assert camera.stops >= 1
    assert any(status["camera"] == "missing" for status in statuses)
    assert any("camera_missing" in status["errors"] for status in statuses)


def test_stop_releases_the_camera_and_the_detector(tmp_path: Path) -> None:
    detector = FakeDetector()
    pipeline, camera, _sink, channel = build(tmp_path, detector=detector)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
        wait_until(lambda: camera.stops >= 1)
        assert pipeline._detector is None  # type: ignore[attr-defined]
    finally:
        channel.close()
        channel.unlink()

    assert camera.stops >= 1
    assert not camera.running


def test_the_stage_timings_are_logged_once_per_period(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cpsa import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "TIMING_PERIOD_S", 0.01)
    pipeline, camera, sink, channel = build(tmp_path)
    try:
        with (
            caplog.at_level(logging.INFO, logger="cpsa.pipeline"),
            RunningPipeline(pipeline),
        ):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 8, delay=0.02)
            wait_until(lambda: "showcase stages over" in caplog.text)
    finally:
        channel.close()
        channel.unlink()

    assert "showcase stages over" in caplog.text
    assert "track=" in caplog.text
    assert "total_fps=" in caplog.text


def test_status_reports_starting_before_the_loop_and_running_after(tmp_path: Path) -> None:
    pipeline, camera, sink, channel = build(tmp_path)
    try:
        assert pipeline.status()["state"] == "starting"
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera)
            wait_until(lambda: sink.topics("status"))
            running = pipeline.status()
    finally:
        channel.close()
        channel.unlink()

    assert running["state"] == "running"
    assert running["camera"] == "ok"
    assert running["model"] == "yolox_nano"
    assert running["fps"] is not None and running["fps"] >= 0.0
