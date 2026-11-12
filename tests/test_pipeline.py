"""The detection loop: pipelining, publishing, throttling and camera recovery."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cpsa.camera import CameraNotFound, VideoFrame
from cpsa.channel import CommandChannel, FrameChannel
from cpsa.config import (
    CameraConfig,
    LogConfig,
    ModelConfig,
    MovenetConfig,
    ShowcaseConfig,
    SmoothingConfig,
    StillConfig,
    TrackerConfig,
    WebConfig,
)
from cpsa.movenet import MovenetPose
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
        self.closed = False
        self._token = 0

    def submit(self, rgb: np.ndarray) -> int:
        self.calls.append("submit")
        self._token += 1
        return self._token

    def wait(self, token: int) -> list[Detection]:
        self.calls.append("wait")
        return list(self.detections)

    def close(self) -> None:
        """Release the graph and the buffers, as a model swap does (R2-G)."""
        self.closed = True
        self.calls.append("close")


class SyncDetector:
    """A detector with only ``detect``: the synchronous path."""

    def __init__(self, detections: Sequence[Detection] = ()) -> None:
        self.detections = list(detections)
        self.stage_ms = {"preprocess": 1.0, "dpu": 2.0, "decode": 0.5}
        self.calls = 0

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        self.calls += 1
        return list(self.detections)


class FakePoseDetector:
    """The MoveNet runner as the loop sees it: a token and one pose."""

    def __init__(self, pose: MovenetPose | None = None) -> None:
        self.pose = pose if pose is not None else DEFAULT_POSE
        self.stage_ms = {"preprocess": 0.4, "dpu": 1.5, "decode": 0.3}
        self.calls: list[str] = []
        self.closed = False
        self._token = 0

    def submit(self, rgb: np.ndarray) -> int:
        self.calls.append("submit")
        self._token += 1
        return self._token

    def wait(self, token: int) -> MovenetPose:
        self.calls.append("wait")
        return self.pose

    def close(self) -> None:
        """Release the graph and the buffers, as a model swap does (R2-G)."""
        self.closed = True
        self.calls.append("close")


class FakeQueue:
    """The ``multiprocessing`` queue surface the command channel needs."""

    def __init__(self, maxsize: int = 8) -> None:
        self._items: deque[tuple[str, dict[str, Any]]] = deque()
        self._maxsize = maxsize

    def put_nowait(self, item: tuple[str, dict[str, Any]]) -> None:
        if len(self._items) >= self._maxsize:
            import queue

            raise queue.Full
        self._items.append(item)

    def get_nowait(self) -> tuple[str, dict[str, Any]]:
        if not self._items:
            import queue

            raise queue.Empty
        return self._items.popleft()

    def close(self) -> None:
        pass


PERSON = Detection(box=(1.0, 1.0, 5.0, 3.0), score=0.75, class_id=0)
BOTTLE = Detection(box=(2.0, 1.2, 3.0, 2.8), score=0.42, class_id=39)
KEYPOINTS = np.array(
    [[1.0 + index % 6, 1.0 + index // 6, 0.9] for index in range(17)], dtype=np.float32
)
DEFAULT_POSE = MovenetPose(keypoints=KEYPOINTS, score=0.75)

DISABLED_STILL = StillConfig(enabled=False, motion_thr=1.0, force_every_s=0.5)
STILL = StillConfig(enabled=True, motion_thr=1.0, force_every_s=30.0)

DEFAULT_TRACKER = TrackerConfig(
    enabled=True,
    iou_match=0.3,
    alpha=0.35,
    fast_alpha=0.8,
    deadband_iou=0.9,
    min_hits=2,
    max_misses=6,
    still=DISABLED_STILL,
)

DEFAULT_SMOOTHING = SmoothingConfig(
    enabled=True,
    min_cutoff=1.0,
    beta=0.02,
    d_cutoff=1.0,
    deadband_px=1.5,
    show_score=0.35,
    hide_score=0.25,
    show_frames=2,
    hide_frames=4,
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
    movenet_enabled: bool = True,
    publish_hz: float = 30.0,
) -> ShowcaseConfig:
    model = tmp_path / "yolox_nano.xmodel"
    model.write_bytes(b"fake xmodel")
    movenet_model = tmp_path / "movenet.xmodel"
    movenet_model.write_bytes(b"fake movenet xmodel")
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
        movenet=MovenetConfig(
            path=movenet_model,
            enabled=movenet_enabled,
            publish_hz=publish_hz,
            smoothing=DEFAULT_SMOOTHING,
        ),
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
    pose: FakePoseDetector | None = None,
    camera: FakeCamera | None = None,
    tracker: TrackerConfig | None = None,
    commands: CommandChannel | None = None,
    movenet_enabled: bool = True,
    publish_hz: float = 30.0,
    detector_factory: Callable[[ModelConfig], object] | None = None,
    pose_factory: Callable[[MovenetConfig], FakePoseDetector] | None = None,
) -> tuple[Pipeline, FakeCamera, Sink, FrameChannel]:
    camera = camera if camera is not None else FakeCamera()
    pose = pose if pose is not None else FakePoseDetector()
    sink = Sink()
    channel = FrameChannel.create(WIDTH, HEIGHT)
    pipeline = Pipeline(
        config(
            tmp_path,
            tracker=tracker,
            movenet_enabled=movenet_enabled,
            publish_hz=publish_hz,
        ),
        channel=channel,
        messages=sink,  # type: ignore[arg-type]
        detector_factory=detector_factory
        or (lambda model: detector if detector is not None else FakeDetector()),
        movenet_factory=pose_factory or (lambda settings: pose),
        camera_factory=lambda settings: camera,
        commands=commands,
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
            movenet=pipeline._config.movenet,
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
        movenet_factory=lambda settings: FakePoseDetector(),
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


def test_the_state_counts_a_bottle_inside_the_person_box(tmp_path: Path) -> None:
    detector = FakeDetector([PERSON, BOTTLE])
    pipeline, camera, sink, channel = build(tmp_path, detector=detector)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            # The bottle needs its second frame to be shown (score < 0.6) and
            # the state is published at 10 Hz, so wait for a state with both.
            feed(camera, count=10, value=90, delay=0.04)
            wait_until(
                lambda: any(len(state["counts"]) == 2 for state in sink.topics("showcase.state"))
            )
            state = next(item for item in sink.topics("showcase.state") if len(item["counts"]) == 2)
    finally:
        channel.close()
        channel.unlink()

    assert state["counts"] == [
        {"label": "bottiglia", "count": 1},
        {"label": "persona", "count": 1},
    ]
    assert [item["class_id"] for item in state["objects"]] == [PERSON.class_id, BOTTLE.class_id]


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
        still=DISABLED_STILL,
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


# --------------------------------------------------------------------------- #
# The motion gate in front of the DPU (F-S6)
# --------------------------------------------------------------------------- #


def tracker_with_still(**overrides: object) -> TrackerConfig:
    """The default tracker with the motion gate configured for one test."""
    settings: dict[str, object] = {
        "enabled": True,
        "motion_thr": 1.0,
        "force_every_s": 30.0,
    }
    settings.update(overrides)
    return replace(DEFAULT_TRACKER, still=StillConfig(**settings))  # type: ignore[arg-type]


def test_a_still_scene_stops_the_detector_and_keeps_the_published_objects(
    tmp_path: Path,
) -> None:
    detector = FakeDetector([PERSON])
    pipeline, camera, sink, channel = build(
        tmp_path, detector=detector, tracker=tracker_with_still()
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 16, delay=0.02)
            wait_until(lambda: len(sink.topics("showcase.state")) >= 2)
            submits = detector.calls.count("submit")
            before = sink.topics("showcase.state")[-1]["objects"]

            # No frame moves the 8x4 camera: the loop draws the same boxes and
            # keeps publishing state, but the DPU is not touched again.
            feed(camera, 16, delay=0.02)
            wait_until(lambda: len(sink.topics("showcase.state")) >= 3)
            time.sleep(0.05)
            submits_after = detector.calls.count("submit")
            after = sink.topics("showcase.state")[-1]["objects"]
    finally:
        channel.close()
        channel.unlink()

    assert submits > 0
    assert submits_after == submits
    assert after == before
    assert [item["label"] for item in after] == ["persona"]


def test_motion_goes_to_the_detector_at_once(tmp_path: Path) -> None:
    detector = FakeDetector([PERSON])
    pipeline, camera, sink, channel = build(
        tmp_path, detector=detector, tracker=tracker_with_still()
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 12, delay=0.02)
            wait_until(lambda: len(sink.topics("showcase.state")) >= 2)
            submits = detector.calls.count("submit")

            camera.push(200)
            wait_until(lambda: detector.calls.count("submit") > submits)
            submits_after = detector.calls.count("submit")
    finally:
        channel.close()
        channel.unlink()

    assert submits_after == submits + 1


def test_a_still_scene_is_refreshed_every_force_every_s(tmp_path: Path) -> None:
    detector = FakeDetector([PERSON])
    pipeline, camera, sink, channel = build(
        tmp_path,
        detector=detector,
        tracker=tracker_with_still(force_every_s=0.15),
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 6, delay=0.02)
            wait_until(lambda: sink.topics("showcase.state"))
            submits = detector.calls.count("submit")

            # 14 still frames over about 0.4 s: the gate skips most of them but
            # the refresh timer lets a new still object through every 0.15 s.
            feed(camera, 14, delay=0.03)
            time.sleep(0.1)
            submitted = detector.calls.count("submit") - submits
    finally:
        channel.close()
        channel.unlink()

    assert 2 <= submitted < 14


def test_a_disabled_gate_infers_on_every_frame(tmp_path: Path) -> None:
    detector = FakeDetector([PERSON])
    pipeline, camera, sink, channel = build(
        tmp_path, detector=detector, tracker=tracker_with_still(enabled=False)
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 6, delay=0.02)
            wait_until(lambda: sink.topics("showcase.state"))
            submits = detector.calls.count("submit")

            feed(camera, 10, delay=0.03)
            time.sleep(0.1)
            submitted = detector.calls.count("submit") - submits
    finally:
        channel.close()
        channel.unlink()

    assert submitted == 10


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
        assert pipeline._runner is None  # type: ignore[attr-defined]
    finally:
        channel.close()
        channel.unlink()

    assert detector.closed
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
    assert "movenet_dpu" not in caplog.text
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


# --------------------------------------------------------------------------- #
# One DPU model at a time (R2-G)
# --------------------------------------------------------------------------- #


def test_start_loads_yolox_and_leaves_movenet_alone(tmp_path: Path) -> None:
    builds: list[str] = []
    pose = FakePoseDetector()

    def pose_factory(settings: MovenetConfig) -> FakePoseDetector:
        builds.append(settings.name)
        return pose

    detector = FakeDetector([PERSON])
    pipeline, camera, sink, channel = build(tmp_path, detector=detector, pose_factory=pose_factory)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 3)
            wait_until(lambda: sink.topics("showcase.state"))
            status = pipeline.status()
    finally:
        channel.close()
        channel.unlink()

    assert builds == []
    assert pose.calls == []
    assert status["active_model"] == "yolox"
    assert status["switching"] is False
    assert status["swap_ms"] is None


def test_the_swap_drains_and_releases_before_it_builds_the_other_model(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class RecordingDetector(FakeDetector):
        def submit(self, rgb: np.ndarray) -> int:
            events.append("yolox.submit")
            return super().submit(rgb)

        def wait(self, token: int) -> list[Detection]:
            events.append("yolox.wait")
            return super().wait(token)

        def close(self) -> None:
            events.append("yolox.close")
            super().close()

    class RecordingPose(FakePoseDetector):
        def submit(self, rgb: np.ndarray) -> int:
            events.append("movenet.submit")
            return super().submit(rgb)

        def wait(self, token: int) -> MovenetPose:
            events.append("movenet.wait")
            return super().wait(token)

        def close(self) -> None:
            events.append("movenet.close")
            super().close()

    def pose_factory(settings: MovenetConfig) -> RecordingPose:
        events.append("movenet.build")
        return RecordingPose()

    pipeline, camera, _sink, channel = build(
        tmp_path,
        detector_factory=lambda model: RecordingDetector([PERSON]),
        pose_factory=pose_factory,
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 3)
            wait_until(lambda: "yolox.wait" in events)
            assert pipeline.set_model("movenet") is True
            wait_until(lambda: pipeline.status()["active_model"] == "movenet")
            feed(camera, 3)
            wait_until(lambda: "movenet.wait" in events)
            time.sleep(0.05)
    finally:
        channel.close()
        channel.unlink()

    released = events.index("yolox.close")
    assert released < events.index("movenet.build")
    # The job in flight was waited for before the graph and its buffers went.
    assert events[:released].count("yolox.submit") == events[:released].count("yolox.wait")
    # And YOLOX never touches the DPU again while MoveNet is loaded.
    assert "yolox.submit" not in events[released:]
    assert "movenet.submit" in events[released:]


def test_a_set_to_the_loaded_model_is_a_no_op(tmp_path: Path) -> None:
    detector = FakeDetector([PERSON])
    pipeline, camera, _sink, channel = build(tmp_path, detector=detector)
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 2)
            assert pipeline.set_model("yolox") is True
            time.sleep(0.1)
            status = pipeline.status()
            assert pipeline._requested_model is None  # type: ignore[attr-defined]
            assert not detector.closed
    finally:
        channel.close()
        channel.unlink()

    assert status["active_model"] == "yolox"
    assert status["swap_ms"] is None


def test_switching_is_visible_in_status_while_the_model_loads(tmp_path: Path) -> None:
    loading = threading.Event()
    release = threading.Event()

    def pose_factory(settings: MovenetConfig) -> FakePoseDetector:
        loading.set()
        assert release.wait(2.0)
        return FakePoseDetector()

    pipeline, camera, _sink, channel = build(
        tmp_path,
        detector_factory=lambda model: FakeDetector([PERSON]),
        pose_factory=pose_factory,
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 2)
            assert pipeline.set_model("movenet") is True
            wait_until(loading.is_set)
            during = pipeline.status()
            release.set()
            wait_until(lambda: pipeline.status()["active_model"] == "movenet")
            after = pipeline.status()
    finally:
        channel.close()
        channel.unlink()

    assert during["switching"] is True
    assert during["active_model"] == "yolox"
    assert after["switching"] is False
    assert after["swap_ms"] is not None and after["swap_ms"] >= 0.0


def test_a_second_set_during_a_swap_queues_the_way_back(tmp_path: Path) -> None:
    loading = threading.Event()
    release = threading.Event()
    builds: list[str] = []

    def pose_factory(settings: MovenetConfig) -> FakePoseDetector:
        builds.append(settings.name)
        loading.set()
        assert release.wait(2.0)
        return FakePoseDetector()

    pipeline, camera, _sink, channel = build(
        tmp_path,
        detector_factory=lambda model: FakeDetector([PERSON]),
        pose_factory=pose_factory,
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 2)
            assert pipeline.set_model("movenet") is True
            wait_until(loading.is_set)
            # The dashboard clicks YOLOX again while MoveNet is still loading:
            # the request is kept, so the loop swaps back after it.
            assert pipeline.set_model("yolox") is True
            release.set()
            wait_until(lambda: pipeline.status()["active_model"] == "movenet")
            wait_until(lambda: pipeline.status()["active_model"] == "yolox")
            back = pipeline.status()
    finally:
        channel.close()
        channel.unlink()

    assert builds == ["movenet"]
    assert back["switching"] is False
    assert back["swap_ms"] is not None


def test_a_failed_load_restores_the_previous_model(tmp_path: Path) -> None:
    builds: list[str] = []

    def pose_factory(settings: MovenetConfig) -> FakePoseDetector:
        builds.append(settings.name)
        raise RuntimeError("no VART for movenet")

    pipeline, camera, sink, channel = build(
        tmp_path,
        detector_factory=lambda model: FakeDetector([PERSON]),
        pose_factory=pose_factory,
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 2)
            wait_until(lambda: sink.topics("showcase.state"))
            assert pipeline.set_model("movenet") is True
            wait_until(lambda: pipeline.status()["swap_ms"] is not None)
            failed = pipeline.status()
            states_before = len(sink.topics("showcase.state"))
            # The restored model keeps the loop alive, and its good frames do
            # not hide the failure the swap reported.
            feed(camera, 4, delay=0.06)
            wait_until(lambda: sink.topics("showcase.state"))
            time.sleep(0.05)
            recovered = pipeline.status()
    finally:
        channel.close()
        channel.unlink()

    assert builds == ["movenet"]
    assert failed["active_model"] == "yolox"
    assert failed["switching"] is False
    assert "dpu_error" in failed["errors"]
    assert sink.topics("showcase.state")
    assert len(sink.topics("showcase.state")) >= states_before
    assert "dpu_error" in recovered["errors"]


def test_each_view_runs_and_publishes_only_its_own_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cpsa import pipeline as pipeline_module

    drawn: list[str] = []

    def boxes(rgb: np.ndarray, objects: object) -> np.ndarray:
        drawn.append("boxes")
        return rgb

    def skeleton(rgb: np.ndarray, keypoints: object) -> np.ndarray:
        drawn.append("skeleton")
        return rgb

    monkeypatch.setattr(pipeline_module, "draw_overlay", boxes)
    monkeypatch.setattr(pipeline_module, "draw_skeleton", skeleton)
    pipeline, camera, sink, channel = build(
        tmp_path,
        detector_factory=lambda model: FakeDetector([PERSON]),
        pose_factory=lambda settings: FakePoseDetector(),
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 3)
            wait_until(lambda: "boxes" in drawn)
            objects_before = len(sink.topics("showcase.state"))

            assert pipeline.set_model("movenet") is True
            wait_until(lambda: pipeline.status()["active_model"] == "movenet")
            feed(camera, 4, delay=0.08)
            wait_until(lambda: "skeleton" in drawn)
            time.sleep(0.1)
            objects_during = len(sink.topics("showcase.state"))
            poses = len(sink.topics("pose.state"))

            assert pipeline.set_model("yolox") is True
            wait_until(lambda: pipeline.status()["active_model"] == "yolox")
            poses_after_swap = len(sink.topics("pose.state"))
            feed(camera, 4, delay=0.08)
            wait_until(lambda: len(sink.topics("showcase.state")) > objects_during)
            time.sleep(0.1)
            poses_end = len(sink.topics("pose.state"))
    finally:
        channel.close()
        channel.unlink()

    assert drawn[0] == "boxes"
    assert poses >= 1
    # F-E1: each state belongs to the model that produced it.
    assert objects_during == objects_before
    assert poses_end == poses_after_swap


def test_pose_state_is_throttled_to_the_configured_rate(tmp_path: Path) -> None:
    pipeline, camera, sink, channel = build(
        tmp_path,
        publish_hz=5.0,
        detector_factory=lambda model: FakeDetector([PERSON]),
        pose_factory=lambda settings: FakePoseDetector(),
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            assert pipeline.set_model("movenet") is True
            wait_until(lambda: pipeline.status()["active_model"] == "movenet")
            objects = len(sink.topics("showcase.state"))
            feed(camera, 8, delay=0.07)
            wait_until(lambda: sink.topics("pose.state"))
            time.sleep(0.05)
    finally:
        channel.close()
        channel.unlink()

    poses = sink.topics("pose.state")
    assert 1 <= len(poses) <= 4
    assert len(sink.topics("showcase.state")) == objects


def test_a_model_set_command_reaches_the_pipeline(tmp_path: Path) -> None:
    commands = CommandChannel(FakeQueue())
    pipeline, camera, sink, channel = build(
        tmp_path,
        commands=commands,
        detector_factory=lambda model: FakeDetector([PERSON]),
        pose_factory=lambda settings: FakePoseDetector(),
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            commands.put("model.set", {"model": "movenet"})
            wait_until(lambda: pipeline.status()["active_model"] == "movenet")
            feed(camera)
            wait_until(lambda: sink.topics("status")[-1]["active_model"] == "movenet")
    finally:
        channel.close()
        channel.unlink()

    assert sink.topics("status")[-1]["active_model"] == "movenet"
    assert sink.topics("status")[-1]["swap_ms"] is not None


def test_an_unknown_model_is_rejected_without_switching(tmp_path: Path) -> None:
    builds: list[str] = []

    def pose_factory(settings: MovenetConfig) -> FakePoseDetector:
        builds.append(settings.name)
        return FakePoseDetector()

    pipeline, camera, _sink, channel = build(
        tmp_path,
        detector_factory=lambda model: FakeDetector([PERSON]),
        pose_factory=pose_factory,
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            assert pipeline.set_model("medical") is False
            assert pipeline.status()["active_model"] == "yolox"
            assert pipeline.set_model("movenet") is True
            wait_until(lambda: pipeline.status()["active_model"] == "movenet")
    finally:
        channel.close()
        channel.unlink()

    assert builds == ["movenet"]


def test_movenet_disabled_rejects_the_swap(tmp_path: Path) -> None:
    calls: list[str] = []

    def factory(settings: object) -> FakePoseDetector:
        calls.append("built")
        return FakePoseDetector()

    sink = Sink()
    channel = FrameChannel.create(WIDTH, HEIGHT)
    camera = FakeCamera()
    pipeline = Pipeline(
        config(tmp_path, movenet_enabled=False),
        channel=channel,
        messages=sink,  # type: ignore[arg-type]
        detector_factory=lambda model: FakeDetector([PERSON]),
        movenet_factory=factory,
        camera_factory=lambda settings: camera,
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            feed(camera, 4)
            wait_until(lambda: sink.topics("showcase.state"))
            time.sleep(0.1)
            assert pipeline.set_model("movenet") is False
    finally:
        channel.close()
        channel.unlink()

    assert calls == []
    assert sink.topics("pose.state") == []
    assert pipeline.status()["active_model"] == "yolox"


def test_a_movenet_frame_failure_does_not_stop_the_loop(tmp_path: Path) -> None:
    class FlakyPose(FakePoseDetector):
        def submit(self, rgb: np.ndarray) -> int:
            if not self.calls:
                self.calls.append("submit")
                raise RuntimeError("MoveNet DPU job failed")
            return super().submit(rgb)

    pose = FlakyPose()
    pipeline, camera, sink, channel = build(
        tmp_path,
        detector_factory=lambda model: FakeDetector([PERSON]),
        pose=pose,
    )
    try:
        with RunningPipeline(pipeline):
            wait_until(lambda: camera.on_frame is not None)
            assert pipeline.set_model("movenet") is True
            wait_until(lambda: pipeline.status()["active_model"] == "movenet")
            camera.push()
            wait_until(
                lambda: sink.topics("status") and "dpu_error" in sink.topics("status")[-1]["errors"]
            )
            feed(camera, 4)
            wait_until(lambda: sink.topics("pose.state"))
            wait_until(lambda: "dpu_error" not in pipeline.status()["errors"])
    finally:
        channel.close()
        channel.unlink()

    # The loop keeps submitting after the failed frame, and the error of a
    # failed frame is cleared by the next good one.
    assert pose.calls.count("submit") >= 2
    assert "dpu_error" not in pipeline.status()["errors"]
