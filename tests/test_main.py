"""Starting the showcase: the two processes, the exit codes and the signals."""

from __future__ import annotations

import asyncio
import json
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from aiohttp import ClientSession

from cpsa import __main__ as entry
from cpsa.camera import VideoFrame
from cpsa.channel import MessageChannel
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
from cpsa.yolox import Detection

WIDTH, HEIGHT = 16, 8
PERSON = Detection(box=(2.0, 1.0, 8.0, 6.0), score=0.75, class_id=0)

SMOOTHING = SmoothingConfig(
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


class FakeQueue:
    """The ``multiprocessing`` queue surface, in one process."""

    def __init__(self, maxsize: int = 64) -> None:
        self._items: deque[tuple[str, dict[str, Any]]] = deque()
        self.maxsize = maxsize

    def put_nowait(self, item: tuple[str, dict[str, Any]]) -> None:
        if len(self._items) >= self.maxsize:
            raise queue.Full
        self._items.append(item)

    def get_nowait(self) -> tuple[str, dict[str, Any]]:
        if not self._items:
            raise queue.Empty
        return self._items.popleft()

    def close(self) -> None:
        pass


class FakeCamera:
    def __init__(self) -> None:
        self.running = False
        self.starts = 0
        self.on_frame: Callable[[VideoFrame], None] | None = None

    def start(self, on_frame: Callable[[VideoFrame], None], on_error: Any = None) -> None:
        self.on_frame = on_frame
        self.running = True
        self.starts += 1

    def stop(self) -> None:
        self.running = False

    def push(self, value: int = 60) -> None:
        assert self.running and self.on_frame is not None
        rgb = np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)
        self.on_frame(
            VideoFrame(seq=0, t=time.monotonic(), width=WIDTH, height=HEIGHT, decode=lambda: rgb)
        )


class FakeDetector:
    def __init__(self) -> None:
        self.stage_ms = {"preprocess": 1.0, "dpu": 2.0, "decode": 0.5}

    def submit(self, rgb: np.ndarray) -> int:
        return 1

    def wait(self, token: int) -> list[Detection]:
        return [PERSON]

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        return [PERSON]


class FakePoseDetector:
    """The MoveNet runner of process A, without the VART bindings."""

    def __init__(self) -> None:
        self.stage_ms = {"preprocess": 0.4, "dpu": 1.5, "decode": 0.3}
        self.keypoints = np.tile(np.array([4.0, 2.0, 0.9], dtype=np.float32), (17, 1))

    def submit(self, rgb: np.ndarray) -> int:
        return 1

    def wait(self, token: int) -> MovenetPose:
        return MovenetPose(keypoints=self.keypoints, score=0.9)


def free_port() -> int:
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        return int(socket_.getsockname()[1])


def showcase_config(
    tmp_path: Path, *, static_dir: Path | None = None, port: int | None = None
) -> ShowcaseConfig:
    model = tmp_path / "yolox_nano.xmodel"
    if not model.exists():
        model.write_bytes(b"fake xmodel")
    movenet = tmp_path / "movenet.xmodel"
    if not movenet.exists():
        movenet.write_bytes(b"fake movenet xmodel")
    static = static_dir if static_dir is not None else tmp_path / "dist"
    static.mkdir(parents=True, exist_ok=True)
    (static / "index.html").write_text("<html>cpsa</html>", encoding="utf-8")
    return ShowcaseConfig(
        camera=CameraConfig(
            device="auto",
            match="RealSense",
            width=WIDTH,
            height=HEIGHT,
            fps=60.0,
            exposure=100,
            gain=200,
            white_balance=4000,
            backlight_compensation=0,
            saturation=110,
            power_line_hz=50,
        ),
        tracker=TrackerConfig(
            enabled=True,
            iou_match=0.3,
            alpha=0.35,
            fast_alpha=0.8,
            deadband_iou=0.9,
            min_hits=2,
            max_misses=6,
            still=StillConfig(enabled=True, motion_thr=1.0, force_every_s=0.5),
        ),
        model=ModelConfig(path=model, conf_thr=0.35, iou_thr=0.45),
        movenet=MovenetConfig(path=movenet, enabled=True, publish_hz=30.0, smoothing=SMOOTHING),
        labels="it",
        web=WebConfig(
            host="127.0.0.1",
            port=port if port is not None else free_port(),
            video_fps=60.0,
            jpeg_quality=70,
            static_dir=static,
        ),
        log=LogConfig(level="INFO"),
    )


def wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.01)


def topics(channel: MessageChannel, topic: str) -> list[dict[str, Any]]:
    """Drain the queue through a fresh handle: ``run`` closes the one it was given."""
    reader = MessageChannel(channel.queue)
    return [data for name, data in reader.drain() if name == topic]


def http_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local
        return dict(json.loads(response.read()))


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


def test_an_unreadable_configuration_exits_with_two(tmp_path: Path) -> None:
    assert entry.main(["--config", str(tmp_path / "nope.yaml")]) == entry.EXIT_START


def test_a_bad_configuration_exits_with_two(tmp_path: Path) -> None:
    broken = tmp_path / "showcase.yaml"
    broken.write_text("camera: [unclosed", encoding="utf-8")

    assert entry.main(["--config", str(broken)]) == entry.EXIT_START


def test_the_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--version"])

    assert excinfo.value.code == 0
    assert "cpsa-showcase" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# run(): exit codes and the published error
# --------------------------------------------------------------------------- #


def test_an_orderly_stop_returns_zero(tmp_path: Path) -> None:
    config = showcase_config(tmp_path)
    messages = MessageChannel(FakeQueue())
    camera = FakeCamera()
    stop = threading.Event()
    result: dict[str, int] = {}

    def target() -> None:
        result["code"] = entry.run(
            config,
            stop=stop,
            messages=messages,
            detector_factory=lambda model: FakeDetector(),
            movenet_factory=lambda settings: FakePoseDetector(),
            camera_factory=lambda settings: camera,
            start_server=False,
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        wait_until(lambda: camera.on_frame is not None)
        camera.push()
        wait_until(
            lambda: any(status["state"] == "running" for status in topics(messages, "status"))
        )
    finally:
        stop.set()
        thread.join(10.0)

    assert not thread.is_alive()
    assert result["code"] == entry.EXIT_OK
    assert not camera.running


def test_a_missing_model_exits_with_two_and_publishes_the_error(tmp_path: Path) -> None:
    config = showcase_config(tmp_path)
    config.model.path.unlink()
    messages = MessageChannel(FakeQueue())
    camera = FakeCamera()

    code = entry.run(
        config,
        messages=messages,
        detector_factory=lambda model: FakeDetector(),
        movenet_factory=lambda settings: FakePoseDetector(),
        camera_factory=lambda settings: camera,
        start_server=False,
    )

    published = topics(messages, "status")
    assert code == entry.EXIT_START
    assert camera.starts == 0
    assert published[-1]["state"] == "error"
    assert published[-1]["errors"] == ["model_missing"]


def test_a_missing_movenet_model_exits_with_two(tmp_path: Path) -> None:
    config = showcase_config(tmp_path)
    config.movenet.path.unlink()
    messages = MessageChannel(FakeQueue())
    camera = FakeCamera()

    code = entry.run(
        config,
        messages=messages,
        detector_factory=lambda model: FakeDetector(),
        movenet_factory=lambda settings: FakePoseDetector(),
        camera_factory=lambda settings: camera,
        start_server=False,
    )

    published = topics(messages, "status")
    assert code == entry.EXIT_START
    assert camera.starts == 0
    assert published[-1]["state"] == "error"
    assert published[-1]["errors"] == ["model_missing"]


def test_a_detector_that_cannot_load_exits_with_two(tmp_path: Path) -> None:
    config = showcase_config(tmp_path)
    messages = MessageChannel(FakeQueue())

    def factory(model: object) -> FakeDetector:
        raise RuntimeError("no VART on this machine")

    code = entry.run(
        config,
        messages=messages,
        detector_factory=factory,
        movenet_factory=lambda settings: FakePoseDetector(),
        camera_factory=lambda settings: FakeCamera(),
        start_server=False,
    )

    assert code == entry.EXIT_START
    assert topics(messages, "status")[-1]["errors"] == ["dpu_error"]


# --------------------------------------------------------------------------- #
# The two processes
# --------------------------------------------------------------------------- #


def test_a_model_set_command_switches_the_view_of_process_a(tmp_path: Path) -> None:
    """The whole chain: client -> server B -> command channel -> pipeline A."""
    config = showcase_config(tmp_path)
    camera = FakeCamera()
    stop = threading.Event()
    result: dict[str, int] = {}

    def target() -> None:
        result["code"] = entry.run(
            config,
            stop=stop,
            detector_factory=lambda model: FakeDetector(),
            movenet_factory=lambda settings: FakePoseDetector(),
            camera_factory=lambda settings: camera,
            start_server=True,
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{config.web.port}/ws"
    try:
        wait_until(lambda: camera.on_frame is not None)

        def switch_view() -> str:
            async def scenario() -> str:
                async with ClientSession() as session:
                    ws = await session.ws_connect(url)
                    try:
                        await ws.send_json(
                            {"v": 1, "type": "model.set", "data": {"model": "movenet"}}
                        )
                        for _ in range(200):
                            message = await ws.receive_json(timeout=1.0)
                            if (
                                message["type"] == "status"
                                and message["data"]["active_model"] == "movenet"
                            ):
                                return str(message["data"]["active_model"])
                        raise AssertionError("the view never switched")
                    finally:
                        await ws.close()

            return asyncio.run(scenario())

        active = switch_view()
    finally:
        stop.set()
        thread.join(15.0)

    assert not thread.is_alive()
    assert result["code"] == entry.EXIT_OK
    assert active == "movenet"


def test_the_two_processes_serve_the_dashboard_together(tmp_path: Path) -> None:
    """A (camera, DPU, overlay) feeds B (HTTP) through the shared channel."""
    config = showcase_config(tmp_path)
    camera = FakeCamera()
    stop = threading.Event()
    result: dict[str, int] = {}

    def target() -> None:
        result["code"] = entry.run(
            config,
            stop=stop,
            detector_factory=lambda model: FakeDetector(),
            movenet_factory=lambda settings: FakePoseDetector(),
            camera_factory=lambda settings: camera,
            start_server=True,
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{config.web.port}"
    try:
        wait_until(lambda: camera.on_frame is not None)

        def running() -> bool:
            try:
                return http_json(f"{base}/health")["state"] == "running"
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                return False

        # Frames travel A -> shared memory -> B, and the status A publishes
        # through the queue is what /health reads in the other process.
        camera.push()
        camera.push()
        wait_until(running)
        health = http_json(f"{base}/health")
        assert health["state"] == "running"
        assert health["fps"] is not None and health["fps"] >= 0.0
        with urllib.request.urlopen(f"{base}/", timeout=2.0) as response:  # noqa: S310 - local
            assert response.status == 200
            assert b"cpsa" in response.read()
    finally:
        stop.set()
        thread.join(15.0)

    assert not thread.is_alive()
    assert result["code"] == entry.EXIT_OK


def test_the_shared_memory_of_a_run_is_unlinked(tmp_path: Path) -> None:
    """``run`` owns the channel: no shared-memory segment may leak."""
    config = showcase_config(tmp_path)
    camera = FakeCamera()
    stop = threading.Event()
    result: dict[str, int] = {}

    def target() -> None:
        result["code"] = entry.run(
            config,
            stop=stop,
            detector_factory=lambda model: FakeDetector(),
            movenet_factory=lambda settings: FakePoseDetector(),
            camera_factory=lambda settings: camera,
            start_server=False,
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        wait_until(lambda: camera.on_frame is not None)
    finally:
        stop.set()
        thread.join(10.0)

    assert result["code"] == entry.EXIT_OK
    leaked = list(Path("/dev/shm").glob("cpsa-*"))
    assert leaked == []
