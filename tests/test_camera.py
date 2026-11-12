"""The UVC camera: node selection, controls, frames and failure reporting (NFR-004)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

import numpy as np
import pytest

from cpsa.camera import (
    DEPTH_FOURCC,
    RGB_FOURCC,
    UvcCamera,
    VideoNode,
    _Captured,
    resolve_device,
    select_video_node,
    yuyv_to_rgb,
)
from cpsa.config import CameraConfig

REAL_F200_CARD = "Intel(R) RealSense(TM) 3D Camer"


def camera_config(**overrides: object) -> CameraConfig:
    settings: dict[str, object] = {
        "device": "/dev/video-test",
        "match": "RealSense",
        "width": 8,
        "height": 4,
        "fps": 30.0,
        "exposure": 100,
        "gain": 200,
        "white_balance": 4000,
        "backlight_compensation": 0,
        "saturation": 110,
        "power_line_hz": 50,
    }
    settings.update(overrides)
    return CameraConfig(**settings)  # type: ignore[arg-type]


def yuyv(width: int, height: int, value: int = 128) -> bytes:
    return bytes([value]) * (width * height * 2)


class FakeStream:
    """A ``_Stream`` that hands out scripted captures, then idles."""

    def __init__(self, captures: Sequence[_Captured], error: Exception | None = None) -> None:
        self._captures = list(captures)
        self._error = error
        self.controls: dict[str, int] = {}
        self.closed = False
        self._lock = threading.Lock()

    def set_control(self, name: str, value: int) -> None:
        self.controls[name] = value

    def read(self) -> _Captured | None:
        with self._lock:
            if self._captures:
                return self._captures.pop(0)
            error = self._error
        if error is not None:
            raise error
        time.sleep(0.001)
        return None

    def close(self) -> None:
        self.closed = True


class Factory:
    def __init__(self, stream: FakeStream) -> None:
        self.stream = stream
        self.calls: list[tuple[str, int, int, float]] = []

    def __call__(self, device: str, width: int, height: int, fps: float) -> FakeStream:
        self.calls.append((device, width, height, fps))
        return self.stream


def wait_until(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.002)


def make_camera(stream: FakeStream, **overrides: object) -> tuple[UvcCamera, Factory]:
    factory = Factory(stream)
    camera = UvcCamera(camera_config(**overrides), stream_factory=factory)
    return camera, factory


def rgb_node(path: str = "/dev/video0", card: str = REAL_F200_CARD) -> VideoNode:
    return VideoNode(path, card, frozenset({RGB_FOURCC}))


def depth_node(path: str = "/dev/video1", card: str = REAL_F200_CARD) -> VideoNode:
    return VideoNode(path, card, frozenset({DEPTH_FOURCC}))


# --------------------------------------------------------------------------- #
# Node selection
# --------------------------------------------------------------------------- #


def test_the_rgb_node_is_chosen_by_card_and_formats() -> None:
    nodes = [
        VideoNode("/dev/video0", "Integrated Webcam", frozenset({RGB_FOURCC})),
        depth_node(),
        rgb_node("/dev/video2"),
    ]

    assert select_video_node(nodes, match="RealSense").path == "/dev/video2"


def test_a_node_offering_yuyv_and_z16_is_not_the_rgb_node() -> None:
    both = VideoNode("/dev/video0", "F200", frozenset({RGB_FOURCC, DEPTH_FOURCC}))

    with pytest.raises(RuntimeError):
        select_video_node([both], match="F200")


def test_match_is_case_insensitive_and_skips_other_cards() -> None:
    nodes = [rgb_node("/dev/video0", "creative vf0800"), rgb_node("/dev/video1", "F200 RGB")]

    assert select_video_node(nodes, match="f200").path == "/dev/video1"


def test_the_default_match_finds_the_truncated_kernel_card_name() -> None:
    nodes = [rgb_node("/dev/video4", REAL_F200_CARD)]

    assert select_video_node(nodes, match="RealSense").path == "/dev/video4"


def test_no_candidate_lists_what_was_found() -> None:
    nodes = [VideoNode("/dev/video0", "Integrated Webcam", frozenset({"MJPG"}))]

    with pytest.raises(RuntimeError, match="no camera node matching"):
        select_video_node(nodes, match="RealSense")


def test_an_explicit_device_is_returned_unchanged_without_listing() -> None:
    def lister() -> list[VideoNode]:
        raise AssertionError("an explicit device must not probe V4L2")

    assert resolve_device("/dev/video7", lister=lister) == "/dev/video7"


def test_auto_device_is_resolved_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="cpsa.camera"):
        resolved = resolve_device(
            "auto", match="F200", lister=lambda: [depth_node(card="F200"), rgb_node(card="F200")]
        )

    assert resolved == "/dev/video0"
    assert "Camera device 'auto' selected /dev/video0" in caplog.text


def test_auto_without_a_camera_raises() -> None:
    with pytest.raises(RuntimeError, match="no V4L2 nodes"):
        resolve_device("auto", lister=list)


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #


def test_frames_carry_v4l2_timestamps_and_the_configured_controls() -> None:
    data = yuyv(8, 4, 128)
    stream = FakeStream(
        [
            _Captured(ts=5.0, data=data, monotonic=True),
            _Captured(ts=5.033, data=data, monotonic=True),
        ]
    )
    camera, factory = make_camera(stream)
    frames = []
    camera.start(frames.append)
    try:
        wait_until(lambda: len(frames) == 2)
    finally:
        camera.stop()

    assert factory.calls == [("/dev/video-test", 8, 4, 30.0)]
    assert [frame.seq for frame in frames] == [0, 1]
    assert [frame.t for frame in frames] == [5.0, 5.033]
    assert stream.controls == {
        "auto_exposure": 1,
        "exposure_time_absolute": 100,
        "gain": 200,
        "power_line_frequency": 1,
        "white_balance_automatic": 0,
        "white_balance_temperature": 4000,
        "backlight_compensation": 0,
        "saturation": 110,
    }
    rgb = frames[0].decode()
    assert rgb.shape == (4, 8, 3) and rgb.dtype == np.uint8
    assert np.all(np.abs(rgb.astype(int) - 128) <= 5)
    assert stream.closed and not camera.running


def test_auto_white_balance_leaves_the_temperature_control_alone() -> None:
    stream = FakeStream([])
    camera, _ = make_camera(stream, white_balance="auto")
    camera.start(lambda frame: None)
    try:
        assert stream.controls["white_balance_automatic"] == 1
        assert stream.controls["backlight_compensation"] == 0
        assert "white_balance_temperature" not in stream.controls
    finally:
        camera.stop()


def test_the_image_controls_are_applied_again_after_a_restart() -> None:
    stream = FakeStream([])
    camera, _ = make_camera(stream)
    camera.start(lambda frame: None)
    camera.stop()
    stream.controls.clear()

    camera.start(lambda frame: None)
    try:
        assert stream.controls == {
            "auto_exposure": 1,
            "exposure_time_absolute": 100,
            "gain": 200,
            "power_line_frequency": 1,
            "white_balance_automatic": 0,
            "white_balance_temperature": 4000,
            "backlight_compensation": 0,
            "saturation": 110,
        }
    finally:
        camera.stop()


def test_non_monotonic_buffer_timestamps_fall_back_and_warn_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = yuyv(8, 4)
    stream = FakeStream(
        [
            _Captured(ts=1.0, data=data, monotonic=False),
            _Captured(ts=2.0, data=data, monotonic=False),
        ]
    )
    camera, _ = make_camera(stream)
    before = time.monotonic()
    frames = []
    camera.start(frames.append)
    try:
        wait_until(lambda: len(frames) == 2)
    finally:
        camera.stop()
    after = time.monotonic()

    assert all(before - 1.0 <= frame.t <= after + 1.0 for frame in frames)
    assert frames[0].t != 1.0
    assert caplog.text.count("not CLOCK_MONOTONIC") == 1


def test_a_stream_error_is_reported_once_and_stops_delivery() -> None:
    stream = FakeStream([_Captured(ts=1.0, data=yuyv(8, 4), monotonic=True)], error=OSError("lost"))
    camera, _ = make_camera(stream)
    frames: list[object] = []
    errors: list[Exception] = []
    camera.start(frames.append, errors.append)
    try:
        wait_until(lambda: errors)
    finally:
        camera.stop()

    assert len(frames) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], OSError)
    assert not camera.running


def test_a_failing_frame_callback_does_not_stop_delivery() -> None:
    data = yuyv(8, 4)
    stream = FakeStream(
        [
            _Captured(ts=1.0, data=data, monotonic=True),
            _Captured(ts=1.033, data=data, monotonic=True),
        ]
    )
    camera, _ = make_camera(stream)
    frames = []
    calls = 0

    def on_frame(frame: object) -> None:
        nonlocal calls
        calls += 1
        frames.append(frame)
        if calls == 1:
            raise RuntimeError("consumer failed")

    camera.start(on_frame)
    try:
        wait_until(lambda: calls == 2)
    finally:
        camera.stop()

    assert [frame.seq for frame in frames] == [0, 1]


def test_start_is_exclusive_and_stop_is_idempotent() -> None:
    stream = FakeStream([])
    camera, _ = make_camera(stream)
    camera.start(lambda frame: None)
    try:
        with pytest.raises(RuntimeError):
            camera.start(lambda frame: None)
    finally:
        camera.stop()
    camera.stop()

    assert stream.closed


def test_a_control_failure_closes_the_stream_and_raises() -> None:
    class BrokenStream(FakeStream):
        def set_control(self, name: str, value: int) -> None:
            raise RuntimeError("no such control")

    stream = BrokenStream([])
    camera, _ = make_camera(stream)

    with pytest.raises(RuntimeError):
        camera.start(lambda frame: None)

    assert stream.closed
    assert not camera.running


def test_the_camera_can_be_started_again_after_it_stopped() -> None:
    stream = FakeStream([_Captured(ts=1.0, data=yuyv(8, 4), monotonic=True)])
    camera, factory = make_camera(stream)
    frames = []
    camera.start(frames.append)
    wait_until(lambda: len(frames) == 1)
    camera.stop()

    camera.start(frames.append)
    try:
        wait_until(lambda: len(factory.calls) == 2)
    finally:
        camera.stop()

    assert factory.calls[0] == factory.calls[1]


def test_auto_resolves_the_node_again_on_a_restart() -> None:
    listing = [rgb_node("/dev/video0")]
    factory = Factory(FakeStream([]))
    camera = UvcCamera(
        camera_config(device="auto"), device_lister=lambda: list(listing), stream_factory=factory
    )
    camera.start(lambda frame: None)
    camera.stop()
    listing[0] = rgb_node("/dev/video3")
    camera.start(lambda frame: None)
    camera.stop()

    assert [call[0] for call in factory.calls] == ["/dev/video0", "/dev/video3"]


# --------------------------------------------------------------------------- #
# YUYV conversion
# --------------------------------------------------------------------------- #


def test_yuyv_to_rgb_decodes_one_frame() -> None:
    width, height = 64, 48
    raw = np.random.default_rng(5).integers(0, 256, size=width * height * 2, dtype=np.uint8)

    rgb = yuyv_to_rgb(raw.tobytes(), width, height)

    assert rgb.shape == (height, width, 3)
    assert rgb.dtype == np.uint8


def test_yuyv_to_rgb_rejects_a_truncated_buffer() -> None:
    with pytest.raises(ValueError):
        yuyv_to_rgb(b"\x80\x80", 8, 4)
