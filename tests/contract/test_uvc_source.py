"""UvcVideoSource contract: timestamps, lazy YUYV decode and error handling (FR-001)."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import numpy as np
import pytest

from cpsa.adapters.camera.uvc import UvcVideoSource, _Captured, decode_yuyv
from cpsa.vision.types import VideoFrame


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
        self.calls: list[tuple[str, int, int, float, int]] = []

    def __call__(
        self, device: str, width: int, height: int, fps: float, exposure: int
    ) -> FakeStream:
        self.calls.append((device, width, height, fps, exposure))
        return self.stream


def wait_until(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.002)


def make_source(stream: FakeStream, **overrides: object) -> tuple[UvcVideoSource, Factory]:
    factory = Factory(stream)
    settings: dict[str, object] = {
        "width": 8,
        "height": 4,
        "fps": 30.0,
        "exposure": 300,
    }
    settings.update(overrides)
    source = UvcVideoSource("/dev/video-test", stream_factory=factory, **settings)  # type: ignore[arg-type]
    return source, factory


def test_delivers_raw_frames_with_v4l2_timestamps_and_applies_controls() -> None:
    data = yuyv(8, 4, 128)
    stream = FakeStream(
        [
            _Captured(ts=5.0, data=data, monotonic=True),
            _Captured(ts=5.033, data=data, monotonic=True),
        ]
    )
    source, factory = make_source(stream)
    frames: list[VideoFrame] = []
    source.start(frames.append)
    try:
        wait_until(lambda: len(frames) == 2)
    finally:
        source.stop()

    assert factory.calls == [("/dev/video-test", 8, 4, 30.0, 300)]
    assert [frame.seq for frame in frames] == [0, 1]
    assert [frame.t for frame in frames] == [5.0, 5.033]
    assert stream.controls == {"auto_exposure": 1, "exposure_time_absolute": 300}
    rgb = frames[0].decode()
    assert rgb.shape == (4, 8, 3) and rgb.dtype == np.uint8
    assert np.all(np.abs(rgb.astype(int) - 128) <= 5)
    assert stream.closed and not source.running


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
    source, _ = make_source(stream)
    before = time.monotonic()
    frames: list[VideoFrame] = []
    source.start(frames.append)
    try:
        wait_until(lambda: len(frames) == 2)
    finally:
        source.stop()
    after = time.monotonic()

    assert all(before - 1.0 <= frame.t <= after + 1.0 for frame in frames)
    assert frames[0].t != 1.0
    assert caplog.text.count("not CLOCK_MONOTONIC") == 1


def test_stream_error_is_reported_once_and_stops_delivery() -> None:
    data = yuyv(8, 4)
    stream = FakeStream([_Captured(ts=1.0, data=data, monotonic=True)], error=OSError("lost"))
    source, _ = make_source(stream)
    frames: list[VideoFrame] = []
    errors: list[Exception] = []
    source.start(frames.append, errors.append)
    try:
        wait_until(lambda: errors)
    finally:
        source.stop()

    assert len(frames) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], OSError)
    assert not source.running


def test_a_failing_frame_callback_does_not_stop_delivery() -> None:
    data = yuyv(8, 4)
    stream = FakeStream(
        [
            _Captured(ts=1.0, data=data, monotonic=True),
            _Captured(ts=1.033, data=data, monotonic=True),
        ]
    )
    source, _ = make_source(stream)
    frames: list[VideoFrame] = []
    calls = 0

    def on_frame(frame: VideoFrame) -> None:
        nonlocal calls
        calls += 1
        frames.append(frame)
        if calls == 1:
            raise RuntimeError("consumer failed")

    source.start(on_frame)
    try:
        wait_until(lambda: calls == 2)
    finally:
        source.stop()

    assert [frame.seq for frame in frames] == [0, 1]


def test_start_is_exclusive_and_stop_is_idempotent() -> None:
    stream = FakeStream([])
    source, _ = make_source(stream)
    source.start(lambda frame: None)
    try:
        with pytest.raises(RuntimeError):
            source.start(lambda frame: None)
    finally:
        source.stop()
    source.stop()
    assert stream.closed


def test_control_failure_closes_the_stream_and_raises() -> None:
    class BrokenStream(FakeStream):
        def set_control(self, name: str, value: int) -> None:
            raise RuntimeError("no such control")

    stream = BrokenStream([])
    source, _ = make_source(stream)

    with pytest.raises(RuntimeError):
        source.start(lambda frame: None)
    assert stream.closed
    assert not source.running


@pytest.mark.parametrize(
    ("width", "height", "fps", "exposure"),
    [(0, 4, 30.0, 300), (7, 4, 30.0, 300), (8, 0, 30.0, 300), (8, 4, 0.0, 300), (8, 4, 30.0, 0)],
)
def test_invalid_settings_are_rejected(width: int, height: int, fps: float, exposure: int) -> None:
    with pytest.raises(ValueError):
        UvcVideoSource(
            "/dev/video-test",
            width=width,
            height=height,
            fps=fps,
            exposure=exposure,
            stream_factory=Factory(FakeStream([])),
        )


def test_decode_yuyv_rejects_a_truncated_buffer() -> None:
    with pytest.raises(ValueError):
        decode_yuyv(b"\x80\x80", 8, 4)
