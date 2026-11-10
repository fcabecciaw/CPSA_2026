"""F200 adapter contract: pairing, XU set/verify, start errors, lifecycle.

The V4L2 layer is replaced by fakes shaped like linuxpy 0.25: a non-blocking
``Device``, a ``VideoCapture.buffer`` that only offers ``raw_read``, and a
``Frame`` whose timestamp and flags come from its ``v4l2_buffer``. The stub's
``read``/``wait_read`` raise if the adapter ever falls back to them, which is
the linuxpy path that fails with ``'module' object is not callable``.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import struct
import threading
import time
from typing import Any

import numpy as np
import pytest

from cpsa import ports
from cpsa.adapters.camera import f200
from cpsa.vision.pairing import PairingStats
from tests import fakes
from tests.contract.vendor_stubs import wait_until

WIDTH, HEIGHT = 640, 480
RGB_PATH = "/dev/video0"
DEPTH_PATH = "/dev/video2"
MONOTONIC_FLAG = 0x00002000


def yuyv_frame(luma: int = 128) -> bytes:
    pairs = np.full((HEIGHT, WIDTH // 2, 4), 128, dtype=np.uint8)
    pairs[:, :, 0] = luma
    pairs[:, :, 2] = luma
    return pairs.tobytes()


def z16_frame() -> tuple[bytes, np.ndarray[Any, np.dtype[np.uint16]]]:
    values = np.arange(WIDTH * HEIGHT, dtype=np.uint16).reshape(HEIGHT, WIDTH)
    return values.tobytes(), values


class FakeControl:
    value: int | None = None


class FakeTimeval:
    def __init__(self, secs: int, usecs: int) -> None:
        self.secs = secs
        self.usecs = usecs


class FakeV4l2Buffer:
    def __init__(self, flags: int, secs: int, usecs: int) -> None:
        self.flags = flags
        self.timestamp = FakeTimeval(secs, usecs)


class FakeFrame:
    """Same public attributes as ``linuxpy.video.device.Frame``."""

    def __init__(self, flags: int, *, data: bytes = b"frame", ts: float = 12.5) -> None:
        secs = int(ts)
        usecs = round((ts - secs) * 1_000_000)
        self.buff = FakeV4l2Buffer(flags, secs, usecs)
        self.data = data

    @property
    def flags(self) -> int:
        return self.buff.flags

    @property
    def timestamp(self) -> float:
        return self.buff.timestamp.secs + self.buff.timestamp.usecs * 1e-6


class FakeIO:
    """linuxpy's ``IO.select`` is the module itself, not ``select.select``."""

    select = object()


class FakeDevice:
    def __init__(self) -> None:
        self.controls = {
            "auto_exposure": FakeControl(),
            "exposure_time_absolute": FakeControl(),
        }
        self.io = FakeIO()
        self.blocking = False
        self.closed = False

    @property
    def is_blocking(self) -> bool:
        return self.blocking

    def fileno(self) -> int:
        return 7

    def close(self) -> None:
        self.closed = True


class FakeBuffer:
    def __init__(
        self,
        frames: list[FakeFrame] | None = None,
        error_after: int | None = None,
        *,
        empty_delay: float = 0.002,
    ) -> None:
        self.frames = list(frames or [])
        self.error_after = error_after
        self.empty_delay = empty_delay
        self.calls: list[str] = []
        self._started_at = time.monotonic()
        self._first_ts = self.frames[0].timestamp if self.frames else None

    def raw_read(self) -> FakeFrame:
        self.calls.append("raw_read")
        if self.error_after is not None and len(self.calls) > self.error_after:
            raise OSError("simulated device error")
        if not self.frames:
            if self.empty_delay:
                time.sleep(self.empty_delay)
            raise BlockingIOError(errno.EAGAIN, "no frame yet")
        frame = self.frames[0]
        if self._first_ts is not None:
            due = (frame.timestamp - self._first_ts) - (time.monotonic() - self._started_at)
            if due > 0:
                time.sleep(due)
        return self.frames.pop(0)

    def read(self) -> FakeFrame:
        raise AssertionError("adapter must call raw_read, not linuxpy wait_read")

    def wait_read(self) -> FakeFrame:
        raise AssertionError("adapter must not use linuxpy wait_read")


class FakeCapture:
    def __init__(self, buffer: FakeBuffer) -> None:
        self.buffer = buffer
        self.closed = False

    def close(self) -> None:
        self.closed = True


def captured_frame(ts: float, data: bytes, monotonic: bool = True) -> FakeFrame:
    return FakeFrame(MONOTONIC_FLAG if monotonic else 0, data=data, ts=ts)


def fake_stream(
    path: str,
    frames: list[FakeFrame] | None = None,
    *,
    error_after: int | None = None,
    empty_delay: float = 0.002,
) -> tuple[f200._LinuxpyStream, FakeDevice, FakeCapture]:
    device = FakeDevice()
    capture = FakeCapture(FakeBuffer(frames, error_after=error_after, empty_delay=empty_delay))
    stream = f200._LinuxpyStream(device, capture, path, MONOTONIC_FLAG)
    return stream, device, capture


@pytest.fixture(autouse=True)
def readable_select(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the adapter's own select report data; linuxpy's wait must be unused."""
    monkeypatch.setattr(f200.select, "select", lambda *args: ([7], [], []))


class VideoFactory:
    def __init__(
        self,
        rgb_frames: list[FakeFrame],
        depth_frames: list[FakeFrame],
        *,
        rgb_error_after: int | None = None,
        depth_error_after: int | None = None,
    ) -> None:
        self.frames = {RGB_PATH: rgb_frames, DEPTH_PATH: depth_frames}
        self.errors = {RGB_PATH: rgb_error_after, DEPTH_PATH: depth_error_after}
        self.calls: dict[str, dict[str, Any]] = {}
        self.streams: dict[str, f200._LinuxpyStream] = {}
        self.devices: dict[str, FakeDevice] = {}
        self.captures: dict[str, FakeCapture] = {}

    def __call__(self, path: str, **kwargs: Any) -> f200._LinuxpyStream:
        self.calls[path] = kwargs
        stream, device, capture = fake_stream(
            path, self.frames[path], error_after=self.errors[path]
        )
        for name, value in (kwargs.get("controls") or {}).items():
            stream.set_control(name, value)
        self.streams[path] = stream
        self.devices[path] = device
        self.captures[path] = capture
        return stream


def fake_ioctl(state: dict[int, int], wrong: dict[int, int] | None = None) -> Any:
    def ioctl(fd: int, request: int, argument: bytearray) -> None:
        assert fd >= 0
        assert request == f200.UVCIOC_CTRL_QUERY
        unit, selector, query, size, address = struct.unpack("BBBxHxxQ", bytes(argument))
        assert unit == f200.XU_UNIT
        if query == f200.XU_GET_LEN:
            ctypes.memmove(address, struct.pack("<H", 1), 2)
        elif query == f200.XU_SET_CUR:
            state[selector] = int.from_bytes(ctypes.string_at(address, size), "little")
        elif query == f200.XU_GET_CUR:
            value = (wrong or {}).get(selector, state[selector])
            ctypes.memmove(address, value.to_bytes(size, "little"), size)
        else:
            raise AssertionError(f"unexpected XU query {query:#x}")

    return ioctl


@pytest.fixture
def xu(monkeypatch: pytest.MonkeyPatch) -> dict[int, int]:
    state = {1: 0, 2: 0, 5: 0, 6: 0}

    def open_xu(path: str) -> f200._FdXu:
        return f200._FdXu(os.open(os.devnull, os.O_RDWR), path)

    monkeypatch.setattr(f200, "_open_xu", open_xu)
    monkeypatch.setattr(f200.fcntl, "ioctl", fake_ioctl(state))
    return state


def make_source() -> f200.F200Source:
    return f200.F200Source(
        RGB_PATH,
        DEPTH_PATH,
        width=WIDTH,
        height=HEIGHT,
        fps=30,
        rgb_exposure=300,
        xu=f200.XuSettings(laser_power=16, accuracy=2, filter=6, confidence=1),
    )


def test_start_pairs_frames_and_applies_settings(
    monkeypatch: pytest.MonkeyPatch, xu: dict[int, int]
) -> None:
    depth_payload, depth_values = z16_frame()
    factory = VideoFactory(
        [
            captured_frame(100.000, yuyv_frame()),
            captured_frame(100.033, yuyv_frame()),
        ],
        [
            captured_frame(100.002, depth_payload),
            captured_frame(100.035, depth_payload),
        ],
    )
    monkeypatch.setattr(f200, "_open_video", factory)
    received: list[ports.RgbdFrame] = []
    source = make_source()
    assert isinstance(source, ports.FrameSource)

    source.start(received.append)
    try:
        assert wait_until(lambda: len(received) == 2)
    finally:
        source.stop()

    assert [frame.seq for frame in received] == [0, 1]
    assert [frame.ts_rgb for frame in received] == pytest.approx([100.000, 100.033])
    assert [frame.ts_depth for frame in received] == pytest.approx([100.002, 100.035])
    for frame in received:
        assert frame.rgb.shape == (HEIGHT, WIDTH, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.depth.shape == (HEIGHT, WIDTH)
        assert frame.depth.dtype == np.uint16
        np.testing.assert_array_equal(frame.depth, depth_values)
        assert np.all(frame.rgb == 130)

    assert xu == {1: 16, 2: 2, 5: 6, 6: 1}
    assert factory.calls[RGB_PATH]["fourcc"] == f200.RGB_FOURCC
    assert factory.calls[DEPTH_PATH]["fourcc"] == f200.DEPTH_FOURCC
    assert factory.calls[RGB_PATH]["controls"] == {
        "auto_exposure": f200.RGB_AUTO_EXPOSURE_MANUAL,
        "exposure_time_absolute": 300,
    }
    assert "controls" not in factory.calls[DEPTH_PATH]
    assert factory.devices[RGB_PATH].controls["auto_exposure"].value == 1
    assert factory.devices[RGB_PATH].controls["exposure_time_absolute"].value == 300
    assert factory.devices[DEPTH_PATH].controls["auto_exposure"].value is None
    assert factory.captures[RGB_PATH].closed
    assert factory.captures[DEPTH_PATH].closed
    assert factory.devices[RGB_PATH].closed
    assert factory.devices[DEPTH_PATH].closed
    assert source.running is False
    assert source.stats == PairingStats(2, 0, 0)


def test_conversion_happens_only_in_the_delivery_thread(
    monkeypatch: pytest.MonkeyPatch, xu: dict[int, int]
) -> None:
    depth_payload, _ = z16_frame()
    factory = VideoFactory(
        [captured_frame(70.0, yuyv_frame())],
        [captured_frame(70.001, depth_payload)],
    )
    calls: list[tuple[str, str]] = []
    real_rgb = f200.yuyv_to_rgb
    real_depth = f200.z16_to_depth

    def tracked_rgb(data: bytes, width: int, height: int) -> np.ndarray[Any, Any]:
        calls.append(("rgb", threading.current_thread().name))
        return real_rgb(data, width, height)

    def tracked_depth(data: bytes, width: int, height: int) -> np.ndarray[Any, Any]:
        calls.append(("depth", threading.current_thread().name))
        return real_depth(data, width, height)

    monkeypatch.setattr(f200, "yuyv_to_rgb", tracked_rgb)
    monkeypatch.setattr(f200, "z16_to_depth", tracked_depth)
    monkeypatch.setattr(f200, "_open_video", factory)
    source = make_source()
    received: list[ports.RgbdFrame] = []

    source.start(received.append)
    try:
        assert wait_until(lambda: len(received) == 1)
    finally:
        source.stop()

    assert calls
    assert all(name == "cpsa-camera-delivery" for _, name in calls)
    assert {kind for kind, _ in calls} == {"rgb", "depth"}


def test_unpaired_raw_frames_are_not_converted(
    monkeypatch: pytest.MonkeyPatch, xu: dict[int, int]
) -> None:
    factory = VideoFactory(
        [captured_frame(0.0, yuyv_frame())],
        [captured_frame(1.0, z16_frame()[0])],
    )
    converted: list[str] = []
    monkeypatch.setattr(f200, "yuyv_to_rgb", lambda *args: converted.append("rgb") or np.empty(0))
    monkeypatch.setattr(
        f200, "z16_to_depth", lambda *args: converted.append("depth") or np.empty(0)
    )
    monkeypatch.setattr(f200, "_open_video", factory)
    source = make_source()

    source.start(lambda frame: None)
    try:
        assert wait_until(lambda: source.stats is not None and source.stats.dropped_rgb == 1)
    finally:
        source.stop()

    assert converted == []


def test_restart_reapplies_the_xu_values(
    monkeypatch: pytest.MonkeyPatch, xu: dict[int, int]
) -> None:
    factory = VideoFactory([], [])
    monkeypatch.setattr(f200, "_open_video", factory)
    source = make_source()

    source.start(lambda frame: None)
    source.stop()
    assert xu == {1: 16, 2: 2, 5: 6, 6: 1}

    for selector in xu:
        xu[selector] = 0
    source.start(lambda frame: None)
    source.stop()
    assert xu == {1: 16, 2: 2, 5: 6, 6: 1}


def test_yuyv_to_rgb_uses_bt601_colour_and_pixel_order() -> None:
    # Two red-ish pixels (Y=81, U=90, V=240) followed by two grey pixels.
    data = bytes([81, 90, 81, 240, 128, 128, 128, 128])
    rgb = f200.yuyv_to_rgb(data, 4, 1)

    assert rgb.shape == (1, 4, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] > 200 and rgb[0, 1, 0] > 200
    assert rgb[0, 0, 1] < 50 and rgb[0, 0, 2] < 50
    np.testing.assert_array_equal(rgb[0, 2], [130, 130, 130])


def test_z16_to_depth_rejects_short_frames() -> None:
    with pytest.raises(ValueError, match="Z16"):
        f200.z16_to_depth(b"\x00\x00", WIDTH, HEIGHT)


def test_read_xu_reads_back_the_four_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {1: 16, 2: 2, 5: 6, 6: 1}
    fd = os.open(os.devnull, os.O_RDWR)
    monkeypatch.setattr(f200, "_open_xu", lambda path: f200._FdXu(fd, path))
    monkeypatch.setattr(f200.fcntl, "ioctl", fake_ioctl(state))

    assert f200.read_xu(DEPTH_PATH) == {
        "laser_power": 16,
        "accuracy": 2,
        "filter": 6,
        "confidence": 1,
    }


def test_xu_read_back_mismatch_names_the_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {1: 16, 2: 2, 5: 6, 6: 1}
    fd = os.open(os.devnull, os.O_RDWR)
    monkeypatch.setattr(f200, "_open_xu", lambda path: f200._FdXu(fd, path))
    monkeypatch.setattr(f200.fcntl, "ioctl", fake_ioctl(state, wrong={6: 14}))
    opened: list[str] = []

    def open_video(path: str, **kwargs: Any) -> f200._LinuxpyStream:
        opened.append(path)
        return fake_stream(path)[0]

    monkeypatch.setattr(f200, "_open_video", open_video)
    source = make_source()

    with pytest.raises(RuntimeError) as info:
        source.start(lambda frame: None)
    message = str(info.value)
    assert "confidence" in message
    assert DEPTH_PATH in message
    assert opened == []
    assert source.running is False
    source.stop()  # idempotent after a failed start


def test_start_error_names_the_device_path_and_closes_the_other_stream(
    monkeypatch: pytest.MonkeyPatch, xu: dict[int, int]
) -> None:
    rgb_stream, rgb_device, rgb_capture = fake_stream(RGB_PATH)

    def open_video(path: str, **kwargs: Any) -> f200._LinuxpyStream:
        if path == DEPTH_PATH:
            raise OSError("device disappeared")
        return rgb_stream

    monkeypatch.setattr(f200, "_open_video", open_video)
    source = make_source()

    with pytest.raises(RuntimeError) as info:
        source.start(lambda frame: None)
    assert DEPTH_PATH in str(info.value)
    assert "device disappeared" in str(info.value)
    assert rgb_capture.closed
    assert rgb_device.closed
    assert source.running is False


def test_capture_error_is_logged_and_stops_capture(
    monkeypatch: pytest.MonkeyPatch,
    xu: dict[int, int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    depth_payload, _ = z16_frame()
    factory = VideoFactory(
        [captured_frame(50.0, yuyv_frame())],
        [captured_frame(50.001, depth_payload)],
        rgb_error_after=1,
        depth_error_after=1,
    )
    monkeypatch.setattr(f200, "_open_video", factory)
    source = make_source()

    with caplog.at_level(logging.ERROR, logger="cpsa.adapters.camera.f200"):
        source.start(lambda frame: None)
        try:
            assert wait_until(lambda: not source.running)
        finally:
            source.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert any("capture stopped by an error" in message for message in messages)
    assert any(RGB_PATH in message or DEPTH_PATH in message for message in messages)
    assert factory.captures[RGB_PATH].closed
    assert factory.captures[DEPTH_PATH].closed


def test_timestamp_fallback_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    source = make_source()
    with caplog.at_level(logging.WARNING, logger="cpsa.adapters.camera.f200"):
        assert source._timestamp(f200._Captured(5.0, b"", False)) != 5.0
        assert source._timestamp(f200._Captured(5.0, b"", True)) == 5.0
        assert source._timestamp(f200._Captured(5.0, b"", False)) != 5.0
    warnings = [record for record in caplog.records if "not CLOCK_MONOTONIC" in record.getMessage()]
    assert len(warnings) == 1


def test_linuxpy_stream_reads_raw_buffer_timestamp_and_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice()
    buffer = FakeBuffer([FakeFrame(MONOTONIC_FLAG, ts=12.5)], empty_delay=0)
    capture = FakeCapture(buffer)
    assert not callable(device.io.select)  # the linuxpy 0.25 wait_read trap

    stream = f200._LinuxpyStream(device, capture, DEPTH_PATH, MONOTONIC_FLAG)
    captured = stream.read()

    assert captured == f200._Captured(12.5, b"frame", True)
    assert buffer.calls == ["raw_read"]
    assert stream.read() is None  # select woke but DQBUF reported EAGAIN
    assert buffer.calls == ["raw_read", "raw_read"]

    stream.set_control("auto_exposure", 1)
    assert device.controls["auto_exposure"].value == 1
    with pytest.raises(RuntimeError, match="not available"):
        stream.set_control("missing", 1)

    stream.close()
    assert capture.closed and device.closed


def test_linuxpy_stream_uses_fallback_flag_and_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice()
    buffer = FakeBuffer([FakeFrame(0)], empty_delay=0)
    stream = f200._LinuxpyStream(device, FakeCapture(buffer), DEPTH_PATH, MONOTONIC_FLAG)
    monkeypatch.setattr(f200.select, "select", lambda *args: ([], [], []))
    assert stream.read() is None
    assert buffer.calls == []

    monkeypatch.setattr(f200.select, "select", lambda *args: ([7], [], []))
    captured = stream.read()
    assert captured is not None
    assert captured.monotonic is False
    assert buffer.calls == ["raw_read"]


def test_linuxpy_timestamp_attributes_match_the_stub() -> None:
    device_module = pytest.importorskip("linuxpy.video.device")
    raw_module = pytest.importorskip("linuxpy.video.raw")

    assert int(device_module.BufferFlag.TIMESTAMP_MONOTONIC) == MONOTONIC_FLAG
    buff = raw_module.v4l2_buffer()
    buff.flags = MONOTONIC_FLAG
    buff.timestamp.secs = 12
    buff.timestamp.usecs = 500000
    format_ = device_module.Format(
        width=2,
        height=2,
        pixel_format=device_module.PixelFormat.Z16,
        size=4,
        bytes_per_line=4,
    )
    real_frame = device_module.Frame(b"abcd", buff, format_)

    assert real_frame.timestamp == 12.5
    assert int(real_frame.flags) & MONOTONIC_FLAG


def test_fake_frame_source_implements_the_port() -> None:
    source = fakes.FakeFrameSource()
    assert isinstance(source, ports.FrameSource)
    received: list[ports.RgbdFrame] = []
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.zeros((2, 2), dtype=np.uint16)
    frame = ports.RgbdFrame(0, rgb, depth, 1.0, 1.001)

    source.start(received.append)
    source.emit(frame)
    source.stop()

    assert source.started and source.stopped
    assert received[0] is frame
