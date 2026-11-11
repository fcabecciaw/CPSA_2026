"""IrDepthSource contract: INZI discovery, decode, timestamps and errors.

The V4L2 layer is replaced by a fake stream shaped like the F200 one: it hands
out scripted buffers with a dequeue timestamp, then idles.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from cpsa.adapters.camera import f200, irdepth
from cpsa.adapters.camera.devices import VideoNode
from cpsa.adapters.camera.irdepth import IrDepthSource
from cpsa.vision.irdepth import IrDepthFrame

WIDTH, HEIGHT = 8, 4
RGB_NODE = VideoNode("/dev/video0", "Intel(R) RealSense(TM) 3D Camer", frozenset({"YUYV"}))
INZI_NODE = VideoNode("/dev/video3", "Intel(R) RealSense(TM) 3D Camer", frozenset({"INZI", "Z16"}))

DEPTH = np.array(
    [
        [0x1234, 0, 7, 0xABCD, 0, 0, 300, 0],
        [0, 5, 0, 9, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint16,
)
IR = np.arange(HEIGHT * WIDTH, dtype=np.uint8).reshape(HEIGHT, WIDTH)


def inzi_payload(depth: np.ndarray, ir: np.ndarray) -> bytes:
    pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[:, :, 0] = (depth & 0xFF).astype(np.uint8)
    pixels[:, :, 1] = (depth >> 8).astype(np.uint8)
    pixels[:, :, 2] = ir
    return pixels.tobytes()


@dataclass
class Captured:
    ts: float
    data: bytes
    monotonic: bool


class FakeStream:
    """A ``_Stream`` that hands out scripted captures, then idles."""

    def __init__(self, captures: Sequence[Captured]) -> None:
        self._captures = list(captures)
        self.closed = False
        self._lock = threading.Lock()

    def read(self) -> Captured | None:
        with self._lock:
            if self._captures:
                return self._captures.pop(0)
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


def wait_until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.002)


def make_source(
    stream: FakeStream, **overrides: object
) -> tuple[IrDepthSource, Factory, list[tuple[str, f200.XuSettings]]]:
    factory = Factory(stream)
    applied: list[tuple[str, f200.XuSettings]] = []
    settings: dict[str, object] = {
        "device": "auto",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": 30.0,
        "device_lister": lambda: [RGB_NODE, INZI_NODE],
        "stream_factory": factory,
        "xu_applier": lambda device, xu: applied.append((device, xu)),
    }
    settings.update(overrides)
    return IrDepthSource(**settings), factory, applied


def test_auto_selects_the_inzi_node_and_delivers_decoded_frames() -> None:
    payload = inzi_payload(DEPTH, IR)
    stream = FakeStream([Captured(5.0, payload, True), Captured(5.033, payload, True)])
    source, factory, applied = make_source(stream)
    frames: list[IrDepthFrame] = []
    source.start(frames.append)
    try:
        wait_until(lambda: len(frames) == 2)
    finally:
        source.stop()

    assert factory.calls == [("/dev/video3", WIDTH, HEIGHT, 30.0)]
    assert applied == [("/dev/video3", irdepth.DEFAULT_XU)]
    assert applied[0][1].confidence == 1
    assert applied[0][1].filter == 6
    assert [frame.seq for frame in frames] == [0, 1]
    assert [frame.t for frame in frames] == [5.0, 5.033]
    assert (frames[0].width, frames[0].height) == (WIDTH, HEIGHT)
    np.testing.assert_array_equal(frames[0].ir, IR)
    np.testing.assert_array_equal(frames[0].depth_raw, DEPTH)
    assert frames[0].decode_ms >= 0.0
    assert stream.closed
    assert not source.running


def test_explicit_device_skips_discovery_and_applies_the_custom_xu() -> None:
    def lister() -> list[VideoNode]:
        raise AssertionError("an explicit node must not be resolved")

    xu = f200.XuSettings(laser_power=0, accuracy=1, filter=6, confidence=1)
    source, factory, applied = make_source(
        FakeStream([Captured(1.0, inzi_payload(DEPTH, IR), True)]),
        device="/dev/video-test",
        device_lister=lister,
        xu=xu,
    )
    frames: list[IrDepthFrame] = []
    source.start(frames.append)
    try:
        wait_until(lambda: len(frames) == 1)
    finally:
        source.stop()

    assert factory.calls == [("/dev/video-test", WIDTH, HEIGHT, 30.0)]
    assert applied == [("/dev/video-test", xu)]


def test_non_monotonic_timestamps_fall_back_and_warn_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = inzi_payload(DEPTH, IR)
    stream = FakeStream([Captured(1.0, payload, False), Captured(2.0, payload, False)])
    source, _, _ = make_source(stream)
    frames: list[IrDepthFrame] = []
    before = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="cpsa.adapters.camera.irdepth"):
        source.start(frames.append)
        try:
            wait_until(lambda: len(frames) == 2)
        finally:
            source.stop()
    after = time.monotonic()

    assert all(before <= frame.t <= after for frame in frames)
    warnings = [record for record in caplog.records if "CLOCK_MONOTONIC" in record.message]
    assert len(warnings) == 1


def test_malformed_frame_is_reported_and_stops_the_capture() -> None:
    stream = FakeStream([Captured(1.0, b"\x00\x01\x02", True)])
    source, _, _ = make_source(stream)
    frames: list[IrDepthFrame] = []
    errors: list[Exception] = []
    source.start(frames.append, errors.append)
    try:
        wait_until(lambda: len(errors) == 1)
    finally:
        source.stop()

    assert isinstance(errors[0], ValueError)
    assert frames == []
    assert stream.closed


def test_start_fails_when_no_node_offers_inzi() -> None:
    source, factory, _ = make_source(FakeStream([]), device_lister=lambda: [RGB_NODE])

    with pytest.raises(RuntimeError, match="no camera node matching"):
        source.start(lambda frame: None)
    assert factory.calls == []
    assert not source.running


def test_the_default_stream_requests_the_inzi_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: dict[str, object] = {}

    def fake_open_video(
        device: str, *, width: int, height: int, fourcc: str, fps: float
    ) -> FakeStream:
        opened.update(device=device, width=width, height=height, fourcc=fourcc, fps=fps)
        return FakeStream([])

    monkeypatch.setattr(irdepth, "_open_video", fake_open_video)
    source = IrDepthSource(
        "/dev/video3",
        width=WIDTH,
        height=HEIGHT,
        fps=30.0,
        xu_applier=lambda device, xu: None,
    )
    source.start(lambda frame: None)
    try:
        assert opened == {
            "device": "/dev/video3",
            "width": WIDTH,
            "height": HEIGHT,
            "fourcc": "INZI",
            "fps": 30.0,
        }
    finally:
        source.stop()


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid INZI size"):
        IrDepthSource(width=0, height=HEIGHT)
    with pytest.raises(ValueError, match="fps must be positive"):
        IrDepthSource(fps=0.0)
    with pytest.raises(ValueError, match="match must not be empty"):
        IrDepthSource(match=" ")
