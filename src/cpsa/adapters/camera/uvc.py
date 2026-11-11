"""RGB-only UVC capture over V4L2 (NFR-004).

The source requests packed YUYV frames and hands out raw bytes: ``VideoFrame``
decodes lazily, so the capture thread never pays the colour conversion unless a
frame is analyzed or saved. Timestamps come from the V4L2 buffer
(``CLOCK_MONOTONIC``); a driver without the monotonic flag falls back to
``time.monotonic()`` at dequeue and warns once, as the F200 source does.
"""

from __future__ import annotations

import errno
import logging
import select
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cpsa.adapters.camera.devices import (
    DEFAULT_MATCH,
    VideoNode,
    list_video_nodes,
    resolve_device,
)
from cpsa.adapters.camera.f200 import yuyv_to_rgb
from cpsa.vision.types import VideoFrame

logger = logging.getLogger(__name__)

__all__ = ["UvcVideoSource"]

FOURCC = "YUYV"
AUTO_EXPOSURE_MANUAL = 1
_READ_TIMEOUT_S = 0.5


@dataclass(frozen=True)
class _Captured:
    ts: float
    data: bytes
    monotonic: bool


class _Stream(Protocol):
    def set_control(self, name: str, value: int) -> None: ...

    def read(self) -> _Captured | None: ...

    def close(self) -> None: ...


_StreamFactory = Callable[[str, int, int, float, int], _Stream]


class _LinuxpyStream:
    """Thin wrapper over linuxpy's mmap capture exposing buffer timestamps."""

    def __init__(self, device: Any, capture: Any, path: str, monotonic_flag: int) -> None:
        self._device = device
        self._capture = capture
        self._path = path
        self._monotonic_flag = monotonic_flag

    def set_control(self, name: str, value: int) -> None:
        controls = self._device.controls
        try:
            control = controls[name]
        except KeyError as exc:
            raise RuntimeError(f"{self._path}: control '{name}' not available") from exc
        control.value = value

    def read(self) -> _Captured | None:
        readable, _, _ = select.select([self._device.fileno()], [], [], _READ_TIMEOUT_S)
        if not readable:
            return None
        try:
            # ``VideoCapture.buffer.read()`` routes to ``wait_read()`` on a
            # non-blocking fd, and linuxpy 0.25's wait_read calls the select
            # module instead of select.select. Wait here and dequeue directly.
            frame = self._capture.buffer.raw_read()
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return None
            raise
        monotonic = bool(int(frame.flags) & self._monotonic_flag)
        return _Captured(float(frame.timestamp), bytes(frame.data), monotonic)

    def close(self) -> None:
        try:
            self._capture.close()
        finally:
            self._device.close()


def _open_stream(device: str, width: int, height: int, fps: float, exposure: int) -> _Stream:
    """Open one V4L2 node in YUYV and apply the manual exposure."""
    try:
        from linuxpy.video.device import BufferFlag, Device, VideoCapture
    except ImportError as exc:
        raise RuntimeError(
            "linuxpy is required for camera capture; install the 'camera' extra"
        ) from exc

    node = Device(device, blocking=False)
    capture = None
    try:
        node.open()
        capture = VideoCapture(node, size=4)
        capture.set_format(width, height, FOURCC)
        capture.set_fps(fps)
        actual = capture.get_format()
        if actual.width != width or actual.height != height or actual.pixel_format.name != FOURCC:
            raise RuntimeError(
                f"driver selected {actual.width}x{actual.height} "
                f"{actual.pixel_format.name}, expected {width}x{height} {FOURCC}"
            )
        node.controls["auto_exposure"].value = AUTO_EXPOSURE_MANUAL
        node.controls["exposure_time_absolute"].value = exposure
        capture.open()
    except Exception:
        if capture is not None:
            try:
                capture.close()
            except Exception:
                logger.warning("failed to close %s after a start error", device, exc_info=True)
        try:
            node.close()
        except Exception:
            logger.warning("failed to close %s after a start error", device, exc_info=True)
        raise
    return _LinuxpyStream(node, capture, device, int(BufferFlag.TIMESTAMP_MONOTONIC))


class UvcVideoSource:
    """``VideoSource`` for one RGB-only UVC camera.

    Args:
        device: V4L2 node, for example ``/dev/video0``, or ``"auto"`` to pick
            the node whose card name contains ``match`` at every :meth:`start`.
        width, height: requested YUYV frame size.
        fps: requested frame rate.
        exposure: manual ``exposure_time_absolute`` value.
        match: case-insensitive card-name fragment used for ``"auto"``.
        device_lister: test seam returning the V4L2 nodes for ``"auto"``;
            defaults to :func:`cpsa.adapters.camera.devices.list_video_nodes`.
        stream_factory: test seam; defaults to the linuxpy implementation.
    """

    def __init__(
        self,
        device: str,
        *,
        width: int,
        height: int,
        fps: float,
        exposure: int,
        match: str = DEFAULT_MATCH,
        device_lister: Callable[[], Sequence[VideoNode]] | None = None,
        stream_factory: _StreamFactory | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or width % 2:
            raise ValueError(f"invalid YUYV size {width}x{height}")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if exposure <= 0:
            raise ValueError("exposure must be positive")
        if not match.strip():
            raise ValueError("match must not be empty")
        self._device = device
        self._match = match
        self._lister = device_lister if device_lister is not None else list_video_nodes
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._exposure = int(exposure)
        self._factory = stream_factory if stream_factory is not None else _open_stream
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: _Stream | None = None
        self._running = False
        self._warned_monotonic = False

    @property
    def running(self) -> bool:
        """True while the delivery thread is active."""
        return self._running and not self._stop.is_set()

    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Open the camera and deliver raw YUYV frames from one thread."""
        with self._lifecycle:
            if self._running:
                raise RuntimeError("UvcVideoSource is already started")
            self._stop = threading.Event()
            self._warned_monotonic = False
            device = resolve_device(self._device, match=self._match, lister=self._lister)
            stream = self._factory(device, self._width, self._height, self._fps, self._exposure)
            try:
                stream.set_control("auto_exposure", AUTO_EXPOSURE_MANUAL)
                stream.set_control("exposure_time_absolute", self._exposure)
            except Exception:
                _close_quietly(stream)
                raise
            self._stream = stream
            self._thread = threading.Thread(
                target=self._deliver,
                args=(on_frame, on_error),
                name="cpsa-uvc-capture",
                daemon=True,
            )
            self._running = True
            try:
                self._thread.start()
            except Exception:
                self._running = False
                _close_quietly(stream)
                self._stream = None
                raise

    def stop(self) -> None:
        """Stop delivery, join the thread and close the device."""
        with self._lifecycle:
            if not self._running:
                return
            self._stop.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning("%s did not stop within 2 s", thread.name)
            stream = self._stream
            if stream is not None:
                _close_quietly(stream)
            self._stream = None
            self._thread = None
            self._running = False

    def _deliver(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None,
    ) -> None:
        stream = self._stream
        if stream is None:  # pragma: no cover - start always sets it
            return
        seq = 0
        while not self._stop.is_set():
            try:
                captured = stream.read()
                if captured is None:
                    continue
                frame = self._frame(seq, captured)
                seq += 1
                try:
                    on_frame(frame)
                except Exception:
                    logger.exception("camera frame callback failed")
            except Exception as exc:
                if self._stop.is_set():
                    return
                logger.exception("%s capture stopped by an error", self._device)
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:
                        logger.exception("camera error callback failed")
                return

    def _frame(self, seq: int, captured: _Captured) -> VideoFrame:
        timestamp = captured.ts if captured.monotonic else self._fallback_timestamp()
        data = captured.data
        width = self._width
        height = self._height
        return VideoFrame(
            seq=seq,
            t=timestamp,
            width=width,
            height=height,
            decode=lambda: decode_yuyv(data, width, height),
        )

    def _fallback_timestamp(self) -> float:
        if not self._warned_monotonic:
            self._warned_monotonic = True
            logger.warning(
                "V4L2 buffer timestamp is not CLOCK_MONOTONIC; "
                "falling back to time.monotonic() at dequeue"
            )
        return time.monotonic()


def decode_yuyv(data: bytes, width: int, height: int) -> NDArray[np.uint8]:
    """Convert packed YUYV bytes to RGB with OpenCV, or the numpy path."""
    try:
        import cv2
    except ImportError:
        return yuyv_to_rgb(data, width, height)
    pairs = np.frombuffer(data, dtype=np.uint8)
    expected = width * height * 2
    if pairs.size != expected:
        raise ValueError(f"expected {expected} YUYV bytes, got {pairs.size}")
    converted = cv2.cvtColor(pairs.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUYV)
    return np.asarray(converted, dtype=np.uint8)


def _close_quietly(stream: _Stream) -> None:
    try:
        stream.close()
    except Exception:
        logger.warning("failed to close the camera stream", exc_info=True)
