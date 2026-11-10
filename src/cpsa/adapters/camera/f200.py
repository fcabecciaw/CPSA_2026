"""Creative VF0800 (Intel RealSense F200) RGB/depth capture over V4L2.

The capture dependency is :mod:`linuxpy`, chosen after inspecting its 0.25
source: it already exposes the depth fourcc ``Z16`` (``PixelFormat.Z16``), mmap
streaming through ``VideoCapture``/``MemoryMap``, and per-buffer timestamps plus
the ``V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC`` flag on ``Frame.flags``. A custom
ctypes/fcntl mmap implementation would duplicate that code without adding
anything for the two fixed nodes.

The depth extension unit (unit 6, GUID ``a55751a1-f3c5-4a5e-8d5a-6854b8fa2716``)
is not part of the generic V4L2 control API, so ``UVCIOC_CTRL_QUERY`` is issued
directly with fcntl/ctypes (allowed in the adapter layer). The values are
volatile: they are applied and read back on every ``F200Source.start``.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import logging
import os
import queue
import select
import struct
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cpsa.ports import RgbdFrame
from cpsa.vision.pairing import PairedFrame, Pairer, PairingStats

logger = logging.getLogger(__name__)

# UVC extension unit of the F200 depth node.
XU_UNIT = 6
UVCIOC_CTRL_QUERY = (3 << 30) | (16 << 16) | (ord("u") << 8) | 0x21

XU_SET_CUR = 0x01
XU_GET_CUR = 0x81
XU_GET_LEN = 0x85

_XU_LENGTH_QUERY_SIZE = 2
_XU_SELECTORS = {
    "laser_power": 1,
    "accuracy": 2,
    "filter": 5,
    "confidence": 6,
}
_XU_SETTINGS = ("laser_power", "accuracy", "filter", "confidence")
_XU_SETTINGS_EXPECTED_LENGTH = 1

RGB_FOURCC = "YUYV"
DEPTH_FOURCC = "Z16 "
RGB_AUTO_EXPOSURE_MANUAL = 1

_READ_TIMEOUT_S = 0.5

# Cached 8-bit BT.601 coefficients for YUYV -> RGB.
_Y_OFFSET = 16.0
_UV_OFFSET = 128.0
_Y_SCALE = 1.164383
_R_V = 1.596027
_G_U = 0.391762
_G_V = 0.812968
_B_U = 2.017232


# Fixed-point BT.601 lookup tables. Kept in the int16 range: the largest
# luma/chroma contribution is below 512, so sums cannot overflow.
_Y_LUT = np.array([round(_Y_SCALE * (value - _Y_OFFSET)) for value in range(256)], dtype=np.int16)
_R_V_LUT = np.array([round(_R_V * (value - _UV_OFFSET)) for value in range(256)], dtype=np.int16)
_G_U_LUT = np.array([round(-_G_U * (value - _UV_OFFSET)) for value in range(256)], dtype=np.int16)
_G_V_LUT = np.array([round(-_G_V * (value - _UV_OFFSET)) for value in range(256)], dtype=np.int16)
_B_U_LUT = np.array([round(_B_U * (value - _UV_OFFSET)) for value in range(256)], dtype=np.int16)


@dataclass(frozen=True)
class XuSettings:
    """Depth extension-unit values applied on every camera open."""

    laser_power: int
    accuracy: int
    filter: int
    confidence: int


@dataclass(frozen=True)
class _Captured:
    ts: float
    data: bytes
    monotonic: bool


class _VideoStream(Protocol):
    def set_control(self, name: str, value: int) -> None: ...

    def read(self) -> _Captured | None: ...

    def close(self) -> None: ...


class _XuDevice(Protocol):
    def length(self, selector: int) -> int: ...

    def get(self, selector: int) -> int: ...

    def set(self, selector: int, value: int) -> None: ...

    def close(self) -> None: ...


def _to_uint8(values: NDArray[np.int16]) -> NDArray[np.uint8]:
    return np.clip(values, 0, 255).astype(np.uint8)


def yuyv_to_rgb(data: bytes, width: int, height: int) -> NDArray[np.uint8]:
    """Convert one packed YUYV frame to an ``(height, width, 3)`` RGB uint8 array.

    Uses integer BT.601 lookup tables and writes the two pixels of each YUYV
    macropixel separately, avoiding full-width chroma temporaries.
    """
    if width <= 0 or height <= 0 or width % 2:
        raise ValueError(f"invalid YUYV size {width}x{height}")
    raw = np.frombuffer(data, dtype=np.uint8)
    expected = width * height * 2
    if raw.size != expected:
        raise ValueError(f"expected {expected} YUYV bytes, got {raw.size}")
    pairs_in = raw.reshape(height, width // 2, 4)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    pairs_out = rgb.reshape(height, width // 2, 2, 3)

    y0 = _Y_LUT[pairs_in[:, :, 0]]
    y1 = _Y_LUT[pairs_in[:, :, 2]]
    u = pairs_in[:, :, 1]
    v = pairs_in[:, :, 3]
    r_v = _R_V_LUT[v]
    g_u = _G_U_LUT[u]
    g_v = _G_V_LUT[v]
    b_u = _B_U_LUT[u]

    pairs_out[:, :, 0, 0] = _to_uint8(y0 + r_v)
    pairs_out[:, :, 0, 1] = _to_uint8(y0 + g_u + g_v)
    pairs_out[:, :, 0, 2] = _to_uint8(y0 + b_u)
    pairs_out[:, :, 1, 0] = _to_uint8(y1 + r_v)
    pairs_out[:, :, 1, 1] = _to_uint8(y1 + g_u + g_v)
    pairs_out[:, :, 1, 2] = _to_uint8(y1 + b_u)
    return rgb


def z16_to_depth(data: bytes, width: int, height: int) -> NDArray[np.uint16]:
    """Convert one raw ``Z16`` frame to an ``(height, width)`` uint16 array."""
    expected = width * height * 2
    if len(data) != expected:
        raise ValueError(f"expected {expected} Z16 bytes, got {len(data)}")
    return np.frombuffer(data, dtype="<u2").reshape(height, width).copy()


class _FdXu:
    """``UVCIOC_CTRL_QUERY`` access to the depth extension unit on one fd."""

    def __init__(self, fd: int, device: str) -> None:
        self._fd = fd
        self._device = device

    def _query(self, selector: int, query: int, size: int, value: int | None = None) -> int:
        data = ctypes.create_string_buffer(size)
        if value is not None:
            data.raw = int(value).to_bytes(size, "little")
        request = struct.pack("BBBxHxxQ", XU_UNIT, selector, query, size, ctypes.addressof(data))
        try:
            fcntl.ioctl(self._fd, UVCIOC_CTRL_QUERY, bytearray(request))
        except OSError as exc:
            raise RuntimeError(
                f"{self._device}: UVCIOC_CTRL_QUERY failed for selector {selector}: {exc}"
            ) from exc
        return int.from_bytes(data.raw, "little")

    def length(self, selector: int) -> int:
        return self._query(selector, XU_GET_LEN, _XU_LENGTH_QUERY_SIZE)

    def get(self, selector: int) -> int:
        return self._query(selector, XU_GET_CUR, self.length(selector))

    def set(self, selector: int, value: int) -> None:
        size = self.length(selector)
        if value < 0 or value >= 1 << (8 * size):
            raise RuntimeError(
                f"{self._device}: XU control {selector} value {value} does not fit {size} bytes"
            )
        self._query(selector, XU_SET_CUR, size, value)

    def close(self) -> None:
        os.close(self._fd)


def _open_xu(device: str) -> _FdXu:
    try:
        fd = os.open(device, os.O_RDWR)
    except OSError as exc:
        raise RuntimeError(f"{device}: cannot open node for XU access: {exc}") from exc
    return _FdXu(fd, device)


def _configure_xu(xu: _XuDevice, settings: XuSettings) -> None:
    for name in _XU_SETTINGS:
        selector = _XU_SELECTORS[name]
        wanted = getattr(settings, name)
        length = xu.length(selector)
        if length != _XU_SETTINGS_EXPECTED_LENGTH:
            raise RuntimeError(f"depth XU control '{name}' has unexpected length {length}")
        xu.set(selector, wanted)
        actual = xu.get(selector)
        if actual != wanted:
            raise RuntimeError(f"depth XU control '{name}' read back {actual}, expected {wanted}")


def _apply_xu(device: str, settings: XuSettings) -> None:
    xu = _open_xu(device)
    try:
        try:
            _configure_xu(xu, settings)
        except RuntimeError as exc:
            raise RuntimeError(f"{device}: {exc}") from exc
    finally:
        xu.close()


def read_xu(device: str) -> dict[str, int]:
    """Read the current values of the four depth XU controls from ``device``."""
    xu = _open_xu(device)
    try:
        return {name: xu.get(_XU_SELECTORS[name]) for name in _XU_SETTINGS}
    finally:
        xu.close()


class _LinuxpyStream:
    """Thin wrapper over linuxpy's mmap capture exposing buffer timestamps."""

    def __init__(
        self,
        device: Any,
        capture: Any,
        path: str,
        monotonic_flag: int,
    ) -> None:
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
        readable, _, _ = select.select(
            [self._device.fileno()],
            [],
            [],
            _READ_TIMEOUT_S,
        )
        if not readable:
            return None
        try:
            # ``VideoCapture.buffer.read()`` routes to ``wait_read()`` on a
            # non-blocking fd, and linuxpy 0.25's wait_read calls the select
            # *module* instead of select.select. Do the wait here and use the
            # raw read, which only dequeues an already-ready buffer.
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


def _open_video(
    path: str,
    *,
    width: int,
    height: int,
    fourcc: str,
    fps: float,
    controls: Mapping[str, int] | None = None,
) -> _VideoStream:
    try:
        from linuxpy.video.device import BufferFlag, Device, VideoCapture
    except ImportError as exc:
        raise RuntimeError(
            "linuxpy is required for camera capture; install the 'camera' extra"
        ) from exc

    # Keep the fd non-blocking and do the readiness wait ourselves: linuxpy
    # 0.25's wait_read() calls the select module instead of select.select.
    device = Device(path, blocking=False)
    capture = None
    try:
        device.open()
        capture = VideoCapture(device, size=4)
        capture.set_format(width, height, fourcc)
        capture.set_fps(fps)
        actual = capture.get_format()
        expected_fourcc = fourcc.strip()
        if (
            actual.width != width
            or actual.height != height
            or actual.pixel_format.name != expected_fourcc
        ):
            raise RuntimeError(
                f"driver selected {actual.width}x{actual.height} "
                f"{actual.pixel_format.name}, expected {width}x{height} {expected_fourcc}"
            )
        for name, value in (controls or {}).items():
            device.controls[name].value = value
        capture.open()
    except Exception:
        if capture is not None:
            try:
                capture.close()
            except Exception:
                logger.warning("failed to close %s after a start error", path, exc_info=True)
        try:
            device.close()
        except Exception:
            logger.warning("failed to close %s after a start error", path, exc_info=True)
        raise
    return _LinuxpyStream(device, capture, path, int(BufferFlag.TIMESTAMP_MONOTONIC))


class F200Source:
    """``FrameSource`` for the F200 RGB/depth pair over two V4L2 nodes."""

    def __init__(
        self,
        rgb_device: str,
        depth_device: str,
        *,
        width: int,
        height: int,
        fps: float,
        rgb_exposure: int,
        xu: XuSettings,
    ) -> None:
        self._rgb_device = rgb_device
        self._depth_device = depth_device
        self._width = width
        self._height = height
        self._fps = fps
        self._rgb_exposure = rgb_exposure
        self._xu = xu
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        self._pairs: queue.Queue[PairedFrame] = queue.Queue()
        self._pairer: Pairer | None = None
        self._streams: tuple[_VideoStream, ...] = ()
        self._capture_threads: tuple[threading.Thread, ...] = ()
        self._delivery_thread: threading.Thread | None = None
        self._warned_monotonic = False
        self._warning_lock = threading.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        """True while capture threads are delivering frames."""
        return self._running and not self._stop.is_set()

    @property
    def stats(self) -> PairingStats | None:
        """Pairing counters of the current or most recent run."""
        pairer = self._pairer
        return None if pairer is None else pairer.stats()

    def start(self, on_frame: Callable[[RgbdFrame], None]) -> None:
        """Open both nodes and deliver paired frames to ``on_frame``."""
        with self._lifecycle:
            if self._running:
                raise RuntimeError("F200Source is already started")
            self._stop = threading.Event()
            self._pairs = queue.Queue()
            self._pairer = Pairer(self._fps)
            self._warned_monotonic = False
            self._streams = ()

            _apply_xu(self._depth_device, self._xu)
            try:
                rgb_stream = _open_video(
                    self._rgb_device,
                    width=self._width,
                    height=self._height,
                    fourcc=RGB_FOURCC,
                    fps=self._fps,
                    controls={
                        "auto_exposure": RGB_AUTO_EXPOSURE_MANUAL,
                        "exposure_time_absolute": self._rgb_exposure,
                    },
                )
            except Exception as exc:
                raise RuntimeError(f"{self._rgb_device}: {exc}") from exc
            try:
                depth_stream = _open_video(
                    self._depth_device,
                    width=self._width,
                    height=self._height,
                    fourcc=DEPTH_FOURCC,
                    fps=self._fps,
                )
            except Exception as exc:
                _close_quietly(rgb_stream)
                raise RuntimeError(f"{self._depth_device}: {exc}") from exc

            self._streams = (rgb_stream, depth_stream)
            pairer = self._pairer
            rgb_thread = threading.Thread(
                target=self._capture_rgb,
                args=(rgb_stream, pairer),
                name="cpsa-camera-rgb",
                daemon=True,
            )
            depth_thread = threading.Thread(
                target=self._capture_depth,
                args=(depth_stream, pairer),
                name="cpsa-camera-depth",
                daemon=True,
            )
            delivery_thread = threading.Thread(
                target=self._deliver,
                args=(on_frame,),
                name="cpsa-camera-delivery",
                daemon=True,
            )
            self._capture_threads = (rgb_thread, depth_thread)
            self._delivery_thread = delivery_thread
            self._running = True
            try:
                for thread in (*self._capture_threads, delivery_thread):
                    thread.start()
            except Exception:
                self._stop.set()
                self._running = False
                for stream in self._streams:
                    _close_quietly(stream)
                self._streams = ()
                raise

    def stop(self) -> None:
        """Stop delivery, join the worker threads and close both devices."""
        with self._lifecycle:
            if not self._running:
                return
            self._stop.set()
            current = threading.current_thread()
            for thread in self._capture_threads:
                if thread is not current:
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        logger.warning("%s did not stop within 2 s", thread.name)
            if self._pairer is not None:
                self._pairer.flush()
            delivery_thread = self._delivery_thread
            if delivery_thread is not None and delivery_thread is not current:
                delivery_thread.join(timeout=5.0)
                if delivery_thread.is_alive():
                    logger.warning("%s did not stop within 5 s", delivery_thread.name)
            for stream in self._streams:
                _close_quietly(stream)
            self._streams = ()
            self._capture_threads = ()
            self._delivery_thread = None
            self._running = False

    def _capture_rgb(self, stream: _VideoStream, pairer: Pairer) -> None:
        while not self._stop.is_set():
            try:
                captured = stream.read()
                if captured is None:
                    continue
                self._publish(pairer.push_rgb(self._timestamp(captured), captured.data))
            except Exception:
                logger.exception("%s capture stopped by an error", self._rgb_device)
                self._stop.set()
                return

    def _capture_depth(self, stream: _VideoStream, pairer: Pairer) -> None:
        while not self._stop.is_set():
            try:
                captured = stream.read()
                if captured is None:
                    continue
                self._publish(pairer.push_depth(self._timestamp(captured), captured.data))
            except Exception:
                logger.exception("%s capture stopped by an error", self._depth_device)
                self._stop.set()
                return

    def _timestamp(self, captured: _Captured) -> float:
        if captured.monotonic:
            return captured.ts
        with self._warning_lock:
            if not self._warned_monotonic:
                logger.warning(
                    "V4L2 buffer timestamp is not CLOCK_MONOTONIC; "
                    "falling back to time.monotonic() at dequeue"
                )
                self._warned_monotonic = True
        return time.monotonic()

    def _publish(self, pair: PairedFrame | None) -> None:
        if pair is not None:
            self._pairs.put(pair)

    def _deliver(self, on_frame: Callable[[RgbdFrame], None]) -> None:
        while True:
            try:
                pair = self._pairs.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                rgb = yuyv_to_rgb(pair.rgb, self._width, self._height)
                depth = z16_to_depth(pair.depth, self._width, self._height)
            except Exception:
                logger.exception("camera frame conversion failed")
                self._stop.set()
                return
            frame = RgbdFrame(
                seq=pair.seq,
                rgb=rgb,
                depth=depth,
                ts_rgb=pair.ts_rgb,
                ts_depth=pair.ts_depth,
            )
            try:
                on_frame(frame)
            except Exception:
                logger.exception("camera frame callback failed")


def _close_quietly(stream: _VideoStream) -> None:
    try:
        stream.close()
    except Exception:
        logger.warning("failed to close a camera stream", exc_info=True)
