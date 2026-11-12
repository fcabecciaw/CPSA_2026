"""The RGB camera: V4L2 capture, node auto-selection and YUYV conversion.

The demo camera re-enumerates when its USB cable is re-plugged: the RGB node
that was ``/dev/video0`` may come back as ``/dev/video1``, so a hard-coded path
silently captures nothing. With ``camera.device: auto`` the node is resolved by
card name at every :meth:`UvcCamera.start`, which is also what makes the
pipeline's re-open loop work: a re-plug is a new node, found again.

Selection is by card name and offered pixel formats: the F200's RGB node offers
``YUYV`` and no ``Z16``, its depth node the opposite. Explicit ``/dev/videoN``
paths are used unchanged.

The capture thread hands out :class:`VideoFrame` objects that keep the raw YUYV
bytes; the conversion to ``(H, W, 3)`` RGB runs in the pipeline loop with
:func:`cv2.cvtColor`, the fastest conversion on the KV260's Cortex-A53.
"""

from __future__ import annotations

import errno
import logging
import re
import select
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from cpsa.config import WHITE_BALANCE_AUTO, CameraConfig

logger = logging.getLogger(__name__)

__all__ = [
    "AUTO",
    "DEPTH_FOURCC",
    "DEFAULT_MATCH",
    "RGB_FOURCC",
    "CameraError",
    "CameraNotFound",
    "UvcCamera",
    "VideoFrame",
    "VideoNode",
    "list_video_nodes",
    "resolve_device",
    "select_video_node",
    "yuyv_to_rgb",
]

AUTO = "auto"
"""Device value that asks the camera to discover its V4L2 node."""

DEFAULT_MATCH = "RealSense"
"""Card-name fragment that the demo camera reports over V4L2.

The kernel truncates the F200 card name to 31 characters
(``"Intel(R) RealSense(TM) 3D Camer"``), so it never contains ``"F200"``.
"""

RGB_FOURCC = "YUYV"
DEPTH_FOURCC = "Z16"

FOURCC = "YUYV"
POWER_LINE_MENU: dict[int, int] = {0: 0, 50: 1, 60: 2}
"""UVC ``power_line_frequency`` menu index by mains frequency in Hz."""

AUTO_EXPOSURE_MANUAL = 1
_READ_TIMEOUT_S = 0.5


class CameraError(RuntimeError):
    """The node is there but cannot be opened or configured."""


class CameraNotFound(CameraError):
    """No V4L2 node matches the configured device."""


@dataclass(frozen=True)
class VideoFrame:
    """One camera frame. ``decode`` returns ``(height, width, 3)`` RGB uint8.

    Decoding may be costly (a YUYV conversion), so the capture thread keeps the
    raw buffer and only the frame the pipeline analyzes is converted.
    """

    seq: int
    t: float
    width: int
    height: int
    decode: Callable[[], NDArray[np.uint8]]


@dataclass(frozen=True)
class VideoNode:
    """One V4L2 node: its device path, card name and offered pixel formats."""

    path: str
    card: str
    formats: frozenset[str]

    def __str__(self) -> str:
        offered = ", ".join(sorted(self.formats)) or "no pixel formats"
        return f"{self.path} {self.card!r} [{offered}]"


def _node_index(path: str) -> int:
    """Numeric suffix of a ``videoN`` path, so video2 sorts before video10."""

    match = re.search(r"(\d+)$", path)
    return -1 if match is None else int(match.group(1))


def list_video_nodes(directory: Path = Path("/dev")) -> list[VideoNode]:
    """Enumerate the V4L2 capture nodes in ``directory``.

    Nodes that cannot be queried (metadata nodes, busy devices, permission
    errors) are skipped: a working camera may sit next to them.

    Raises:
        RuntimeError: If ``linuxpy`` is unavailable.
    """

    try:
        from linuxpy.video.device import Device
    except ImportError as exc:  # pragma: no cover - depends on the board
        raise RuntimeError("linuxpy is required for camera auto-detection") from exc

    nodes: list[VideoNode] = []
    for path in sorted(directory.glob("video*"), key=lambda entry: _node_index(str(entry))):
        try:
            with Device(str(path), read_write=False) as device:
                card = device.info.card
                formats = frozenset(fmt.pixel_format.name for fmt in device.info.formats)
        except Exception:
            logger.debug("Skipping %s: it cannot be queried", path, exc_info=True)
            continue
        nodes.append(VideoNode(path=str(path), card=card, formats=formats))
    return nodes


def select_video_node(
    nodes: Sequence[VideoNode],
    *,
    match: str,
    require: str = RGB_FOURCC,
    forbid: tuple[str, ...] = (DEPTH_FOURCC,),
) -> VideoNode:
    """Return the first node whose card contains ``match`` and whose formats fit.

    Args:
        nodes: Candidate nodes, already in preference order.
        match: Case-insensitive fragment of the V4L2 card name.
        require: Pixel format the node must offer.
        forbid: Pixel formats the node must not offer; the RGB rule excludes the
            depth format so the two nodes of one camera never collide.

    Raises:
        CameraNotFound: If no candidate fits; the message lists what was found.
    """

    wanted = match.casefold()
    excluded = frozenset(forbid)
    for node in nodes:
        if (
            wanted in node.card.casefold()
            and require in node.formats
            and not excluded & node.formats
        ):
            return node
    found = "; ".join(str(node) for node in nodes) or "no V4L2 nodes"
    constraints = require if not forbid else f"{require} without {'/'.join(forbid)}"
    raise CameraNotFound(f"no camera node matching {match!r} with {constraints}; found: {found}")


def resolve_device(
    device: str,
    *,
    match: str = DEFAULT_MATCH,
    require: str = RGB_FOURCC,
    forbid: tuple[str, ...] = (DEPTH_FOURCC,),
    lister: Callable[[], Sequence[VideoNode]] = list_video_nodes,
) -> str:
    """Resolve a configured device to a V4L2 path.

    An explicit path is returned unchanged; ``"auto"`` picks the matching node
    and logs the choice, so the journal records which node served a run.

    Args:
        device: Configured device path, or ``"auto"``.
        match: Card-name fragment used for ``"auto"``.
        require: Pixel format the node must offer.
        forbid: Pixel formats the node must not offer.
        lister: Test seam returning the available nodes.

    Raises:
        CameraNotFound: If ``device`` is ``"auto"`` and no node matches.
    """

    if device.strip().casefold() != AUTO:
        return device
    node = select_video_node(list(lister()), match=match, require=require, forbid=forbid)
    logger.info(
        "Camera device 'auto' selected %s (card %r, formats %s)",
        node.path,
        node.card,
        ", ".join(sorted(node.formats)),
    )
    return node.path


def yuyv_to_rgb(data: bytes, width: int, height: int, *, mirror: bool = False) -> NDArray[np.uint8]:
    """Convert one packed YUYV frame to an ``(height, width, 3)`` RGB uint8 array.

    ``cv2.cvtColor`` is the conversion the showcase uses: about an order of
    magnitude faster than the NumPy BT.601 path it replaces on the KV260, where
    the colour conversion of one 640x480 frame is a real share of the 16.7 ms of
    a 60 fps frame.

    Raises:
        ValueError: If the buffer is not one packed YUYV frame of ``width`` x
            ``height``.
    """

    pairs = np.frombuffer(data, dtype=np.uint8)
    expected = width * height * 2
    if width <= 0 or height <= 0 or width % 2 or pairs.size != expected:
        raise ValueError(f"expected {expected} YUYV bytes for {width}x{height}, got {pairs.size}")
    converted = cv2.cvtColor(pairs.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUYV)
    if mirror:
        # A mirror is what people expect when they face a screen: flipping the
        # frame before detection keeps the drawn labels readable and makes
        # every coordinate (boxes, keypoints, gaze) match what the screen shows.
        converted = cv2.flip(converted, 1)
    return np.asarray(converted, dtype=np.uint8)


@dataclass(frozen=True)
class _Captured:
    ts: float
    data: bytes
    monotonic: bool


class _Stream(Protocol):
    def set_control(self, name: str, value: int) -> None: ...

    def read(self) -> _Captured | None: ...

    def close(self) -> None: ...


_StreamFactory = Callable[[str, int, int, float], _Stream]


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
            raise CameraError(f"{self._path}: control '{name}' not available") from exc
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


def _open_stream(device: str, width: int, height: int, fps: float) -> _Stream:
    """Open one V4L2 node in YUYV at the requested geometry."""

    try:
        from linuxpy.video.device import BufferFlag, Device, VideoCapture
    except ImportError as exc:  # pragma: no cover - depends on the board
        raise CameraError("linuxpy is required for camera capture") from exc

    node = Device(device, blocking=False)
    capture = None
    try:
        node.open()
        capture = VideoCapture(node, size=4)
        capture.set_format(width, height, FOURCC)
        capture.set_fps(fps)
        actual = capture.get_format()
        if actual.width != width or actual.height != height or actual.pixel_format.name != FOURCC:
            raise CameraError(
                f"driver selected {actual.width}x{actual.height} "
                f"{actual.pixel_format.name}, expected {width}x{height} {FOURCC}"
            )
        node.controls["auto_exposure"].value = AUTO_EXPOSURE_MANUAL
        capture.open()
    except FileNotFoundError as exc:
        raise CameraNotFound(f"{device}: {exc}") from exc
    except CameraError:
        _close_capture(capture, node, device)
        raise
    except Exception as exc:
        _close_capture(capture, node, device)
        raise CameraError(f"{device}: {exc}") from exc
    return _LinuxpyStream(node, capture, device, int(BufferFlag.TIMESTAMP_MONOTONIC))


def _close_capture(capture: Any, node: Any, device: str) -> None:
    """Best-effort close of a capture that failed to start."""
    if capture is not None:
        try:
            capture.close()
        except Exception:
            logger.warning("failed to close %s after a start error", device, exc_info=True)
    try:
        node.close()
    except Exception:
        logger.warning("failed to close %s after a start error", device, exc_info=True)


class UvcCamera:
    """One RGB-only UVC camera at ``camera`` settings, capturing in a thread.

    Args:
        config: F-K3 camera settings; ``device: auto`` resolves the node at
            every :meth:`start`, by card name.
        device_lister: Test seam returning the V4L2 nodes for ``"auto"``.
        stream_factory: Test seam; defaults to the linuxpy implementation.
    """

    def __init__(
        self,
        config: CameraConfig,
        *,
        device_lister: Callable[[], Sequence[VideoNode]] | None = None,
        stream_factory: _StreamFactory | None = None,
    ) -> None:
        self._config = config
        self._match = config.match
        self._lister = device_lister if device_lister is not None else list_video_nodes
        self._factory = stream_factory if stream_factory is not None else _open_stream
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: _Stream | None = None
        self._running = False
        self._warned_monotonic = False

    @property
    def config(self) -> CameraConfig:
        """The settings this camera captures with."""
        return self._config

    @property
    def running(self) -> bool:
        """True while the delivery thread is active."""
        return self._running and not self._stop.is_set()

    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Open the camera and deliver raw YUYV frames from one thread.

        Raises:
            CameraNotFound: If ``device: auto`` finds no matching node, or the
                node disappeared between discovery and open.
            CameraError: If the node cannot be configured or the thread cannot
                start.
            RuntimeError: If this camera is already running.
        """

        with self._lifecycle:
            if self._running:
                raise RuntimeError("the camera is already started")
            self._stop = threading.Event()
            self._warned_monotonic = False
            device = resolve_device(self._config.device, match=self._match, lister=self._lister)
            stream = self._factory(
                device, self._config.width, self._config.height, self._config.fps
            )
            try:
                self._apply_controls(stream)
            except Exception:
                _close_quietly(stream)
                raise
            self._stream = stream
            self._thread = threading.Thread(
                target=self._deliver,
                args=(on_frame, on_error),
                name="cpsa-camera",
                daemon=True,
            )
            self._running = True
            try:
                self._thread.start()
            except Exception as exc:
                self._running = False
                _close_quietly(stream)
                self._stream = None
                raise CameraError(f"cannot start the capture thread: {exc}") from exc

    def _apply_controls(self, stream: _Stream) -> None:
        """Set the manual image controls at every open.

        The auto-exposure, auto-white-balance and backlight-compensation
        controls of the F200's ISP are switched off (or set explicitly): left
        on, they keep re-adjusting the image, which is what made the gain look
        too bright and the boxes drift. The manual white balance is set after
        its automatic flag, as the UVC driver only accepts a temperature while
        auto is off.
        """

        config = self._config
        stream.set_control("auto_exposure", AUTO_EXPOSURE_MANUAL)
        stream.set_control("exposure_time_absolute", config.exposure)
        stream.set_control("gain", config.gain)
        stream.set_control("power_line_frequency", POWER_LINE_MENU[config.power_line_hz])
        if config.white_balance == WHITE_BALANCE_AUTO:
            stream.set_control("white_balance_automatic", 1)
        else:
            stream.set_control("white_balance_automatic", 0)
            stream.set_control("white_balance_temperature", int(config.white_balance))
        stream.set_control("backlight_compensation", config.backlight_compensation)
        stream.set_control("saturation", config.saturation)

    def set_controls(
        self,
        *,
        gain: int | None = None,
        exposure: int | None = None,
        saturation: int | None = None,
    ) -> None:
        """Change gain, exposure or saturation of the running stream (live calibration).

        The values replace those of :attr:`config`, so a re-open after a
        re-plug keeps them; on an open stream only the controls whose value
        changed are written, through the same V4L2 ``set_control`` the start
        uses. The capture thread is not involved: the caller runs this off the
        frame loop (the pipeline's calibration worker), as a UVC control
        transfer can take a few milliseconds.

        Raises:
            CameraError: If the stream rejects a control.
        """

        changes: dict[str, int] = {}
        if gain is not None:
            changes["gain"] = int(gain)
        if exposure is not None:
            changes["exposure"] = int(exposure)
        if saturation is not None:
            changes["saturation"] = int(saturation)
        with self._lifecycle:
            previous = self._config
            self._config = replace(
                previous,
                gain=changes.get("gain", previous.gain),
                exposure=changes.get("exposure", previous.exposure),
                saturation=changes.get("saturation", previous.saturation),
            )
            stream = self._stream if self._running else None
            if stream is None:
                return
            if "exposure" in changes and changes["exposure"] != previous.exposure:
                stream.set_control("exposure_time_absolute", changes["exposure"])
            if "gain" in changes and changes["gain"] != previous.gain:
                stream.set_control("gain", changes["gain"])
            if "saturation" in changes and changes["saturation"] != previous.saturation:
                stream.set_control("saturation", changes["saturation"])

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
                logger.exception("%s capture stopped by an error", self._config.device)
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:
                        logger.exception("camera error callback failed")
                return

    def _frame(self, seq: int, captured: _Captured) -> VideoFrame:
        timestamp = captured.ts if captured.monotonic else self._fallback_timestamp()
        data = captured.data
        width = self._config.width
        height = self._config.height
        mirror = self._config.mirror
        return VideoFrame(
            seq=seq,
            t=timestamp,
            width=width,
            height=height,
            decode=lambda: yuyv_to_rgb(data, width, height, mirror=mirror),
        )

    def _fallback_timestamp(self) -> float:
        if not self._warned_monotonic:
            self._warned_monotonic = True
            logger.warning(
                "V4L2 buffer timestamp is not CLOCK_MONOTONIC; "
                "falling back to time.monotonic() at dequeue"
            )
        return time.monotonic()


def _close_quietly(stream: _Stream) -> None:
    try:
        stream.close()
    except Exception:
        logger.warning("failed to close the camera stream", exc_info=True)
