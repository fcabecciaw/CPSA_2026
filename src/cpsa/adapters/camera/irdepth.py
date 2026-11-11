"""Intel RealSense F200 ``INZI`` IR/depth capture over V4L2 (hands demos).

The F200 depth node offers ``INZI``: one 640x480 frame of three interleaved
bytes per pixel, Z16 depth followed by 8-bit IR. IR and depth are therefore
pixel-aligned and arrive on the same clock, which is what the hand mode needs.

The node is chosen like the F200 source chooses its nodes: ``"auto"`` picks the
node whose card name matches and that offers ``INZI``, so a USB re-plug that
renumbers ``/dev/videoN`` is recovered at the next :meth:`IrDepthSource.start`.
The depth extension-unit settings are the F200 ones (``confidence``, ``filter``
and the rest), applied and read back on every open.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the installed extra
    cv2 = None  # type: ignore[assignment]

from cpsa.adapters.camera.devices import (
    AUTO,
    DEFAULT_MATCH,
    VideoNode,
    list_video_nodes,
    resolve_device,
)
from cpsa.adapters.camera.f200 import XuSettings, _apply_xu, _open_video
from cpsa.vision.irdepth import (
    CLAHE_CLIP,
    CLAHE_TILES,
    IrDepthFrame,
    decode_inzi,
    normalize_ir,
)

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_XU", "INZI_FOURCC", "IrDepthSource", "ir_to_input"]

_CHANNELS = 3

INZI_FOURCC = "INZI"
"""Pixel format of the F200 node that carries aligned depth and IR."""

DEFAULT_XU = XuSettings(laser_power=16, accuracy=2, filter=6, confidence=1)
"""Depth XU values of the demo configuration (``config/board.demo.yaml``)."""

_STOP_TIMEOUT_S = 2.0


def ir_to_input(ir: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Turn one ``(H, W)`` IR image into the ``(H, W, 3)`` model input.

    The image is min-max normalised to the full 8-bit range (a flat image maps
    to zeros), contrast-limited adaptive equalisation is applied with
    :data:`cpsa.vision.irdepth.CLAHE_CLIP` and
    :data:`cpsa.vision.irdepth.CLAHE_TILES`, and the result is replicated to
    three channels.

    Raises:
        ValueError: If ``ir`` is not a 2-D uint8 array.
        RuntimeError: If OpenCV is not installed.
    """

    normalised = normalize_ir(ir)
    if cv2 is None:  # pragma: no cover - depends on the installed extra
        raise RuntimeError(
            "OpenCV is required for the hand IR preprocessing; "
            "install the 'vision' extra (uv sync --extra vision)"
        )
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
    enhanced = np.asarray(clahe.apply(normalised), dtype=np.uint8)
    return np.repeat(enhanced[:, :, np.newaxis], _CHANNELS, axis=2)


class _Captured(Protocol):
    """One dequeued V4L2 buffer, as the F200 capture helpers return it."""

    @property
    def ts(self) -> float: ...

    @property
    def data(self) -> bytes: ...

    @property
    def monotonic(self) -> bool: ...


class _Stream(Protocol):
    def read(self) -> _Captured | None: ...

    def close(self) -> None: ...


_StreamFactory = Callable[[str, int, int, float], _Stream]
_XuApplier = Callable[[str, XuSettings], None]


def _open_stream(device: str, width: int, height: int, fps: float) -> _Stream:
    return _open_video(device, width=width, height=height, fourcc=INZI_FOURCC, fps=fps)


class IrDepthSource:
    """IR/depth source for the F200 ``INZI`` node.

    Args:
        device: V4L2 node, or ``"auto"`` to pick the node whose card name
            contains ``match`` and that offers ``INZI`` at every :meth:`start`.
        width, height: requested frame size; the hand demos use 640x480.
        fps: requested frame rate.
        xu: depth extension-unit values applied on every open; ``None`` skips
            them.
        match: case-insensitive card-name fragment used for ``"auto"``.
        device_lister: test seam returning the V4L2 nodes for ``"auto"``;
            defaults to :func:`cpsa.adapters.camera.devices.list_video_nodes`.
        stream_factory: test seam; defaults to the linuxpy implementation.
        xu_applier: test seam applying the XU values; defaults to the F200
            ``UVCIOC_CTRL_QUERY`` implementation.

    Raises:
        ValueError: If a size is not positive, ``fps`` is not positive or
            ``match`` is empty.
    """

    def __init__(
        self,
        device: str = AUTO,
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        xu: XuSettings | None = DEFAULT_XU,
        match: str = DEFAULT_MATCH,
        device_lister: Callable[[], Sequence[VideoNode]] | None = None,
        stream_factory: _StreamFactory | None = None,
        xu_applier: _XuApplier | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid INZI size {width}x{height}")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if not match.strip():
            raise ValueError("match must not be empty")
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._xu = xu
        self._match = match
        self._lister = device_lister if device_lister is not None else list_video_nodes
        self._factory = stream_factory if stream_factory is not None else _open_stream
        self._apply_xu = xu_applier if xu_applier is not None else _apply_xu
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
        on_frame: Callable[[IrDepthFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Open the node and deliver decoded IR/depth frames from one thread."""

        with self._lifecycle:
            if self._running:
                raise RuntimeError("IrDepthSource is already started")
            self._stop = threading.Event()
            self._warned_monotonic = False
            device = resolve_device(
                self._device,
                match=self._match,
                require=INZI_FOURCC,
                forbid=(),
                lister=self._lister,
            )
            if self._xu is not None:
                self._apply_xu(device, self._xu)
            stream = self._factory(device, self._width, self._height, self._fps)
            self._stream = stream
            self._thread = threading.Thread(
                target=self._deliver,
                args=(on_frame, on_error),
                name="cpsa-irdepth-capture",
                daemon=True,
            )
            self._running = True
            try:
                self._thread.start()
            except Exception:
                self._running = False
                self._thread = None
                self._stream = None
                _close_quietly(stream)
                raise

    def stop(self) -> None:
        """Stop delivery, join the thread and close the node."""

        with self._lifecycle:
            if not self._running:
                return
            self._stop.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=_STOP_TIMEOUT_S)
                if thread.is_alive():
                    logger.warning("%s did not stop within %.0f s", thread.name, _STOP_TIMEOUT_S)
            stream = self._stream
            if stream is not None:
                _close_quietly(stream)
            self._stream = None
            self._thread = None
            self._running = False

    def _deliver(
        self,
        on_frame: Callable[[IrDepthFrame], None],
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
                started = time.perf_counter()
                ir, depth_raw = decode_inzi(captured.data, self._width, self._height)
                decode_ms = (time.perf_counter() - started) * 1000.0
                frame = IrDepthFrame(
                    seq=seq,
                    t=captured.ts if captured.monotonic else self._fallback_timestamp(),
                    width=self._width,
                    height=self._height,
                    ir=ir,
                    depth_raw=depth_raw,
                    decode_ms=decode_ms,
                )
                seq += 1
                try:
                    on_frame(frame)
                except Exception:
                    logger.exception("IR/depth frame callback failed")
            except Exception as exc:
                if self._stop.is_set():
                    return
                logger.exception("%s capture stopped by an error", self._device)
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:
                        logger.exception("IR/depth error callback failed")
                return

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
        logger.warning("failed to close the IR/depth stream", exc_info=True)
