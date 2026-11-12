"""Inter-process state of the showcase: the video frame and the messages.

The showcase runs as two processes (see ``docs/plans/showcase-only.md``): the
pipeline (camera, DPU, overlay) and the server (HTTP, WebSocket, JPEG). They
share exactly two things here:

* :class:`FrameChannel` — one fixed-size shared-memory buffer carrying the
  latest overlaid RGB frame. The writer copies into buffer ``seq % 2`` and then
  publishes ``seq``; the reader copies the buffer of the newest published
  sequence and re-reads the sequence, so a frame is either complete or skipped,
  never torn. The writer never waits for the reader, and the reader never waits
  for the writer: a slow server drops frames instead of slowing the pipeline.
* :class:`MessageChannel` — a bounded queue of ``(topic, payload)`` snapshots.
  When it is full the oldest message is discarded, because every message is a
  snapshot of the current state and only the newest one matters.

Both are plain Python on top of ``multiprocessing`` primitives, so a test can
drive them in one process with a fake queue.
"""

from __future__ import annotations

import contextlib
import os
import queue
import struct
from collections.abc import Iterator
from multiprocessing import Queue, shared_memory
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

__all__ = ["FRAME_MAGIC", "FrameChannel", "MessageChannel"]

FRAME_MAGIC = b"CPSAFR01"
"""First eight bytes of the shared-memory header; a wrong magic is a stale buffer."""

_NAME_PREFIX = "cpsa-"
"""Shared-memory names carry this prefix, so ``ls /dev/shm`` shows the showcase."""

_HEADER_SIZE = 64
"""Header bytes before the two frame buffers: magic, sequence and geometry."""

_SEQUENCE = struct.Struct("<Q")
"""Published sequence number at header offset 8."""

_GEOMETRY = struct.Struct("<II")
"""``width`` and ``height`` at header offset 16."""

_READ_RETRIES = 4
"""Copies a reader attempts before giving up on a frame the writer keeps replacing."""


class FrameChannel:
    """The latest RGB frame in shared memory, guarded by a published sequence.

    A writer (:meth:`write`) publishes one frame at a time; a reader
    (:meth:`read`) copies the newest published frame it has not seen yet.
    Neither side blocks the other.

    Args:
        name: Shared-memory name; ``None`` creates a new one.
        width, height: Frame geometry in pixels, RGB uint8.
        create: ``True`` to allocate the segment, ``False`` to attach by name.

    Raises:
        ValueError: If the geometry is not positive or the attached segment is
            too small or carries the wrong magic.
    """

    def __init__(self, name: str | None, width: int, height: int, *, create: bool) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid frame geometry {width}x{height}")
        self._width = int(width)
        self._height = int(height)
        self._frame_bytes = self._width * self._height * 3
        size = _HEADER_SIZE + 2 * self._frame_bytes
        self._owner = create
        self._shm = (
            shared_memory.SharedMemory(name=name, create=True, size=size)
            if create
            else shared_memory.SharedMemory(name=name, create=False)
        )
        if self._shm.size < size:
            self.close()
            raise ValueError(
                f"shared memory {self._shm.name!r} holds {self._shm.size} bytes, need {size}"
            )
        buffer = self._shm.buf
        if buffer is None:  # pragma: no cover - a closed SharedMemory handle
            self.close()
            raise ValueError(f"shared memory {self._shm.name!r} is already closed")
        self._buffer: memoryview = buffer
        if create:
            _SEQUENCE.pack_into(self._buffer, 8, 0)
            _GEOMETRY.pack_into(self._buffer, 16, self._width, self._height)
            self._buffer[0:8] = FRAME_MAGIC
        elif bytes(self._buffer[0:8]) != FRAME_MAGIC:
            self.close()
            raise ValueError(f"shared memory {self._shm.name!r} is not a frame channel")
        self._frames_written = 0
        self._frames_read = 0
        self._frames_dropped = 0
        self._last_seq = 0

    @classmethod
    def create(cls, width: int, height: int, name: str | None = None) -> FrameChannel:
        """Allocate a channel and return the owning handle."""
        if name is None:
            name = f"{_NAME_PREFIX}{os.getpid()}-{uuid4().hex[:8]}"
        return cls(name, width, height, create=True)

    @classmethod
    def attach(cls, name: str, width: int, height: int) -> FrameChannel:
        """Attach to a channel another process created."""
        return cls(name, width, height, create=False)

    @property
    def name(self) -> str:
        """Shared-memory name, what the other process attaches by."""
        return str(self._shm.name)

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def last_seq(self) -> int:
        """Sequence of the newest frame this handle wrote or read."""
        return self._last_seq

    @property
    def frames_written(self) -> int:
        """Frames this handle published (writer side)."""
        return self._frames_written

    @property
    def frames_read(self) -> int:
        """Frames this handle copied (reader side)."""
        return self._frames_read

    @property
    def frames_dropped(self) -> int:
        """Frames a reader skipped because a newer one had replaced them."""
        return self._frames_dropped

    def write(self, image: NDArray[np.uint8]) -> int:
        """Publish one RGB frame and return its sequence number.

        The copy is a single ``memcpy`` into the slot the sequence selects, and
        the sequence is published only afterwards, so a reader never sees a
        half-written frame. The call never blocks.

        Raises:
            ValueError: If ``image`` is not an ``(H, W, 3)`` uint8 frame of this
                channel's geometry.
        """

        frame = np.asarray(image)
        if frame.dtype != np.uint8 or frame.shape != (self._height, self._width, 3):
            raise ValueError(
                f"expected a ({self._height}, {self._width}, 3) uint8 frame, "
                f"got {frame.shape} {frame.dtype}"
            )
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        seq = self._last_seq + 1
        np.copyto(self._view(seq % 2), frame)
        _SEQUENCE.pack_into(self._buffer, 8, seq)
        self._last_seq = seq
        self._frames_written += 1
        return seq

    def read(self, last_seq: int = 0) -> tuple[int, NDArray[np.uint8]] | None:
        """Return the newest frame newer than ``last_seq``, or ``None``.

        The returned array is a fresh copy, so the caller may hold it while the
        writer publishes the next frames. A frame replaced twice while it is
        being copied is retried; if the writer keeps winning, ``None`` is
        returned and the caller simply reads the next one.

        Raises:
            ValueError: If ``last_seq`` is negative.
        """

        if last_seq < 0:
            raise ValueError(f"last_seq must not be negative, got {last_seq}")
        seq = _SEQUENCE.unpack_from(self._buffer, 8)[0]
        for _ in range(_READ_RETRIES):
            if seq <= last_seq:
                return None
            frame = np.empty((self._height, self._width, 3), dtype=np.uint8)
            np.copyto(frame, self._view(seq % 2))
            published = _SEQUENCE.unpack_from(self._buffer, 8)[0]
            if published == seq:
                self._frames_read += 1
                self._frames_dropped += seq - last_seq - 1
                self._last_seq = seq
                return seq, frame
            seq = published
        return None

    def _view(self, slot: int) -> NDArray[np.uint8]:
        """Mutable view of one frame buffer."""
        offset = _HEADER_SIZE + slot * self._frame_bytes
        view: NDArray[np.uint8] = np.ndarray(
            (self._height, self._width, 3),
            dtype=np.uint8,
            buffer=self._buffer,
            offset=offset,
        )
        return view

    def close(self) -> None:
        """Close this handle; the creator must also :meth:`unlink`."""
        with contextlib.suppress(FileNotFoundError):  # pragma: no cover - already unlinked
            self._shm.close()

    def unlink(self) -> None:
        """Remove the shared-memory segment (creator only, after both closed)."""
        if not self._owner:
            return
        self._owner = False
        with contextlib.suppress(FileNotFoundError):  # pragma: no cover - already unlinked
            self._shm.unlink()

    def __enter__(self) -> FrameChannel:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
        self.unlink()


class MessageChannel:
    """A bounded queue of ``(topic, payload)`` snapshots that drops the oldest.

    The pipeline must never wait for the server: if the server is not draining,
    the next state replaces the last one and the queue never grows.

    Args:
        messages: A ``multiprocessing``-style queue; ``None`` creates one of
            ``maxsize``. A test passes a fake with ``put_nowait``/``get_nowait``.
        maxsize: Capacity of the queue when this instance creates it.
    """

    def __init__(self, messages: Any | None = None, *, maxsize: int = 16) -> None:
        if messages is None:
            messages = Queue(maxsize=maxsize)
        self._queue = messages
        self._dropped = 0
        self._closed = False

    @property
    def queue(self) -> Any:
        """The underlying queue, to hand to the other process."""
        return self._queue

    @property
    def dropped(self) -> int:
        """Messages discarded because the consumer was behind."""
        return self._dropped

    @property
    def closed(self) -> bool:
        """True once :meth:`close` ran."""
        return self._closed

    def put(self, topic: str, data: dict[str, Any]) -> bool:
        """Offer one snapshot, discarding the oldest message when full.

        Returns:
            ``True`` when the message was queued, ``False`` when the queue was
            still full after dropping one (the message is then lost, and counted
            in :attr:`dropped`).
        """

        if self._closed:
            return False
        item = (str(topic), dict(data))
        if self._put(item):
            return True
        if self._discard_oldest() and self._put(item):
            return True
        self._dropped += 1
        return False

    def get_nowait(self) -> tuple[str, dict[str, Any]] | None:
        """Return the oldest queued message, or ``None`` when the queue is empty."""
        if self._closed:
            return None
        try:
            topic, data = self._queue.get_nowait()
        except queue.Empty:
            return None
        return str(topic), dict(data)

    def drain(self, limit: int = 64) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield up to ``limit`` queued messages, oldest first."""
        for _ in range(limit):
            message = self.get_nowait()
            if message is None:
                return
            yield message

    def close(self) -> None:
        """Stop accepting messages and release the queue."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self._queue, "close", None)
        if callable(close):
            close()

    def _put(self, item: tuple[str, dict[str, Any]]) -> bool:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            return False
        return True

    def _discard_oldest(self) -> bool:
        """Drop one queued message; ``False`` when there was nothing to drop."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            return False
        self._dropped += 1
        return True
