"""The frame channel and the message channel: ordering, drops and no tearing."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import pytest

from cpsa.channel import FrameChannel, MessageChannel


def test_the_frame_channel_publishes_frames_in_order() -> None:
    with FrameChannel.create(4, 2) as channel:
        writer = FrameChannel.attach(channel.name, 4, 2)
        try:
            first = np.full((2, 4, 3), 7, dtype=np.uint8)
            second = np.full((2, 4, 3), 9, dtype=np.uint8)

            assert channel.write(first) == 1
            seq, frame = writer.read(0)  # type: ignore[misc]
            assert seq == 1
            assert np.array_equal(frame, first)

            assert channel.write(second) == 2
            seq, frame = writer.read(1)  # type: ignore[misc]
            assert seq == 2
            assert np.array_equal(frame, second)
        finally:
            writer.close()


def test_a_reader_that_lags_reads_the_newest_frame_and_counts_the_drops() -> None:
    with FrameChannel.create(4, 2) as channel:
        for value in (1, 2, 3, 4, 5):
            channel.write(np.full((2, 4, 3), value, dtype=np.uint8))

        seq, frame = channel.read(channel.last_seq - 5)  # type: ignore[misc]

        assert seq == 5
        assert np.all(frame == 5)
        assert channel.frames_dropped == 4
        assert channel.frames_written == 5


def test_reading_a_frame_twice_returns_nothing_new() -> None:
    with FrameChannel.create(4, 2) as channel:
        channel.write(np.zeros((2, 4, 3), dtype=np.uint8))

        seq, _ = channel.read(0)  # type: ignore[misc]
        assert seq == 1
        assert channel.read(seq) is None


def test_the_writer_rejects_a_frame_of_the_wrong_shape_or_dtype() -> None:
    with FrameChannel.create(4, 2) as channel:
        with pytest.raises(ValueError):
            channel.write(np.zeros((4, 2, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            channel.write(np.zeros((2, 4, 3), dtype=np.float32))


def test_a_reader_never_sees_a_torn_frame() -> None:
    """The writer alternates two solid colours; every read must be one colour."""
    width, height = 64, 48
    with FrameChannel.create(width, height) as channel:
        stop = threading.Event()
        errors: list[str] = []

        def write_forever() -> None:
            value = 0
            while not stop.is_set():
                channel.write(np.full((height, width, 3), value, dtype=np.uint8))
                value = (value + 1) % 256

        thread = threading.Thread(target=write_forever, daemon=True)
        thread.start()
        try:
            seen = 0
            last = 0
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline and seen < 200:
                result = channel.read(last)
                if result is None:
                    continue
                seq, frame = result
                last = seq
                seen += 1
                if not (np.all(frame == frame[0, 0]) and frame[0, 0, 0] == frame[-1, -1, 0]):
                    errors.append(f"torn frame at seq {seq}")
                    break
        finally:
            stop.set()
            thread.join()
        assert errors == []
        assert seen > 0


def test_attaching_to_foreign_memory_is_rejected() -> None:
    from multiprocessing import shared_memory

    foreign = shared_memory.SharedMemory(create=True, size=4096)
    try:
        with pytest.raises(ValueError, match="not a frame channel"):
            FrameChannel.attach(foreign.name, 2, 2)
    finally:
        foreign.close()
        foreign.unlink()


def test_the_geometry_must_be_positive() -> None:
    with pytest.raises(ValueError):
        FrameChannel.create(0, 2)


class FakeQueue:
    """A queue with the ``multiprocessing`` surface, for the drop policy."""

    def __init__(self, maxsize: int) -> None:
        self._items: deque[tuple[str, dict[str, Any]]] = deque()
        self._maxsize = maxsize
        self.closed = False

    def put_nowait(self, item: tuple[str, dict[str, Any]]) -> None:
        if len(self._items) >= self._maxsize:
            raise queue.Full
        self._items.append(item)

    def get_nowait(self) -> tuple[str, dict[str, Any]]:
        if not self._items:
            raise queue.Empty
        return self._items.popleft()

    def close(self) -> None:
        self.closed = True


def test_the_message_channel_drops_the_oldest_when_full() -> None:
    queue_ = FakeQueue(maxsize=2)
    channel = MessageChannel(queue_)

    for index in range(5):
        assert channel.put("status", {"n": index}) is True

    assert channel.dropped == 3
    assert list(channel.drain()) == [("status", {"n": 3}), ("status", {"n": 4})]


def test_a_message_that_cannot_be_queued_is_counted() -> None:
    queue_ = FakeQueue(maxsize=0)
    channel = MessageChannel(queue_)

    assert channel.put("status", {"n": 1}) is False
    assert channel.dropped == 1
    assert channel.get_nowait() is None


def test_the_message_channel_copies_the_payload() -> None:
    channel = MessageChannel(FakeQueue(maxsize=2))
    payload = {"state": "running"}

    channel.put("status", payload)
    payload["state"] = "error"

    topic, data = channel.get_nowait()  # type: ignore[misc]
    assert topic == "status"
    assert data == {"state": "running"}


def test_a_closed_channel_takes_nothing_and_returns_nothing() -> None:
    queue_ = FakeQueue(maxsize=2)
    channel = MessageChannel(queue_)
    channel.put("status", {"n": 1})

    channel.close()

    assert channel.closed is True
    assert queue_.closed is True
    assert channel.put("status", {"n": 2}) is False
    assert channel.get_nowait() is None


def test_two_handles_of_one_queue_see_the_same_messages() -> None:
    queue_ = FakeQueue(maxsize=4)
    writer = MessageChannel(queue_)
    reader = MessageChannel(queue_)

    writer.put("showcase.state", {"objects": []})

    assert reader.get_nowait() == ("showcase.state", {"objects": []})
