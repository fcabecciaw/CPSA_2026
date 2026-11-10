"""Thread-safe pairing of RGB and depth frames by their capture timestamps.

Each capture thread pushes raw frames into one :class:`Pairer`. Frames are kept
in short timestamp-ordered per-stream buffers so a stream may arrive later than
the other without losing pairs. A frame is emitted only when the other buffer
holds a partner within ``1 / (2 * fps)``; frames that can no longer match are
dropped and counted.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

Image = bytes


@dataclass(frozen=True)
class PairedFrame:
    """A matched RGB/depth pair with the timestamps of both captures."""

    seq: int
    rgb: Image
    depth: Image
    ts_rgb: float
    ts_depth: float


@dataclass(frozen=True)
class PairingStats:
    """Counters of one :class:`Pairer`, taken as a consistent snapshot."""

    paired: int
    dropped_rgb: int
    dropped_depth: int


@dataclass(frozen=True)
class _Pending:
    ts: float
    image: Image


class Pairer:
    """Pair RGB and depth frames arriving from two threads.

    Each stream keeps a timestamp-ordered buffer bounded by ``horizon_s`` and
    ``max_pending`` frames. A new frame is matched with the nearest timestamp in
    the other buffer; unmatched frames stay buffered until a later partner
    arrives. A frame is dropped only when its timestamp is older than the newest
    timestamp seen on the other stream minus the tolerance (it can no longer
    match any future frame) or when a buffer bound evicts it.

    Timestamps must be monotonic per stream, and frames are pushed in capture
    order, so emitted pairs are in timestamp order. Pushes and counter reads are
    serialized by an internal lock.
    """

    def __init__(
        self,
        fps: float,
        *,
        horizon_s: float = 0.5,
        max_pending: int = 32,
    ) -> None:
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
        if isinstance(horizon_s, bool) or not math.isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError("horizon_s must be a finite positive number")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        self._tolerance = 0.5 / fps
        self._match_tolerance = self._tolerance + 1e-9
        self._horizon = horizon_s
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._rgb: list[_Pending] = []
        self._depth: list[_Pending] = []
        self._newest_rgb: float | None = None
        self._newest_depth: float | None = None
        self._paired = 0
        self._dropped_rgb = 0
        self._dropped_depth = 0

    @property
    def tolerance(self) -> float:
        """Maximum allowed RGB/depth timestamp difference in seconds."""
        return self._tolerance

    @property
    def pending(self) -> tuple[int, int]:
        """Number of buffered (RGB, depth) frames awaiting a partner."""
        with self._lock:
            return len(self._rgb), len(self._depth)

    def push_rgb(self, ts: float, image: Image) -> PairedFrame | None:
        """Add an RGB frame and return the emitted pair, if any."""
        return self._push(ts, image, depth=False)

    def push_depth(self, ts: float, image: Image) -> PairedFrame | None:
        """Add a depth frame and return the emitted pair, if any."""
        return self._push(ts, image, depth=True)

    def stats(self) -> PairingStats:
        """Return the current pair and drop counters."""
        with self._lock:
            return PairingStats(self._paired, self._dropped_rgb, self._dropped_depth)

    def flush(self) -> None:
        """Count frames still waiting for a partner as dropped."""
        with self._lock:
            self._dropped_rgb += len(self._rgb)
            self._dropped_depth += len(self._depth)
            self._rgb.clear()
            self._depth.clear()

    def _push(self, ts: float, image: Image, *, depth: bool) -> PairedFrame | None:
        if not math.isfinite(ts):
            raise ValueError("timestamp must be finite")
        with self._lock:
            current = _Pending(ts, image)
            if depth:
                self._newest_depth = self._newer(self._newest_depth, ts)
                other_index = self._nearest(self._rgb, ts)
                if other_index is None:
                    self._insert(self._depth, current)
                    pair = None
                else:
                    other = self._rgb.pop(other_index)
                    pair = self._emit(other, current)
            else:
                self._newest_rgb = self._newer(self._newest_rgb, ts)
                other_index = self._nearest(self._depth, ts)
                if other_index is None:
                    self._insert(self._rgb, current)
                    pair = None
                else:
                    other = self._depth.pop(other_index)
                    pair = self._emit(current, other)
            self._drop_unmatchable()
            self._enforce_bounds()
            return pair

    @staticmethod
    def _newer(current: float | None, candidate: float) -> float:
        return candidate if current is None else max(current, candidate)

    @staticmethod
    def _insert(buffer: list[_Pending], item: _Pending) -> None:
        index = len(buffer)
        while index > 0 and buffer[index - 1].ts > item.ts:
            index -= 1
        buffer.insert(index, item)

    def _nearest(self, buffer: list[_Pending], ts: float) -> int | None:
        best_index: int | None = None
        best_difference = self._match_tolerance
        for index, pending in enumerate(buffer):
            if pending.ts > ts + self._match_tolerance:
                break
            difference = abs(pending.ts - ts)
            if difference < best_difference or (
                difference == best_difference and best_index is None
            ):
                best_index = index
                best_difference = difference
        return best_index

    def _drop_unmatchable(self) -> None:
        if self._newest_depth is not None:
            self._dropped_rgb += self._pop_older_than(
                self._rgb, self._newest_depth - self._match_tolerance
            )
        if self._newest_rgb is not None:
            self._dropped_depth += self._pop_older_than(
                self._depth, self._newest_rgb - self._match_tolerance
            )

    def _enforce_bounds(self) -> None:
        self._dropped_rgb += self._trim(self._rgb, self._newest_rgb)
        self._dropped_depth += self._trim(self._depth, self._newest_depth)

    def _trim(self, buffer: list[_Pending], newest: float | None) -> int:
        dropped = 0
        if newest is not None:
            dropped += self._pop_older_than(buffer, newest - self._horizon)
        while len(buffer) > self._max_pending:
            buffer.pop(0)
            dropped += 1
        return dropped

    @staticmethod
    def _pop_older_than(buffer: list[_Pending], cutoff: float) -> int:
        dropped = 0
        while buffer and buffer[0].ts < cutoff:
            buffer.pop(0)
            dropped += 1
        return dropped

    def _emit(self, rgb: _Pending, depth: _Pending) -> PairedFrame:
        frame = PairedFrame(
            seq=self._paired,
            rgb=rgb.image,
            depth=depth.image,
            ts_rgb=rgb.ts,
            ts_depth=depth.ts,
        )
        self._paired += 1
        return frame
