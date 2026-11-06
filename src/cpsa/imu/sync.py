"""Left/right wrist IMU stream synchronization."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import SampleKind, Side

logger = logging.getLogger(__name__)

RowCallback = Callable[[NDArray[np.float32], float], None]

_SAMPLE_LENGTH = {SampleKind.ACC: 3, SampleKind.GYR: 3, SampleKind.QUAT: 4}


class MonotonicClock(Protocol):
    """The part of ``cpsa.ports.Clock`` the synchronizer needs."""

    def monotonic(self) -> float: ...


@dataclass
class _Triplet:
    """Latest acc/gyr/quat sample of one wrist, each with its arrival time."""

    acc: tuple[float, ...] | None = None
    gyr: tuple[float, ...] | None = None
    quat: tuple[float, ...] | None = None
    ts_acc: float = 0.0
    ts_gyr: float = 0.0
    ts_quat: float = 0.0

    def ready(self) -> bool:
        return self.acc is not None and self.gyr is not None and self.quat is not None

    def newest_ts(self) -> float:
        return max(self.ts_acc, self.ts_gyr, self.ts_quat)

    def clear(self) -> None:
        self.acc = self.gyr = self.quat = None
        self.ts_acc = self.ts_gyr = self.ts_quat = 0.0

    def values(self) -> tuple[float, ...]:
        """Return acc, gyr and quat concatenated; only valid when ``ready()``."""
        if self.acc is None or self.gyr is None or self.quat is None:
            raise RuntimeError("triplet is incomplete")
        return self.acc + self.gyr + self.quat

    def set(self, kind: SampleKind, values: tuple[float, ...], ts: float) -> None:
        if kind is SampleKind.ACC:
            self.acc, self.ts_acc = values, ts
        elif kind is SampleKind.GYR:
            self.gyr, self.ts_gyr = values, ts
        else:
            self.quat, self.ts_quat = values, ts


class IMUSynchronizer:
    """Combine left and right wrist triplets into raw 20-value rows.

    A row is emitted when both wrists hold a complete (acc, gyr, quat) triplet whose
    newest timestamps differ by at most ``max_skew_s``; both triplets are then cleared.
    Otherwise the older triplet is dropped so the streams resynchronize quickly. A
    complete triplet whose newest sample is older than ``stale_s`` is discarded.

    Row layout (raw units: mg, dps, quaternion x10000): R acc xyz, R gyr xyz,
    R quat xyzw, L acc xyz, L gyr xyz, L quat xyzw. The row timestamp is the newer
    of the two wrists' newest timestamps.

    ``update`` is thread-safe (called from one BLE thread per wrist); ``on_row`` is
    invoked outside the internal lock.
    """

    def __init__(
        self,
        max_skew_s: float,
        stale_s: float,
        on_row: RowCallback,
        clock: MonotonicClock,
    ) -> None:
        self._max_skew_s = max_skew_s
        self._stale_s = stale_s
        self._on_row = on_row
        self._clock = clock
        self._lock = threading.Lock()
        self._left = _Triplet()
        self._right = _Triplet()

    def update(self, side: Side, kind: SampleKind, values: Sequence[float], ts: float) -> None:
        try:
            sample = tuple(float(values[i]) for i in range(_SAMPLE_LENGTH[kind]))
        except (TypeError, ValueError, IndexError) as exc:
            logger.warning("Bad %s sample for %s wrist: %s", kind.value, side.value, exc)
            return

        with self._lock:
            (self._left if side is Side.LEFT else self._right).set(kind, sample, ts)
            row = self._try_combine_locked()

        if row is not None:
            self._on_row(*row)

    def _try_combine_locked(self) -> tuple[NDArray[np.float32], float] | None:
        left, right = self._left, self._right

        now = self._clock.monotonic()
        if left.ready() and now - left.newest_ts() > self._stale_s:
            left.clear()
        if right.ready() and now - right.newest_ts() > self._stale_s:
            right.clear()

        if not (left.ready() and right.ready()):
            return None

        t_left, t_right = left.newest_ts(), right.newest_ts()
        if abs(t_left - t_right) <= self._max_skew_s:
            row = np.asarray(right.values() + left.values(), dtype=np.float32)
            left.clear()
            right.clear()
            return row, max(t_left, t_right)

        if t_left + self._max_skew_s < t_right:
            left.clear()
        elif t_right + self._max_skew_s < t_left:
            right.clear()
        return None
