"""Schema-v1 session writer: one writer thread behind a bounded frame queue.

Frames are accepted only inside the active trial windows. A pre-roll ring keeps the
last ``pre_s`` seconds of frames in memory so the frames captured just before a cue
are written as well. IMU rows and trial records take a separate, unbounded queue so
the BLE callbacks never wait for disk I/O.

Layout of ``<root>/<session_id>/``::

    session.json
    calibration.json        (optional copy)
    frames.csv
    rgb/NNNNNN.jpg          JPEG quality 95, RGB order
    depth/NNNNNN.png        uint16 PNG, raw Z16
    imu.csv
    trials.jsonl
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import queue
import shutil
import socket
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import SampleKind, Side
from cpsa.ports import Clock, RgbdFrame

logger = logging.getLogger(__name__)

FRAMES_HEADER = ("seq", "trial_id", "ts_rgb", "ts_depth", "rgb", "depth")
IMU_HEADER = ("ts", "side", "kind", "v0", "v1", "v2", "v3")
_STATUSES = ("ok", "bad")
_JPEG_QUALITY = 95
_RING_MARGIN_S = 0.5


class RecordingError(RuntimeError):
    """A session cannot be written."""


@dataclass(frozen=True)
class XuValues:
    """Depth extension-unit values copied into ``session.json``."""

    laser_power: int
    accuracy: int
    filter: int
    confidence: int


@dataclass(frozen=True)
class CameraInfo:
    """The camera settings recorded in ``session.json``."""

    rgb_device: str
    depth_device: str
    width: int
    height: int
    fps: float
    rgb_exposure: int
    xu: XuValues


@dataclass(frozen=True)
class ImuInfo:
    """The BlueCoin settings recorded in ``session.json``.

    ``max_skew_ms`` and ``stale_ms`` are the synchronizer settings used during the
    recording; ``cpsa-replay-imu`` reuses them so replay behaves like the live run.
    """

    enabled: bool
    left: str | None
    right: str | None
    max_skew_ms: float | None = None
    stale_ms: float | None = None


@dataclass(frozen=True)
class TrialRecord:
    """One ``trials.jsonl`` line. ``class_name`` is serialized as ``class``."""

    trial_id: int
    class_name: str
    hand: str
    target: str
    cue_ts: float
    start_ts: float
    end_ts: float
    status: str
    note: str
    distance_cm: int

    def to_json(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "class": self.class_name,
            "hand": self.hand,
            "target": self.target,
            "cue_ts": self.cue_ts,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "status": self.status,
            "note": self.note,
            "distance_cm": self.distance_cm,
        }


@dataclass(frozen=True)
class SessionCounters:
    """Counters written into ``session.json`` at finalize time."""

    frames_written: int
    frames_dropped_queue: int
    pairs_dropped: int | None
    imu_samples: dict[str, dict[str, int]]
    imu_rate_hz: dict[str, dict[str, float]]
    trials: dict[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "frames_written": self.frames_written,
            "frames_dropped_queue": self.frames_dropped_queue,
            "pairs_dropped": self.pairs_dropped,
            "imu_samples": self.imu_samples,
            "imu_rate_hz": self.imu_rate_hz,
            "trials": self.trials,
        }


@dataclass(frozen=True)
class WriterSnapshot:
    """A consistent-enough counter snapshot for the periodic progress line."""

    bytes_written: int
    frames_written: int
    frames_dropped_queue: int
    imu_samples: int


@dataclass(frozen=True)
class _QueuedFrame:
    trial_id: int
    frame: RgbdFrame


@dataclass
class _Window:
    trial_id: int
    start_ts: float
    end_ts: float


@dataclass
class _ImuStats:
    count: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0

    def add(self, ts: float) -> None:
        if self.count == 0:
            self.first_ts = ts
            self.last_ts = ts
        else:
            self.first_ts = min(self.first_ts, ts)
            self.last_ts = max(self.last_ts, ts)
        self.count += 1

    def rate_hz(self) -> float:
        duration = self.last_ts - self.first_ts
        if self.count < 2 or duration <= 0:
            return 0.0
        return round((self.count - 1) / duration, 3)


class SessionWriter:
    """Write one schema-v1 session.

    ``write_frame`` and ``write_sample`` are safe to call from capture and BLE
    threads: they only enqueue and never wait for disk. ``start`` starts the writer
    thread; calling it is optional when nothing must be written before ``finalize``
    (``finalize`` starts the thread if needed).
    """

    def __init__(
        self,
        *,
        root: Path,
        session_id: str,
        participant_id: str,
        camera: CameraInfo,
        imu: ImuInfo,
        protocol_sha256: str,
        protocol_seed: int,
        clock: Clock,
        created: datetime | None = None,
        calibration_path: Path | None = None,
        notes: str = "",
        pre_s: float = 1.0,
        post_s: float = 1.0,
        queue_size: int = 90,
    ) -> None:
        if pre_s < 0 or post_s < 0:
            raise ValueError("pre_s and post_s must be >= 0")
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        if not session_id or "/" in session_id or "\\" in session_id:
            raise ValueError(f"invalid session_id {session_id!r}")

        self.path = root / session_id
        self.session_id = session_id
        self.participant_id = participant_id
        self._camera = camera
        self._imu = imu
        self._protocol_sha256 = protocol_sha256
        self._protocol_seed = protocol_seed
        self._notes = notes
        self._pre_s = float(pre_s)
        self._post_s = float(post_s)
        self._created = created if created is not None else clock.now()

        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "rgb").mkdir()
        (self.path / "depth").mkdir()

        self._calibration: dict[str, str] | None = None
        if calibration_path is not None:
            if not calibration_path.is_file():
                raise RecordingError(f"calibration file not found: {calibration_path}")
            destination = self.path / "calibration.json"
            shutil.copyfile(calibration_path, destination)
            self._calibration = {
                "file": destination.name,
                "sha256": _sha256(destination),
            }

        self._frames_file = (self.path / "frames.csv").open("w", encoding="utf-8", newline="")
        self._frames_csv = csv.writer(self._frames_file, lineterminator="\n")
        self._frames_csv.writerow(FRAMES_HEADER)
        self._imu_file = (self.path / "imu.csv").open("w", encoding="utf-8", newline="")
        self._imu_csv = csv.writer(self._imu_file, lineterminator="\n")
        self._imu_csv.writerow(IMU_HEADER)
        (self.path / "trials.jsonl").touch()

        self._frames: queue.Queue[_QueuedFrame | None] = queue.Queue(maxsize=queue_size)
        self._imu_queue: queue.SimpleQueue[tuple[str, str, tuple[float, ...], float]] = (
            queue.SimpleQueue()
        )
        self._trial_queue: queue.SimpleQueue[TrialRecord] = queue.SimpleQueue()

        self._lock = threading.Lock()
        self._ring: deque[tuple[float, RgbdFrame]] = deque()
        self._windows: list[_Window] = []
        self._dropped_queue = 0
        self._finalized = False

        self._stats_lock = threading.Lock()
        self._bytes_written = 0
        self._frames_written = 0
        self._imu_stats: dict[tuple[str, str], _ImuStats] = {}
        self._trial_status = {"ok": 0, "bad": 0}

        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._counters = SessionCounters(
            frames_written=0,
            frames_dropped_queue=0,
            pairs_dropped=None,
            imu_samples={},
            imu_rate_hz={},
            trials={"ok": 0, "bad": 0},
        )
        self._write_session_json()

    # -- capture-side API -------------------------------------------------

    def start(self) -> None:
        """Start the writer thread once."""
        with self._start_lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name="cpsa-session-writer", daemon=True
            )
            self._thread.start()

    def begin_trial(self, trial_id: int, start_ts: float, end_ts: float) -> None:
        """Open a recording window and flush the matching pre-roll frames.

        ``start_ts`` is normally ``cue_ts - pre_s`` and ``end_ts`` is
        ``trial_end_ts + post_s``.
        """
        if end_ts < start_ts:
            raise ValueError("trial window end must not precede its start")
        with self._lock:
            if self._finalized:
                return
            window = _Window(trial_id, float(start_ts), float(end_ts))
            self._windows.append(window)
            self._flush_ring_locked(window)

    def extend_trial(self, trial_id: int, end_ts: float) -> None:
        """Extend an open window, for example after the operator paused a trial."""
        with self._lock:
            if self._finalized:
                return
            for window in self._windows:
                if window.trial_id == trial_id:
                    if end_ts > window.end_ts:
                        window.end_ts = float(end_ts)
                        self._flush_ring_locked(window)
                    return

    def write_frame(self, frame: RgbdFrame) -> None:
        """Capture callback: enqueue a frame if it belongs to an open window."""
        ts = float(frame.ts_rgb)
        with self._lock:
            if self._finalized:
                return
            window = self._window_for_locked(ts)
            if window is not None:
                self._enqueue_locked(window.trial_id, frame)
                return
            self._ring.append((ts, frame))
            horizon = ts - (self._pre_s + _RING_MARGIN_S)
            while self._ring and self._ring[0][0] < horizon:
                self._ring.popleft()

    def write_sample(
        self, side: Side, kind: SampleKind, values: tuple[float, ...], ts: float
    ) -> None:
        """Sensor callback: enqueue one raw IMU sample without waiting."""
        expected = 4 if kind is SampleKind.QUAT else 3
        if len(values) != expected:
            logger.warning(
                "Ignoring %s/%s sample with %d values", side.value, kind.value, len(values)
            )
            return
        try:
            sample = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric %s/%s sample", side.value, kind.value)
            return
        self._imu_queue.put((side.value, kind.value, sample, float(ts)))

    def record_trial(self, record: TrialRecord) -> None:
        """Queue one trial record; it is written before finalize returns."""
        if record.status not in _STATUSES:
            raise ValueError(f"invalid trial status {record.status!r}")
        self._trial_queue.put(record)

    def snapshot(self) -> WriterSnapshot:
        """Return current progress counters for the periodic report."""
        with self._lock:
            dropped = self._dropped_queue
        with self._stats_lock:
            return WriterSnapshot(
                bytes_written=self._bytes_written,
                frames_written=self._frames_written,
                frames_dropped_queue=dropped,
                imu_samples=sum(stats.count for stats in self._imu_stats.values()),
            )

    # -- finalize ---------------------------------------------------------

    def finalize(self, pairs_dropped: int | None = None) -> SessionCounters:
        """Flush both queues, close the files and rewrite ``session.json`` atomically."""
        with self._lock:
            if self._finalized:
                return self._counters
            self._finalized = True

        self.start()
        self._frames.put(None)
        thread = self._thread
        if thread is not None:
            thread.join()

        self._frames_file.flush()
        self._frames_file.close()
        self._imu_file.flush()
        self._imu_file.close()

        with self._lock:
            dropped = self._dropped_queue
        imu_samples: dict[str, dict[str, int]] = {}
        imu_rates: dict[str, dict[str, float]] = {}
        for (side, kind), stats in self._imu_stats.items():
            imu_samples.setdefault(side, {})[kind] = stats.count
            imu_rates.setdefault(side, {})[kind] = stats.rate_hz()
        self._counters = SessionCounters(
            frames_written=self._frames_written,
            frames_dropped_queue=dropped,
            pairs_dropped=pairs_dropped,
            imu_samples=imu_samples,
            imu_rate_hz=imu_rates,
            trials=dict(self._trial_status),
        )
        self._write_session_json()
        return self._counters

    # -- writer thread ----------------------------------------------------

    def _run(self) -> None:
        while True:
            self._drain_imu()
            self._drain_trials()
            try:
                item = self._frames.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self._write_frame(item)
            except Exception:
                logger.exception("Failed to write frame seq=%d", item.frame.seq)
        self._drain_imu()
        self._drain_trials()

    def _drain_imu(self) -> None:
        while True:
            try:
                side, kind, values, ts = self._imu_queue.get_nowait()
            except queue.Empty:
                return
            padded: list[object] = [float(ts), side, kind, *values]
            padded.extend([""] * (len(IMU_HEADER) - len(padded)))
            self._imu_csv.writerow(padded)
            with self._stats_lock:
                stats = self._imu_stats.setdefault((side, kind), _ImuStats())
                stats.add(float(ts))

    def _drain_trials(self) -> None:
        while True:
            try:
                record = self._trial_queue.get_nowait()
            except queue.Empty:
                return
            with (self.path / "trials.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record.to_json()) + "\n")
            self._trial_status[record.status] = self._trial_status.get(record.status, 0) + 1

    def _write_frame(self, item: _QueuedFrame) -> None:
        frame = item.frame
        seq = int(frame.seq)
        if seq < 0:
            raise ValueError(f"invalid frame seq {seq}")
        stem = f"{seq:06d}"
        rgb_rel = f"rgb/{stem}.jpg"
        depth_rel = f"depth/{stem}.png"
        rgb_path = self.path / rgb_rel
        depth_path = self.path / depth_rel

        _encode_jpeg(frame.rgb, rgb_path)
        _encode_png(frame.depth, depth_path)

        self._frames_csv.writerow(
            [seq, item.trial_id, float(frame.ts_rgb), float(frame.ts_depth), rgb_rel, depth_rel]
        )
        written = rgb_path.stat().st_size + depth_path.stat().st_size
        with self._stats_lock:
            self._bytes_written += written
            self._frames_written += 1

    # -- state helpers ----------------------------------------------------

    def _window_for_locked(self, ts: float) -> _Window | None:
        for window in self._windows:
            if window.start_ts <= ts <= window.end_ts:
                return window
        return None

    def _flush_ring_locked(self, window: _Window) -> None:
        keep: deque[tuple[float, RgbdFrame]] = deque()
        for ts, frame in self._ring:
            if window.start_ts <= ts <= window.end_ts:
                self._enqueue_locked(window.trial_id, frame)
            elif ts > window.end_ts:
                keep.append((ts, frame))
        self._ring = keep

    def _enqueue_locked(self, trial_id: int, frame: RgbdFrame) -> None:
        try:
            self._frames.put_nowait(_QueuedFrame(trial_id, frame))
        except queue.Full:
            self._dropped_queue += 1

    def _session_document(self) -> dict[str, object]:
        imu: dict[str, object] = {
            "enabled": self._imu.enabled,
            "left": self._imu.left,
            "right": self._imu.right,
        }
        if self._imu.max_skew_ms is not None:
            imu["max_skew_ms"] = self._imu.max_skew_ms
        if self._imu.stale_ms is not None:
            imu["stale_ms"] = self._imu.stale_ms
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "created": self._created.isoformat(),
            "participant_id": self.participant_id,
            "camera": asdict(self._camera),
            "calibration": self._calibration,
            "imu": imu,
            "protocol": {
                "sha256": self._protocol_sha256,
                "seed": self._protocol_seed,
            },
            "counters": self._counters.to_json(),
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
            },
            "notes": self._notes,
        }

    def _write_session_json(self) -> None:
        document = self._session_document()
        temporary = self.path / "session.json.tmp"
        temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path / "session.json")


def _encode_jpeg(rgb: NDArray[np.uint8], path: Path) -> None:
    from PIL import Image

    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected uint8 RGB image, got {array.dtype} {array.shape}")
    Image.fromarray(np.ascontiguousarray(array)).save(path, format="JPEG", quality=_JPEG_QUALITY)


def _encode_png(depth: NDArray[np.uint16], path: Path) -> None:
    from PIL import Image

    array = np.asarray(depth)
    if array.dtype != np.uint16 or array.ndim != 2:
        raise ValueError(f"expected uint16 depth image, got {array.dtype} {array.shape}")
    Image.fromarray(np.ascontiguousarray(array)).save(path, format="PNG")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
