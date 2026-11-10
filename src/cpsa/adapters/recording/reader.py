"""Schema-v1 session reader used by dataset building and IMU replay.

``Session.open(path)`` validates the schema version, the mandatory files and the
column/JSON keys, then exposes the trial list, a numpy frame index and the raw IMU
rows. Frames are decoded lazily so opening a large session stays cheap.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from cpsa.adapters.recording.writer import FRAMES_HEADER, IMU_HEADER, TrialRecord

SCHEMA_VERSION = 1
REQUIRED_FILES = ("session.json", "frames.csv", "imu.csv", "trials.jsonl")
IMU_DTYPE = np.dtype(
    [
        ("ts", "<f8"),
        ("side", "U5"),
        ("kind", "U4"),
        ("v0", "<f8"),
        ("v1", "<f8"),
        ("v2", "<f8"),
        ("v3", "<f8"),
    ]
)
_TRIAL_KEYS = (
    "trial_id",
    "class",
    "hand",
    "target",
    "cue_ts",
    "start_ts",
    "end_ts",
    "status",
    "note",
    "distance_cm",
)


class SessionFormatError(ValueError):
    """A session is missing a file, key or column, or has an invalid value."""


@dataclass(frozen=True)
class FrameIndex:
    """Column arrays of ``frames.csv`` plus the paths of the image files."""

    seq: NDArray[np.int64]
    trial_id: NDArray[np.int64]
    ts_rgb: NDArray[np.float64]
    ts_depth: NDArray[np.float64]
    rgb_paths: tuple[Path, ...]
    depth_paths: tuple[Path, ...]

    def __len__(self) -> int:
        return int(self.seq.size)


@dataclass(frozen=True)
class FrameData:
    """One lazily decoded RGB/depth pair."""

    rgb: NDArray[np.uint8]
    depth: NDArray[np.uint16]


class Session:
    """A validated schema-v1 session directory."""

    def __init__(
        self,
        *,
        path: Path,
        meta: Mapping[str, object],
        trials: tuple[TrialRecord, ...],
        frames: FrameIndex,
        imu: NDArray[np.void],
        calibration_path: Path | None,
    ) -> None:
        self.path = path
        self.meta = dict(meta)
        self.trials = trials
        self.frames = frames
        self.imu = imu
        self.calibration_path = calibration_path

        self.schema_version = 1
        self.session_id = str(meta["session_id"])
        self.participant_id = str(meta["participant_id"])
        self.created = _parse_created(meta["created"])
        imu_meta = _mapping(meta.get("imu"), "session.json:imu")
        self.imu_enabled = _boolean(imu_meta.get("enabled", False), "session.json:imu.enabled")
        self.imu_left = _optional_text(imu_meta.get("left"), "session.json:imu.left")
        self.imu_right = _optional_text(imu_meta.get("right"), "session.json:imu.right")
        self.sync_max_skew_s = (
            _number(imu_meta.get("max_skew_ms", 30.0), "session.json:imu.max_skew_ms") / 1000.0
        )
        self.sync_stale_s = (
            _number(imu_meta.get("stale_ms", 50.0), "session.json:imu.stale_ms") / 1000.0
        )

    @classmethod
    def open(cls, path: Path | str) -> Session:
        """Read and validate the session rooted at ``path``."""
        root = Path(path)
        if not root.is_dir():
            raise SessionFormatError(f"{root}: not a session directory")
        for name in REQUIRED_FILES:
            if not (root / name).is_file():
                raise SessionFormatError(f"{name}: missing required file in {root}")

        meta = _read_json(root / "session.json")
        version = meta.get("schema_version")
        if version != SCHEMA_VERSION:
            raise SessionFormatError(
                f"session.json: schema_version must be {SCHEMA_VERSION}, got {version!r}"
            )
        for key in ("session_id", "participant_id", "created", "imu"):
            if key not in meta:
                raise SessionFormatError(f"session.json: missing key '{key}'")

        calibration_path = _calibration_path(root, meta.get("calibration"))
        frames = _read_frames(root)
        trials = _read_trials(root / "trials.jsonl")
        imu = _read_imu(root / "imu.csv")
        return cls(
            path=root,
            meta=meta,
            trials=trials,
            frames=frames,
            imu=imu,
            calibration_path=calibration_path,
        )

    def rgb(self, index: int) -> NDArray[np.uint8]:
        """Decode and return frame ``index`` as an ``(H, W, 3)`` RGB uint8 array."""
        path = self.frames.rgb_paths[index]
        try:
            from PIL import Image

            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError) as exc:
            relative = _relative(self.path, path)
            raise SessionFormatError(f"{relative}: cannot decode JPEG: {exc}") from exc

    def depth(self, index: int) -> NDArray[np.uint16]:
        """Decode and return frame ``index`` as an ``(H, W)`` raw Z16 array."""
        path = self.frames.depth_paths[index]
        try:
            from PIL import Image

            with Image.open(path) as image:
                array = np.asarray(image)
        except (OSError, ValueError) as exc:
            relative = _relative(self.path, path)
            raise SessionFormatError(f"{relative}: cannot decode PNG: {exc}") from exc
        if array.dtype != np.uint16:
            array = array.astype(np.uint16)
        return cast(NDArray[np.uint16], array)

    def load_frame(self, index: int) -> FrameData:
        """Decode both images of frame ``index``."""
        return FrameData(rgb=self.rgb(index), depth=self.depth(index))


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SessionFormatError(f"{path.name}: cannot read file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SessionFormatError(
            f"{path.name}: invalid JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise SessionFormatError(f"{path.name}: expected a JSON object")
    return data


def _read_frames(root: Path) -> FrameIndex:
    path = root / "frames.csv"
    seq: list[int] = []
    trial_id: list[int] = []
    ts_rgb: list[float] = []
    ts_depth: list[float] = []
    rgb_paths: list[Path] = []
    depth_paths: list[Path] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise SessionFormatError("frames.csv: missing header")
        if tuple(header) != FRAMES_HEADER:
            raise SessionFormatError(
                f"frames.csv: expected columns {','.join(FRAMES_HEADER)}, got {','.join(header)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(FRAMES_HEADER):
                raise SessionFormatError(
                    f"frames.csv: line {line_number}: expected {len(FRAMES_HEADER)} columns, "
                    f"got {len(row)}"
                )
            try:
                seq.append(int(row[0]))
                trial_id.append(int(row[1]))
                ts_rgb.append(float(row[2]))
                ts_depth.append(float(row[3]))
            except ValueError as exc:
                raise SessionFormatError(
                    f"frames.csv: line {line_number}: invalid number: {exc}"
                ) from exc
            rgb_paths.append(_session_path(root, row[4], "frames.csv:rgb"))
            depth_paths.append(_session_path(root, row[5], "frames.csv:depth"))
    return FrameIndex(
        seq=np.asarray(seq, dtype=np.int64),
        trial_id=np.asarray(trial_id, dtype=np.int64),
        ts_rgb=np.asarray(ts_rgb, dtype=np.float64),
        ts_depth=np.asarray(ts_depth, dtype=np.float64),
        rgb_paths=tuple(rgb_paths),
        depth_paths=tuple(depth_paths),
    )


def _read_trials(path: Path) -> tuple[TrialRecord, ...]:
    trials: list[TrialRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionFormatError(
                    f"trials.jsonl: line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise SessionFormatError(f"trials.jsonl: line {line_number}: expected an object")
            for key in _TRIAL_KEYS:
                if key not in row:
                    raise SessionFormatError(
                        f"trials.jsonl: line {line_number}: missing key '{key}'"
                    )
            status = str(row["status"])
            if status not in ("ok", "bad"):
                raise SessionFormatError(
                    f"trials.jsonl: line {line_number}: status must be 'ok' or 'bad', "
                    f"got {status!r}"
                )
            trials.append(
                TrialRecord(
                    trial_id=_integer(
                        row["trial_id"], f"trials.jsonl: line {line_number}:trial_id"
                    ),
                    class_name=str(row["class"]),
                    hand=str(row["hand"]),
                    target=str(row["target"]),
                    cue_ts=_number(row["cue_ts"], f"trials.jsonl: line {line_number}:cue_ts"),
                    start_ts=_number(row["start_ts"], f"trials.jsonl: line {line_number}:start_ts"),
                    end_ts=_number(row["end_ts"], f"trials.jsonl: line {line_number}:end_ts"),
                    status=status,
                    note=str(row["note"]),
                    distance_cm=_integer(
                        row["distance_cm"], f"trials.jsonl: line {line_number}:distance_cm"
                    ),
                )
            )
    return tuple(trials)


def _read_imu(path: Path) -> NDArray[np.void]:
    records: list[tuple[float, str, str, float, float, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise SessionFormatError("imu.csv: missing header")
        if tuple(header) != IMU_HEADER:
            raise SessionFormatError(
                f"imu.csv: expected columns {','.join(IMU_HEADER)}, got {','.join(header)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(IMU_HEADER):
                raise SessionFormatError(
                    f"imu.csv: line {line_number}: expected {len(IMU_HEADER)} columns, "
                    f"got {len(row)}"
                )
            ts = _csv_number(row[0], f"imu.csv: line {line_number}:ts")
            side = row[1].strip()
            kind = row[2].strip()
            if side not in ("left", "right"):
                raise SessionFormatError(
                    f"imu.csv: line {line_number}: side must be left or right, got {side!r}"
                )
            if kind not in ("acc", "gyr", "quat"):
                raise SessionFormatError(
                    f"imu.csv: line {line_number}: kind must be acc, gyr or quat, got {kind!r}"
                )
            values = [_optional_number(value, f"imu.csv: line {line_number}") for value in row[3:7]]
            if kind == "quat" and any(value is None for value in values):
                raise SessionFormatError(
                    f"imu.csv: line {line_number}: quaternion needs four values"
                )
            if kind != "quat" and values[3] is not None:
                raise SessionFormatError(
                    f"imu.csv: line {line_number}: {kind} has only three values"
                )
            filled = [float("nan") if value is None else value for value in values]
            records.append((ts, side, kind, filled[0], filled[1], filled[2], filled[3]))
    return np.asarray(records, dtype=IMU_DTYPE)


def _calibration_path(root: Path, value: object) -> Path | None:
    if value is None:
        if (root / "calibration.json").exists():
            raise SessionFormatError(
                "session.json: calibration is null but calibration.json exists"
            )
        return None
    if not isinstance(value, Mapping):
        raise SessionFormatError("session.json:calibration: expected an object or null")
    file_value = value.get("file")
    if not isinstance(file_value, str) or not file_value:
        raise SessionFormatError("session.json:calibration: missing key 'file'")
    path = _session_path(root, file_value, "session.json:calibration.file")
    if not path.is_file():
        raise SessionFormatError(f"{path.name}: missing calibration file")
    return path


def _session_path(root: Path, value: str, where: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SessionFormatError(f"{where}: path must stay inside the session, got {value!r}")
    return root / relative


def _mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SessionFormatError(f"{where}: expected an object")
    return value


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionFormatError(f"{where}: expected a number")
    return float(value)


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionFormatError(f"{where}: expected an integer")
    return value


def _csv_number(value: str, where: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise SessionFormatError(f"{where}: invalid number {value!r}") from exc


def _optional_number(value: str, where: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise SessionFormatError(f"{where}: invalid number {text!r}") from exc


def _boolean(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise SessionFormatError(f"{where}: expected true or false")
    return value


def _optional_text(value: object, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SessionFormatError(f"{where}: expected a string or null")
    return value


def _parse_created(value: object) -> datetime:
    if not isinstance(value, str):
        raise SessionFormatError("session.json:created: expected an ISO timestamp string")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise SessionFormatError(f"session.json:created: invalid ISO timestamp: {exc}") from exc


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
