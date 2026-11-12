"""Live calibration of the showcase: YOLOX zoom and the camera image (debug panel).

The operator tunes five values from the dashboard's debug panel while the demo
runs: the YOLOX centre crop (``zoom``, ``zoom_center``) and three camera
controls (``gain``, ``exposure``, ``saturation``). ``config/showcase.yaml``
holds the defaults; the accepted values are written to
``~/cpsa/config/calibration.json`` and loaded at the next start as overrides on
top of it, so a calibration survives a restart until ``calibration.reset``
removes the file.

Everything here is pure except :class:`CalibrationStore` (one small JSON file)
and :class:`LatestWorker` (one thread that applies the newest value off the
frame loop). :func:`parse_update` is the one validator: the server uses it to
answer ``error bad_command``, the pipeline to re-check what it drains and the
store to read the file back.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generic, TypeVar

from cpsa.config import GAIN_RANGE, SATURATION_RANGE, ZOOM_RANGE, ShowcaseConfig

logger = logging.getLogger("cpsa.calibration")

__all__ = [
    "CALIBRATION_KEYS",
    "CAMERA_KEYS",
    "DEFAULT_CALIBRATION_PATH",
    "EXPOSURE_CHOICES",
    "Calibration",
    "CalibrationStore",
    "LatestWorker",
    "parse_update",
]

DEFAULT_CALIBRATION_PATH = Path("~/cpsa/config/calibration.json")
"""Where the accepted values live on the board (the repository is ``~/cpsa``)."""

EXPOSURE_CHOICES = (100, 200)
"""``exposure_time_absolute`` the panel offers: 1/100 s and 1/50 s, flicker-safe at 50 Hz."""

CAMERA_KEYS = ("gain", "exposure", "saturation")
"""The calibration values that are V4L2 controls of the open camera stream."""

CALIBRATION_KEYS = ("zoom", "zoom_center", *CAMERA_KEYS)
"""Every value ``calibration.set`` accepts, in the order ``status`` lists them."""


@dataclass(frozen=True)
class Calibration:
    """The five values the debug panel tunes, as the pipeline applies them."""

    zoom: float
    zoom_center: tuple[float, float]
    gain: int
    exposure: int
    saturation: int

    @classmethod
    def from_config(cls, config: ShowcaseConfig) -> Calibration:
        """The values ``config/showcase.yaml`` sets: what ``calibration.reset`` restores."""
        return cls(
            zoom=config.model.zoom,
            zoom_center=config.model.zoom_center,
            gain=config.camera.gain,
            exposure=config.camera.exposure,
            saturation=config.camera.saturation,
        )

    def updated(self, update: Mapping[str, Any]) -> Calibration:
        """Return a copy with the keys of a :func:`parse_update` result replaced."""
        return replace(self, **dict(update))

    def applied_to(self, config: ShowcaseConfig) -> ShowcaseConfig:
        """Return ``config`` with these values in its ``model`` and ``camera`` sections."""
        return replace(
            config,
            model=replace(config.model, zoom=self.zoom, zoom_center=self.zoom_center),
            camera=replace(
                config.camera,
                gain=self.gain,
                exposure=self.exposure,
                saturation=self.saturation,
            ),
        )

    def camera_controls(self) -> dict[str, int]:
        """The V4L2 part: ``{"gain", "exposure", "saturation"}``."""
        return {"gain": self.gain, "exposure": self.exposure, "saturation": self.saturation}

    def to_json(self) -> dict[str, Any]:
        """The ``status.calibration`` object and the content of the file."""
        return {
            "zoom": round(self.zoom, 4),
            "zoom_center": [round(self.zoom_center[0], 4), round(self.zoom_center[1], 4)],
            "gain": self.gain,
            "exposure": self.exposure,
            "saturation": self.saturation,
        }


def parse_update(data: object) -> dict[str, Any]:
    """Validate a ``calibration.set`` payload and return its normalized values.

    Any non-empty subset of :data:`CALIBRATION_KEYS` is accepted: ``zoom`` a
    number in ``1.0..2.0``, ``zoom_center`` two numbers in ``0..1``, ``gain`` an
    integer in ``64..2540``, ``exposure`` one of :data:`EXPOSURE_CHOICES` and
    ``saturation`` an integer in ``0..255``.

    Raises:
        ValueError: With a short reason for the client, on an empty payload, an
            unknown key or a value out of range.
    """

    if not isinstance(data, Mapping):
        raise ValueError("calibration data must be an object")
    unknown = sorted(str(key) for key in data if key not in CALIBRATION_KEYS)
    if unknown:
        raise ValueError(f"unknown calibration key {unknown[0]!r}")
    if not data:
        raise ValueError("calibration.set needs at least one value")
    update: dict[str, Any] = {}
    if "zoom" in data:
        update["zoom"] = _number(data["zoom"], "zoom", *ZOOM_RANGE)
    if "zoom_center" in data:
        pair = data["zoom_center"]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("zoom_center must be [x, y] fractions in 0..1")
        update["zoom_center"] = (
            _number(pair[0], "zoom_center", 0.0, 1.0),
            _number(pair[1], "zoom_center", 0.0, 1.0),
        )
    if "gain" in data:
        update["gain"] = _integer(data["gain"], "gain", *GAIN_RANGE)
    if "exposure" in data:
        exposure = data["exposure"]
        if _whole(exposure) not in EXPOSURE_CHOICES:
            choices = " or ".join(str(choice) for choice in EXPOSURE_CHOICES)
            raise ValueError(f"exposure must be {choices}")
        update["exposure"] = _whole(exposure)
    if "saturation" in data:
        update["saturation"] = _integer(data["saturation"], "saturation", *SATURATION_RANGE)
    return update


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in {minimum:g}..{maximum:g}")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be a number in {minimum:g}..{maximum:g}")
    return number


def _whole(value: object) -> int | None:
    """``value`` as an int when it is a whole number (``200`` or ``200.0``), else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    number = _whole(value)
    if number is None or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return number


class CalibrationStore:
    """The JSON file that keeps the accepted values across restarts.

    Args:
        path: The file; ``~`` is expanded. Its directory is created on save.
    """

    def __init__(self, path: Path | str = DEFAULT_CALIBRATION_PATH) -> None:
        self._path = Path(path).expanduser()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, Any]:
        """Return the stored overrides, or ``{}`` when there is no usable file.

        A missing file is the normal case; an unreadable or invalid one is
        logged and ignored, so a bad calibration can never stop the demo.
        """

        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            logger.warning("cannot read the calibration %s: %s", self._path, exc)
            return {}
        try:
            return parse_update(json.loads(text))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("ignoring the calibration %s: %s", self._path, exc)
            return {}

    def save(self, calibration: Calibration) -> None:
        """Write the values atomically: a temporary file, fsync, then rename."""

        directory = self._path.parent
        directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps(calibration.to_json(), indent=2) + "\n"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise

    def remove(self) -> None:
        """Delete the file, so the next start uses ``config/showcase.yaml`` alone."""
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()


T = TypeVar("T")


class LatestWorker(Generic[T]):
    """Run ``apply`` on the newest submitted value, in its own daemon thread.

    The frame loop only stores the value and sets an event: a slow V4L2 control
    transfer or an fsync on the SD card never delays a frame. Values submitted
    while one is being applied collapse into the newest, so a slider dragged
    quickly costs one application, not one per step.
    """

    def __init__(self, apply: Callable[[T], None], *, name: str) -> None:
        self._apply = apply
        self._name = name
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._pending: tuple[T] | None = None
        self._closed = False
        self._thread: threading.Thread | None = None

    def submit(self, value: T) -> None:
        """Queue ``value`` in place of any value not applied yet."""
        with self._lock:
            if self._closed:
                return
            self._pending = (value,)
            self._idle.clear()
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()
        self._wake.set()

    def wait_idle(self, timeout: float = 2.0) -> bool:
        """Wait until every submitted value is applied; ``False`` on timeout."""
        return self._idle.wait(timeout)

    def close(self, timeout: float = 2.0) -> None:
        """Apply what is pending, then stop the thread."""
        with self._lock:
            self._closed = True
            thread = self._thread
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                self._wake.clear()
                pending = self._pending
                self._pending = None
            if pending is not None:
                try:
                    self._apply(pending[0])
                except Exception:
                    logger.exception("%s failed to apply a calibration", self._name)
            with self._lock:
                if self._pending is None:
                    self._idle.set()
                    if self._closed:
                        return
