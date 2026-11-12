#!/usr/bin/env python3
"""Record labelled RGB runs for the rock/paper/scissors detector, with a live preview.

Runs **on the board** with the repo venv (``/home/kria/cpsa/.venv``)::

    .venv/bin/python scripts/capture_studio.py --out /home/kria/rps_capture_20261112

The script opens the F200 RGB node, keeps reading frames so the browser preview
stays live, and records one labelled run at a time on request.  Open
``http://10.42.0.83:8080/`` from a laptop: the page shows the live video, a REC
indicator with the remaining seconds and the frame count, the run form and the
list of runs already recorded.

Each run is written to ``<out>/<run_id>_<tilt>_<label>/`` as ``rgb_XXXXX.jpg``
(q95, 640x480, RGB order preserved) plus ``meta.json``; ``<out>/runs.json`` is
the atomically updated index of every run.

Safety
------
Same rule as ``scripts/capture_ir_frames.py``: on 2026-11-11 the board froze when
a capture ran next to the demo services, so the script refuses to open the
camera while ``cpsa-supervisor`` or ``cpsa-dpu`` is active::

    sudo systemctl stop cpsa-supervisor cpsa-dpu

``--allow-services`` skips that check (only for a deliberate coexistence test).
The camera is always closed, also on Ctrl-C or an error, and a run in progress is
finalised (``meta.json`` + ``runs.json``) before the process exits.  A run is
bounded by ``--max-run-seconds`` even if the page asks for more.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
from aiohttp import web
from board_services import ServiceCheckError, active_services, services_refusal

from cpsa.adapters.camera.devices import resolve_device
from cpsa.adapters.camera.uvc import UvcVideoSource
from cpsa.vision.types import VideoFrame

__all__ = [
    "COUNTDOWN_S",
    "DEFAULT_EVERY",
    "DEFAULT_MAX_RUN_SECONDS",
    "DEFAULT_PORT",
    "DEFAULT_SECONDS",
    "INDEX_HTML",
    "LABELS",
    "TILTS",
    "BusyError",
    "RunRequest",
    "Studio",
    "ValidationError",
    "active_services",
    "create_app",
    "main",
    "parse_args",
    "parse_run_request",
    "serve",
]

WIDTH = 640
"""Width of one RGB frame written by a run."""

HEIGHT = 480
"""Height of one RGB frame written by a run."""

FPS = 30.0
"""Requested camera rate."""

DEFAULT_EXPOSURE = 300
DEFAULT_GAIN = 160
DEFAULT_POWER_LINE_HZ = 50

LABELS = ("rock", "paper", "scissors", "negative")
"""Label of a run; ``negative`` covers no hand and other hand shapes."""

TILTS = ("top", "bust", "frontal")
"""Camera tilt of a run: near top-down, tilted with part of the bust, half-bust."""

DEFAULT_SECONDS = 60
DEFAULT_EVERY = 5
DEFAULT_MAX_RUN_SECONDS = 120.0
DEFAULT_PORT = 8080
DEFAULT_OUT_PREFIX = "/home/kria/rps_capture_"
BOARD_HOST = "10.42.0.83"
"""Board address shown in the startup banner when the server binds 0.0.0.0."""

COUNTDOWN_S = 3.0
"""Time between Start and the first delivered frame of the run."""

PREVIEW_FPS = 15.0
"""Preview rate; the camera itself keeps running at 30 fps."""

PREVIEW_TOLERANCE = 0.95
"""Fraction of the preview interval a frame must wait, to absorb float rounding."""

PREVIEW_QUALITY = 70
SAVE_QUALITY = 95
MAX_NOTE_CHARS = 200

IDLE = "idle"
COUNTDOWN = "countdown"
RECORDING = "recording"

BOUNDARY = "cpsa-rps-frame"
"""Multipart boundary of ``GET /video.mjpeg``."""

LOGGER = logging.getLogger("cpsa.capture_studio")


class StudioError(RuntimeError):
    """Base class for the errors the page reports with a 4xx status."""


class ValidationError(StudioError, ValueError):
    """The page asked for a run the studio cannot record."""


class BusyError(StudioError):
    """A run is already counting down or recording."""


class _Source(Protocol):
    """The part of :class:`UvcVideoSource` the studio uses."""

    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None: ...

    def stop(self) -> None: ...


SourceFactory = Callable[[str, argparse.Namespace], _Source]
ServiceCheck = Callable[[], Sequence[str]]
NodeResolver = Callable[[str, str], str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValidationError(f"{name} must be a number")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number") from None
    return parsed


def _parse_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValidationError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number") from None


@dataclass(frozen=True)
class RunRequest:
    """What the page asks for: one labelled run of a given length and tilt.

    Args:
        label: one of :data:`LABELS`.
        tilt: one of :data:`TILTS`.
        seconds: requested duration; the studio clamps it to ``max_run_seconds``.
        every: keep every K-th delivered frame.
        note: free text stored in ``meta.json``.
    """

    label: str
    tilt: str
    seconds: float = DEFAULT_SECONDS
    every: int = DEFAULT_EVERY
    note: str = ""

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValidationError(f"label must be one of {', '.join(LABELS)}")
        if self.tilt not in TILTS:
            raise ValidationError(f"tilt must be one of {', '.join(TILTS)}")
        if self.seconds <= 0:
            raise ValidationError("seconds must be positive")
        if self.every <= 0:
            raise ValidationError("every must be positive")
        if len(self.note) > MAX_NOTE_CHARS:
            raise ValidationError(f"note must be at most {MAX_NOTE_CHARS} characters")


def parse_run_request(payload: Mapping[str, Any]) -> RunRequest:
    """Build a :class:`RunRequest` from a form/JSON payload, or raise.

    Raises:
        ValidationError: If a field is missing, unknown or out of range.
    """

    return RunRequest(
        label=str(payload.get("label", "")).strip(),
        tilt=str(payload.get("tilt", "")).strip(),
        seconds=_parse_float(payload.get("seconds", DEFAULT_SECONDS), "seconds"),
        every=_parse_int(payload.get("every", DEFAULT_EVERY), "every"),
        note=str(payload.get("note", "")).strip(),
    )


@dataclass
class _Run:
    """One run in progress: its metadata, counters and written frames."""

    run_id: str
    request: RunRequest
    directory: Path
    duration_s: float
    every: int
    started_at: str
    start_monotonic: float
    delivered: int = 0
    saved: int = 0
    frames: list[dict[str, Any]] = field(default_factory=list)


class Studio:
    """Own the camera frames, the preview slot and at most one active run.

    The camera thread calls :meth:`on_frame`; the HTTP handlers call
    :meth:`start`, :meth:`stop`, :meth:`status` and :meth:`latest_preview` from
    the event loop.  All state sits behind one lock, and the preview is a single
    latest-frame slot: a slow browser never queues frames and never slows the
    camera loop beyond the cost of one JPEG encode.

    Args:
        out: output root; every run gets one directory below it.
        clock: test seam; defaults to :func:`time.monotonic`.
        camera: camera description stored in each ``meta.json`` (node, controls).
        max_run_seconds: hard bound on one run, whatever the page asks for.
        countdown_s: seconds between Start and the first recorded frame.
        preview_fps: preview frames per second offered to the browsers.
    """

    def __init__(
        self,
        out: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        camera: Mapping[str, Any] | None = None,
        max_run_seconds: float = DEFAULT_MAX_RUN_SECONDS,
        countdown_s: float = COUNTDOWN_S,
        preview_fps: float = PREVIEW_FPS,
        preview_quality: int = PREVIEW_QUALITY,
        save_quality: int = SAVE_QUALITY,
    ) -> None:
        if max_run_seconds <= 0:
            raise ValueError("max_run_seconds must be positive")
        if countdown_s < 0:
            raise ValueError("countdown_s must not be negative")
        if preview_fps <= 0:
            raise ValueError("preview_fps must be positive")
        self.out = Path(out)
        self.camera = dict(camera or {})
        self.max_run_seconds = float(max_run_seconds)
        self.countdown_s = float(countdown_s)
        self.preview_fps = float(preview_fps)
        self.preview_quality = int(preview_quality)
        self.save_quality = int(save_quality)
        self._clock = clock
        self._lock = threading.Lock()
        self._phase = IDLE
        self._run: _Run | None = None
        self._countdown_deadline = 0.0
        self._last_preview: float | None = None
        self._preview: tuple[int, bytes] | None = None
        self._preview_seq = 0
        self._delivered = 0
        self._error: str | None = None
        self._runs: list[dict[str, Any]] = _load_runs(self.out)

    # -- camera side -----------------------------------------------------

    def on_frame(self, frame: VideoFrame) -> None:
        """Consume one camera frame: preview always, recording only while active.

        Decoding, the JPEG encode and the file write happen on the camera thread
        while the lock is held: a browser can only ever read the latest slot, so
        the HTTP side cannot queue frames or grow memory.  The lock is held for
        the duration of one frame (a few milliseconds), never across a request.
        """
        now = self._clock()
        with self._lock:
            self._delivered += 1
            self._advance(now)
            recording = self._phase == RECORDING and self._run is not None
            # 5 % tolerance: a 30 fps camera then yields a steady 15 fps preview
            # instead of dropping to 10 fps when 1/15 does not divide 1/30 exactly.
            preview_due = self._last_preview is None or (
                now - self._last_preview >= PREVIEW_TOLERANCE * (1.0 / self.preview_fps)
            )
            if not recording and not preview_due:
                return
            rgb = frame.decode()
            if preview_due:
                self._offer_preview(now, rgb)
            run = self._run
            if recording and run is not None:
                index = run.delivered
                run.delivered += 1
                if index % run.every == 0:
                    self._write_frame(run, frame, rgb)
            self._advance(now)

    def on_camera_error(self, exc: Exception) -> None:
        """Stop (and finalise) the active run when the camera delivers an error."""
        LOGGER.error("camera error: %s", exc)
        with self._lock:
            self._error = f"{type(exc).__name__}: {exc}"
            self._advance(self._clock(), force_error=True)

    def _offer_preview(self, now: float, rgb: Any) -> None:
        """Encode one preview frame into the latest slot; caller holds the lock."""
        ok, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.preview_quality],
        )
        if not ok:
            LOGGER.warning("could not encode a preview frame")
            return
        self._preview_seq += 1
        self._preview = (self._preview_seq, buffer.tobytes())
        self._last_preview = now

    def _write_frame(self, run: _Run, frame: VideoFrame, rgb: Any) -> None:
        """Write one kept frame as q95 JPEG (RGB → BGR for OpenCV)."""
        path = run.directory / f"rgb_{run.saved:05d}.jpg"
        written = cv2.imwrite(
            str(path),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.save_quality],
        )
        if not written:
            self._error = f"could not write {path}"
            return
        run.saved += 1
        run.frames.append(
            {
                "file": path.name,
                "seq": frame.seq,
                "t_monotonic": round(float(frame.t), 6),
                "captured_at": _now_iso(),
            }
        )

    # -- run lifecycle ---------------------------------------------------

    def start(self, request: RunRequest) -> str:
        """Begin the countdown for ``request`` and return the new run id.

        Raises:
            BusyError: If a run is already counting down or recording.
        """

        with self._lock:
            if self._phase != IDLE:
                raise BusyError(f"a run is already {self._phase}")
            self.out.mkdir(parents=True, exist_ok=True)
            now = self._clock()
            run_id = self._new_run_id()
            duration = min(float(request.seconds), self.max_run_seconds)
            directory = self.out / f"{run_id}_{request.tilt}_{request.label}"
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except OSError as exc:
                raise StudioError(f"cannot create {directory}: {exc}") from exc
            self._run = _Run(
                run_id=run_id,
                request=request,
                directory=directory,
                duration_s=duration,
                every=request.every,
                started_at=_now_iso(),
                start_monotonic=now + self.countdown_s,
            )
            self._phase = COUNTDOWN
            self._countdown_deadline = now + self.countdown_s
            self._error = None
            LOGGER.info(
                "run %s: label=%s tilt=%s %.0f s every=%d -> %s",
                run_id,
                request.label,
                request.tilt,
                duration,
                request.every,
                directory,
            )
            return run_id

    def stop(self, reason: str = "user") -> dict[str, Any] | None:
        """Finalise the active run; returns its summary, or ``None`` when idle."""
        with self._lock:
            if self._phase == IDLE:
                return None
            if self._phase == COUNTDOWN:
                self._abort_countdown()
                return None
            return self._finish(reason)

    def shutdown(self) -> None:
        """Finalise a run still in progress, so Ctrl-C never loses metadata."""
        with self._lock:
            if self._phase == RECORDING:
                self._finish("shutdown")
            elif self._phase == COUNTDOWN:
                self._abort_countdown()

    def poll(self, now: float | None = None) -> None:
        """Advance the phase; called by the camera loop and by the HTTP ticker."""
        with self._lock:
            self._advance(self._clock() if now is None else now)

    def _advance(self, now: float, *, force_error: bool = False) -> None:
        """Move countdown → recording → idle; caller holds the lock."""
        if force_error and self._phase != IDLE:
            self._finish("error")
            return
        if self._phase == COUNTDOWN and now >= self._countdown_deadline:
            self._phase = RECORDING
            if self._run is not None:
                self._run.started_at = _now_iso()
        if self._phase == RECORDING and self._run is not None:
            elapsed = now - self._run.start_monotonic
            if self._error:
                self._finish("error")
            elif elapsed >= self._run.duration_s:
                self._finish(
                    "max_seconds"
                    if self._run.request.seconds > self.max_run_seconds
                    else "duration"
                )

    def _abort_countdown(self) -> None:
        run = self._run
        self._phase = IDLE
        self._run = None
        if run is not None:
            with contextlib.suppress(OSError):
                run.directory.rmdir()

    def _finish(self, reason: str) -> dict[str, Any]:
        """Write ``meta.json`` and the ``runs.json`` index; caller holds the lock."""
        run = self._run
        if run is None:
            self._phase = IDLE
            return {}
        finished_at = _now_iso()
        meta = {
            "run_id": run.run_id,
            "label": run.request.label,
            "tilt": run.request.tilt,
            "note": run.request.note,
            "started_at": run.started_at,
            "finished_at": finished_at,
            "stopped_reason": reason,
            "requested_seconds": run.request.seconds,
            "duration_s": round(run.duration_s, 3),
            "elapsed_s": round(max(0.0, self._clock() - run.start_monotonic), 3),
            "every": run.every,
            "delivered": run.delivered,
            "saved": run.saved,
            "error": self._error,
            "camera": {
                "node": self.camera.get("node"),
                "width": self.camera.get("width", WIDTH),
                "height": self.camera.get("height", HEIGHT),
                "fps": self.camera.get("fps", FPS),
                "controls": self.camera.get("controls", {}),
            },
            "frames": run.frames,
        }
        try:
            (run.directory / "meta.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            self._error = f"could not write meta.json: {exc}"
        summary = {
            "run_id": run.run_id,
            "label": run.request.label,
            "tilt": run.request.tilt,
            "note": run.request.note,
            "frames": run.saved,
            "duration_s": meta["elapsed_s"],
            "started_at": run.started_at,
            "finished_at": finished_at,
            "stopped_reason": reason,
            "directory": run.directory.name,
        }
        self._runs.append(summary)
        _write_runs(self.out, self._runs)
        self._phase = IDLE
        self._run = None
        LOGGER.info("finished run %s: %d frames (%s)", run.run_id, run.saved, reason)
        return summary

    def _new_run_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        candidate = stamp
        suffix = 1
        while any(self.out.glob(f"{candidate}_*")) or any(
            run["run_id"] == candidate for run in self._runs
        ):
            suffix += 1
            candidate = f"{stamp}-{suffix}"
        return candidate

    # -- read-only views -------------------------------------------------

    @property
    def delivered(self) -> int:
        """Frames delivered by the camera since start."""
        return self._delivered

    def latest_preview(self) -> tuple[int, bytes] | None:
        """The newest preview frame as ``(seq, jpeg)``, or ``None`` before the first."""
        return self._preview

    def status(self) -> dict[str, Any]:
        """The JSON payload behind ``GET /api/status``."""
        with self._lock:
            now = self._clock()
            payload: dict[str, Any] = {
                "phase": self._phase,
                "delivered": self._delivered,
                "error": self._error,
                "out": str(self.out),
                "camera": dict(self.camera),
                "labels": list(LABELS),
                "tilts": list(TILTS),
                "max_run_seconds": self.max_run_seconds,
                "defaults": {"seconds": int(DEFAULT_SECONDS), "every": DEFAULT_EVERY},
                "runs": list(reversed(self._runs)),
            }
            run = self._run
            if run is not None:
                payload["run"] = {
                    "run_id": run.run_id,
                    "label": run.request.label,
                    "tilt": run.request.tilt,
                    "note": run.request.note,
                    "every": run.every,
                    "frames": run.saved,
                    "delivered": run.delivered,
                    "duration_s": round(run.duration_s, 3),
                }
                if self._phase == COUNTDOWN:
                    payload["countdown_s"] = max(0.0, self._countdown_deadline - now)
                else:
                    elapsed = min(run.duration_s, max(0.0, now - run.start_monotonic))
                    payload["elapsed_s"] = round(elapsed, 2)
                    payload["remaining_s"] = round(max(0.0, run.duration_s - elapsed), 2)
            return payload


def _load_runs(out: Path) -> list[dict[str, Any]]:
    path = Path(out) / "runs.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    runs = payload.get("runs") if isinstance(payload, dict) else payload
    return [run for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Replace ``path`` with ``payload`` without leaving a half-written file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_runs(out: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    try:
        _write_json_atomic(Path(out) / "runs.json", {"updated_at": _now_iso(), "runs": list(runs)})
    except OSError as exc:
        LOGGER.error("could not update %s: %s", Path(out) / "runs.json", exc)


# -- HTTP ----------------------------------------------------------------

STUDIO_KEY: web.AppKey[Studio] = web.AppKey("studio", Studio)
TICK_S_KEY: web.AppKey[float] = web.AppKey("tick_s", float)
TICKER_KEY: web.AppKey[asyncio.Task[None]] = web.AppKey("ticker", asyncio.Task)

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RPS capture studio</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 15px/1.4 system-ui, sans-serif; background: #111; color: #eee; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 20px; margin: 0 0 12px; }
  .stage { position: relative; background: #000; border-radius: 6px; overflow: hidden; }
  .stage img { display: block; width: 100%; height: auto; }
  .badge { position: absolute; top: 12px; left: 12px; padding: 6px 12px; border-radius: 4px;
           font-weight: 700; letter-spacing: .08em; background: #444; }
  .badge.rec { background: #c62828; }
  .badge.rec::before { content: "\\25CF  "; }
  .countdown { position: absolute; inset: 0; display: none; place-items: center;
               font-size: 180px; font-weight: 800; color: #fff; background: rgba(0,0,0,.45); }
  .countdown.on { display: grid; }
  .stats { position: absolute; top: 12px; right: 12px; text-align: right;
           font-variant-numeric: tabular-nums; }
  .stats div { background: rgba(0,0,0,.55); padding: 4px 10px; margin-bottom: 4px;
               border-radius: 4px; }
  form { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;
         margin: 16px 0; align-items: end; }
  label { display: block; font-size: 12px; color: #aaa; margin-bottom: 3px; }
  input, select, button { font: inherit; padding: 7px 8px; border-radius: 4px;
                          border: 1px solid #555; background: #1c1c1c; color: #eee;
                          width: 100%; box-sizing: border-box; }
  button { cursor: pointer; border: 0; font-weight: 600; }
  button:disabled { opacity: .45; cursor: default; }
  #start { background: #2e7d32; }
  #stop { background: #b71c1c; }
  #msg { min-height: 20px; color: #ffb74d; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #2a2a2a; }
  th { color: #999; font-weight: 500; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<main>
  <h1>RPS capture studio <span id="out" style="font-size:12px;color:#777"></span></h1>
  <div class="stage">
    <img id="video" src="/video.mjpeg" alt="live camera">
    <div id="badge" class="badge">IDLE</div>
    <div id="countdown" class="countdown">3</div>
    <div id="stats" class="stats"></div>
  </div>
  <div id="msg"></div>
  <form id="form">
    <div><label for="label">Label</label>
      <select id="label"><option>rock</option><option>paper</option>
        <option>scissors</option><option>negative</option></select></div>
    <div><label for="tilt">Tilt</label>
      <select id="tilt"><option>top</option><option>bust</option>
        <option>frontal</option></select></div>
    <div><label for="seconds">Seconds</label>
      <input id="seconds" type="number" min="1" step="1" value="60"></div>
    <div><label for="every">Every K</label>
      <input id="every" type="number" min="1" step="1" value="5"></div>
    <div><label for="note">Note</label>
      <input id="note" type="text" maxlength="200" placeholder="optional"></div>
    <div><button id="start" type="submit">Start</button></div>
    <div><button id="stop" type="button">Stop</button></div>
  </form>
  <table>
    <thead><tr><th>Run</th><th>Label</th><th>Tilt</th><th>Frames</th><th>Seconds</th><th>Note</th></tr></thead>
    <tbody id="runs"></tbody>
  </table>
</main>
<script>
const $ = (id) => document.getElementById(id);
let busy = false;

function setForm(s) {
  const seconds = $("seconds");
  const every = $("every");
  if (s.defaults && !seconds.dataset.touched) seconds.value = s.defaults.seconds;
  if (s.defaults && !every.dataset.touched) every.value = s.defaults.every;
}

function render(s) {
  setForm(s);
  $("out").textContent = s.out || "";
  const badge = $("badge");
  const stats = $("stats");
  const countdown = $("countdown");
  const run = s.run || null;
  const recording = s.phase === "recording";
  badge.className = "badge" + (recording ? " rec" : "");
  badge.textContent = recording ? "REC" : s.phase.toUpperCase();
  if (s.phase === "countdown" && s.countdown_s !== undefined) {
    countdown.classList.add("on");
    countdown.textContent = String(Math.max(1, Math.ceil(s.countdown_s)));
  } else {
    countdown.classList.remove("on");
  }
  stats.innerHTML = "";
  if (run) {
    stats.innerHTML =
      "<div>" + run.label + " / " + run.tilt + "</div>" +
      (recording ? "<div>" + Number(s.remaining_s).toFixed(1) + " s left</div>" : "") +
      "<div>" + ((run.frames)|0) + " frames</div>";
  }
  $("start").disabled = busy || s.phase !== "idle";
  $("stop").disabled = busy || s.phase === "idle";
  $("msg").textContent = s.error || "";
  const body = $("runs");
  body.innerHTML = "";
  for (const r of s.runs || []) {
    const tr = document.createElement("tr");
    for (const value of [r.run_id, r.label, r.tilt, r.frames,
                         Number(r.duration_s).toFixed(1), r.note || ""]) {
      const td = document.createElement("td");
      td.textContent = value;
      if (typeof value === "number") td.className = "num";
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

async function refresh() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    render(await response.json());
  } catch (error) {
    $("msg").textContent = "studio offline: " + error;
  }
}

async function post(path, payload) {
  busy = true;
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) $("msg").textContent = body.error || ("HTTP " + response.status);
    else $("msg").textContent = "";
  } finally {
    busy = false;
    refresh();
  }
}

$("form").addEventListener("submit", (event) => {
  event.preventDefault();
  post("/api/start", {
    label: $("label").value,
    tilt: $("tilt").value,
    seconds: Number($("seconds").value),
    every: Number($("every").value),
    note: $("note").value,
  });
});
$("stop").addEventListener("click", () => post("/api/stop"));
$("seconds").addEventListener("input", (e) => { e.target.dataset.touched = "1"; });
$("every").addEventListener("input", (e) => { e.target.dataset.touched = "1"; });
refresh();
setInterval(refresh, 500);
</script>
</body>
</html>
"""


def _studio(request: web.Request) -> Studio:
    return request.app[STUDIO_KEY]


async def _index(request: web.Request) -> web.Response:
    return web.Response(text=INDEX_HTML, content_type="text/html", charset="utf-8")


async def _api_status(request: web.Request) -> web.Response:
    studio = _studio(request)
    studio.poll()
    return web.json_response(studio.status())


async def _api_start(request: web.Request) -> web.Response:
    studio = _studio(request)
    try:
        payload = json.loads(await request.text())
    except (ValueError, UnicodeDecodeError):
        return web.json_response({"ok": False, "error": "expected a JSON body"}, status=400)
    if not isinstance(payload, Mapping):
        return web.json_response({"ok": False, "error": "expected a JSON object"}, status=400)
    try:
        run_id = studio.start(parse_run_request(payload))
    except ValidationError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except BusyError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)
    except StudioError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)
    return web.json_response({"ok": True, "run_id": run_id})


async def _api_stop(request: web.Request) -> web.Response:
    studio = _studio(request)
    summary = studio.stop()
    return web.json_response({"ok": True, "run": summary})


async def _write_part(response: web.StreamResponse, seq: int, jpeg: bytes) -> None:
    """Write one MJPEG part; aiohttp 3.14 made ``write`` awaitable."""
    header = (
        f"--{BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpeg)}\r\n"
        f"X-Frame-Seq: {seq}\r\n\r\n"
    ).encode("ascii")
    written = response.write(header + jpeg + b"\r\n")
    if inspect.isawaitable(written):
        await written
    else:
        await response.drain()


async def _video(request: web.Request) -> web.StreamResponse:
    """Stream the newest preview frame to every client, never blocking the camera."""
    studio = _studio(request)
    response = web.StreamResponse(
        headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Connection": "close",
        }
    )
    await response.prepare(request)
    interval = 1.0 / studio.preview_fps
    seen = -1
    try:
        while True:
            item = studio.latest_preview()
            if item is not None and item[0] != seen:
                seen = item[0]
                await _write_part(response, seen, item[1])
            await asyncio.sleep(interval)
    except ConnectionResetError:
        pass
    return response


async def _tick(app: web.Application) -> None:
    """Finalise a run that reached its duration even when no frames arrive."""
    studio = app[STUDIO_KEY]
    interval = app[TICK_S_KEY]
    while True:
        studio.poll()
        await asyncio.sleep(interval)


def create_app(studio: Studio, *, tick_s: float = 0.2) -> web.Application:
    """Build the aiohttp application that serves the page, the preview and the API."""
    app = web.Application()
    app[STUDIO_KEY] = studio
    app[TICK_S_KEY] = tick_s
    app.router.add_get("/", _index)
    app.router.add_get("/index.html", _index)
    app.router.add_get("/video.mjpeg", _video)
    app.router.add_get("/api/status", _api_status)
    app.router.add_post("/api/start", _api_start)
    app.router.add_post("/api/stop", _api_stop)

    async def _start_ticker(app: web.Application) -> None:
        app[TICKER_KEY] = asyncio.create_task(_tick(app))

    async def _stop_ticker(app: web.Application) -> None:
        task = app[TICKER_KEY]
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    app.on_startup.append(_start_ticker)
    app.on_cleanup.append(_stop_ticker)
    return app


async def serve(
    studio: Studio,
    *,
    host: str,
    port: int,
    stop: asyncio.Event | None = None,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Serve the studio until ``stop`` is set, or until SIGINT/SIGTERM arrives."""
    app = create_app(studio)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    event = stop if stop is not None else asyncio.Event()
    if stop is None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, event.set)
    if on_ready is not None:
        on_ready()
    try:
        await event.wait()
    finally:
        await runner.cleanup()


# -- CLI -----------------------------------------------------------------


def _resolve_node(device: str, match: str) -> str:
    """Resolve ``device`` to the RGB node the UVC source will open."""
    return resolve_device(device, match=match)


def _make_source(node: str, args: argparse.Namespace) -> UvcVideoSource:
    return UvcVideoSource(
        node,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        exposure=args.exposure,
        gain=args.gain,
        power_line_hz=args.power_line_hz,
        match=args.match,
    )


def camera_description(node: str, args: argparse.Namespace) -> dict[str, Any]:
    """The camera block written into every ``meta.json`` and served by the API."""
    return {
        "node": node,
        "requested": args.device,
        "match": args.match,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "controls": {
            "exposure": args.exposure,
            "gain": args.gain,
            "power_line_hz": args.power_line_hz,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(f"{DEFAULT_OUT_PREFIX}{datetime.now().strftime('%Y%m%d')}"),
        help=f"output root, one directory per run (default {DEFAULT_OUT_PREFIX}<date>)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"HTTP port (default {DEFAULT_PORT})"
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    parser.add_argument(
        "--device",
        default="auto",
        help="V4L2 node, or 'auto' to pick the RGB node by --match (default: auto)",
    )
    parser.add_argument(
        "--match",
        default="RealSense",
        help="case-insensitive fragment of the camera card name (default: RealSense)",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=DEFAULT_EXPOSURE,
        help=f"manual exposure_time_absolute (default {DEFAULT_EXPOSURE})",
    )
    parser.add_argument(
        "--gain", type=int, default=DEFAULT_GAIN, help=f"manual V4L2 gain (default {DEFAULT_GAIN})"
    )
    parser.add_argument(
        "--power-line-hz",
        type=int,
        default=DEFAULT_POWER_LINE_HZ,
        choices=(0, 50, 60),
        help=f"anti-flicker mains frequency (default {DEFAULT_POWER_LINE_HZ})",
    )
    parser.add_argument(
        "--max-run-seconds",
        type=float,
        default=DEFAULT_MAX_RUN_SECONDS,
        help=f"hard bound on one run (default {DEFAULT_MAX_RUN_SECONDS:g} s)",
    )
    parser.add_argument(
        "--allow-services",
        action="store_true",
        help="skip the cpsa-supervisor/cpsa-dpu check (do not use during a demo)",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    service_check: ServiceCheck = active_services,
    resolver: NodeResolver = _resolve_node,
    source_factory: SourceFactory = _make_source,
    clock: Callable[[], float] = time.monotonic,
    serve_forever: Callable[[Studio, argparse.Namespace], None] | None = None,
) -> int:
    """Run the studio on the board; returns the process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.max_run_seconds <= 0:
        print("--max-run-seconds must be positive", file=sys.stderr)
        return 2
    if not 0 < args.port < 65536:
        print("--port must be between 1 and 65535", file=sys.stderr)
        return 2
    if not args.match.strip():
        print("--match must not be empty", file=sys.stderr)
        return 2

    if not args.allow_services:
        try:
            busy = service_check()
        except ServiceCheckError as exc:
            print(f"{exc}; pass --allow-services to skip the check", file=sys.stderr)
            return 2
        if busy:
            print(services_refusal(busy), file=sys.stderr)
            return 2

    try:
        node = resolver(args.device, args.match)
    except Exception as exc:
        print(f"cannot resolve the RGB camera: {exc}", file=sys.stderr)
        return 1

    studio = Studio(
        args.out,
        clock=clock,
        camera=camera_description(node, args),
        max_run_seconds=args.max_run_seconds,
    )
    source = source_factory(node, args)
    serve_loop = serve_forever if serve_forever is not None else _serve_args(args)
    try:
        try:
            source.start(studio.on_frame, studio.on_camera_error)
        except Exception as exc:
            print(f"cannot open the RGB camera: {exc}", file=sys.stderr)
            return 1
        shown = args.host if args.host not in ("", "0.0.0.0", "::") else BOARD_HOST
        print(f"recording into {args.out}")
        print(f"open http://{shown}:{args.port}/ in the laptop browser")
        try:
            serve_loop(studio, args)
        except OSError as exc:
            print(f"cannot serve on port {args.port}: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("interrupted; closing the camera", file=sys.stderr)
    finally:
        studio.shutdown()
        source.stop()
    print(f"recorded {len(studio.status()['runs'])} runs under {args.out}")
    return 0


def _serve_args(args: argparse.Namespace) -> Callable[[Studio, argparse.Namespace], None]:
    def run(studio: Studio, options: argparse.Namespace) -> None:
        asyncio.run(serve(studio, host=options.host, port=options.port))

    return run


if __name__ == "__main__":
    raise SystemExit(main())
