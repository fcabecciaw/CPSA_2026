"""Terminal console, non-blocking operator keys and the cue beep.

The console is plain stdout, so it works over SSH without a GUI toolkit. Keys are
read in cbreak mode (canonical mode and echo off, signals kept) and restored on
close; a non-TTY stdin is left untouched so the controller can run in tests.
"""

from __future__ import annotations

import contextlib
import math
import os
import queue
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import tty
import wave
from pathlib import Path
from typing import Any, TextIO

from cpsa.recorder.protocol import TrialSpec
from cpsa.recorder.session import SAFETY

_POLL_S = 0.05
_RULE = "=" * 64


class TerminalConsole:
    """Operator-facing text output with optional ANSI screen clearing."""

    def __init__(self, stream: TextIO = sys.stdout, clear: bool | None = None) -> None:
        self._stream = stream
        self._clear_screen = stream.isatty() if clear is None else clear

    def _write(self, text: str = "") -> None:
        self._stream.write(text + "\n")
        self._stream.flush()

    def _clear(self) -> None:
        if self._clear_screen:
            self._stream.write("\033[2J\033[H")

    def show_safety(self) -> None:
        self._clear()
        self._write(_RULE)
        self._write(" CPSA recording session")
        self._write(_RULE)
        for line in SAFETY.splitlines():
            self._write(f" {line}")
        self._write(_RULE)
        self._write(" Keys: b = mark trial bad, p = pause/resume, q = finish")

    def show_block(self, distance_cm: int, block: int, blocks: int) -> None:
        self._clear()
        self._write(_RULE)
        self._write(f" BLOCK {block}/{blocks} — distance {distance_cm} cm")
        self._write(_RULE)
        self._write(f" Set the participant at {distance_cm} cm from the camera.")
        self._write(" Press Enter when ready to start this block (q to finish).")

    def show_trial(self, spec: TrialSpec, number: int, total: int) -> None:
        self._clear()
        self._write(_RULE)
        self._write(
            f" BLOCK {spec.block_index + 1}  TRIAL {number}/{total}  ({spec.distance_cm} cm)"
        )
        self._write(_RULE)
        self._write()
        self._write(f"   {spec.class_name.upper().replace('_', ' ')}")
        self._write()
        self._write(f"   class: {spec.class_name} ({spec.category})")
        self._write(f"   hand: {spec.hand}    target: {spec.target}")
        self._write(f"   duration: {spec.duration_s:g} s")
        self._write(_RULE)
        self._write(f" {SAFETY.splitlines()[0]}")

    def show_countdown(self, seconds: int) -> None:
        self._write(f"   starting in {seconds} ...")

    def show_go(self, spec: TrialSpec) -> None:
        self._write()
        self._write(f"   >>> GO — {spec.class_name} ({spec.hand}) <<<")
        self._write()

    def show_rest(self, seconds: float) -> None:
        self._write(f" Rest {seconds:g} s — relax.")

    def show_paused(self, paused: bool) -> None:
        self._write(" PAUSED — press p to resume." if paused else " RESUMED.")

    def show_bad(self, trial_id: int | None) -> None:
        if trial_id is None:
            self._write(" Marked bad.")
        else:
            self._write(f" Trial {trial_id} marked bad.")


class RawKeySource:
    """Read single key presses from a terminal in cbreak mode."""

    def __init__(self, fd: int = 0) -> None:
        self._fd = fd
        self._keys: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved: list[Any] | None = None
        self._is_tty = os.isatty(fd)

    def start(self) -> None:
        """Enter cbreak mode and start the reader thread (no-op without a TTY)."""
        if self._thread is not None or not self._is_tty:
            return
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._read, name="cpsa-keys", daemon=True)
        self._thread.start()

    def poll(self) -> str | None:
        try:
            return self._keys.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
        if self._saved is not None:
            with contextlib.suppress(termios.error, OSError):
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None
        self._thread = None

    def _read(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], _POLL_S)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                data = os.read(self._fd, 1)
            except OSError:
                return
            if not data:
                return
            key = data.decode("utf-8", errors="ignore")
            if key == "\x1b":
                # Ignore the ESC of an arrow/function-key sequence; the remaining
                # bytes are harmless characters the controller does not use.
                continue
            self._keys.put(key)


class Beep:
    """Short cue tone through ``aplay``, or the terminal bell when it is missing."""

    def __init__(self) -> None:
        self._aplay = shutil.which("aplay")
        self._path: Path | None = None
        self._processes: list[subprocess.Popen[bytes]] = []
        if self._aplay is not None:
            self._path = _write_beep_wav()

    def play(self) -> None:
        if self._aplay is not None and self._path is not None:
            try:
                process = subprocess.Popen(
                    [self._aplay, "-q", str(self._path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                self._bell()
                return
            self._processes = [item for item in self._processes if item.poll() is None]
            self._processes.append(process)
            return
        self._bell()

    def close(self) -> None:
        for process in self._processes:
            with contextlib.suppress(OSError):
                process.terminate()
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                process.wait(timeout=0.2)
        self._processes.clear()
        if self._path is not None:
            with contextlib.suppress(OSError):
                self._path.unlink()
            self._path = None

    @staticmethod
    def _bell() -> None:
        sys.stderr.write("\a")
        sys.stderr.flush()


def _write_beep_wav() -> Path | None:
    descriptor, name = tempfile.mkstemp(prefix="cpsa-cue-", suffix=".wav")
    os.close(descriptor)
    path = Path(name)
    try:
        rate = 44100
        duration_s = 0.12
        frequency_hz = 880.0
        frames = bytearray()
        for index in range(int(rate * duration_s)):
            sample = int(0.35 * 32767 * math.sin(2 * math.pi * frequency_hz * index / rate))
            frames.extend(struct.pack("<h", sample))
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(bytes(frames))
    except OSError:
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    return path
