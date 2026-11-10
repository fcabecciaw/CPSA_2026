"""Protocol execution: cue, trial window, operator keys and trial records.

The controller is terminal-independent so a test can drive it with a scripted key
source, a fake clock and a recording fake writer. It does not touch hardware.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol as TypingProtocol

from cpsa.adapters.recording.writer import TrialRecord
from cpsa.ports import Clock
from cpsa.recorder.protocol import Protocol, TrialSpec

POLL_INTERVAL_S = 0.05
SAFETY = (
    "Simulate gently. No real hits. Use the foam head for impacts.\nStop at once if anything hurts."
)


@dataclass(frozen=True)
class RunResult:
    """How a session ended."""

    completed_trials: int
    aborted: bool


class Console(TypingProtocol):
    """Operator-facing output; the terminal implementation prints to stdout."""

    def show_safety(self) -> None: ...
    def show_block(self, distance_cm: int, block: int, blocks: int) -> None: ...
    def show_trial(self, spec: TrialSpec, number: int, total: int) -> None: ...
    def show_countdown(self, seconds: int) -> None: ...
    def show_go(self, spec: TrialSpec) -> None: ...
    def show_rest(self, seconds: float) -> None: ...
    def show_paused(self, paused: bool) -> None: ...
    def show_bad(self, trial_id: int | None) -> None: ...


class KeySource(TypingProtocol):
    """Non-blocking operator key source."""

    def poll(self) -> str | None: ...
    def close(self) -> None: ...


class TrialRecorder(TypingProtocol):
    """The subset of :class:`~cpsa.adapters.recording.writer.SessionWriter` used here."""

    def begin_trial(self, trial_id: int, start_ts: float, end_ts: float) -> None: ...
    def extend_trial(self, trial_id: int, end_ts: float) -> None: ...
    def record_trial(self, record: TrialRecord) -> None: ...


class SessionController:
    """Run the trial protocol against a recorder and clock."""

    def __init__(
        self,
        *,
        protocol: Protocol,
        trials: tuple[TrialSpec, ...],
        recorder: TrialRecorder,
        clock: Clock,
        keys: KeySource,
        console: Console,
        beep: Callable[[], None] = lambda: None,
        sleep: Callable[[float], None] = time.sleep,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.protocol = protocol
        self.trials = trials
        self.recorder = recorder
        self.clock = clock
        self.keys = keys
        self.console = console
        self.beep = beep
        self._sleep = sleep
        self._stop_event = stop_event
        self._paused = False
        self._quit = False
        self._bad = False
        self._current_trial: int | None = None
        self._pending: TrialRecord | None = None

    def run(self) -> RunResult:
        """Run every distance block until the protocol ends or the operator quits."""
        self.console.show_safety()
        completed = 0
        blocks = len(self.protocol.distances_cm)
        for block_index, distance_cm in enumerate(self.protocol.distances_cm):
            block = [spec for spec in self.trials if spec.block_index == block_index]
            self.console.show_block(distance_cm, block_index + 1, blocks)
            if not self._wait_for_enter():
                return self._finish(completed, aborted=True)
            for position, spec in enumerate(block, start=1):
                if not self._run_trial(spec, position, len(block)):
                    return self._finish(completed, aborted=True)
                completed += 1
                if position < len(block):
                    self.console.show_rest(self.protocol.rest_s)
                    if not self._wait(self.protocol.rest_s):
                        return self._finish(completed, aborted=True)
        return self._finish(completed, aborted=False)

    # -- trial phases -----------------------------------------------------

    def _run_trial(self, spec: TrialSpec, number: int, total: int) -> bool:
        self._flush_pending()
        self.console.show_trial(spec, number, total)
        cue_ts = self._countdown()
        if cue_ts is None:
            return False

        self._current_trial = spec.trial_id
        self._bad = False
        trial_end_ts = cue_ts + spec.duration_s
        self.recorder.begin_trial(
            spec.trial_id,
            cue_ts - self.protocol.pre_s,
            trial_end_ts + self.protocol.post_s,
        )
        self.beep()
        self.console.show_go(spec)

        completed = self._wait(spec.duration_s)
        actual_end_ts = self.clock.monotonic()
        self.recorder.extend_trial(spec.trial_id, actual_end_ts + self.protocol.post_s)

        notes: list[str] = []
        if self._bad:
            notes.append("operator marked bad")
        if not completed:
            notes.append("aborted by operator")
        self._pending = TrialRecord(
            trial_id=spec.trial_id,
            class_name=spec.class_name,
            hand=spec.hand,
            target=spec.target,
            cue_ts=cue_ts,
            start_ts=cue_ts,
            end_ts=actual_end_ts,
            status="bad" if notes else "ok",
            note="; ".join(notes),
            distance_cm=spec.distance_cm,
        )
        self._current_trial = None
        return completed

    def _countdown(self) -> float | None:
        remaining = self.protocol.cue_lead_s
        shown = -1
        while remaining > 0:
            current = int(math.ceil(remaining))
            if current != shown:
                self.console.show_countdown(current)
                shown = current
            if not self._wait(min(1.0, remaining)):
                return None
            remaining -= min(1.0, remaining)
        return self.clock.monotonic()

    def _finish(self, completed: int, *, aborted: bool) -> RunResult:
        self._flush_pending()
        return RunResult(completed_trials=completed, aborted=aborted)

    def _flush_pending(self) -> None:
        if self._pending is not None:
            self.recorder.record_trial(self._pending)
            self._pending = None

    # -- waiting and keys -------------------------------------------------

    def _wait(self, seconds: float) -> bool:
        deadline = self.clock.monotonic() + max(0.0, seconds)
        while True:
            self._handle_keys()
            if self._quit or self._stop_requested():
                return False
            now = self.clock.monotonic()
            if now >= deadline:
                return True
            if self._paused:
                # Paused time must not count against the phase, and a paused trial
                # keeps its recording window open so its frames are not lost.
                deadline += POLL_INTERVAL_S
                self._extend_active_window()
                self._sleep(POLL_INTERVAL_S)
            else:
                self._sleep(min(POLL_INTERVAL_S, deadline - now))

    def _wait_for_enter(self) -> bool:
        while True:
            if self._handle_keys():
                return True
            if self._quit or self._stop_requested():
                return False
            self._sleep(POLL_INTERVAL_S)

    def _handle_keys(self) -> bool:
        enter = False
        while True:
            raw = self.keys.poll()
            if raw is None:
                return enter
            key = raw.lower() if len(raw) == 1 else raw
            if key in ("q", "\x03"):
                self._quit = True
            elif key in ("\r", "\n", " "):
                enter = True
            elif key == "p":
                self._paused = not self._paused
                self.console.show_paused(self._paused)
                if self._paused:
                    self._extend_active_window()
            elif key == "b":
                self._bad = True
                if self._pending is not None and self._current_trial is None:
                    note = self._pending.note
                    if "operator marked bad" not in note:
                        note = f"{note}; operator marked bad" if note else "operator marked bad"
                    self._pending = replace(self._pending, status="bad", note=note)
                self.console.show_bad(self._current_trial)

    def _stop_requested(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _extend_active_window(self) -> None:
        if self._current_trial is not None:
            self.recorder.extend_trial(
                self._current_trial, self.clock.monotonic() + self.protocol.post_s
            )
