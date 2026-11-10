"""Session controller: cue/window timing, operator keys and trial records."""

from __future__ import annotations

from dataclasses import replace

from cpsa.adapters.recording.writer import TrialRecord
from cpsa.recorder.protocol import Protocol, TrialSpec, parse, plan_trials
from cpsa.recorder.session import RunResult, SessionController
from tests.fakes import FakeClock

START = 100.0


class ScriptedKeys:
    def __init__(self, clock: FakeClock, script: list[tuple[float, str]]) -> None:
        self._clock = clock
        self._script = script
        self.closed = False

    def poll(self) -> str | None:
        if self._script and self._clock.monotonic() >= self._script[0][0]:
            return self._script.pop(0)[1]
        return None

    def close(self) -> None:
        self.closed = True


class RecordingConsole:
    def __init__(self) -> None:
        self.safety = 0
        self.blocks: list[tuple[int, int, int]] = []
        self.trials: list[TrialSpec] = []
        self.countdowns: list[int] = []
        self.goes: list[TrialSpec] = []
        self.rests: list[float] = []
        self.paused: list[bool] = []
        self.bad: list[int | None] = []

    def show_safety(self) -> None:
        self.safety += 1

    def show_block(self, distance_cm: int, block: int, blocks: int) -> None:
        self.blocks.append((distance_cm, block, blocks))

    def show_trial(self, spec: TrialSpec, number: int, total: int) -> None:
        self.trials.append(spec)

    def show_countdown(self, seconds: int) -> None:
        self.countdowns.append(seconds)

    def show_go(self, spec: TrialSpec) -> None:
        self.goes.append(spec)

    def show_rest(self, seconds: float) -> None:
        self.rests.append(seconds)

    def show_paused(self, paused: bool) -> None:
        self.paused.append(paused)

    def show_bad(self, trial_id: int | None) -> None:
        self.bad.append(trial_id)


class RecordingRecorder:
    def __init__(self) -> None:
        self.windows: list[tuple[int, float, float]] = []
        self.extensions: list[tuple[int, float]] = []
        self.trials: list[TrialRecord] = []

    def begin_trial(self, trial_id: int, start_ts: float, end_ts: float) -> None:
        self.windows.append((trial_id, start_ts, end_ts))

    def extend_trial(self, trial_id: int, end_ts: float) -> None:
        self.extensions.append((trial_id, end_ts))

    def record_trial(self, record: TrialRecord) -> None:
        self.trials.append(record)


def protocol(**changes: object) -> Protocol:
    data: dict[str, object] = {
        "seed": 3,
        "distances_cm": [55],
        "pre_s": 1.0,
        "post_s": 0.5,
        "rest_s": 0.2,
        "cue_lead_s": 0.2,
        "classes": [
            {
                "name": "fist_temple",
                "category": "dangerous",
                "hand": "both",
                "target": "temple",
                "reps": 1,
                "duration_s": 0.3,
            }
        ],
    }
    data.update(changes)
    return parse(data)


def controller(
    protocol_document: Protocol,
    *,
    keys: ScriptedKeys,
    clock: FakeClock,
    recorder: RecordingRecorder,
    console: RecordingConsole,
) -> SessionController:
    return SessionController(
        protocol=protocol_document,
        trials=plan_trials(protocol_document),
        recorder=recorder,
        clock=clock,
        keys=keys,
        console=console,
        beep=lambda: None,
        sleep=clock.advance,
    )


def test_runs_every_trial_and_records_windows() -> None:
    clock = FakeClock(START)
    keys = ScriptedKeys(clock, [(START, "\r")])
    recorder = RecordingRecorder()
    console = RecordingConsole()

    result = controller(
        protocol(), keys=keys, clock=clock, recorder=recorder, console=console
    ).run()

    assert result == RunResult(completed_trials=2, aborted=False)
    assert console.safety == 1
    assert console.blocks == [(55, 1, 1)]
    assert len(console.goes) == 2
    assert len(recorder.windows) == 2
    assert sorted(record.hand for record in recorder.trials) == ["L", "R"]
    assert all(record.status == "ok" for record in recorder.trials)
    assert all(record.distance_cm == 55 for record in recorder.trials)
    for record in recorder.trials:
        assert record.end_ts - record.start_ts >= 0.3 - 1e-9


def test_b_marks_the_current_trial_bad() -> None:
    clock = FakeClock(START)
    keys = ScriptedKeys(clock, [(START, "\r"), (START + 0.25, "b")])
    recorder = RecordingRecorder()
    console = RecordingConsole()

    result = controller(
        protocol(), keys=keys, clock=clock, recorder=recorder, console=console
    ).run()

    assert not result.aborted
    assert recorder.trials[0].status == "bad"
    assert "operator marked bad" in recorder.trials[0].note
    assert console.bad == [1]


def test_pause_extends_the_trial_window() -> None:
    clock = FakeClock(START)
    keys = ScriptedKeys(clock, [(START, "\r"), (START + 0.25, "p"), (START + 0.6, "p")])
    recorder = RecordingRecorder()
    console = RecordingConsole()

    result = controller(
        protocol(), keys=keys, clock=clock, recorder=recorder, console=console
    ).run()

    assert not result.aborted
    assert console.paused == [True, False]
    # The trial is 0.3 s long and starts after the 0.2 s countdown; the 0.35 s
    # pause must extend its end well beyond the unpaused 100.5 s.
    assert recorder.trials[0].end_ts > START + 0.8
    assert recorder.extensions


def test_q_during_a_trial_aborts_and_records_it_bad() -> None:
    clock = FakeClock(START)
    keys = ScriptedKeys(clock, [(START, "\r"), (START + 0.25, "q")])
    recorder = RecordingRecorder()
    console = RecordingConsole()

    result = controller(
        protocol(), keys=keys, clock=clock, recorder=recorder, console=console
    ).run()

    assert result == RunResult(completed_trials=0, aborted=True)
    assert recorder.trials[0].status == "bad"
    assert "aborted by operator" in recorder.trials[0].note


def test_b_during_rest_marks_the_finished_trial_bad() -> None:
    clock = FakeClock(START)
    keys = ScriptedKeys(
        clock,
        [(START, "\r"), (START + 0.6, "b"), (START + 0.7, "q")],
    )
    recorder = RecordingRecorder()
    console = RecordingConsole()

    result = controller(
        protocol(), keys=keys, clock=clock, recorder=recorder, console=console
    ).run()

    assert result.aborted
    assert recorder.trials[0].status == "bad"
    assert "operator marked bad" in recorder.trials[0].note


def test_stop_event_aborts_cleanly() -> None:
    import threading

    clock = FakeClock(START)
    keys = ScriptedKeys(clock, [(START, "\r")])
    recorder = RecordingRecorder()
    console = RecordingConsole()
    stop = threading.Event()
    stop.set()

    runner = SessionController(
        protocol=protocol(),
        trials=plan_trials(protocol()),
        recorder=recorder,
        clock=clock,
        keys=keys,
        console=console,
        sleep=lambda seconds: (clock.advance(seconds), stop.set())[1],
        stop_event=stop,
    )
    result = runner.run()
    assert result.aborted
    assert recorder.trials == []


def test_replace_keeps_trial_records_immutable() -> None:
    record = TrialRecord(1, "idle", "L", "none", 1.0, 1.0, 2.0, "ok", "", 55)
    marked = replace(record, status="bad", note="operator marked bad")
    assert record.status == "ok"
    assert marked.status == "bad"
