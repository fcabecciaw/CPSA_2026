import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from cpsa.adapters.verifier_stub import StubVerifier
from cpsa.application.dispatcher import Dispatcher
from cpsa.application.slot import LatestSlot
from cpsa.domain.model import (
    Actuation,
    ActuatorKind,
    Decision,
    DecisionRecord,
    Detection,
    GateReason,
    Tag,
    Verdict,
    VisionAssessment,
)
from cpsa.ports import DetectionObserver, Verifier
from tests.fakes import FakeClock, MemorySink, RecordingActuator, RecordingGui, ScriptedVerifier

STRONG_META = {"duty": 100, "duration": 900}
MILD_META = {"duty": 40, "duration": 700}


class AssessmentVerifier:
    """Returns a scripted assessment (or bare verdict) and records verify/observe calls."""

    def __init__(self, result: VisionAssessment | Verdict) -> None:
        self.result = result
        self.calls: list[Detection] = []
        self.observed: list[Detection] = []
        self.journal: list[str] = []

    def verify(self, detection: Detection) -> VisionAssessment | Verdict:
        self.journal.append("verify")
        self.calls.append(detection)
        return self.result

    def observe(self, detection: Detection) -> None:
        self.journal.append("observe")
        self.observed.append(detection)


class FailingObserver(AssessmentVerifier):
    def observe(self, detection: Detection) -> None:
        raise RuntimeError("episode store unavailable")


class Harness:
    def __init__(
        self,
        verifier: Verifier,
        actuators: Sequence[RecordingActuator] | None = None,
        deadline_s: float = 1.0,
    ) -> None:
        self.clock = FakeClock()
        self.slot: LatestSlot[Detection] = LatestSlot()
        self.verifier = verifier
        self.meta = RecordingActuator("meta_AA", ActuatorKind.METAMOTION)
        self.actuators = list(actuators) if actuators is not None else [self.meta]
        self.sink = MemorySink()
        self.gui = RecordingGui()
        self.dispatcher = Dispatcher(
            self.slot,
            verifier,
            self.actuators,
            self.sink,
            self.gui,
            self.clock,
            cooldown_s=5.0,
            deadline_s=deadline_s,
        )

    def detect(self, tag: Tag, end_ts: float = 0.0) -> Detection:
        return Detection(tag, self.clock.now(), end_ts, (0.5,) * 18)


@pytest.fixture
def harness_factory() -> Iterator[Callable[..., Harness]]:
    created: list[Harness] = []

    def factory(*args: object, **kwargs: object) -> Harness:
        harness = Harness(*args, **kwargs)  # type: ignore[arg-type]
        created.append(harness)
        return harness

    yield factory
    for harness in created:
        harness.dispatcher.stop()


def test_rejected_detection_is_not_actuated_and_is_logged(
    harness_factory: Callable[..., Harness],
) -> None:
    h = harness_factory(ScriptedVerifier(Verdict.REJECTED))
    record = h.dispatcher.handle(h.detect(Tag.DANGEROUS, end_ts=42.0))

    assert h.meta.executed == []
    rejection = VisionAssessment(Verdict.REJECTED)
    assert record == DecisionRecord(
        timestamp=h.clock.now(),
        window_end_ts=42.0,
        tag=Tag.DANGEROUS,
        tag_changed=True,
        decision=Decision(Verdict.REJECTED, GateReason.REJECTED, False, (), None, rejection),
    )
    assert h.sink.records == [record]
    assert h.gui.records == [record]


def _sleep_past_deadline(_: Detection) -> Verdict:
    time.sleep(0.5)
    return Verdict.REJECTED


def _raise(_: Detection) -> Verdict:
    raise RuntimeError("camera unavailable")


def _not_a_verdict(_: Detection) -> Verdict:
    return "REJECTED"  # type: ignore[return-value]


def _malformed_assessment(_: Detection) -> VisionAssessment:
    return VisionAssessment("CONFIRMED")  # type: ignore[arg-type]


def _unhashable_assessment(_: Detection) -> VisionAssessment:
    return VisionAssessment([])  # type: ignore[arg-type]


def _malformed_severity(_: Detection) -> VisionAssessment:
    return VisionAssessment(Verdict.CONFIRMED, "DANGEROUS")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("verifier", "verdict", "reason", "logged"),
    [
        (
            ScriptedVerifier(Verdict.CONFIRMED),
            Verdict.CONFIRMED,
            GateReason.CONFIRMED,
            VisionAssessment(Verdict.CONFIRMED),
        ),
        (
            ScriptedVerifier(Verdict.UNDECIDED),
            Verdict.UNDECIDED,
            GateReason.UNDECIDED,
            VisionAssessment(Verdict.UNDECIDED),
        ),
        (
            ScriptedVerifier(behavior=_sleep_past_deadline),
            Verdict.UNDECIDED,
            GateReason.TIMEOUT,
            None,
        ),
        (
            ScriptedVerifier(behavior=_raise),
            Verdict.UNDECIDED,
            GateReason.VERIFIER_ERROR,
            None,
        ),
        (
            ScriptedVerifier(behavior=_not_a_verdict),
            Verdict.UNDECIDED,
            GateReason.VERIFIER_ERROR,
            None,
        ),
        (
            ScriptedVerifier(behavior=_malformed_assessment),
            Verdict.UNDECIDED,
            GateReason.VERIFIER_ERROR,
            VisionAssessment(Verdict.UNDECIDED),
        ),
        (
            ScriptedVerifier(behavior=_unhashable_assessment),
            Verdict.UNDECIDED,
            GateReason.VERIFIER_ERROR,
            VisionAssessment(Verdict.UNDECIDED),
        ),
        (
            ScriptedVerifier(behavior=_malformed_severity),
            Verdict.UNDECIDED,
            GateReason.VERIFIER_ERROR,
            VisionAssessment(Verdict.UNDECIDED),
        ),
    ],
    ids=[
        "confirmed",
        "undecided",
        "timeout",
        "error",
        "invalid-verdict",
        "malformed-assessment",
        "unhashable-verdict",
        "invalid-severity",
    ],
)
def test_fail_open_outcomes_actuate_with_their_reason(
    harness_factory: Callable[..., Harness],
    verifier: ScriptedVerifier,
    verdict: Verdict,
    reason: GateReason,
    logged: VisionAssessment | None,
) -> None:
    h = harness_factory(verifier, deadline_s=0.05)
    record = h.dispatcher.handle(h.detect(Tag.DANGEROUS))

    assert h.meta.executed == [STRONG_META]
    assert record is not None
    assert record.decision == Decision(
        verdict, reason, True, (Actuation("meta_AA", STRONG_META),), Tag.DANGEROUS, logged
    )


@pytest.mark.parametrize(
    ("behavior", "message"),
    [
        (_unhashable_assessment, "invalid verdict"),
        (_malformed_severity, "invalid severity"),
    ],
    ids=["unhashable-verdict", "invalid-severity"],
)
def test_malformed_assessment_is_logged_and_fails_open_to_undecided(
    harness_factory: Callable[..., Harness],
    behavior: Callable[[Detection], VisionAssessment],
    message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = harness_factory(ScriptedVerifier(behavior=behavior), deadline_s=0.05)

    record = h.dispatcher.handle(h.detect(Tag.DANGEROUS))

    assert record is not None and record.decision is not None
    assert record.decision.verdict is Verdict.UNDECIDED
    assert record.decision.reason is GateReason.VERIFIER_ERROR
    assert record.decision.assessment == VisionAssessment(Verdict.UNDECIDED)
    assert record.decision.severity is Tag.DANGEROUS
    assert message in caplog.text


@pytest.mark.parametrize(
    ("imu_tag", "vision_severity", "expected_meta"),
    [
        (Tag.DANGEROUS, Tag.NON_DANGEROUS, MILD_META),
        (Tag.NON_DANGEROUS, Tag.DANGEROUS, STRONG_META),
        (Tag.NON_DANGEROUS, Tag.NON_DANGEROUS, MILD_META),
        (Tag.DANGEROUS, Tag.DANGEROUS, STRONG_META),
        (Tag.DANGEROUS, None, STRONG_META),
    ],
    ids=[
        "downgrade-dangerous-to-mild",
        "upgrade-mild-to-dangerous",
        "mild-both",
        "strong-both",
        "no-vision-grade",
    ],
)
def test_confirmed_vision_severity_drives_the_actuation(
    harness_factory: Callable[..., Harness],
    imu_tag: Tag,
    vision_severity: Tag | None,
    expected_meta: Mapping[str, object],
) -> None:
    assessment = VisionAssessment(Verdict.CONFIRMED, vision_severity)
    h = harness_factory(AssessmentVerifier(assessment))
    record = h.dispatcher.handle(h.detect(imu_tag))

    assert h.meta.executed == [expected_meta]
    assert record is not None and record.decision is not None
    expected_severity = vision_severity if vision_severity is not None else imu_tag
    assert record.decision.severity is expected_severity
    assert record.decision.assessment is assessment


def test_undecided_assessment_actuates_as_the_imu_tag(
    harness_factory: Callable[..., Harness],
) -> None:
    assessment = VisionAssessment(Verdict.UNDECIDED, Tag.DANGEROUS)
    h = harness_factory(AssessmentVerifier(assessment))
    record = h.dispatcher.handle(h.detect(Tag.NON_DANGEROUS))

    assert h.meta.executed == [MILD_META]
    assert record is not None and record.decision is not None
    assert record.decision.verdict is Verdict.UNDECIDED
    assert record.decision.severity is Tag.NON_DANGEROUS
    assert record.decision.assessment is assessment


def test_rejected_assessment_does_not_actuate(harness_factory: Callable[..., Harness]) -> None:
    assessment = VisionAssessment(Verdict.REJECTED, Tag.DANGEROUS, {"from_episode": 1.0}, "ep-1")
    h = harness_factory(AssessmentVerifier(assessment))
    record = h.dispatcher.handle(h.detect(Tag.DANGEROUS))

    assert h.meta.executed == []
    assert record is not None and record.decision is not None
    assert not record.decision.actuated
    assert record.decision.severity is None
    assert record.decision.assessment is assessment


def test_observer_receives_every_stereotypy_detection(
    harness_factory: Callable[..., Harness],
) -> None:
    verifier = AssessmentVerifier(VisionAssessment(Verdict.UNDECIDED))
    h = harness_factory(verifier)

    actuating = h.detect(Tag.DANGEROUS)
    h.dispatcher.handle(actuating)
    h.clock.advance(1.0)
    inside_cooldown = h.detect(Tag.DANGEROUS)
    assert h.dispatcher.handle(inside_cooldown) is None
    h.clock.advance(1.0)
    h.dispatcher.handle(h.detect(Tag.NON_STEREOTIPY))

    assert h.dispatcher.flush_observer()
    assert verifier.observed == [actuating, inside_cooldown]
    assert verifier.calls == [actuating]
    assert verifier.journal.count("observe") == 2
    assert verifier.journal.count("verify") == 1


def test_observer_receives_rejected_detections(harness_factory: Callable[..., Harness]) -> None:
    verifier = AssessmentVerifier(VisionAssessment(Verdict.REJECTED))
    h = harness_factory(verifier)
    detection = h.detect(Tag.NON_DANGEROUS)
    assert h.dispatcher.handle(detection) is not None

    assert h.dispatcher.flush_observer()
    assert verifier.observed == [detection]
    assert h.meta.executed == []


def test_failing_observer_does_not_block_actuation(
    harness_factory: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    h = harness_factory(FailingObserver(VisionAssessment(Verdict.UNDECIDED)))
    record = h.dispatcher.handle(h.detect(Tag.DANGEROUS))

    assert h.meta.executed == [STRONG_META]
    assert record is not None and record.decision is not None and record.decision.actuated
    assert h.dispatcher.flush_observer()
    assert "Verifier observer failed" in caplog.text


class HungObserver(AssessmentVerifier):
    """Observe blocks until released, to prove handle and stop never wait on it."""

    def __init__(self, result: VisionAssessment | Verdict) -> None:
        super().__init__(result)
        self.entered = threading.Event()
        self.release = threading.Event()

    def observe(self, detection: Detection) -> None:
        self.observed.append(detection)
        self.entered.set()
        self.release.wait(timeout=5.0)

    def close(self) -> None:
        self.release.set()


def test_hung_observer_does_not_block_actuation_or_stop(
    harness_factory: Callable[..., Harness],
) -> None:
    verifier = HungObserver(VisionAssessment(Verdict.UNDECIDED))
    h = harness_factory(verifier, deadline_s=0.05)
    try:
        started = time.monotonic()
        record = h.dispatcher.handle(h.detect(Tag.DANGEROUS))
        elapsed = time.monotonic() - started

        assert record is not None and record.decision is not None and record.decision.actuated
        assert h.meta.executed == [STRONG_META]
        assert elapsed < 1.0
        assert verifier.entered.wait(timeout=5.0)

        started = time.monotonic()
        h.dispatcher.stop()
        assert time.monotonic() - started < 3.0
    finally:
        verifier.close()


def test_full_observer_queue_drops_and_logs(
    harness_factory: Callable[..., Harness], caplog: pytest.LogCaptureFixture
) -> None:
    verifier = HungObserver(VisionAssessment(Verdict.UNDECIDED))
    h = harness_factory(verifier, deadline_s=0.05)
    try:
        h.dispatcher.handle(h.detect(Tag.DANGEROUS))
        assert verifier.entered.wait(timeout=5.0)
        for _ in range(70):
            h.dispatcher.handle(h.detect(Tag.DANGEROUS))
    finally:
        verifier.close()

    assert h.dispatcher.flush_observer(timeout=10.0)
    assert len(verifier.observed) == 65
    assert "Observer queue full" in caplog.text


def test_verifier_without_observer_still_dispatches(
    harness_factory: Callable[..., Harness],
) -> None:
    h = harness_factory(ScriptedVerifier(Verdict.UNDECIDED))
    assert not isinstance(h.verifier, DetectionObserver)
    assert h.dispatcher.handle(h.detect(Tag.DANGEROUS)) is not None
    assert h.meta.executed == [STRONG_META]


def test_cooldown_and_tag_changes(harness_factory: Callable[..., Harness]) -> None:
    h = harness_factory(ScriptedVerifier(Verdict.UNDECIDED))
    handle = h.dispatcher.handle

    assert handle(h.detect(Tag.DANGEROUS)) is not None
    h.clock.advance(4.9)
    assert handle(h.detect(Tag.DANGEROUS)) is None  # same tag inside cooldown
    h.clock.advance(0.1)
    assert handle(h.detect(Tag.DANGEROUS)) is not None  # cooldown elapsed
    h.clock.advance(0.1)
    changed = handle(h.detect(Tag.NON_DANGEROUS))  # tag change actuates immediately
    h.clock.advance(0.1)
    calm = handle(h.detect(Tag.NON_STEREOTIPY))

    assert h.meta.executed == [STRONG_META, STRONG_META, MILD_META]
    assert changed is not None and changed.tag_changed and changed.decision is not None
    assert calm is not None and calm.tag_changed and calm.decision is None
    assert [r.tag for r in h.sink.records] == [
        Tag.DANGEROUS,
        Tag.DANGEROUS,
        Tag.NON_DANGEROUS,
        Tag.NON_STEREOTIPY,
    ]


def test_rejection_does_not_start_the_cooldown(harness_factory: Callable[..., Harness]) -> None:
    verdicts = iter([Verdict.REJECTED, Verdict.CONFIRMED])
    h = harness_factory(ScriptedVerifier(behavior=lambda _: next(verdicts)))
    h.dispatcher.handle(h.detect(Tag.DANGEROUS))
    h.clock.advance(1.0)
    h.dispatcher.handle(h.detect(Tag.DANGEROUS))
    assert h.meta.executed == [STRONG_META]


def test_without_actuators_detections_are_still_verified_and_logged(
    harness_factory: Callable[..., Harness],
) -> None:
    h = harness_factory(ScriptedVerifier(Verdict.CONFIRMED), actuators=[])
    record = h.dispatcher.handle(h.detect(Tag.DANGEROUS))
    assert len(h.verifier.calls) == 1
    assert record is not None and record.decision == Decision(
        Verdict.CONFIRMED,
        GateReason.CONFIRMED,
        False,
        (),
        None,
        VisionAssessment(Verdict.CONFIRMED),
    )
    assert h.sink.records == [record]


class FailingActuator(RecordingActuator):
    def execute(self, params: Mapping[str, object]) -> None:
        raise OSError("device gone")


def test_failing_actuator_does_not_prevent_the_others(
    harness_factory: Callable[..., Harness],
) -> None:
    led = RecordingActuator("led_10.0.0.2", ActuatorKind.LED_STRIP)
    actuators = [FailingActuator("meta_AA"), led]
    h = harness_factory(ScriptedVerifier(Verdict.UNDECIDED), actuators=actuators)
    record = h.dispatcher.handle(h.detect(Tag.DANGEROUS))
    assert len(led.executed) == 1
    assert record is not None and record.decision is not None and record.decision.actuated


def test_blocked_verifier_never_blocks_producers_and_newest_detection_wins(
    harness_factory: Callable[..., Harness],
) -> None:
    entered, release = threading.Event(), threading.Event()

    def gated(_: Detection) -> Verdict:
        entered.set()
        release.wait(timeout=5)
        return Verdict.CONFIRMED

    h = harness_factory(ScriptedVerifier(behavior=gated), deadline_s=5.0)
    h.dispatcher.start()
    try:
        h.slot.put(h.detect(Tag.DANGEROUS, end_ts=1.0))
        assert entered.wait(timeout=5)

        started = time.monotonic()
        h.slot.put(h.detect(Tag.NON_DANGEROUS, end_ts=2.0))
        h.slot.put(h.detect(Tag.NON_DANGEROUS, end_ts=3.0))
        assert time.monotonic() - started < 0.1
    finally:
        release.set()

    deadline = time.monotonic() + 5
    while len(h.sink.records) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [d.window_end_ts for d in h.verifier.calls] == [1.0, 3.0]
    assert [r.window_end_ts for r in h.sink.records] == [1.0, 3.0]


def test_stub_verifier_reproduces_imu_only_actuation(
    harness_factory: Callable[..., Harness],
) -> None:
    verifier = StubVerifier()
    assert not isinstance(verifier, DetectionObserver)
    h = harness_factory(verifier)
    # (seconds since previous detection, tag); IMU-only rule: actuate stereotypy tags
    # on a tag change or once the 5 s cooldown has elapsed.
    sequence = [
        (0.0, Tag.DANGEROUS),  # actuate strong
        (1.0, Tag.DANGEROUS),  # cooldown
        (4.0, Tag.DANGEROUS),  # cooldown elapsed: strong
        (0.5, Tag.NON_DANGEROUS),  # tag change: mild
        (0.5, Tag.NON_STEREOTIPY),  # not a stereotypy
        (1.0, Tag.NON_DANGEROUS),  # tag change: mild
        (1.0, Tag.NO_CLASS),  # not a stereotypy
    ]
    for elapsed, tag in sequence:
        h.clock.advance(elapsed)
        h.dispatcher.handle(h.detect(tag))

    assert h.meta.executed == [STRONG_META, STRONG_META, MILD_META, MILD_META]
    decisions = [r.decision for r in h.sink.records if r.decision is not None]
    assert [d.severity for d in decisions] == [
        Tag.DANGEROUS,
        Tag.DANGEROUS,
        Tag.NON_DANGEROUS,
        Tag.NON_DANGEROUS,
    ]
    assert all(d.assessment == VisionAssessment(Verdict.UNDECIDED) for d in decisions)


HUNG_VERIFIER_PROCESS = """
import sys, threading
from cpsa.application.dispatcher import Dispatcher
from cpsa.application.slot import LatestSlot
from cpsa.board.__main__ import install_signal_handlers
from cpsa.domain.model import Detection, Tag
from tests.fakes import FakeClock, MemorySink, RecordingGui, ScriptedVerifier

entered = threading.Event()
def hang(_):
    entered.set()
    threading.Event().wait()  # never returns

clock = FakeClock()
slot = LatestSlot()
dispatcher = Dispatcher(slot, ScriptedVerifier(behavior=hang), [], MemorySink(), RecordingGui(),
                        clock, cooldown_s=5.0, deadline_s=0.05)
stop = threading.Event()
install_signal_handlers(stop)
dispatcher.start()
slot.put(Detection(Tag.DANGEROUS, clock.now(), 0.0, ()))
entered.wait(5)
print("ready", flush=True)
stop.wait()
dispatcher.stop()
sys.exit(0)
"""


def test_hung_verifier_does_not_prevent_process_exit_on_sigterm() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(
        [sys.executable, "-c", HUNG_VERIFIER_PROCESS],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
