import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence

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
)
from tests.fakes import FakeClock, MemorySink, RecordingActuator, RecordingGui, ScriptedVerifier

STRONG_META = {"duty": 100, "duration": 900}
MILD_META = {"duty": 40, "duration": 700}


class Harness:
    def __init__(
        self,
        verifier: ScriptedVerifier,
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
    assert record == DecisionRecord(
        timestamp=h.clock.now(),
        window_end_ts=42.0,
        tag=Tag.DANGEROUS,
        tag_changed=True,
        decision=Decision(Verdict.REJECTED, GateReason.REJECTED, False, ()),
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


@pytest.mark.parametrize(
    ("verifier", "verdict", "reason"),
    [
        (ScriptedVerifier(Verdict.CONFIRMED), Verdict.CONFIRMED, GateReason.CONFIRMED),
        (ScriptedVerifier(Verdict.UNDECIDED), Verdict.UNDECIDED, GateReason.UNDECIDED),
        (ScriptedVerifier(behavior=_sleep_past_deadline), Verdict.UNDECIDED, GateReason.TIMEOUT),
        (ScriptedVerifier(behavior=_raise), Verdict.UNDECIDED, GateReason.VERIFIER_ERROR),
        (ScriptedVerifier(behavior=_not_a_verdict), Verdict.UNDECIDED, GateReason.VERIFIER_ERROR),
    ],
    ids=["confirmed", "undecided", "timeout", "error", "invalid-verdict"],
)
def test_fail_open_outcomes_actuate_with_their_reason(
    harness_factory: Callable[..., Harness],
    verifier: ScriptedVerifier,
    verdict: Verdict,
    reason: GateReason,
) -> None:
    h = harness_factory(verifier, deadline_s=0.05)
    record = h.dispatcher.handle(h.detect(Tag.DANGEROUS))

    assert h.meta.executed == [STRONG_META]
    assert record is not None
    assert record.decision == Decision(verdict, reason, True, (Actuation("meta_AA", STRONG_META),))


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
        Verdict.CONFIRMED, GateReason.CONFIRMED, False, ()
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
    h = harness_factory(StubVerifier())  # type: ignore[arg-type]
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
