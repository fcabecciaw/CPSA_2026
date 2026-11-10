"""Verification-gated actuation of detections."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from cpsa.application.slot import LatestSlot
from cpsa.domain import gating, policy
from cpsa.domain.model import (
    Decision,
    DecisionRecord,
    Detection,
    GateReason,
    Tag,
    Verdict,
    VisionAssessment,
)
from cpsa.ports import Actuator, Clock, DetectionObserver, EventSink, Gui, Verifier

logger = logging.getLogger(__name__)

_REASON_FOR_VERDICT = {
    Verdict.CONFIRMED: GateReason.CONFIRMED,
    Verdict.UNDECIDED: GateReason.UNDECIDED,
    Verdict.REJECTED: GateReason.REJECTED,
}

_OBSERVER_QUEUE_MAX = 64
_OBSERVER_JOIN_TIMEOUT_S = 1.0


@dataclass
class _Request:
    detection: Detection
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: BaseException | None = None
    abandoned: bool = False


@dataclass(frozen=True)
class _Verification:
    """One verifier outcome after deadline and error handling.

    ``assessment`` is ``None`` on timeout or verifier error, where nothing usable
    came back and the gate fails open.
    """

    verdict: Verdict
    reason: GateReason
    assessment: VisionAssessment | None


class _VerificationWorker:
    """Runs verifier calls one at a time on a daemon thread.

    The caller waits for each result only up to its deadline. A daemon thread (not a
    ``ThreadPoolExecutor``, whose workers are joined at interpreter exit) is used so a
    verifier that never returns cannot keep the process alive after shutdown. While
    the verifier hangs, later requests time out and fail open; an abandoned request
    still waiting is replaced by the newest one, so requests never pile up.
    """

    def __init__(self, verifier: Verifier) -> None:
        self._verifier = verifier
        self._requests: LatestSlot[_Request] = LatestSlot()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="verifier", daemon=True)
        self._start_lock = threading.Lock()

    def submit(self, detection: Detection) -> _Request:
        with self._start_lock:
            if not self._thread.is_alive() and not self._stop.is_set():
                self._thread.start()
        request = _Request(detection)
        self._requests.put(request)
        return request

    def stop(self) -> None:
        """Ask the worker to exit; a call in progress is not waited for."""
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            request = self._requests.get(timeout=0.5)
            if request is None or request.abandoned:
                continue
            try:
                request.result = self._verifier.verify(request.detection)
            except BaseException as exc:
                request.error = exc
            request.done.set()


class _ObservationWorker:
    """Deliver ``DetectionObserver.observe`` calls on one bounded daemon thread.

    ``Dispatcher.handle`` never blocks on an observer: a detection is appended to
    a bounded queue and the thread delivers it. When the queue is full the newest
    observation is dropped and logged. The thread is a daemon and shutdown joins
    it only with a timeout, so a hung observer cannot keep the process alive or
    hold up actuation.
    """

    def __init__(self, observer: DetectionObserver, queue_max: int = _OBSERVER_QUEUE_MAX) -> None:
        self._observer = observer
        self._queue_max = queue_max
        self._queue: deque[Detection] = deque()
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, name="observer", daemon=True)
        self._start_lock = threading.Lock()

    def submit(self, detection: Detection) -> None:
        """Queue one observation without waiting for the observer."""
        with self._start_lock:
            if not self._thread.is_alive() and not self._stop.is_set():
                self._thread.start()
        with self._cond:
            if len(self._queue) >= self._queue_max:
                self._dropped += 1
                logger.warning(
                    "Observer queue full (%d detections); dropping observation of %s",
                    self._queue_max,
                    detection.tag.name,
                )
                return
            self._queue.append(detection)
            self._submitted += 1
            self._cond.notify_all()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every accepted observation was delivered; for tests and shutdown checks."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._processed < self._submitted:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._cond.wait(remaining)
        return True

    def stop(self) -> None:
        """Stop the thread; a hung observer is abandoned after the join timeout."""
        with self._cond:
            self._stop.set()
            self._cond.notify_all()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=_OBSERVER_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    "Observer thread did not stop within %.1f s; abandoning it",
                    _OBSERVER_JOIN_TIMEOUT_S,
                )

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                if not self._queue:
                    self._cond.wait(timeout=0.5)
                if not self._queue:
                    continue
                detection = self._queue.popleft()
            try:
                self._observer.observe(detection)
            except Exception:
                logger.exception("Verifier observer failed for detection %s", detection.tag.name)
            with self._cond:
                self._processed += 1
                self._cond.notify_all()


class Dispatcher:
    """Consume detections, verify them under a deadline, actuate, and record events.

    For each detection: a tag change resets the cooldown; a stereotypy tag outside
    the cooldown is sent to the verifier; unless the verifier rejects it in time,
    the policy's actuations are executed (fail-open on timeout or verifier error).
    A confirmed vision assessment may upgrade or downgrade the IMU tag. Every
    stereotypy detection is also offered to a verifier that observes detections,
    including those inside the cooldown. Every tag change and every decision is
    written to the sink and published to the GUI.
    """

    def __init__(
        self,
        slot: LatestSlot[Detection],
        verifier: Verifier,
        actuators: Sequence[Actuator],
        sink: EventSink,
        gui: Gui,
        clock: Clock,
        cooldown_s: float,
        deadline_s: float,
    ) -> None:
        self._slot = slot
        self._verifier = verifier
        self._observer: DetectionObserver | None = (
            verifier if isinstance(verifier, DetectionObserver) else None
        )
        self._observer_worker = (
            None if self._observer is None else _ObservationWorker(self._observer)
        )
        self._actuators = {actuator.id: actuator for actuator in actuators}
        self._sink = sink
        self._gui = gui
        self._clock = clock
        self._cooldown_s = cooldown_s
        self._deadline_s = deadline_s
        self._last_tag: Tag | None = None
        self._last_actuation: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dispatcher", daemon=True)
        self._worker = _VerificationWorker(verifier)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join()
        self._worker.stop()
        if self._observer_worker is not None:
            self._observer_worker.stop()

    def flush_observer(self, timeout: float = 5.0) -> bool:
        """Wait until every queued observation was delivered; for tests and shutdown checks."""
        worker = self._observer_worker
        return True if worker is None else worker.flush(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            detection = self._slot.get(timeout=0.5)
            if detection is None:
                continue
            try:
                self.handle(detection)
            except Exception:
                logger.exception("Failed to dispatch detection %s", detection.tag.name)

    def handle(self, detection: Detection) -> DecisionRecord | None:
        """Process one detection; return the record written, if any."""
        now = self._clock.monotonic()
        tag_changed = detection.tag != self._last_tag
        if tag_changed:
            logger.info("Tag changed: %s -> %s", _name(self._last_tag), detection.tag.name)
            self._last_tag = detection.tag
            self._last_actuation = None

        if detection.tag.is_stereotypy:
            self._observe(detection)

        decision = None
        if gating.should_evaluate(detection.tag, self._last_actuation, now, self._cooldown_s):
            decision = self._decide(detection)
            if decision.reason is not GateReason.REJECTED:
                self._last_actuation = now

        if not (tag_changed or decision):
            return None
        record = DecisionRecord(
            timestamp=self._clock.now(),
            window_end_ts=detection.window_end_ts,
            tag=detection.tag,
            tag_changed=tag_changed,
            decision=decision,
        )
        self._publish(record)
        return record

    def _observe(self, detection: Detection) -> None:
        """Offer a stereotypy detection to the verifier's episode, if it has one.

        Delivery is asynchronous: a hung or slow observer never delays the
        verification, actuation or publishing path of ``handle``.
        """
        worker = self._observer_worker
        if worker is None:
            return
        try:
            worker.submit(detection)
        except Exception:
            logger.exception("Verifier observer failed for detection %s", detection.tag.name)

    def _decide(self, detection: Detection) -> Decision:
        verified = self._verify_with_deadline(detection)
        if not gating.allows_actuation(verified.verdict):
            logger.info("Detection %s rejected by the verifier", detection.tag.name)
            return Decision(verified.verdict, verified.reason, False, (), None, verified.assessment)

        severity = policy.effective_severity(detection.tag, verified.assessment)
        actuations = policy.actuations_for(severity, list(self._actuators))
        for actuation in actuations:
            try:
                self._actuators[actuation.actuator_id].execute(actuation.params)
            except Exception:
                logger.exception("Actuator %s failed", actuation.actuator_id)
        logger.info(
            "Detection %s: verdict %s (%s), severity %s, %d actuation(s)",
            detection.tag.name,
            verified.verdict.name,
            verified.reason.name,
            severity.name,
            len(actuations),
        )
        return Decision(
            verified.verdict,
            verified.reason,
            bool(actuations),
            tuple(actuations),
            severity if actuations else None,
            verified.assessment,
        )

    def _verify_with_deadline(self, detection: Detection) -> _Verification:
        request = self._worker.submit(detection)
        if not request.done.wait(timeout=self._deadline_s):
            request.abandoned = True
            logger.warning("Verifier exceeded the %.3f s deadline; failing open", self._deadline_s)
            return _Verification(Verdict.UNDECIDED, GateReason.TIMEOUT, None)
        if request.error is not None:
            logger.error("Verifier failed; failing open", exc_info=request.error)
            return _Verification(Verdict.UNDECIDED, GateReason.VERIFIER_ERROR, None)
        assessment = gating.as_assessment(request.result)
        if assessment is None:
            logger.error(
                "Verifier returned %r, not a verdict or assessment; failing open", request.result
            )
            return _Verification(Verdict.UNDECIDED, GateReason.VERIFIER_ERROR, None)
        normalized = _validated_assessment(assessment)
        if normalized is not assessment:
            return _Verification(Verdict.UNDECIDED, GateReason.VERIFIER_ERROR, normalized)
        reason = _REASON_FOR_VERDICT.get(assessment.verdict)
        if reason is None:
            logger.error(
                "Verifier returned %r with an unknown verdict; failing open", request.result
            )
            return _Verification(Verdict.UNDECIDED, GateReason.VERIFIER_ERROR, None)
        return _Verification(assessment.verdict, reason, assessment)

    def _publish(self, record: DecisionRecord) -> None:
        try:
            self._sink.write(record)
        except Exception:
            logger.exception("Event sink failed to write a record")
        try:
            self._gui.publish(record)
        except Exception:
            logger.exception("GUI failed to publish a record")


def _validated_assessment(assessment: VisionAssessment) -> VisionAssessment:
    """Return ``assessment`` or a safe ``UNDECIDED`` copy when its types are invalid.

    A malformed verifier result must never raise on the path to actuation. An
    invalid verdict or severity is logged and normalized to ``UNDECIDED`` with no
    severity, which makes the gate use the IMU tag.
    """
    safe = VisionAssessment(Verdict.UNDECIDED, None, assessment.features, assessment.episode_id)
    if not isinstance(assessment.verdict, Verdict):
        logger.error(
            "Verifier assessment has an invalid verdict %r; treating it as UNDECIDED",
            assessment.verdict,
        )
        return safe
    if assessment.severity is not None and not isinstance(assessment.severity, Tag):
        logger.error(
            "Verifier assessment has an invalid severity %r; treating it as UNDECIDED",
            assessment.severity,
        )
        return safe
    return assessment


def _name(tag: Tag | None) -> str:
    return tag.name if tag is not None else "None"
