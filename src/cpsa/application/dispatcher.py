"""Verification-gated actuation of detections."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from cpsa.application.slot import LatestSlot
from cpsa.domain import gating, policy
from cpsa.domain.model import Decision, DecisionRecord, Detection, GateReason, Tag, Verdict
from cpsa.ports import Actuator, Clock, EventSink, Gui, Verifier

logger = logging.getLogger(__name__)

_REASON_FOR_VERDICT = {
    Verdict.CONFIRMED: GateReason.CONFIRMED,
    Verdict.UNDECIDED: GateReason.UNDECIDED,
    Verdict.REJECTED: GateReason.REJECTED,
}


@dataclass
class _Request:
    detection: Detection
    done: threading.Event = field(default_factory=threading.Event)
    verdict: object = None
    error: BaseException | None = None
    abandoned: bool = False


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
                request.verdict = self._verifier.verify(request.detection)
            except BaseException as exc:
                request.error = exc
            request.done.set()


class Dispatcher:
    """Consume detections, verify them under a deadline, actuate, and record events.

    For each detection: a tag change resets the cooldown; a stereotypy tag outside
    the cooldown is sent to the verifier; unless the verifier rejects it in time,
    the policy's actuations are executed (fail-open on timeout or verifier error).
    Every tag change and every decision is written to the sink and published to
    the GUI.
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
        if self._thread.is_alive():
            self._thread.join()
        self._worker.stop()

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

    def _decide(self, detection: Detection) -> Decision:
        verdict, reason = self._verify_with_deadline(detection)
        if not gating.allows_actuation(verdict):
            logger.info("Detection %s rejected by the verifier", detection.tag.name)
            return Decision(verdict, GateReason.REJECTED, False, ())

        actuations = policy.actuations_for(detection.tag, list(self._actuators))
        for actuation in actuations:
            try:
                self._actuators[actuation.actuator_id].execute(actuation.params)
            except Exception:
                logger.exception("Actuator %s failed", actuation.actuator_id)
        logger.info(
            "Detection %s: verdict %s (%s), %d actuation(s)",
            detection.tag.name,
            verdict.name,
            reason.name,
            len(actuations),
        )
        return Decision(verdict, reason, bool(actuations), tuple(actuations))

    def _verify_with_deadline(self, detection: Detection) -> tuple[Verdict, GateReason]:
        request = self._worker.submit(detection)
        if not request.done.wait(timeout=self._deadline_s):
            request.abandoned = True
            logger.warning("Verifier exceeded the %.3f s deadline; failing open", self._deadline_s)
            return Verdict.UNDECIDED, GateReason.TIMEOUT
        if request.error is not None:
            logger.error("Verifier failed; failing open", exc_info=request.error)
            return Verdict.UNDECIDED, GateReason.VERIFIER_ERROR
        verdict = request.verdict
        if not isinstance(verdict, Verdict):
            logger.error("Verifier returned %r, not a Verdict; failing open", verdict)
            return Verdict.UNDECIDED, GateReason.VERIFIER_ERROR
        return verdict, _REASON_FOR_VERDICT[verdict]

    def _publish(self, record: DecisionRecord) -> None:
        try:
            self._sink.write(record)
        except Exception:
            logger.exception("Event sink failed to write a record")
        try:
            self._gui.publish(record)
        except Exception:
            logger.exception("GUI failed to publish a record")


def _name(tag: Tag | None) -> str:
    return tag.name if tag is not None else "None"
