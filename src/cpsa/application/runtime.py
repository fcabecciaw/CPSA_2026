"""Component graph of the running system and its start/stop ordering."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Literal

from cpsa.application.classification import ClassificationStage
from cpsa.application.config import Config
from cpsa.application.dispatcher import Dispatcher
from cpsa.application.slot import LatestSlot
from cpsa.application.vision_verifier import VisionVerifier
from cpsa.domain.model import Detection, SampleKind, Side
from cpsa.imu.sync import IMUSynchronizer
from cpsa.imu.window import Windower
from cpsa.ports import (
    Actuator,
    Classifier,
    Clock,
    EventSink,
    FeatureExtractor,
    Gui,
    SampleCallback,
    SensorSource,
    Verifier,
)

logger = logging.getLogger(__name__)

#: Both wrists must have delivered a sample within this window to be ``connected`` (F3).
_IMU_STALE_S = 1.0


class NullSensorSource:
    """A ``SensorSource`` with no devices and no I/O.

    The camera-only medical mode (FR-026) has no IMU: this source scans for
    nothing, accepts the sample callback and never calls it.
    """

    def discover(self, timeout_s: float) -> set[str]:
        """Return no device names."""
        return set()

    def start(self, on_sample: SampleCallback) -> None:
        """Accept the callback and stay silent."""

    def stop(self) -> None:
        """Nothing to close."""


class Runtime:
    """Starts and stops the pipeline components in a fixed order.

    Start: GUI, dispatcher, vision verifier, then sensors (so no sample arrives
    before its consumers). Stop: GUI, dispatcher, sensors, vision verifier,
    actuators; a failing stop is logged and the remaining components are still
    stopped.
    """

    def __init__(
        self,
        source: SensorSource,
        synchronizer: IMUSynchronizer,
        dispatcher: Dispatcher,
        actuators: Sequence[Actuator],
        gui: Gui,
        vision: VisionVerifier | None = None,
    ) -> None:
        self._source = source
        self._synchronizer = synchronizer
        self._dispatcher = dispatcher
        self._actuators = list(actuators)
        self._gui = gui
        self._vision = vision
        self._imu_rx_lock = threading.Lock()
        self._imu_last_rx: dict[Side, float] = {}

    def start(self) -> None:
        try:
            with self._imu_rx_lock:
                self._imu_last_rx.clear()
            self._gui.start()
            self._dispatcher.start()
            if self._vision is not None:
                self._vision.start()
            self._source.start(self._on_sample)
        except Exception:
            self.stop()
            raise
        logger.info("Runtime started with %d actuator(s)", len(self._actuators))

    def _on_sample(
        self, side: Side, kind: SampleKind, values: tuple[float, ...], ts: float
    ) -> None:
        """Tee sensor samples to the IMU synchronizer and the vision episode feed."""
        with self._imu_rx_lock:
            self._imu_last_rx[side] = ts
        self._synchronizer.update(side, kind, values, ts)
        if self._vision is not None:
            self._vision.on_imu_sample(side, kind, values, ts)

    def imu_status(self, now: float) -> Literal["connected", "lost"]:
        """Whether both wrists delivered a sample within the last second (F3).

        Fed by :meth:`_on_sample` with the sample's monotonic timestamp, so the
        status follows the sensor stream even when no wall-clock time passes
        between two reads.
        """
        with self._imu_rx_lock:
            left = self._imu_last_rx.get(Side.LEFT)
            right = self._imu_last_rx.get(Side.RIGHT)
        if left is None or right is None:
            return "lost"
        if now - left > _IMU_STALE_S or now - right > _IMU_STALE_S:
            return "lost"
        return "connected"

    def stop(self) -> None:
        steps: list[tuple[str, Callable[[], None]]] = [
            ("GUI", self._gui.stop),
            ("dispatcher", self._dispatcher.stop),
            ("sensors", self._source.stop),
        ]
        if self._vision is not None:
            steps.append(("vision verifier", self._vision.stop))
        steps += [(f"actuator {a.id}", a.stop) for a in self._actuators]
        for name, stop in steps:
            try:
                stop()
            except Exception:
                logger.exception("Failed to stop %s", name)
        logger.info("Runtime stopped")


def build_runtime(
    config: Config,
    *,
    source: SensorSource,
    extractor: FeatureExtractor,
    classifier: Classifier,
    verifier: Verifier,
    actuators: Sequence[Actuator],
    sink: EventSink,
    gui: Gui,
    clock: Clock,
) -> Runtime:
    """Wire sensors -> synchronizer -> windower -> classification -> dispatcher."""
    slot: LatestSlot[Detection] = LatestSlot()
    stage = ClassificationStage(extractor, classifier, slot, clock)
    windower = Windower(stage.on_window)
    synchronizer = IMUSynchronizer(
        max_skew_s=config.sync.max_skew_ms / 1000.0,
        stale_s=config.sync.stale_ms / 1000.0,
        on_row=windower.add,
        clock=clock,
    )
    dispatcher = Dispatcher(
        slot,
        verifier,
        actuators,
        sink,
        gui,
        clock,
        cooldown_s=config.dispatch.cooldown_s,
        deadline_s=config.dispatch.verification_deadline_s,
    )
    vision = verifier if isinstance(verifier, VisionVerifier) else None
    return Runtime(source, synchronizer, dispatcher, actuators, gui, vision)


def build_vision_runtime(
    config: Config,
    *,
    verifier: VisionVerifier,
    actuators: Sequence[Actuator],
    sink: EventSink,
    gui: Gui,
    clock: Clock,
) -> tuple[Runtime, Callable[[Detection], None]]:
    """Wire the camera-only pipeline (FR-026, F10).

    The verifier runs continuously and turns each trigger fire into a
    ``Detection``; the returned callable puts it into the dispatcher's slot,
    where the IMU classification stage puts its detections in ``build_runtime``.
    The source is a no-op: vision mode has no IMU, and the synchronizer exists
    only because ``Runtime`` owns one start/stop order.
    """
    slot: LatestSlot[Detection] = LatestSlot()
    dispatcher = Dispatcher(
        slot,
        verifier,
        actuators,
        sink,
        gui,
        clock,
        cooldown_s=config.dispatch.cooldown_s,
        deadline_s=config.dispatch.verification_deadline_s,
    )
    synchronizer = IMUSynchronizer(
        max_skew_s=config.sync.max_skew_ms / 1000.0,
        stale_s=config.sync.stale_ms / 1000.0,
        on_row=_discard_row,
        clock=clock,
    )
    runtime = Runtime(NullSensorSource(), synchronizer, dispatcher, actuators, gui, verifier)
    return runtime, slot.put


def _discard_row(*_row: object) -> None:
    """Sink for the synchronizer that never receives a sample in vision mode."""
