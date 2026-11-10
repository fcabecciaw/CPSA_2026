"""Component graph of the running system and its start/stop ordering."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

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
    SensorSource,
    Verifier,
)

logger = logging.getLogger(__name__)


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

    def start(self) -> None:
        try:
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
        self._synchronizer.update(side, kind, values, ts)
        if self._vision is not None:
            self._vision.on_imu_sample(side, kind, values, ts)

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
