"""``cpsa-board --config PATH``: run the pipeline on the board until SIGINT/SIGTERM.

Exit codes: 0 after an orderly stop, 1 on an unexpected error, 2 when the
configuration is invalid or asks for the camera-only medical trigger, the
``board`` or ``vision`` extra is not installed, a pose model is missing, a
BlueCoin is missing, or the native libraries cannot be loaded.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cpsa.adapters.actuators.factory import start_actuators
from cpsa.adapters.clock import SystemClock
from cpsa.adapters.event_log import JsonlEventSink, configure_logging
from cpsa.adapters.gui_null import NullGui
from cpsa.adapters.native.classifier import (
    NativeClassifier,
    NativeFeatureExtractor,
    NativeLibraryError,
)
from cpsa.adapters.verifier_stub import StubVerifier
from cpsa.application.config import BlueCoinsConfig, Config, ConfigError, VisionConfig, load
from cpsa.application.runtime import NullSensorSource, build_runtime
from cpsa.domain.model import Detection
from cpsa.ports import (
    Actuator,
    Classifier,
    Clock,
    EventSink,
    FeatureExtractor,
    Gui,
    PoseEstimator,
    SensorSource,
    Verifier,
    VisionObserver,
)
from cpsa.vision.trigger import VisionTrigger
from cpsa.vision.types import KeypointLayout

logger = logging.getLogger("cpsa.board")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STARTUP = 2


class VisionStartupError(RuntimeError):
    """The vision dependencies or pose model cannot be built from the configuration."""


@dataclass(frozen=True)
class BoardAdapters:
    """Adapters used by :func:`run`. Hardware is touched only by the callables."""

    source: SensorSource
    load_classifier: Callable[[], tuple[FeatureExtractor, Classifier]]
    start_actuators: Callable[[], Sequence[Actuator]]
    verifier: Verifier
    gui: Gui
    sink: EventSink
    clock: Clock


def _load_native() -> tuple[FeatureExtractor, Classifier]:
    # Extractor first, then classifier: the baseline initialization order.
    return NativeFeatureExtractor(), NativeClassifier()


def _build_pose_estimator(settings: VisionConfig, layout: KeypointLayout) -> PoseEstimator:
    """Select the DPU or ONNX pose adapter from ``vision.model.kind``."""
    model = settings.model
    if model.kind == "onnx":
        from cpsa.adapters.pose.onnx import OnnxPoseEstimator

        return OnnxPoseEstimator(
            model.path,
            layout,
            (model.input_height, model.input_width),
            conf_thr=model.conf_thr,
            iou_thr=model.iou_thr,
        )
    from cpsa.adapters.pose.dpu import DpuPoseEstimator

    return DpuPoseEstimator(model.path, layout, conf_thr=model.conf_thr, iou_thr=model.iou_thr)


def _build_verifier(
    config: Config,
    clock: Clock,
    *,
    observer: VisionObserver | None = None,
    continuous: bool = False,
    trigger: VisionTrigger | None = None,
    on_detection: Callable[[Detection], None] | None = None,
) -> Verifier:
    """Return the vision verifier when enabled, else the stub.

    Imports stay local so the runner remains importable without the camera,
    OpenCV and model-runtime dependencies of the vision pipeline. The F8
    continuous-mode keywords are passed only when ``continuous`` is set, so the
    IMU construction path stays exactly as it was (AC-026).
    """
    if not config.vision.enabled:
        return StubVerifier()

    from cpsa.adapters.camera.uvc import UvcVideoSource
    from cpsa.adapters.evidence import FileEvidenceSink
    from cpsa.application.vision_verifier import VisionVerifier
    from cpsa.vision.layouts import get_layout

    settings = config.vision
    layout = get_layout(settings.layout)
    camera = settings.camera
    evidence = (
        FileEvidenceSink(settings.evidence.base_path, settings.evidence.retention_days, layout)
        if settings.evidence.enabled
        else None
    )
    try:
        source = UvcVideoSource(
            camera.device,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
            exposure=camera.exposure,
            gain=camera.gain,
            power_line_hz=camera.power_line_hz,
            match=camera.match,
        )
        pose = _build_pose_estimator(settings, layout)
        if continuous:
            return VisionVerifier(
                source,
                pose,
                evidence,
                settings,
                clock,
                layout,
                log_dir=config.log.base_path,
                observer=observer,
                continuous=True,
                trigger=trigger,
                on_detection=on_detection,
            )
        return VisionVerifier(
            source,
            pose,
            evidence,
            settings,
            clock,
            layout,
            log_dir=config.log.base_path,
            observer=observer,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise VisionStartupError(str(exc)) from exc


def board_adapters(
    config: Config,
    *,
    gui: Gui | None = None,
    observer: VisionObserver | None = None,
    clock: Clock | None = None,
    continuous: bool = False,
    trigger: VisionTrigger | None = None,
    on_detection: Callable[[Detection], None] | None = None,
) -> BoardAdapters:
    """Construct the real adapters. The single place where the verifier and GUI are chosen.

    ``gui``, ``observer`` and ``clock`` are injection seams for the supervisor's
    medical mode; the F8 keywords reach the verifier only in continuous mode.
    With ``medical.trigger = vision`` the IMU source is a no-op and the BLE stack
    is never imported (FR-026); the defaults keep ``cpsa-board`` identical.
    """
    shared_clock = SystemClock() if clock is None else clock
    source: SensorSource
    if config.medical.trigger == "vision":
        source = NullSensorSource()
    else:
        # Imported here so the runner stays importable (and testable) without the
        # board-only BLE libraries.
        from cpsa.adapters.bluecoin.source import BlueCoinSource

        source = BlueCoinSource(config.bluecoins.left, config.bluecoins.right)
    return BoardAdapters(
        source=source,
        load_classifier=_load_native,
        start_actuators=lambda: start_actuators(config.actuators),
        verifier=_build_verifier(
            config,
            shared_clock,
            observer=observer,
            continuous=continuous,
            trigger=trigger,
            on_detection=on_detection,
        ),
        gui=NullGui() if gui is None else gui,
        sink=JsonlEventSink(config.log.base_path),
        clock=shared_clock,
    )


def discover_bluecoins(
    source: SensorSource, settings: BlueCoinsConfig, stop_event: threading.Event
) -> set[str] | None:
    """Scan until both BlueCoins are found or retries run out.

    Returns the names still missing (empty when both were found), or ``None`` if a
    stop was requested while waiting.
    """
    expected = {settings.left, settings.right}
    missing = expected - source.discover(settings.scan_timeout_s)
    for attempt in range(1, settings.discovery_retries + 1):
        if not missing:
            break
        logger.warning(
            "Waiting for BlueCoin devices %s; retry %d/%d in %s s",
            ", ".join(sorted(missing)),
            attempt,
            settings.discovery_retries,
            settings.retry_interval_s,
        )
        if stop_event.wait(settings.retry_interval_s):
            return None
        missing = expected - source.discover(settings.scan_timeout_s)
    return missing


def run(config: Config, adapters: BoardAdapters, stop_event: threading.Event) -> int:
    """Start the system, wait for ``stop_event``, then stop it in order."""
    try:
        missing = discover_bluecoins(adapters.source, config.bluecoins, stop_event)
        if missing is None:
            logger.info("Stop requested during BlueCoin discovery")
            return EXIT_OK
        if missing:
            logger.error(
                "BlueCoin devices not found after %d retries: %s",
                config.bluecoins.discovery_retries,
                ", ".join(sorted(missing)),
            )
            return EXIT_STARTUP

        try:
            extractor, classifier = adapters.load_classifier()
        except NativeLibraryError as exc:
            logger.error("%s", exc)
            return EXIT_STARTUP

        runtime = build_runtime(
            config,
            source=adapters.source,
            extractor=extractor,
            classifier=classifier,
            verifier=adapters.verifier,
            actuators=adapters.start_actuators(),
            sink=adapters.sink,
            gui=adapters.gui,
            clock=adapters.clock,
        )
        runtime.start()
        try:
            logger.info("System running; send SIGINT or SIGTERM to stop")
            stop_event.wait()
        finally:
            runtime.stop()
        return EXIT_OK
    except Exception:
        logger.exception("Unexpected error")
        return EXIT_ERROR


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Set ``stop_event`` on SIGINT or SIGTERM."""

    def request_stop(signum: int, _frame: object) -> None:
        logger.info("Received %s; stopping", signal.Signals(signum).name)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cpsa-board",
        description="Run the CPSA pipeline on the board until SIGINT or SIGTERM.",
    )
    parser.add_argument("--config", required=True, type=Path, help="board YAML configuration")
    args = parser.parse_args(argv)

    try:
        config = load(args.config)
    except ConfigError as exc:
        print(f"cpsa-board: {exc}", file=sys.stderr)
        return EXIT_STARTUP

    # cpsa-board runs the IMU-triggered pipeline: the camera-only trigger belongs
    # to cpsa-supervisor's medical mode. Reject it here, before any adapter is
    # built, so a swapped config fails with this message instead of a BlueCoin
    # discovery error from the no-op source (R-05).
    if config.medical.trigger == "vision":
        print(
            "cpsa-board: medical.trigger is 'vision', but cpsa-board is IMU-only; "
            "run the camera-only medical mode with cpsa-supervisor, or set "
            "medical.trigger: imu in the configuration",
            file=sys.stderr,
        )
        return EXIT_STARTUP

    configure_logging(config.log.base_path, config.log.level)
    try:
        adapters = board_adapters(config)
    except VisionStartupError as exc:
        logger.error("Vision pipeline cannot start: %s", exc)
        return EXIT_STARTUP
    except ImportError as exc:
        logger.error("Board dependencies are missing (install the 'board' extra): %s", exc)
        return EXIT_STARTUP
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    return run(config, adapters, stop_event)


if __name__ == "__main__":
    sys.exit(main())
