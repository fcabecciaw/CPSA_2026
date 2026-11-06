"""``cpsa-board --config PATH``: run the pipeline on the board until SIGINT/SIGTERM.

Exit codes: 0 after an orderly stop, 1 on an unexpected error, 2 when the
configuration is invalid, the ``board`` extra is not installed, a BlueCoin is
missing, or the native libraries cannot be loaded.
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
from cpsa.application.config import BlueCoinsConfig, Config, ConfigError, load
from cpsa.application.runtime import build_runtime
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

logger = logging.getLogger("cpsa.board")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STARTUP = 2


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


def board_adapters(config: Config) -> BoardAdapters:
    """Construct the real adapters. The single place where the verifier and GUI are chosen."""
    # Imported here so the runner stays importable (and testable) without the
    # board-only BLE libraries.
    from cpsa.adapters.bluecoin.source import BlueCoinSource

    return BoardAdapters(
        source=BlueCoinSource(config.bluecoins.left, config.bluecoins.right),
        load_classifier=_load_native,
        start_actuators=lambda: start_actuators(config.actuators),
        verifier=StubVerifier(),
        gui=NullGui(),
        sink=JsonlEventSink(config.log.base_path),
        clock=SystemClock(),
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

    configure_logging(config.log.base_path, config.log.level)
    try:
        adapters = board_adapters(config)
    except ImportError as exc:
        logger.error("Board dependencies are missing (install the 'board' extra): %s", exc)
        return EXIT_STARTUP
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    return run(config, adapters, stop_event)


if __name__ == "__main__":
    sys.exit(main())
