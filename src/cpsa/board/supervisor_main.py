"""``cpsa-supervisor --config PATH``: serve the UI and run the default mode.

Exit codes: 0 after an orderly stop, 1 on an unexpected error, 2 when the
configuration is invalid, the server extra is missing, or the medical mode cannot
be imported (for example when the ``board`` extra is not installed; use
``--fake-medical`` then).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from cpsa.adapters.clock import SystemClock
from cpsa.adapters.event_log import configure_logging
from cpsa.adapters.web.log_forward import install as install_log_forward
from cpsa.application.config import Config, ConfigError
from cpsa.application.config import load as load_config
from cpsa.board.fake_mode import FakeMedicalMode
from cpsa.supervisor.contract import Mode
from cpsa.supervisor.supervisor import Supervisor

logger = logging.getLogger("cpsa.supervisor")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STARTUP = 2
SHUTDOWN_TIMEOUT_S = 5.0
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "docs" / "protocol" / "v1" / "examples"


class StartupError(RuntimeError):
    """The supervisor cannot start with this configuration or environment."""


def _server_version() -> str:
    try:
        return version("cpsa")
    except PackageNotFoundError:
        return "0.1.0"


def _load_medical_mode(config: Config, fake: bool) -> Mode:
    """Build the medical mode, or explain why this installation cannot."""
    if fake:
        return FakeMedicalMode(EXAMPLES_DIR)
    try:
        from cpsa.board.medical_mode import MedicalMode
    except ImportError as exc:
        raise StartupError(
            "the medical mode is not available (the 'board' extra or its board "
            f"dependencies are not installed): {exc}; use --fake-medical to run "
            "the dashboard without hardware"
        ) from exc
    return MedicalMode(config)


def _build_server(
    config: Config, on_command: Callable[[str, dict[str, Any], int | None], None]
) -> Any:
    try:
        from cpsa.adapters.web.server import WebServer
    except ImportError as exc:
        raise StartupError(
            f"the web server is not available (install the 'server' extra): {exc}"
        ) from exc
    return WebServer(config.server, on_command, _server_version())


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Set ``stop_event`` on SIGINT or SIGTERM."""

    def request_stop(signum: int, _frame: object) -> None:
        logger.info("Received %s; stopping", signal.Signals(signum).name)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cpsa-supervisor",
        description="Serve the demo UI and run the configured mode until SIGINT or SIGTERM.",
    )
    parser.add_argument("--config", required=True, type=Path, help="board YAML configuration")
    parser.add_argument(
        "--fake-medical",
        action="store_true",
        help="replay docs/protocol/v1/examples instead of running the real medical mode",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"cpsa-supervisor: {exc}", file=sys.stderr)
        return EXIT_STARTUP

    configure_logging(config.log.base_path, config.log.level)
    default_mode = config.supervisor.default_mode
    try:
        modes = {"medical": _load_medical_mode(config, args.fake_medical)}
        if default_mode != "idle" and default_mode not in modes:
            raise StartupError(f"unknown supervisor default_mode {default_mode!r}")
        supervisor_holder: list[Supervisor] = []

        def on_command(command: str, data: dict[str, Any], ref_seq: int | None) -> None:
            if command == "mode.set" and supervisor_holder:
                supervisor_holder[0].request_mode(str(data["mode"]), ref_seq)

        server = _build_server(config, on_command)
        # Device and connection log lines reach the dashboard as `log` events.
        install_log_forward(server)
        supervisor = Supervisor(modes, server, config, SystemClock())
        supervisor_holder.append(supervisor)
    except StartupError as exc:
        logger.error("%s", exc)
        return EXIT_STARTUP

    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    try:
        server.start()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return EXIT_STARTUP

    try:
        supervisor.start_default()
        logger.info(
            "Supervisor running mode %r; UI at http://%s:%d/ (Ctrl+C to stop)",
            default_mode,
            config.server.host,
            server.port,
        )
        stop_event.wait()
        return EXIT_OK
    except Exception:
        logger.exception("Unexpected error")
        return EXIT_ERROR
    finally:
        supervisor.shutdown(SHUTDOWN_TIMEOUT_S)
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
