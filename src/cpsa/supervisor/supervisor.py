"""Run exactly one demo mode at a time and publish its status (IF-301..304, IF-701..703).

Pure coordination code: no asyncio and no hardware. Mode switches run on one
worker thread, status is published on another, and both reach clients only through
the thread-safe :class:`cpsa.ports.Publisher` port.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

from cpsa.application.config import Config
from cpsa.ports import Clock, Publisher
from cpsa.protocol import system
from cpsa.supervisor.contract import Mode, ModeContext, ModeStartError

logger = logging.getLogger("cpsa.supervisor")

IDLE = "idle"
"""The built-in mode that owns no devices."""

STATUS_INTERVAL_S = 1.0
"""``status`` is published at this rate, and on any change (IF-301)."""

SWITCH_TIMEOUT_S = 5.0
"""Default stop timeout handed to a mode that is being switched out."""


class Supervisor:
    """Owns the mode lifecycle and the ``status`` topic.

    ``modes`` maps mode names to their implementation; ``idle`` is built in and
    must not be present. The constructor does no I/O: call :meth:`start_default`
    to start the status thread and the configured default mode, and
    :meth:`shutdown` to release everything.
    """

    def __init__(
        self,
        modes: Mapping[str, Mode],
        publisher: Publisher,
        config: Config,
        clock: Clock,
        *,
        status_interval_s: float = STATUS_INTERVAL_S,
        switch_timeout_s: float = SWITCH_TIMEOUT_S,
    ) -> None:
        if status_interval_s <= 0.0:
            raise ValueError("status_interval_s must be positive")
        if switch_timeout_s <= 0.0:
            raise ValueError("switch_timeout_s must be positive")
        self._modes = dict(modes)
        self._publisher = publisher
        self._config = config
        self._clock = clock
        self._status_interval_s = status_interval_s
        self._switch_timeout_s = switch_timeout_s
        self._lock = threading.RLock()
        self._started_at = clock.monotonic()
        self._mode = IDLE
        self._mode_state = "stopped"
        self._errors: list[str] = []
        self._active: Mode | None = None
        self._active_stop_event: threading.Event | None = None
        self._switching = False
        self._closing = False
        self._status_thread: threading.Thread | None = None
        self._status_stop = threading.Event()
        self._switch_thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start_default(self) -> None:
        """Start the status thread and the configured default mode (FR-205)."""
        default = self._config.supervisor.default_mode
        if default != IDLE and default not in self._modes:
            raise ValueError(f"default mode {default!r} is not registered")
        with self._lock:
            if self._status_thread is None:
                self._status_thread = threading.Thread(
                    target=self._status_loop, name="cpsa-supervisor-status", daemon=True
                )
                self._status_thread.start()
        self._publish_status()
        self.request_mode(default, None)

    def shutdown(self, timeout_s: float) -> None:
        """Refuse new switches, wait for an in-flight one and stop whatever is active."""
        with self._lock:
            self._closing = True
            self._status_stop.set()
            in_flight = self._active_stop_event
        if in_flight is not None:
            # Registered before mode.start, so this cancels a start that is still
            # blocking (F4, R-03).
            in_flight.set()
        status_thread = self._status_thread
        if status_thread is not None:
            status_thread.join(timeout_s)
        switch_thread = self._switch_thread
        if switch_thread is not None and switch_thread.is_alive():
            switch_thread.join(timeout_s)
        with self._lock:
            active = self._active
            stop_event = self._active_stop_event
            self._active = None
            self._active_stop_event = None
            self._mode = IDLE
            self._mode_state = "stopped"
        if stop_event is not None:
            stop_event.set()
        if active is not None:
            try:
                active.stop(timeout_s)
            except Exception:
                logger.exception("Mode %s failed to stop during shutdown", active.name)

    # -- commands ----------------------------------------------------------

    def request_mode(self, mode: str, ref_seq: int | None = None) -> None:
        """Handle ``mode.set`` (IF-302); the switch, when needed, runs on a worker."""
        if mode != IDLE and mode not in self._modes:
            self._publisher.event(
                "error", system.error("bad_command", f"unknown mode {mode!r}", ref_seq)
            )
            return
        with self._lock:
            if self._closing:
                self._publisher.event(
                    "error",
                    system.error("internal", "the supervisor is shutting down", ref_seq),
                )
                return
            if self._switching:
                self._publisher.event(
                    "error",
                    system.error("busy", "a mode switch is already in progress", ref_seq),
                )
                return
            if mode == self._mode and self._mode_state in ("stopped", "running"):
                self._publisher.event("mode.progress", system.mode_progress(mode, "ready"))
                return
            self._switching = True
            self._switch_thread = threading.Thread(
                target=self._switch, args=(mode,), name="cpsa-mode-switch", daemon=True
            )
            self._switch_thread.start()

    def status(self) -> dict[str, Any]:
        """The ``status`` payload that clients should currently see (IF-301)."""
        return self._status_payload()

    # -- switch state machine ---------------------------------------------

    def _switch(self, target: str) -> None:
        try:
            self._run_switch(target)
        except Exception as exc:
            logger.exception("Unexpected failure while switching to %s", target)
            self._fail(target, "internal", str(exc))

    def _run_switch(self, target: str) -> None:
        with self._lock:
            previous = self._active
            previous_stop_event = self._active_stop_event
            self._mode = target
            self._mode_state = "stopping"
        self._progress(target, "stopping")
        self._publish_status()

        if previous is not None:
            if previous_stop_event is not None:
                previous_stop_event.set()
            released = previous.stop(self._switch_timeout_s)
            if not released:
                logger.error(
                    "Mode %s did not release its devices within %.1f s",
                    previous.name,
                    self._switch_timeout_s,
                )
                # Keep the reference: the mode still holds its devices, so the
                # next switch and shutdown() must try to stop it again (R-06).
                self._fail(
                    target,
                    "internal",
                    f"mode {previous.name} did not release its devices",
                    keep_active=True,
                )
                return
        with self._lock:
            self._active = None
            self._active_stop_event = None
            self._mode_state = "stopped"
        self._progress(target, "releasing")

        self._progress(target, "starting")
        stop_event = threading.Event()
        with self._lock:
            self._mode_state = "starting"
            # Registered before start() so shutdown() can cancel a blocking
            # start and stop whatever ends up active (F4, R-03).
            self._active_stop_event = stop_event
            if self._closing:
                # shutdown() ran after the previous mode was released and before
                # this event was registered, so it saw nothing to set (R-20).
                stop_event.set()
        self._publish_status()

        if target == IDLE:
            with self._lock:
                self._errors = []
                self._mode_state = "stopped"
                self._active_stop_event = None
            self._finish(target, "ready")
            return

        mode = self._modes[target]
        started = False
        if stop_event.is_set():
            logger.info("Stop requested before %s started; not starting it", target)
        else:
            context = ModeContext(
                publisher=self._publisher, config=self._config, stop_event=stop_event
            )
            try:
                mode.start(context)
                started = True
            except ModeStartError as exc:
                logger.error("Mode %s failed to start: %s", target, exc.message)
                self._fail(target, exc.code, exc.message)
                return
            except Exception as exc:
                logger.exception("Mode %s failed to start", target)
                self._fail(target, "internal", str(exc))
                return

        with self._lock:
            cancelled = self._closing or stop_event.is_set()
            if not cancelled:
                self._active = mode
                self._mode_state = "running"
                self._errors = []
        if cancelled:
            logger.info("Stop requested while %s was starting; releasing it", target)
            if started:
                try:
                    mode.stop(self._switch_timeout_s)
                except Exception:
                    logger.exception("Mode %s failed to stop after a cancelled start", target)
            with self._lock:
                self._active = None
                self._active_stop_event = None
                self._errors = []
                self._mode = IDLE
                self._mode_state = "stopped"
                self._switching = False
            return
        self._finish(target, "ready")

    def _finish(self, target: str, step: str, error: Mapping[str, str] | None = None) -> None:
        with self._lock:
            self._switching = False
        self._publisher.event("mode.progress", system.mode_progress(target, step, error))
        self._publish_status()

    def _fail(
        self,
        target: str,
        code: str,
        message: str,
        *,
        keep_active: bool = False,
    ) -> None:
        with self._lock:
            if not keep_active:
                self._active = None
                self._active_stop_event = None
            self._mode = IDLE
            self._mode_state = "failed"
            self._errors = [code]
        self._finish(target, "failed", {"code": code, "message": message})

    def _progress(self, target: str, step: str) -> None:
        self._publisher.event("mode.progress", system.mode_progress(target, step))

    # -- status ------------------------------------------------------------

    def _status_loop(self) -> None:
        while not self._closing:
            started = time.monotonic()
            try:
                self._publish_status()
            except Exception:
                logger.exception("Publishing status failed")
            elapsed = time.monotonic() - started
            self._status_stop.wait(max(0.0, self._status_interval_s - elapsed))

    def _status_payload(self) -> dict[str, Any]:
        with self._lock:
            mode = self._mode
            mode_state = self._mode_state
            errors = list(self._errors)
            active = self._active
            uptime_s = self._clock.monotonic() - self._started_at

        health: Mapping[str, Any] = {}
        if active is not None and mode_state == "running":
            try:
                health = active.health()
            except Exception:
                logger.exception("Mode %s health() failed", active.name)
        health_errors = health.get("errors") or ()
        all_errors = sorted({*errors, *health_errors})
        return system.status(
            mode=mode,
            mode_state=mode_state,
            perception={key: health.get(key) for key in ("fps", "dpu_ms", "camera", "model")},
            # Take the IMU from health whatever the mode is called: this layer is
            # mode-agnostic (R-07).
            imu=health.get("imu", "n/a"),
            errors=all_errors,
            uptime_s=uptime_s,
        )

    def _publish_status(self) -> None:
        """Publish ``status`` now: every tick, and on any change (IF-301).

        Building the payload and publishing it happen under one lock section, so a
        periodic tick cannot replace a newer transition with an older payload.
        """
        with self._lock:
            self._publisher.state("status", self._status_payload())


__all__ = ["IDLE", "STATUS_INTERVAL_S", "SWITCH_TIMEOUT_S", "Supervisor"]
