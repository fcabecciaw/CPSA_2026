"""Discover and start the enabled actuators.

A device module (and therefore its vendor library) is imported only when its kind
is enabled, so a board without, e.g., ``flux_led`` installed can still run with
the LED strip disabled.
"""

from __future__ import annotations

import logging
from typing import Protocol

from cpsa.ports import Actuator

logger = logging.getLogger(__name__)


class _Retry(Protocol):
    @property
    def scan_timeout_s(self) -> float: ...
    @property
    def fast_attempts(self) -> int: ...
    @property
    def interval_s(self) -> float: ...
    @property
    def sleep_s(self) -> float: ...


class _Enable(Protocol):
    @property
    def enable(self) -> bool: ...


class _Speaker(_Enable, Protocol):
    @property
    def mac(self) -> str: ...


class _LedStrip(_Enable, Protocol):
    @property
    def duration_ms(self) -> int: ...


class ActuatorSettings(Protocol):
    """Read-only structure of the ``actuators`` configuration section."""

    @property
    def retry(self) -> _Retry: ...
    @property
    def metamotion(self) -> _Enable: ...
    @property
    def speaker(self) -> _Speaker: ...
    @property
    def led_strip(self) -> _LedStrip: ...


def start_actuators(settings: ActuatorSettings) -> list[Actuator]:
    """Scan for and start every enabled actuator; return those that started.

    Devices connect (and reconnect) in their own threads, so an actuator is
    returned even if its device is not reachable yet.
    """
    retry = settings.retry
    threads: list = []

    if settings.metamotion.enable:
        from cpsa.adapters.actuators import metamotion

        for mac in metamotion.scan_metamotion_devices(retry.scan_timeout_s):
            threads.append(
                metamotion.MetaMotionThread(
                    mac, retry.fast_attempts, retry.interval_s, retry.sleep_s
                )
            )

    if settings.speaker.enable:
        from cpsa.adapters.actuators import speaker

        threads.append(
            speaker.SpeakerThread(
                settings.speaker.mac, retry.fast_attempts, retry.interval_s, retry.sleep_s
            )
        )

    if settings.led_strip.enable:
        from cpsa.adapters.actuators import led_strip

        for ip in led_strip.scan_led_devices(retry.scan_timeout_s):
            threads.append(
                led_strip.LedThread(
                    ip,
                    settings.led_strip.duration_ms,
                    retry.fast_attempts,
                    retry.interval_s,
                    retry.sleep_s,
                )
            )

    started: list[Actuator] = []
    for thread in threads:
        try:
            thread.start()
        except Exception:
            logger.exception("Actuator %s failed to start", thread.id)
            continue
        started.append(thread)
        logger.info("Actuator started: %s", thread.id)
    if not started:
        logger.warning("No actuator available; detection, verification and logging continue")
    return started
