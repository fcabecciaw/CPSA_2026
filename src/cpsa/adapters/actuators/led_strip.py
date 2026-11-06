# led_strip.py
# Actuator module for Wi-Fi LED strip control using flux_led
#
# Author: Francesco Urru
# GitHub: https://github.com/frarvo
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

import contextlib
import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

import flux_led
from flux_led import BulbScanner

from cpsa.adapters.ble_lock import ble_reconnection_lock
from cpsa.domain.model import ACTUATOR_ID_PREFIX, ActuatorKind

logger = logging.getLogger(__name__)


def scan_led_devices(timeout_s: float) -> list[str]:
    """Scan the network for flux_led Wi-Fi LED devices and return their IP addresses."""
    logger.info("Starting Wi-Fi LED scan for %s seconds...", timeout_s)
    ip_list: list[str] = []

    try:
        scanner = BulbScanner()
        devices = scanner.scan(timeout_s)

        for device in devices:
            ipaddr = device.get("ipaddr")
            model = device.get("model")
            if ipaddr:
                logger.info("Found LED device: %s (model: %s)", ipaddr, model)
                ip_list.append(ipaddr)

        if not ip_list:
            logger.warning("No LED devices found")

    except Exception:
        logger.exception("LED scan error")

    return ip_list


class LedThread(threading.Thread):
    """
    Actuator managing a Wi-Fi LED strip using flux_led.

    A color or pattern request turns the strip on; it turns itself off after
    ``duration_ms`` unless the request carries its own ``duration``.
    """

    kind = ActuatorKind.LED_STRIP

    def __init__(
        self,
        ip_address: str,
        duration_ms: int,
        fast_attempts: int,
        interval_s: float,
        sleep_s: float,
    ):
        super().__init__(daemon=True)
        self.ip_address = ip_address
        self.id = f"{ACTUATOR_ID_PREFIX[self.kind]}{ip_address}"

        self._stop_event = threading.Event()
        self._event = threading.Event()
        self._disconnect_event = threading.Event()

        self._pattern: int | None = None
        self._color: tuple[int, int, int, int] | None = None
        self._speed: int = 100
        self._intensity: int = 100
        self._off_timer: threading.Timer | None = None
        self._pending_duration_ms = 0

        self._bulb: Any = None  # flux_led.WifiLedBulb once connected

        self._fast_retry_attempts = int(fast_attempts)
        self._retry_interval = interval_s
        self._retry_sleep = sleep_s
        self._duration = int(duration_ms)

    def run(self) -> None:
        logger.info(f"[LedStrip {self.ip_address}] Thread started")
        self._connect()
        self._wait_for_event()

    def _wait_for_event(self) -> None:
        while not self._stop_event.is_set():
            if self._event.wait(timeout=0.5):
                self._event.clear()

                if self._disconnect_event.is_set():
                    self._disconnect_event.clear()
                    with ble_reconnection_lock:
                        self._reconnection_attempts()

                if self._bulb:
                    self._process_action()

        if self._bulb:
            self._turn_off()

    def _connect(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._bulb = flux_led.WifiLedBulb(self.ip_address)
                self._bulb.refreshState()
                self._connection_feedback()
                logger.info(f"[LedStrip {self.ip_address}] Connected successfully")
                return
            except Exception as exc:
                logger.error(f"[LedStrip {self.ip_address}] Connection error: {exc}")
                time.sleep(self._retry_interval)

    def _connection_feedback(self) -> None:
        try:
            self._bulb.setRgbw(0, 0, 0, 255, brightness=50)
            time.sleep(0.3)
            self._bulb.setRgbw(0, 0, 0, 0)
            time.sleep(0.2)
            self._bulb.setRgbw(0, 0, 0, 255, brightness=50)
            time.sleep(0.3)
            self._bulb.setRgbw(0, 0, 0, 0)
        except Exception:
            pass

    def _reconnection_attempts(self) -> None:
        logger.info(f"[LedStrip {self.ip_address}] Starting reconnection attempts...")

        for attempt in range(1, self._fast_retry_attempts + 1):
            if self._stop_event.is_set():
                return
            if self._try_reconnect():
                return
            logger.warning(
                "[LedStrip %s] Fast retry %d/%d failed",
                self.ip_address,
                attempt,
                self._fast_retry_attempts,
            )
            time.sleep(self._retry_interval)

        logger.warning(
            f"[LedStrip {self.ip_address}] Switching to slow retries every {self._retry_sleep}s"
        )
        while not self._stop_event.is_set():
            if self._try_reconnect():
                return
            logger.warning(f"[LedStrip {self.ip_address}] Slow retry failed")
            time.sleep(self._retry_sleep)

    def _try_reconnect(self) -> bool:
        try:
            self._bulb = flux_led.WifiLedBulb(self.ip_address)
            self._bulb.refreshState()
            self._connection_feedback()
            logger.info(f"[LedStrip {self.ip_address}] Reconnected successfully")
            return True
        except Exception:
            return False

    def execute(self, params: Mapping[str, object]) -> None:
        """Show ``color`` (r, g, b, w) or effect ``pattern`` at ``intensity`` and ``speed``.

        Optional ``duration`` (ms) overrides the configured auto-off delay.
        """
        self._configure(
            pattern=params.get("pattern"),
            color=params.get("color"),
            intensity=params.get("intensity", 100),
            speed=params.get("speed", 100),
            duration=params.get("duration"),
        )

    def _configure(self, *, pattern, color, intensity, speed, duration) -> None:
        self._pattern = pattern
        # Clamp color channels to 0..255
        if color is not None:
            try:
                r, g, b, w = color
                r = max(0, min(int(r), 255))
                g = max(0, min(int(g), 255))
                b = max(0, min(int(b), 255))
                w = max(0, min(int(w), 255))
                self._color = (r, g, b, w)
            except Exception:
                self._color = None
        else:
            self._color = None

        self._intensity = max(0, min(intensity, 100))
        self._speed = max(1, min(speed, 100))
        self._pending_duration_ms = self._duration if duration is None else max(0, int(duration))
        self._event.set()

    def _process_action(self) -> None:
        try:
            if self._color:
                r, g, b, w = self._color
                if self._intensity <= 0 or (r == 0 and g == 0 and b == 0 and w == 0):
                    self._turn_off()
                    return
                logger.info(
                    f"[LedStrip {self.ip_address}] Setting color ({r},{g},{b},{w}) "
                    f"intensity={self._intensity}%"
                )
                self._bulb.setRgbw(r, g, b, w, brightness=self._intensity)
                self._schedule_off_timer(self._pending_duration_ms)
            elif self._pattern is not None:
                if self._intensity <= 0:
                    self._turn_off()
                logger.info(
                    f"[LedStrip {self.ip_address}] Setting pattern {self._pattern} "
                    f"speed={self._speed}% intensity={self._intensity}%"
                )
                self._bulb.set_effect(
                    effect=self._pattern, speed=self._speed, brightness=self._intensity
                )
                self._schedule_off_timer(self._pending_duration_ms)

        except Exception as exc:
            logger.error(f"[LedStrip {self.ip_address}] Action error: {exc}")
            self._disconnect_event.set()
            self._event.set()

    def _turn_off(self) -> None:
        try:
            if self._bulb:
                try:
                    self._bulb.turnOff()
                except Exception as e:
                    logger.info(f"[LedStrip {self.ip_address}] turnOff() failed: {e}")
                with contextlib.suppress(Exception):
                    self._bulb.setRgbw(0, 0, 0, 0, brightness=0)
                logger.info(f"[LedStrip {self.ip_address}] Turned off")
        except Exception as exc:
            logger.error(f"[LedStrip {self.ip_address}] TurnOff error: {exc}")

    def stop(self) -> None:
        self._stop_event.set()
        self._event.set()
        try:
            if self._off_timer:
                self._off_timer.cancel()
        except Exception:
            pass
        if threading.current_thread() is not self and self.is_alive():
            self.join()
        logger.info(f"[LedStrip {self.ip_address}] Thread stopped")

    def _schedule_off_timer(self, ms: int) -> None:
        try:
            if self._off_timer:
                self._off_timer.cancel()
        except Exception:
            pass
        self._off_timer = None
        if ms > 0 and not self._stop_event.is_set():
            self._off_timer = threading.Timer(ms / 1000.0, self._auto_off)
            self._off_timer.daemon = True
            self._off_timer.start()

    def _auto_off(self) -> None:
        if self._stop_event.is_set():
            return
        try:
            if self._bulb:
                self._bulb.turnOff()
                logger.info(f"[LedStrip {self.ip_address}] Auto OFF")
        except Exception as e:
            logger.warning(f"[LedStrip {self.ip_address}] Auto OFF error: {e}")
