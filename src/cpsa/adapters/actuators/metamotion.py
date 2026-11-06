# metamotion.py
# Actuator module for vibration control using MetaMotionRL BLE device
#
# Author: Francesco Urru
# Github: https://github.com/frarvo
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

import contextlib
import logging
import threading
import time
from collections.abc import Mapping

from bluepy.btle import Scanner
from mbientlab.metawear import MetaWear, libmetawear
from mbientlab.warble import WarbleException

from cpsa.adapters.ble_lock import ble_reconnection_lock
from cpsa.domain.model import ACTUATOR_ID_PREFIX, ActuatorKind

logger = logging.getLogger(__name__)


def scan_metamotion_devices(timeout_s: float) -> list[str]:
    """Scan for nearby MetaWear BLE devices and return their MAC addresses."""
    logger.info("Starting MetaMotion BLE scan for %s seconds...", timeout_s)
    mac_list = []
    try:
        devices = Scanner().scan(timeout_s)
    except Exception:
        logger.exception("MetaMotion scan error")
        return []

    for dev in devices:
        name = dev.getValueText(9)  # 9 = Complete Local Name
        if name == "MetaWear":
            logger.info("Found MetaWear at %s", dev.addr)
            mac_list.append(dev.addr)

    if not mac_list:
        logger.warning("No MetaWear devices found")
    return mac_list


class MetaMotionThread(threading.Thread):
    """
    Actuator managing the BLE connection and vibration motor of a MetaMotion device.

    - Connects to the device at a known MAC address
    - Waits for vibration requests (``execute``)
    - Reconnects automatically after an unexpected disconnection
    """

    kind = ActuatorKind.METAMOTION

    def __init__(self, mac_address: str, fast_attempts: int, interval_s: float, sleep_s: float):
        super().__init__(daemon=True)
        self.id = f"{ACTUATOR_ID_PREFIX[self.kind]}{mac_address}"
        self.mac_address = mac_address
        self.stop_event = threading.Event()
        self.event = threading.Event()
        self.disconnect_event = threading.Event()
        self.vibration_duty = 100
        self.vibration_time = 500
        self.device = None
        self.vibration_lock = threading.Lock()

        self.fast_retry_attempts = int(fast_attempts)
        self.retry_interval = int(interval_s)
        self.retry_sleep = int(sleep_s)

    def run(self):
        logger.info("[MetaMotion %s] Thread started", self.mac_address)
        self._connect_meta_device()
        self._wait_for_event()

    def _connect_meta_device(self):
        """Connect to the device; on success give haptic connection feedback."""
        addr = self.mac_address
        self.device = MetaWear(addr)
        self.device.on_disconnect = lambda status: self._on_disconnection(status)

        while not getattr(self.device, "is_connected", False):
            if self.stop_event.is_set():
                return
            try:
                self.device.connect()
                time.sleep(1)
            except WarbleException as e:
                logger.error("[MetaMotion %s] Connection failed: %s", addr, e)
                time.sleep(self.retry_interval)

        logger.info("[MetaMotion %s] Connected successfully", addr)
        self._connection_feedback()

    def _connection_feedback(self):
        """Brief double vibration confirming the connection."""
        try:
            libmetawear.mbl_mw_haptic_start_motor(self.device.board, 100, 200)
            time.sleep(0.250)
            libmetawear.mbl_mw_haptic_start_motor(self.device.board, 100, 200)
            time.sleep(1)
        except Exception:
            # Never raise during shutdown
            pass

    def _wait_for_event(self):
        """Process vibration requests and reconnections until stopped."""
        while not self.stop_event.is_set():
            if self.event.wait(timeout=0.5):
                self.event.clear()

                if self.disconnect_event.is_set():
                    self.disconnect_event.clear()
                    with ble_reconnection_lock:
                        self._reconnection_attempts()

                if self.device and getattr(self.device, "is_connected", False):
                    self._process_vibration()

        if self.device and getattr(self.device, "is_connected", False):
            self._disconnect_device()

    def _process_vibration(self):
        """Start the motor with the current duty cycle and duration."""
        try:
            if self.device and getattr(self.device, "is_connected", False):
                with self.vibration_lock:
                    duty = max(0, min(int(self.vibration_duty), 100))
                    duration = max(1, int(self.vibration_time))
                logger.info(
                    "[MetaMotion %s] Vibrating %dms at %d%%", self.mac_address, duration, duty
                )
                libmetawear.mbl_mw_haptic_start_motor(self.device.board, duty, duration)
        except Exception:
            logger.exception("[MetaMotion %s] Vibration failed", self.mac_address)

    def set_vibration(self, duty_cycle, time_ms):
        """Store the vibration parameters and wake the thread to apply them."""
        with self.vibration_lock:
            self.vibration_duty = duty_cycle
            self.vibration_time = time_ms
        self.event.set()

    def execute(self, params: Mapping[str, object]) -> None:
        """Vibrate with ``duty`` (0-100, default 100) for ``duration`` ms (default 500)."""
        self.set_vibration(params.get("duty", 100), params.get("duration", 500))

    def stop(self):
        """Stop the thread; the device is disconnected by the thread before it exits."""
        self.stop_event.set()
        self.event.set()
        # Prevent callbacks during shutdown
        try:
            if self.device:
                self.device.on_disconnect = None
        except Exception:
            pass

        if threading.current_thread() is not self and self.is_alive():
            self.join()
        logger.info("[MetaMotion %s] Thread stopped", self.mac_address)

    def _disconnect_device(self):
        try:
            if self.device:
                # Disable callbacks before disconnect to avoid teardown races
                with contextlib.suppress(Exception):
                    self.device.on_disconnect = None

                if getattr(self.device, "is_connected", False):
                    self.device.disconnect()
                    logger.info("[MetaMotion %s] Disconnected manually", self.mac_address)
        except Exception:
            logger.exception("[MetaMotion %s] Disconnection error", self.mac_address)

    def _on_disconnection(self, status):
        """MetaWear callback on disconnection; non-zero status means unexpected."""
        if status != 0:
            logger.warning(
                "[MetaMotion %s] Disconnected unexpectedly (status = %s)", self.mac_address, status
            )
            self.disconnect_event.set()
            self.event.set()
        else:
            logger.info("[MetaMotion %s] Disconnected successfully", self.mac_address)

    def _reconnection_attempts(self):
        """Reconnect: ``fast_retry_attempts`` fast retries, then retry until stopped."""
        logger.info("[MetaMotion %s] Starting reconnection procedure", self.mac_address)

        # Recreate the device object
        self.device = MetaWear(self.mac_address)
        self.device.on_disconnect = lambda status: self._on_disconnection(status)

        # Phase 1: fast retries
        for attempt in range(self.fast_retry_attempts):
            if self.stop_event.is_set():
                return
            try:
                self.device.connect()
                logger.info("[MetaMotion %s] Reconnected successfully", self.mac_address)
                self._connection_feedback()
                return
            except Exception as e:
                logger.error(
                    "[MetaMotion %s] Retry %d/%d failed: %s",
                    self.mac_address,
                    attempt + 1,
                    self.fast_retry_attempts,
                    e,
                )
                for _ in range(self.retry_interval):
                    if self.stop_event.is_set():
                        return
                    time.sleep(1)

        # Phase 2: slow retries
        logger.info(
            "[MetaMotion %s] All fast retries failed. Sleeping for %ss before retrying",
            self.mac_address,
            self.retry_sleep,
        )
        while not self.stop_event.is_set():
            try:
                self.device.connect()
                logger.info("[MetaMotion %s] Reconnected successfully", self.mac_address)
                with contextlib.suppress(Exception):
                    self._connection_feedback()
                return
            except Exception as e:
                logger.warning("[MetaMotion %s] Slow retry failed: %s", self.mac_address, e)
                for _ in range(self.retry_interval):
                    if self.stop_event.is_set():
                        return
                    time.sleep(1)
