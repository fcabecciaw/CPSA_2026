# speaker.py
# Actuator module for Bluetooth speaker control and audio playback.
#
# Author:
# GitHub: https://github.com/frarvo
# Repository: https://github.com/frarvo/STOPme
# License: MIT

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path

from dbus_fast.aio import MessageBus
from dbus_fast.constants import BusType
from dbus_fast.errors import DBusError
from dbus_fast.signature import Variant
from playsound import playsound

from cpsa.adapters.ble_lock import ble_reconnection_lock
from cpsa.domain.model import ACTUATOR_ID_PREFIX, ActuatorKind

logger = logging.getLogger(__name__)

AUDIO_PACKAGE = "cpsa.adapters.assets.audio"
MISSING_AUDIO = "missing_audio_file.mp3"
CONNECTED_AUDIO = "speaker_connected.mp3"


def resolve_audio(name: str) -> Path:
    """Return the path of a bundled audio asset, or of the "missing file" clip."""
    path = Path(str(files(AUDIO_PACKAGE) / name))
    if not path.is_file():
        logger.warning("Missing audio file %s; playing %s instead", name, MISSING_AUDIO)
        path = Path(str(files(AUDIO_PACKAGE) / MISSING_AUDIO))
    return path


class _BlueZSpeakerClient:
    """
    Private helper around BlueZ D-Bus APIs using dbus-fast.
    Keeps all Bluetooth management inside speaker.py without subprocess.
    """

    BLUEZ_SERVICE = "org.bluez"
    OBJ_MANAGER_PATH = "/"
    OBJ_MANAGER_IFACE = "org.freedesktop.DBus.ObjectManager"
    PROPERTIES_IFACE = "org.freedesktop.DBus.Properties"
    ADAPTER_IFACE = "org.bluez.Adapter1"
    DEVICE_IFACE = "org.bluez.Device1"

    def __init__(self):
        self.bus = None
        self.adapter_path = None
        self.object_manager = None
        self._device_path_cache = {}

    @staticmethod
    def _unwrap(value):
        """
        Unwraps dbus-fast Variant values into plain Python values.
        """
        return value.value if isinstance(value, Variant) else value

    async def connect(self):
        """
        Connects to the system bus and resolves the default Bluetooth adapter.
        """
        if self.bus is not None:
            return

        self.bus = await MessageBus(bus_type=BusType.SYSTEM).connect()

        introspection = await self.bus.introspect(self.BLUEZ_SERVICE, self.OBJ_MANAGER_PATH)
        obj = self.bus.get_proxy_object(self.BLUEZ_SERVICE, self.OBJ_MANAGER_PATH, introspection)
        self.object_manager = obj.get_interface(self.OBJ_MANAGER_IFACE)

        managed = await self.object_manager.call_get_managed_objects()

        for path, interfaces in managed.items():
            if self.ADAPTER_IFACE in interfaces:
                self.adapter_path = path
                break

        if not self.adapter_path:
            raise RuntimeError("No BlueZ adapter found")

    async def close(self):
        """
        Closes the D-Bus connection if open.
        """
        if self.bus is not None:
            try:
                self.bus.disconnect()
            except Exception:
                pass
            finally:
                self.bus = None
                self.adapter_path = None
                self.object_manager = None
                self._device_path_cache.clear()

    async def _get_managed_objects(self):
        """
        Returns all BlueZ managed objects.
        """
        await self.connect()
        return await self.object_manager.call_get_managed_objects()

    async def _get_adapter_interfaces(self):
        """
        Returns adapter and properties interfaces for the active adapter.
        """
        await self.connect()

        introspection = await self.bus.introspect(self.BLUEZ_SERVICE, self.adapter_path)
        obj = self.bus.get_proxy_object(self.BLUEZ_SERVICE, self.adapter_path, introspection)
        adapter = obj.get_interface(self.ADAPTER_IFACE)
        props = obj.get_interface(self.PROPERTIES_IFACE)
        return adapter, props

    async def _get_device_interfaces(self, device_path: str):
        """
        Returns device and properties interfaces for a BlueZ device path.
        """
        await self.connect()

        introspection = await self.bus.introspect(self.BLUEZ_SERVICE, device_path)
        obj = self.bus.get_proxy_object(self.BLUEZ_SERVICE, device_path, introspection)
        device = obj.get_interface(self.DEVICE_IFACE)
        props = obj.get_interface(self.PROPERTIES_IFACE)
        return device, props

    async def _resolve_device_path(self, mac_address: str) -> str | None:
        """
        Resolves a MAC address to a BlueZ device object path.
        """
        mac_upper = mac_address.upper()

        if mac_upper in self._device_path_cache:
            return self._device_path_cache[mac_upper]

        managed = await self._get_managed_objects()

        for path, interfaces in managed.items():
            dev = interfaces.get(self.DEVICE_IFACE)
            if not dev:
                continue

            address = self._unwrap(dev.get("Address"))
            if address and str(address).upper() == mac_upper:
                self._device_path_cache[mac_upper] = path
                return path

        return None

    async def _set_trusted(self, device_path: str, value: bool = True):
        """
        Sets the Trusted property on a device.
        """
        _, props = await self._get_device_interfaces(device_path)
        await props.call_set(self.DEVICE_IFACE, "Trusted", Variant("b", value))

    async def is_connected(self, mac_address: str) -> bool:
        """
        Returns True if the device is currently connected.
        """
        device_path = await self._resolve_device_path(mac_address)
        if not device_path:
            return False

        _, props = await self._get_device_interfaces(device_path)
        value = await props.call_get(self.DEVICE_IFACE, "Connected")
        return bool(self._unwrap(value))

    async def connect_device(self, mac_address: str) -> bool:
        """
        Connects to a device by MAC address.
        """
        device_path = await self._resolve_device_path(mac_address)
        if not device_path:
            raise RuntimeError(f"BlueZ device path not found for {mac_address}")

        device, _ = await self._get_device_interfaces(device_path)

        with contextlib.suppress(Exception):
            await self._set_trusted(device_path, True)

        await device.call_connect()
        return await self.is_connected(mac_address)

    async def disconnect_device(self, mac_address: str) -> bool:
        """
        Disconnects a device by MAC address.
        Returns True if it ends up disconnected.
        """
        device_path = await self._resolve_device_path(mac_address)
        if not device_path:
            return True

        device, _ = await self._get_device_interfaces(device_path)

        with contextlib.suppress(DBusError):
            await device.call_disconnect()

        return not await self.is_connected(mac_address)


class SpeakerThread(threading.Thread):
    """
    Actuator managing a Bluetooth speaker connection and audio playback.

    - Keeps a BlueZ connection with a known device and reconnects when it drops
    - Plays bundled audio assets on request (``execute``)
    """

    kind = ActuatorKind.SPEAKER

    def __init__(self, mac_address: str, fast_attempts: int, interval_s: float, sleep_s: float):
        super().__init__(daemon=True)
        self.mac_address = mac_address.upper()
        self.id = f"{ACTUATOR_ID_PREFIX[self.kind]}{self.mac_address}"
        self.event = threading.Event()
        self.stop_event = threading.Event()
        self.file: Path | None = None
        self.connected = False

        self.client = _BlueZSpeakerClient()
        self.loop = None

        self.fast_retry_attempts = int(fast_attempts)
        self.retry_interval = interval_s
        self.retry_sleep = sleep_s

    def _run_async(self, coro):
        """
        Runs a coroutine on the thread-owned event loop.
        Must only be called from this thread after loop initialization.
        """
        if self.loop is None:
            raise RuntimeError("Speaker event loop is not initialized")
        return self.loop.run_until_complete(coro)

    def run(self):
        """Keep the connection alive and play audio when requested."""
        logger.info("[Speaker %s] Thread started", self.mac_address)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            while not self.stop_event.is_set():
                is_conn = self._is_connected_internal()
                self.connected = is_conn

                if not is_conn:
                    with ble_reconnection_lock:
                        self._reconnection_attempts()
                    self.connected = self._is_connected_internal()

                if self.event.is_set():
                    try:
                        if self.connected and self.file:
                            p = Path(self.file)
                            if p.exists():
                                if not self.stop_event.is_set() and self._is_connected_internal():
                                    logger.info(
                                        "[Speaker %s] Playing audio: %s", self.mac_address, p
                                    )
                                    playsound(str(p))
                                else:
                                    logger.warning(
                                        "[Speaker %s] Skipping playback (stopping or disconnected)",
                                        self.mac_address,
                                    )
                            else:
                                logger.warning(
                                    "[Speaker %s] File not found: %s", self.mac_address, self.file
                                )
                        elif not self.file:
                            logger.warning("[Speaker %s] No audio file provided", self.mac_address)
                    except Exception:
                        logger.exception(
                            "[Speaker %s] Error during audio playback", self.mac_address
                        )
                    finally:
                        self.event.clear()

                time.sleep(1)
        finally:
            with contextlib.suppress(Exception):
                self._disconnect_internal()

            with contextlib.suppress(Exception):
                self._run_async(self.client.close())

            with contextlib.suppress(Exception):
                self.loop.close()

            self.loop = None
            self.connected = False

    def execute(self, params: Mapping[str, object]) -> None:
        """Play the bundled audio asset named by ``file``; skipped while disconnected."""
        name = params.get("file")
        if not name:
            logger.warning("[Speaker %s] No audio file provided", self.mac_address)
            return
        if not self.connected:
            logger.warning("[Speaker %s] Not connected, skipping actuation", self.mac_address)
            return
        self._request_playback(resolve_audio(str(name)))

    def _request_playback(self, path: Path) -> None:
        self.file = path
        self.event.set()

    def stop(self):
        """Stop the thread; the speaker is disconnected by the thread before it exits."""
        self.stop_event.set()
        self.event.set()

        if threading.current_thread() is not self and self.is_alive():
            self.join()

        logger.info("[Speaker %s] Thread stopped", self.mac_address)

    def _connect_internal(self):
        """Connect through BlueZ D-Bus. Must only be called from the speaker thread."""
        try:
            was_connected = self._is_connected_internal()
            ok = self._run_async(self.client.connect_device(self.mac_address))
            self.connected = bool(ok)

            if self.connected:
                logger.info("[Speaker %s] Connected successfully", self.mac_address)
                if not was_connected:
                    self._connection_feedback()
            else:
                logger.warning("[Speaker %s] Connection failed", self.mac_address)
        except Exception:
            logger.exception("[Speaker %s] Exception during connection", self.mac_address)
            self.connected = False

    def _disconnect_internal(self):
        """Disconnect through BlueZ D-Bus. Must only be called from the speaker thread."""
        try:
            was_connected = self._is_connected_internal()
            self._run_async(self.client.disconnect_device(self.mac_address))
            self.connected = False

            if was_connected:
                logger.info("[Speaker %s] Disconnected", self.mac_address)
        except Exception:
            logger.exception("[Speaker %s] Exception during disconnection", self.mac_address)

    def _is_connected_internal(self) -> bool:
        """Check the connection. Must only be called from the speaker thread."""
        try:
            return bool(self._run_async(self.client.is_connected(self.mac_address)))
        except Exception:
            logger.exception("[Speaker %s] Error checking status", self.mac_address)
            return False

    def _reconnection_attempts(self):
        """Reconnect with fast retries, then slow retries until stopped.

        Must only be called from the speaker thread.
        """
        logger.info("[Speaker %s] Starting reconnection procedure", self.mac_address)

        for attempt in range(self.fast_retry_attempts):
            if self.stop_event.is_set():
                return
            if self.stop_event.wait(self.retry_interval):
                return

            self._connect_internal()
            if self.connected:
                logger.info("[Speaker %s] Reconnected successfully", self.mac_address)
                return

            logger.warning(
                "[Speaker %s] Retry %d/%d failed",
                self.mac_address,
                attempt + 1,
                self.fast_retry_attempts,
            )

        logger.info(
            "[Speaker %s] All fast retries failed. Retrying every %ss",
            self.mac_address,
            self.retry_sleep,
        )

        while not self.stop_event.is_set():
            if self.stop_event.wait(self.retry_sleep):
                return

            self._connect_internal()
            if self.connected:
                logger.info("[Speaker %s] Reconnected successfully", self.mac_address)
                return

            logger.warning("[Speaker %s] Slow retry failed", self.mac_address)

    def _connection_feedback(self):
        """Play a voice line confirming the connection."""
        self._request_playback(resolve_audio(CONNECTED_AUDIO))
