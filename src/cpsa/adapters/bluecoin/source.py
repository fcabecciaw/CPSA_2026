# source.py
# BLE interface for the two BlueCoin wrist sensors (STMicroelectronics): discovery,
# connection, notifications and automatic reconnection.
#
# Author: Francesco Urru
# Github: https://github.com/frarvo
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

import contextlib
import logging
import threading
import time
from typing import Any

from blue_st_sdk.features.feature_accelerometer import FeatureAccelerometer
from blue_st_sdk.features.feature_gyroscope import FeatureGyroscope
from blue_st_sdk.manager import Manager, ManagerListener
from blue_st_sdk.node import NodeListener, NodeStatus
from bluepy.btle import BTLEDisconnectError

from cpsa.adapters.ble_lock import ble_reconnection_lock
from cpsa.adapters.bluecoin.fusion_feature import (
    FeatureMemsSensorFusionCompact,
    register_fusion_feature,
)
from cpsa.adapters.bluecoin.listeners import SampleListener
from cpsa.domain.model import SampleKind, Side
from cpsa.ports import SampleCallback

logger = logging.getLogger(__name__)

_FEATURES = (
    (SampleKind.ACC, FeatureAccelerometer),
    (SampleKind.GYR, FeatureGyroscope),
    (SampleKind.QUAT, FeatureMemsSensorFusionCompact),
)


class BlueCoinSource:
    """``SensorSource`` for the left and right BlueCoin, identified by BLE name."""

    def __init__(self, left_name: str, right_name: str):
        register_fusion_feature()  # before any scan builds the node features
        self._names = {Side.LEFT: left_name, Side.RIGHT: right_name}
        self._nodes: dict[str, Any] = {}
        self._threads: list[BlueCoinThread] = []

    def discover(self, timeout_s: float) -> set[str]:
        """Scan for BlueCoin nodes; remember them by name and return the names found."""
        self._nodes = {}
        for node in scan_bluecoin_devices(timeout_s):
            try:
                name = node.get_name()
            except Exception:
                logger.warning("Cannot read the name of a scanned node", exc_info=True)
                continue
            if name:
                self._nodes[name] = node
        return set(self._nodes)

    def start(self, on_sample: SampleCallback) -> None:
        """Connect both wrists and stream their samples to ``on_sample``.

        Raises ``RuntimeError`` naming the devices not found by the last ``discover``.
        """
        missing = sorted(name for name in self._names.values() if name not in self._nodes)
        if missing:
            raise RuntimeError(f"BlueCoin devices not discovered: {', '.join(missing)}")

        for side, name in self._names.items():
            node = self._nodes[name]
            features, listeners = [], []
            try:
                for kind, feature_class in _FEATURES:
                    feature = node.get_feature(feature_class)
                    if feature:
                        features.append(feature)
                        listeners.append(SampleListener(side, kind, on_sample))
                    else:
                        logger.warning("%s is missing the %s feature", name, kind.value)
            except Exception:
                logger.exception("Error retrieving features of %s", name)
                continue
            if not features:
                logger.warning("No features available on %s", name)
                continue
            thread = BlueCoinThread(node, features, listeners, device_id=f"{side.value}:{name}")
            thread.start()
            self._threads.append(thread)
            logger.info(
                "BlueCoin %s (%s) started with %d features", name, side.value, len(features)
            )

    def stop(self) -> None:
        for thread in self._threads:
            try:
                thread.stop()
            except Exception:
                logger.exception("Error stopping BlueCoin thread %s", thread.device_id)
        self._threads.clear()


class BlueCoinManagerListener(ManagerListener):
    """Logs BLE discovery events."""

    def on_discovery_change(self, manager, enabled):
        logger.info("BlueCoin discovery %s", "started" if enabled else "stopped")

    def on_node_discovered(self, manager, node):
        try:
            logger.info("Found BlueCoin node: %s - %s", node.get_name(), node.get_tag())
        except Exception:
            logger.info("Found BlueCoin node (name/tag unavailable)")


class BlueCoinNodeListener(NodeListener):
    """Logs connection and disconnection events of a node."""

    def on_connect(self, node):
        logger.info("BlueCoin %s: %s connected", node.get_name(), node.get_tag())

    def on_disconnect(self, node, unexpected=False):
        logger.info(
            "BlueCoin %s: %s disconnected%s",
            node.get_name(),
            node.get_tag(),
            " unexpectedly" if unexpected else "",
        )

    def on_status_change(self, node, new_status, old_status):
        logger.info("BlueCoin status changed from %s to %s", old_status, new_status)


def scan_bluecoin_devices(timeout_s: float) -> list:
    """Scan for BlueCoin nodes for ``timeout_s`` seconds and return the nodes found."""
    nodes = []
    logger.info("Starting BLE scan for %s seconds...", timeout_s)
    manager = Manager.instance()
    listener = BlueCoinManagerListener()
    manager.add_listener(listener)
    try:
        manager.discover(timeout_s)
        nodes = manager.get_nodes()
    except Exception:
        logger.exception("Error during BlueCoin discovery")
    finally:
        with contextlib.suppress(Exception):
            manager.remove_listener(listener)

    if not nodes:
        logger.warning("No BlueCoin devices found")
    else:
        logger.info("Total BlueCoin devices found: %d", len(nodes))
    return nodes


class BlueCoinThread(threading.Thread):
    """
    Thread managing the connection to one BlueCoin with several features.
    Parameters:
        node: BlueST Node object representing the BlueCoin device.
        features: list of Feature.
        feature_listeners: list of FeatureListener, same length as features.
        device_id: Identifier string for logs.
    """

    def __init__(self, node, features, feature_listeners, device_id: str):
        super().__init__(daemon=True)
        self.node = node
        self.node_listener = BlueCoinNodeListener()
        self.device_id = device_id
        self.stop_event = threading.Event()
        self.features = list(features)
        self.feature_listeners = list(feature_listeners)
        if len(self.features) != len(self.feature_listeners):
            raise ValueError("features and feature_listeners must have the same length")

    # Thread lifecycle
    def run(self):
        logger.info("[%s] Thread started", self.device_id)
        ok = self._connect()
        if not ok or self.stop_event.is_set():
            self._cleanup()
            return
        self._start_notifications()
        self._listen()

    def stop(self):
        self.stop_event.set()
        if self.is_alive():
            self.join()
        logger.info("[%s] Thread stopped", self.device_id)

    # Thread internals
    def _connect(self) -> bool:
        self.node.add_listener(self.node_listener)
        while not self.node.connect():
            if self.stop_event.is_set():
                return False
            logger.warning("[%s] Connection failed, retrying...", self.device_id)
            # 1 second interruptible wait
            for _ in range(10):
                if self.stop_event.is_set():
                    return False
                time.sleep(0.1)
        logger.info("[%s] Connected successfully", self.device_id)
        # Attach listeners for all features
        for feature, listener in zip(self.features, self.feature_listeners, strict=True):
            try:
                feature.add_listener(listener)
            except Exception:
                logger.exception("[%s] add_listener error", self.device_id)
        return True

    def _start_notifications(self):
        # Enable BLE notifications for all features
        for feature in self.features:
            try:
                self.node.enable_notifications(feature)
            except Exception:
                logger.exception("[%s] enable_notifications error", self.device_id)

    def _stop_notifications(self):
        # Disable BLE notifications and remove listeners for all features
        for feature, listener in zip(self.features, self.feature_listeners, strict=True):
            with contextlib.suppress(Exception):
                self.node.disable_notifications(feature)
            with contextlib.suppress(Exception):
                feature.remove_listener(listener)

    def _listen(self):
        while not self.stop_event.is_set():
            try:
                if self.node.get_status() != NodeStatus.CONNECTED:
                    self._handle_reconnection()
                else:
                    self.node.wait_for_notifications(0.05)
            except BTLEDisconnectError:
                logger.error("[%s] BTLE exception caught", self.device_id)
                self._handle_reconnection()
            except Exception:
                logger.exception("[%s] wait_for_notifications error", self.device_id)
                self._handle_reconnection()
        self._cleanup()

    def _handle_reconnection(self):
        self._stop_notifications()
        with ble_reconnection_lock:
            while not self.stop_event.is_set():
                try:
                    logger.info("[%s] Attempting reconnection...", self.device_id)
                    if self.node.connect():
                        logger.info("[%s] Reconnected successfully", self.device_id)
                        # Reattach listeners and re-enable notifications on reconnection
                        for feature, listener in zip(
                            self.features, self.feature_listeners, strict=True
                        ):
                            with contextlib.suppress(Exception):
                                feature.add_listener(listener)
                        self._start_notifications()
                        return
                except BTLEDisconnectError:
                    logger.warning("[%s] Reconnection failed, retrying...", self.device_id)
                except Exception:
                    logger.exception("[%s] Reconnection error", self.device_id)
                # 2 seconds interruptible wait
                for _ in range(20):
                    if self.stop_event.is_set():
                        return
                    time.sleep(0.1)

    def _cleanup(self):
        # Never raise during shutdown
        with contextlib.suppress(Exception):
            self._stop_notifications()
        # Remove listener before disconnect
        with contextlib.suppress(Exception):
            self.node.remove_listener(self.node_listener)
        try:
            if self.node.get_status() == NodeStatus.CONNECTED:
                self.node.disconnect()
                logger.info("[%s] Disconnected cleanly", self.device_id)
        except Exception:
            pass
