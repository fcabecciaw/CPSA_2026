"""Minimal stand-ins for the board-only vendor libraries.

They implement just enough of each vendor API for the hardware adapters to be
imported and driven in-process; they do not emulate device behavior beyond that.
"""

from __future__ import annotations

import sys
import time
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

HARDWARE_ADAPTER_MODULES = (
    "cpsa.adapters.actuators.metamotion",
    "cpsa.adapters.actuators.speaker",
    "cpsa.adapters.actuators.led_strip",
    "cpsa.adapters.bluecoin.source",
    "cpsa.adapters.bluecoin.listeners",
    "cpsa.adapters.bluecoin.fusion_feature",
)


@dataclass
class Vendor:
    """Shared state of the stubs, inspected and scripted by tests."""

    ble_devices: list[Any] = field(default_factory=list)
    led_devices: list[dict[str, str]] = field(default_factory=list)
    haptic_calls: list[tuple[int, int]] = field(default_factory=list)
    bulbs: list[Any] = field(default_factory=list)
    bluecoin_nodes: list[Any] = field(default_factory=list)


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _build(vendor: Vendor) -> dict[str, types.ModuleType]:
    class BTLEDisconnectError(Exception):
        pass

    class Scanner:
        def scan(self, timeout: float) -> list[Any]:
            return list(vendor.ble_devices)

    class MetaWear:
        def __init__(self, address: str) -> None:
            self.address = address
            self.is_connected = False
            self.board = address
            self.on_disconnect = None

        def connect(self) -> None:
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False

    class LibMetaWear:
        @staticmethod
        def mbl_mw_haptic_start_motor(board: str, duty: int, duration: int) -> None:
            vendor.haptic_calls.append((duty, duration))

    class WifiLedBulb:
        def __init__(self, ip: str) -> None:
            self.ip = ip
            self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
            vendor.bulbs.append(self)

        def refreshState(self) -> None:  # noqa: N802 - vendor API name
            pass

        def setRgbw(self, *args: Any, **kwargs: Any) -> None:  # noqa: N802
            self.calls.append(("setRgbw", args, kwargs))

        def set_effect(self, **kwargs: Any) -> None:
            self.calls.append(("set_effect", (), kwargs))

        def turnOff(self) -> None:  # noqa: N802
            self.calls.append(("turnOff", (), {}))

    class BulbScanner:
        def scan(self, timeout: float) -> list[dict[str, str]]:
            return list(vendor.led_devices)

    class Listener:
        def __init__(self) -> None:
            pass

    class Feature:
        def __init__(self, *args: Any) -> None:
            self.listeners: list[Any] = []

        def add_listener(self, listener: Any) -> None:
            self.listeners.append(listener)

        def remove_listener(self, listener: Any) -> None:
            self.listeners.remove(listener)

    class NodeStatus:
        CONNECTED = "CONNECTED"
        IDLE = "IDLE"

    class Manager:
        _instance: Manager | None = None

        @classmethod
        def instance(cls) -> Manager:
            return cls()

        def add_listener(self, listener: Any) -> None:
            pass

        def remove_listener(self, listener: Any) -> None:
            pass

        def discover(self, timeout: float) -> None:
            pass

        def get_nodes(self) -> list[Any]:
            return list(vendor.bluecoin_nodes)

    class FieldType:
        Float = "Float"

    return {
        "bluepy": _module("bluepy"),
        "bluepy.btle": _module(
            "bluepy.btle", Scanner=Scanner, BTLEDisconnectError=BTLEDisconnectError
        ),
        "mbientlab": _module("mbientlab"),
        "mbientlab.metawear": _module(
            "mbientlab.metawear", MetaWear=MetaWear, libmetawear=LibMetaWear()
        ),
        "mbientlab.warble": _module(
            "mbientlab.warble", WarbleException=type("W", (Exception,), {})
        ),
        "flux_led": _module("flux_led", WifiLedBulb=WifiLedBulb, BulbScanner=BulbScanner),
        "playsound": _module("playsound", playsound=lambda path: None),
        "dbus_fast": _module("dbus_fast"),
        "dbus_fast.aio": _module("dbus_fast.aio", MessageBus=object),
        "dbus_fast.constants": _module("dbus_fast.constants", BusType=types.SimpleNamespace()),
        "dbus_fast.errors": _module("dbus_fast.errors", DBusError=type("D", (Exception,), {})),
        "dbus_fast.signature": _module("dbus_fast.signature", Variant=object),
        "blue_st_sdk": _module("blue_st_sdk"),
        "blue_st_sdk.feature": _module(
            "blue_st_sdk.feature",
            Feature=Feature,
            FeatureListener=Listener,
            Sample=object,
            ExtractedData=object,
        ),
        "blue_st_sdk.features": _module("blue_st_sdk.features"),
        "blue_st_sdk.features.field": _module(
            "blue_st_sdk.features.field", Field=lambda *a: a, FieldType=FieldType
        ),
        "blue_st_sdk.features.feature_accelerometer": _module(
            "blue_st_sdk.features.feature_accelerometer",
            FeatureAccelerometer=type("FeatureAccelerometer", (Feature,), {}),
        ),
        "blue_st_sdk.features.feature_gyroscope": _module(
            "blue_st_sdk.features.feature_gyroscope",
            FeatureGyroscope=type("FeatureGyroscope", (Feature,), {}),
        ),
        "blue_st_sdk.manager": _module(
            "blue_st_sdk.manager", Manager=Manager, ManagerListener=Listener
        ),
        "blue_st_sdk.node": _module(
            "blue_st_sdk.node", NodeStatus=NodeStatus, NodeListener=Listener
        ),
        "blue_st_sdk.utils": _module("blue_st_sdk.utils"),
        "blue_st_sdk.utils.number_conversion": _module(
            "blue_st_sdk.utils.number_conversion", LittleEndian=object
        ),
        "blue_st_sdk.utils.blue_st_exceptions": _module(
            "blue_st_sdk.utils.blue_st_exceptions",
            BlueSTInvalidOperationException=type("O", (Exception,), {}),
            BlueSTInvalidDataException=type("I", (Exception,), {}),
        ),
    }


def purge_hardware_adapters() -> None:
    """Forget imported hardware adapter modules so the next import starts fresh."""
    for name in HARDWARE_ADAPTER_MODULES:
        sys.modules.pop(name, None)
        package, _, attr = name.rpartition(".")
        if package in sys.modules and hasattr(sys.modules[package], attr):
            delattr(sys.modules[package], attr)


@pytest.fixture
def vendor(monkeypatch: pytest.MonkeyPatch) -> Iterator[Vendor]:
    """Install the vendor stubs for one test and remove every trace afterwards."""
    state = Vendor()
    purge_hardware_adapters()
    for name, module in _build(state).items():
        monkeypatch.setitem(sys.modules, name, module)
    yield state
    purge_hardware_adapters()


def wait_until(condition: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True
