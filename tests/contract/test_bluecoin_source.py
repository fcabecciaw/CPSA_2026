import time
from types import SimpleNamespace
from typing import Any

import pytest

from cpsa import ports
from cpsa.domain.model import SampleKind, Side
from tests.contract.vendor_stubs import Vendor, wait_until


class FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.features: dict[str, Any] = {}
        self.connected = False
        self.disconnected = False

    def get_name(self) -> str:
        return self.name

    def get_tag(self) -> str:
        return "tag"

    def get_feature(self, feature_class: type) -> Any:
        return self.features.setdefault(feature_class.__name__, feature_class(self))

    def add_listener(self, listener: Any) -> None:
        pass

    def remove_listener(self, listener: Any) -> None:
        pass

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.disconnected = True

    def get_status(self) -> str:
        return "CONNECTED"

    def enable_notifications(self, feature: Any) -> None:
        pass

    def disable_notifications(self, feature: Any) -> None:
        pass

    def wait_for_notifications(self, timeout: float) -> None:
        time.sleep(timeout)


def sample(*values: float) -> SimpleNamespace:
    return SimpleNamespace(get_data=lambda: list(values))


def test_discover_returns_names_found(vendor: Vendor) -> None:
    from cpsa.adapters.bluecoin.source import BlueCoinSource

    vendor.bluecoin_nodes = [FakeNode("CPSA_L2"), FakeNode("OTHER")]
    source = BlueCoinSource("CPSA_L2", "CPSA_R2")
    assert isinstance(source, ports.SensorSource)
    assert source.discover(0.01) == {"CPSA_L2", "OTHER"}

    with pytest.raises(RuntimeError, match="CPSA_R2"):
        source.start(lambda *args: None)


def test_samples_are_forwarded_with_side_and_kind(vendor: Vendor) -> None:
    from cpsa.adapters.bluecoin.source import BlueCoinSource

    left, right = FakeNode("CPSA_L2"), FakeNode("CPSA_R2")
    vendor.bluecoin_nodes = [left, right]
    received: list[tuple[Side, SampleKind, tuple[float, ...]]] = []
    source = BlueCoinSource("CPSA_L2", "CPSA_R2")
    source.discover(0.01)
    source.start(lambda side, kind, values, ts: received.append((side, kind, values)))
    try:
        assert wait_until(
            lambda: all(f.listeners for n in (left, right) for f in n.features.values())
        )
        quat = left.features["FeatureMemsSensorFusionCompact"]
        quat.listeners[0].on_update(quat, sample(1, 2, 3, 4))
        acc = right.features["FeatureAccelerometer"]
        acc.listeners[0].on_update(acc, sample(10, 20, 30))
        acc.listeners[0].on_update(acc, sample(10))  # malformed: ignored
    finally:
        source.stop()

    assert received == [
        (Side.LEFT, SampleKind.QUAT, (1.0, 2.0, 3.0, 4.0)),
        (Side.RIGHT, SampleKind.ACC, (10.0, 20.0, 30.0)),
    ]
    assert left.disconnected and right.disconnected


def test_source_registers_the_quaternion_feature_before_discovery(vendor: Vendor) -> None:
    """The stock SDK has no class for mask 0x100; without it no node exposes quaternions."""
    import sys

    from cpsa.adapters.bluecoin.fusion_feature import (
        FUSION_COMPACT_MASK,
        FeatureMemsSensorFusionCompact,
    )
    from cpsa.adapters.bluecoin.source import BlueCoinSource

    definitions = sys.modules["blue_st_sdk.utils.ble_node_definitions"].FeatureCharacteristic
    definitions.BASE_MASK_TO_FEATURE_DIC.clear()
    definitions.SENSOR_TILE_BOX_MASK_TO_FEATURE_DIC.clear()

    BlueCoinSource("CPSA_L2", "CPSA_R2")

    for table in (
        definitions.BASE_MASK_TO_FEATURE_DIC,
        definitions.SENSOR_TILE_BOX_MASK_TO_FEATURE_DIC,
    ):
        assert table[FUSION_COMPACT_MASK] is FeatureMemsSensorFusionCompact
