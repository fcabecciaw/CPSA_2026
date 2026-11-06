import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cpsa import ports
from cpsa.adapters.actuators.factory import start_actuators
from cpsa.application.config import ActuatorsConfig, LedStripConfig, MetaMotionConfig, load
from cpsa.domain.model import ActuatorKind
from tests.application.test_config import SHIPPED
from tests.contract.vendor_stubs import Vendor, purge_hardware_adapters, wait_until

DEVICE_MODULES = {
    "metamotion": "cpsa.adapters.actuators.metamotion",
    "speaker": "cpsa.adapters.actuators.speaker",
    "led_strip": "cpsa.adapters.actuators.led_strip",
}


def settings(**enabled: bool) -> ActuatorsConfig:
    base = load(SHIPPED).actuators
    return replace(
        base,
        retry=replace(base.retry, interval_s=0.01, sleep_s=0.01),
        metamotion=MetaMotionConfig(enable=enabled.get("metamotion", False)),
        speaker=replace(base.speaker, enable=enabled.get("speaker", False)),
        led_strip=LedStripConfig(enable=enabled.get("led_strip", False), duration_ms=50),
    )


def test_all_disabled_needs_no_vendor_library() -> None:
    purge_hardware_adapters()  # no vendor stubs installed in this test
    assert start_actuators(settings()) == []
    assert not set(DEVICE_MODULES.values()) & set(sys.modules)


def test_only_enabled_kinds_are_started_and_metamotion_vibrates(vendor: Vendor) -> None:
    vendor.ble_devices = [
        SimpleNamespace(addr="aa:bb", getValueText=lambda code: "MetaWear"),
        SimpleNamespace(addr="cc:dd", getValueText=lambda code: "Headphones"),
    ]
    actuators = start_actuators(settings(metamotion=True))
    try:
        assert [(a.id, a.kind) for a in actuators] == [("meta_aa:bb", ActuatorKind.METAMOTION)]
        assert DEVICE_MODULES["metamotion"] in sys.modules
        assert DEVICE_MODULES["speaker"] not in sys.modules
        assert DEVICE_MODULES["led_strip"] not in sys.modules

        (meta,) = actuators
        assert isinstance(meta, ports.Actuator)
        # Connection feedback: two short pulses.
        assert wait_until(lambda: vendor.haptic_calls == [(100, 200), (100, 200)])
        meta.execute({"duty": 40, "duration": 700})
        assert wait_until(lambda: len(vendor.haptic_calls) == 3)
        assert vendor.haptic_calls[-1] == (40, 700)
    finally:
        for actuator in actuators:
            actuator.stop()


def test_led_strip_shows_color_then_turns_itself_off(vendor: Vendor) -> None:
    vendor.led_devices = [{"ipaddr": "10.0.0.2", "model": "x"}]
    (led,) = start_actuators(settings(led_strip=True))
    try:
        assert (led.id, led.kind) == ("led_10.0.0.2", ActuatorKind.LED_STRIP)
        assert isinstance(led, ports.Actuator)
        assert wait_until(lambda: len(vendor.bulbs) == 1 and len(vendor.bulbs[0].calls) == 4)
        led.execute({"color": (255, 0, 0, 0), "intensity": 100, "speed": 100})
        calls = vendor.bulbs[0].calls
        assert wait_until(lambda: ("turnOff", (), {}) in calls[4:])
        assert calls[4] == ("setRgbw", (255, 0, 0, 0), {"brightness": 100})
    finally:
        led.stop()


def test_speaker_plays_bundled_assets_only_while_connected(vendor: Vendor) -> None:
    from cpsa.adapters.actuators import speaker

    thread = speaker.SpeakerThread("d4:8c:49:c9:dc:9a", 1, 0.01, 0.01)  # not started
    assert (thread.id, thread.kind) == ("speaker_D4:8C:49:C9:DC:9A", ActuatorKind.SPEAKER)
    assert isinstance(thread, ports.Actuator)

    thread.execute({"file": "stereotipia_pericolosa_1.mp3"})
    assert not thread.event.is_set()  # not connected: skipped

    thread.connected = True
    thread.execute({"file": "stereotipia_pericolosa_1.mp3"})
    assert thread.event.is_set()
    assert (
        thread.file
        == Path(speaker.__file__).parents[1] / "assets/audio/stereotipia_pericolosa_1.mp3"
    )
    assert thread.file.is_file()

    thread.execute({"file": "no_such_clip.mp3"})
    assert thread.file.name == "missing_audio_file.mp3"


def test_policy_audio_assets_are_bundled(vendor: Vendor) -> None:
    from cpsa.adapters.actuators.speaker import resolve_audio
    from cpsa.domain.model import Tag
    from cpsa.domain.policy import actuations_for

    for tag in (Tag.NON_DANGEROUS, Tag.DANGEROUS):
        (actuation,) = actuations_for(tag, ["speaker_X"])
        assert resolve_audio(str(actuation.params["file"])).name == actuation.params["file"]


def test_metamotion_reconnection_honors_subsecond_intervals(
    vendor: Vendor, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cpsa.adapters.actuators import metamotion

    attempts: list[float] = []
    thread = metamotion.MetaMotionThread("aa:bb", fast_attempts=2, interval_s=0.2, sleep_s=0.2)

    class Unreachable(metamotion.MetaWear):  # type: ignore[misc,name-defined]
        def connect(self) -> None:
            attempts.append(time.monotonic())
            if len(attempts) == 3:
                thread.stop_event.set()
            raise OSError("device unreachable")

    monkeypatch.setattr(metamotion, "MetaWear", Unreachable)
    thread._reconnection_attempts()

    gaps = [later - earlier for earlier, later in itertools.pairwise(attempts)]
    assert len(attempts) == 3
    assert all(gap >= 0.19 for gap in gaps), gaps
