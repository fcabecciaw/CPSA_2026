"""``WebGui`` publishes the medical topics and never raises into its callers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np

from cpsa.adapters.gui_web import HANDS_MAX_HZ, WebGui
from cpsa.domain.model import (
    Decision,
    DecisionRecord,
    FaceRegion,
    GateReason,
    HandShape,
    Side,
    StereotypyLabel,
    Tag,
    Verdict,
)
from cpsa.vision.types import Contact, EpisodeReport, KeypointLayout
from tests.contract.protocol_schemas import validate_server_message
from tests.fakes import FakeClock
from tests.unit import vision_fixtures as fx

LAYOUT = KeypointLayout(
    name="TEST",
    names=("a", "b", "c"),
    face=(),
    face_regions=(),
    left_hand=(),
    right_hand=(),
    edges=(),
)


class FakePublisher:
    """Record every state and event exactly as the server receives it."""

    def __init__(self) -> None:
        self.states: list[tuple[str, Mapping[str, Any]]] = []
        self.events: list[tuple[str, Mapping[str, Any]]] = []

    def state(self, type: str, data: Mapping[str, Any]) -> None:
        self.states.append((type, data))

    def event(self, type: str, data: Mapping[str, Any]) -> None:
        self.events.append((type, data))


class BrokenPublisher:
    def state(self, type: str, data: Mapping[str, Any]) -> None:
        raise RuntimeError("client gone")

    def event(self, type: str, data: Mapping[str, Any]) -> None:
        raise RuntimeError("client gone")


def record(decision_value: Decision | None = None) -> DecisionRecord:
    return DecisionRecord(
        timestamp=datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc),
        window_end_ts=1000.0,
        tag=Tag.DANGEROUS,
        tag_changed=True,
        decision=decision_value,
    )


def report() -> EpisodeReport:
    return EpisodeReport(
        episode_id="ep-20261112-0001",
        t_start=1000.5,
        t_end=1002.0,
        active=False,
        contacts=(
            Contact(
                side=Side.RIGHT,
                region=FaceRegion.MOUTH,
                t_start=1001.5,
                duration_s=0.6,
                min_distance_fw=0.2,
                peak_closing_speed_fwps=0.3,
                decel_fwps2=0.1,
                hand_shape=HandShape.FIST,
                head_recoil_fw=0.0,
                head_recoil_speed_fwps=0.0,
            ),
        ),
        contact_count=1,
        repetition_hz=1.4,
        periodicity=0.8,
        interval_cv=0.1,
        speed_trend_fwps_per_s=0.0,
        peak_severity=Tag.DANGEROUS,
        labels=frozenset({StereotypyLabel.HAND_TO_FACE}),
        clock_offset_s=0.0,
        imu_consistency=0.9,
    )


def tracked_frame() -> Any:
    return fx.make_frame(
        7,
        42.0,
        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        conf=np.array([0.9, 0.5, 0.1]),
    )


def make_gui(publisher: object) -> tuple[WebGui, FakeClock]:
    clock = FakeClock(1000.0)
    return WebGui(publisher, LAYOUT, 640, 480, clock), clock  # type: ignore[arg-type]


def test_publish_sends_one_decision_event_and_updates_the_state() -> None:
    publisher = FakePublisher()
    gui, _ = make_gui(publisher)
    value = Decision(
        verdict=Verdict.CONFIRMED,
        reason=GateReason.CONFIRMED,
        actuated=True,
        actuations=(),
        severity=Tag.DANGEROUS,
    )

    gui.publish(record(value))

    assert [message_type for message_type, _ in publisher.events] == ["medical.decision"]
    payload = publisher.events[0][1]
    validate_server_message(
        {
            "v": 1,
            "type": "medical.decision",
            "seq": 1,
            "t": 1000.0,
            "wall": "2026-11-12T08:00:00Z",
            "data": payload,
        }
    )
    assert gui.tag is Tag.DANGEROUS
    snapshot = gui.snapshot_state("connected")
    assert snapshot["last_decision"] == payload
    assert snapshot["tag"] == "DANGEROUS"


def test_hands_are_throttled_to_at_most_15_hz() -> None:
    publisher = FakePublisher()
    gui, clock = make_gui(publisher)
    frame = tracked_frame()

    gui.on_tracked(frame)
    clock.advance(0.5 / HANDS_MAX_HZ)
    gui.on_tracked(frame)
    clock.advance(0.5 / HANDS_MAX_HZ)
    gui.on_tracked(frame)

    assert [message_type for message_type, _ in publisher.states] == [
        "medical.hands",
        "medical.hands",
    ]
    payload = publisher.states[-1][1]
    validate_server_message(
        {
            "v": 1,
            "type": "medical.hands",
            "seq": 1,
            "t": 1000.0,
            "wall": "2026-11-12T08:00:00Z",
            "data": payload,
        }
    )
    assert payload["keypoints"][0] == [10.0, 20.0, 0.9]


def test_episode_start_and_report_update_the_state_and_send_events() -> None:
    publisher = FakePublisher()
    gui, _ = make_gui(publisher)

    gui.on_episode("ep-20261112-0001", 1000.5, True)

    assert publisher.events[0][0] == "medical.episode"
    assert publisher.events[0][1]["active"] is True
    assert gui.snapshot_state("connected")["episode"] == {
        "episode_id": "ep-20261112-0001",
        "t_start": 1000.5,
        "active": True,
    }

    gui.on_episode("ep-20261112-0001", 1000.5, False)
    gui.on_report(report())

    assert [message_type for message_type, _ in publisher.events] == [
        "medical.episode",
        "medical.episode",
    ]
    final = publisher.events[-1][1]
    assert final["episode_id"] == "ep-20261112-0001"
    assert final["contact_count"] == 1
    assert final["contacts"][0]["hand_shape"] == "fist"
    assert gui.snapshot_state("connected")["episode"] == {
        "episode_id": "ep-20261112-0001",
        "t_start": 1000.5,
        "active": False,
    }


def test_publisher_failures_never_escape_an_observer_callback() -> None:
    gui, _ = make_gui(BrokenPublisher())

    gui.publish(record())
    gui.on_episode("ep", 1.0, True)
    gui.on_episode("ep", 1.0, False)
    gui.on_tracked(tracked_frame())
    gui.on_report(report())
