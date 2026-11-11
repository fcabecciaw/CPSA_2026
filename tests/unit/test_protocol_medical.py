"""``cpsa.protocol.medical`` payloads validate against the v1 schemas (IF-451..454)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from cpsa.domain.model import (
    Actuation,
    ActuatorKind,
    Decision,
    DecisionRecord,
    FaceRegion,
    GateReason,
    HandShape,
    Side,
    StereotypyLabel,
    Tag,
    Verdict,
    VisionAssessment,
)
from cpsa.protocol.medical import ImuStatus, decision, episode, hands, state
from cpsa.vision.types import Contact, EpisodeReport, KeypointLayout, TrackedFrame
from tests.contract.protocol_schemas import validate_server_message
from tests.unit import vision_fixtures as fx

WALL = "2026-11-12T08:00:00Z"
LAYOUT = KeypointLayout(
    name="TEST",
    names=("a", "b", "c"),
    face=(),
    face_regions=(),
    left_hand=(),
    right_hand=(),
    edges=(),
)


def envelope(message_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"v": 1, "type": message_type, "seq": 1, "t": 1000.0, "wall": WALL, "data": data}


def record(
    tag: Tag = Tag.DANGEROUS,
    *,
    tag_changed: bool = True,
    decision_value: Decision | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        timestamp=datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc),
        window_end_ts=1000.0,
        tag=tag,
        tag_changed=tag_changed,
        decision=decision_value,
    )


def report(**overrides: Any) -> EpisodeReport:
    contact = Contact(
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
    )
    base = EpisodeReport(
        episode_id="ep-20261112-0001",
        t_start=1000.5,
        t_end=1002.0,
        active=False,
        contacts=(contact,),
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
    return replace(base, **overrides)


def frame(xy: np.ndarray, conf: np.ndarray) -> TrackedFrame:
    return fx.make_frame(7, 42.0, xy, conf=conf)


# ---------------------------------------------------------------------------
# medical.decision
# ---------------------------------------------------------------------------


def test_decision_with_vision_and_actuations_matches_the_schema() -> None:
    assessment = VisionAssessment(
        verdict=Verdict.CONFIRMED,
        severity=Tag.DANGEROUS,
        features={"contact_count": 2.0, "repetition_hz": 1.4},
        episode_id="ep-20261112-0001",
    )
    value = Decision(
        verdict=Verdict.CONFIRMED,
        reason=GateReason.CONFIRMED,
        actuated=True,
        actuations=(
            Actuation("meta_AA", {"duty": 100, "duration": 900}),
            Actuation("speaker_CC", {"file": "stereotipia_pericolosa_1.mp3"}),
        ),
        severity=Tag.DANGEROUS,
        assessment=assessment,
    )

    payload = decision(record(decision_value=value))

    validate_server_message(envelope("medical.decision", payload))
    assert payload["tag"] == "DANGEROUS"
    assert payload["tag_changed"] is True
    assert payload["verdict"] == "CONFIRMED"
    assert payload["reason"] == "CONFIRMED"
    assert payload["severity"] == "DANGEROUS"
    assert payload["vision"] == {
        "verdict": "CONFIRMED",
        "severity": "DANGEROUS",
        "episode_id": "ep-20261112-0001",
        "features": {"contact_count": 2.0, "repetition_hz": 1.4},
    }
    assert payload["actuations"] == [
        {"actuator_id": "meta_AA", "kind": "metamotion"},
        {"actuator_id": "speaker_CC", "kind": "speaker"},
    ]


def test_decision_without_vision_uses_nulls_and_no_actuations() -> None:
    value = Decision(
        verdict=Verdict.UNDECIDED,
        reason=GateReason.TIMEOUT,
        actuated=False,
        actuations=(),
        severity=None,
        assessment=None,
    )

    payload = decision(record(decision_value=value))

    validate_server_message(envelope("medical.decision", payload))
    assert payload["verdict"] == "UNDECIDED"
    assert payload["reason"] == "TIMEOUT"
    assert payload["severity"] is None
    assert payload["vision"] is None
    assert payload["actuations"] == []


def test_tag_only_record_has_null_decision_fields() -> None:
    payload = decision(record(tag=Tag.NON_STEREOTIPY, tag_changed=True))

    validate_server_message(envelope("medical.decision", payload))
    assert payload["tag"] == "NON_STEREOTIPY"
    assert payload["verdict"] is None
    assert payload["reason"] is None
    assert payload["severity"] is None
    assert payload["vision"] is None
    assert payload["actuations"] == []


def test_actuator_kind_falls_back_to_the_policy_parameters() -> None:
    payload = decision(
        record(
            decision_value=Decision(
                verdict=Verdict.CONFIRMED,
                reason=GateReason.CONFIRMED,
                actuated=True,
                actuations=(
                    Actuation("external-1", {"file": "alarm.mp3"}),
                    Actuation("external-2", {"color": (255, 0, 0, 0)}),
                    Actuation("external-3", {}),
                ),
                severity=Tag.DANGEROUS,
            )
        )
    )

    kinds = [actuation["kind"] for actuation in payload["actuations"]]
    assert kinds == [ActuatorKind.SPEAKER.value, ActuatorKind.LED_STRIP.value, None]


# ---------------------------------------------------------------------------
# medical.episode
# ---------------------------------------------------------------------------


def test_episode_with_contacts_matches_the_schema() -> None:
    payload = episode(report())

    validate_server_message(envelope("medical.episode", payload))
    assert payload["episode_id"] == "ep-20261112-0001"
    assert payload["active"] is False
    assert payload["contact_count"] == 1
    assert payload["repetition_hz"] == 1.4
    assert payload["peak_severity"] == "DANGEROUS"
    assert payload["labels"] == ["hand_to_face"]
    assert payload["contacts"] == [
        {
            "side": "right",
            "region": "mouth",
            "t_start": 1001.5,
            "duration_s": 0.6,
            "hand_shape": "fist",
        }
    ]


def test_episode_without_contacts_or_severity_matches_the_schema() -> None:
    payload = episode(
        report(
            contacts=(),
            contact_count=0,
            repetition_hz=None,
            peak_severity=None,
            labels=frozenset(),
        )
    )

    validate_server_message(envelope("medical.episode", payload))
    assert payload["contacts"] == []
    assert payload["repetition_hz"] is None
    assert payload["peak_severity"] is None
    assert payload["labels"] == []


def test_episode_labels_have_a_deterministic_order() -> None:
    labels = frozenset({StereotypyLabel.ROCKING, StereotypyLabel.HAND_TO_FACE})
    payload = episode(report(labels=labels))
    assert payload["labels"] == ["hand_to_face", "rocking"]


# ---------------------------------------------------------------------------
# medical.hands
# ---------------------------------------------------------------------------


def test_hands_matches_the_schema_in_frame_pixels() -> None:
    tracked = frame(
        np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        np.array([0.9, 0.5, 0.1]),
    )

    payload = hands(tracked, LAYOUT, 640, 480)

    validate_server_message(envelope("medical.hands", payload))
    assert payload["layout"] == "TEST"
    assert payload["w"] == 640 and payload["h"] == 480
    assert payload["keypoints"] == [[10.0, 20.0, 0.9], [30.0, 40.0, 0.5], [50.0, 60.0, 0.1]]
    assert payload["subject_box"] == [0.0, 0.0, 50.0, 60.0]


def test_hands_reports_unseen_keypoints_at_the_origin() -> None:
    tracked = frame(
        np.array([[np.nan, np.nan], [30.0, 40.0], [np.nan, np.nan]]),
        np.array([0.0, 0.5, 0.0]),
    )

    payload = hands(replace(tracked, box=None), LAYOUT, 640, 480)

    validate_server_message(envelope("medical.hands", payload))
    assert payload["keypoints"] == [[0.0, 0.0, 0.0], [30.0, 40.0, 0.5], [0.0, 0.0, 0.0]]
    assert payload["subject_box"] is None


# ---------------------------------------------------------------------------
# medical.state
# ---------------------------------------------------------------------------


def test_state_without_episode_or_decision_matches_the_schema() -> None:
    payload = state("searching", None, None, None)

    validate_server_message(envelope("medical.state", payload))
    assert payload == {
        "imu": "searching",
        "tag": None,
        "episode": None,
        "last_decision": None,
        "trigger": "imu",
    }


def test_state_carries_the_vision_trigger_mode() -> None:
    payload = state("n/a", None, None, None, trigger="vision")

    validate_server_message(envelope("medical.state", payload))
    assert payload["imu"] == "n/a"
    assert payload["trigger"] == "vision"


def test_state_carries_the_episode_and_the_last_decision() -> None:
    last = decision(record(tag=Tag.DANGEROUS))
    payload = state(
        "connected",
        Tag.DANGEROUS,
        {"episode_id": "ep-20261112-0001", "t_start": 1000.5, "active": True},
        last,
    )

    validate_server_message(envelope("medical.state", payload))
    assert payload["imu"] == "connected"
    assert payload["tag"] == "DANGEROUS"
    assert payload["episode"] == {
        "episode_id": "ep-20261112-0001",
        "t_start": 1000.5,
        "active": True,
    }
    assert payload["last_decision"] == last


def test_non_finite_vision_features_become_null() -> None:
    value = Decision(
        verdict=Verdict.CONFIRMED,
        reason=GateReason.CONFIRMED,
        actuated=False,
        actuations=(),
        severity=Tag.DANGEROUS,
        assessment=VisionAssessment(
            verdict=Verdict.CONFIRMED,
            severity=Tag.DANGEROUS,
            features={"good": 1.0, "bad": float("nan")},
            episode_id=None,
        ),
    )

    payload = decision(record(decision_value=value))

    validate_server_message(envelope("medical.decision", payload))
    assert payload["vision"]["features"] == {"good": 1.0, "bad": None}


@pytest.mark.parametrize("imu", ["n/a", "searching", "connected", "lost"])
def test_every_imu_status_is_a_valid_state(imu: ImuStatus) -> None:
    validate_server_message(envelope("medical.state", state(imu, None, None, None)))
