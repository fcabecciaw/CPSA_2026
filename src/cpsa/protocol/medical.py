"""Builders for the ``medical.*`` payloads (interface IF-451..IF-454).

Pure functions: they map the domain and vision types to a JSON-ready ``data``
object. The envelope (``v``, ``seq``, ``t``, ``wall``) is added by the server per
connection (IF-102). Conventions from ``docs/protocol/v1/README.md`` apply:
``Tag`` is serialized by name, every other enum by value, and a missing or
not-applicable value is ``null``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final, Literal

from cpsa.domain.model import (
    Actuation,
    ActuatorKind,
    Decision,
    DecisionRecord,
    Tag,
    VisionAssessment,
    actuator_kind_of,
)
from cpsa.vision.types import EpisodeReport, KeypointLayout, TrackedFrame

#: ``medical.state.imu`` and ``Mode.health()["imu"]`` values (IF-301, F4).
ImuStatus = Literal["n/a", "searching", "connected", "lost"]

#: Distinctive actuator-policy parameter keys, used only when the id prefix is unknown.
_KIND_BY_PARAM: Final[Mapping[str, ActuatorKind]] = {
    "file": ActuatorKind.SPEAKER,
    "color": ActuatorKind.LED_STRIP,
    "duty": ActuatorKind.METAMOTION,
}


def decision(record: DecisionRecord) -> dict[str, Any]:
    """Build the ``medical.decision`` payload for one record (IF-452).

    A record that only reports a tag change has no decision: every decision field
    is then ``null`` (``actuations`` is an empty list, as the schema requires).
    """
    payload: dict[str, Any] = {
        "tag": record.tag.name,
        "tag_changed": bool(record.tag_changed),
    }
    payload.update(_decision_fields(record.decision))
    return payload


def episode(report: EpisodeReport) -> dict[str, Any]:
    """Build the ``medical.episode`` payload for one episode report (IF-453)."""
    return {
        "episode_id": report.episode_id,
        "active": bool(report.active),
        "contact_count": int(report.contact_count),
        "repetition_hz": _number(report.repetition_hz),
        "peak_severity": None if report.peak_severity is None else report.peak_severity.name,
        "labels": sorted(label.value for label in report.labels),
        "contacts": [
            {
                "side": contact.side.value,
                "region": contact.region.value,
                "t_start": float(contact.t_start),
                "duration_s": float(contact.duration_s),
                "hand_shape": contact.hand_shape.value,
            }
            for contact in report.contacts
        ],
    }


def hands(tracked: TrackedFrame, layout: KeypointLayout, w: int, h: int) -> dict[str, Any]:
    """Build the ``medical.hands`` payload for one analyzed frame (IF-454).

    Keypoints are in camera-frame pixels, in ``layout`` order. A keypoint the
    tracker never measured has non-finite coordinates and confidence 0.
    """
    keypoints = [
        [_coordinate(x), _coordinate(y), _coordinate(conf)]
        for (x, y), conf in zip(tracked.xy, tracked.conf, strict=True)
    ]
    box = tracked.box
    return {
        "layout": layout.name,
        "w": int(w),
        "h": int(h),
        "keypoints": keypoints,
        "subject_box": None if box is None else [_coordinate(value) for value in box],
    }


def state(
    imu: ImuStatus,
    tag: Tag | None,
    episode: Mapping[str, Any] | None,
    last_decision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the ``medical.state`` payload (IF-451)."""
    return {
        "imu": imu,
        "tag": None if tag is None else tag.name,
        "episode": None if episode is None else dict(episode),
        "last_decision": None if last_decision is None else dict(last_decision),
    }


def _decision_fields(decision: Decision | None) -> dict[str, Any]:
    if decision is None:
        return {
            "verdict": None,
            "reason": None,
            "severity": None,
            "vision": None,
            "actuations": [],
        }
    return {
        "verdict": decision.verdict.value,
        "reason": decision.reason.value,
        "severity": None if decision.severity is None else decision.severity.name,
        "vision": _vision(decision.assessment),
        "actuations": [_actuation(actuation) for actuation in decision.actuations],
    }


def _vision(assessment: VisionAssessment | None) -> dict[str, Any] | None:
    if assessment is None:
        return None
    return {
        "verdict": assessment.verdict.value,
        "severity": None if assessment.severity is None else assessment.severity.name,
        "episode_id": assessment.episode_id,
        "features": {name: _number(value) for name, value in assessment.features.items()},
    }


def _actuation(actuation: Actuation) -> dict[str, Any]:
    kind = actuator_kind_of(actuation.actuator_id)
    if kind is None:
        kind = _kind_from_params(actuation.params)
    return {
        "actuator_id": actuation.actuator_id,
        "kind": None if kind is None else kind.value,
    }


def _kind_from_params(params: Mapping[str, object]) -> ActuatorKind | None:
    """Identify a kind from its policy parameters when the id has no known prefix."""
    for key, kind in _KIND_BY_PARAM.items():
        if key in params:
            return kind
    return None


def _number(value: float | None) -> float | None:
    """Return ``value`` as a JSON number, or ``None`` when absent or not finite."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _coordinate(value: float) -> float:
    """Return a finite coordinate; an unseen keypoint is reported at the origin."""
    number = float(value)
    return number if math.isfinite(number) else 0.0
