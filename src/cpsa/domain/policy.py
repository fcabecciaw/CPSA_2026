"""Minimal actuation policy: every available actuator, mild for tag 1, strong for tag 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cpsa.domain.model import (
    Actuation,
    ActuatorKind,
    Tag,
    Verdict,
    VisionAssessment,
    actuator_kind_of,
)

_PARAMS: Mapping[Tag, Mapping[ActuatorKind, Mapping[str, object]]] = {
    Tag.NON_DANGEROUS: {
        ActuatorKind.LED_STRIP: {"color": (255, 255, 255, 0), "intensity": 50, "speed": 100},
        ActuatorKind.METAMOTION: {"duty": 40, "duration": 700},
        ActuatorKind.SPEAKER: {"file": "stereotipia_non_pericolosa_1.mp3"},
    },
    Tag.DANGEROUS: {
        ActuatorKind.LED_STRIP: {"color": (255, 0, 0, 0), "intensity": 100, "speed": 100},
        ActuatorKind.METAMOTION: {"duty": 100, "duration": 900},
        ActuatorKind.SPEAKER: {"file": "stereotipia_pericolosa_1.mp3"},
    },
}


def actuations_for(tag: Tag, actuator_ids: Sequence[str]) -> list[Actuation]:
    """Return one actuation per known actuator id for a stereotypy tag, else none."""
    params_by_kind = _PARAMS.get(tag)
    if params_by_kind is None:
        return []
    actuations = []
    for actuator_id in actuator_ids:
        kind = actuator_kind_of(actuator_id)
        if kind is not None:
            actuations.append(Actuation(actuator_id, dict(params_by_kind[kind])))
    return actuations


def effective_severity(tag: Tag, assessment: VisionAssessment | None) -> Tag:
    """Return the tag that drives actuation for one detection (FR-017, AC-009).

    A confirmed assessment carrying a stereotypy grade overrides the IMU tag, so
    vision can upgrade or downgrade the actuation. Without that grade the IMU tag
    is used, which keeps the fail-open and IMU-only paths unchanged.
    """
    if assessment is None or assessment.verdict is not Verdict.CONFIRMED:
        return tag
    severity = assessment.severity
    if severity is Tag.NON_DANGEROUS or severity is Tag.DANGEROUS:
        return severity
    return tag
