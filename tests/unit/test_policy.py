import pytest

from cpsa.domain.model import Actuation, Tag, Verdict, VisionAssessment
from cpsa.domain.policy import actuations_for, effective_severity

IDS = ["meta_AA:BB", "speaker_D4:8C", "led_192.168.1.5"]


def test_tag_1_is_mild_on_every_actuator() -> None:
    assert actuations_for(Tag.NON_DANGEROUS, IDS) == [
        Actuation("meta_AA:BB", {"duty": 40, "duration": 700}),
        Actuation("speaker_D4:8C", {"file": "stereotipia_non_pericolosa_1.mp3"}),
        Actuation("led_192.168.1.5", {"color": (255, 255, 255, 0), "intensity": 50, "speed": 100}),
    ]


def test_tag_2_is_strong_on_every_actuator() -> None:
    assert actuations_for(Tag.DANGEROUS, IDS) == [
        Actuation("meta_AA:BB", {"duty": 100, "duration": 900}),
        Actuation("speaker_D4:8C", {"file": "stereotipia_pericolosa_1.mp3"}),
        Actuation("led_192.168.1.5", {"color": (255, 0, 0, 0), "intensity": 100, "speed": 100}),
    ]


@pytest.mark.parametrize("tag", [Tag.NO_CLASS, Tag.NON_STEREOTIPY])
def test_non_stereotypy_produces_no_actuation(tag: Tag) -> None:
    assert actuations_for(tag, IDS) == []


def test_unknown_prefix_is_skipped() -> None:
    assert [a.actuator_id for a in actuations_for(Tag.DANGEROUS, ["fan_1", "meta_X"])] == ["meta_X"]


def test_returned_params_are_independent_copies() -> None:
    first = actuations_for(Tag.DANGEROUS, ["meta_X"])[0]
    first.params["duty"] = 0  # type: ignore[index]
    assert actuations_for(Tag.DANGEROUS, ["meta_X"])[0].params["duty"] == 100


@pytest.mark.parametrize(
    ("imu_tag", "vision_severity", "expected"),
    [
        (Tag.DANGEROUS, Tag.NON_DANGEROUS, Tag.NON_DANGEROUS),
        (Tag.NON_DANGEROUS, Tag.DANGEROUS, Tag.DANGEROUS),
        (Tag.NON_DANGEROUS, Tag.NON_DANGEROUS, Tag.NON_DANGEROUS),
        (Tag.DANGEROUS, Tag.DANGEROUS, Tag.DANGEROUS),
        (Tag.DANGEROUS, None, Tag.DANGEROUS),
        (Tag.NON_DANGEROUS, Tag.NON_STEREOTIPY, Tag.NON_DANGEROUS),
    ],
    ids=[
        "downgrade",
        "upgrade",
        "mild-both",
        "strong-both",
        "no-vision-grade",
        "not-a-severity",
    ],
)
def test_confirmed_assessment_severity_overrides_the_imu_tag(
    imu_tag: Tag, vision_severity: Tag | None, expected: Tag
) -> None:
    assessment = VisionAssessment(Verdict.CONFIRMED, vision_severity)
    assert effective_severity(imu_tag, assessment) is expected


@pytest.mark.parametrize("verdict", [Verdict.UNDECIDED, Verdict.REJECTED])
def test_non_confirmed_assessment_keeps_the_imu_tag(verdict: Verdict) -> None:
    assessment = VisionAssessment(verdict, Tag.DANGEROUS)
    assert effective_severity(Tag.NON_DANGEROUS, assessment) is Tag.NON_DANGEROUS


def test_missing_assessment_keeps_the_imu_tag() -> None:
    assert effective_severity(Tag.DANGEROUS, None) is Tag.DANGEROUS
