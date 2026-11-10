import pytest

from cpsa.domain import gating
from cpsa.domain.model import Tag, Verdict, VisionAssessment


@pytest.mark.parametrize("tag", [Tag.NON_DANGEROUS, Tag.DANGEROUS])
def test_stereotypy_without_previous_actuation_is_evaluated(tag: Tag) -> None:
    assert gating.should_evaluate(tag, None, now=10.0, cooldown_s=5.0)


@pytest.mark.parametrize("tag", [Tag.NO_CLASS, Tag.NON_STEREOTIPY])
def test_non_stereotypy_is_never_evaluated(tag: Tag) -> None:
    assert not gating.should_evaluate(tag, None, now=10.0, cooldown_s=5.0)


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [(4.999, False), (5.0, True), (7.0, True)],
)
def test_cooldown_boundary(elapsed: float, expected: bool) -> None:
    assert gating.should_evaluate(Tag.DANGEROUS, 100.0, 100.0 + elapsed, 5.0) is expected


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [(Verdict.CONFIRMED, True), (Verdict.UNDECIDED, True), (Verdict.REJECTED, False)],
)
def test_only_rejection_blocks_actuation(verdict: Verdict, expected: bool) -> None:
    assert gating.allows_actuation(verdict) is expected


def test_bare_verdict_becomes_an_assessment_without_severity() -> None:
    assert gating.as_assessment(Verdict.CONFIRMED) == VisionAssessment(Verdict.CONFIRMED)


def test_assessment_passes_through_unchanged() -> None:
    assessment = VisionAssessment(Verdict.CONFIRMED, Tag.DANGEROUS, {"contact_count": 2.0}, "ep-1")
    assert gating.as_assessment(assessment) is assessment


@pytest.mark.parametrize("value", ["CONFIRMED", None, 1, object()], ids=repr)
def test_unknown_verifier_output_has_no_assessment(value: object) -> None:
    assert gating.as_assessment(value) is None
