import pytest

from cpsa.domain import gating
from cpsa.domain.model import Tag, Verdict


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
