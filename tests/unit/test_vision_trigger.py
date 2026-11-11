"""Camera-only episode trigger: starts, follow-ups and refractory (FR-028, FR-029).

AC-021 covers the start conditions (2 contacts in 10 s, 1 does not, 2 twelve
seconds apart do not, 3 s of flapping does, 2 s does not) and AC-022 the
refractory period after an episode ends.
"""

from __future__ import annotations

import pytest

from cpsa.domain.model import Tag
from cpsa.vision.trigger import TriggerConfig, TriggerFire, VisionTrigger

STEP_S = 0.1


def _trigger(
    min_contacts: int = 2,
    contact_window_s: float = 10.0,
    oscillation_min_s: float = 3.0,
    refractory_s: float = 5.0,
) -> VisionTrigger:
    return VisionTrigger(
        TriggerConfig(
            min_contacts=min_contacts,
            contact_window_s=contact_window_s,
            oscillation_min_s=oscillation_min_s,
            refractory_s=refractory_s,
        )
    )


def _drive(
    trigger: VisionTrigger,
    start_s: float,
    end_s: float,
    *,
    active: bool = True,
    episode_active: bool = False,
) -> list[TriggerFire]:
    """Feed one oscillation sample per frame and collect the fires."""
    fires: list[TriggerFire] = []
    step = 0
    while True:
        t = round(start_s + step * STEP_S, 9)
        if t > end_s + 1e-9:
            return fires
        trigger.oscillation(t, active)
        fire = trigger.poll(t, episode_active=episode_active)
        if fire is not None:
            fires.append(fire)
        step += 1


# ---------------------------------------------------------------------------
# AC-021: start conditions from contacts
# ---------------------------------------------------------------------------


def test_two_contacts_within_the_window_start_an_episode() -> None:
    trigger = _trigger()
    trigger.contact(1.0, Tag.NON_DANGEROUS)
    trigger.contact(5.0, Tag.DANGEROUS)

    assert trigger.poll(5.0, episode_active=False) == TriggerFire(5.0, Tag.DANGEROUS, "contacts")


def test_one_contact_does_not_start_an_episode() -> None:
    trigger = _trigger()
    trigger.contact(1.0, Tag.DANGEROUS)

    assert trigger.poll(1.0, episode_active=False) is None
    assert trigger.poll(8.0, episode_active=False) is None


def test_two_contacts_wider_apart_than_the_window_do_not_start_an_episode() -> None:
    trigger = _trigger()
    trigger.contact(0.0, Tag.DANGEROUS)
    trigger.contact(12.0, Tag.DANGEROUS)

    assert trigger.poll(12.0, episode_active=False) is None


def test_the_start_severity_is_the_maximum_of_the_window() -> None:
    trigger = _trigger()
    trigger.contact(1.0, Tag.DANGEROUS)
    trigger.contact(2.0, Tag.NON_DANGEROUS)
    trigger.contact(3.0, Tag.NON_DANGEROUS)

    fire = trigger.poll(3.0, episode_active=False)

    assert fire is not None
    assert fire.severity is Tag.DANGEROUS


def test_a_contact_counts_once_however_often_it_is_graded() -> None:
    trigger = _trigger(min_contacts=1)
    trigger.contact(1.0, Tag.NON_DANGEROUS)
    trigger.contact(1.0, Tag.DANGEROUS)

    fire = trigger.poll(1.0, episode_active=False)
    assert fire is not None
    assert fire.severity is Tag.DANGEROUS
    # The repeated grade is the same event, so it is not a follow-up.
    assert trigger.poll(1.1, episode_active=True) is None


def test_every_new_contact_is_a_follow_up_with_its_own_severity() -> None:
    trigger = _trigger()
    trigger.contact(1.0, Tag.NON_DANGEROUS)
    trigger.contact(2.0, Tag.DANGEROUS)
    assert trigger.poll(2.0, episode_active=False) is not None

    trigger.contact(3.0, Tag.NON_DANGEROUS)

    assert trigger.poll(3.0, episode_active=True) == TriggerFire(3.0, Tag.NON_DANGEROUS, "contacts")
    assert trigger.poll(3.1, episode_active=True) is None


def test_a_start_fire_needs_contacts_that_have_not_fired_yet() -> None:
    trigger = _trigger()
    trigger.contact(1.0, Tag.DANGEROUS)
    trigger.contact(5.0, Tag.DANGEROUS)

    assert trigger.poll(5.0, episode_active=False) == TriggerFire(5.0, Tag.DANGEROUS, "contacts")
    # The start did not become an episode (the verifier may still reject it), so
    # the next frames must not fire the same two contacts again.
    assert trigger.poll(5.1, episode_active=False) is None
    assert trigger.poll(5.2, episode_active=False) is None

    # A new start needs a fresh pair inside the window; its severity is the
    # maximum of those fresh contacts.
    trigger.contact(6.0, Tag.NON_DANGEROUS)
    assert trigger.poll(6.0, episode_active=False) is None
    trigger.contact(7.0, Tag.DANGEROUS)
    assert trigger.poll(7.0, episode_active=False) == TriggerFire(7.0, Tag.DANGEROUS, "contacts")


def test_a_counted_contact_that_leaves_the_window_makes_room_for_a_new_start() -> None:
    trigger = _trigger()
    trigger.contact(1.0, Tag.DANGEROUS)
    trigger.contact(5.0, Tag.DANGEROUS)
    assert trigger.poll(5.0, episode_active=False) is not None

    # Contact 1.0 leaves the window at 11.0; only the counted 5.0 one is left.
    assert trigger.poll(12.0, episode_active=False) is None

    trigger.contact(13.0, Tag.NON_DANGEROUS)
    trigger.contact(14.0, Tag.NON_DANGEROUS)

    assert trigger.poll(14.0, episode_active=False) == TriggerFire(
        14.0, Tag.NON_DANGEROUS, "contacts"
    )


# ---------------------------------------------------------------------------
# AC-021: start conditions from oscillation
# ---------------------------------------------------------------------------


def test_three_seconds_of_flapping_start_an_episode() -> None:
    trigger = _trigger()

    fires = _drive(trigger, 0.0, 3.0)

    assert fires == [TriggerFire(3.0, Tag.NON_DANGEROUS, "oscillation")]


def test_two_seconds_of_flapping_do_not_start_an_episode() -> None:
    trigger = _trigger()

    assert _drive(trigger, 0.0, 2.0) == []


def test_a_gap_restarts_the_required_oscillation_run() -> None:
    trigger = _trigger()
    assert _drive(trigger, 0.0, 2.0) == []

    trigger.oscillation(2.1, False)

    # Continuity restarts at 2.2, so 3 s of it are only complete at 5.2.
    assert _drive(trigger, 2.2, 5.1) == []
    assert _drive(trigger, 5.2, 5.2) == [TriggerFire(5.2, Tag.NON_DANGEROUS, "oscillation")]


def test_oscillation_follow_ups_fire_once_per_required_run() -> None:
    trigger = _trigger()
    assert _drive(trigger, 0.0, 3.0) == [TriggerFire(3.0, Tag.NON_DANGEROUS, "oscillation")]

    follow_ups = _drive(trigger, 3.1, 6.0, episode_active=True)

    assert follow_ups == [TriggerFire(6.0, Tag.NON_DANGEROUS, "oscillation")]


# ---------------------------------------------------------------------------
# AC-022: refractory and clear
# ---------------------------------------------------------------------------


def test_episode_end_returns_a_clear_fire() -> None:
    trigger = _trigger()

    assert trigger.episode_ended(10.0) == TriggerFire(10.0, Tag.NON_STEREOTIPY, "clear")


def test_no_start_within_the_refractory_period_and_one_after_it() -> None:
    trigger = _trigger(refractory_s=5.0)
    trigger.episode_ended(10.0)
    trigger.contact(11.0, Tag.DANGEROUS)
    trigger.contact(12.0, Tag.DANGEROUS)

    assert trigger.poll(12.0, episode_active=False) is None
    assert trigger.poll(14.9, episode_active=False) is None
    assert trigger.poll(15.0, episode_active=False) == TriggerFire(15.0, Tag.DANGEROUS, "contacts")


def test_the_contacts_of_the_finished_episode_do_not_count_again() -> None:
    trigger = _trigger()
    trigger.contact(0.0, Tag.DANGEROUS)
    trigger.contact(1.0, Tag.DANGEROUS)
    assert trigger.poll(1.0, episode_active=False) is not None

    trigger.episode_ended(2.0)
    trigger.contact(3.0, Tag.DANGEROUS)

    # A single new contact after the refractory is not a new episode.
    assert trigger.poll(7.1, episode_active=False) is None


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def test_a_contact_needs_a_finite_start_and_a_stereotypy_grade() -> None:
    trigger = _trigger()

    with pytest.raises(ValueError, match="finite"):
        trigger.contact(float("nan"), Tag.DANGEROUS)
    with pytest.raises(ValueError, match="stereotypy"):
        trigger.contact(1.0, Tag.NON_STEREOTIPY)


def test_poll_and_oscillation_reject_non_finite_times() -> None:
    trigger = _trigger()

    with pytest.raises(ValueError, match="finite"):
        trigger.oscillation(float("inf"), True)
    with pytest.raises(ValueError, match="finite"):
        trigger.poll(float("nan"), episode_active=False)
    with pytest.raises(ValueError, match="finite"):
        trigger.episode_ended(float("nan"))
