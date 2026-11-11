"""The rock/paper/scissors state machine under a fake clock (AC-09)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from cpsa.vision.hand3d import Hand3D
from cpsa.vision.rps_game import (
    OUTCOMES,
    RPS_MOVES,
    RPS_PHASES,
    RpsGame,
    RpsGameConfig,
    RpsScore,
    RpsState,
)

READY_S = 0.5
STEP_S = 0.7
SHOOT_S = 0.4
RESULT_S = 3.0
RESET_S = 20.0
COUNTDOWN_START = READY_S
SHOOT_START = COUNTDOWN_START + 3 * STEP_S
RESULT_START = SHOOT_START + SHOOT_S


def hand(area: float = 10.0, offset: float = 0.0) -> Hand3D:
    """A hand whose box area is ``area`` (the game plays the largest one)."""

    kpts2d = np.zeros((21, 3), dtype=np.float32)
    kpts2d[:, 0] = offset + np.arange(21, dtype=np.float32) * 2.0
    kpts2d[:, 1] = offset + np.arange(21, dtype=np.float32)
    kpts2d[:, 2] = 0.9
    kpts3d = np.zeros((21, 4), dtype=np.float32)
    return Hand3D(
        score=0.9,
        box=(offset, offset, offset + area, offset + area),
        kpts2d=kpts2d,
        kpts3d=kpts3d,
    )


class ScriptedRandom:
    """A ``SystemRandom`` stand-in that returns the scripted moves in order."""

    def __init__(self, *moves: str) -> None:
        self.moves = list(moves) or ["rock"]
        self.calls = 0

    def choice(self, seq: Sequence[str]) -> str:
        assert set(seq) == set(RPS_MOVES)
        move = self.moves[min(self.calls, len(self.moves) - 1)]
        self.calls += 1
        return move


def constant(name: str) -> Callable[[Hand3D], tuple[str, float]]:
    return lambda _hand: (name, 0.9)


def game(rng: ScriptedRandom | None = None) -> RpsGame:
    return RpsGame(
        RpsGameConfig(
            ready_s=READY_S,
            countdown_step_s=STEP_S,
            shoot_s=SHOOT_S,
            result_s=RESULT_S,
            score_reset_s=RESET_S,
        ),
        rng=rng if rng is not None else ScriptedRandom("rock"),
    )


def walk_to_shoot(game_: RpsGame) -> RpsState:
    """Take the game from waiting to the start of the shoot window."""

    assert game_.update(0.0, [hand()], constant("rock")).phase == "ready"
    assert game_.update(READY_S, [hand()], constant("rock")).count == 3
    assert game_.update(COUNTDOWN_START + STEP_S, [hand()], constant("rock")).count == 2
    assert game_.update(COUNTDOWN_START + 2 * STEP_S, [hand()], constant("rock")).count == 1
    state = game_.update(SHOOT_START, [hand()], constant("rock"))
    assert state.phase == "shoot"
    return state


# -- phases ----------------------------------------------------------------


def test_the_vocabulary_matches_the_protocol() -> None:
    from cpsa.protocol import hands as protocol_hands

    assert RPS_PHASES == protocol_hands.RPS_PHASES
    assert RPS_MOVES == protocol_hands.RPS_MOVES
    assert OUTCOMES == protocol_hands.OUTCOMES


def test_a_new_game_waits_for_a_hand() -> None:
    state = game().update(0.0, [], constant("rock"))

    assert state.phase == "waiting"
    assert state.count is None
    assert state.player is None
    assert state.hand is None
    assert state.score == RpsScore()


def test_one_hand_for_ready_s_starts_the_countdown() -> None:
    game_ = game()

    assert game_.update(10.0, [hand()], constant("rock")).phase == "ready"
    assert game_.update(10.0 + READY_S - 0.01, [hand()], constant("rock")).phase == "ready"
    state = game_.update(10.0 + READY_S, [hand()], constant("rock"))

    assert state.phase == "countdown"
    assert state.count == 3


def test_the_countdown_steps_three_two_one_every_step() -> None:
    game_ = game()
    game_.update(0.0, [hand()], constant("rock"))

    counts = [
        game_.update(COUNTDOWN_START + index * STEP_S, [hand()], constant("rock")).count
        for index in range(3)
    ]

    assert counts == [3, 2, 1]
    assert game_.update(SHOOT_START, [hand()], constant("rock")).phase == "shoot"


def test_a_hand_leaving_during_ready_returns_to_waiting() -> None:
    game_ = game()
    game_.update(0.0, [hand()], constant("rock"))

    state = game_.update(0.2, [], constant("rock"))

    assert state.phase == "waiting"
    # Back to ready needs a full ready_s again.
    assert game_.update(0.3, [hand()], constant("rock")).phase == "ready"
    assert game_.update(0.7, [hand()], constant("rock")).phase == "ready"


# -- voting and outcomes ---------------------------------------------------


def test_the_majority_vote_decides_and_the_score_updates() -> None:
    rng = ScriptedRandom("scissors")
    game_ = game(rng)

    walk_to_shoot(game_)

    game_.update(SHOOT_START + 0.1, [hand()], constant("rock"))
    game_.update(SHOOT_START + 0.2, [hand()], constant("rock"))
    game_.update(SHOOT_START + 0.3, [hand()], constant("paper"))
    state = game_.update(RESULT_START, [hand()], constant("paper"))

    assert state.phase == "result"
    assert state.player == "rock"
    assert state.player_p == pytest.approx(2 / 3)
    assert state.board == "scissors"
    assert state.outcome == "win"
    assert state.score == RpsScore(player=1, board=0, draws=0)
    assert rng.calls == 1


@pytest.mark.parametrize(
    ("player", "board", "outcome", "score"),
    [
        ("rock", "scissors", "win", RpsScore(1, 0, 0)),
        ("rock", "paper", "lose", RpsScore(0, 1, 0)),
        ("paper", "paper", "draw", RpsScore(0, 0, 1)),
        ("scissors", "paper", "win", RpsScore(1, 0, 0)),
    ],
)
def test_the_win_lose_draw_table(player: str, board: str, outcome: str, score: RpsScore) -> None:
    rng = ScriptedRandom(board)
    game_ = game(rng)
    walk_to_shoot(game_)

    game_.update(SHOOT_START + 0.1, [hand()], constant(player))
    state = game_.update(RESULT_START, [hand()], constant(player))

    assert (state.player, state.board, state.outcome) == (player, board, outcome)
    assert state.score == score


def test_a_vote_share_below_sixty_percent_is_unclear() -> None:
    rng = ScriptedRandom("paper")
    game_ = game(rng)
    walk_to_shoot(game_)

    for index, name in enumerate(("rock", "rock", "paper", "scissors")):
        game_.update(SHOOT_START + 0.05 * (index + 1), [hand()], constant(name))
    state = game_.update(RESULT_START, [hand()], constant("rock"))

    assert state.outcome == "unclear"
    assert state.player_p == pytest.approx(0.5)
    assert state.board is None
    assert state.score == RpsScore()
    assert rng.calls == 0


def test_a_none_majority_is_unclear() -> None:
    rng = ScriptedRandom("paper")
    game_ = game(rng)
    walk_to_shoot(game_)

    game_.update(SHOOT_START + 0.1, [hand()], constant("none"))
    state = game_.update(RESULT_START, [hand()], constant("none"))

    assert state.outcome == "unclear"
    assert state.player == "none"
    assert state.board is None
    assert state.score == RpsScore()


def test_an_unknown_class_votes_as_none() -> None:
    game_ = game()
    walk_to_shoot(game_)

    game_.update(SHOOT_START + 0.1, [hand()], constant("horns"))
    state = game_.update(RESULT_START, [hand()], constant("horns"))

    assert state.player == "none"
    assert state.outcome == "unclear"


def test_no_votes_during_the_shoot_window_is_unclear() -> None:
    game_ = game()
    walk_to_shoot(game_)

    state = game_.update(RESULT_START, [], constant("rock"))

    assert state.phase == "result"
    assert state.outcome == "unclear"
    assert state.player is None
    assert state.hand is None


def test_the_largest_hand_plays() -> None:
    game_ = game()
    small = hand(area=5.0, offset=0.0)
    large = hand(area=50.0, offset=100.0)
    seen: list[Hand3D] = []

    def classify(selected: Hand3D) -> tuple[str, float]:
        seen.append(selected)
        return "rock", 0.9

    game_.update(0.0, [small, large], classify)
    assert game_.update(0.0, [large, small], classify).hand is large
    walk_to_shoot(game_)
    game_.update(SHOOT_START + 0.1, [small, large], classify)
    game_.update(RESULT_START, [small, large], classify)

    assert seen and all(selected is large for selected in seen)


# -- result and score ------------------------------------------------------


def test_the_result_is_shown_for_three_seconds_then_the_next_round_readies() -> None:
    game_ = game(ScriptedRandom("scissors"))
    walk_to_shoot(game_)
    game_.update(SHOOT_START + 0.1, [hand()], constant("rock"))
    assert game_.update(RESULT_START, [hand()], constant("rock")).phase == "result"

    state = game_.update(RESULT_START + RESULT_S - 0.01, [hand()], constant("rock"))
    assert state.phase == "result"
    state = game_.update(RESULT_START + RESULT_S, [hand()], constant("rock"))
    assert state.phase == "ready"
    assert state.player is None and state.player_p is None and state.outcome is None
    assert state.score == RpsScore(player=1, board=0, draws=0)
    # The next round needs a fresh ready_s.
    assert game_.update(RESULT_START + RESULT_S + READY_S, [hand()], constant("rock")).phase == (
        "countdown"
    )


def test_the_score_resets_after_twenty_seconds_without_a_hand() -> None:
    game_ = game(ScriptedRandom("scissors"))
    walk_to_shoot(game_)
    game_.update(SHOOT_START + 0.1, [hand()], constant("rock"))
    state = game_.update(RESULT_START, [hand()], constant("rock"))
    assert state.score.player == 1

    last_hand = RESULT_START + RESULT_S
    game_.update(last_hand, [hand()], constant("rock"))
    assert game_.update(last_hand + 10.0, [], constant("rock")).score.player == 1
    state = game_.update(last_hand + RESET_S, [], constant("rock"))

    assert state.score == RpsScore()
    assert state.phase == "waiting"


def test_the_hand_of_the_current_frame_is_reported() -> None:
    game_ = game()
    current = hand(area=7.0)

    state = game_.update(0.0, [current], constant("rock"))

    assert state.hand is current


def test_state_is_immutable_and_the_snapshot_is_a_copy() -> None:
    game_ = game()
    snapshot = game_.state

    game_.update(0.0, [hand()], constant("rock"))

    assert snapshot.phase == "waiting"
    assert game_.state.phase == "ready"


# -- configuration ---------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["ready_s", "countdown_step_s", "shoot_s", "result_s", "score_reset_s"],
)
def test_zero_durations_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RpsGameConfig(**{field: 0.0})


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_min_share_must_be_a_fraction(value: float) -> None:
    with pytest.raises(ValueError, match="min_share"):
        RpsGameConfig(min_share=value)
