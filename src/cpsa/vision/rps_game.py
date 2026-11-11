"""The rock/paper/scissors round state machine (AC-09).

Pure game logic: the caller supplies the monotonic time of every update, a
``classify`` callable mapping a hand to ``(class, probability)``, and the random
source that picks the board's move. Nothing here reads the clock, the camera or
the network, so the transitions are exact under a fake clock in the tests.

Phases: ``waiting`` (no hand) -> ``ready`` (one hand for ``ready_s``) ->
``countdown`` (3, 2, 1 every ``countdown_step_s``) -> ``shoot`` (majority vote
over ``shoot_s``) -> ``result`` (``result_s``) -> ``ready``. The largest visible
hand plays. A vote share below ``min_share``, or a ``none`` majority, gives
``unclear`` and leaves the score alone; the score resets after ``score_reset_s``
without a hand.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from cpsa.vision.hand3d import Hand3D

__all__ = [
    "OUTCOMES",
    "RPS_MOVES",
    "RPS_PHASES",
    "RpsGame",
    "RpsGameConfig",
    "RpsScore",
    "RpsState",
]

WAITING = "waiting"
READY = "ready"
COUNTDOWN = "countdown"
SHOOT = "shoot"
RESULT = "result"

RPS_PHASES: tuple[str, ...] = (WAITING, READY, COUNTDOWN, SHOOT, RESULT)
"""The ``rps.state.phase`` values, in round order."""

RPS_MOVES: tuple[str, ...] = ("rock", "paper", "scissors")
"""The three playable gestures."""

OUTCOMES: tuple[str, ...] = ("win", "lose", "draw", "unclear")
"""The ``rps.state.outcome`` values."""

NONE_CLASS = "none"
"""The classifier class that is not a gesture."""

_BEATS: dict[str, str] = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
_COUNTDOWN_FROM = 3
_TIME_EPS = 1e-9
"""Boundary tolerance: a step duration counts as reached at its nominal time."""


class RpsRandom(Protocol):
    """The random seam: anything with ``random.Random.choice``."""

    def choice(self, seq: Sequence[str]) -> str: ...


@dataclass(frozen=True)
class RpsScore:
    """Rounds won by the player, by the board, and drawn."""

    player: int = 0
    board: int = 0
    draws: int = 0


@dataclass(frozen=True)
class RpsGameConfig:
    """Round timings and thresholds.

    Raises:
        ValueError: If a duration is not positive or ``min_share`` is not in
            ``[0, 1]``.
    """

    ready_s: float = 0.5
    countdown_step_s: float = 0.7
    shoot_s: float = 0.4
    result_s: float = 3.0
    score_reset_s: float = 20.0
    min_share: float = 0.6

    def __post_init__(self) -> None:
        for name in ("ready_s", "countdown_step_s", "shoot_s", "result_s", "score_reset_s"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.min_share <= 1.0:
            raise ValueError(f"min_share must be in [0, 1], got {self.min_share}")


@dataclass(frozen=True)
class RpsState:
    """One snapshot of the round, ready for :func:`cpsa.protocol.hands.rps_state`."""

    phase: str = WAITING
    count: int | None = None
    player: str | None = None
    player_p: float | None = None
    board: str | None = None
    outcome: str | None = None
    score: RpsScore = field(default_factory=RpsScore)
    hand: Hand3D | None = None


class RpsGame:
    """Advance one rock/paper/scissors game with :meth:`update`.

    Args:
        config: Round timings and thresholds.
        rng: Random source for the board's move; defaults to
            ``random.SystemRandom()``. Tests pass a seeded
            ``random.Random`` or a scripted stand-in.
    """

    def __init__(
        self,
        config: RpsGameConfig | None = None,
        *,
        rng: RpsRandom | None = None,
    ) -> None:
        self._config = config if config is not None else RpsGameConfig()
        self._rng: RpsRandom = rng if rng is not None else random.SystemRandom()
        self._phase = WAITING
        self._count: int | None = None
        self._player: str | None = None
        self._player_p: float | None = None
        self._board: str | None = None
        self._outcome: str | None = None
        self._score = RpsScore()
        self._hand: Hand3D | None = None
        self._ready_since: float | None = None
        self._countdown_start: float | None = None
        self._shoot_start: float | None = None
        self._result_start: float | None = None
        self._votes: list[str] = []
        self._last_hand_seen: float | None = None

    @property
    def state(self) -> RpsState:
        """The current snapshot."""

        return RpsState(
            phase=self._phase,
            count=self._count,
            player=self._player,
            player_p=self._player_p,
            board=self._board,
            outcome=self._outcome,
            score=self._score,
            hand=self._hand,
        )

    def update(
        self,
        now: float,
        hands: Sequence[Hand3D],
        classify: Callable[[Hand3D], tuple[str, float]],
    ) -> RpsState:
        """Advance the machine to ``now`` and return the new snapshot.

        ``classify`` is called once per update during the shoot window, for the
        largest visible hand. Exceptions it raises propagate to the caller,
        which owns error reporting; the round then keeps its phase.
        """

        hand = _largest(hands)
        self._hand = hand
        if hand is None:
            if self._phase == READY:
                self._enter_waiting()
            if self._last_hand_seen is not None and _reached(
                now, self._last_hand_seen, self._config.score_reset_s
            ):
                self._score = RpsScore()
                self._last_hand_seen = None
                if self._phase == WAITING:
                    self._reset_round()
        else:
            self._last_hand_seen = now
            if self._phase == WAITING:
                self._phase = READY
                self._ready_since = now
                self._count = None

        if self._phase == READY:
            ready_since = self._ready_since if self._ready_since is not None else now
            if _reached(now, ready_since, self._config.ready_s):
                self._phase = COUNTDOWN
                self._countdown_start = ready_since + self._config.ready_s
        entered_shoot = False
        if self._phase == COUNTDOWN:
            entered_shoot = self._advance_countdown(now)
        if self._phase == SHOOT:
            shoot_start = self._shoot_start if self._shoot_start is not None else now
            if not _reached(now, shoot_start, self._config.shoot_s):
                if hand is not None and not entered_shoot:
                    self._votes.append(_vote(classify, hand))
            else:
                self._decide(shoot_start + self._config.shoot_s)
        if self._phase == RESULT:
            result_start = self._result_start if self._result_start is not None else now
            if _reached(now, result_start, self._config.result_s):
                self._reset_round()
                if hand is None:
                    self._phase = WAITING
                else:
                    # The hand is visible, so the next round may start right away.
                    self._phase = READY
                    self._ready_since = now
        return self.state

    # -- phases ------------------------------------------------------------

    def _advance_countdown(self, now: float) -> bool:
        """Show the next count, or enter the shoot window and return ``True``."""

        start = self._countdown_start if self._countdown_start is not None else now
        elapsed = max(0.0, now - start)
        step = int((elapsed + _TIME_EPS) / self._config.countdown_step_s)
        if step < _COUNTDOWN_FROM:
            self._count = _COUNTDOWN_FROM - step
            return False
        self._phase = SHOOT
        self._count = None
        self._shoot_start = start + _COUNTDOWN_FROM * self._config.countdown_step_s
        self._votes = []
        return True

    def _decide(self, ended_at: float) -> None:
        self._phase = RESULT
        self._result_start = ended_at
        self._count = None
        if not self._votes:
            self._player = None
            self._player_p = None
            self._board = None
            self._outcome = "unclear"
            return
        player, votes = Counter(self._votes).most_common(1)[0]
        share = votes / len(self._votes)
        self._player = player
        self._player_p = share
        if player == NONE_CLASS or share < self._config.min_share:
            self._board = None
            self._outcome = "unclear"
            return
        board = self._rng.choice(RPS_MOVES)
        if board not in RPS_MOVES:  # pragma: no cover - a broken random seam
            raise ValueError(f"random source returned an unknown move {board!r}")
        self._board = board
        if board == player:
            self._outcome = "draw"
            self._score = RpsScore(self._score.player, self._score.board, self._score.draws + 1)
        elif _BEATS[player] == board:
            self._outcome = "win"
            self._score = RpsScore(self._score.player + 1, self._score.board, self._score.draws)
        else:
            self._outcome = "lose"
            self._score = RpsScore(self._score.player, self._score.board + 1, self._score.draws)

    def _enter_waiting(self) -> None:
        self._phase = WAITING
        self._reset_round()

    def _reset_round(self) -> None:
        """Forget the finished round; the score and the hand survive."""

        self._count = None
        self._player = None
        self._player_p = None
        self._board = None
        self._outcome = None
        self._ready_since = None
        self._countdown_start = None
        self._shoot_start = None
        self._result_start = None
        self._votes = []


def _reached(now: float, start: float, duration: float) -> bool:
    """Whether ``duration`` seconds have passed since ``start`` (inclusive at the boundary)."""

    return now - start >= duration - _TIME_EPS


def _largest(hands: Sequence[Hand3D]) -> Hand3D | None:
    """The hand with the largest box area, or ``None`` when no hand is visible."""

    best: Hand3D | None = None
    best_area = -1.0
    for hand in hands:
        x1, y1, x2, y2 = hand.box
        area = abs((x2 - x1) * (y2 - y1))
        if area > best_area:
            best, best_area = hand, area
    return best


def _vote(classify: Callable[[Hand3D], tuple[str, float]], hand: Hand3D) -> str:
    """Classify one hand for the majority vote; an unknown class counts as ``none``."""

    name, _ = classify(hand)
    return name if name in RPS_MOVES or name == NONE_CLASS else NONE_CLASS
