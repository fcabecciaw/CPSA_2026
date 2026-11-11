"""Builders for the ``hologram.hands`` and ``rps.state`` payloads (F-E).

Pure functions: they map the hand-perception types and the game snapshot to a
JSON-ready ``data`` object; the server adds the envelope (IF-102). Conventions
from ``docs/protocol/v1/README.md`` apply: a missing or not-applicable value is
``null``, never an absent key, and ``kpts3d`` entries are ``null`` for a
keypoint without a valid depth measurement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from cpsa.vision.hand3d import Hand3D, HandFrame

__all__ = ["OUTCOMES", "RPS_MOVES", "RPS_PHASES", "hologram_hands", "rps_state"]

RPS_PHASES: tuple[str, ...] = ("waiting", "ready", "countdown", "shoot", "result")
"""``rps.state.phase`` values, in round order."""

RPS_MOVES: tuple[str, ...] = ("rock", "paper", "scissors")
"""``rps.state.player`` / ``.board`` gesture values; ``player`` may add ``none``."""

OUTCOMES: tuple[str, ...] = ("win", "lose", "draw", "unclear")
"""``rps.state.outcome`` values."""

PLAYER_CLASSES: tuple[str, ...] = (*RPS_MOVES, "none")
"""``rps.state.player`` values: a gesture, or the classifier's ``none``."""

_KEYPOINTS = 21
_COORDINATES_2D = 3
_COORDINATES_3D = 4
_VALID = 0.5


def hologram_hands(frame: HandFrame) -> dict[str, Any]:
    """Build the ``hologram.hands`` payload for one analyzed IR frame (F-E).

    Args:
        frame: The hand-perception output; ``kpts2d`` is ``(21, 3)`` in IR
            pixels and ``kpts3d`` is ``(21, 4)`` metric with a ``0/1`` valid
            flag.

    Raises:
        ValueError: If a hand does not carry exactly 21 keypoints.
    """

    timing = {str(key): _number(value) for key, value in frame.timing.items()}
    return {
        "seq": int(frame.seq),
        "t": _finite(frame.t),
        "width": int(frame.width),
        "height": int(frame.height),
        "hands": [_hand(hand) for hand in frame.hands],
        "timing": {
            "dpu_ms": _number(frame.timing.get("dpu_ms")) or 0.0,
            "total_ms": math.fsum(value for value in timing.values() if value is not None),
        },
    }


def rps_state(
    phase: str,
    *,
    count: int | None = None,
    player: str | None = None,
    player_p: float | None = None,
    board: str | None = None,
    outcome: str | None = None,
    score: Mapping[str, int] | None = None,
    kpts2d: Any | None = None,
) -> dict[str, Any]:
    """Build the ``rps.state`` payload (F-E).

    ``kpts2d`` is the ``(21, 3)`` keypoint array of the hand that plays, or
    ``None`` when no hand is visible. ``player_p`` is the vote share of
    ``player``; the schema requires it in ``[0, 1]`` or ``null``.

    Raises:
        ValueError: If ``phase``, ``player``, ``board`` or ``outcome`` is not a
            protocol value, or ``kpts2d`` does not hold 21 keypoints.
    """

    if phase not in RPS_PHASES:
        raise ValueError(f"unknown rps phase {phase!r}, expected one of {RPS_PHASES}")
    if count is not None and count not in (1, 2, 3):
        raise ValueError(f"rps count must be 3, 2, 1 or null, got {count!r}")
    if player is not None and player not in PLAYER_CLASSES:
        raise ValueError(f"unknown rps player {player!r}, expected one of {PLAYER_CLASSES}")
    if board is not None and board not in RPS_MOVES:
        raise ValueError(f"unknown rps board move {board!r}, expected one of {RPS_MOVES}")
    if outcome is not None and outcome not in OUTCOMES:
        raise ValueError(f"unknown rps outcome {outcome!r}, expected one of {OUTCOMES}")
    return {
        "phase": phase,
        "count": None if count is None else int(count),
        "player": player,
        "player_p": _probability(player_p),
        "board": board,
        "outcome": outcome,
        "score": _score(score),
        "hand": None if kpts2d is None else {"kpts2d": _kpts2d(kpts2d)},
    }


def _hand(hand: Hand3D) -> dict[str, Any]:
    return {
        "score": _finite(hand.score),
        "box": [_finite(value) for value in hand.box],
        "kpts2d": _kpts2d(hand.kpts2d),
        "kpts3d": _kpts3d(hand.kpts3d),
    }


def _kpts2d(values: Any) -> list[list[float]]:
    """Return 21 ``[u, v, conf]`` rows; ``values`` is any 21-row sequence.

    The array is duck-typed (a NumPy array or a list of rows both work) so this
    protocol layer needs no third-party import.
    """

    rows = _keypoint_rows(values, "keypoints", _COORDINATES_2D)
    return [[_finite(row[0]), _finite(row[1]), _finite(row[2])] for row in rows]


def _kpts3d(values: Any) -> list[list[float] | None]:
    """Return 21 ``[x, y, z]`` rows, ``None`` where the valid flag is unset."""

    rows = _keypoint_rows(values, "3D keypoints", _COORDINATES_3D)
    lifted: list[list[float] | None] = []
    for row in rows:
        if not float(row[3]) >= _VALID:
            lifted.append(None)
            continue
        lifted.append([_finite(row[0]), _finite(row[1]), _finite(row[2])])
    return lifted


def _keypoint_rows(values: Any, name: str, minimum_columns: int) -> list[Any]:
    """Check that ``values`` holds exactly 21 rows of at least N columns."""

    try:
        rows = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be (21, K), got {values!r}") from exc
    if len(rows) != _KEYPOINTS:
        raise ValueError(f"{name} must be (21, K), got {len(rows)} rows")
    for row in rows:
        if not hasattr(row, "__len__") or len(row) < minimum_columns:
            raise ValueError(f"{name} must be (21, K), got a row without {minimum_columns} values")
    return rows


def _score(score: Mapping[str, int] | None) -> dict[str, int]:
    values = {} if score is None else score
    return {
        "player": _count(values.get("player")),
        "board": _count(values.get("board")),
        "draws": _count(values.get("draws")),
    }


def _count(value: int | None) -> int:
    return 0 if value is None else max(0, int(value))


def _probability(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return min(1.0, max(0.0, number))


def _number(value: float | None) -> float | None:
    """Return ``value`` as a JSON number, or ``None`` when absent or not finite."""

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite(value: float) -> float:
    """Return a finite coordinate; a non-finite measurement is reported at the origin."""

    number = float(value)
    return number if math.isfinite(number) else 0.0
