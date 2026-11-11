"""The ``hologram.hands`` and ``rps.state`` protocol builders (F-E, AC-10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import numpy as np
import pytest

from cpsa.protocol.hands import OUTCOMES, RPS_MOVES, RPS_PHASES, hologram_hands, rps_state
from cpsa.protocol.system import envelope
from cpsa.supervisor.contract import ModeName
from cpsa.vision.hand3d import Hand3D, HandFrame
from tests.contract.protocol_schemas import SCHEMA_DIR, validate_server_message

MODE_SCHEMAS = ("hello", "mode.progress", "mode.set", "status")


def keypoints() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = 300.0 + 3.0 * np.arange(21)
    points[:, 1] = 240.0 - 2.0 * np.arange(21)
    points[:, 2] = 0.9
    return points


def lifted() -> np.ndarray:
    lifted_points = np.zeros((21, 4), dtype=np.float32)
    lifted_points[:, 0] = np.linspace(-0.1, 0.1, 21)
    lifted_points[:, 1] = np.linspace(0.05, -0.05, 21)
    lifted_points[:, 2] = 0.55
    lifted_points[:, 3] = 1.0
    lifted_points[3, 3] = 0.0
    lifted_points[3, :3] = 0.0
    return lifted_points


def hand(score: float = 0.93) -> Hand3D:
    return Hand3D(
        score=score,
        box=(284.0, 150.0, 392.0, 302.0),
        kpts2d=keypoints(),
        kpts3d=lifted(),
    )


def frame(*hands: Hand3D) -> HandFrame:
    return HandFrame(
        seq=7,
        t=1012.5,
        width=640,
        height=480,
        hands=hands,
        timing={
            "decode_ms": 1.5,
            "preprocess_ms": 0.7,
            "dpu_ms": 6.2,
            "post_ms": 0.4,
            "lift_ms": 2.6,
        },
    )


def message(kind: str, data: dict[str, object]) -> dict[str, object]:
    return envelope(kind, data, 3, 1012.6, "2026-11-12T08:00:00Z")


# -- hologram.hands --------------------------------------------------------


def test_hologram_hands_builds_one_entry_per_hand() -> None:
    payload = hologram_hands(frame(hand(0.9), hand(0.7)))

    assert payload["seq"] == 7
    assert payload["t"] == pytest.approx(1012.5)
    assert (payload["width"], payload["height"]) == (640, 480)
    assert [entry["score"] for entry in payload["hands"]] == [
        pytest.approx(0.9),
        pytest.approx(0.7),
    ]
    assert payload["hands"][0]["box"] == [284.0, 150.0, 392.0, 302.0]
    assert len(payload["hands"][0]["kpts2d"]) == 21
    assert len(payload["hands"][0]["kpts3d"]) == 21
    assert payload["hands"][0]["kpts2d"][0] == pytest.approx([300.0, 240.0, 0.9])
    assert payload["timing"]["dpu_ms"] == pytest.approx(6.2)
    assert payload["timing"]["total_ms"] == pytest.approx(11.4)


def test_hologram_hands_reports_an_invalid_depth_as_null() -> None:
    payload = hologram_hands(frame(hand()))
    lifted_points = payload["hands"][0]["kpts3d"]

    assert lifted_points[3] is None
    assert lifted_points[0] == pytest.approx([-0.1, 0.05, 0.55])


def test_hologram_hands_validates_against_the_schema() -> None:
    validate_server_message(message("hologram.hands", hologram_hands(frame())))


def test_hologram_hands_allows_a_frame_without_hands() -> None:
    payload = hologram_hands(frame())

    assert payload["hands"] == []
    validate_server_message(message("hologram.hands", payload))


def test_hologram_hands_rejects_a_hand_without_twenty_one_keypoints() -> None:
    broken = Hand3D(
        score=0.9,
        box=(0.0, 0.0, 1.0, 1.0),
        kpts2d=np.zeros((20, 3), dtype=np.float32),
        kpts3d=lifted(),
    )

    with pytest.raises(ValueError, match="21"):
        hologram_hands(frame(broken))


def test_hologram_hands_reports_non_finite_measurements_at_the_origin() -> None:
    points = keypoints()
    points[5, 0] = np.nan
    broken = Hand3D(score=0.9, box=(0.0, 0.0, 1.0, 1.0), kpts2d=points, kpts3d=lifted())

    payload = hologram_hands(frame(broken))

    assert payload["hands"][0]["kpts2d"][5][0] == 0.0


# -- rps.state -------------------------------------------------------------


@pytest.mark.parametrize("phase", RPS_PHASES)
def test_every_rps_phase_validates_against_the_schema(phase: str) -> None:
    payload = rps_state(phase)

    assert payload["phase"] == phase
    assert set(payload) == {
        "phase",
        "count",
        "player",
        "player_p",
        "board",
        "outcome",
        "score",
        "hand",
    }
    validate_server_message(message("rps.state", payload))


def test_rps_state_carries_the_round_result_and_the_hand() -> None:
    payload = rps_state(
        "result",
        player="paper",
        player_p=0.86,
        board="rock",
        outcome="win",
        score={"player": 3, "board": 1, "draws": 2},
        kpts2d=keypoints(),
    )

    assert payload["player"] == "paper"
    assert payload["player_p"] == pytest.approx(0.86)
    assert payload["board"] == "rock"
    assert payload["outcome"] == "win"
    assert payload["score"] == {"player": 3, "board": 1, "draws": 2}
    assert payload["hand"] is not None
    assert len(payload["hand"]["kpts2d"]) == 21
    validate_server_message(message("rps.state", payload))


def test_rps_state_defaults_every_optional_field() -> None:
    payload = rps_state("waiting")

    assert payload["count"] is None
    assert payload["player"] is None
    assert payload["player_p"] is None
    assert payload["board"] is None
    assert payload["outcome"] is None
    assert payload["score"] == {"player": 0, "board": 0, "draws": 0}
    assert payload["hand"] is None


def test_rps_state_accepts_the_none_class_and_the_unclear_outcome() -> None:
    payload = rps_state("result", player="none", player_p=1.0, outcome="unclear")

    assert payload["player"] == "none"
    assert payload["outcome"] == "unclear"
    validate_server_message(message("rps.state", payload))


def test_rps_state_normalises_the_probability_and_the_score() -> None:
    payload = rps_state(
        "result",
        player="rock",
        player_p=1.7,
        board="scissors",
        outcome="win",
        score={"player": -2, "board": 1, "draws": 0},
    )

    assert payload["player_p"] == pytest.approx(1.0)
    assert payload["score"] == {"player": 0, "board": 1, "draws": 0}
    assert rps_state("result", player_p=float("nan"))["player_p"] is None


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("phase", "dance"),
        ("player", "horns"),
        ("board", "none"),
        ("outcome", "maybe"),
        ("count", 4),
    ],
)
def test_rps_state_rejects_unknown_protocol_values(keyword: str, value: object) -> None:
    arguments: dict[str, object] = {
        "phase": "countdown",
        "count": 3,
        "player": None,
        "board": None,
        "outcome": None,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError):
        rps_state(**arguments)  # type: ignore[arg-type]


def test_rps_state_keeps_the_vocabulary_in_step_with_the_game() -> None:
    from cpsa.vision import rps_game

    assert RPS_PHASES == rps_game.RPS_PHASES
    assert RPS_MOVES == rps_game.RPS_MOVES
    assert OUTCOMES == rps_game.OUTCOMES


# -- schemas and enums -----------------------------------------------------


def test_the_python_mode_enum_includes_the_hand_modes() -> None:
    names = set(get_args(ModeName))

    assert {"hologram", "rps"} <= names
    assert {"idle", "lab", "medical"} <= names


@pytest.mark.parametrize("name", MODE_SCHEMAS)
def test_the_json_mode_enums_include_the_hand_modes(name: str) -> None:
    schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))

    if name == "hello":
        values = schema["properties"]["modes"]["items"]["enum"]
    else:
        values = schema["properties"]["mode"]["enum"]

    assert {"hologram", "rps"} <= set(values)
    assert {"idle", "lab", "medical"} <= set(values)


def test_the_new_schemas_have_examples() -> None:
    examples = SCHEMA_DIR / "examples"

    for name in ("hologram.hands", "rps.state"):
        assert (SCHEMA_DIR / f"{name}.schema.json").is_file()
        assert (examples / f"{name}.json").is_file()


def test_the_example_hand_stays_inside_the_frame() -> None:
    example = json.loads(
        (SCHEMA_DIR / "examples" / "hologram.hands.json").read_text(encoding="utf-8")
    )
    data = example["data"]

    assert len(data["hands"]) == 1
    assert data["width"] > 0 and data["height"] > 0
    assert all(0.0 <= point[0] <= data["width"] for point in data["hands"][0]["kpts2d"])


def test_repo_root_relative_examples_are_discovered_by_the_contract_tests() -> None:
    """The contract tests parametrise every schema; no schema may sit outside v1."""

    assert Path(SCHEMA_DIR).name == "v1"
