"""Protocol parsing, validation and deterministic trial scheduling."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from cpsa.recorder.protocol import (
    ProtocolError,
    load,
    parse,
    plan_trials,
    protocol_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "config" / "protocol.yaml"


def valid() -> dict[str, Any]:
    return {
        "seed": 7,
        "distances_cm": [55],
        "pre_s": 1.0,
        "post_s": 1.0,
        "rest_s": 0.5,
        "cue_lead_s": 2.0,
        "classes": [
            {
                "name": "fist_temple",
                "category": "dangerous",
                "hand": "both",
                "target": "temple",
                "reps": 2,
                "duration_s": 2.0,
            }
        ],
    }


def without(data: dict[str, Any], dotted: str) -> dict[str, Any]:
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node[part]
    del node[leaf]
    return data


def with_value(data: dict[str, Any], dotted: str, value: object) -> dict[str, Any]:
    if dotted.startswith("classes[0]."):
        data["classes"][0][dotted.split(".", 1)[1]] = value
        return data
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node[part]
    node[leaf] = value
    return data


def test_shipped_protocol_is_valid_and_covers_every_category() -> None:
    protocol = load(SHIPPED)
    assert protocol.seed == 20261106
    assert protocol.distances_cm == (45, 55, 65)
    categories = {item.category for item in protocol.classes}
    assert categories == {"dangerous", "confounder", "stereotypy", "idle"}
    names = {item.name for item in protocol.classes}
    assert {
        "fist_temple",
        "fist_forehead",
        "slap_cheek",
        "finger_eye",
        "press_eye",
        "hand_mouth_bite",
        "scratch_face",
        "scratch_nose",
        "adjust_glasses",
        "drink",
        "phone_ear",
        "touch_hair",
        "cover_yawn",
        "chin_rest",
        "hand_flapping",
        "rocking",
        "idle",
    } <= names
    assert len(plan_trials(protocol)) == 279


def test_same_seed_produces_the_same_order() -> None:
    protocol = parse(valid())
    assert plan_trials(protocol) == plan_trials(parse(copy.deepcopy(valid())))


def test_different_seed_changes_the_order() -> None:
    first = plan_trials(parse(valid()))
    other = valid()
    other["seed"] = 8
    second = plan_trials(parse(other))
    assert [trial.class_name for trial in first] == [trial.class_name for trial in second]
    assert [trial.hand for trial in first] != [trial.hand for trial in second]


def test_both_expands_into_left_and_right_trials() -> None:
    trials = plan_trials(parse(valid()))
    assert len(trials) == 4
    assert sorted(trial.hand for trial in trials) == ["L", "L", "R", "R"]
    assert {trial.rep for trial in trials} == {1, 2}


def test_single_hand_is_not_expanded() -> None:
    data = valid()
    data["classes"][0]["hand"] = "R"
    data["classes"][0]["reps"] = 1
    trials = plan_trials(parse(data))
    assert len(trials) == 1
    assert trials[0].hand == "R"


def test_each_distance_is_a_block_with_its_own_order() -> None:
    data = valid()
    data["distances_cm"] = [45, 65]
    data["classes"][0]["reps"] = 1
    trials = plan_trials(parse(data))
    assert len(trials) == 4
    assert [trial.block_index for trial in trials].count(0) == 2
    assert [trial.block_index for trial in trials].count(1) == 2
    assert {trial.distance_cm for trial in trials if trial.block_index == 0} == {45}
    assert {trial.distance_cm for trial in trials if trial.block_index == 1} == {65}
    assert [trial.trial_id for trial in trials] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "key",
    ["seed", "distances_cm", "pre_s", "post_s", "rest_s", "cue_lead_s", "classes"],
)
def test_missing_root_key_is_named(key: str) -> None:
    with pytest.raises(ProtocolError) as info:
        parse(without(valid(), key))
    assert info.value.field.startswith(key)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seed", "early"),
        ("seed", 1.5),
        ("distances_cm", []),
        ("distances_cm", [55, 0]),
        ("pre_s", -1),
        ("post_s", "soon"),
        ("rest_s", -0.5),
        ("cue_lead_s", -1),
        ("classes[0].hand", "B"),
        ("classes[0].category", "unknown"),
        ("classes[0].reps", 0),
        ("classes[0].reps", 1.5),
        ("classes[0].duration_s", 0),
        ("classes[0].name", "bad name"),
        ("classes[0].target", ""),
    ],
)
def test_invalid_value_names_the_field(key: str, value: object) -> None:
    with pytest.raises(ProtocolError) as info:
        parse(with_value(valid(), key, value))
    assert info.value.field.startswith(key)


def test_unknown_keys_are_rejected() -> None:
    data = valid()
    data["unknown"] = True
    with pytest.raises(ProtocolError) as info:
        parse(data)
    assert info.value.field == "unknown"

    class_data = valid()
    class_data["classes"][0]["extra"] = 1
    with pytest.raises(ProtocolError) as info:
        parse(class_data)
    assert info.value.field == "classes[0].extra"


def test_duplicate_class_names_are_rejected() -> None:
    data = valid()
    data["classes"].append(copy.deepcopy(data["classes"][0]))
    with pytest.raises(ProtocolError) as info:
        parse(data)
    assert info.value.field == "classes"
    assert "duplicate" in str(info.value)


def test_unreadable_and_malformed_files_name_the_path(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="missing.yaml"):
        load(tmp_path / "missing.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("seed: [unclosed\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="broken.yaml"):
        load(broken)


def test_protocol_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "protocol.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    first = protocol_sha256(path)
    assert len(first) == 64
    assert first == protocol_sha256(path)
