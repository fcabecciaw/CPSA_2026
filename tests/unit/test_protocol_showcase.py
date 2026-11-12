"""The ``showcase.state`` protocol builder and its schema (F-S2, AC-S4)."""

from __future__ import annotations

import json
import math
from typing import get_args

import pytest

from cpsa.protocol.showcase import MAX_OBJECTS, ShowcaseObject, showcase_state
from cpsa.protocol.system import envelope
from cpsa.supervisor.contract import ModeName
from tests.contract.protocol_schemas import SCHEMA_DIR, validate_server_message

MODE_SCHEMAS = ("hello", "mode.progress", "mode.set", "status")


def obj(
    label: str = "persona",
    class_id: int = 0,
    score: float = 0.9,
    box: tuple[float, float, float, float] = (10.0, 20.0, 110.0, 220.0),
) -> ShowcaseObject:
    return ShowcaseObject(label=label, class_id=class_id, score=score, box=box)


def state(objects: list[ShowcaseObject], **overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "fps": 29.74,
        "dpu_ms": 4.14,
        "total_ms": 20.96,
    }
    arguments.update(overrides)
    return showcase_state(
        "yolox_nano",
        640,
        480,
        objects,
        **arguments,  # type: ignore[arg-type]
    )


# -- the payload -----------------------------------------------------------


def test_showcase_state_carries_the_frame_and_the_timings() -> None:
    payload = state([obj()])

    assert payload == {
        "model": "yolox_nano",
        "width": 640,
        "height": 480,
        "fps": 29.7,
        "dpu_ms": 4.1,
        "total_ms": 21.0,
        "objects": [
            {
                "label": "persona",
                "class_id": 0,
                "score": 0.9,
                "box": [10.0, 20.0, 110.0, 220.0],
            }
        ],
        "counts": [{"label": "persona", "count": 1}],
    }


def test_an_empty_frame_publishes_empty_lists_and_null_timings() -> None:
    payload = state([], fps=None, dpu_ms=None, total_ms=None)

    assert payload["objects"] == []
    assert payload["counts"] == []
    assert payload["fps"] is None
    assert payload["dpu_ms"] is None
    assert payload["total_ms"] is None


def test_the_objects_are_ranked_by_descending_score() -> None:
    objects = [
        obj(label="auto", class_id=2, score=0.4),
        obj(label="persona", class_id=0, score=0.9),
        obj(label="tazza", class_id=41, score=0.7),
    ]

    payload = state(objects)

    assert [entry["label"] for entry in payload["objects"]] == ["persona", "tazza", "auto"]


def test_the_display_list_stops_at_the_cap_but_the_counts_do_not() -> None:
    objects = [obj(label="persona", class_id=0, score=0.9 - 0.01 * index) for index in range(30)]

    payload = state(objects)

    assert len(payload["objects"]) == MAX_OBJECTS
    assert payload["counts"] == [{"label": "persona", "count": 30}]


def test_the_counts_are_ordered_by_count_and_then_label() -> None:
    objects = [
        obj(label="tazza", class_id=41),
        obj(label="auto", class_id=2),
        obj(label="persona", class_id=0),
        obj(label="persona", class_id=0),
    ]

    payload = state(objects)

    assert payload["counts"] == [
        {"label": "persona", "count": 2},
        {"label": "auto", "count": 1},
        {"label": "tazza", "count": 1},
    ]


def test_non_finite_measurements_never_reach_the_wire() -> None:
    payload = state(
        [obj(score=math.nan, box=(math.inf, 0.0, 10.0, 20.0))],
        fps=math.nan,
        dpu_ms=math.inf,
        total_ms=-math.inf,
    )

    assert payload["fps"] is None
    assert payload["dpu_ms"] is None
    assert payload["total_ms"] is None
    entry = payload["objects"][0]
    assert entry["score"] == 0.0
    assert entry["box"] == [0.0, 0.0, 10.0, 20.0]


def test_scores_outside_the_unit_interval_are_clamped() -> None:
    payload = state([obj(score=1.4), obj(score=-0.3)])

    assert [entry["score"] for entry in payload["objects"]] == [1.0, 0.0]


def test_the_scores_keep_three_decimals_and_the_pixels_one() -> None:
    payload = state([obj(score=0.91249, box=(1.25, 2.5, 3.75, 4.125))])

    entry = payload["objects"][0]
    assert entry["score"] == 0.912
    assert entry["box"] == [1.2, 2.5, 3.8, 4.1]


def test_the_builder_output_validates_against_the_schema() -> None:
    payload = state([obj(), obj(label="tazza", class_id=41, score=0.41)])

    validate_server_message(envelope("showcase.state", payload, 4, 1001.4, "2026-11-12T08:00:00Z"))


# -- schemas and enums -----------------------------------------------------


def test_the_python_mode_enum_includes_the_showcase_mode() -> None:
    names = set(get_args(ModeName))

    assert "showcase" in names
    assert {"idle", "lab", "hologram", "medical", "rps"} <= names


@pytest.mark.parametrize("name", MODE_SCHEMAS)
def test_the_json_mode_enums_include_the_showcase_mode(name: str) -> None:
    schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))

    if name == "hello":
        values = schema["properties"]["modes"]["items"]["enum"]
    else:
        values = schema["properties"]["mode"]["enum"]

    assert "showcase" in values
    assert {"idle", "lab", "hologram", "medical", "rps"} <= set(values)


def test_the_showcase_schema_has_an_example() -> None:
    examples = SCHEMA_DIR / "examples"

    assert (SCHEMA_DIR / "showcase.state.schema.json").is_file()
    assert (examples / "showcase.state.json").is_file()


def test_the_schema_caps_the_display_list_where_the_builder_does() -> None:
    schema = json.loads((SCHEMA_DIR / "showcase.state.schema.json").read_text(encoding="utf-8"))

    assert schema["properties"]["objects"]["maxItems"] == MAX_OBJECTS
    assert schema["required"] == [
        "model",
        "width",
        "height",
        "fps",
        "dpu_ms",
        "total_ms",
        "objects",
        "counts",
    ]


def test_the_example_lists_exactly_the_counts_of_its_objects() -> None:
    example = json.loads(
        (SCHEMA_DIR / "examples" / "showcase.state.json").read_text(encoding="utf-8")
    )
    data = example["data"]

    labels = [entry["label"] for entry in data["objects"]]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    assert data["counts"] == [
        {"label": label, "count": count}
        for label, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
    assert [entry["score"] for entry in data["objects"]] == sorted(
        (entry["score"] for entry in data["objects"]), reverse=True
    )
    assert all(0.0 <= entry["box"][0] <= data["width"] for entry in data["objects"])
