"""The protocol v1 builders: ranking, counts, rounding and envelope (F-K1)."""

from __future__ import annotations

import math

from cpsa import protocol


def _object(label: str, class_id: int, score: float) -> protocol.ShowcaseObject:
    return protocol.ShowcaseObject(
        label=label, class_id=class_id, score=score, box=(1.0, 2.0, 3.0, 4.0)
    )


def test_the_envelope_carries_the_sequence_and_both_clocks() -> None:
    message = protocol.envelope(
        "hello", {"server_version": "0.1.0"}, 3, 12.5, "2026-11-12T15:00:00Z"
    )

    assert message == {
        "v": 1,
        "type": "hello",
        "seq": 3,
        "t": 12.5,
        "wall": "2026-11-12T15:00:00Z",
        "data": {"server_version": "0.1.0"},
    }
    assert message["data"] is not protocol.hello("0.1.0")


def test_hello_names_the_protocol_version() -> None:
    assert protocol.hello("1.2.3") == {"server_version": "1.2.3", "protocol_version": 1}


def test_status_keeps_unknown_numbers_null() -> None:
    payload = protocol.status(
        state="starting",
        camera="missing",
        model="yolox_nano",
        fps=None,
        dpu_ms=None,
        errors=["camera_missing"],
        uptime_s=0.0,
    )

    assert payload["fps"] is None and payload["dpu_ms"] is None
    assert payload["errors"] == ["camera_missing"]
    assert payload["uptime_s"] == 0.0


def test_status_rounds_timings_and_rejects_a_negative_uptime() -> None:
    payload = protocol.status(
        state="running",
        camera="ok",
        model="m",
        fps=59.44,
        dpu_ms=4.66,
        errors=(),
        uptime_s=-3.0,
    )

    assert payload["fps"] == 59.4
    assert payload["dpu_ms"] == 4.7
    assert payload["uptime_s"] == 0.0


def test_status_reports_a_non_finite_measurement_as_null() -> None:
    payload = protocol.status(
        state="running",
        camera="ok",
        model="m",
        fps=math.nan,
        dpu_ms=math.inf,
        errors=[],
        uptime_s=math.nan,
    )

    assert payload["fps"] is None
    assert payload["dpu_ms"] is None
    assert payload["uptime_s"] == 0.0


def test_showcase_state_ranks_by_score_and_counts_by_label() -> None:
    payload = protocol.showcase_state(
        "yolox_nano",
        640,
        480,
        [
            _object("persona", 0, 0.5),
            _object("tazza", 41, 0.9),
            _object("persona", 0, 0.7),
        ],
    )

    assert [item["label"] for item in payload["objects"]] == ["tazza", "persona", "persona"]
    assert payload["counts"] == [
        {"label": "persona", "count": 2},
        {"label": "tazza", "count": 1},
    ]


def test_showcase_state_caps_the_object_list() -> None:
    objects = [_object(f"c{index}", index, 1.0 - index / 100.0) for index in range(30)]

    payload = protocol.showcase_state("m", 640, 480, objects)

    assert len(payload["objects"]) == protocol.MAX_OBJECTS == 20
    assert payload["objects"][0]["label"] == "c0"
    assert len(payload["counts"]) == 30


def test_showcase_state_rounds_and_clamps_values() -> None:
    item = protocol.ShowcaseObject(
        label="persona", class_id=0, score=2.0, box=(1.234, math.nan, -5.0, 480.0)
    )

    payload = protocol.showcase_state("m", 640, 480, [item], fps=59.96, total_ms=11.24)

    assert payload["fps"] == 60.0
    assert payload["total_ms"] == 11.2
    entry = payload["objects"][0]
    assert entry["score"] == 1.0
    assert entry["box"] == [1.2, 0.0, -5.0, 480.0]


def test_error_omits_an_absent_ref_seq() -> None:
    assert protocol.error("bad_command", "nope") == {
        "code": "bad_command",
        "message": "nope",
    }
    assert protocol.error("bad_command", "nope", 4)["ref_seq"] == 4


def test_pong_echoes_the_id_unchanged() -> None:
    marker = {"nested": [1, 2]}

    assert protocol.pong(marker)["id"] is marker


def test_showcase_state_handles_an_empty_frame() -> None:
    payload = protocol.showcase_state("m", 640, 480, [])

    assert payload["objects"] == []
    assert payload["counts"] == []
    assert payload["fps"] is None
