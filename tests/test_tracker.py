"""The box tracker: deadband stability, motion, dropouts, births and cost."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from cpsa.config import ConfigError, TrackerConfig, load, parse
from cpsa.tracker import BIRTH_SCORE, Tracker
from cpsa.yolox import Detection

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "config" / "showcase.yaml"


def tracker_config(**overrides: object) -> TrackerConfig:
    settings: dict[str, object] = {
        "enabled": True,
        "iou_match": 0.3,
        "alpha": 0.35,
        "fast_alpha": 0.8,
        "deadband_iou": 0.9,
        "min_hits": 2,
        "max_misses": 6,
    }
    settings.update(overrides)
    return TrackerConfig(**settings)  # type: ignore[arg-type]


def detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    score: float = 0.9,
    class_id: int = 0,
) -> Detection:
    return Detection(box=(x1, y1, x2, y2), score=score, class_id=class_id)


def shown_box(output: list[Detection]) -> tuple[float, float, float, float]:
    assert len(output) == 1
    return output[0].box


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #


def test_a_still_object_with_jitter_keeps_one_displayed_box() -> None:
    tracker = Tracker(tracker_config())
    boxes = [
        shown_box(tracker.update([detection(100 + dx, 100, 200 + dx, 200)]))
        for dx in (0, -3, 3, -1, 1, 3, -3, 0, 2)
    ]

    assert set(boxes) == {(100.0, 100.0, 200.0, 200.0)}


def test_a_moving_object_is_followed_after_a_jump() -> None:
    tracker = Tracker(tracker_config())
    tracker.update([detection(100, 100, 200, 200)])
    truth = (140.0, 100.0, 240.0, 200.0)

    distances = []
    for _ in range(3):
        box = shown_box(tracker.update([detection(*truth)]))
        distances.append(max(abs(a - b) for a, b in zip(box, truth, strict=True)))

    assert distances[0] <= 10.0
    assert all(distance <= 10.0 for distance in distances)


def test_a_slow_move_leaves_the_deadband_only_when_it_is_big_enough() -> None:
    tracker = Tracker(tracker_config())
    tracker.update([detection(100, 100, 200, 200)])
    # 1 px per frame on a 100 px box is inside the deadband: a still box.
    boxes = [
        shown_box(tracker.update([detection(100 + step, 100, 200 + step, 200)]))
        for step in range(1, 4)
    ]

    assert set(boxes) == {(100.0, 100.0, 200.0, 200.0)}


def test_a_one_frame_dropout_keeps_the_box() -> None:
    tracker = Tracker(tracker_config())
    born = shown_box(tracker.update([detection(100, 100, 200, 200)]))
    tracker.update([detection(101, 100, 201, 200)])

    dropped = tracker.update([])
    assert [item.box for item in dropped] == [born]
    assert tracker.tracks == 1

    again = tracker.update([detection(100, 100, 200, 200)])
    assert [item.box for item in again] == [born]


def test_a_track_dies_after_max_misses() -> None:
    tracker = Tracker(tracker_config(max_misses=6))
    tracker.update([detection(100, 100, 200, 200)])

    for _ in range(6):
        assert tracker.update([])
    assert tracker.update([]) == []
    assert tracker.tracks == 0


def test_two_same_class_objects_never_swap() -> None:
    tracker = Tracker(tracker_config())
    for step in range(12):
        left_truth = (100.0 + 3.0 * step, 100.0, 160.0 + 3.0 * step, 160.0)
        right_truth = (300.0 - 3.0 * step, 100.0, 360.0 - 3.0 * step, 160.0)
        output = tracker.update(
            [detection(*left_truth), detection(*right_truth)],
        )
        boxes = sorted((item.box for item in output), key=lambda box: box[0])
        assert boxes[0] == pytest.approx(left_truth, abs=10.0)
        assert boxes[1] == pytest.approx(right_truth, abs=10.0)


# --------------------------------------------------------------------------- #
# Birth, score and classes
# --------------------------------------------------------------------------- #


def test_a_new_object_shows_only_after_min_hits() -> None:
    tracker = Tracker(tracker_config(min_hits=2))

    assert tracker.update([detection(100, 100, 200, 200, score=0.4)]) == []
    shown = tracker.update([detection(100, 100, 200, 200, score=0.4)])

    assert len(shown) == 1


def test_a_high_score_object_shows_at_once() -> None:
    tracker = Tracker(tracker_config(min_hits=3))

    shown = tracker.update([detection(100, 100, 200, 200, score=BIRTH_SCORE)])

    assert len(shown) == 1


def test_the_shown_score_is_an_ema() -> None:
    tracker = Tracker(tracker_config(alpha=0.35))
    tracker.update([detection(100, 100, 200, 200, score=0.9)])

    output = tracker.update([detection(100, 100, 200, 200, score=0.5)])

    assert output[0].score == pytest.approx(0.9 + 0.35 * (0.5 - 0.9))


def test_tracks_only_match_the_same_class() -> None:
    tracker = Tracker(tracker_config())
    tracker.update([detection(100, 100, 200, 200, class_id=0)])

    output = tracker.update([detection(100, 100, 200, 200, class_id=2)])

    assert {item.class_id for item in output} == {0, 2}
    assert tracker.tracks == 2


def test_a_disabled_tracker_returns_the_raw_boxes() -> None:
    tracker = Tracker(tracker_config(enabled=False))
    detections = [detection(1, 2, 3, 4, score=0.7)]

    assert tracker.update(detections) == detections


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def test_twenty_boxes_cost_less_than_a_fraction_of_a_frame() -> None:
    tracker = Tracker(tracker_config())
    detections = [
        detection(10.0 + 40.0 * index, 10.0, 70.0 + 40.0 * index, 70.0) for index in range(20)
    ]
    for _ in range(10):
        tracker.update(detections)

    # The host is shared with the other booth processes: the fastest frame is
    # the honest estimate of the work one update costs, while a mean measures
    # the scheduler. A regression raises both.
    samples = []
    for _ in range(300):
        started = time.perf_counter()
        tracker.update(detections)
        samples.append((time.perf_counter() - started) * 1000.0)
    per_frame_ms = min(samples)

    assert per_frame_ms < 0.3, f"{per_frame_ms:.3f} ms per frame for 20 boxes"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def document() -> dict[str, Any]:
    data = yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_the_shipped_tracker_configuration_is_valid() -> None:
    config = load(SHIPPED)

    assert config.tracker == TrackerConfig(
        enabled=True,
        iou_match=0.3,
        alpha=0.35,
        fast_alpha=0.8,
        deadband_iou=0.9,
        min_hits=2,
        max_misses=6,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("enabled", "yes"),
        ("iou_match", 1.5),
        ("alpha", -0.1),
        ("fast_alpha", 2),
        ("deadband_iou", -1),
        ("min_hits", -1),
        ("max_misses", -1),
    ],
)
def test_an_invalid_tracker_value_is_named(key: str, value: object) -> None:
    data = document()
    data["tracker"][key] = value

    with pytest.raises(ConfigError) as excinfo:
        parse(data)
    assert excinfo.value.key == f"tracker.{key}"
