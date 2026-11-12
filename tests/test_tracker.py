"""The box tracker: deadband stability, motion, dropouts, births and cost."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from cpsa.config import ConfigError, StillConfig, TrackerConfig, load, parse
from cpsa.tracker import BIRTH_SCORE, Tracker
from cpsa.yolox import OBJECT_CLASSES, PERSON_CLASSES, Detection

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "config" / "showcase.yaml"

STILL = StillConfig(enabled=True, motion_thr=1.0, force_every_s=0.5)


def tracker_config(**overrides: object) -> TrackerConfig:
    settings: dict[str, object] = {
        "enabled": True,
        "iou_match": 0.3,
        "alpha": 0.35,
        "fast_alpha": 0.8,
        "deadband_iou": 0.9,
        "min_hits": 2,
        "max_misses": 6,
        "still": STILL,
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


def test_a_different_class_in_the_same_box_keeps_the_track() -> None:
    """The label is the track's business: a re-labelled detection is voted on."""
    tracker = Tracker(tracker_config())
    person = detection(100, 100, 200, 200, class_id=0, score=0.9)
    bottle = detection(100, 100, 200, 200, class_id=39, score=0.9)

    tracker.update([person], now=0.0)
    first = tracker.update([bottle], now=0.02)

    # The track survives the class change; the label waits for the vote.
    assert tracker.tracks == 1
    assert [item.class_id for item in first] == [0]

    for step in range(1, 45):
        output = tracker.update([bottle], now=0.02 + 0.02 * step)

    assert tracker.tracks == 1
    assert [item.class_id for item in output] == [39]


def test_a_class_flicker_never_kills_the_track() -> None:
    """90 frames of cup/bowl noise on one box stay one track with one label."""
    tracker = Tracker(tracker_config())
    cup, bowl = 41, 45

    labels = [
        tracker.update(
            [detection(100, 100, 200, 200, class_id=cup if index % 2 else bowl, score=0.9)],
            now=index * (1.0 / 60.0),
        )[0].class_id
        for index in range(90)
    ]

    assert tracker.tracks == 1
    assert set(labels) == {bowl}


def test_a_bottle_held_inside_a_person_box_keeps_its_track() -> None:
    """The bottle overlaps the person, so only the class may separate the tracks."""
    tracker = Tracker(tracker_config())
    person = detection(100, 100, 340, 440, score=0.9, class_id=0)
    bottle = detection(170, 260, 230, 400, score=0.4, class_id=39)

    tracker.update([person, bottle])
    output = tracker.update([person, bottle])

    assert tracker.tracks == 2
    assert sorted(item.class_id for item in output) == [0, 39]
    boxes = {item.class_id: item.box for item in output}
    assert boxes[0] == pytest.approx(person.box)
    assert boxes[39] == pytest.approx(bottle.box)


def test_a_bottle_never_takes_over_the_person_track() -> None:
    tracker = Tracker(tracker_config())
    person = detection(100, 100, 340, 440, score=0.9, class_id=0)
    tracker.update([person], now=0.0)
    tracker.update([person], now=1 / 60)

    # The person leaves, a bottle appears in the same box: the person's track
    # must survive its misses, not be relabelled as the bottle.
    output = tracker.update([detection(100, 100, 340, 440, score=0.8, class_id=39)], now=2 / 60)

    assert tracker.tracks == 1
    assert [item.class_id for item in output] == [0]


# --------------------------------------------------------------------------- #
# The alternating passes: each class group ages on its own frames
# --------------------------------------------------------------------------- #

PERSON = detection(100, 100, 340, 440, score=0.9, class_id=0)
BOTTLE = detection(170, 260, 230, 400, score=0.5, class_id=39)


def two_tracks(*, max_misses: int = 2) -> Tracker:
    """A tracker holding one person and one bottle, both matched twice."""
    tracker = Tracker(tracker_config(max_misses=max_misses))
    tracker.update([PERSON, BOTTLE], now=0.0)
    tracker.update([PERSON, BOTTLE], now=0.05)
    return tracker


def box_of(output: list[Detection], class_id: int) -> tuple[float, float, float, float]:
    return next(item.box for item in output if item.class_id == class_id)


def test_a_class_an_alternate_pass_does_not_look_for_is_not_missed() -> None:
    tracker = two_tracks()
    original = box_of(tracker.update([PERSON, BOTTLE], now=0.1), 39)

    # Ten whole-frame passes report persons only: the bottle is not "missed",
    # it simply was not looked for, so it keeps its box.
    for step in range(10):
        shown = tracker.update([PERSON], now=0.2 + step * 0.05, covered=PERSON_CLASSES)
        assert tracker.tracks == 2
        assert box_of(shown, 39) == original


def test_a_class_ages_out_on_its_own_pass_only() -> None:
    tracker = two_tracks()
    # The person's passes do not touch the bottle, however many miss it.
    for step in range(6):
        tracker.update([PERSON], now=0.1 + step * 0.05, covered=PERSON_CLASSES)
    assert tracker.tracks == 2

    # The bottle's own passes do: max_misses is 2, so the third absence drops it.
    for step in range(2):
        assert tracker.update([], now=0.5 + step * 0.05, covered=OBJECT_CLASSES)
    output = tracker.update([], now=0.7, covered=OBJECT_CLASSES)
    assert tracker.tracks == 1
    assert [item.class_id for item in output] == [0]


def test_a_person_pass_cannot_relabel_an_object_track() -> None:
    tracker = Tracker(tracker_config())
    tracker.update([BOTTLE], now=0.0)
    tracker.update([BOTTLE], now=0.05)
    bottle_box = box_of(tracker.update([BOTTLE], now=0.1), 39)

    # A whole-frame pass reports a person in the bottle's box: the bottle's
    # track is not a candidate at all, so it survives as a bottle ...
    output = tracker.update(
        [detection(*BOTTLE.box, score=0.9, class_id=0)],
        now=0.15,
        covered=PERSON_CLASSES,
    )
    assert box_of(output, 39) == bottle_box

    # ... and the zoomed pass still finds it exactly where it was.
    later = tracker.update([BOTTLE], now=0.2, covered=OBJECT_CLASSES)
    assert box_of(later, 39) == bottle_box


def test_without_a_covered_set_every_frame_looks_for_every_class() -> None:
    tracker = two_tracks()
    for step in range(3):
        tracker.update([PERSON], now=0.1 + step * 0.05)

    assert tracker.tracks == 1


def test_a_disabled_tracker_returns_the_raw_boxes() -> None:
    tracker = Tracker(tracker_config(enabled=False))
    detections = [detection(1, 2, 3, 4, score=0.7)]

    assert tracker.update(detections) == detections


# --------------------------------------------------------------------------- #
# Settled tracks: a box that has not moved for a second
# --------------------------------------------------------------------------- #


def settled_tracker(*, seconds: float = 1.2) -> Tracker:
    """A tracker holding one settled person box of 100x100 at ``(100, 100)``."""
    tracker = Tracker(tracker_config())
    person = detection(100, 100, 200, 200, score=0.8, class_id=0)
    for step in range(int(round(seconds / 0.05))):
        tracker.update([person], now=0.05 * step)
    return tracker


def test_a_box_that_held_for_a_second_settles() -> None:
    """A jitter the normal deadband follows no longer moves a settled box."""
    jitter = (108.0, 100.0, 208.0, 200.0)  # 8 px: past 4% of the diagonal, inside 8%
    person = detection(100, 100, 200, 200, score=0.8, class_id=0)

    fresh = Tracker(tracker_config())
    fresh.update([person], now=0.0)
    moving = [
        shown_box(fresh.update([detection(*jitter, score=0.8)], now=0.02 * step))
        for step in range(1, 6)
    ]

    settled = settled_tracker()
    still = [
        shown_box(settled.update([detection(*jitter, score=0.8)], now=1.2 + 0.02 * step))
        for step in range(1, 6)
    ]

    assert moving[-1] != person.box
    assert set(still) == {person.box}


def test_a_settled_box_follows_only_a_sustained_move() -> None:
    tracker = settled_tracker()
    truth = (145.0, 100.0, 245.0, 200.0)  # 45 px: still matched, far outside the settled deadband

    boxes = [
        shown_box(tracker.update([detection(*truth, score=0.8)], now=1.2 + 0.02 * step))
        for step in range(1, 6)
    ]

    assert boxes[:2] == [(100.0, 100.0, 200.0, 200.0)] * 2
    assert boxes[2] == pytest.approx(truth, abs=15.0)


def test_a_one_frame_glitch_never_moves_a_settled_box() -> None:
    tracker = settled_tracker()
    tracker.update([detection(130.0, 100.0, 230.0, 200.0, score=0.8)], now=1.2)
    tracker.update([detection(100.0, 100.0, 200.0, 200.0, score=0.8)], now=1.22)

    box = shown_box(tracker.update([detection(100.0, 100.0, 200.0, 200.0, score=0.8)], now=1.24))

    assert box == (100.0, 100.0, 200.0, 200.0)


def test_a_settled_box_survives_thirty_missed_frames() -> None:
    tracker = settled_tracker()

    for step in range(30):
        assert tracker.update([], now=1.2 + 0.02 * step)

    assert tracker.tracks == 1
    assert tracker.update([], now=2.0) == []


def test_a_settled_score_is_quantized_to_five_percent() -> None:
    tracker = settled_tracker()
    output = tracker.update([detection(100, 100, 200, 200, score=0.79, class_id=0)], now=1.6)

    score = output[0].score
    assert score == pytest.approx(round(score * 20) / 20)
    assert score == pytest.approx(0.80)


def test_a_settled_score_is_a_slow_ema() -> None:
    tracker = settled_tracker(seconds=1.2)
    before = settled_tracker(seconds=1.2).update([], now=1.3)[0].score

    # 20 frames of score 0.4 move the slow EMA by a fifth of the way at most.
    for step in range(20):
        output = tracker.update(
            [detection(100, 100, 200, 200, score=0.4, class_id=0)], now=1.2 + 0.02 * step
        )

    assert output[0].score < before - 0.1
    assert output[0].score > 0.4


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark
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
        still=STILL,
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
