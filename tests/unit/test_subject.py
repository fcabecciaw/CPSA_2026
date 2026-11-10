"""Subject selection, face width and framing state (AC-002, AC-003)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa.vision.layouts import COCO17, HALFBUST_62
from cpsa.vision.subject import SubjectConfig, framing_state, select_subject
from cpsa.vision.tracker2d import KeypointTracker2D, Tracker2DConfig
from cpsa.vision.types import FramingState, KeypointLayout, PersonPose, PoseFrame

_LAYOUT = HALFBUST_62
_SIZE = len(_LAYOUT)


def _points(scale: float = 1.0) -> dict[str, tuple[float, float]]:
    """Synthetic front-facing upper body around ``(320, 240)``."""
    return {
        "nose": (320.0, 200.0),
        "left_eye": (300.0, 190.0),
        "right_eye": (340.0, 190.0),
        "left_ear": (270.0, 200.0),
        "right_ear": (370.0, 200.0),
        "face_jaw_left": (290.0, 220.0),
        "face_jaw_right": (350.0, 220.0),
        "left_shoulder": (320.0 - 70.0 * scale, 380.0 - 120.0 * scale),
        "right_shoulder": (320.0 + 70.0 * scale, 380.0 - 120.0 * scale),
        "left_elbow": (320.0 - 90.0 * scale, 380.0 - 60.0 * scale),
        "right_elbow": (320.0 + 90.0 * scale, 380.0 - 60.0 * scale),
        "left_wrist": (320.0 - 100.0 * scale, 380.0),
        "right_wrist": (320.0 + 100.0 * scale, 380.0),
    }


def _person(
    box: tuple[float, float, float, float],
    points: dict[str, tuple[float, float]] | None = None,
    conf: dict[str, float] | None = None,
    score: float = 0.9,
    layout: KeypointLayout = _LAYOUT,
) -> PersonPose:
    keypoints = _points() if points is None else points
    xy = np.zeros((len(layout), 2), dtype=np.float64)
    confidences = np.zeros(len(layout), dtype=np.float64)
    for name, value in keypoints.items():
        xy[layout.index(name)] = value
    for name, value in (conf if conf is not None else {}).items():
        confidences[layout.index(name)] = value
    if conf is None:
        for name in keypoints:
            confidences[layout.index(name)] = 0.9
    return PersonPose(
        box=np.asarray(box, dtype=np.float64),
        score=score,
        xy=xy,
        conf=confidences,
    )


def _frame(person: PersonPose | None, seq: int = 0, t: float = 0.0) -> PoseFrame:
    persons = () if person is None else (person,)
    return PoseFrame(seq=seq, t=t, width=640, height=480, persons=persons)


def test_framing_half_and_full_body() -> None:
    cfg = SubjectConfig()
    half_body = _person((180.0, 120.0, 460.0, 570.0))
    full_body = _person((180.0, 120.0, 460.0, 1020.0))
    # Auto face width is the jaw distance, 60 px.
    assert framing_state(half_body, _LAYOUT, 60.0, cfg) is FramingState.HALF_BODY
    assert framing_state(full_body, _LAYOUT, 60.0, cfg) is FramingState.FULL_BODY


def test_framing_reports_missing_face_and_hands() -> None:
    cfg = SubjectConfig()
    no_face = _person(
        (180.0, 120.0, 460.0, 570.0),
        conf={
            "nose": 0.1,
            "left_eye": 0.1,
            "right_eye": 0.2,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "left_wrist": 0.9,
            "right_wrist": 0.9,
        },
    )
    no_hands = _person(
        (180.0, 120.0, 460.0, 570.0),
        conf={
            "nose": 0.9,
            "left_eye": 0.9,
            "right_eye": 0.9,
            "left_ear": 0.9,
            "right_ear": 0.9,
            "left_wrist": 0.0,
            "right_wrist": 0.0,
        },
    )
    assert framing_state(no_face, _LAYOUT, 60.0, cfg) is FramingState.FACE_NOT_VISIBLE
    assert framing_state(no_hands, _LAYOUT, 60.0, cfg) is FramingState.HANDS_NOT_VISIBLE
    assert framing_state(None, _LAYOUT, 60.0, cfg) is FramingState.NO_PERSON


def test_half_body_keeps_upper_body_confidences() -> None:
    """Lower-body keypoints must not change upper-body confidences (AC-002)."""
    cfg = Tracker2DConfig(subject=SubjectConfig(face_width_source="ears"))
    tracker = KeypointTracker2D(cfg, COCO17)
    upper = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_wrist", "right_wrist")
    points = {
        "nose": (320.0, 200.0),
        "left_eye": (300.0, 190.0),
        "right_eye": (340.0, 190.0),
        "left_ear": (270.0, 200.0),
        "right_ear": (370.0, 200.0),
        "left_shoulder": (250.0, 380.0),
        "right_shoulder": (390.0, 380.0),
        "left_elbow": (230.0, 460.0),
        "right_elbow": (410.0, 460.0),
        "left_wrist": (220.0, 540.0),
        "right_wrist": (420.0, 540.0),
    }
    conf = {name: 0.85 for name in upper}
    conf.update({"left_shoulder": 0.85, "right_shoulder": 0.85})
    half = _person((180.0, 120.0, 460.0, 520.0), points, conf, layout=COCO17)
    full = _person((180.0, 100.0, 460.0, 1100.0), points, conf, layout=COCO17)

    half_frame = tracker.update(_frame(half))
    assert half_frame.framing is FramingState.HALF_BODY
    for name in upper:
        assert half_frame.conf[COCO17.index(name)] == pytest.approx(0.85)
        assert half_frame.status[COCO17.index(name)] == 0

    full_frame = tracker.update(_frame(full, seq=1, t=1.0 / 30.0))
    assert full_frame.framing is FramingState.FULL_BODY
    for name in upper:
        assert full_frame.conf[COCO17.index(name)] == pytest.approx(0.85)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ears", 100.0),
        ("eyes", 88.0),
        ("jaw", 60.0),
        ("auto", 60.0),
    ],
)
def test_face_width_sources(source: str, expected: float) -> None:
    cfg = Tracker2DConfig(subject=SubjectConfig(face_width_source=source))
    tracked = KeypointTracker2D(cfg, _LAYOUT).update(_frame(_person((180.0, 120.0, 460.0, 570.0))))
    assert tracked.face_width_px == pytest.approx(expected)


def test_auto_face_width_skips_missing_sources() -> None:
    cfg = Tracker2DConfig(subject=SubjectConfig(face_width_source="auto"))
    conf = {name: 0.9 for name in _points()}
    conf["face_jaw_left"] = 0.0
    conf["face_jaw_right"] = 0.0
    person = _person((180.0, 120.0, 460.0, 570.0), conf=conf)
    tracked = KeypointTracker2D(cfg, _LAYOUT).update(_frame(person))
    assert tracked.face_width_px == pytest.approx(100.0)


def test_face_width_is_a_stabilizing_median() -> None:
    cfg = Tracker2DConfig(subject=SubjectConfig(face_width_source="eyes", face_width_window=5))
    tracker = KeypointTracker2D(cfg, _LAYOUT)
    half_distances = (15.0, 25.0, 20.0, 10.0, 30.0)  # eye distance / 2
    widths = []
    for index, half in enumerate(half_distances):
        points = _points()
        points["left_eye"] = (320.0 - half, 190.0)
        points["right_eye"] = (320.0 + half, 190.0)
        tracked = tracker.update(_frame(_person((180.0, 120.0, 460.0, 570.0), points), seq=index))
        widths.append(tracked.face_width_px)
    # Face widths: 66, 110, 88, 44, 132 -> median 88.
    assert widths[-1] == pytest.approx(88.0)


def test_two_persons_give_one_subject_per_rule() -> None:
    centered = _person((300.0, 220.0, 340.0, 260.0), score=0.4)
    large_corner = _person((0.0, 0.0, 200.0, 200.0), score=0.9)
    persons = (large_corner, centered)

    largest = select_subject(persons, 640, 480, SubjectConfig(rule="largest"))
    central = select_subject(persons, 640, 480, SubjectConfig(rule="central"))
    assert largest is large_corner
    assert central is centered

    cfg = Tracker2DConfig(subject=SubjectConfig(rule="central"))
    tracked = KeypointTracker2D(cfg, _LAYOUT).update(
        PoseFrame(seq=0, t=0.0, width=640, height=480, persons=persons)
    )
    assert tracked.box == pytest.approx(np.asarray(centered.box))
    assert tracked.xy[_LAYOUT.index("nose")] == pytest.approx((320.0, 200.0))


def test_invalid_boxes_are_ignored() -> None:
    empty = _person((100.0, 100.0, 100.0, 200.0))
    invalid = _person((0.0, 0.0, float("nan"), 100.0))
    valid = _person((200.0, 200.0, 300.0, 300.0), score=0.1)
    selected = select_subject((empty, invalid, valid), 640, 480, SubjectConfig())
    assert selected is valid
    assert select_subject((empty, invalid), 640, 480, SubjectConfig()) is None


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SubjectConfig(rule="closest")
    with pytest.raises(ValueError):
        SubjectConfig(face_width_source="nose")
    with pytest.raises(ValueError):
        SubjectConfig(face_width_window=0)
    with pytest.raises(ValueError):
        SubjectConfig(full_body_ratio=0.0)
