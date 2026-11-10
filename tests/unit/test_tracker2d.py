"""2D tracker: filtering, dropout statuses, gating and model swap (AC-004, AC-015)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa.domain.model import Side
from cpsa.vision.layouts import HALFBUST_62
from cpsa.vision.skeleton import SkeletonConfig
from cpsa.vision.subject import SubjectConfig
from cpsa.vision.tracker2d import KeypointTracker2D, Tracker2DConfig
from cpsa.vision.types import (
    KeypointLayout,
    KeypointStatus,
    PersonPose,
    PoseFrame,
    TrackedFrame,
)

_LAYOUT: KeypointLayout = HALFBUST_62
_SIZE = len(_LAYOUT)
_NOSE = _LAYOUT.index("nose")
_DT = 1.0 / 30.0


def _person(
    points: dict[str, tuple[float, float]],
    conf: float | dict[str, float] = 0.9,
    box: tuple[float, float, float, float] = (180.0, 120.0, 460.0, 720.0),
    score: float = 0.9,
) -> PersonPose:
    xy = np.zeros((_SIZE, 2), dtype=np.float64)
    confidences = np.zeros(_SIZE, dtype=np.float64)
    for name, value in points.items():
        index = _LAYOUT.index(name)
        xy[index] = value
        confidences[index] = conf[name] if isinstance(conf, dict) else conf
    return PersonPose(
        box=np.asarray(box, dtype=np.float64),
        score=score,
        xy=xy,
        conf=confidences,
    )


def _frame(seq: int, t: float, person: PersonPose | None) -> PoseFrame:
    persons = () if person is None else (person,)
    return PoseFrame(seq=seq, t=t, width=640, height=480, persons=persons)


def _body_points() -> dict[str, tuple[float, float]]:
    return {
        "nose": (320.0, 200.0),
        "left_eye": (300.0, 190.0),
        "right_eye": (340.0, 190.0),
        "left_ear": (270.0, 200.0),
        "right_ear": (370.0, 200.0),
        "left_shoulder": (250.0, 260.0),
        "right_shoulder": (390.0, 260.0),
        "left_elbow": (230.0, 340.0),
        "right_elbow": (410.0, 340.0),
        "left_wrist": (220.0, 420.0),
        "right_wrist": (420.0, 420.0),
    }


def _trajectory(
    frame_count: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(frame_count, dtype=np.float64) * _DT
    truth = np.stack(
        [
            320.0 + 40.0 * np.sin(2.0 * np.pi * 0.7 * t),
            200.0 + 15.0 * np.cos(2.0 * np.pi * 0.7 * t),
        ],
        axis=1,
    )
    rng = np.random.default_rng(4)
    observed = truth + rng.normal(0.0, 3.0, truth.shape)
    return t, truth, observed


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def test_filter_beats_raw_differencing() -> None:
    t, truth, observed = _trajectory()
    tracker = KeypointTracker2D(Tracker2DConfig(), _LAYOUT)
    tracked = [
        tracker.update(_frame(index, t[index], _person({"nose": observed[index]})))
        for index in range(len(t))
    ]
    estimate = np.array([frame.xy[_NOSE] for frame in tracked])
    velocity = np.array([frame.vel[_NOSE] for frame in tracked])

    truth_velocity = np.stack(
        [
            40.0 * 2.0 * np.pi * 0.7 * np.cos(2.0 * np.pi * 0.7 * t),
            -15.0 * 2.0 * np.pi * 0.7 * np.sin(2.0 * np.pi * 0.7 * t),
        ],
        axis=1,
    )
    raw_velocity = (observed[2:] - observed[:-2]) / (2.0 * _DT)
    raw_position_rmse = _rmse(observed - truth)
    filtered_position_rmse = _rmse(estimate - truth)
    raw_velocity_rmse = _rmse(raw_velocity - truth_velocity[1:-1])
    filtered_velocity_rmse = _rmse(velocity[5:-2] - truth_velocity[5:-2])

    assert filtered_position_rmse < 0.9 * raw_position_rmse
    assert filtered_velocity_rmse < 0.75 * raw_velocity_rmse

    # P-01: every keypoint carries a finite, positive acceleration uncertainty.
    for frame in tracked:
        assert np.all(np.isfinite(frame.acc_std))
        assert np.all(frame.acc_std > 0.0)

    smoothed = tracker.smoothed()
    assert len(smoothed) == len(tracked)
    smoothed_rmse = _rmse(np.array([frame.xy[_NOSE] for frame in smoothed]) - truth)
    assert smoothed_rmse < filtered_position_rmse


def test_short_dropout_is_predicted_and_long_dropout_is_lost() -> None:
    frame_count = 60
    t = np.arange(frame_count, dtype=np.float64) * _DT
    truth = np.stack([320.0 + 60.0 * t, 200.0 + 10.0 * t], axis=1)
    short_gap = set(range(20, 23))
    long_gap = set(range(40, 56))
    tracker = KeypointTracker2D(
        Tracker2DConfig(max_predict_frames=5),
        _LAYOUT,
    )
    tracked: list[TrackedFrame] = []
    for index in range(frame_count):
        confidence = 0.0 if index in short_gap | long_gap else 0.9
        person = _person({"nose": truth[index]}, conf={"nose": confidence})
        tracked.append(tracker.update(_frame(index, t[index], person)))

    status = [int(frame.status[_NOSE]) for frame in tracked]
    assert status[19] == int(KeypointStatus.MEASURED)
    assert all(status[index] == int(KeypointStatus.PREDICTED) for index in range(20, 23))
    assert status[23] == int(KeypointStatus.MEASURED)
    assert all(status[index] == int(KeypointStatus.PREDICTED) for index in range(40, 45))
    assert all(status[index] == int(KeypointStatus.LOST) for index in range(45, 56))
    # A lost keypoint is re-initialized by its next accepted measurement.
    assert status[56] == int(KeypointStatus.MEASURED)
    assert tracked[56].xy[_NOSE] == pytest.approx(truth[56])


def test_no_person_still_predicts_then_marks_lost() -> None:
    tracker = KeypointTracker2D(Tracker2DConfig(max_predict_frames=5), _LAYOUT)
    for index in range(3):
        tracker.update(_frame(index, index * _DT, _person({"nose": (320.0, 200.0)})))
    status = []
    for index in range(3, 10):
        status.append(int(tracker.update(_frame(index, index * _DT, None)).status[_NOSE]))
    assert status[:5] == [int(KeypointStatus.PREDICTED)] * 5
    assert status[5:] == [int(KeypointStatus.LOST)] * 2


def test_chi_square_gate_rejects_an_outlier() -> None:
    tracker = KeypointTracker2D(Tracker2DConfig(max_predict_frames=5), _LAYOUT)
    t = np.arange(30, dtype=np.float64) * _DT
    truth = np.stack([320.0 + 120.0 * t, np.full_like(t, 200.0)], axis=1)
    outlier = 24
    observed = truth.copy()
    observed[outlier] = observed[outlier] + np.array([300.0, 0.0])

    tracked = []
    for index in range(len(t)):
        person = _person({"nose": observed[index]}, conf={"nose": 0.95})
        tracked.append(tracker.update(_frame(index, t[index], person)))

    assert int(tracked[outlier].status[_NOSE]) == int(KeypointStatus.PREDICTED)
    error = float(np.linalg.norm(tracked[outlier].xy[_NOSE] - truth[outlier]))
    assert error < 20.0
    assert int(tracked[outlier + 1].status[_NOSE]) == int(KeypointStatus.MEASURED)


def test_skeleton_rejects_a_bone_breaking_measurement() -> None:
    cfg = Tracker2DConfig(
        subject=SubjectConfig(face_width_source="ears", face_width_window=1),
        skeleton=SkeletonConfig(ref_min_frames=3, ref_frames=6),
    )
    tracker = KeypointTracker2D(cfg, _LAYOUT)
    for index in range(4):
        tracker.update(_frame(index, index * _DT, _person(_body_points())))

    broken = _body_points()
    broken["left_wrist"] = (220.0, 900.0)
    conf = {name: 0.9 for name in broken}
    conf["left_wrist"] = 0.4
    tracked = tracker.update(_frame(4, 4 * _DT, _person(broken, conf=conf)))

    assert int(tracked.status[_LAYOUT.index("left_wrist")]) == int(KeypointStatus.PREDICTED)
    assert int(tracked.status[_LAYOUT.index("left_elbow")]) == int(KeypointStatus.MEASURED)
    assert float(np.linalg.norm(tracked.xy[_LAYOUT.index("left_wrist")] - (220.0, 900.0))) > 50.0
    assert tracked.foreshortening[Side.LEFT] == pytest.approx(1.0, rel=0.1)


def test_variable_dt_from_timestamps() -> None:
    rng = np.random.default_rng(11)
    times = np.cumsum(rng.uniform(0.02, 0.06, 40))
    tracker = KeypointTracker2D(Tracker2DConfig(), _LAYOUT)
    velocity = np.array([150.0, -50.0])
    start = np.array([200.0, 300.0])
    last: TrackedFrame | None = None
    for index, t in enumerate(times):
        position = start + velocity * t
        last = tracker.update(_frame(index, t, _person({"nose": position})))
    assert last is not None
    assert last.vel[_NOSE] == pytest.approx(velocity, rel=0.1, abs=15.0)


class _ConstantVelocityMotion:
    """Test-only motion model: a different F/Q with the same six-dimensional layout."""

    dim = 6

    def __init__(self, accel_std_pxps2: float = 60.0) -> None:
        self._variance = accel_std_pxps2 * accel_std_pxps2

    def transition(self, dt: float) -> np.ndarray:
        transition = np.eye(6, dtype=np.float64)
        transition[0, 2] = dt
        transition[1, 3] = dt
        return transition

    def process_noise(self, dt: float) -> np.ndarray:
        q = self._variance
        covariance = np.zeros((6, 6), dtype=np.float64)
        covariance[0, 0] = q * dt**3 / 3.0
        covariance[1, 1] = q * dt**3 / 3.0
        covariance[0, 2] = q * dt**2 / 2.0
        covariance[2, 0] = covariance[0, 2]
        covariance[1, 3] = q * dt**2 / 2.0
        covariance[3, 1] = covariance[1, 3]
        covariance[2, 2] = q * dt
        covariance[3, 3] = q * dt
        return covariance


def _consume(frames: list[TrackedFrame]) -> dict[str, float]:
    """Representative consumer: it only reads the frozen ``TrackedFrame`` fields."""
    positions = np.array([frame.xy[_NOSE] for frame in frames])
    accelerations = np.array([frame.acc[_NOSE] for frame in frames])
    measured = sum(
        1 for frame in frames if int(frame.status[_NOSE]) == int(KeypointStatus.MEASURED)
    )
    return {
        "frames": float(len(frames)),
        "measured": float(measured),
        "finite": float(np.all(np.isfinite(positions))),
        "max_abs_acc": float(np.max(np.abs(accelerations))),
    }


def test_test_only_motion_model_plugs_in_without_consumer_change() -> None:
    t, _, observed = _trajectory(frame_count=45)
    default = KeypointTracker2D(Tracker2DConfig(), _LAYOUT)
    swapped = KeypointTracker2D(Tracker2DConfig(), _LAYOUT, motion=_ConstantVelocityMotion())
    default_frames = [
        default.update(_frame(index, t[index], _person({"nose": observed[index]})))
        for index in range(len(t))
    ]
    swapped_frames = [
        swapped.update(_frame(index, t[index], _person({"nose": observed[index]})))
        for index in range(len(t))
    ]

    default_summary = _consume(default_frames)
    swapped_summary = _consume(swapped_frames)
    assert default_summary["frames"] == swapped_summary["frames"] == len(t)
    assert default_summary["finite"] == swapped_summary["finite"] == 1.0
    assert swapped_summary["measured"] > 0.5 * len(t)
    # The swapped model has no acceleration state dynamics, so the same consumer
    # sees a different acceleration trace without any change to its own code.
    assert swapped_summary["max_abs_acc"] < 1e-6
    assert default_summary["max_abs_acc"] > 1.0
    assert len(swapped.smoothed()) == len(t)


def test_reset_keeps_skeleton_references_and_reset_session_clears_them() -> None:
    cfg = Tracker2DConfig(
        subject=SubjectConfig(face_width_source="ears", face_width_window=1),
        skeleton=SkeletonConfig(ref_min_frames=4, ref_frames=8),
    )
    tracker = KeypointTracker2D(cfg, _LAYOUT)
    for index in range(6):
        tracker.update(_frame(index, index * _DT, _person(_body_points())))

    references = dict(tracker.skeleton.reference_fw)
    assert len(references) == 5
    tracked = tracker.update(_frame(6, 6 * _DT, _person(_body_points())))
    assert tracked.face_width_px == pytest.approx(100.0)
    assert tracked.foreshortening[Side.LEFT] == pytest.approx(1.0, rel=0.05)

    tracker.reset()
    assert tracker.smoothed() == []
    assert dict(tracker.skeleton.reference_fw) == references
    empty = tracker.update(_frame(7, 7 * _DT, None))
    assert np.all(empty.status == int(KeypointStatus.LOST))

    for index in range(6):
        tracker.update(_frame(10 + index, (10 + index) * _DT, _person(_body_points())))
    assert dict(tracker.skeleton.reference_fw) == references
    assert len(tracker.smoothed()) == 7  # the empty frame is part of this episode

    tracker.reset_session()
    assert dict(tracker.skeleton.reference_fw) == {}
    for index in range(4):
        tracker.update(_frame(20 + index, (20 + index) * _DT, _person(_body_points())))
    assert dict(tracker.skeleton.reference_fw) == pytest.approx(references)


def test_invalid_person_shape_raises() -> None:
    tracker = KeypointTracker2D(Tracker2DConfig(), _LAYOUT)
    with pytest.raises(ValueError):
        tracker.update(
            PoseFrame(
                seq=0,
                t=0.0,
                width=640,
                height=480,
                persons=(
                    PersonPose(
                        box=np.array([0.0, 0.0, 100.0, 100.0]),
                        score=0.9,
                        xy=np.zeros((_SIZE - 1, 2)),
                        conf=np.zeros(_SIZE),
                    ),
                ),
            )
        )


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        Tracker2DConfig(min_conf=0.0)
    with pytest.raises(ValueError):
        Tracker2DConfig(max_predict_frames=-1)
    with pytest.raises(ValueError):
        Tracker2DConfig(meas_std_px=0.0)

    class _BadDimension:
        dim = 4

        def transition(self, dt: float) -> np.ndarray:
            return np.eye(4)

        def process_noise(self, dt: float) -> np.ndarray:
            return np.zeros((4, 4))

    with pytest.raises(ValueError):
        KeypointTracker2D(Tracker2DConfig(), _LAYOUT, motion=_BadDimension())
