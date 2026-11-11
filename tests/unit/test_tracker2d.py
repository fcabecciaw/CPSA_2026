"""2D tracker: filtering, dropout statuses, gating and model swap (AC-004, AC-015)."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa.domain.model import HandShape, Side
from cpsa.vision.layouts import HALFBUST_62
from cpsa.vision.skeleton import SkeletonConfig
from cpsa.vision.subject import SubjectConfig, framing_state, select_subject
from cpsa.vision.tracker2d import KeypointTracker2D, Tracker2DConfig, _FrameRecord
from cpsa.vision.types import (
    FramingState,
    KeypointLayout,
    KeypointStatus,
    PersonPose,
    PoseFrame,
    TrackedFrame,
)
from tests.unit import vision_fixtures as fx
from tests.unit.test_oscillation import _flapping_frames, _rocking_frames

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


# ----------------------------------------------------------------------
# Batched-update equivalence oracle
# ----------------------------------------------------------------------
#
# ``KeypointTracker2D.update`` batches the per-frame Kalman math over the
# keypoints. The class below is the per-keypoint implementation it replaced,
# copied verbatim, and acts as the oracle for every equivalence check. Only the
# shared identity matrix moved to a class attribute because the vectorized
# tracker no longer keeps one.

_STATE_DIM = 6


class _ReferenceKeypointTracker(KeypointTracker2D):
    """Per-keypoint reference filter: the pre-vectorization ``update``."""

    _identity = np.eye(_STATE_DIM, dtype=np.float64)

    def update(self, frame: PoseFrame) -> TrackedFrame:
        """Filter one pose frame and return the tracked subject."""
        person = select_subject(frame.persons, frame.width, frame.height, self._cfg.subject)
        if person is None:
            face_width = self._face_width.value
            framing = FramingState.NO_PERSON
            box: np.ndarray | None = None
            accepted = np.zeros(self._size, dtype=np.bool_)
            conf = np.zeros(self._size, dtype=np.float64)
            points = np.zeros((self._size, 2), dtype=np.float64)
        else:
            points, person_conf = self._validate_person(person)
            face_width = self._face_width.update(person, self._layout)
            framing = framing_state(person, self._layout, face_width, self._cfg.subject)
            box = np.asarray(person.box, dtype=np.float64).copy()
            accepted = self.skeleton.observe(points, person_conf, face_width)
            conf = person_conf.copy()

        measured = accepted & np.isfinite(conf) & (conf >= self._cfg.min_conf)
        dt = self._time_step(frame.t)
        transition = np.asarray(self._motion.transition(dt), dtype=np.float64)
        noise = np.asarray(self._motion.process_noise(dt), dtype=np.float64)
        if transition.shape != (_STATE_DIM, _STATE_DIM):
            raise ValueError(f"motion transition must have shape ({_STATE_DIM}, {_STATE_DIM})")
        if noise.shape != (_STATE_DIM, _STATE_DIM):
            raise ValueError(f"motion process noise must have shape ({_STATE_DIM}, {_STATE_DIM})")

        x_pred = self._x.copy()
        p_pred = self._p.copy()
        predict = self._initialized & (self._missed <= self._cfg.max_predict_frames)
        for index in np.flatnonzero(predict):
            keypoint = int(index)
            x_pred[keypoint] = transition @ self._x[keypoint]
            p_pred[keypoint] = transition @ self._p[keypoint] @ transition.T + noise

        x_filt = x_pred.copy()
        p_filt = p_pred.copy()
        status = np.full(self._size, int(KeypointStatus.LOST), dtype=np.int8)
        conf_out = self._conf.copy()

        for keypoint in range(self._size):
            if measured[keypoint]:
                if (
                    not self._initialized[keypoint]
                    or self._missed[keypoint] > self._cfg.max_predict_frames
                ):
                    self._initialize(x_filt, p_filt, keypoint, points[keypoint])
                    status[keypoint] = int(KeypointStatus.MEASURED)
                    conf_out[keypoint] = float(conf[keypoint])
                else:
                    update = self._measurement_update(
                        x_pred[keypoint], p_pred[keypoint], points[keypoint], float(conf[keypoint])
                    )
                    if update is None:
                        self._advance_dropout(keypoint, status)
                    else:
                        gain, innovation = update
                        x_filt[keypoint] = x_pred[keypoint] + gain @ innovation
                        p_filt[keypoint] = (
                            self._identity - gain @ self._measurement_matrix
                        ) @ p_pred[keypoint]
                        p_filt[keypoint] = 0.5 * (p_filt[keypoint] + p_filt[keypoint].T)
                        status[keypoint] = int(KeypointStatus.MEASURED)
                        conf_out[keypoint] = float(conf[keypoint])
                        self._missed[keypoint] = 0
            elif self._initialized[keypoint]:
                self._advance_dropout(keypoint, status)

        self._x = x_filt
        self._p = p_filt
        self._conf = conf_out
        self._last_t = float(frame.t)

        tracked = self._build_frame(
            frame.seq,
            frame.t,
            framing,
            box,
            x_filt,
            p_filt,
            status,
            conf_out,
            face_width,
        )
        self._history.append(
            _FrameRecord(
                seq=frame.seq,
                t=float(frame.t),
                framing=framing,
                box=box,
                face_width_px=face_width,
                status=status.copy(),
                conf=conf_out.copy(),
                initialized=self._initialized.copy(),
                segment=self._segment.copy(),
                x_pred=x_pred,
                p_pred=p_pred,
                x_filt=x_filt.copy(),
                p_filt=p_filt.copy(),
                dt=dt,
            )
        )
        return tracked

    def _initialize(
        self,
        x_filt: np.ndarray,
        p_filt: np.ndarray,
        keypoint: int,
        position: np.ndarray,
    ) -> None:
        x_filt[keypoint] = 0.0
        x_filt[keypoint, 0:2] = position
        p_filt[keypoint] = self._initial_covariance
        self._initialized[keypoint] = True
        self._segment[keypoint] += 1
        self._missed[keypoint] = 0

    def _measurement_update(
        self,
        x_pred: np.ndarray,
        p_pred: np.ndarray,
        position: np.ndarray,
        conf: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(gain, innovation)`` when the measurement passes the gate."""
        noise = np.asarray(self._measurement.noise(conf), dtype=np.float64)
        if noise.shape != (2, 2):
            raise ValueError(f"measurement noise must have shape (2, 2), got {noise.shape}")
        innovation = position - self._measurement_matrix @ x_pred
        covariance = self._measurement_matrix @ p_pred @ self._measurement_matrix.T + noise
        try:
            distance = float(innovation @ np.linalg.solve(covariance, innovation))
            gain = np.asarray(
                p_pred @ self._measurement_matrix.T @ np.linalg.pinv(covariance),
                dtype=np.float64,
            )
        except (np.linalg.LinAlgError, ValueError):
            return None
        if not np.isfinite(distance) or distance > self._cfg.gate_chi2:
            return None
        return gain, innovation

    def _advance_dropout(self, keypoint: int, status: np.ndarray) -> None:  # type: ignore[override]
        self._missed[keypoint] += 1
        if self._missed[keypoint] > self._cfg.max_predict_frames:
            status[keypoint] = int(KeypointStatus.LOST)
        else:
            status[keypoint] = int(KeypointStatus.PREDICTED)


class _TiltedMeasurement:
    """Test-only measurement model with a non-canonical observation matrix."""

    def observe_matrix(self) -> np.ndarray:
        matrix = np.zeros((2, _STATE_DIM), dtype=np.float64)
        matrix[0, 0] = 1.0
        matrix[1, 0] = 0.5
        matrix[1, 1] = 0.5
        return matrix

    def noise(self, conf: float) -> np.ndarray:
        variance = (4.0 / max(conf, 0.05)) ** 2
        return np.array([[variance, 0.25 * variance], [0.25 * variance, variance]])


def _pose(seq: int, t: float, points: np.ndarray, conf: float | np.ndarray) -> PoseFrame:
    """Build a single-person pose frame from keypoint positions and confidences."""
    xy = np.array(points, dtype=np.float64, copy=True)
    confidences = np.asarray(conf, dtype=np.float64)
    if confidences.ndim == 0:
        confidences = np.full(xy.shape[0], float(confidences), dtype=np.float64)
    visible = xy[np.all(np.isfinite(xy), axis=1)]
    box = np.array(
        [visible[:, 0].min(), visible[:, 1].min(), visible[:, 0].max(), visible[:, 1].max()],
        dtype=np.float64,
    )
    return PoseFrame(
        seq=seq,
        t=float(t),
        width=640,
        height=480,
        persons=(PersonPose(box=box, score=0.9, xy=xy, conf=confidences),),
    )


def _scene_noisy_trajectory() -> list[PoseFrame]:
    """Filtered nose trajectory with irregular timestamps, as in the filter tests."""
    roots = np.arange(60, dtype=np.float64) * _DT
    times = np.cumsum(np.linspace(0.02, 0.06, 60))
    observed = np.stack(
        [320.0 + 40.0 * np.sin(2.0 * np.pi * 0.7 * roots), 200.0 + 15.0 * np.cos(3.0 * times)],
        axis=1,
    )
    frames = []
    for index in range(60):
        points = np.zeros((_SIZE, 2), dtype=np.float64)
        points[_NOSE] = observed[index]
        frames.append(_pose(index, times[index], points, 0.9))
    return frames


def _scene_neutral() -> list[PoseFrame]:
    """Relaxed seated scene: all 62 keypoints measured with sensor noise."""
    rng = np.random.default_rng(5)
    frames = []
    for index in range(45):
        xy = fx.neutral_xy() + rng.normal(0.0, 1.5, (fx.KEYPOINT_COUNT, 2))
        frames.append(_pose(index, index * fx.DT, xy, 0.9))
    return frames


def _scene_hand_to_face() -> list[PoseFrame]:
    """Right hand approaching the chin: fast motion plus a foreshortening change."""
    tracked = fx.distance_profile(
        [(10.0, 2.5), (11.5, 0.3)],
        side=Side.RIGHT,
        shape=HandShape.POINT_PINCH,
        anchor=fx.face_anchor("face_chin"),
    )
    return [_pose(frame.seq, frame.t, frame.xy, 0.9) for frame in tracked]


def _scene_rocking() -> list[PoseFrame]:
    """Shoulder-midpoint rocking frames from the oscillation helpers."""
    return [_pose(frame.seq, frame.t, frame.xy, 0.9) for frame in _rocking_frames()]


def _scene_flapping() -> list[PoseFrame]:
    """Fast wrist-flapping frames from the oscillation helpers."""
    return [_pose(frame.seq, frame.t, frame.xy, 0.9) for frame in _flapping_frames()]


def _scene_dropouts_outliers_and_missing_person() -> list[PoseFrame]:
    """Random confidence dropouts, a gated outlier and a frame without a person."""
    rng = np.random.default_rng(21)
    base = fx.neutral_xy()
    frames = []
    for index in range(60):
        conf = np.full(fx.KEYPOINT_COUNT, 0.9)
        if index in {12, 13, 30, 46, 47, 48, 49, 50}:
            conf[rng.choice(fx.KEYPOINT_COUNT, 8, replace=False)] = 0.0
        if index == 33:
            frames.append(PoseFrame(seq=index, t=index * fx.DT, width=640, height=480, persons=()))
            continue
        xy = base + np.array([2.0 * index, 0.0])
        if index == 20:
            xy[fx.LAYOUT.index("nose")] += np.array([250.0, 0.0])
        frames.append(_pose(index, index * fx.DT, xy, conf))
    return frames


def _scene_reinitialization() -> list[PoseFrame]:
    """A gap longer than ``max_predict_frames``: LOST keypoints re-initialize."""
    base = fx.neutral_xy()
    frames = []
    for index in range(40):
        conf = np.full(fx.KEYPOINT_COUNT, 0.9)
        if 8 <= index < 20:
            conf[:] = 0.0
        frames.append(_pose(index, index * fx.DT, base + np.array([3.0 * index, 0.0]), conf))
    return frames


_EQUIVALENCE_SCENES = {
    "noisy-trajectory": _scene_noisy_trajectory,
    "neutral": _scene_neutral,
    "hand-to-face": _scene_hand_to_face,
    "rocking": _scene_rocking,
    "flapping": _scene_flapping,
    "dropouts-outliers-missing": _scene_dropouts_outliers_and_missing_person,
    "reinitialization": _scene_reinitialization,
}


def _assert_same_frame(actual: TrackedFrame, expected: TrackedFrame, tolerance: float) -> None:
    """Assert that two tracked frames carry the same content within ``tolerance``."""
    assert actual.seq == expected.seq
    assert actual.t == expected.t
    assert actual.framing is expected.framing
    if expected.box is None:
        assert actual.box is None
    else:
        np.testing.assert_allclose(actual.box, expected.box, rtol=0.0, atol=tolerance)
    if expected.face_width_px is None:
        assert actual.face_width_px is None
    else:
        assert actual.face_width_px == pytest.approx(expected.face_width_px, abs=tolerance)
    for field in ("xy", "vel", "acc", "xy_std", "vel_std", "acc_std", "conf"):
        np.testing.assert_allclose(
            getattr(actual, field),
            getattr(expected, field),
            rtol=tolerance,
            atol=tolerance,
            err_msg=field,
        )
    np.testing.assert_array_equal(actual.status, expected.status)
    assert set(actual.foreshortening) == set(expected.foreshortening)
    for side, value in expected.foreshortening.items():
        if value is None:
            assert actual.foreshortening[side] is None
        else:
            assert actual.foreshortening[side] == pytest.approx(value, rel=tolerance, abs=tolerance)


def _assert_same_record(actual: _FrameRecord, expected: _FrameRecord) -> None:
    """Assert that one smoother history record carries the same content."""
    assert actual.seq == expected.seq
    assert actual.t == expected.t
    assert actual.framing is expected.framing
    assert actual.dt == expected.dt
    assert actual.face_width_px == pytest.approx(expected.face_width_px)
    if expected.box is None:
        assert actual.box is None
    else:
        np.testing.assert_allclose(actual.box, expected.box, rtol=0.0, atol=1e-9)
    np.testing.assert_array_equal(actual.status, expected.status)
    np.testing.assert_array_equal(actual.initialized, expected.initialized)
    np.testing.assert_array_equal(actual.segment, expected.segment)
    for field in ("conf", "x_pred", "p_pred", "x_filt", "p_filt"):
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), rtol=1e-9, atol=1e-9, err_msg=field
        )


def _assert_same_episode(
    batched: KeypointTracker2D, reference: KeypointTracker2D, frames: list[PoseFrame]
) -> None:
    """Drive both trackers through ``frames`` and compare filtered and smoothed output."""
    for frame in frames:
        _assert_same_frame(batched.update(frame), reference.update(frame), 1e-9)
    assert len(batched._history) == len(reference._history)
    for actual_record, expected_record in zip(batched._history, reference._history, strict=True):
        _assert_same_record(actual_record, expected_record)
    smoothed = batched.smoothed()
    expected = reference.smoothed()
    assert len(smoothed) == len(expected) == len(frames)
    for actual, reference_frame in zip(smoothed, expected, strict=True):
        _assert_same_frame(actual, reference_frame, 1e-7)


@pytest.mark.parametrize("scene", list(_EQUIVALENCE_SCENES.values()), ids=list(_EQUIVALENCE_SCENES))
def test_batched_update_matches_the_per_keypoint_reference(scene) -> None:
    frames = scene()

    _assert_same_episode(
        KeypointTracker2D(Tracker2DConfig(), _LAYOUT),
        _ReferenceKeypointTracker(Tracker2DConfig(), _LAYOUT),
        frames,
    )


def test_batched_update_matches_the_reference_with_swapped_models() -> None:
    frames = _scene_neutral() + _scene_flapping()[:30]
    cfg = Tracker2DConfig(jerk_std_pxps3=250.0, meas_std_px=5.0)

    _assert_same_episode(
        KeypointTracker2D(
            cfg, _LAYOUT, motion=_ConstantVelocityMotion(), measurement=_TiltedMeasurement()
        ),
        _ReferenceKeypointTracker(
            cfg, _LAYOUT, motion=_ConstantVelocityMotion(), measurement=_TiltedMeasurement()
        ),
        frames,
    )


def test_equivalence_scenes_exercise_dropout_gating_and_reinitialization() -> None:
    tracker = KeypointTracker2D(Tracker2DConfig(), _LAYOUT)
    statuses = [
        tracker.update(frame).status for frame in _scene_dropouts_outliers_and_missing_person()
    ]
    assert any(np.any(status == int(KeypointStatus.PREDICTED)) for status in statuses)
    assert statuses[33][_NOSE] == int(KeypointStatus.PREDICTED)  # no person: predicted
    assert statuses[20][_NOSE] == int(KeypointStatus.PREDICTED)  # 250 px outlier gated
    assert statuses[21][_NOSE] == int(KeypointStatus.MEASURED)

    tracker = KeypointTracker2D(Tracker2DConfig(), _LAYOUT)
    statuses = [tracker.update(frame).status[_NOSE] for frame in _scene_reinitialization()]
    assert statuses[0] == int(KeypointStatus.MEASURED)
    assert statuses[13] == int(KeypointStatus.LOST)
    assert statuses[19] == int(KeypointStatus.LOST)
    assert statuses[20] == int(KeypointStatus.MEASURED)  # re-initialized after the gap


def test_pass_through_returns_the_raw_detections_without_filtering() -> None:
    tracker = KeypointTracker2D(Tracker2DConfig(filter=False), _LAYOUT)
    points = _body_points()
    tracker.update(_frame(0, 0.0, _person(points)))
    moved = {name: (x + 40.0, y) for name, (x, y) in points.items()}

    tracked = tracker.update(_frame(1, _DT, _person(moved)))

    nose = moved["nose"]
    assert tuple(tracked.xy[_NOSE]) == pytest.approx(nose, abs=1e-9)
    assert tracked.status[_NOSE] == KeypointStatus.MEASURED
    gone = tracker.update(_frame(2, 2 * _DT, None))
    assert gone.status[_NOSE] == KeypointStatus.LOST
