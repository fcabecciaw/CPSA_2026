"""Per-keypoint 2D constant-acceleration Kalman tracker (FR-007, FR-008, NFR-001).

Each keypoint carries a six-dimensional state ``[x, y, vx, vy, ax, ay]`` in
pixels and seconds. Measurements below a confidence threshold, measurements
rejected by the skeleton plausibility check and measurements outside the
chi-square gate are replaced by a prediction; after ``max_predict_frames``
consecutive predictions the keypoint becomes ``LOST`` and is re-initialized by
its next accepted measurement (FR-008).

The motion and measurement models are protocols, so a different filter (for
example a later articulated EKF) plugs in without changing any consumer
(NFR-001). The forward filter stores its predictions and covariances so
:meth:`KeypointTracker2D.smoothed` can run a Rauch-Tung-Striebel pass over the
frames since the last reset.

The tracker output is a :class:`~cpsa.vision.types.TrackedFrame` per input
frame, including the filtered position, velocity and acceleration, their
one-sigma uncertainties (``acc_std`` is required by FR-007), the per-keypoint
status, the framing state and the forearm foreshortening ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cpsa.vision.skeleton import SkeletonConfig, SkeletonModel
from cpsa.vision.subject import SubjectConfig, _FaceWidthTracker, framing_state, select_subject
from cpsa.vision.types import (
    FramingState,
    KeypointLayout,
    KeypointStatus,
    PersonPose,
    PoseFrame,
    TrackedFrame,
)

__all__ = ["KeypointTracker2D", "MeasurementModel", "MotionModel", "Tracker2DConfig"]

_STATE_DIM = 6
_POSITION = slice(0, 2)
_VELOCITY = slice(2, 4)
_ACCELERATION = slice(4, 6)
_TINY_VARIANCE = 1e-12
_MIN_CONF = 1e-3
_MAX_CONF = 1.0


class MotionModel(Protocol):
    """Swappable linear motion model (NFR-001).

    The tracker state layout is ``[x, y, vx, vy, ax, ay]`` and ``dim`` must be
    6. ``transition(dt)`` returns the state-transition matrix ``F`` and
    ``process_noise(dt)`` the process covariance ``Q``. A model whose
    ``F``/``Q`` are not the linear operators the RTS smoother needs may set a
    ``smoothing = False`` attribute; ``smoothed`` then returns the filtered
    frames instead.
    """

    @property
    def dim(self) -> int:
        """State dimension, which must be 6 for this tracker."""
        ...

    def transition(self, dt: float) -> NDArray[np.float64]:
        """Return the ``(6, 6)`` state-transition matrix for a step ``dt``."""
        ...

    def process_noise(self, dt: float) -> NDArray[np.float64]:
        """Return the ``(6, 6)`` process-noise covariance for a step ``dt``."""
        ...


class MeasurementModel(Protocol):
    """Swappable position measurement model (NFR-001)."""

    def observe_matrix(self) -> NDArray[np.float64]:
        """Return the ``(2, 6)`` matrix mapping the state to the position."""
        ...

    def noise(self, conf: float) -> NDArray[np.float64]:
        """Return the ``(2, 2)`` measurement covariance at confidence ``conf``."""
        ...


@dataclass(frozen=True)
class _ConstantAcceleration:
    """Constant-acceleration model driven by white jerk noise."""

    jerk_std_pxps3: float

    @property
    def dim(self) -> int:
        return _STATE_DIM

    def transition(self, dt: float) -> NDArray[np.float64]:
        transition = np.eye(_STATE_DIM, dtype=np.float64)
        half_dt2 = 0.5 * dt * dt
        transition[0, 2] = dt
        transition[0, 4] = half_dt2
        transition[1, 3] = dt
        transition[1, 5] = half_dt2
        transition[2, 4] = dt
        transition[3, 5] = dt
        return transition

    def process_noise(self, dt: float) -> NDArray[np.float64]:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        block = np.array(
            [
                [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
                [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
                [dt3 / 6.0, dt2 / 2.0, dt],
            ],
            dtype=np.float64,
        )
        block *= self.jerk_std_pxps3 * self.jerk_std_pxps3
        covariance = np.zeros((_STATE_DIM, _STATE_DIM), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                covariance[2 * row, 2 * column] = block[row, column]
                covariance[2 * row + 1, 2 * column + 1] = block[row, column]
        return covariance


@dataclass(frozen=True)
class _PositionMeasurement:
    """Position measurement whose variance grows as ``1 / conf``."""

    std_px: float

    def observe_matrix(self) -> NDArray[np.float64]:
        matrix = np.zeros((2, _STATE_DIM), dtype=np.float64)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        return matrix

    def noise(self, conf: float) -> NDArray[np.float64]:
        confidence = min(max(float(conf), _MIN_CONF), _MAX_CONF)
        variance = self.std_px * self.std_px / confidence
        return np.diag(np.array([variance, variance], dtype=np.float64))


@dataclass(frozen=True)
class Tracker2DConfig:
    """Filter, gating and dropout settings.

    ``subject`` and ``skeleton`` hold the nested framing, face-width and
    bone-reference settings. ``min_conf`` is the lowest detection confidence
    accepted as a measurement. ``gate_chi2`` is the two-degree-of-freedom
    chi-square gate. ``max_predict_frames`` is the number of consecutive
    missing frames a keypoint is predicted before it becomes ``LOST``.
    ``meas_std_px`` is the position noise at full confidence and
    ``jerk_std_pxps3`` the white-jerk density. ``init_*_std`` set the initial
    uncertainty of a newly initialized keypoint.
    """

    subject: SubjectConfig = field(default_factory=SubjectConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    min_conf: float = 0.3
    gate_chi2: float = 9.21
    max_predict_frames: int = 5
    meas_std_px: float = 3.0
    jerk_std_pxps3: float = 1000.0
    init_pos_std_px: float = 20.0
    init_vel_std_pxps: float = 1000.0
    init_acc_std_pxps2: float = 10000.0

    def __post_init__(self) -> None:
        if not 0.0 < self.min_conf <= 1.0:
            raise ValueError("min_conf must be in (0, 1]")
        if not self.gate_chi2 > 0.0:
            raise ValueError("gate_chi2 must be positive")
        if self.max_predict_frames < 0:
            raise ValueError("max_predict_frames must not be negative")
        for name in (
            "meas_std_px",
            "jerk_std_pxps3",
            "init_pos_std_px",
            "init_vel_std_pxps",
            "init_acc_std_pxps2",
        ):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class _FrameRecord:
    """One forward filter step, kept for the RTS smoother."""

    seq: int
    t: float
    framing: FramingState
    box: NDArray[np.float64] | None
    face_width_px: float | None
    status: NDArray[np.int8]
    conf: NDArray[np.float64]
    initialized: NDArray[np.bool_]
    segment: NDArray[np.int64]
    x_pred: NDArray[np.float64]
    p_pred: NDArray[np.float64]
    x_filt: NDArray[np.float64]
    p_filt: NDArray[np.float64]
    dt: float


class KeypointTracker2D:
    """Track one subject's keypoints through an episode.

    ``motion`` and ``measurement`` default to a constant-acceleration model with
    white jerk and a position measurement whose variance scales with ``1/conf``.
    Both may be replaced by any object implementing the protocol.
    """

    def __init__(
        self,
        cfg: Tracker2DConfig,
        layout: KeypointLayout,
        motion: MotionModel | None = None,
        measurement: MeasurementModel | None = None,
    ) -> None:
        self._cfg = cfg
        self._layout = layout
        self._size = len(layout)
        self._motion: MotionModel = (
            motion if motion is not None else _ConstantAcceleration(cfg.jerk_std_pxps3)
        )
        if self._motion.dim != _STATE_DIM:
            raise ValueError(f"motion model dim must be {_STATE_DIM}, got {self._motion.dim}")
        self._measurement: MeasurementModel = (
            measurement if measurement is not None else _PositionMeasurement(cfg.meas_std_px)
        )
        matrix = np.asarray(self._measurement.observe_matrix(), dtype=np.float64)
        if matrix.shape != (2, _STATE_DIM):
            raise ValueError(
                f"observe_matrix must have shape (2, {_STATE_DIM}), got {matrix.shape}"
            )
        self._measurement_matrix = matrix
        self._identity = np.eye(_STATE_DIM, dtype=np.float64)

        self.skeleton = SkeletonModel(cfg.skeleton, layout)
        self._face_width = _FaceWidthTracker(cfg.subject)
        self._initial_covariance = self._make_initial_covariance()
        self._x = np.zeros((self._size, _STATE_DIM), dtype=np.float64)
        self._p = np.tile(self._initial_covariance, (self._size, 1, 1))
        self._initialized = np.zeros(self._size, dtype=np.bool_)
        self._segment = np.zeros(self._size, dtype=np.int64)
        self._missed = np.zeros(self._size, dtype=np.int64)
        self._conf = np.zeros(self._size, dtype=np.float64)
        self._last_t: float | None = None
        self._history: list[_FrameRecord] = []

    def _make_initial_covariance(self) -> NDArray[np.float64]:
        cfg = self._cfg
        variances = np.array(
            [
                cfg.init_pos_std_px**2,
                cfg.init_pos_std_px**2,
                cfg.init_vel_std_pxps**2,
                cfg.init_vel_std_pxps**2,
                cfg.init_acc_std_pxps2**2,
                cfg.init_acc_std_pxps2**2,
            ],
            dtype=np.float64,
        )
        return np.diag(variances)

    # ------------------------------------------------------------------
    # Session and episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Start a new episode: clear the filters and the smoother history.

        Skeleton references are session-long (FR-009), so they survive this
        call. Use :meth:`reset_session` to clear them as well.
        """
        self._x.fill(0.0)
        self._p[...] = self._initial_covariance
        self._initialized.fill(False)
        self._segment.fill(0)
        self._missed.fill(0)
        self._conf.fill(0.0)
        self._last_t = None
        self._history.clear()
        self._face_width.reset()

    def reset_session(self) -> None:
        """Start a new session: clear the filters, the history and the skeleton."""
        self.reset()
        self.skeleton.reset()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, frame: PoseFrame) -> TrackedFrame:
        """Filter one pose frame and return the tracked subject."""
        person = select_subject(frame.persons, frame.width, frame.height, self._cfg.subject)
        if person is None:
            face_width = self._face_width.value
            framing = FramingState.NO_PERSON
            box: NDArray[np.float64] | None = None
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

    def _validate_person(
        self, person: PersonPose
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        xy = np.asarray(person.xy, dtype=np.float64)
        conf = np.asarray(person.conf, dtype=np.float64)
        if xy.shape != (self._size, 2):
            raise ValueError(f"person xy must have shape ({self._size}, 2), got {xy.shape}")
        if conf.shape != (self._size,):
            raise ValueError(f"person conf must have shape ({self._size},), got {conf.shape}")
        return xy, conf

    def _time_step(self, t: float) -> float:
        if self._last_t is None:
            return 0.0
        dt = float(t) - self._last_t
        if not np.isfinite(dt) or dt < 0.0:
            return 0.0
        return dt

    def _initialize(
        self,
        x_filt: NDArray[np.float64],
        p_filt: NDArray[np.float64],
        keypoint: int,
        position: NDArray[np.float64],
    ) -> None:
        x_filt[keypoint] = 0.0
        x_filt[keypoint, 0:2] = position
        p_filt[keypoint] = self._initial_covariance
        self._initialized[keypoint] = True
        self._segment[keypoint] += 1
        self._missed[keypoint] = 0

    def _measurement_update(
        self,
        x_pred: NDArray[np.float64],
        p_pred: NDArray[np.float64],
        position: NDArray[np.float64],
        conf: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
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

    def _advance_dropout(self, keypoint: int, status: NDArray[np.int8]) -> None:
        self._missed[keypoint] += 1
        if self._missed[keypoint] > self._cfg.max_predict_frames:
            status[keypoint] = int(KeypointStatus.LOST)
        else:
            status[keypoint] = int(KeypointStatus.PREDICTED)

    # ------------------------------------------------------------------
    # Frame construction and smoothing
    # ------------------------------------------------------------------

    def _build_frame(
        self,
        seq: int,
        t: float,
        framing: FramingState,
        box: NDArray[np.float64] | None,
        x: NDArray[np.float64],
        p: NDArray[np.float64],
        status: NDArray[np.int8],
        conf: NDArray[np.float64],
        face_width_px: float | None,
    ) -> TrackedFrame:
        diagonals = np.einsum("kii->ki", p)
        std = np.sqrt(np.maximum(diagonals, _TINY_VARIANCE))
        valid = status != int(KeypointStatus.LOST)
        points = np.where(valid[:, None], x[:, 0:2], np.nan)
        return TrackedFrame(
            seq=seq,
            t=float(t),
            framing=framing,
            box=None if box is None else box.copy(),
            xy=np.array(x[:, _POSITION], dtype=np.float64, copy=True),
            vel=np.array(x[:, _VELOCITY], dtype=np.float64, copy=True),
            acc=np.array(x[:, _ACCELERATION], dtype=np.float64, copy=True),
            xy_std=np.array(std[:, _POSITION], dtype=np.float64, copy=True),
            vel_std=np.array(std[:, _VELOCITY], dtype=np.float64, copy=True),
            acc_std=np.array(std[:, _ACCELERATION], dtype=np.float64, copy=True),
            status=np.asarray(status, dtype=np.int8),
            conf=np.array(conf, dtype=np.float64, copy=True),
            face_width_px=face_width_px,
            foreshortening=self.skeleton.foreshortening(points),
        )

    def smoothed(self) -> list[TrackedFrame]:
        """Return RTS-smoothed frames over the history since the last reset.

        The forward pass is causal; this pass uses the whole episode for the
        stage-2 report. A model that cannot expose linear ``F``/``Q`` (it sets
        ``smoothing = False``) gets the filtered frames instead.
        """
        if not self._history:
            return []
        if not bool(getattr(self._motion, "smoothing", True)):
            return [self._record_frame(record) for record in self._history]
        try:
            x_smooth, p_smooth = self._rts_states()
        except (np.linalg.LinAlgError, NotImplementedError, ValueError):
            return [self._record_frame(record) for record in self._history]
        return [
            self._record_frame(record, x_smooth[index], p_smooth[index])
            for index, record in enumerate(self._history)
        ]

    def _record_frame(
        self,
        record: _FrameRecord,
        x: NDArray[np.float64] | None = None,
        p: NDArray[np.float64] | None = None,
    ) -> TrackedFrame:
        return self._build_frame(
            record.seq,
            record.t,
            record.framing,
            record.box,
            record.x_filt if x is None else x,
            record.p_filt if p is None else p,
            record.status,
            record.conf,
            record.face_width_px,
        )

    def _rts_states(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        records = self._history
        count = len(records)
        x_smooth = np.array([record.x_filt for record in records], dtype=np.float64)
        p_smooth = np.array([record.p_filt for record in records], dtype=np.float64)
        lost = int(KeypointStatus.LOST)
        for keypoint in range(self._size):
            start = 0
            while start < count:
                if (
                    not records[start].initialized[keypoint]
                    or records[start].status[keypoint] == lost
                ):
                    start += 1
                    continue
                end = start
                while (
                    end + 1 < count
                    and records[end + 1].initialized[keypoint]
                    and records[end + 1].status[keypoint] != lost
                    and records[end + 1].segment[keypoint] == records[start].segment[keypoint]
                ):
                    end += 1
                for index in range(end - 1, start - 1, -1):
                    dt = records[index + 1].dt
                    if dt <= 0.0:
                        continue
                    transition = np.asarray(self._motion.transition(dt), dtype=np.float64)
                    predicted = records[index + 1].p_pred[keypoint]
                    gain = p_smooth[index, keypoint] @ transition.T @ np.linalg.pinv(predicted)
                    x_smooth[index, keypoint] = x_smooth[index, keypoint] + gain @ (
                        x_smooth[index + 1, keypoint] - records[index + 1].x_pred[keypoint]
                    )
                    p_smooth[index, keypoint] = (
                        p_smooth[index, keypoint]
                        + gain @ (p_smooth[index + 1, keypoint] - predicted) @ gain.T
                    )
                    p_smooth[index, keypoint] = 0.5 * (
                        p_smooth[index, keypoint] + p_smooth[index, keypoint].T
                    )
                start = end + 1
        return x_smooth, p_smooth
