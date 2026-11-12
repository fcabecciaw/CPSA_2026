"""The ``hologram`` and ``rps`` supervisor modes (AC-08).

Both modes run the same perception chain and differ only in what they publish:
:class:`HologramMode` streams every :class:`~cpsa.vision.hand3d.HandFrame` as
``hologram.hands``; :class:`RpsMode` feeds the frames to
:class:`~cpsa.vision.rps_game.RpsGame` with
:class:`~cpsa.vision.rps_classifier.RpsClassifier` on the largest hand and
publishes ``rps.state``. One :class:`~cpsa.application.hand_perception.HandPerception`
per running mode owns the IR/depth source, so :meth:`stop` always releases the
shared camera before the supervisor starts another mode.

A missing model file fails the start with ``model_missing`` before the camera is
touched; a missing INZI node fails it with ``camera_missing``.

Both modes also feed ``/video.mjpeg`` (F14) with the preprocessed IR image of
every analyzed frame, the way the medical mode feeds its RGB frames: the
dashboard shows the camera on the left and the keypoints on the right.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cpsa.adapters.camera.irdepth import IrDepthSource, ir_to_input
from cpsa.application.config import Config, HandsConfig
from cpsa.application.hand_perception import HandPerception, IrDepthSourceLike
from cpsa.ports import PoseEstimator
from cpsa.protocol.hands import hologram_hands, rps_state
from cpsa.supervisor.contract import ModeContext, ModeStartError
from cpsa.vision.calibration import DepthModel, Intrinsics
from cpsa.vision.hand3d import Hand3D, HandFrame
from cpsa.vision.layouts import HAND21
from cpsa.vision.rps_classifier import RpsClassifier
from cpsa.vision.rps_game import RpsGame, RpsGameConfig, RpsRandom, RpsState

logger = logging.getLogger("cpsa.board.hands")

__all__ = ["HologramMode", "RpsMode"]

IR_WIDTH = 640
"""Width of one INZI IR/depth frame."""

IR_HEIGHT = 480
"""Height of one INZI IR/depth frame."""


COUNTDOWN_PUBLISH_S = 0.1
"""``rps.state`` is republished at 10 Hz during the countdown."""

_CAMERA_CODES = {"camera_missing": "missing", "camera_error": "error"}


class GestureClassifier(Protocol):
    """The part of :class:`~cpsa.vision.rps_classifier.RpsClassifier` the mode uses."""

    def predict(self, kpts2d: NDArray[np.float32]) -> tuple[str, float]: ...


SourceFactory = Callable[[HandsConfig], IrDepthSourceLike]
EstimatorFactory = Callable[[HandsConfig], PoseEstimator]
ClassifierFactory = Callable[[Path], GestureClassifier]


def _default_source(hands: HandsConfig) -> IrDepthSource:
    return IrDepthSource(
        device="auto",
        width=IR_WIDTH,
        height=IR_HEIGHT,
        fps=hands.camera.fps,
        match=hands.camera.match,
    )


def _default_estimator(hands: HandsConfig) -> PoseEstimator:
    from cpsa.adapters.pose.dpu import DpuPoseEstimator

    return DpuPoseEstimator(
        hands.model.path,
        HAND21,
        conf_thr=hands.model.conf_thr,
        iou_thr=hands.model.iou_thr,
    )


def _default_classifier(path: Path) -> GestureClassifier:
    return RpsClassifier.load(path)


def _require_hands(config: Config) -> HandsConfig:
    if config.hands is None:
        raise ValueError("the hand modes need a 'hands' configuration section")
    return config.hands


class _HandMode:
    """Shared lifecycle, health and publishing for the two hand modes."""

    name = ""

    def __init__(
        self,
        config: Config,
        *,
        source_factory: SourceFactory | None = None,
        estimator_factory: EstimatorFactory | None = None,
    ) -> None:
        self._config = config
        self._hands = _require_hands(config)
        self._source_factory: SourceFactory = (
            source_factory if source_factory is not None else _default_source
        )
        self._estimator_factory: EstimatorFactory = (
            estimator_factory if estimator_factory is not None else _default_estimator
        )
        self._lock = threading.Lock()
        self._perception: HandPerception | None = None
        self._ctx: ModeContext | None = None
        self._running = False
        self._camera_error: str | None = None
        self._errors: list[str] = []
        # F14: the preprocessed IR image of the frame being handled, handed to
        # the dashboard's MJPEG stream instead of the RGB camera the medical
        # mode feeds.
        self._model_input: NDArray[np.uint8] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self, ctx: ModeContext) -> None:
        """Open the IR/depth source, start perception and return once it runs.

        Raises :class:`ModeStartError` with ``model_missing``, ``camera_missing``,
        ``camera_error``, ``dpu_error`` or ``internal``; a failed start releases
        whatever it acquired.
        """
        with self._lock:
            if self._running:
                raise ModeStartError("busy", f"{self.name} mode is already running")
            self._camera_error = None
            self._errors = []
            self._model_input = None
        if ctx.stop_event.is_set():
            logger.info("Stop requested before %s mode started", self.name)
            return

        hands = self._hands
        model_path = hands.model.path
        if not model_path.is_file():
            self._fail("model_missing")
            raise ModeStartError("model_missing", f"hand model not found: {model_path}")
        try:
            self._prepare()
        except ModeStartError as exc:
            self._fail(exc.code)
            raise
        except Exception as exc:
            logger.exception("%s mode preparation failed", self.name)
            self._fail("internal")
            raise ModeStartError("internal", str(exc)) from exc

        try:
            estimator = self._estimator_factory(hands)
        except FileNotFoundError as exc:
            self._fail("model_missing")
            raise ModeStartError("model_missing", str(exc)) from exc
        except RuntimeError as exc:
            self._fail("dpu_error")
            raise ModeStartError("dpu_error", str(exc)) from exc
        except Exception as exc:
            logger.exception("%s mode could not build the hand estimator", self.name)
            self._fail("internal")
            raise ModeStartError("internal", str(exc)) from exc

        perception: HandPerception | None = None
        try:
            perception = self._build_perception(estimator)
            perception.start(self._on_frame)
        except FileNotFoundError as exc:
            _release_quietly(perception)
            self._fail("camera_missing")
            raise ModeStartError("camera_missing", str(exc)) from exc
        except Exception as exc:
            _release_quietly(perception)
            self._fail("camera_error")
            raise ModeStartError("camera_error", str(exc)) from exc

        if ctx.stop_event.is_set():
            # A stop during the (possibly blocking) camera open releases it now.
            logger.info("Stop requested while %s mode was starting", self.name)
            perception.stop(0.0)
            return
        with self._lock:
            self._perception = perception
            self._ctx = ctx
            self._running = True
        self._started(ctx)
        logger.info("%s mode started", self.name)

    def stop(self, timeout_s: float) -> bool:
        """Stop perception and release the camera; return whether it stopped in time."""
        with self._lock:
            perception = self._perception
            self._perception = None
            self._ctx = None
            self._running = False
            self._camera_error = None
            self._errors = []
            self._model_input = None
        released = True if perception is None else perception.stop(timeout_s)
        self._release()
        logger.info("%s mode stopped", self.name)
        return released

    # -- health ------------------------------------------------------------

    def health(self) -> Mapping[str, Any]:
        """Return the F4 keys: ``imu``, ``camera``, ``model``, ``fps``, ``dpu_ms``, ``errors``."""
        with self._lock:
            running = self._running
            perception = self._perception
            camera_error = self._camera_error
            errors = list(self._errors)
        camera: str | None
        fps: float | None = None
        dpu_ms: float | None = None
        if camera_error is not None:
            camera = camera_error
        elif running and perception is not None:
            snapshot = perception.health()
            fps = float(snapshot["fps"])
            dpu_ms = float(snapshot["dpu_ms"])
            camera = "ok" if snapshot["camera"] else "error"
        else:
            camera = None
        return {
            "imu": "n/a",
            "camera": camera,
            "model": self._hands.model.path.stem,
            "fps": fps,
            "dpu_ms": dpu_ms,
            "errors": errors,
        }

    # -- hooks -------------------------------------------------------------

    def _prepare(self) -> None:
        """Acquire mode-specific resources before the camera opens."""

    def _started(self, ctx: ModeContext) -> None:
        """Publish the first state of a mode that is now running."""

    def _release(self) -> None:
        """Drop mode-specific resources after perception stopped."""

    def _handle(self, ctx: ModeContext, frame: HandFrame) -> None:
        raise NotImplementedError

    # -- perception --------------------------------------------------------

    def _build_perception(self, estimator: PoseEstimator) -> HandPerception:
        hands = self._hands
        return HandPerception(
            self._source_factory(hands),
            estimator,
            Intrinsics(
                fx=hands.intrinsics.fx,
                fy=hands.intrinsics.fy,
                cx=hands.intrinsics.cx,
                cy=hands.intrinsics.cy,
                width=IR_WIDTH,
                height=IR_HEIGHT,
            ),
            DepthModel(scale_m=hands.depth.scale_m, offset_m=hands.depth.offset_m),
            preprocess=self._preprocess,
            max_fps=hands.camera.fps,
        )

    def _preprocess(self, ir: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Preprocess one IR frame and keep the model input for the video stream.

        Runs on the perception thread, just before the estimator, and the same
        thread calls :meth:`_on_frame` afterwards, so the image offered there is
        the one of the frame being handled. ``ir_to_input`` returns a new array
        every time, so the encoder never sees it change under its hands.
        """

        model_input = ir_to_input(ir)
        with self._lock:
            self._model_input = model_input
        return model_input

    def _on_frame(self, frame: HandFrame) -> None:
        with self._lock:
            ctx = self._ctx
            running = self._running
            model_input = self._model_input
        if ctx is None or not running:
            return
        self._offer_video(ctx, frame.seq, model_input)
        try:
            self._handle(ctx, frame)
        except Exception:
            logger.exception("%s mode failed on frame %d", self.name, frame.seq)

    def _offer_video(
        self,
        ctx: ModeContext,
        seq: int,
        image: NDArray[np.uint8] | None,
    ) -> None:
        """Offer the preprocessed IR frame to ``/video.mjpeg`` (F14).

        The dashboard shows the IR image the model saw next to the keypoints, so
        no overlay is drawn on it. The server ignores frames while the stream is
        disabled or no client is connected; a publisher without live video (the
        fake publisher, a test double) simply has no ``offer_video_frame``.
        """

        if image is None:
            return
        offer = getattr(ctx.publisher, "offer_video_frame", None)
        if not callable(offer):
            return
        try:
            offer(seq, image)
        except Exception:
            logger.exception("Publisher rejected a hand video frame")

    # -- helpers -----------------------------------------------------------

    def _fail(self, code: str) -> None:
        with self._lock:
            self._errors = [code]
            self._camera_error = _CAMERA_CODES.get(code)


class HologramMode(_HandMode):
    """``hologram``: publish the 3D hands of every analyzed IR frame."""

    name = "hologram"

    def _handle(self, ctx: ModeContext, frame: HandFrame) -> None:
        self._publish_state(ctx, hologram_hands(frame))

    def _publish_state(self, ctx: ModeContext, payload: Mapping[str, Any]) -> None:
        try:
            ctx.publisher.state("hologram.hands", payload)
        except Exception:
            logger.exception("Publisher rejected hologram.hands")


class RpsMode(_HandMode):
    """``rps``: play rock/paper/scissors and publish the round state."""

    name = "rps"

    def __init__(
        self,
        config: Config,
        *,
        source_factory: SourceFactory | None = None,
        estimator_factory: EstimatorFactory | None = None,
        classifier_factory: ClassifierFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        rng: RpsRandom | None = None,
    ) -> None:
        super().__init__(config, source_factory=source_factory, estimator_factory=estimator_factory)
        self._classifier_factory: ClassifierFactory = (
            classifier_factory if classifier_factory is not None else _default_classifier
        )
        self._clock = clock
        self._rng = rng
        self._classifier: GestureClassifier | None = None
        self._game: RpsGame | None = None
        self._last_key: tuple[Any, ...] | None = None
        self._last_publish = 0.0

    def _prepare(self) -> None:
        rps = self._hands.rps
        if rps is None:
            raise ModeStartError("model_missing", "no hands.rps.classifier_path configured")
        if not rps.classifier_path.is_file():
            raise ModeStartError(
                "model_missing", f"rps classifier not found: {rps.classifier_path}"
            )
        try:
            self._classifier = self._classifier_factory(rps.classifier_path)
        except FileNotFoundError as exc:
            raise ModeStartError("model_missing", str(exc)) from exc
        except ValueError as exc:
            raise ModeStartError("asset_invalid", str(exc)) from exc
        self._game = RpsGame(
            RpsGameConfig(
                ready_s=rps.ready_s,
                countdown_step_s=rps.countdown_step_s,
                shoot_s=rps.shoot_s,
                result_s=rps.result_s,
                score_reset_s=rps.score_reset_s,
                min_share=rps.min_share,
            ),
            rng=self._rng,
        )
        self._last_key = None
        self._last_publish = 0.0

    def _started(self, ctx: ModeContext) -> None:
        self._publish_if_changed(ctx, self._clock(), force=True)

    def _release(self) -> None:
        self._classifier = None
        self._game = None
        self._last_key = None
        self._last_publish = 0.0

    def _handle(self, ctx: ModeContext, frame: HandFrame) -> None:
        game = self._game
        if game is None:
            return
        now = self._clock()
        state = game.update(now, frame.hands, self._classify)
        self._publish_if_changed(ctx, now, state=state)

    def _classify(self, hand: Hand3D) -> tuple[str, float]:
        classifier = self._classifier
        if classifier is None:
            return ("none", 0.0)
        try:
            return classifier.predict(hand.kpts2d)
        except Exception:
            logger.exception("rps classification failed on one hand")
            return ("none", 0.0)

    def _publish_if_changed(
        self,
        ctx: ModeContext,
        now: float,
        *,
        state: RpsState | None = None,
        force: bool = False,
    ) -> None:
        game = self._game
        if game is None:
            return
        snapshot = state if state is not None else game.state
        key = _state_key(snapshot)
        countdown_tick = (
            snapshot.phase == "countdown" and now - self._last_publish >= COUNTDOWN_PUBLISH_S
        )
        if not force and key == self._last_key and not countdown_tick:
            return
        self._last_key = key
        self._last_publish = now
        payload = rps_state(
            snapshot.phase,
            count=snapshot.count,
            player=snapshot.player,
            player_p=snapshot.player_p,
            board=snapshot.board,
            outcome=snapshot.outcome,
            score={
                "player": snapshot.score.player,
                "board": snapshot.score.board,
                "draws": snapshot.score.draws,
            },
            kpts2d=None if snapshot.hand is None else snapshot.hand.kpts2d,
        )
        try:
            ctx.publisher.state("rps.state", payload)
        except Exception:
            logger.exception("Publisher rejected rps.state")


def _state_key(state: RpsState) -> tuple[Any, ...]:
    """The published fields that make ``rps.state`` change (not the live hand)."""

    return (
        state.phase,
        state.count,
        state.player,
        None if state.player_p is None else round(state.player_p, 4),
        state.board,
        state.outcome,
        state.score.player,
        state.score.board,
        state.score.draws,
    )


def _release_quietly(perception: HandPerception | None) -> None:
    """Release a partially started perception without masking the start error."""

    if perception is None:
        return
    try:
        perception.stop(0.0)
    except Exception:
        logger.exception("Failed to stop hand perception after a failed start")
