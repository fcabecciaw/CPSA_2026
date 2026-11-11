"""The ``medical`` supervisor mode: the existing board runtime behind the Publisher port.

IF-704: the mode wraps ``build_runtime`` and the dispatcher without changing their
behavior; its ``Gui`` (and the verifier's ``VisionObserver``) publishes IF-451..454
through :class:`~cpsa.adapters.gui_web.WebGui`. The perception service that will
own the camera in a later slice is deferred (plan D4), so the mode still builds the
same adapters as ``cpsa-board``. With ``medical.trigger = vision`` it wraps
``build_vision_runtime`` instead: the vision trigger replaces the BlueCoin
detection, and no IMU is scanned or connected (FR-026, FR-027).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from cpsa.adapters.clock import SystemClock
from cpsa.adapters.gui_web import WebGui
from cpsa.adapters.native.classifier import NativeLibraryError
from cpsa.application.config import Config
from cpsa.application.runtime import Runtime, build_runtime, build_vision_runtime
from cpsa.application.vision_verifier import VisionVerifier
from cpsa.board.__main__ import (
    BoardAdapters,
    VisionStartupError,
    board_adapters,
    discover_bluecoins,
)
from cpsa.domain.model import Detection
from cpsa.ports import Actuator, Clock, Verifier
from cpsa.protocol.medical import ImuStatus
from cpsa.protocol.medical import state as medical_state
from cpsa.supervisor.contract import ModeContext, ModeStartError
from cpsa.vision.layouts import get_layout
from cpsa.vision.trigger import VisionTrigger

logger = logging.getLogger("cpsa.board.medical")

#: IF-451: ``medical.state`` is published at 1 Hz (and on change, from the GUI).
STATE_PERIOD_S = 1.0

_CAMERA_CODES = frozenset({"camera_missing", "camera_error"})


class MedicalMode:
    """Wrap the medical runtime as a supervisor :class:`~cpsa.supervisor.contract.Mode`."""

    name = "medical"

    def __init__(self, config: Config) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._runtime: Runtime | None = None
        self._verifier: Verifier | None = None
        self._clock: Clock | None = None
        self._state_stop = threading.Event()
        self._state_thread: threading.Thread | None = None
        self._running = False
        self._imu_state: ImuStatus = "n/a"
        self._camera_error: str | None = None
        self._errors: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self, ctx: ModeContext) -> None:
        """Discover the wrists, build the runtime and start it; block until running.

        With ``medical.trigger = vision`` the mode is camera-only (FR-026): it
        skips the BlueCoin discovery and the native classifier, wires the
        continuous verifier and the vision trigger, and reports the IMU as
        ``"n/a"``. A failed start releases everything it acquired and raises
        :class:`ModeStartError` with an IF-304 code. ``ctx.stop_event`` is the
        supervisor's cancellation handle: a stop requested before or during the
        BlueCoin discovery, the classifier load, the actuator start or the vision
        start releases whatever was acquired and returns cleanly (F4).
        """
        vision_mode = self._vision_trigger
        # The verifier is built before the runtime that owns the slot, so its
        # callback forwards to the dispatcher feed once F10 returns it.
        dispatch_detection: Callable[[Detection], None] | None = None

        def on_detection(detection: Detection) -> None:
            if dispatch_detection is not None:
                dispatch_detection(detection)

        with self._lock:
            if self._runtime is not None:
                raise ModeStartError("busy", "medical mode is already running")
            self._imu_state = "n/a" if vision_mode else "searching"
            self._camera_error = None
            self._errors = []
        if self._stopped(ctx):
            self._cancelled()
            return
        self._publish(
            ctx,
            medical_state(self._imu_state, None, None, None, trigger=self._config.medical.trigger),
        )

        config = self._config
        clock = SystemClock()
        trigger = VisionTrigger(config.medical.vision_trigger) if vision_mode else None
        adapters: BoardAdapters | None = None
        actuators: Sequence[Actuator] = ()
        runtime: Runtime | None = None
        try:
            layout = get_layout(config.vision.layout)
            web_gui = WebGui(
                ctx.publisher,
                layout,
                config.vision.camera.width,
                config.vision.camera.height,
                clock,
                trigger=config.medical.trigger,
            )
            try:
                adapters = board_adapters(
                    config,
                    gui=web_gui,
                    observer=web_gui,
                    clock=clock,
                    continuous=vision_mode,
                    trigger=trigger,
                    on_detection=on_detection if vision_mode else None,
                )
            except VisionStartupError as exc:
                raise ModeStartError(_vision_start_code(exc), str(exc)) from exc
            if self._stopped(ctx):
                self._cancelled(adapters)
                return
            if vision_mode:
                actuators = adapters.start_actuators()
                if self._stopped(ctx):
                    self._cancelled(adapters, actuators=actuators)
                    return
                runtime, dispatch_detection = build_vision_runtime(
                    config,
                    # build_vision_runtime's contract is a continuous verifier;
                    # the seam keeps the Verifier port type for ``cpsa-board``.
                    verifier=cast(VisionVerifier, adapters.verifier),
                    actuators=actuators,
                    sink=adapters.sink,
                    gui=web_gui,
                    clock=adapters.clock,
                )
            else:
                missing = discover_bluecoins(adapters.source, config.bluecoins, ctx.stop_event)
                if missing is None or ctx.stop_event.is_set():
                    logger.info(
                        "Stop requested during BlueCoin discovery; medical mode not started"
                    )
                    self._cancelled(adapters)
                    return
                if missing:
                    raise ModeStartError(
                        "imu_not_found",
                        "BlueCoin devices not found: " + ", ".join(sorted(missing)),
                    )
                try:
                    extractor, classifier = adapters.load_classifier()
                except NativeLibraryError as exc:
                    raise ModeStartError("internal", str(exc)) from exc
                if self._stopped(ctx):
                    self._cancelled(adapters)
                    return
                actuators = adapters.start_actuators()
                if self._stopped(ctx):
                    self._cancelled(adapters, actuators=actuators)
                    return
                runtime = build_runtime(
                    config,
                    source=adapters.source,
                    extractor=extractor,
                    classifier=classifier,
                    verifier=adapters.verifier,
                    actuators=actuators,
                    sink=adapters.sink,
                    gui=web_gui,
                    clock=adapters.clock,
                )
            try:
                runtime.start()
            except Exception as exc:
                recorded = _verifier_camera_error(adapters.verifier)
                if recorded is not None:
                    raise ModeStartError(_camera_code(recorded), str(recorded)) from exc
                if isinstance(exc, FileNotFoundError):
                    raise ModeStartError("camera_missing", str(exc)) from exc
                raise ModeStartError("internal", str(exc)) from exc

            # The camera opens inside runtime.start(): a stop requested while it
            # blocked must release the runtime and the verifier, not leave them.
            if self._stopped(ctx):
                self._cancelled(adapters, runtime=runtime, actuators=actuators)
                return
            recorded = _verifier_camera_error(adapters.verifier)
            if recorded is not None:
                raise ModeStartError(_camera_code(recorded), str(recorded))
        except ModeStartError as exc:
            self._fail(exc.code)
            self._release(adapters, runtime, actuators)
            raise
        except Exception as exc:
            logger.exception("Medical mode failed to start")
            self._fail("internal")
            self._release(adapters, runtime, actuators)
            raise ModeStartError("internal", str(exc)) from exc

        assert adapters is not None and runtime is not None  # both set on the success path
        if self._stopped(ctx):
            # The supervisor asked to stop while start was finishing; release now
            # rather than leave a running mode nobody owns.
            self._cancelled(adapters, runtime=runtime, actuators=actuators)
            return
        with self._lock:
            self._runtime = runtime
            self._verifier = adapters.verifier
            self._clock = clock
            self._running = True
        self._publish_state(ctx, web_gui, clock, runtime)
        self._state_stop = threading.Event()
        thread = threading.Thread(
            target=self._publish_loop,
            args=(ctx, web_gui, clock),
            name="cpsa-medical-state",
            daemon=True,
        )
        self._state_thread = thread
        thread.start()
        if self._stopped(ctx):
            # The supervisor asked to stop while start was finishing; release now
            # rather than leave a running mode nobody owns.
            self.stop(0.0)
            return
        logger.info("Medical mode started")

    def stop(self, timeout_s: float) -> bool:
        """Stop the state publisher and the runtime; return whether the thread stopped."""
        self._state_stop.set()
        thread = self._state_thread
        released = True
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_s)))
            released = not thread.is_alive()
        with self._lock:
            runtime = self._runtime
            self._runtime = None
            self._state_thread = None
            self._verifier = None
            self._clock = None
            self._running = False
            self._imu_state = "n/a"
            self._camera_error = None
            self._errors = []
        if runtime is not None:
            runtime.stop()
        logger.info("Medical mode stopped")
        return released

    # -- health ------------------------------------------------------------

    def health(self) -> Mapping[str, Any]:
        """Return the F4 health keys: imu, camera, model, fps, dpu_ms, timing, errors.

        With ``medical.trigger = vision`` the mode has no IMU, so ``imu`` is
        always ``"n/a"`` (FR-026, F11). ``fps``, ``dpu_ms`` and ``timing`` come from
        the vision verifier's rolling performance snapshot (F12/F13); they are
        ``None`` when there is no verifier or it has no data yet.
        """
        with self._lock:
            running = self._running
            runtime = self._runtime
            verifier = self._verifier
            clock = self._clock
            stored_imu = self._imu_state
            camera_error = self._camera_error
            errors = list(self._errors)
        imu: ImuStatus
        if self._vision_trigger:
            imu = "n/a"
        elif running and runtime is not None and clock is not None:
            imu = runtime.imu_status(clock.monotonic())
        else:
            imu = stored_imu
        camera: str | None
        if not self._config.vision.enabled:
            camera = None
        elif camera_error is not None:
            camera = camera_error
        else:
            recorded = None if verifier is None else _verifier_camera_error(verifier)
            camera = "ok" if running else None
            if recorded is not None:
                camera = _camera_code(recorded)
        model = None if not self._config.vision.enabled else self._config.vision.model.path.stem
        perf = _verifier_perf(verifier)
        return {
            "imu": imu,
            "camera": camera,
            "model": model,
            "fps": perf.get("analyzed_fps"),
            "dpu_ms": perf.get("dpu_ms"),
            "timing": perf or None,
            "errors": errors,
        }

    # -- state publisher ---------------------------------------------------

    def _publish_loop(self, ctx: ModeContext, web_gui: WebGui, clock: Clock) -> None:
        while not self._state_stop.wait(STATE_PERIOD_S):
            with self._lock:
                runtime = self._runtime
            if runtime is None:
                return
            self._publish_state(ctx, web_gui, clock, runtime)

    def _publish_state(
        self, ctx: ModeContext, web_gui: WebGui, clock: Clock, runtime: Runtime
    ) -> None:
        try:
            imu: ImuStatus = (
                "n/a" if self._vision_trigger else runtime.imu_status(clock.monotonic())
            )
            with self._lock:
                self._imu_state = imu
            self._publish(ctx, web_gui.snapshot_state(imu))
        except Exception:
            logger.exception("Failed to publish medical.state")

    @property
    def _vision_trigger(self) -> bool:
        """Whether the medical mode runs camera-only (FR-026)."""
        return self._config.medical.trigger == "vision"

    @staticmethod
    def _publish(ctx: ModeContext, payload: Mapping[str, Any]) -> None:
        try:
            ctx.publisher.state("medical.state", payload)
        except Exception:
            logger.exception("Publisher rejected medical.state")

    # -- release -----------------------------------------------------------

    def _fail(self, code: str) -> None:
        with self._lock:
            self._errors = [code]
            self._imu_state = "n/a"
            self._camera_error = code if code in _CAMERA_CODES else None

    @staticmethod
    def _stopped(ctx: ModeContext) -> bool:
        """Whether the supervisor asked this start to stop (F4)."""
        return ctx.stop_event.is_set()

    def _cancelled(
        self,
        adapters: BoardAdapters | None = None,
        *,
        runtime: Runtime | None = None,
        actuators: Sequence[Actuator] = (),
    ) -> None:
        """Release a start cancelled through ``ctx.stop_event`` and reset the state."""
        self._release(adapters, runtime, actuators)
        with self._lock:
            self._imu_state = "n/a"
            self._camera_error = None
            self._errors = []
            self._running = False

    @staticmethod
    def _release(
        adapters: BoardAdapters | None,
        runtime: Runtime | None,
        actuators: Sequence[Actuator],
    ) -> None:
        """Release whatever a failed (or cancelled) start acquired."""
        if runtime is not None:
            # Runtime.stop() releases the GUI, dispatcher, sensors, verifier and
            # actuators, so stopping the actuators again is redundant.
            try:
                runtime.stop()
            except Exception:
                logger.exception("Failed to stop the medical runtime")
        else:
            for actuator in actuators:
                try:
                    actuator.stop()
                except Exception:
                    logger.exception("Failed to stop actuator %s", getattr(actuator, "id", "?"))
        if adapters is not None:
            stop = getattr(adapters.verifier, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    logger.exception("Failed to stop the vision verifier")


def _verifier_perf(verifier: Verifier | None) -> dict[str, Any]:
    """Return the verifier's rolling performance snapshot, or ``{}`` (F12).

    ``Verifier`` (F1) has no performance member, so the snapshot is read from the
    public ``perf_snapshot`` that ``VisionVerifier`` exposes. Implementations without
    it (the stub, test doubles) report no metrics.
    """
    if verifier is None:
        return {}
    snapshot = getattr(verifier, "perf_snapshot", None)
    if not callable(snapshot):
        return {}
    try:
        perf = snapshot()
    except Exception:
        logger.exception("Vision verifier performance snapshot failed")
        return {}
    return dict(perf) if isinstance(perf, Mapping) else {}


def _verifier_camera_error(verifier: Verifier) -> BaseException | None:
    """Return the camera error the verifier recorded for its fail-open gate, if any.

    ``Verifier`` (F1) does not have a health member, so the signal is read from
    the public read-only ``health_error`` that ``VisionVerifier`` exposes (R-05).
    Implementations without it (the stub, test doubles) simply report no error.
    """
    error = getattr(verifier, "health_error", None)
    return error if isinstance(error, BaseException) else None


def _camera_code(error: BaseException) -> str:
    """Map a video-source failure to the IF-304 code (a missing device node vs. the rest)."""
    if isinstance(error, FileNotFoundError):
        return "camera_missing"
    return "camera_error"


def _vision_start_code(exc: VisionStartupError) -> str:
    """Map a vision construction failure to an IF-304 code by its cause (R-04).

    ``_build_verifier`` wraps every construction error in the same
    ``VisionStartupError``, so the cause type is the only origin it preserves:
    a missing model file is ``model_missing``; a ``RuntimeError`` is the pose
    estimator (``xir``/``vart`` or ``onnxruntime`` bindings missing, or the
    graph is unusable), so ``dpu_error``; a ``ValueError`` is the UVC source
    rejecting its stream format, so a ``camera_*`` code. Camera failures that
    happen later, when the source opens, keep the ``camera_*`` codes too.
    """
    cause = exc.__cause__
    if isinstance(cause, FileNotFoundError):
        return "model_missing"
    if isinstance(cause, RuntimeError):
        return "dpu_error"
    if isinstance(cause, ValueError):
        return "camera_error"
    return "internal"
