"""Replay the protocol v1 examples as a ``medical`` mode (no hardware needed).

``cpsa-supervisor --fake-medical`` uses this mode so the dashboard can be built
and tested on a PC. It publishes the example payloads on a loop: ``medical.state``
at 1 Hz, ``medical.hands`` at 10 Hz with a small synthetic motion, and a
``medical.decision`` plus ``medical.episode`` event every 5 s.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cpsa.supervisor.contract import ModeContext, ModeStartError

logger = logging.getLogger("cpsa.fake")

STATE_PERIOD_S = 1.0
HANDS_PERIOD_S = 0.1
EVENT_PERIOD_S = 5.0
MOTION_PX = 8.0


class FakeMedicalMode:
    """A :class:`cpsa.supervisor.contract.Mode` that replays recorded messages."""

    name = "medical"

    def __init__(self, examples_dir: Path) -> None:
        self._examples_dir = examples_dir
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._context: ModeContext | None = None
        self._state: Mapping[str, Any] = {}
        self._hands: Mapping[str, Any] = {}
        self._decision: Mapping[str, Any] = {}
        self._episode: Mapping[str, Any] = {}

    def start(self, ctx: ModeContext) -> None:
        """Load the examples and start replaying them on a worker thread."""
        self._context = ctx
        self._stop.clear()
        self._state = self._load("medical.state")
        self._hands = self._load("medical.hands")
        self._decision = self._load("medical.decision")
        self._episode = self._load("medical.episode")
        ctx.publisher.state("medical.state", self._state)
        ctx.publisher.state("medical.hands", self._hands)
        ctx.publisher.event("medical.decision", self._decision)
        ctx.publisher.event("medical.episode", self._episode)
        self._thread = threading.Thread(target=self._replay, name="cpsa-fake-medical", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float) -> bool:
        """Stop replaying; ``True`` when the thread ended within ``timeout_s``."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        stopped = thread is None or not thread.is_alive()
        if not stopped:
            logger.error("Fake medical mode did not stop within %.1f s", timeout_s)
        self._thread = None
        return stopped

    def health(self) -> Mapping[str, Any]:
        """A fixed healthy status so the dashboard shows plausible values."""
        return {
            "imu": "connected",
            "camera": "ok",
            "model": "fake",
            "fps": round(1.0 / HANDS_PERIOD_S, 1),
            "dpu_ms": None,
            "errors": [],
        }

    def _load(self, name: str) -> Mapping[str, Any]:
        path = self._examples_dir / f"{name}.json"
        try:
            message = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModeStartError("asset_invalid", f"cannot read {path}: {exc}") from exc
        data = message.get("data") if isinstance(message, dict) else None
        if not isinstance(data, dict):
            raise ModeStartError("asset_invalid", f"{path} has no data object")
        return data

    def _replay(self) -> None:
        start = time.monotonic()
        next_state = start
        next_hands = start
        next_event = start + EVENT_PERIOD_S
        hands_tick = 0
        while not self._stop.is_set():
            context = self._context
            if context is None or context.stop_event.is_set():
                return
            now = time.monotonic()
            if now >= next_state:
                context.publisher.state("medical.state", self._state)
                next_state += STATE_PERIOD_S
            if now >= next_hands:
                context.publisher.state("medical.hands", _animated(self._hands, hands_tick))
                hands_tick += 1
                next_hands += HANDS_PERIOD_S
            if now >= next_event:
                context.publisher.event("medical.decision", self._decision)
                context.publisher.event("medical.episode", self._episode)
                next_event += EVENT_PERIOD_S
            self._stop.wait(max(0.0, min(next_state, next_hands, next_event) - time.monotonic()))


def _animated(hands: Mapping[str, Any], tick: int) -> dict[str, Any]:
    """Return the hands payload with a small sinusoidal motion on every keypoint."""
    keypoints = hands.get("keypoints")
    if not isinstance(keypoints, list):
        return dict(hands)
    phase = 2.0 * math.pi * tick / 50.0
    moved = []
    for index, point in enumerate(keypoints):
        if not isinstance(point, list) or len(point) < 3:
            moved.append(point)
            continue
        x, y, conf = point[0], point[1], point[2]
        moved.append(
            [
                float(x) + MOTION_PX * math.sin(phase + index * 0.1),
                float(y) + MOTION_PX * math.cos(phase + index * 0.1),
                conf,
            ]
        )
    payload = dict(hands)
    payload["keypoints"] = moved
    return payload
