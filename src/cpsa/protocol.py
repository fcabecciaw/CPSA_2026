"""Protocol v1 messages of the showcase (F-K1).

Pure builders: every function returns the ``data`` object (or, for
:func:`envelope`, the whole message) as plain JSON-ready values. The server
stamps the envelope per connection with its own ``seq``, the monotonic ``t`` and
the wall-clock ``wall``, so nothing here reads a clock or a socket.

The frozen set of server messages is ``hello``, ``status``,
``showcase.state``, ``pose.state``, ``error`` and ``pong``; the client commands
are ``ping`` and ``model.set``. A missing or not-applicable value is ``null``,
never an absent key, so the dashboard never has to guess whether a field is
``0`` or unknown.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

__all__ = [
    "ACTIVE_MODELS",
    "KEYPOINTS",
    "MAX_OBJECTS",
    "NUM_KEYPOINTS",
    "PROTOCOL_VERSION",
    "ShowcaseObject",
    "envelope",
    "error",
    "hello",
    "pong",
    "pose_state",
    "showcase_state",
    "status",
]

PROTOCOL_VERSION = 1
"""Version of the envelope, the messages and the JSON Schemas in ``docs/protocol/v1``."""

MAX_OBJECTS = 20
"""``showcase.state.objects`` holds at most this many detections, best first."""

ACTIVE_MODELS = ("yolox", "movenet")
"""Values of ``status.active_model`` and of the ``model.set`` command (R2-G)."""

KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
"""COCO-17 order of ``pose.state.keypoints`` (F-E1)."""

NUM_KEYPOINTS = len(KEYPOINTS)

_SCORE_DIGITS = 3
"""Scores keep three decimals: ``0.913`` is already more than the label shows."""

_PIXEL_DIGITS = 1
"""Pixel coordinates are rounded to a tenth of a pixel."""

_TIMING_DIGITS = 1
"""Milliseconds are rounded to a tenth."""


class ShowcaseObject(NamedTuple):
    """One detection as the wire carries it: label, COCO class and source box."""

    label: str
    class_id: int
    score: float
    box: tuple[float, float, float, float]
    """``x1, y1, x2, y2`` in source-image pixels."""


def envelope(
    type: str,
    data: Mapping[str, Any],
    seq: int,
    t: float,
    wall: str,
) -> dict[str, Any]:
    """Wrap ``data`` in the protocol v1 envelope (F-K1)."""
    return {
        "v": PROTOCOL_VERSION,
        "type": type,
        "seq": int(seq),
        "t": float(t),
        "wall": wall,
        "data": dict(data),
    }


def hello(server_version: str) -> dict[str, Any]:
    """The first message of every connection (F-K1)."""
    return {
        "server_version": str(server_version),
        "protocol_version": PROTOCOL_VERSION,
    }


def status(
    *,
    state: str,
    camera: str,
    model: str,
    fps: float | None,
    dpu_ms: float | None,
    errors: Sequence[str],
    uptime_s: float,
    active_model: str = ACTIVE_MODELS[0],
) -> dict[str, Any]:
    """The process snapshot, published at 1 Hz and on change (F-K1).

    ``state`` is ``starting``, ``running`` or ``error``; ``camera`` is ``ok``
    while frames arrive, ``missing`` when the node is not there and ``error``
    when it is but cannot be used. ``errors`` holds the codes of the conditions
    the process reports: ``model_missing``, ``dpu_error``, ``camera_missing``,
    ``camera_error`` or ``internal``. ``active_model`` is the model whose
    overlay the dashboard shows (R2-G): both run, the switch only picks the view.
    """
    return {
        "state": str(state),
        "camera": str(camera),
        "model": str(model),
        "active_model": str(active_model),
        "fps": _number(fps),
        "dpu_ms": _number(dpu_ms),
        "errors": [str(code) for code in errors],
        "uptime_s": max(0.0, _finite(uptime_s, 0.0)),
    }


def showcase_state(
    model: str,
    width: int,
    height: int,
    objects: Sequence[ShowcaseObject],
    *,
    fps: float | None = None,
    dpu_ms: float | None = None,
    total_ms: float | None = None,
) -> dict[str, Any]:
    """Build the ``showcase.state`` payload for one analyzed frame (F-S2, F-K1).

    Args:
        model: Name of the model that produced ``objects``.
        width, height: Size of the frame the boxes refer to, in pixels.
        objects: Every detection of the frame, in any order; the payload ranks
            them by score and counts them by label.
        fps: Processed frames per second, or ``None`` when not measured yet.
        dpu_ms: Last inference time on the DPU, or ``None``.
        total_ms: Last whole-chain time (preprocess, inference, decode, draw).

    Returns:
        The ``data`` object of a ``showcase.state`` message: ``objects`` holds
        the best :data:`MAX_OBJECTS` detections by descending score, ``counts``
        one entry per label by descending count and then label.
    """

    ranked = sorted(objects, key=lambda item: (-_score(item.score), item.class_id, item.label))
    counts = Counter(item.label for item in objects)
    return {
        "model": str(model),
        "width": int(width),
        "height": int(height),
        "fps": _number(fps),
        "dpu_ms": _number(dpu_ms),
        "total_ms": _number(total_ms),
        "objects": [
            {
                "label": str(item.label),
                "class_id": int(item.class_id),
                "score": round(_score(item.score), _SCORE_DIGITS),
                "box": [round(_pixel(value), _PIXEL_DIGITS) for value in item.box],
            }
            for item in ranked[:MAX_OBJECTS]
        ],
        "counts": [
            {"label": label, "count": count}
            for label, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ],
    }


def pose_state(
    model: str,
    width: int,
    height: int,
    keypoints: Sequence[Sequence[float]],
    *,
    score: float,
    fps: float | None = None,
    dpu_ms: float | None = None,
    total_ms: float | None = None,
) -> dict[str, Any]:
    """Build the ``pose.state`` payload of one MoveNet frame (F-E1, R2-G).

    Args:
        model: Name of the model that produced the pose (``movenet``).
        width, height: Size of the frame the keypoints refer to, in pixels.
        keypoints: :data:`NUM_KEYPOINTS` triples ``(x, y, score)`` in COCO-17
            order, in source-image pixels.
        score: Mean keypoint score; below ``0.2`` when nobody is in the frame.
        fps: Processed frames per second, or ``None`` when not measured yet.
        dpu_ms: Last MoveNet inference on the DPU, or ``None``.
        total_ms: Last whole-chain time of the frame, or ``None``.

    Returns:
        The ``data`` object of a ``pose.state`` message.

    Raises:
        ValueError: If ``keypoints`` does not hold :data:`NUM_KEYPOINTS` triples.
    """

    if len(keypoints) != NUM_KEYPOINTS:
        raise ValueError(f"keypoints must hold {NUM_KEYPOINTS} entries, got {len(keypoints)}")
    return {
        "model": str(model),
        "width": int(width),
        "height": int(height),
        "fps": _number(fps),
        "dpu_ms": _number(dpu_ms),
        "total_ms": _number(total_ms),
        "score": round(_score(score), _SCORE_DIGITS),
        "keypoints": [
            [
                round(_pixel(float(point[0])), _PIXEL_DIGITS),
                round(_pixel(float(point[1])), _PIXEL_DIGITS),
                round(_score(float(point[2])), _SCORE_DIGITS),
            ]
            for point in keypoints
        ],
    }


def error(code: str, message: str, ref_seq: int | None = None) -> dict[str, Any]:
    """An ``error`` event; ``ref_seq`` identifies the command it answers."""
    data: dict[str, Any] = {"code": str(code), "message": str(message)}
    if ref_seq is not None:
        data["ref_seq"] = int(ref_seq)
    return data


def pong(message_id: Any) -> dict[str, Any]:
    """The answer to ``ping``; ``id`` is echoed unchanged."""
    return {"id": message_id}


def _score(value: float) -> float:
    """Return a finite score clamped to ``0..1``; a non-finite one counts as zero."""

    return min(1.0, max(0.0, _finite(value, 0.0)))


def _pixel(value: float) -> float:
    """Return a finite pixel coordinate; a non-finite one is reported at the origin."""

    return _finite(value, 0.0)


def _number(value: float | None) -> float | None:
    """Return ``value`` as a JSON number rounded to one decimal, or ``None``."""

    if value is None:
        return None
    number = _finite(value, math.nan)
    return None if math.isnan(number) else round(number, _TIMING_DIGITS)


def _finite(value: object, fallback: float) -> float:
    """Return ``value`` as a finite float, or ``fallback``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    number = float(value)
    return number if math.isfinite(number) else fallback
