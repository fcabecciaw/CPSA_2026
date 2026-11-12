"""Builder for the ``showcase.state`` payload (F-S2).

Pure function: it maps the object detections of one analyzed RGB frame, already
translated to labels and source-image pixels by the mode, to a JSON-ready
``data`` object; the envelope (``v``, ``seq``, ``t``, ``wall``) is added by the
server per connection (IF-102). Conventions from ``docs/protocol/v1/README.md``
apply: a missing or not-applicable value is ``null``, never an absent key.

The order matters to the dashboard, so it is fixed here: ``objects`` is the
best-first display list (at most :data:`MAX_OBJECTS`), ``counts`` covers *every*
object handed in - the label histogram is what the right pane shows large, and
capping it at the display list would under-report a crowded frame.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, NamedTuple

__all__ = ["MAX_OBJECTS", "ShowcaseObject", "showcase_state"]

MAX_OBJECTS = 20
"""``showcase.state.objects`` holds at most this many detections, best first."""

_SCORE_DIGITS = 3
"""Scores keep three decimals: ``0.913`` is already more than the label shows."""

_PIXEL_DIGITS = 1
"""Pixel coordinates are rounded to a tenth of a pixel."""

_TIMING_DIGITS = 1
"""Milliseconds are rounded like the ``status.perception.timing`` fields."""


class ShowcaseObject(NamedTuple):
    """One detection as the wire carries it: label, COCO class and source box."""

    label: str
    class_id: int
    score: float
    box: tuple[float, float, float, float]
    """``x1, y1, x2, y2`` in source-image pixels."""


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
    """Build the ``showcase.state`` payload for one analyzed frame (F-S2).

    Args:
        model: Name of the model that produced ``objects`` (``showcase.default_model``).
        width, height: Size of the frame the boxes refer to, in pixels.
        objects: Every detection of the frame, in any order; the payload ranks
            them by score and counts them by label.
        fps: Processed frames per second, or ``None`` when not measured yet.
        dpu_ms: Last inference time on the DPU, or ``None``.
        total_ms: Last whole-chain time (preprocess, inference, decode, draw), or ``None``.

    Returns:
        The ``data`` object of a ``showcase.state`` message: ``objects`` holds the
        best :data:`MAX_OBJECTS` detections by descending score, ``counts`` one
        entry per label by descending count and then label.
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


def _score(value: float) -> float:
    """Return a finite score clamped to ``0..1``; a non-finite one counts as zero."""

    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _pixel(value: float) -> float:
    """Return a finite pixel coordinate; a non-finite one is reported at the origin."""

    number = float(value)
    return number if math.isfinite(number) else 0.0


def _number(value: float | None) -> float | None:
    """Return ``value`` as a JSON number rounded to one decimal, or ``None``."""

    if value is None:
        return None
    number = float(value)
    return round(number, _TIMING_DIGITS) if math.isfinite(number) else None
