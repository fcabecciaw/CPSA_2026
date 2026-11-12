"""Drawing the detections on the analysed frame.

The overlay is drawn on the board, on the same frame the detector saw, so the
video and the boxes can never drift apart however fast the browser is. Each box
is a 2 px rectangle in the class colour plus a ``"<label> <score>%"`` tag; the
colours come from a fixed palette indexed by class, so a class keeps its colour
across frames and runs.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from cpsa.protocol import ShowcaseObject

__all__ = ["class_color", "draw_overlay"]

_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1
_LABEL_PADDING = 3
_LABEL_GAP = 4

_PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (245, 130, 48),
    (0, 130, 200),
    (240, 50, 230),
    (0, 128, 128),
    (170, 110, 40),
    (145, 30, 180),
    (70, 240, 240),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 0),
)
"""Box colours in RGB, one per class modulo the table length."""


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return the RGB colour of a class: stable across frames and runs."""
    return _PALETTE[abs(int(class_id)) % len(_PALETTE)]


def draw_overlay(
    rgb: NDArray[np.uint8],
    objects: Sequence[ShowcaseObject],
) -> NDArray[np.uint8]:
    """Return a copy of ``rgb`` with a 2 px box and a ``"<label> <score>%"`` tag.

    The image is RGB (the order the JPEG encoder and the shared frame channel
    expect) and the copy is what the video shows, so the frame being analyzed
    can never change under it.
    """
    overlay = np.array(rgb, copy=True)
    height, width = overlay.shape[:2]
    for item in objects:
        colour = class_color(item.class_id)
        x1, y1, x2, y2 = _clip(item.box, width, height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, _BOX_THICKNESS)
        text = f"{item.label} {int(round(item.score * 100.0))}%"
        (text_width, text_height), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        label_height = text_height + 2 * _LABEL_GAP
        top = y1 - label_height
        if top < 0:
            # No room above the box: put the tag just inside its top edge.
            top = y1 + _LABEL_GAP
        top = max(0, min(top, height - label_height))
        bottom = min(height - 1, top + label_height)
        right = min(width - 1, x1 + text_width + 2 * _LABEL_PADDING)
        cv2.rectangle(overlay, (x1, top), (right, bottom), colour, -1)
        cv2.putText(
            overlay,
            text,
            (x1 + _LABEL_PADDING, bottom - _LABEL_GAP),
            _FONT,
            _FONT_SCALE,
            _text_color(colour),
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return overlay


def _clip(
    box: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Clip a float box to the image and return integer corners inside it."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
        return (0, 0, 0, 0)
    left = min(max(int(round(x1)), 0), max(width - 1, 0))
    top = min(max(int(round(y1)), 0), max(height - 1, 0))
    right = min(max(int(round(x2)), left), max(width - 1, 0))
    bottom = min(max(int(round(y2)), top), max(height - 1, 0))
    return (left, top, right, bottom)


def _text_color(colour: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black on a light box colour, white on a dark one."""
    red, green, blue = colour
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 140.0 else (255, 255, 255)
