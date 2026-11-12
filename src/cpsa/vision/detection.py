"""Object detection contracts shared by the detector adapters (F-S1).

A detector turns one RGB frame into a list of :class:`Detection` in source
pixels.  The types are plain numpy/py-dataclass so the vision layer, the
application layer and the board mode can depend on them without importing a
DPU runtime: only the adapter in :mod:`cpsa.adapters.detect.yolox_dpu` binds
the compiled model.

Coordinates are pixels of the frame passed to :meth:`ObjectDetector.detect`,
``x`` right and ``y`` down, and boxes are clipped to the frame.  Scores are
``0..1`` and ``class_id`` is the model family's native index (COCO-80 for
YOLOX, where ``0`` is ``person``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["Detection", "ObjectDetector"]


@dataclass(frozen=True)
class Detection:
    """One detected object in source-image pixels.

    Attributes:
        box: ``(x1, y1, x2, y2)`` in source pixels, clipped to the image and
            ordered so ``x2 >= x1`` and ``y2 >= y1``.
        score: Objectness times the best class probability, in ``0..1``.
        class_id: Model-native class index; COCO-80 for YOLOX.
    """

    box: tuple[float, float, float, float]
    score: float
    class_id: int


class ObjectDetector(Protocol):
    """A single-frame object detector.

    Implementations are stateful only for timing and buffers: ``detect`` may be
    called once per frame from the board mode's loop.
    """

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input ``(height, width)`` in pixels."""

    @property
    def stage_ms(self) -> dict[str, float]:
        """Per-stage timings of the last call, in milliseconds.

        Keys are ``preprocess``, ``dpu`` and ``decode``.
        """

    def detect(self, rgb: NDArray[np.uint8]) -> list[Detection]:
        """Detect objects in one ``(H, W, 3)`` RGB uint8 frame.

        Returns:
            Detections in score-descending order.
        """
