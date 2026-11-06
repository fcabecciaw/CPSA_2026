"""Turn preprocessed IMU windows into detections."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from cpsa.application.slot import LatestSlot
from cpsa.domain.model import Detection
from cpsa.ports import Classifier, Clock, FeatureExtractor

logger = logging.getLogger(__name__)


class ClassificationStage:
    """Extract features, classify, and publish a :class:`Detection` per window.

    Warm-up reproduces the baseline native call sequence: the first window is
    dropped without reaching the extractor, and the first successfully extracted
    window only calibrates the extractor's internal state (its features are
    discarded). Every later window is classified.

    Not thread-safe: windows must be delivered sequentially (the ``Windower`` does).
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        classifier: Classifier,
        slot: LatestSlot[Detection],
        clock: Clock,
    ) -> None:
        self._extractor = extractor
        self._classifier = classifier
        self._slot = slot
        self._clock = clock
        self._windows = 0
        self._calibrated = False

    def on_window(self, window: NDArray[np.float32], end_ts: float) -> None:
        self._windows += 1
        number = self._windows
        if number == 1:
            logger.info("Window #1 skipped for sensor stabilization")
            return

        try:
            features = self._extractor.extract(window)
        except Exception:
            logger.exception("Feature extraction failed for window #%d; window skipped", number)
            return

        if not self._calibrated:
            self._calibrated = True
            logger.info("Window #%d calibrated the feature extractor (features discarded)", number)
            return

        try:
            tag = self._classifier.classify(features)
        except Exception:
            logger.exception("Classification failed for window #%d; window skipped", number)
            return

        logger.info("Window #%d classified as %s", number, tag.name)
        self._slot.put(
            Detection(
                tag=tag,
                timestamp=self._clock.now(),
                window_end_ts=end_ts,
                features=tuple(float(value) for value in features),
            )
        )
