# BlueCoin feature listeners: forward accelerometer, gyroscope and quaternion
# samples to the runtime's sample callback.
#
# Author: Francesco Urru
# Github: https://github.com/frarvo
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

import logging
import time
from collections.abc import Sequence

from blue_st_sdk.feature import FeatureListener

from cpsa.domain.model import SampleKind, Side
from cpsa.ports import SampleCallback

logger = logging.getLogger(__name__)

_LENGTH = {SampleKind.ACC: 3, SampleKind.GYR: 3, SampleKind.QUAT: 4}


def _to_floats(data: Sequence, n: int) -> tuple[float, ...] | None:
    """Convert the first n items of data to floats."""
    if not data or len(data) < n:
        return None
    try:
        return tuple(float(data[i]) for i in range(n))
    except (TypeError, ValueError):
        return None


class SampleListener(FeatureListener):
    """Forward one feature's samples of one wrist, stamped with ``time.monotonic()``."""

    def __init__(self, side: Side, kind: SampleKind, on_sample: SampleCallback):
        super().__init__()
        self.side = side
        self.kind = kind
        self._on_sample = on_sample

    def on_update(self, feature, sample):
        try:
            values = _to_floats(sample.get_data(), _LENGTH[self.kind])
            if not values:
                return
            self._on_sample(self.side, self.kind, values, time.monotonic())
        except Exception:
            logger.exception("%s %s listener failed", self.side.value, self.kind.value)
