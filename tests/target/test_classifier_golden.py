"""Classifier equivalence with the baseline at ``fcf3ec9`` (FR-004, AC-003).

Runs only on aarch64 with ``pytest -m target``. Replays the raw rows recorded in
``fixtures/classifier_golden.npz`` through ``Windower`` -> ``ClassificationStage``
with the native adapters and requires features and tags identical to the baseline.
See ``fixtures/README.md`` for the fixture schema and how it was captured.
"""

from __future__ import annotations

import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

FIXTURE = Path(__file__).parent / "fixtures" / "classifier_golden.npz"

pytestmark = [
    pytest.mark.target,
    pytest.mark.skipif(platform.machine() != "aarch64", reason="native libraries are aarch64-only"),
]


def _set_names() -> list[str]:
    if not FIXTURE.exists():
        return []
    with np.load(FIXTURE) as data:
        return sorted({key.rsplit("_", 1)[0] for key in data.files})


def replay(
    rows: NDArray[np.float32], ts: NDArray[np.float64]
) -> tuple[list[list[float]], list[int]]:
    """Run one recorded stream through the production pipeline stages."""
    from cpsa.adapters.native.classifier import NativeClassifier, NativeFeatureExtractor
    from cpsa.application.classification import ClassificationStage
    from cpsa.application.slot import LatestSlot
    from cpsa.domain.model import Detection
    from cpsa.imu.window import Windower
    from tests.fakes import FakeClock

    detections: list[Detection] = []

    class RecordingSlot(LatestSlot[Detection]):
        def put(self, item: Detection) -> None:
            detections.append(item)

    # Same construction order as the baseline: extractor library, then classifier.
    extractor = NativeFeatureExtractor()
    classifier = NativeClassifier()
    stage = ClassificationStage(extractor, classifier, RecordingSlot(), FakeClock())
    windower = Windower(stage.on_window)
    for row, row_ts in zip(rows, ts, strict=True):
        windower.add(np.asarray(row, dtype=np.float32), float(row_ts))
    return [list(d.features) for d in detections], [int(d.tag) for d in detections]


@pytest.mark.parametrize("set_name", _set_names() or ["<missing>"])
def test_features_and_tags_match_baseline(set_name: str) -> None:
    if not FIXTURE.exists():
        pytest.skip("golden fixture not captured yet (PLAN.md Task 0, see fixtures/README.md)")
    with np.load(FIXTURE) as data:
        rows, ts = data[f"{set_name}_rows"], data[f"{set_name}_ts"]
        expected_features, expected_tags = data[f"{set_name}_features"], data[f"{set_name}_tags"]

    # The native extractor keeps process-global state, so each recorded stream is
    # replayed in a fresh interpreter, exactly as it was captured.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        features, tags = pool.submit(replay, rows, ts).result()

    assert np.array_equal(np.asarray(features, dtype=np.float32), expected_features)
    assert np.array_equal(np.asarray(tags), expected_tags)
