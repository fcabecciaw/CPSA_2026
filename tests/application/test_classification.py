import numpy as np
from numpy.typing import NDArray

from cpsa.application.classification import ClassificationStage
from cpsa.application.slot import LatestSlot
from cpsa.domain.model import ClassificationError, Detection, Tag
from tests.fakes import FakeClassifier, FakeClock, FakeExtractor


def window(marker: float) -> NDArray[np.float32]:
    return np.full((150, 20), marker, dtype=np.float32)


def make(
    tags: tuple[Tag, ...] = (Tag.DANGEROUS,),
) -> tuple[ClassificationStage, FakeExtractor, FakeClassifier, LatestSlot[Detection], FakeClock]:
    extractor, classifier = FakeExtractor(), FakeClassifier(tags)
    slot: LatestSlot[Detection] = LatestSlot()
    clock = FakeClock()
    return (
        ClassificationStage(extractor, classifier, slot, clock),
        extractor,
        classifier,
        slot,
        clock,
    )


def test_warm_up_skips_window_1_and_calibrates_on_window_2() -> None:
    stage, extractor, classifier, slot, _ = make()

    stage.on_window(window(1), 1.0)
    assert extractor.windows == []

    stage.on_window(window(2), 2.0)
    assert [float(w[0, 0]) for w in extractor.windows] == [2.0]
    assert classifier.calls == []
    assert slot.get(timeout=0) is None

    stage.on_window(window(3), 3.0)
    assert [float(w[0, 0]) for w in extractor.windows] == [2.0, 3.0]
    assert len(classifier.calls) == 1


def test_classified_window_publishes_a_detection() -> None:
    stage, extractor, _, slot, clock = make((Tag.NON_DANGEROUS,))
    for i in range(3):
        stage.on_window(window(i), 10.0 + i)

    assert slot.get(timeout=0) == Detection(
        tag=Tag.NON_DANGEROUS,
        timestamp=clock.now(),
        window_end_ts=12.0,
        features=tuple(float(v) for v in extractor.features),
    )


def test_extraction_failure_skips_the_window_and_keeps_warm_up_pending() -> None:
    stage, extractor, classifier, slot, _ = make()
    stage.on_window(window(1), 1.0)
    extractor.error = RuntimeError("native failure")
    stage.on_window(window(2), 2.0)
    extractor.error = None

    stage.on_window(window(3), 3.0)  # calibrates instead of window #2
    assert classifier.calls == []
    stage.on_window(window(4), 4.0)
    assert slot.get(timeout=0) is not None


def test_classification_failure_skips_the_window() -> None:
    stage, _, classifier, slot, _ = make()
    classifier.error = ClassificationError("unknown code 9")
    for i in range(3):
        stage.on_window(window(i), float(i))
    assert slot.get(timeout=0) is None

    classifier.error = None
    stage.on_window(window(4), 4.0)
    assert slot.get(timeout=0) is not None
