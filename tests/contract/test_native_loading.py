"""Native adapters fail clearly where the aarch64 libraries cannot be loaded."""

import platform

import pytest

from cpsa.adapters.native.classifier import (
    CLASSIFIER_LIB,
    EXTRACTOR_LIB,
    NativeClassifier,
    NativeFeatureExtractor,
    NativeLibraryError,
)

pytestmark = pytest.mark.skipif(
    platform.machine() == "aarch64", reason="the libraries load on aarch64"
)


@pytest.mark.parametrize(
    ("adapter", "library"),
    [(NativeFeatureExtractor, EXTRACTOR_LIB), (NativeClassifier, CLASSIFIER_LIB)],
)
def test_construction_names_library_host_and_required_architecture(
    adapter: type, library: object
) -> None:
    with pytest.raises(NativeLibraryError) as info:
        adapter()
    message = str(info.value)
    assert str(library) in message
    assert platform.machine() in message
    assert "aarch64" in message
