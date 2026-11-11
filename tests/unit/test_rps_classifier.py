"""``RpsClassifier`` and ``rps_features`` parity with the training export (F-D, AC-06)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.vision.rps_classifier import (
    FEATURE_VERSION,
    RPS_CLASSES,
    RpsClassifier,
    rps_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_NPZ = REPO_ROOT / "models" / "rps_mlp.npz"


def reference_features(kpts2d: NDArray[np.float32]) -> NDArray[np.float64]:
    """An independent implementation of the version 1 features.

    Written as an explicit rotation matrix product, so a mistake shared with
    ``rps_features`` is unlikely.
    """

    points = np.asarray(kpts2d, dtype=np.float64)[:, :2]
    centred = points - points[0]
    middle = centred[9]
    scale = math.hypot(*middle)
    if scale < 1e-6:
        return np.zeros(42)
    theta = -math.pi / 2.0 - math.atan2(middle[1], middle[0])
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    return (centred @ rotation.T / scale).reshape(42)


def reference_forward(
    features: NDArray[np.float64], weights: dict[str, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """The documented 42-64-64-4 forward pass, independent of the runtime."""

    hidden = np.maximum(features @ weights["w1"] + weights["b1"], 0.0)
    hidden = np.maximum(hidden @ weights["w2"] + weights["b2"], 0.0)
    logits = hidden @ weights["w3"] + weights["b3"]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def hand_points() -> NDArray[np.float32]:
    """A plausible 21x3 hand: wrist at the origin, knuckle 9 above it."""

    rng = np.random.default_rng(7)
    points = rng.uniform(-0.2, 0.2, size=(21, 3)).astype(np.float32)
    points[:, 2] = 0.9
    points[0, :2] = (0.5, 0.5)
    points[9, :2] = (0.5, 0.2)
    return points


def fixture_weights(seed: int = 3) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, 0.3, size=(42, 64)),
        "b1": rng.normal(0.0, 0.1, size=64),
        "w2": rng.normal(0.0, 0.3, size=(64, 64)),
        "b2": rng.normal(0.0, 0.1, size=64),
        "w3": rng.normal(0.0, 0.3, size=(64, 4)),
        "b3": rng.normal(0.0, 0.1, size=4),
    }


def write_fixture(
    path: Path,
    *,
    feature_version: object = FEATURE_VERSION,
    classes: tuple[str, ...] = RPS_CLASSES,
    samples: int = 6,
) -> dict[str, NDArray[np.float64]]:
    weights = fixture_weights()
    rng = np.random.default_rng(11)
    test_input = rng.random((samples, 21, 3), dtype=np.float32)
    test_input[:, :, :2] += 0.1
    features = np.stack([reference_features(hand) for hand in test_input])
    test_proba = reference_forward(features, weights)
    np.savez(
        path,
        **{key: value.astype(np.float32) for key, value in weights.items()},
        feature_version=np.array(feature_version),
        classes=np.array(classes),
        test_input=test_input.astype(np.float32),
        test_proba=test_proba.astype(np.float32),
    )
    return weights


# -- features --------------------------------------------------------------


def test_features_are_forty_two_float32_values() -> None:
    features = rps_features(hand_points())

    assert features.shape == (42,)
    assert features.dtype == np.float32


def test_the_wrist_is_at_the_origin_and_the_middle_knuckle_points_up() -> None:
    features = rps_features(hand_points())

    assert features[0] == pytest.approx(0.0, abs=1e-6)
    assert features[1] == pytest.approx(0.0, abs=1e-6)
    assert features[18] == pytest.approx(0.0, abs=1e-6)
    assert features[19] == pytest.approx(-1.0, abs=1e-6)


def test_features_are_scale_rotation_and_translation_invariant() -> None:
    reference = rps_features(hand_points())

    moved = hand_points()
    moved[:, :2] = moved[:, :2] * 4.0 + 12.0
    np.testing.assert_allclose(rps_features(moved), reference, atol=1e-5)

    # A hand rotated in the image plane produces the same canonical features.
    angle = math.radians(35.0)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    rotated = hand_points()
    rotated[:, :2] = rotated[:, :2] @ rotation.T
    np.testing.assert_allclose(rps_features(rotated), reference, atol=1e-5)


def test_the_confidence_column_is_ignored() -> None:
    points = hand_points()
    reference = rps_features(points)

    points[:, 2] = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    np.testing.assert_array_equal(rps_features(points), reference)


def test_a_degenerate_hand_yields_zeros() -> None:
    points = hand_points()
    points[9, :2] = points[0, :2]

    np.testing.assert_array_equal(rps_features(points), np.zeros(42, dtype=np.float32))


def test_non_finite_coordinates_yield_zeros() -> None:
    points = hand_points()
    points[4, 0] = np.nan

    np.testing.assert_array_equal(rps_features(points), np.zeros(42, dtype=np.float32))


def test_a_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="21"):
        rps_features(np.zeros((20, 3), dtype=np.float32))


# -- loading and inference -------------------------------------------------


def test_the_fixture_parity_vector_matches_the_runtime(tmp_path: Path) -> None:
    path = tmp_path / "rps_mlp.npz"
    write_fixture(path)
    with np.load(path) as data:
        test_input = np.asarray(data["test_input"])
        test_proba = np.asarray(data["test_proba"])

    classifier = RpsClassifier.load(path)

    assert classifier.classes == RPS_CLASSES
    assert classifier.feature_version == FEATURE_VERSION
    np.testing.assert_allclose(classifier.predict_proba(test_input), test_proba, atol=1e-5)


def test_predict_returns_the_most_likely_class_and_its_probability(tmp_path: Path) -> None:
    path = tmp_path / "rps_mlp.npz"
    write_fixture(path)
    classifier = RpsClassifier.load(path)
    points = hand_points()

    name, probability = classifier.predict(points)
    probabilities = classifier.predict_proba(points[np.newaxis, :, :])[0]

    assert name == RPS_CLASSES[int(np.argmax(probabilities))]
    assert probability == pytest.approx(float(np.max(probabilities)))
    assert probabilities.sum() == pytest.approx(1.0)


def test_an_unknown_feature_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rps_mlp.npz"
    write_fixture(path, feature_version="v2")

    with pytest.raises(ValueError, match="feature version"):
        RpsClassifier.load(path)


def test_other_classes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rps_mlp.npz"
    write_fixture(path, classes=("rock", "paper", "scissors", "other"))

    with pytest.raises(ValueError, match="classes"):
        RpsClassifier.load(path)


def test_a_missing_key_is_named(tmp_path: Path) -> None:
    path = tmp_path / "rps_mlp.npz"
    weights = fixture_weights()
    np.savez(
        path,
        **{key: value.astype(np.float32) for key, value in weights.items() if key != "b3"},
        feature_version=np.array(FEATURE_VERSION),
        classes=np.array(RPS_CLASSES),
    )

    with pytest.raises(ValueError, match="b3"):
        RpsClassifier.load(path)


def test_a_wrong_batch_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rps_mlp.npz"
    write_fixture(path)
    classifier = RpsClassifier.load(path)

    with pytest.raises(ValueError, match="N, 21, 3"):
        classifier.predict_proba(np.zeros((2, 20, 3), dtype=np.float32))


# -- the real export (T2) --------------------------------------------------


@pytest.mark.skipif(not REAL_NPZ.is_file(), reason="models/rps_mlp.npz is produced by T2")
def test_the_trained_classifier_parity_vector_matches_the_runtime() -> None:
    with np.load(REAL_NPZ) as data:
        test_input = np.asarray(data["test_input"])
        test_proba = np.asarray(data["test_proba"])

    classifier = RpsClassifier.load(REAL_NPZ)

    assert classifier.classes == RPS_CLASSES
    np.testing.assert_allclose(classifier.predict_proba(test_input), test_proba, atol=1e-5)
