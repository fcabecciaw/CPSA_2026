"""Tests for the rock/paper/scissors landmark classifier (synthetic data only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cpsa_ml.gestures import (
    FEATURE_DIM,
    FEATURE_VERSION,
    RPS_CLASSES,
    classify_ratios,
    export_npz,
    finger_ratios,
    forward_proba,
    label_to_class,
    load_parity,
    load_split,
    main,
    rps_features,
    rule_baseline,
)

# Finger MCP offsets and directions, relative to the wrist and multiplied by
# the hand scale; index, middle, ring and pinky.
MCPS = ((-0.25, -0.30), (-0.05, -0.32), (0.15, -0.28), (0.30, -0.22))
DIRECTIONS = ((-0.35, -1.0), (0.0, -1.0), (0.15, -1.0), (0.40, -0.90))
TIP_INDICES = (8, 12, 16, 20)
MCP_INDICES = (5, 9, 13, 17)


def make_hand(
    extended: tuple[bool, bool, bool, bool],
    *,
    wrist: tuple[float, float] = (0.5, 0.72),
    scale: float = 0.2,
) -> np.ndarray:
    """Return plausible ``(21, 3)`` HaGRID-style landmarks for one hand."""

    origin = np.asarray(wrist, dtype=np.float64)
    points = np.zeros((21, 2), dtype=np.float64)
    points[0] = origin
    # Thumb 1-4: a short chained digit to the side of the wrist.
    thumb = np.array([(-0.10, -0.01), (-0.16, -0.04), (-0.20, -0.07), (-0.22, -0.10)])
    points[1:5] = origin + thumb * scale
    for finger, (mcp, direction, is_extended) in enumerate(
        zip(MCPS, DIRECTIONS, extended, strict=True)
    ):
        knuckle = origin + np.asarray(mcp) * scale
        # An extended finger reaches up; a curled one folds back onto the palm.
        reach = np.asarray(direction) * scale * (1.4 if is_extended else -0.25)
        points[MCP_INDICES[finger]] = knuckle
        for joint in range(1, 4):
            points[MCP_INDICES[finger] + joint] = knuckle + reach * (joint / 4.0)
        points[TIP_INDICES[finger]] = knuckle + reach
    keypoints = np.zeros((21, 3), dtype=np.float32)
    keypoints[:, :2] = points
    keypoints[:, 2] = 1.0
    return keypoints


def write_annotations(
    root: Path,
    split: str,
    gesture: str,
    hands: list[np.ndarray],
    *,
    labels: list[str] | None = None,
) -> Path:
    """Write one HaGRID-style ``<gesture>.json`` with ``hands`` (one per image)."""

    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for index, hand in enumerate(hands):
        entry = {
            "bboxes": [[0.2, 0.2, 0.4, 0.4]],
            "hand_landmarks": [hand[:, :2].tolist()],
            "labels": [labels[index] if labels else gesture],
        }
        data[f"image_{index:05d}"] = entry
    path = split_dir / f"{gesture}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def make_dataset(
    root: Path, *, per_class: int = 12, splits: tuple[str, ...] = ("train", "val", "test")
) -> Path:
    """Write a small, separable annotation tree and return its root directory."""

    files = {
        "fist": ("fist", (False, False, False, False)),
        "palm": ("palm", (True, True, True, True)),
        "peace": ("peace", (True, True, False, False)),
        "one": ("one", (True, False, False, False)),
        "three": ("three", (True, True, True, False)),
        "rock": ("rock", (True, False, False, True)),
    }
    for split in splits:
        for gesture, (label, extended) in files.items():
            hands = [
                make_hand(extended, wrist=(0.4 + 0.01 * (i % 5), 0.7)) for i in range(per_class)
            ]
            write_annotations(
                root / "annotations", split, gesture, hands, labels=[label] * per_class
            )
    return root / "annotations"


def test_rps_features_geometry():
    hand = make_hand((True, True, True, True))
    features = rps_features(hand)
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32
    # Wrist at the origin, middle knuckle at (0, -1) by construction.
    assert np.allclose(features[:2], 0.0, atol=1e-6)
    assert abs(float(features[18])) < 1e-5
    assert features[19] == pytest.approx(-1.0, abs=1e-5)


def test_rps_features_invariant_to_translation_rotation_and_scale():
    hand = make_hand((True, True, False, False), wrist=(0.55, 0.65), scale=0.25)
    reference = rps_features(hand)

    angle = np.deg2rad(37.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = hand.copy()
    moved[:, :2] = (moved[:, :2] - moved[0, :2]) @ rotation.T * 1.7 + np.array([0.1, 0.9])
    assert np.allclose(rps_features(moved), reference, atol=1e-5)

    flipped = hand.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    mirrored = rps_features(flipped)
    assert np.allclose(mirrored, reference * np.array([-1.0, 1.0] * 21), atol=1e-5)


def test_rps_features_ignores_confidence_and_rejects_bad_shapes():
    hand = make_hand((True, False, True, False))
    reference = rps_features(hand)
    hand[:, 2] = np.linspace(0.0, 1.0, 21)
    assert np.allclose(rps_features(hand), reference)

    degenerate = make_hand((True, True, True, True))
    degenerate[9, :2] = degenerate[0, :2]
    assert np.array_equal(rps_features(degenerate), np.zeros(FEATURE_DIM, dtype=np.float32))

    with pytest.raises(ValueError, match="21, 3"):
        rps_features(np.zeros((20, 3), dtype=np.float32))


def test_label_mapping_includes_hagrid_rock_as_none():
    assert label_to_class("fist") == RPS_CLASSES.index("rock")
    assert label_to_class("palm") == RPS_CLASSES.index("paper")
    assert label_to_class("Stop_Inverted") == RPS_CLASSES.index("paper")
    assert label_to_class("peace") == RPS_CLASSES.index("scissors")
    assert label_to_class("peace_inverted") == RPS_CLASSES.index("scissors")
    # HaGRID's own "rock" label is the horns sign.
    assert label_to_class("rock") == RPS_CLASSES.index("none")
    assert label_to_class("no_gesture") == RPS_CLASSES.index("none")
    assert label_to_class("unknown_gesture") == RPS_CLASSES.index("none")


def test_load_split_maps_labels_skips_bad_hands_and_caps_none(tmp_path: Path):
    root = make_dataset(tmp_path, per_class=5, splits=("train",))
    # A malformed hand and an entry without labels exercise the reader guards.
    path = root / "train" / "fist.json"
    data = json.loads(path.read_text())
    first = next(iter(data))
    data[first]["hand_landmarks"].append([[0.0, 0.0]] * 20)
    data[first]["labels"].append("fist")
    data["second_image"] = {
        "hand_landmarks": [make_hand((False, False, False, False))[:, :2].tolist()]
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    split = load_split(root, "train", seed=3)
    counts = split.counts()
    assert counts["rock"] == 6  # 5 labelled + 1 without a label, from the file name
    assert counts["paper"] == 5
    assert counts["scissors"] == 5
    # 5 * one/like/rock(horns) = 15 none hands, capped at 2 * largest minority.
    assert counts["none"] == 12
    assert len(split) == sum(counts.values())
    assert split.features.shape == (len(split), FEATURE_DIM)
    assert split.ratios.shape == (len(split), 4)
    assert split.features.dtype == np.float32
    assert split.labels.dtype == np.int64


def test_load_split_keeps_keypoints_for_parity(tmp_path: Path):
    root = make_dataset(tmp_path, per_class=4, splits=("test",))
    split = load_split(root, "test", keep_kpts=True)
    assert split.kpts2d is not None
    assert split.kpts2d.shape == (len(split), 21, 3)
    assert np.allclose(split.kpts2d[:, :, 2], 1.0)
    parity_input, parity_features = load_parity(split, seed=0)
    assert parity_input.shape == (16, 21, 3)
    assert parity_features.shape == (16, FEATURE_DIM)
    assert parity_features.dtype == np.float32


def test_finger_ratios_and_rule_baseline(tmp_path: Path):
    root = make_dataset(tmp_path, per_class=8, splits=("train", "test"))
    train = load_split(root, "train")
    test = load_split(root, "test")

    paper = make_hand((True, True, True, True))
    fist = make_hand((False, False, False, False))
    assert np.all(finger_ratios(paper) > finger_ratios(fist))
    assert classify_ratios(finger_ratios(paper)[None, :], 1.2)[0] == RPS_CLASSES.index("paper")
    assert classify_ratios(finger_ratios(fist)[None, :], 1.2)[0] == RPS_CLASSES.index("rock")

    baseline = rule_baseline(train.ratios, train.labels, test.ratios, test.labels)
    assert 0.0 <= baseline["test_accuracy"] <= 1.0
    assert 0.0 <= baseline["train_accuracy"] <= 1.0
    assert np.asarray(baseline["confusion_matrix"]).shape == (4, 4)


REQUIRED_KEYS = {
    "w1",
    "b1",
    "w2",
    "b2",
    "w3",
    "b3",
    "feature_version",
    "classes",
    "test_input",
    "test_proba",
}


def test_export_npz_contract(tmp_path: Path):
    weights = {
        "w1": np.zeros((42, 64), dtype=np.float32),
        "b1": np.zeros(64, dtype=np.float32),
        "w2": np.zeros((64, 64), dtype=np.float32),
        "b2": np.zeros(64, dtype=np.float32),
        "w3": np.zeros((64, 4), dtype=np.float32),
        "b3": np.zeros(4, dtype=np.float32),
    }
    test_input = np.stack([make_hand((True, False, True, False)) for _ in range(16)])
    features = np.stack([rps_features(sample) for sample in test_input])
    test_proba = forward_proba(features, weights)
    path = tmp_path / "tiny.npz"

    size = export_npz(path, weights, test_input, test_proba)
    assert size == path.stat().st_size
    assert size < 100 * 1024

    with np.load(path) as data:
        assert set(data.files) >= REQUIRED_KEYS
        assert data["w1"].shape == (42, 64)
        assert data["w2"].shape == (64, 64)
        assert data["w3"].shape == (64, 4)
        assert data["b1"].shape == (64,)
        assert data["b2"].shape == (64,)
        assert data["b3"].shape == (4,)
        assert str(data["feature_version"]) == FEATURE_VERSION
        assert [str(name) for name in data["classes"]] == list(RPS_CLASSES)
        assert data["test_input"].shape == (16, 21, 3)
        assert data["test_proba"].shape == (16, 4)
        assert np.allclose(data["test_proba"].sum(axis=1), 1.0, atol=1e-6)
        recomputed = forward_proba(
            np.stack([rps_features(sample) for sample in data["test_input"]]),
            {name: data[name] for name in ("w1", "b1", "w2", "b2", "w3", "b3")},
        )
        assert np.allclose(recomputed, data["test_proba"], atol=1e-6)


def test_cli_trains_exports_and_reports(tmp_path: Path):
    root = make_dataset(tmp_path, per_class=24)
    out = tmp_path / "models" / "rps_mlp.npz"
    report_path = tmp_path / "build" / "report.json"

    exit_code = main(
        [
            "--annotations",
            str(root),
            "--out",
            str(out),
            "--report",
            str(report_path),
            "--epochs",
            "15",
            "--batch-size",
            "16",
            "--max-per-class",
            "24",
            "--seed",
            "1",
        ]
    )
    assert exit_code == 0
    assert out.is_file()
    assert out.stat().st_size < 100 * 1024

    report = json.loads(report_path.read_text())
    metrics = report["metrics"]
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["target"] == 0.95
    assert metrics["confusion_matrix_order"] == list(RPS_CLASSES)
    matrix = np.asarray(metrics["confusion_matrix"])
    assert matrix.shape == (4, 4)
    assert int(matrix.sum()) == report["dataset"]["splits"]["test"]["samples"]
    assert report["baseline"]["name"] == "finger_extension_rules"
    assert 0.0 <= report["baseline"]["test_accuracy"] <= 1.0
    assert report["artifact"]["parity_samples"] == 16
    assert report["artifact"]["under_limit"] is True
    assert report["model"]["architecture"] == [42, 64, 64, 4]

    # The tiny synthetic set is linearly separable enough to be learned.
    assert metrics["test_accuracy"] >= 0.9

    with np.load(out) as data:
        assert set(data.files) >= REQUIRED_KEYS
        assert data["test_input"].shape == (16, 21, 3)
        assert data["test_proba"].shape == (16, 4)
        weights = {name: data[name] for name in ("w1", "b1", "w2", "b2", "w3", "b3")}
        parity = forward_proba(
            np.stack([rps_features(sample) for sample in data["test_input"]]), weights
        )
        assert np.allclose(parity, data["test_proba"], atol=1e-5)
