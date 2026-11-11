"""Auto-label tests: runtime look, per-frame gates, dataset output, CLI wiring (fake predictor)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from cpsa_ml.datasets.handframe import apply_clahe
from cpsa_ml.datasets.irpseudo import (
    HandPrediction,
    IrPseudoConfig,
    build_dataset,
    label_frame,
    main,
    preprocess_ir,
    split_for,
)

WIDTH = 64
HEIGHT = 48
N_KPT = 21


def make_hand(
    *,
    conf: float = 0.9,
    kpt_conf: float = 0.9,
    low_keypoints: int = 0,
    centre: tuple[float, float] = (0.5, 0.5),
    span: float = 0.4,
) -> HandPrediction:
    """One synthetic hand filling ``span`` of the frame, with ``low_keypoints`` weak points."""

    cx = centre[0] * WIDTH
    cy = centre[1] * HEIGHT
    half_w = 0.5 * span * WIDTH
    half_h = 0.5 * span * HEIGHT
    points = np.zeros((N_KPT, 3), dtype=np.float32)
    strong = N_KPT - low_keypoints
    for index in range(N_KPT):
        x = cx + (index % 5 - 2) * half_w / 4.0
        y = cy + (index // 5 - 2) * half_h / 4.0
        points[index] = (x, y, kpt_conf if index < strong else 0.1)
    return HandPrediction(
        box=(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        conf=conf,
        keypoints=points,
    )


def write_frames(root: Path, count: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        gray = (np.arange(HEIGHT * WIDTH, dtype=np.uint16) + index * 7).reshape(HEIGHT, WIDTH)
        path = root / f"frame_{index:03d}.png"
        Image.fromarray((gray % 256).astype(np.uint8)).save(path)
        paths.append(path)
    return paths


class FakePredictor:
    """Return one prepared prediction list per call, in frame order."""

    def __init__(self, hands: Sequence[Sequence[HandPrediction]]) -> None:
        self.hands = [list(item) for item in hands]
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, image: np.ndarray) -> list[HandPrediction]:
        self.shapes.append(image.shape)
        hands = self.hands[self.calls] if self.calls < len(self.hands) else []
        self.calls += 1
        return hands


def parse_label(text: str) -> tuple[list[float], np.ndarray]:
    fields = text.split()
    box = [float(value) for value in fields[1:5]]
    points = np.array([float(value) for value in fields[5:]], dtype=np.float64).reshape(N_KPT, 3)
    return box, points


def test_preprocessing_is_the_runtime_min_max_clahe_three_channel_look() -> None:
    gray = np.arange(HEIGHT * WIDTH, dtype=np.uint16).reshape(HEIGHT, WIDTH).astype(np.uint8)

    image = preprocess_ir(gray)

    expected = np.repeat(apply_clahe(gray)[:, :, np.newaxis], 3, axis=2)
    assert image.shape == (HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(image, expected)
    np.testing.assert_array_equal(image[:, :, 0], image[:, :, 2])


def test_preprocessing_rejects_a_non_gray_frame() -> None:
    with pytest.raises(ValueError, match="2-D uint8"):
        preprocess_ir(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))


def test_frame_with_every_hand_confident_is_labelled() -> None:
    first = make_hand(centre=(0.3, 0.4))
    second = make_hand(centre=(0.75, 0.6))

    lines, reason, labeled, low = label_frame(
        [first, second], width=WIDTH, height=HEIGHT, conf=0.5, kpt_conf=0.5
    )

    assert reason == ""
    assert len(lines) == 2
    assert labeled == 2 * N_KPT
    assert low == 0
    for line in lines:
        box, points = parse_label(line)
        assert np.all(points[:, 2] == 1.0)
        assert 0.0 < box[0] < 1.0 and 0.0 < box[3] < 1.0


def test_one_low_confidence_hand_drops_the_whole_frame() -> None:
    lines, reason, _, _ = label_frame(
        [make_hand(), make_hand(conf=0.4)],
        width=WIDTH,
        height=HEIGHT,
        conf=0.5,
        kpt_conf=0.5,
    )

    assert lines == []
    assert reason == "low_conf_hand"


def test_hand_needs_fifteen_confident_keypoints() -> None:
    kept, reason, _, _ = label_frame(
        [make_hand(low_keypoints=6)], width=WIDTH, height=HEIGHT, conf=0.5, kpt_conf=0.5
    )
    dropped, drop_reason, _, _ = label_frame(
        [make_hand(low_keypoints=7)], width=WIDTH, height=HEIGHT, conf=0.5, kpt_conf=0.5
    )

    assert reason == "" and len(kept) == 1
    assert dropped == []
    assert drop_reason == "low_keypoints"


def test_weak_keypoints_are_written_with_zero_visibility() -> None:
    lines, reason, labeled, low = label_frame(
        [make_hand(low_keypoints=5)], width=WIDTH, height=HEIGHT, conf=0.5, kpt_conf=0.75
    )

    assert reason == ""
    assert (labeled, low) == (16, 5)
    _, points = parse_label(lines[0])
    assert points[:, 2].sum() == 16.0
    assert np.count_nonzero(points[:, 2] == 0.0) == 5


def test_empty_prediction_drops_the_frame() -> None:
    assert label_frame([], width=WIDTH, height=HEIGHT, conf=0.5, kpt_conf=0.5)[:2] == (
        [],
        "no_hand",
    )


def test_build_dataset_writes_yaml_labels_stats_and_contact_sheet(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", 2)
    predictor = FakePredictor([[make_hand()], [make_hand()]])

    payload = build_dataset(
        IrPseudoConfig(out=tmp_path / "dataset", frames=frames[0].parent, val_ratio=0.0),
        predictor,
    )

    out = tmp_path / "dataset"
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["kpt_shape"] == [N_KPT, 3]
    assert data["flip_idx"] == list(range(N_KPT))
    assert data["names"] == {0: "hand"}
    assert data["train"] == "images/train" and data["val"] == "images/val"

    images = sorted((out / "images" / "train").glob("*.png"))
    labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert [path.name for path in images] == ["frame_000.png", "frame_001.png"]
    assert [path.name for path in labels] == ["frame_000.txt", "frame_001.txt"]
    assert (out / "contact_sheet.png").is_file()

    source = np.array(Image.open(frames[0]))
    stored = np.array(Image.open(images[0]))
    np.testing.assert_array_equal(stored, preprocess_ir(source))

    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats == payload
    assert stats["source"]["frames"] == str(frames[0].parent)
    assert stats["thresholds"] == {"conf": 0.5, "kpt_conf": 0.5, "min_keypoints": 15}
    assert stats["frames"]["kept"] == 2
    assert stats["frames"]["dropped"] == 0
    assert stats["frames"]["splits"]["train"] == {"frames": 2, "hands": 2}
    assert stats["contact_sheet"] == "contact_sheet.png"
    assert predictor.shapes == [(HEIGHT, WIDTH, 3), (HEIGHT, WIDTH, 3)]


def test_dropped_frames_are_counted_and_not_written(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", 3)
    predictor = FakePredictor([[make_hand()], [], [make_hand(conf=0.2)]])

    payload = build_dataset(
        IrPseudoConfig(out=tmp_path / "dataset", frames=frames[0].parent, val_ratio=0.0),
        predictor,
    )

    assert payload["frames"]["kept"] == 1
    assert payload["frames"]["dropped"] == 2
    assert payload["frames"]["drop_reasons"] == {"low_conf_hand": 1, "no_hand": 1}
    assert len(list((tmp_path / "dataset" / "images" / "train").glob("*.png"))) == 1


def test_frames_npy_is_read_with_stable_names(tmp_path: Path) -> None:
    stack = np.stack(
        [
            np.arange(HEIGHT * WIDTH, dtype=np.uint16).reshape(HEIGHT, WIDTH).astype(np.uint8),
            np.flipud(np.arange(HEIGHT * WIDTH, dtype=np.uint16).reshape(HEIGHT, WIDTH)).astype(
                np.uint8
            ),
        ]
    )
    npy = tmp_path / "ir.npy"
    np.save(npy, stack)

    payload = build_dataset(
        IrPseudoConfig(out=tmp_path / "dataset", frames_npy=npy, val_ratio=0.0),
        FakePredictor([[make_hand()], [make_hand()]]),
    )

    assert payload["source"]["frames_npy"] == str(npy)
    train_images = sorted((tmp_path / "dataset" / "images" / "train").iterdir())
    assert [path.name for path in train_images] == ["ir_00000.png", "ir_00001.png"]


def test_unreadable_frame_is_counted(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", 1)
    (frames[0].parent / "broken.png").write_bytes(b"not a png")

    payload = build_dataset(
        IrPseudoConfig(out=tmp_path / "dataset", frames=frames[0].parent, val_ratio=0.0),
        FakePredictor([[make_hand()]]),
    )

    assert payload["frames"]["unreadable"] == 1
    assert payload["frames"]["kept"] == 1
    assert payload["frames"]["drop_reasons"] == {"unreadable": 1}


def make_mixed_dataset(root: Path, count: int = 2) -> None:
    for index in range(count):
        gray = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        gray[:] = 40 + index * 20
        (root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        Image.fromarray(gray).save(root / "images" / "train" / f"old_{index}.png")
        line, _, _, _ = label_frame(
            [make_hand()], width=WIDTH, height=HEIGHT, conf=0.5, kpt_conf=0.5
        )
        (root / "labels" / "train" / f"old_{index}.txt").write_text(line[0] + "\n")


def test_mix_handframe_copies_a_seeded_subset_into_train(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", 2)
    mixed_root = tmp_path / "handframe"
    make_mixed_dataset(mixed_root)

    payload = build_dataset(
        IrPseudoConfig(
            out=tmp_path / "dataset",
            frames=frames[0].parent,
            val_ratio=0.0,
            mix_handframe=mixed_root,
            mix_ratio=1.0,
        ),
        FakePredictor([[make_hand()], [make_hand()]]),
    )

    train_images = sorted((tmp_path / "dataset" / "images" / "train").iterdir())
    assert [path.name for path in train_images] == [
        "frame_000.png",
        "frame_001.png",
        "mix_old_0.png",
        "mix_old_1.png",
    ]
    assert (tmp_path / "dataset" / "labels" / "train" / "mix_old_0.txt").is_file()
    assert list((tmp_path / "dataset" / "images" / "val").iterdir()) == []
    assert payload["mix"] == {
        "root": str(mixed_root),
        "available": 2,
        "requested": 2,
        "copied": 2,
    }
    assert payload["frames"]["splits"]["train"] == {"frames": 4, "hands": 4}


def test_an_existing_dataset_is_refused_without_overwrite(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", 1)
    config = IrPseudoConfig(out=tmp_path / "dataset", frames=frames[0].parent, val_ratio=0.0)
    build_dataset(config, FakePredictor([[make_hand()]]))

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_dataset(config, FakePredictor([[make_hand()]]))

    payload = build_dataset(
        IrPseudoConfig(
            out=tmp_path / "dataset",
            frames=frames[0].parent,
            val_ratio=0.0,
            overwrite=True,
        ),
        FakePredictor([[make_hand()]]),
    )
    assert payload["frames"]["kept"] == 1


def test_config_requires_exactly_one_frame_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one of frames and frames_npy"):
        IrPseudoConfig(out=tmp_path / "out")
    with pytest.raises(ValueError, match="exactly one of frames and frames_npy"):
        IrPseudoConfig(
            out=tmp_path / "out",
            frames=tmp_path / "frames",
            frames_npy=tmp_path / "ir.npy",
        )


def test_split_is_a_stable_seeded_hash() -> None:
    names = [f"frame_{index:03d}" for index in range(200)]
    first = {name: split_for(name, seed=0, val_ratio=0.25) for name in names}

    assert first == {name: split_for(name, seed=0, val_ratio=0.25) for name in names}
    assert first != {name: split_for(name, seed=1, val_ratio=0.25) for name in names}
    assert set(first.values()) == {"train", "val"}


def test_hand_prediction_requires_the_hand21_shape() -> None:
    with pytest.raises(ValueError, match=r"\(21, 3\)"):
        HandPrediction(box=(0.0, 0.0, 10.0, 10.0), conf=0.9, keypoints=np.zeros((17, 3)))


def test_cli_wires_the_thresholds_and_the_predictor(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", 1)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    predictor = FakePredictor([[make_hand()]])
    seen: list[tuple[Path, str]] = []

    def factory(path: Path, device: str) -> FakePredictor:
        seen.append((path, device))
        return predictor

    code = main(
        [
            "--frames",
            str(frames[0].parent),
            "--weights",
            str(weights),
            "--out",
            str(tmp_path / "dataset"),
            "--conf",
            "0.7",
            "--kpt-conf",
            "0.6",
            "--val-ratio",
            "0",
            "--seed",
            "3",
            "--device",
            "cpu",
        ],
        predictor_factory=factory,
    )

    assert code == 0
    assert seen == [(weights, "cpu")]
    stats = json.loads((tmp_path / "dataset" / "stats.json").read_text(encoding="utf-8"))
    assert stats["thresholds"] == {"conf": 0.7, "kpt_conf": 0.6, "min_keypoints": 15}
    assert stats["source"]["weights"] == str(weights)
    assert stats["seed"] == 3


def test_cli_reports_an_empty_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    frames = write_frames(tmp_path / "frames", 1)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")

    def factory(path: Path, device: str) -> FakePredictor:
        return FakePredictor([[]])

    code = main(
        [
            "--frames",
            str(frames[0].parent),
            "--weights",
            str(weights),
            "--out",
            str(tmp_path / "dataset"),
            "--val-ratio",
            "0",
        ],
        predictor_factory=factory,
    )

    assert code == 1
    assert "no frame passed the gates" in capsys.readouterr().err
