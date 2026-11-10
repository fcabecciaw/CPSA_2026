"""Tests for the activation x training-size variant study (no Kaggle, no network)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from cpsa_ml import variant_study
from cpsa_ml.datasets.curation import ExtraDataset
from cpsa_ml.models import load_yolo, model_yaml_path
from cpsa_ml.schema import HALFBUST_62
from cpsa_ml.variant_study import (
    ALBUMENTATIONS_MIN_VERSION,
    AUGMENTATIONS,
    DEFAULT_VARIANTS,
    MOTION_BLUR_DESCRIPTION,
    STUDY_ACTIVATIONS,
    StudyConfig,
    VariantEntry,
    build_albumentations,
    build_metrics_payload,
    build_mixed_data_yaml,
    prepare_study_data,
    publish_artifacts,
    train_study,
    train_variant,
    transfer_coco_weights,
)

# ---------------------------------------------------------------------------
# Matrix rows
# ---------------------------------------------------------------------------


def test_variant_entry_normalizes_activation_and_sizes() -> None:
    entry = VariantEntry("LReLU", 512, [384, 512])
    assert entry.activation == "lrelu"
    assert entry.train_imgsz == 512
    assert entry.deploy_size == (384, 512)
    assert entry.name == "lrelu_512"
    assert entry.config_name == "yolov8n-pose-lrelu.yaml"
    assert entry.to_dict() == {
        "variant": "lrelu_512",
        "activation": "lrelu",
        "config": "yolov8n-pose-lrelu.yaml",
        "train_imgsz": 512,
        "deploy_size": [384, 512],
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"activation": "silu", "train_imgsz": 640, "deploy_size": (480, 640)}, "activation"),
        ({"activation": "hswish", "train_imgsz": 650, "deploy_size": (480, 650)}, "multiple"),
        ({"activation": "hswish", "train_imgsz": 640, "deploy_size": (0, 640)}, "positive"),
        ({"activation": "hswish", "train_imgsz": 640, "deploy_size": (490, 640)}, "multiple"),
    ],
)
def test_variant_entry_rejects_invalid_rows(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VariantEntry(**kwargs)


def test_training_and_deployment_sizes_are_independent() -> None:
    entry = VariantEntry("hswish", 640, (384, 512))
    assert entry.deploy_size == (384, 512)
    assert entry.train_imgsz == 640


def test_default_matrix_is_the_proposed_study() -> None:
    assert [
        (entry.activation, entry.train_imgsz, entry.deploy_size) for entry in DEFAULT_VARIANTS
    ] == [
        ("hswish", 640, (480, 640)),
        ("lrelu", 640, (480, 640)),
        ("lrelu", 512, (384, 512)),
        ("lrelu", 384, (288, 384)),
    ]
    assert [entry.name for entry in DEFAULT_VARIANTS] == [
        "hswish_640",
        "lrelu_640",
        "lrelu_512",
        "lrelu_384",
    ]


def test_default_variants_use_packaged_configs() -> None:
    for entry in DEFAULT_VARIANTS:
        assert entry.activation in STUDY_ACTIVATIONS
        assert model_yaml_path("n", entry.activation).name == entry.config_name


# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------


def _label_line(n_keypoints: int) -> str:
    """One YOLO-pose label row with every keypoint visible."""

    values = [0.0, 0.5, 0.5, 1.0, 1.0] + [0.5, 0.5, 1.0] * n_keypoints
    return " ".join(f"{value:g}" for value in values)


def _dataset(root: Path, *, kpt_shape: tuple[int, int] = (62, 3), crops: int = 2) -> Path:
    """Create the images, labels and ``data.yaml`` of a built dataset."""

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(crops):
            stem = f"{split}-{index}"
            Image.new("RGB", (64, 200)).save(root / "images" / split / f"{stem}.jpg", format="JPEG")
            (root / "labels" / split / f"{stem}.txt").write_text(
                _label_line(kpt_shape[0]) + "\n", encoding="utf-8"
            )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "images/train",
                "val": "images/val",
                "kpt_shape": list(kpt_shape),
                "flip_idx": list(HALFBUST_62.flip_idx)[: kpt_shape[0]],
                "names": {0: "person"},
            }
        ),
        encoding="utf-8",
    )
    return root


def _config(tmp_path: Path, **overrides: object) -> StudyConfig:
    values: dict[str, object] = {
        "data": _dataset(tmp_path / "halfbust") / "data.yaml",
        "coco_weights": tmp_path / "yolov8n-pose.pt",
        "project": tmp_path / "runs",
        "publish": tmp_path / "publish",
    }
    values.update(overrides)
    return StudyConfig(**values)


def test_study_config_defaults(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.variants == DEFAULT_VARIANTS
    assert config.project == tmp_path / "runs"
    assert config.publish == tmp_path / "publish"
    assert config.schema == "halfbust62"
    assert config.seed == 0
    assert config.epochs == 100
    assert config.batch == 32
    assert config.patience == 30
    assert config.augmentations == dict(AUGMENTATIONS)
    assert config.resume == {}
    assert config.full_data is None
    assert config.mixed_data == config.publish / "data_mixed.yaml"
    assert config.min_height == 128
    assert config.hand_repeat == 2
    assert config.face_repeat == 1
    assert config.labels_root is None
    assert config.extra_data == ()

    # Every config owns its augmentation copy.
    config.augmentations["mosaic"] = 0.0
    assert AUGMENTATIONS["mosaic"] == 1.0
    assert _config(tmp_path).augmentations["mosaic"] == 1.0


def test_study_config_normalizes_the_curation_knobs(tmp_path: Path) -> None:
    extra = tmp_path / "hagrid"
    config = _config(
        tmp_path,
        labels_root=str(tmp_path / "halfbust-pl"),
        extra_data=[(str(extra), 2)],
    )
    assert config.labels_root == tmp_path / "halfbust-pl"
    assert config.extra_data == (ExtraDataset(extra, 2),)


def test_augmentations_pin_the_seated_demo_recipe() -> None:
    assert AUGMENTATIONS["cos_lr"] is True
    assert AUGMENTATIONS["degrees"] == 10.0
    assert AUGMENTATIONS["close_mosaic"] == 15
    assert AUGMENTATIONS["mixup"] == 0.0
    assert AUGMENTATIONS["fliplr"] == 0.5
    assert AUGMENTATIONS["scale"] == 0.5
    assert AUGMENTATIONS["optimizer"] == "SGD"
    assert AUGMENTATIONS["lr0"] == 0.01
    assert AUGMENTATIONS["lrf"] == 0.01
    assert AUGMENTATIONS["momentum"] == 0.937
    assert AUGMENTATIONS["warmup_epochs"] == 3.0
    assert AUGMENTATIONS["pose"] == 12.0  # Ultralytics default, deliberately not tuned
    assert AUGMENTATIONS["multi_scale"] == 0.0  # recommendation: off, it costs speed


def test_study_config_normalizes_resume_paths(tmp_path: Path) -> None:
    entry = VariantEntry("hswish", 640, (480, 640))
    config = _config(
        tmp_path,
        variants=(entry,),
        resume={entry.name: str(tmp_path / "last.pt")},
    )
    assert config.resume == {entry.name: tmp_path / "last.pt"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"variants": ()}, "must not be empty"),
        (
            {
                "variants": (
                    VariantEntry("hswish", 640, (480, 640)),
                    VariantEntry("HSWISH", 640, (480, 640)),
                )
            },
            "unique",
        ),
        ({"resume": {"relu_640": "last.pt"}}, "match no variant"),
        ({"epochs": 0}, "epochs"),
        ({"batch": 0}, "batch"),
        ({"workers": -1}, "workers"),
        ({"patience": -1}, "patience"),
        ({"val_conf": 1.5}, "val_conf"),
        ({"val_iou": 0.0}, "val_iou"),
        ({"min_height": -1}, "min_height"),
        ({"hand_repeat": 0}, "hand_repeat"),
        ({"face_repeat": 0}, "face_repeat"),
    ],
)
def test_study_config_rejects_invalid_values(tmp_path: Path, overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **overrides)


# ---------------------------------------------------------------------------
# Mixed dataset
# ---------------------------------------------------------------------------


def test_mixed_data_yaml_concatenates_both_train_dirs(tmp_path: Path) -> None:
    half = _dataset(tmp_path / "halfbust")
    full = _dataset(tmp_path / "fullframe")
    target = tmp_path / "work" / "data_mixed.yaml"

    payload = build_mixed_data_yaml(half / "data.yaml", full, target)

    assert payload["train"] == [str(half / "images" / "train"), str(full / "images" / "train")]
    assert payload["val"] == str(half / "images" / "val")
    assert payload["val_full"] == str(full / "images" / "val")
    assert payload["kpt_shape"] == [62, 3]
    assert payload["flip_idx"] == list(HALFBUST_62.flip_idx)
    assert payload["names"] == {0: "person"}
    assert yaml.safe_load(target.read_text()) == payload


def test_mixed_data_yaml_falls_back_to_half_body_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    half = _dataset(tmp_path / "halfbust")
    target = tmp_path / "data_mixed.yaml"

    with caplog.at_level(logging.WARNING):
        payload = build_mixed_data_yaml(half, None, target)

    assert payload["train"] == [str(half / "images" / "train")]
    assert "val_full" not in payload
    assert "FULL-FRAME DATASET MISSING" in caplog.text
    assert "FULL-FRAME DATASET MISSING" in capsys.readouterr().out

    missing = build_mixed_data_yaml(half, tmp_path / "absent", target)
    assert "val_full" not in missing


def test_mixed_data_yaml_rejects_a_schema_mismatch(tmp_path: Path) -> None:
    half = _dataset(tmp_path / "halfbust")
    full = _dataset(tmp_path / "fullframe", kpt_shape=(17, 3))

    with pytest.raises(ValueError, match="does not match"):
        build_mixed_data_yaml(half, full, tmp_path / "data_mixed.yaml")


def test_mixed_data_yaml_requires_the_half_body_splits(tmp_path: Path) -> None:
    half = tmp_path / "halfbust"
    (half / "images" / "train").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="split directory"):
        build_mixed_data_yaml(half, None, tmp_path / "data_mixed.yaml")


def test_dataset_root_accepts_a_yaml_or_a_directory(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    assert variant_study.dataset_root(root) == root
    assert variant_study.dataset_root(root / "data.yaml") == root


def test_mixed_data_yaml_uses_the_curated_list_and_the_h160_split(tmp_path: Path) -> None:
    half = _dataset(tmp_path / "halfbust")
    curated = tmp_path / "work" / "curated_train.txt"
    curated.parent.mkdir()
    curated.write_text(str(half / "images" / "train" / "train-0.jpg") + "\n", encoding="utf-8")
    val_h160 = tmp_path / "work" / "curated_val_h160.txt"
    val_h160.write_text(str(half / "images" / "val" / "val-0.jpg") + "\n", encoding="utf-8")

    payload = build_mixed_data_yaml(
        half, None, tmp_path / "data_mixed.yaml", curated_train=curated, val_h160=val_h160
    )

    assert payload["train"] == [str(curated)]
    assert payload["val"] == str(half / "images" / "val")
    assert payload["val_h160"] == str(val_h160)


def test_mixed_data_yaml_reports_one_split_per_extra_dataset(tmp_path: Path) -> None:
    half = _dataset(tmp_path / "halfbust")
    extra = _dataset(tmp_path / "hagrid")
    other = _dataset(tmp_path / "other")

    payload = build_mixed_data_yaml(
        half, None, tmp_path / "data_mixed.yaml", extra_roots=[extra, other]
    )

    assert payload["val_extra1"] == str(extra / "images" / "val")
    assert payload["val_extra2"] == str(other / "images" / "val")


def test_mixed_data_yaml_rejects_an_extra_dataset_with_another_schema(tmp_path: Path) -> None:
    half = _dataset(tmp_path / "halfbust")
    extra = _dataset(tmp_path / "hagrid", kpt_shape=(17, 3))

    with pytest.raises(ValueError, match="does not match"):
        build_mixed_data_yaml(half, None, tmp_path / "data_mixed.yaml", extra_roots=[extra])


def test_prepare_study_data_writes_the_curated_lists(tmp_path: Path) -> None:
    extra = _dataset(tmp_path / "hagrid")
    config = _config(tmp_path, extra_data=[(extra, 1)])

    data = prepare_study_data(config)

    assert data.train_list == config.mixed_data.parent / "curated_train.txt"
    assert data.curation.summary_path.is_file()
    assert data.summary["train_images_total"] == 2
    assert data.summary["train_entries"] == 4  # two crops with every keypoint visible
    assert data.summary["extra_entries"] == 2
    lines = data.train_list.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert data.mixed_data["train"] == [str(data.train_list)]
    assert data.mixed_data["val_h160"] == str(data.curation.val_h160_list)
    assert data.mixed_data["val_extra1"] == str(extra / "images" / "val")
    assert config.mixed_data.is_file()


# ---------------------------------------------------------------------------
# COCO initialisation
# ---------------------------------------------------------------------------


class _TinyPoseHead(torch.nn.Module):
    """Stand-in for the Pose head: ``cv2``/``cv3``/``dfl``/``cv4`` branches."""

    def __init__(
        self,
        n_kpt: int,
        nc: int = 1,
        *,
        width: int = 2,
        init: float | None = None,
        branches: tuple[str, ...] = ("cv2", "cv3", "dfl", "cv4"),
    ) -> None:
        super().__init__()
        self.kpt_shape = (n_kpt, 3)
        self.nc = nc
        shapes = {
            "cv2": (4, width),
            "cv3": (nc, width),
            "dfl": (1, 1),
            "cv4": (3 * n_kpt, width),
        }
        for branch in branches:
            rows, columns = shapes[branch]
            layer = torch.nn.Linear(columns, rows, bias=False)
            if init is not None:
                torch.nn.init.constant_(layer.weight, init)
            setattr(self, branch, torch.nn.ModuleList([layer]))


def _head_tensor_names(model: torch.nn.Module, head: str = "model.2.") -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(head)
    }


class _TinyModel(torch.nn.Module):
    def __init__(
        self,
        n_kpt: int,
        *,
        width: int = 2,
        body_layers: int = 2,
        nc: int = 1,
        init: float | None = None,
        branches: tuple[str, ...] = ("cv2", "cv3", "dfl", "cv4"),
    ) -> None:
        super().__init__()
        self.yaml_file = f"tiny-{n_kpt}.yaml"
        layers: list[torch.nn.Module] = []
        for _ in range(body_layers):
            layer = torch.nn.Linear(width, width, bias=False)
            if init is not None:
                torch.nn.init.constant_(layer.weight, init)
            layers.append(layer)
        layers.append(_TinyPoseHead(n_kpt, nc, width=width, init=init, branches=branches))
        self.model = torch.nn.Sequential(*layers)


def _fake_teacher(monkeypatch: pytest.MonkeyPatch, teacher: object) -> None:
    monkeypatch.setattr("ultralytics.YOLO", lambda *args, **kwargs: teacher, raising=True)


def test_transfer_coco_weights_copies_the_matching_pose_head_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student_model = _TinyModel(62, init=62.0)
    teacher_model = _TinyModel(17, init=17.0)
    head_before = _head_tensor_names(student_model)
    _fake_teacher(monkeypatch, SimpleNamespace(model=teacher_model))

    report = transfer_coco_weights(
        tmp_path / "yolov8n-pose.pt", SimpleNamespace(model=student_model)
    )

    assert report["head_prefix"] == "model.2."
    assert report["transferred"] == 5
    assert report["backbone_transferred"] == 2
    assert report["head_transferred"] == 3
    assert report["reinitialised"] == 1
    assert report["mismatched"] == []
    assert report["teacher_kpt_shape"] == [17, 3]
    assert report["student_kpt_shape"] == [62, 3]
    assert report["teacher_nc"] == 1
    assert report["student_nc"] == 1
    assert report["student_parameters"] == sum(p.numel() for p in student_model.parameters())

    # Every head branch whose shape does not depend on the keypoint count was
    # copied from COCO; only cv4 keeps its fresh initialisation.
    assert sorted(report["head_transferred_names"]) == [
        "model.2.cv2.0.weight",
        "model.2.cv3.0.weight",
        "model.2.dfl.0.weight",
    ]
    assert report["reinitialised_names"] == ["model.2.cv4.0.weight"]
    student_state = student_model.state_dict()
    for name in ("model.0.weight", "model.1.weight"):
        assert torch.equal(student_state[name], torch.full_like(student_state[name], 17.0))
    for name in report["head_transferred_names"]:
        assert torch.equal(student_state[name], torch.full_like(student_state[name], 17.0)), name
    assert torch.equal(student_state["model.2.cv4.0.weight"], head_before["model.2.cv4.0.weight"])


def test_transfer_coco_weights_rejects_shape_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student = SimpleNamespace(model=_TinyModel(62, width=4))
    _fake_teacher(monkeypatch, SimpleNamespace(model=_TinyModel(17, width=2)))

    with pytest.raises(RuntimeError, match="no COCO match"):
        transfer_coco_weights(tmp_path / "yolov8n-pose.pt", student)


def test_transfer_coco_weights_rejects_a_different_head_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student = SimpleNamespace(model=_TinyModel(62, body_layers=2))
    _fake_teacher(monkeypatch, SimpleNamespace(model=_TinyModel(17, body_layers=1)))

    with pytest.raises(RuntimeError, match="head layer index"):
        transfer_coco_weights(tmp_path / "yolov8n-pose.pt", student)


def test_transfer_coco_weights_reinitialises_only_the_keypoint_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A head-only student whose only branch is cv4: nothing can be copied.
    student = SimpleNamespace(model=_TinyModel(62, body_layers=0, branches=("cv4",)))
    _fake_teacher(monkeypatch, SimpleNamespace(model=_TinyModel(17, body_layers=0)))

    with pytest.raises(RuntimeError, match="no tensors transferred"):
        transfer_coco_weights(tmp_path / "yolov8n-pose.pt", student)


def test_transfer_coco_weights_rejects_a_mismatching_non_keypoint_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student = SimpleNamespace(model=_TinyModel(62, nc=2))
    _fake_teacher(monkeypatch, SimpleNamespace(model=_TinyModel(17, nc=1)))

    with pytest.raises(RuntimeError, match="cv4 may"):
        transfer_coco_weights(tmp_path / "yolov8n-pose.pt", student)


def test_transfer_coco_weights_against_a_real_coco_pose_model(tmp_path: Path) -> None:
    from ultralytics import YOLO

    teacher = load_yolo("coco17", variant="n", activation="silu")
    checkpoint = tmp_path / "yolov8n-pose.pt"
    teacher.save(str(checkpoint))
    # `save` writes half precision, like the official file; reload before
    # comparing so the comparison uses the values the loader sees.
    teacher = YOLO(str(checkpoint), task="pose", verbose=False)
    teacher_state = teacher.model.state_dict()

    student = load_yolo("halfbust62", variant="n", activation="lrelu")
    report = transfer_coco_weights(checkpoint, student)

    assert report["teacher_kpt_shape"] == [17, 3]
    assert report["student_kpt_shape"] == [62, 3]
    assert report["mismatched"] == []
    assert report["transferred"] > 0
    assert report["head_transferred"] > 0
    assert report["reinitialised"] > 0
    assert all(name.startswith("model.22.cv4.") for name in report["reinitialised_names"])
    branches = {name.split(".")[2] for name in report["head_transferred_names"]}
    assert branches == {"cv2", "cv3", "dfl"}
    student_state = student.model.state_dict()
    for name in report["transferred_names"]:
        assert torch.equal(student_state[name], teacher_state[name]), name


# ---------------------------------------------------------------------------
# Artifacts and metrics
# ---------------------------------------------------------------------------


def test_publish_artifacts_copies_existing_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"best")
    (run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
    published = tmp_path / "published"

    artifacts = publish_artifacts(run_dir, published)

    assert artifacts == {
        "best.pt": str(published / "best.pt"),
        "results.csv": str(published / "results.csv"),
    }
    assert (published / "best.pt").read_bytes() == b"best"
    assert not (published / "last.pt").exists()


def test_publish_artifacts_creates_the_target_directory(tmp_path: Path) -> None:
    assert publish_artifacts(tmp_path / "missing-run", tmp_path / "deep" / "published") == {}
    assert (tmp_path / "deep" / "published").is_dir()


def test_build_metrics_payload_records_the_variant(tmp_path: Path) -> None:
    entry = VariantEntry("lrelu", 512, (384, 512))
    config = _config(tmp_path, seed=3, epochs=2, batch=1, schema="halfbust62")
    payload = build_metrics_payload(
        entry,
        config,
        artifacts={"best.pt": "/out/best.pt"},
        transfer={"transferred": 270},
        results={"metrics/mAP50-95(P)": 0.4},
        results_full={"metrics/mAP50-95(P)": 0.3},
        speed={"inference": 1.5},
        onnx="/out/deploy.onnx",
        versions={"ultralytics": "8.4.203"},
    )

    assert payload["variant"] == "lrelu_512"
    assert payload["activation"] == "lrelu"
    assert payload["config"] == "yolov8n-pose-lrelu.yaml"
    assert payload["train_imgsz"] == 512
    assert payload["deploy_size"] == [384, 512]
    assert payload["weights"] == "/out/best.pt"
    assert payload["onnx"] == "/out/deploy.onnx"
    assert payload["dataset"] == str(config.data)
    assert payload["seed"] == 3
    assert payload["epochs"] == 2
    assert payload["batch"] == 1
    assert payload["transfer"] == {"transferred": 270}
    assert payload["metrics"] == {
        "half": {"metrics/mAP50-95(P)": 0.4},
        "full": {"metrics/mAP50-95(P)": 0.3},
        "h160": None,
    }
    assert payload["speed_ms"] == {"inference": 1.5}
    assert payload["versions"] == {"ultralytics": "8.4.203"}
    assert payload["augmentations"] == dict(AUGMENTATIONS)
    assert payload["custom_augmentations"] is None
    assert payload["curation"] is None
    assert payload["extra_metrics"] == []
    assert payload["extra_data"] == []
    assert payload["mixed_data"] == str(config.mixed_data)
    assert payload["full_data"] is None


def test_build_metrics_payload_reports_the_h160_split_and_the_curation(tmp_path: Path) -> None:
    payload = build_metrics_payload(
        DEFAULT_VARIANTS[0],
        _config(tmp_path, extra_data=[(tmp_path / "hagrid", 2)]),
        artifacts={},
        transfer=None,
        results={"metrics/mAP50-95(P)": 0.4},
        results_h160={"metrics/mAP50-95(P)": 0.5},
        extra_metrics=[{"root": "/data/hagrid", "repeat": 2, "results": {}}],
        curation={"train_entries": 7},
        custom_augmentations=[MOTION_BLUR_DESCRIPTION],
    )

    assert payload["metrics"] == {
        "half": {"metrics/mAP50-95(P)": 0.4},
        "full": None,
        "h160": {"metrics/mAP50-95(P)": 0.5},
    }
    assert payload["curation"] == {"train_entries": 7}
    assert payload["custom_augmentations"] == [MOTION_BLUR_DESCRIPTION]
    assert payload["extra_metrics"] == [{"root": "/data/hagrid", "repeat": 2, "results": {}}]
    assert payload["extra_data"] == [{"root": str(tmp_path / "hagrid"), "repeat": 2}]


def test_build_metrics_payload_marks_missing_evidence(tmp_path: Path) -> None:
    payload = build_metrics_payload(
        DEFAULT_VARIANTS[0],
        _config(tmp_path),
        artifacts={},
        transfer=None,
    )
    assert payload["weights"] is None
    assert payload["onnx"] is None
    assert payload["metrics"] is None
    assert payload["speed_ms"] is None
    assert payload["transfer"] is None
    half_only = build_metrics_payload(
        DEFAULT_VARIANTS[0],
        _config(tmp_path),
        artifacts={},
        transfer=None,
        results={"metrics/mAP50-95(P)": 0.4},
    )
    assert half_only["metrics"] == {
        "half": {"metrics/mAP50-95(P)": 0.4},
        "full": None,
        "h160": None,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class _FakeYOLO:
    """Minimal stand-in for ``ultralytics.YOLO`` used by ``train_variant``."""

    resume_save_dir: Path | None = None
    created: list[_FakeYOLO] = []
    val_calls: list[dict] = []

    def __init__(self, source: object, task: str | None = None, verbose: bool = False) -> None:
        self.source = source
        self.task = task
        self.verbose = verbose
        self.trainer = None
        self.train_kwargs: dict = {}
        self.val_kwargs: dict = {}
        type(self).created.append(self)

    def train(self, **kwargs: object) -> None:
        self.train_kwargs = dict(kwargs)
        if set(kwargs) == {"resume"}:
            save_dir = type(self).resume_save_dir
        else:
            save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        assert save_dir is not None
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"best")
        (save_dir / "weights" / "last.pt").write_bytes(b"last")
        (save_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
        self.trainer = SimpleNamespace(save_dir=str(save_dir))

    def val(self, **kwargs: object) -> SimpleNamespace:
        self.val_kwargs = dict(kwargs)
        type(self).val_calls.append(dict(kwargs))
        scores = {"val": 0.42, "val_full": 0.31, "val_h160": 0.35}
        score = scores.get(str(kwargs.get("split")), 0.21)
        return SimpleNamespace(
            results_dict={"metrics/mAP50-95(P)": score},
            speed={"inference": 1.5},
        )


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import importlib

    import ultralytics

    models = importlib.import_module("cpsa_ml.models")
    onnx_check = importlib.import_module("cpsa_ml.onnx_check")
    train = importlib.import_module("cpsa_ml.train")

    _FakeYOLO.created = []
    _FakeYOLO.val_calls = []
    _FakeYOLO.resume_save_dir = None
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    monkeypatch.setattr(train, "parse_device", lambda device: device)
    monkeypatch.setattr(models, "load_yolo", lambda *args, **kwargs: _FakeYOLO("yaml"))
    monkeypatch.setattr(
        variant_study,
        "transfer_coco_weights",
        lambda source, student: {"transferred": 3, "coco_weights": str(source)},
    )
    export_calls: list[dict] = []

    def export_onnx(model: object, output: Path, **kwargs: object) -> Path:
        export_calls.append({"model": model, "output": Path(output), **kwargs})
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"onnx")
        return Path(output)

    monkeypatch.setattr(onnx_check, "export_onnx", export_onnx)
    monkeypatch.setattr(variant_study, "build_albumentations", lambda: None)
    return SimpleNamespace(export_calls=export_calls, yolo=_FakeYOLO)


def test_train_variant_trains_at_its_size_then_validates_and_exports_at_deploy_size(
    tmp_path: Path, fake_runtime: SimpleNamespace
) -> None:
    entry = VariantEntry("lrelu", 384, (288, 384))
    full = _dataset(tmp_path / "fullframe")
    extra = _dataset(tmp_path / "hagrid")
    config = _config(
        tmp_path,
        variants=(entry,),
        epochs=2,
        batch=4,
        device="cpu",
        val_device="cpu",
        workers=0,
        run_prefix="study",
        full_data=full,
        extra_data=[(extra, 2)],
    )

    summary = train_variant(entry, config)

    built = fake_runtime.yolo.created[0]
    assert built.train_kwargs["imgsz"] == 384
    assert built.train_kwargs["data"] == str(config.mixed_data)
    assert config.mixed_data.is_file()
    mixed = yaml.safe_load(config.mixed_data.read_text(encoding="utf-8"))
    assert mixed["train"] == [
        str(config.mixed_data.parent / "curated_train.txt"),
        str(full / "images" / "train"),
    ]
    assert mixed["val_h160"] == str(config.mixed_data.parent / "curated_val_h160.txt")
    assert mixed["val_extra1"] == str(extra / "images" / "val")
    assert "augmentations" not in built.train_kwargs
    assert built.train_kwargs["epochs"] == 2
    assert built.train_kwargs["batch"] == 4
    assert built.train_kwargs["device"] == "cpu"
    assert built.train_kwargs["seed"] == config.seed
    assert built.train_kwargs["deterministic"] is True
    assert built.train_kwargs["rect"] is False
    assert built.train_kwargs["name"] == "study-lrelu_384"
    assert built.train_kwargs["project"] == str(config.project)
    for key, value in AUGMENTATIONS.items():
        assert built.train_kwargs[key] == value

    validated = fake_runtime.yolo.created[1]
    primary, full_call, h160_call, extra_call = fake_runtime.yolo.val_calls
    assert validated is not fake_runtime.yolo.created[0]
    assert primary["imgsz"] == [288, 384]
    assert primary["split"] == "val"
    assert primary["data"] == str(config.mixed_data)
    assert primary["conf"] == config.val_conf
    assert primary["iou"] == config.val_iou
    assert primary["device"] == "cpu"
    assert full_call["split"] == "val_full"
    assert full_call["data"] == str(config.mixed_data)
    assert h160_call["split"] == "val_h160"
    assert extra_call["split"] == "val_extra1"

    assert fake_runtime.export_calls[0]["imgsz"] == (288, 384)
    assert fake_runtime.export_calls[0]["opset"] == 17

    published = config.publish / "lrelu_384"
    payload = json.loads((published / "metrics_fp32.json").read_text(encoding="utf-8"))
    assert payload["variant"] == "lrelu_384"
    assert payload["train_imgsz"] == 384
    assert payload["deploy_size"] == [288, 384]
    assert payload["metrics"] == {
        "half": {"metrics/mAP50-95(P)": 0.42},
        "full": {"metrics/mAP50-95(P)": 0.31},
        "h160": {"metrics/mAP50-95(P)": 0.35},
    }
    assert payload["extra_metrics"] == [
        {
            "root": str(extra),
            "repeat": 2,
            "split": "val_extra1",
            "results": {"metrics/mAP50-95(P)": 0.21},
        }
    ]
    assert payload["curation"]["train_images_total"] == 2
    assert payload["custom_augmentations"] is None
    assert payload["speed_ms"] == {"inference": 1.5}
    assert payload["onnx"] == str(published / "deploy.onnx")
    assert payload["transfer"]["transferred"] == 3
    assert payload["weights"] == str(published / "best.pt")
    assert (published / "transfer_report.json").is_file()
    assert summary == {
        "variant": "lrelu_384",
        "run_dir": str(config.project / "study-lrelu_384"),
        "artifacts": {
            "best.pt": str(published / "best.pt"),
            "last.pt": str(published / "last.pt"),
            "results.csv": str(published / "results.csv"),
            "deploy.onnx": str(published / "deploy.onnx"),
            "metrics_fp32.json": str(published / "metrics_fp32.json"),
            "curated_summary.json": str(config.mixed_data.parent / "curated_train_summary.json"),
            "transfer_report.json": str(published / "transfer_report.json"),
        },
    }


def test_train_variant_passes_motion_blur_and_records_it(
    tmp_path: Path, fake_runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = VariantEntry("lrelu", 512, (384, 512))
    config = _config(tmp_path, variants=(entry,))
    motion_blur = ["motion-blur"]
    monkeypatch.setattr(variant_study, "build_albumentations", lambda: motion_blur)

    train_variant(entry, config)

    built = fake_runtime.yolo.created[0]
    assert built.train_kwargs["augmentations"] is motion_blur
    payload = json.loads(
        (config.publish / entry.name / "metrics_fp32.json").read_text(encoding="utf-8")
    )
    assert payload["custom_augmentations"] == [MOTION_BLUR_DESCRIPTION]


def test_train_variant_resumes_from_the_entry_checkpoint(
    tmp_path: Path, fake_runtime: SimpleNamespace
) -> None:
    entry = VariantEntry("hswish", 640, (480, 640))
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = _config(tmp_path, variants=(entry,), resume={entry.name: checkpoint})
    published = config.publish / entry.name
    # `/kaggle/working` is empty in a new session; the report travels next to
    # the published checkpoint.
    checkpoint.with_name("transfer_report.json").write_text(
        json.dumps({"transferred": 270}), encoding="utf-8"
    )
    fake_runtime.yolo.resume_save_dir = tmp_path / "resumed-run"

    summary = train_variant(entry, config)

    resumed = fake_runtime.yolo.created[0]
    assert resumed.source == str(checkpoint)
    assert resumed.train_kwargs == {"resume": str(checkpoint)}
    assert summary["run_dir"] == str(tmp_path / "resumed-run")
    payload = json.loads((published / "metrics_fp32.json").read_text(encoding="utf-8"))
    assert payload["transfer"] == {"transferred": 270}
    assert payload["metrics"]["half"] == {"metrics/mAP50-95(P)": 0.42}
    assert payload["metrics"]["full"] is None
    assert (published / "transfer_report.json").is_file()
    assert summary["artifacts"]["transfer_report.json"] == str(published / "transfer_report.json")


def test_train_variant_rejects_a_missing_resume_checkpoint(
    tmp_path: Path, fake_runtime: SimpleNamespace
) -> None:
    entry = VariantEntry("hswish", 640, (480, 640))
    config = _config(tmp_path, variants=(entry,), resume={entry.name: tmp_path / "missing.pt"})

    with pytest.raises(FileNotFoundError, match="resume checkpoint"):
        train_variant(entry, config)
    assert fake_runtime.yolo.created == []


def test_train_study_prepares_the_data_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = DEFAULT_VARIANTS[:3]
    config = _config(tmp_path, variants=entries)
    calls: list[str] = []
    prepared = prepare_study_data(config)
    prepare_calls: list[StudyConfig] = []
    received: list[object] = []

    def fake_prepare(study: StudyConfig) -> variant_study.StudyData:
        prepare_calls.append(study)
        return prepared

    def fake_train(entry: VariantEntry, study: StudyConfig, *, data: object = None) -> dict:
        assert study is config
        calls.append(entry.name)
        received.append(data)
        return {"variant": entry.name, "run_dir": "run", "artifacts": {}}

    monkeypatch.setattr(variant_study, "prepare_study_data", fake_prepare)
    monkeypatch.setattr(variant_study, "train_variant", fake_train)

    summaries = train_study(config)

    assert prepare_calls == [config]
    assert calls == [entry.name for entry in entries]
    assert received == [prepared, prepared, prepared]
    assert [summary["variant"] for summary in summaries] == calls


# ---------------------------------------------------------------------------
# Motion blur (albumentations train argument)
# ---------------------------------------------------------------------------


class _FakeAlbumentations:
    """Stand-in for the ``albumentations`` module: records the transform arguments."""

    def __init__(self, version: str = ALBUMENTATIONS_MIN_VERSION) -> None:
        self.__version__ = version
        self.built: list[tuple[str, dict]] = []

    def _transform(self, name: str, **kwargs: object) -> SimpleNamespace:
        self.built.append((name, dict(kwargs)))
        return SimpleNamespace(name=name, kwargs=kwargs)

    def Blur(self, **kwargs: object) -> SimpleNamespace:  # noqa: N802 - mirrors albumentations
        return self._transform("Blur", **kwargs)

    def MedianBlur(self, **kwargs: object) -> SimpleNamespace:  # noqa: N802
        return self._transform("MedianBlur", **kwargs)

    def ToGray(self, **kwargs: object) -> SimpleNamespace:  # noqa: N802
        return self._transform("ToGray", **kwargs)

    def CLAHE(self, **kwargs: object) -> SimpleNamespace:  # noqa: N802
        return self._transform("CLAHE", **kwargs)

    def RandomBrightnessContrast(self, **kwargs: object) -> SimpleNamespace:  # noqa: N802
        return self._transform("RandomBrightnessContrast", **kwargs)

    def RandomGamma(self, **kwargs: object) -> SimpleNamespace:  # noqa: N802
        return self._transform("RandomGamma", **kwargs)

    def ImageCompression(self, **kwargs: object) -> SimpleNamespace:  # noqa: N802
        return self._transform("ImageCompression", **kwargs)

    def MotionBlur(self, **kwargs: object) -> SimpleNamespace:  # noqa: N802
        return self._transform("MotionBlur", **kwargs)


def _fake_albumentations(
    monkeypatch: pytest.MonkeyPatch, module: _FakeAlbumentations | None
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "albumentations":
            if module is None:
                raise ImportError("No module named 'albumentations'")
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_build_albumentations_adds_motion_blur_to_the_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeAlbumentations()
    _fake_albumentations(monkeypatch, module)

    transforms = build_albumentations()

    assert [name for name, _ in module.built] == [
        "Blur",
        "MedianBlur",
        "ToGray",
        "CLAHE",
        "RandomBrightnessContrast",
        "RandomGamma",
        "ImageCompression",
        "MotionBlur",
    ]
    assert module.built[-1][1] == {"blur_limit": (3, 9), "p": 0.15}
    assert module.built[0][1] == {"p": 0.01}
    assert transforms is not None and len(transforms) == 8


def test_build_albumentations_skips_without_the_library(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_albumentations(monkeypatch, None)

    with caplog.at_level(logging.WARNING):
        assert build_albumentations() is None

    assert "motion blur is SKIPPED" in caplog.text


def test_build_albumentations_skips_an_old_release(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_albumentations(monkeypatch, _FakeAlbumentations("1.3.9"))

    assert build_albumentations() is None
