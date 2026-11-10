"""Tests for the activation x training-size variant study (no Kaggle, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cpsa_ml import variant_study
from cpsa_ml.models import load_yolo, model_yaml_path
from cpsa_ml.variant_study import (
    AUGMENTATIONS,
    DEFAULT_VARIANTS,
    STUDY_ACTIVATIONS,
    StudyConfig,
    VariantEntry,
    build_metrics_payload,
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


def _config(tmp_path: Path, **overrides: object) -> StudyConfig:
    values: dict[str, object] = {
        "data": tmp_path / "data.yaml",
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
    assert config.augmentations == dict(AUGMENTATIONS)
    assert config.resume == {}

    # Every config owns its augmentation copy.
    config.augmentations["mosaic"] = 0.0
    assert AUGMENTATIONS["mosaic"] == 1.0
    assert _config(tmp_path).augmentations["mosaic"] == 1.0


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
    ],
)
def test_study_config_rejects_invalid_values(tmp_path: Path, overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **overrides)


# ---------------------------------------------------------------------------
# COCO initialisation
# ---------------------------------------------------------------------------


class _TinyHead(torch.nn.Module):
    """Stand-in for the keypoint head, with the attributes the report reads."""

    def __init__(self, n_kpt: int, nc: int = 1) -> None:
        super().__init__()
        self.kpt_shape = (n_kpt, 3)
        self.nc = nc
        self.weight = torch.nn.Parameter(torch.full((n_kpt, 2), float(n_kpt)))
        self.bias = torch.nn.Parameter(torch.full((n_kpt,), float(n_kpt)))


class _TinyModel(torch.nn.Module):
    def __init__(self, n_kpt: int, *, width: int = 2, body_layers: int = 2) -> None:
        super().__init__()
        self.yaml_file = f"tiny-{n_kpt}.yaml"
        layers: list[torch.nn.Module] = []
        for _ in range(body_layers):
            layers.append(torch.nn.Linear(width, width, bias=False))
        layers.append(_TinyHead(n_kpt))
        self.model = torch.nn.Sequential(*layers)


def _fake_teacher(monkeypatch: pytest.MonkeyPatch, teacher: object) -> None:
    monkeypatch.setattr("ultralytics.YOLO", lambda *args, **kwargs: teacher, raising=True)


def test_transfer_coco_weights_copies_matching_tensors_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student_model = _TinyModel(62)
    teacher_model = _TinyModel(17)
    head_before = {
        name: tensor.detach().clone()
        for name, tensor in student_model.state_dict().items()
        if name.startswith("model.2.")
    }
    teacher_state = {
        name: tensor.detach().clone() for name, tensor in teacher_model.state_dict().items()
    }
    _fake_teacher(monkeypatch, SimpleNamespace(model=teacher_model))

    report = transfer_coco_weights(
        tmp_path / "yolov8n-pose.pt", SimpleNamespace(model=student_model)
    )

    assert report["head_prefix"] == "model.2."
    assert report["transferred"] == 2
    assert report["reinitialised"] == 2
    assert report["mismatched"] == []
    assert report["teacher_kpt_shape"] == [17, 3]
    assert report["student_kpt_shape"] == [62, 3]
    assert report["teacher_nc"] == 1
    assert report["student_nc"] == 1
    assert report["student_parameters"] == sum(p.numel() for p in student_model.parameters())
    student_state = student_model.state_dict()
    for name in ("model.0.weight", "model.1.weight"):
        assert torch.equal(student_state[name], teacher_state[name])
    for name, tensor in head_before.items():
        assert torch.equal(student_state[name], tensor)


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


def test_transfer_coco_weights_requires_something_to_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student = SimpleNamespace(model=_TinyModel(62, body_layers=0))
    _fake_teacher(monkeypatch, SimpleNamespace(model=_TinyModel(17, body_layers=0)))

    with pytest.raises(RuntimeError, match="no tensors transferred"):
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
    assert report["reinitialised"] > 0
    assert all(name.startswith("model.22.") for name in report["reinitialised_names"])
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
    assert payload["metrics"] == {"metrics/mAP50-95(P)": 0.4}
    assert payload["speed_ms"] == {"inference": 1.5}
    assert payload["versions"] == {"ultralytics": "8.4.203"}
    assert payload["augmentations"] == dict(AUGMENTATIONS)


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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class _FakeYOLO:
    """Minimal stand-in for ``ultralytics.YOLO`` used by ``train_variant``."""

    resume_save_dir: Path | None = None
    created: list[_FakeYOLO] = []

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
        return SimpleNamespace(
            results_dict={"metrics/mAP50-95(P)": 0.42},
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
    return SimpleNamespace(export_calls=export_calls, yolo=_FakeYOLO)


def test_train_variant_trains_at_its_size_then_validates_and_exports_at_deploy_size(
    tmp_path: Path, fake_runtime: SimpleNamespace
) -> None:
    entry = VariantEntry("lrelu", 384, (288, 384))
    config = _config(
        tmp_path,
        variants=(entry,),
        epochs=2,
        batch=4,
        device="cpu",
        val_device="cpu",
        workers=0,
        run_prefix="study",
    )

    summary = train_variant(entry, config)

    built = fake_runtime.yolo.created[0]
    assert built.train_kwargs["imgsz"] == 384
    assert built.train_kwargs["data"] == str(config.data)
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
    assert validated.val_kwargs["imgsz"] == [288, 384]
    assert validated.val_kwargs["split"] == "val"
    assert validated.val_kwargs["conf"] == config.val_conf
    assert validated.val_kwargs["iou"] == config.val_iou
    assert validated.val_kwargs["device"] == "cpu"

    assert fake_runtime.export_calls[0]["imgsz"] == (288, 384)
    assert fake_runtime.export_calls[0]["opset"] == 17

    published = config.publish / "lrelu_384"
    payload = json.loads((published / "metrics_fp32.json").read_text(encoding="utf-8"))
    assert payload["variant"] == "lrelu_384"
    assert payload["train_imgsz"] == 384
    assert payload["deploy_size"] == [288, 384]
    assert payload["metrics"] == {"metrics/mAP50-95(P)": 0.42}
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
            "transfer_report.json": str(published / "transfer_report.json"),
        },
    }


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


def test_train_study_runs_every_row_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = DEFAULT_VARIANTS[:3]
    config = _config(tmp_path, variants=entries)
    calls: list[str] = []

    def fake_train(entry: VariantEntry, study: StudyConfig) -> dict:
        assert study is config
        calls.append(entry.name)
        return {"variant": entry.name, "run_dir": "run", "artifacts": {}}

    monkeypatch.setattr(variant_study, "train_variant", fake_train)

    summaries = train_study(config)

    assert calls == [entry.name for entry in entries]
    assert [summary["variant"] for summary in summaries] == calls
