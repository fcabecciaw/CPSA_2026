"""Activation x training-size variants of YOLOv8n-pose for the DPU accuracy study.

Notebook 04 is a thin wrapper around this module.  Every variant:

* starts from the official COCO ``yolov8n-pose.pt``: every shape-matching
  tensor is copied into the 62-keypoint student -- backbone, neck and the Pose
  head's box (``cv2``), class (``cv3``) and ``dfl`` branches -- and only the
  keypoint branch ``cv4`` is left at its fresh initialisation, because its
  shape depends on the number of keypoints (see
  :func:`transfer_coco_weights`);
* trains on the *mixed* dataset: the half-body ``images/train`` plus the
  full-frame ``images/train`` (see :func:`build_mixed_data_yaml`), so the box
  learns the true person extent while the half-body crops keep dominating;
* validates on the half-body val split (the primary metric Ultralytics uses for
  early stopping and ``best.pt``) and also reports the full-frame val split as
  ``val_full``;
* exports ONNX at its deployment input size;
* publishes ``best.pt``, ``last.pt``, ``results.csv``, ``metrics_fp32.json``
  (``metrics.half`` and ``metrics.full``), ``transfer_report.json`` and
  ``deploy.onnx`` under ``<publish>/<act>_<imgsz>``.

The default variant list is a **proposal** pending approval: the ``hswish``
640 model is the baseline, followed by ``lrelu`` at 640, 512 and 384 pixels of
square training (each paired with the deployment input whose long side matches).
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cpsa_ml.models import model_yaml_path
from cpsa_ml.schema import get_schema

__all__ = [
    "AUGMENTATIONS",
    "DEFAULT_VARIANTS",
    "STUDY_ACTIVATIONS",
    "StudyConfig",
    "VariantEntry",
    "build_metrics_payload",
    "build_mixed_data_yaml",
    "dataset_root",
    "publish_artifacts",
    "train_study",
    "train_variant",
    "transfer_coco_weights",
]

LOGGER = logging.getLogger(__name__)

#: Activations compared by the study (``silu`` is the operator-check control).
STUDY_ACTIVATIONS = ("hswish", "relu", "lrelu")

#: Ultralytics requires sizes that are multiples of the largest stride.
MODEL_STRIDE = 32

#: State-dict entries that are counters, not weights; never copied or reported.
_BOOKKEEPING_SUFFIXES = ("num_batches_tracked",)

#: Ultralytics defaults pinned so a library default change cannot silently make
#: the variants incomparable.  Every entry gets exactly these values.
AUGMENTATIONS: Mapping[str, Any] = {
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 10,
}


@dataclass(frozen=True, slots=True)
class VariantEntry:
    """One trained variant: activation, square training size, deployment input.

    The sizes are independent: a row can reuse a training size at a smaller
    deployment input (and the default matrix pairs each row with the input
    whose long side matches its training size).

    Args:
        activation: ``hswish``, ``relu`` or ``lrelu``; selects
            ``configs/yolov8n-pose-<activation>.yaml``.
        train_imgsz: Square Ultralytics training size in pixels.
        deploy_size: Deployment input as ``(height, width)``.

    Raises:
        ValueError: If the activation, the sizes or their relationship is
            invalid.
    """

    activation: str
    train_imgsz: int
    deploy_size: tuple[int, int]

    def __post_init__(self) -> None:
        activation = str(self.activation).lower()
        train_imgsz = int(self.train_imgsz)
        height, width = (int(self.deploy_size[0]), int(self.deploy_size[1]))
        object.__setattr__(self, "activation", activation)
        object.__setattr__(self, "train_imgsz", train_imgsz)
        object.__setattr__(self, "deploy_size", (height, width))
        if activation not in STUDY_ACTIVATIONS:
            raise ValueError(f"activation must be one of {STUDY_ACTIVATIONS}, got {activation!r}")
        if train_imgsz <= 0 or train_imgsz % MODEL_STRIDE:
            raise ValueError(
                f"train_imgsz must be a positive multiple of {MODEL_STRIDE}, got {train_imgsz}"
            )
        if height <= 0 or width <= 0:
            raise ValueError(f"deploy_size must be positive, got {(height, width)}")
        if height % MODEL_STRIDE or width % MODEL_STRIDE:
            raise ValueError(
                f"deploy_size must use multiples of {MODEL_STRIDE}, got {(height, width)}"
            )

    @property
    def name(self) -> str:
        """Folder/run name, ``<activation>_<train_imgsz>``."""

        return f"{self.activation}_{self.train_imgsz}"

    @property
    def config_name(self) -> str:
        """Exact model YAML used by this variant."""

        return f"yolov8n-pose-{self.activation}.yaml"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description recorded in ``metrics_fp32.json``."""

        return {
            "variant": self.name,
            "activation": self.activation,
            "config": self.config_name,
            "train_imgsz": self.train_imgsz,
            "deploy_size": [self.deploy_size[0], self.deploy_size[1]],
        }


#: Default study matrix. **Proposal pending approval.**
DEFAULT_VARIANTS: tuple[VariantEntry, ...] = (
    VariantEntry("hswish", 640, (480, 640)),  # baseline
    VariantEntry("lrelu", 640, (480, 640)),
    VariantEntry("lrelu", 512, (384, 512)),
    VariantEntry("lrelu", 384, (288, 384)),
)


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """Everything one Kaggle session needs to train the study matrix.

    Args:
        data: Ultralytics ``data.yaml`` of the ``halfbust62`` dataset; its val
            split is the primary metric.
        coco_weights: Official COCO ``yolov8n-pose.pt`` checkpoint.
        project: Ultralytics run directory (``runs/``).
        publish: Output root; each variant writes to ``publish/<name>``.
        schema: Keypoint schema of the deployed graph.
        variants: Rows of the study matrix.
        resume: ``{variant name: last.pt}``; unlisted entries train from COCO.
        seed: Shared seed; Ultralytics also gets ``deterministic=True``.
        epochs: Shared epoch budget.
        batch: Shared batch size (per DDP rank in Ultralytics).
        device: Ultralytics training device, ``"0,1"`` for 2x T4 DDP.
        val_device: Single device for the published validation/export.
        workers: Data-loader workers.
        patience: Early-stopping patience.
        amp: Mixed-precision training.
        val_conf: Confidence threshold for the published FP32 metrics.
        val_iou: NMS IoU for the published FP32 metrics.
        run_prefix: Ultralytics run name prefix; the run name also carries the
            variant name.
        augmentations: Optimisation/augmentation overrides shared by all rows.
        full_data: Full-frame dataset root (or its ``data.yaml``); its
            ``images/train`` joins the training set. Missing -> half-body only.
        mixed_data: Where the generated mixed ``data.yaml`` is written.
    """

    data: Path
    coco_weights: Path
    project: Path = Path("/kaggle/working/runs")
    publish: Path = Path("/kaggle/working/variant_study")
    schema: str = "halfbust62"
    variants: tuple[VariantEntry, ...] = DEFAULT_VARIANTS
    resume: Mapping[str, Path] = field(default_factory=dict)
    seed: int = 0
    epochs: int = 100
    batch: int = 32
    device: str = "0,1"
    val_device: str = "0"
    workers: int = 4
    patience: int = 50
    amp: bool = True
    val_conf: float = 0.001
    val_iou: float = 0.7
    run_prefix: str = "halfbust62-yolov8n"
    augmentations: Mapping[str, Any] = field(default_factory=lambda: dict(AUGMENTATIONS))
    full_data: Path | None = None
    mixed_data: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", Path(self.data))
        object.__setattr__(self, "coco_weights", Path(self.coco_weights))
        object.__setattr__(self, "project", Path(self.project))
        object.__setattr__(self, "publish", Path(self.publish))
        variants = tuple(self.variants)
        object.__setattr__(self, "variants", variants)
        object.__setattr__(
            self, "resume", {str(key): Path(value) for key, value in self.resume.items()}
        )
        object.__setattr__(self, "augmentations", dict(self.augmentations))
        if self.full_data is not None:
            object.__setattr__(self, "full_data", Path(self.full_data))
        mixed = self.mixed_data
        object.__setattr__(
            self,
            "mixed_data",
            Path(mixed) if mixed is not None else self.publish / "data_mixed.yaml",
        )
        if not variants:
            raise ValueError("variants must not be empty")
        names = [entry.name for entry in variants]
        if len(set(names)) != len(names):
            raise ValueError(f"variant names must be unique, got {names}")
        unknown = sorted(set(self.resume) - set(names))
        if unknown:
            raise ValueError(f"resume entries {unknown} match no variant in {names}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch == 0:
            raise ValueError("batch must be non-zero")
        if self.workers < 0:
            raise ValueError(f"workers must be non-negative, got {self.workers}")
        if self.patience < 0:
            raise ValueError(f"patience must be non-negative, got {self.patience}")
        if not 0.0 <= self.val_conf <= 1.0:
            raise ValueError(f"val_conf must be in [0, 1], got {self.val_conf}")
        if not 0.0 < self.val_iou <= 1.0:
            raise ValueError(f"val_iou must be in (0, 1], got {self.val_iou}")


# ---------------------------------------------------------------------------
# Mixed dataset
# ---------------------------------------------------------------------------


def dataset_root(path: Path | str) -> Path:
    """Return the dataset root of a ``data.yaml`` path or a dataset folder."""

    resolved = Path(path)
    if resolved.is_dir():
        return resolved
    return resolved.parent


def _dataset_schema(root: Path, schema: str) -> tuple[list[int], list[int]]:
    """``(kpt_shape, flip_idx)`` of the half-bust schema, checked against ``root``."""

    keypoint_schema = get_schema(schema)
    kpt_shape = list(keypoint_schema.kpt_shape)
    flip_idx = list(keypoint_schema.flip_idx)
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        return kpt_shape, flip_idx
    try:
        declared = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ValueError(f"cannot read {data_yaml}: {exc}") from exc
    for key, expected in (("kpt_shape", kpt_shape), ("flip_idx", flip_idx)):
        value = declared.get(key)
        if value is not None and [int(item) for item in value] != expected:
            raise ValueError(
                f"{data_yaml}: {key} {value} does not match the {keypoint_schema.name} "
                f"schema {expected}"
            )
    return kpt_shape, flip_idx


def build_mixed_data_yaml(
    half_root: Path | str,
    full_root: Path | str | None,
    out: Path | str,
    *,
    schema: str = "halfbust62",
) -> dict[str, Any]:
    """Write the mixed ``data.yaml`` and return its parsed content.

    ``train`` lists both ``images/train`` directories, ``val`` is the half-body
    val split (the primary metric Ultralytics uses for early stopping and
    ``best.pt``) and ``val_full`` is the full-frame val split, which this module
    reports next to it.  Paths are absolute because the YAML lives outside the
    two dataset roots; Ultralytics only resolves relative paths inside the
    ``train``/``val``/``test``/``minival`` keys.

    A missing or incomplete full-frame root is not fatal: the YAML then trains
    on the half-body images only and a loud warning says so, because the box
    can no longer learn the true person extent.

    Args:
        half_root: Half-bust dataset root or its ``data.yaml``.
        full_root: Full-frame dataset root or its ``data.yaml``; ``None`` skips it.
        out: Path of the generated YAML.
        schema: Keypoint schema both datasets were built with.

    Returns:
        The written mapping.

    Raises:
        FileNotFoundError: If the half-body train/val image directories are missing.
        ValueError: If a dataset declares a different keypoint schema.
    """

    half = dataset_root(half_root)
    half_train = half / "images" / "train"
    half_val = half / "images" / "val"
    for directory in (half_train, half_val):
        if not directory.is_dir():
            raise FileNotFoundError(f"half-body split directory not found: {directory}")
    kpt_shape, flip_idx = _dataset_schema(half, schema)

    train: list[str] = [str(half_train)]
    payload: dict[str, Any] = {"train": train, "val": str(half_val)}
    full = dataset_root(full_root) if full_root is not None else None
    full_train = full / "images" / "train" if full is not None else None
    full_val = full / "images" / "val" if full is not None else None
    if (
        full is None
        or full_train is None
        or full_val is None
        or not full_train.is_dir()
        or not full_val.is_dir()
    ):
        warning = (
            "FULL-FRAME DATASET MISSING"
            f" ({full if full is not None else 'full_data=None'}): training on the half-body "
            "dataset only, so the predicted box will not learn the true person extent. "
            "Attach fabiocabeccia/cpsa-fullframe and set FULL_DATA."
        )
        LOGGER.warning(warning)
        print(f"WARNING: {warning}", flush=True)
    else:
        full_kpt_shape, full_flip_idx = _dataset_schema(full, schema)
        if (full_kpt_shape, full_flip_idx) != (kpt_shape, flip_idx):
            raise ValueError(
                f"{full}/data.yaml keypoint schema differs from {half}/data.yaml; "
                "both datasets must be built with the same schema"
            )
        train.append(str(full_train))
        payload["val_full"] = str(full_val)

    payload["kpt_shape"] = kpt_shape
    payload["flip_idx"] = flip_idx
    payload["names"] = {0: "person"}
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# COCO initialisation
# ---------------------------------------------------------------------------


def transfer_coco_weights(source: Path | str, student: Any) -> dict[str, Any]:
    """Copy every shape-matching tensor from COCO weights, Pose head included.

    The Pose head of the student is ``model.22``.  Its box branch ``cv2``, class
    branch ``cv3`` and ``dfl`` do not depend on the keypoint count, so their
    tensors are copied and the student starts from a working person detector.
    Only the keypoint branch ``cv4`` is reinitialised: its last convolution has
    ``3 * n_kpt`` outputs, so 17-keypoint COCO weights cannot fill 62 keypoints.

    Args:
        source: Official ``yolov8n-pose.pt`` (or any 17-keypoint pose checkpoint).
        student: ``ultralytics.YOLO`` wrapper of the variant model.

    Returns:
        A JSON-ready report with counts, tensor names and both keypoint shapes.

    Raises:
        RuntimeError: If the models disagree on the head layer, a backbone/neck
            tensor has no COCO counterpart, a head branch other than ``cv4``
            mismatches, or nothing was copied.
    """

    import torch
    from ultralytics import YOLO

    teacher = YOLO(str(source), task="pose", verbose=False)
    head_prefix = f"model.{len(student.model.model) - 1}."
    if head_prefix != f"model.{len(teacher.model.model) - 1}.":
        raise RuntimeError("student and COCO models disagree on the head layer index")

    student_state = student.model.state_dict()
    teacher_state = teacher.model.state_dict()
    backbone_transferred: list[str] = []
    head_transferred: list[str] = []
    reinitialised: list[str] = []
    mismatched: list[str] = []
    with torch.no_grad():
        for name, target in student_state.items():
            if name.endswith(_BOOKKEEPING_SUFFIXES):
                # BatchNorm counters are bookkeeping, not learned weights.
                continue
            candidate = teacher_state.get(name)
            shapes_match = candidate is not None and candidate.shape == target.shape
            if not name.startswith(head_prefix):
                if not shapes_match:
                    mismatched.append(name)
                    continue
                target.copy_(candidate)
                backbone_transferred.append(name)
                continue
            if shapes_match:
                target.copy_(candidate)
                head_transferred.append(name)
                continue
            branch = name[len(head_prefix) :].split(".", 1)[0]
            if branch in {"cv4", "one2one_cv4"}:
                reinitialised.append(name)
                continue
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} tensors have no COCO match (only the keypoint branch cv4 may "
            f"be reinitialised): {mismatched[:5]}"
        )
    transferred = backbone_transferred + head_transferred
    if not transferred:
        raise RuntimeError(f"no tensors transferred from {source}")

    report = {
        "coco_weights": str(source),
        "student_config": str(getattr(student.model, "yaml_file", "")),
        "head_prefix": head_prefix,
        "transferred": len(transferred),
        "backbone_transferred": len(backbone_transferred),
        "head_transferred": len(head_transferred),
        "reinitialised": len(reinitialised),
        "mismatched": mismatched,
        "transferred_names": transferred,
        "head_transferred_names": head_transferred,
        "reinitialised_names": reinitialised,
        "teacher_nc": int(teacher.model.model[-1].nc),
        "teacher_kpt_shape": [int(value) for value in teacher.model.model[-1].kpt_shape],
        "student_nc": int(student.model.model[-1].nc),
        "student_kpt_shape": [int(value) for value in student.model.model[-1].kpt_shape],
        "student_parameters": int(
            sum(parameter.numel() for parameter in student.model.parameters())
        ),
    }
    print(
        f"transferred {report['transferred']} tensors from {source} "
        f"({report['backbone_transferred']} backbone/neck, {report['head_transferred']} Pose head: "
        f"box/class/dfl); reinitialised {report['reinitialised']} keypoint-branch tensors "
        f"({report['teacher_kpt_shape']} -> {report['student_kpt_shape']}, "
        f"nc {report['teacher_nc']} -> {report['student_nc']})"
    )
    return report


# ---------------------------------------------------------------------------
# Artifacts and metrics
# ---------------------------------------------------------------------------


def publish_artifacts(run_dir: Path, published: Path) -> dict[str, str]:
    """Copy ``best.pt``, ``last.pt`` and ``results.csv`` into ``published``.

    Returns:
        ``{filename: published path}`` for the files that exist.
    """

    source = Path(run_dir)
    target = Path(published)
    target.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for filename in ("best.pt", "last.pt"):
        path = source / "weights" / filename
        if path.is_file():
            shutil.copyfile(path, target / filename)
            artifacts[filename] = str(target / filename)
    results = source / "results.csv"
    if results.is_file():
        shutil.copyfile(results, target / "results.csv")
        artifacts["results.csv"] = str(target / "results.csv")
    return artifacts


def build_metrics_payload(
    entry: VariantEntry,
    config: StudyConfig,
    *,
    artifacts: Mapping[str, str],
    transfer: Mapping[str, Any] | None,
    results: Mapping[str, float] | None = None,
    results_full: Mapping[str, float] | None = None,
    speed: Mapping[str, float] | None = None,
    onnx: str | None = None,
    versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the ``metrics_fp32.json`` payload of one variant.

    ``metrics`` always has the ``half`` (half-body val, the primary metric) and
    ``full`` (full-frame val) sections; a section is ``None`` when that split
    was not evaluated so missing evidence never reads as a pass.
    """

    metrics: dict[str, Any] | None = None
    if results is not None or results_full is not None:
        metrics = {
            "half": {key: float(value) for key, value in results.items()} if results else None,
            "full": (
                {key: float(value) for key, value in results_full.items()} if results_full else None
            ),
        }
    payload: dict[str, Any] = {
        **entry.to_dict(),
        "weights": artifacts.get("best.pt"),
        "onnx": onnx,
        "dataset": str(config.data),
        "mixed_data": str(config.mixed_data),
        "full_data": str(config.full_data) if config.full_data is not None else None,
        "seed": config.seed,
        "epochs": config.epochs,
        "batch": config.batch,
        "schema": config.schema,
        "augmentations": dict(config.augmentations),
        "versions": dict(versions or {}),
        "transfer": dict(transfer) if transfer is not None else None,
        "metrics": metrics,
        "speed_ms": {key: float(value) for key, value in speed.items()} if speed else None,
    }
    return payload


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_variant(entry: VariantEntry, config: StudyConfig) -> dict[str, Any]:
    """Train one matrix row, publish its artifacts and score the FP32 model.

    Args:
        entry: Activation, square training size and deployment input.
        config: Shared paths, schedule and augmentation.

    Returns:
        ``{"variant", "run_dir", "artifacts"}`` for the session summary.
    """

    import torch
    import ultralytics
    from ultralytics import YOLO

    from cpsa_ml.onnx_check import export_onnx
    from cpsa_ml.train import parse_device

    config_path = model_yaml_path("n", entry.activation)
    if config_path.name != entry.config_name:
        raise RuntimeError(f"unexpected config name {config_path.name}")

    mixed = build_mixed_data_yaml(
        config.data, config.full_data, config.mixed_data, schema=config.schema
    )
    val_full = mixed.get("val_full")

    run_name = f"{config.run_prefix}-{entry.name}"
    run_dir = config.project / run_name
    published = config.publish / entry.name
    published.mkdir(parents=True, exist_ok=True)
    transfer_path = published / "transfer_report.json"

    torch.manual_seed(config.seed)
    resume = config.resume.get(entry.name)
    if resume is not None:
        checkpoint = Path(resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint for {entry.name} not found: {checkpoint}")
        print(f"[{entry.name}] resuming from {checkpoint}")
        model = YOLO(str(checkpoint), task="pose", verbose=False)
        model.train(resume=str(checkpoint))
        # `/kaggle/working` does not survive between sessions: the transfer
        # report of the interrupted run travels next to the published last.pt.
        transfer = None
        for candidate in (transfer_path, checkpoint.parent / transfer_path.name):
            if candidate.is_file():
                transfer = json.loads(candidate.read_text(encoding="utf-8"))
                if candidate != transfer_path:
                    transfer_path.write_text(
                        json.dumps(transfer, indent=2) + "\n", encoding="utf-8"
                    )
                break
    else:
        print(f"[{entry.name}] building {config_path} at imgsz={entry.train_imgsz}")
        from cpsa_ml.models import load_yolo

        model = load_yolo(config.schema, variant="n", activation=entry.activation)
        transfer = transfer_coco_weights(config.coco_weights, model)
        transfer_path.write_text(json.dumps(transfer, indent=2) + "\n", encoding="utf-8")
        model.train(
            data=str(config.mixed_data),
            epochs=config.epochs,
            batch=config.batch,
            imgsz=entry.train_imgsz,
            device=parse_device(config.device),
            project=str(config.project),
            name=run_name,
            seed=config.seed,
            deterministic=True,
            rect=False,
            workers=config.workers,
            patience=config.patience,
            amp=config.amp,
            val=True,
            plots=False,
            exist_ok=False,
            **dict(config.augmentations),
        )

    # Ultralytics may have used a suffixed directory (name2, name3, ...) when a
    # previous run exists; collect from the directory it actually used.
    run_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", run_dir))
    artifacts = publish_artifacts(run_dir, published)

    results: dict[str, float] | None = None
    results_full: dict[str, float] | None = None
    speed: dict[str, float] | None = None
    onnx_path: str | None = None
    best = Path(artifacts.get("best.pt", published / "best.pt"))
    if best.is_file():
        trained = YOLO(str(best), task="pose", verbose=False)
        metrics = trained.val(
            data=str(config.mixed_data),
            imgsz=[entry.deploy_size[0], entry.deploy_size[1]],
            split="val",
            conf=config.val_conf,
            iou=config.val_iou,
            plots=False,
            verbose=False,
            device=config.val_device,
        )
        results = metrics.results_dict
        speed = metrics.speed
        if val_full:
            full_metrics = trained.val(
                data=str(config.mixed_data),
                imgsz=[entry.deploy_size[0], entry.deploy_size[1]],
                split="val_full",
                conf=config.val_conf,
                iou=config.val_iou,
                plots=False,
                verbose=False,
                device=config.val_device,
            )
            results_full = full_metrics.results_dict
        exported = export_onnx(
            trained,
            published / "deploy.onnx",
            imgsz=entry.deploy_size,
            opset=17,
            device="cpu",
        )
        onnx_path = str(exported)
        artifacts["deploy.onnx"] = onnx_path

    payload = build_metrics_payload(
        entry,
        config,
        artifacts=artifacts,
        transfer=transfer,
        results=results,
        results_full=results_full,
        speed=speed,
        onnx=onnx_path,
        versions={"ultralytics": ultralytics.__version__, "torch": torch.__version__},
    )
    (published / "metrics_fp32.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    artifacts["metrics_fp32.json"] = str(published / "metrics_fp32.json")
    if transfer_path.is_file():
        artifacts["transfer_report.json"] = str(transfer_path)
    return {"variant": entry.name, "run_dir": str(run_dir), "artifacts": artifacts}


def train_study(config: StudyConfig) -> list[dict[str, Any]]:
    """Train every row of ``config.variants`` in order.

    Returns:
        One summary per variant, as returned by :func:`train_variant`.
    """

    summaries: list[dict[str, Any]] = []
    for entry in config.variants:
        summaries.append(train_variant(entry, config))
    for summary in summaries:
        print(f"{summary['variant']} -> {summary['run_dir']}")
        for kind, path in summary["artifacts"].items():
            print(f"  {kind:22s} {path}")
    return summaries
