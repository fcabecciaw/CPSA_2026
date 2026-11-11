"""Fine-tune the ``hand21`` model on board IR frames and export the DPU ONNX graph.

Starts from an existing checkpoint (the notebook-08 ``best.pt``) at a low
learning rate, with the augmentation recipe of the top-down hand model
(``mosaic=0.5``, ``degrees=180``, ``fliplr=0.5``, hue/saturation off) and runs on
the CPU by default, because the local ``ml/.venv`` torch is CPU-only::

    cd ml
    .venv/bin/python -m cpsa_ml.finetune_hand \\
        --weights build/models/hand_topdown/best.pt \\
        --data build/ir_pseudo/data.yaml \\
        --epochs 20 --imgsz 320 --batch 16 --device cpu \\
        --out build/hand_finetune

Outputs under ``--out``:

* ``best.pt`` — the fine-tuned checkpoint;
* ``best.onnx`` — fixed ``1x3x256x320`` graph at opset 17, via
  :func:`cpsa_ml.onnx_check.export_onnx`;
* ``results.csv`` — the Ultralytics training log.

``--device 0`` (or ``0,1``) trains on CUDA; the ONNX export always runs on the
CPU, because a multi-GPU export is not supported.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cpsa_ml.onnx_check import export_onnx
from cpsa_ml.train import parse_device

__all__ = [
    "DEFAULT_EXPORT_IMGSZ",
    "FineTuneConfig",
    "FineTuneResult",
    "build_train_args",
    "finetune",
    "main",
]

LOGGER = logging.getLogger(__name__)

#: Export size as ``(height, width)``; the deployed hand model input.
DEFAULT_EXPORT_IMGSZ = (256, 320)

DEFAULT_EPOCHS = 20
DEFAULT_IMGSZ = 320
DEFAULT_BATCH = 16
DEFAULT_LR0 = 0.002
DEFAULT_MOSAIC = 0.5
DEFAULT_DEGREES = 180.0
DEFAULT_FLIPLR = 0.5
DEFAULT_HSV_H = 0.0
DEFAULT_HSV_S = 0.0
DEFAULT_HSV_V = 0.3

YoloFactory = Callable[[Path], Any]
Exporter = Callable[..., Path]


@dataclass(frozen=True)
class FineTuneConfig:
    """Weights, data, training recipe and export of one fine-tune run.

    Args:
        weights: Starting checkpoint (a ``hand21`` ``best.pt``).
        data: ``data.yaml`` of the auto-labelled dataset.
        out: Output directory; ``best.pt``, ``best.onnx`` and ``results.csv``
            are written here and Ultralytics writes ``<out>/<name>/`` beside it.
        epochs: Training epochs.
        imgsz: Square training size; the hand model trains at 320.
        batch: Batch size; 16 fits the CPU run.
        device: ``"cpu"``, ``"0"`` or ``"0,1"`` for Ultralytics.
        lr0: Initial learning rate; low, because the run starts from weights.
        mosaic: Mosaic probability; halves the default to keep the real look.
        degrees: Rotation range; hands are seen from above at any angle.
        fliplr: Horizontal flip probability.
        hsv_h, hsv_s, hsv_v: Colour augmentation; IR frames are gray, so only
            brightness (``hsv_v``) is left on.
        seed: Training seed.
        workers: Data-loader workers.
        name: Ultralytics run name below ``out``.
        export_imgsz: ``(height, width)`` of the exported ONNX graph.
        exist_ok: Reuse ``<out>/<name>`` instead of failing.
    """

    weights: Path
    data: Path
    out: Path
    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_IMGSZ
    batch: int = DEFAULT_BATCH
    device: str = "cpu"
    lr0: float = DEFAULT_LR0
    mosaic: float = DEFAULT_MOSAIC
    degrees: float = DEFAULT_DEGREES
    fliplr: float = DEFAULT_FLIPLR
    hsv_h: float = DEFAULT_HSV_H
    hsv_s: float = DEFAULT_HSV_S
    hsv_v: float = DEFAULT_HSV_V
    seed: int = 0
    workers: int = 4
    name: str = "finetune"
    export_imgsz: tuple[int, int] = DEFAULT_EXPORT_IMGSZ
    exist_ok: bool = True

    def __post_init__(self) -> None:
        for field_name in ("weights", "data", "out"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch == 0:
            raise ValueError("batch must be non-zero")
        if self.imgsz <= 0 or self.imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        if not 0.0 < self.lr0 <= 1.0:
            raise ValueError("lr0 must be in (0, 1]")
        if not 0.0 <= self.mosaic <= 1.0:
            raise ValueError("mosaic must be in [0, 1]")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        height, width = (int(value) for value in self.export_imgsz)
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("export_imgsz must be positive multiples of 32")
        object.__setattr__(self, "export_imgsz", (height, width))


@dataclass(frozen=True)
class FineTuneResult:
    """Files produced by :func:`finetune`."""

    best_pt: Path
    best_onnx: Path
    results_csv: Path


def build_train_args(config: FineTuneConfig) -> dict[str, Any]:
    """Return the ``YOLO.train`` arguments of this run.

    The defaults of :class:`FineTuneConfig` pin the recipe; ``close_mosaic`` is
    off because a 20-epoch run would otherwise train the last half without the
    half-strength mosaic augmentation it asks for.
    """

    return {
        "data": str(config.data),
        "epochs": config.epochs,
        "batch": config.batch,
        "imgsz": config.imgsz,
        "device": parse_device(config.device),
        "project": str(config.out),
        "name": config.name,
        "exist_ok": config.exist_ok,
        "seed": config.seed,
        "workers": config.workers,
        "pretrained": True,
        "resume": False,
        "rect": False,
        "val": True,
        "plots": False,
        "cache": False,
        # CPU-only torch locally: AMP stays off so the run does not depend on a
        # CUDA build; a later CUDA run can enable it explicitly.
        "amp": False,
        "lr0": config.lr0,
        "mosaic": config.mosaic,
        "close_mosaic": 0,
        "degrees": config.degrees,
        "fliplr": config.fliplr,
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
    }


def _load_yolo(weights: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(weights), verbose=False)


def _export_device(device: str) -> str:
    """The ONNX export runs on the CPU; multi-GPU export is not supported."""

    return "cpu" if "," in device or device.strip().lower() == "cpu" else device


def _checkpoint(trainer: Any, run_dir: Path) -> Path:
    """Return the checkpoint Ultralytics saved for the run."""

    for candidate in (
        getattr(trainer, "best", None),
        getattr(trainer, "last", None),
        run_dir / "weights" / "best.pt",
        run_dir / "weights" / "last.pt",
    ):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(f"no checkpoint was written under {run_dir}")


def finetune(
    config: FineTuneConfig,
    *,
    yolo_factory: YoloFactory | None = None,
    exporter: Exporter = export_onnx,
) -> FineTuneResult:
    """Train from ``config.weights``, copy the artifacts and export the ONNX graph.

    Args:
        config: Weights, dataset, recipe and output paths.
        yolo_factory: Test seam returning a ``YOLO``-like object; defaults to
            ``ultralytics.YOLO(weights)``.
        exporter: Test seam; defaults to :func:`cpsa_ml.onnx_check.export_onnx`.

    Raises:
        FileNotFoundError: If the weights or the dataset YAML are missing, or
            Ultralytics wrote no checkpoint or ``results.csv``.
    """

    if not config.weights.is_file():
        raise FileNotFoundError(f"weights not found: {config.weights}")
    if not config.data.is_file():
        raise FileNotFoundError(f"data.yaml not found: {config.data}")
    config.out.mkdir(parents=True, exist_ok=True)

    yolo = (yolo_factory or _load_yolo)(config.weights)
    yolo.train(**build_train_args(config))

    trainer = getattr(yolo, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics did not create a trainer")
    run_dir = Path(getattr(trainer, "save_dir", config.out / config.name))
    checkpoint = _checkpoint(trainer, run_dir)

    best_pt = config.out / "best.pt"
    shutil.copy2(checkpoint, best_pt)
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Ultralytics wrote no results.csv under {run_dir}")
    shutil.copy2(results_csv, config.out / "results.csv")

    best_onnx = Path(
        exporter(
            best_pt,
            config.out / "best.onnx",
            imgsz=config.export_imgsz,
            device=_export_device(config.device),
        )
    )
    LOGGER.info("fine-tuned checkpoint: %s", best_pt)
    LOGGER.info("exported graph: %s", best_onnx)
    return FineTuneResult(
        best_pt=best_pt, best_onnx=best_onnx, results_csv=config.out / "results.csv"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="starting best.pt")
    parser.add_argument("--data", type=Path, required=True, help="dataset data.yaml")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default="cpu", help='"cpu", "0" or "0,1"')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="finetune", help="Ultralytics run name")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    yolo_factory: YoloFactory | None = None,
    exporter: Exporter = export_onnx,
) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = FineTuneConfig(
        weights=args.weights,
        data=args.data,
        out=args.out,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        name=args.name,
    )
    result = finetune(config, yolo_factory=yolo_factory, exporter=exporter)
    print(result.best_pt)
    print(result.best_onnx)
    print(result.results_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
