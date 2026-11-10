"""Thin, resumable Ultralytics training wrapper for the CPSA pose models.

Training runs on Kaggle 2x T4 with DDP (``device="0,1"``) and must be resumable
because Kaggle sessions are capped at 12 hours.  Ultralytics trains with square
letterboxed inputs; the deployment graph is exported and audited separately at
the camera's native 480x640 input.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cpsa_ml.models import load_yolo

__all__ = ["TrainConfig", "main", "parse_device", "train"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Training parameters passed to ``ultralytics.YOLO.train``."""

    data: Path
    schema: str = "halfbust62"
    variant: str = "n"
    activation: str = "hswish"
    epochs: int = 100
    batch: int = 32
    imgsz: int | tuple[int, int] = 640
    device: str | Sequence[int] = "cpu"
    project: Path = Path("runs")
    name: str = "cpsa-pose"
    resume: Path | None = None
    seed: int = 0
    workers: int = 4
    patience: int = 50
    amp: bool = True
    val: bool = True
    plots: bool = False
    exist_ok: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", Path(self.data))
        object.__setattr__(self, "project", Path(self.project))
        if self.resume is not None:
            object.__setattr__(self, "resume", Path(self.resume))
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch == 0:
            raise ValueError("batch must be non-zero")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")


def parse_device(device: str | Sequence[int]) -> str | list[int]:
    """Convert ``"0,1"`` to ``[0, 1]`` so Ultralytics uses DDP."""

    if isinstance(device, str):
        if "," in device:
            return [int(part.strip()) for part in device.split(",") if part.strip()]
        return device
    return [int(item) for item in device]


def train(config: TrainConfig) -> Path:
    """Run ``YOLO.train`` and return the best (or last) checkpoint path.

    ``resume`` is a path to a ``last.pt`` checkpoint written by a previous
    session.  Ultralytics restores the optimizer, epoch and training arguments
    from that checkpoint; changed arguments are ignored or warned about by
    Ultralytics, so keep the original data/model settings when resuming.
    """

    if config.resume is not None:
        from ultralytics import YOLO

        if not config.resume.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {config.resume}")
        yolo = YOLO(str(config.resume), verbose=False)
        resume: str | bool = str(config.resume)
    else:
        yolo = load_yolo(
            config.schema,
            variant=config.variant,
            activation=config.activation,
        )
        resume = False

    if isinstance(config.imgsz, (tuple, list)) and config.imgsz[0] != config.imgsz[1]:
        LOGGER.warning(
            "Ultralytics training letterboxes to the configured imgsz; the deployment graph "
            "is exported/audited separately at 480x640."
        )

    args: dict[str, Any] = {
        "data": str(config.data),
        "epochs": config.epochs,
        "batch": config.batch,
        "imgsz": list(config.imgsz) if isinstance(config.imgsz, tuple) else config.imgsz,
        "device": parse_device(config.device),
        "project": str(config.project),
        "name": config.name,
        "resume": resume,
        "seed": config.seed,
        "deterministic": True,
        "rect": False,
        "workers": config.workers,
        "patience": config.patience,
        "amp": config.amp,
        "val": config.val,
        "plots": config.plots,
        "exist_ok": config.exist_ok,
        "pretrained": False,
        "cache": False,
    }
    args.update(config.extra)
    yolo.train(**args)

    trainer = getattr(yolo, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics did not create a trainer")
    for candidate in (getattr(trainer, "best", None), getattr(trainer, "last", None)):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(f"no checkpoint was written under {config.project / config.name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--schema", default="halfbust62")
    parser.add_argument("--variant", choices=("n", "s"), default="n")
    parser.add_argument("--activation", choices=("hswish", "silu"), default="hswish")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help='"cpu", "0", or "0,1" for DDP')
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="cpsa-pose")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    checkpoint = train(
        TrainConfig(
            data=args.data,
            schema=args.schema,
            variant=args.variant,
            activation=args.activation,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            seed=args.seed,
            workers=args.workers,
            patience=args.patience,
            amp=not args.no_amp,
            plots=args.plots,
        )
    )
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
