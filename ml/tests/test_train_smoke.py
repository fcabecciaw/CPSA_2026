"""Slow one-epoch CPU training smoke test (opt in with CPSA_ML_SLOW=1)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cpsa_ml.datasets.halfbust import BuildConfig, build_halfbust_dataset
from cpsa_ml.train import TrainConfig, parse_device, train


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("CPSA_ML_SLOW") != "1",
    reason="set CPSA_ML_SLOW=1 to run the one-epoch training smoke test",
)
def test_one_epoch_cpu_smoke(tmp_path: Path, synthetic_coco: tuple[Path, Path]) -> None:
    coco_root, wholebody_json = synthetic_coco
    data_dir = tmp_path / "dataset"
    build_halfbust_dataset(
        BuildConfig(
            coco_root=coco_root,
            wholebody_json=wholebody_json,
            split="train",
            schema="halfbust62",
            out=data_dir,
            max_images=1,
        )
    )
    checkpoint = train(
        TrainConfig(
            data=data_dir / "data.yaml",
            schema="halfbust62",
            variant="n",
            activation="hswish",
            epochs=1,
            batch=1,
            imgsz=320,
            device="cpu",
            project=tmp_path / "runs",
            name="smoke",
            workers=0,
            patience=0,
            amp=False,
            val=False,
            plots=False,
            exist_ok=True,
            extra={"verbose": False},
        )
    )
    assert checkpoint.is_file()
    assert checkpoint.suffix == ".pt"


def test_parse_device() -> None:
    assert parse_device("cpu") == "cpu"
    assert parse_device("0,1") == [0, 1]
    assert parse_device([0, 1]) == [0, 1]
