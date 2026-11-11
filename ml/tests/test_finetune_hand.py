"""Fine-tune CLI tests: recipe wiring, artifact copying and ONNX export (Ultralytics mocked)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cpsa_ml.finetune_hand import (
    DEFAULT_EXPORT_IMGSZ,
    FineTuneConfig,
    build_train_args,
    finetune,
    main,
)


def make_config(tmp_path: Path, **kwargs: Any) -> FineTuneConfig:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"start")
    data = tmp_path / "data.yaml"
    data.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    values: dict[str, Any] = {
        "weights": weights,
        "data": data,
        "out": tmp_path / "run",
    }
    values.update(kwargs)
    return FineTuneConfig(**values)


class FakeTrainer:
    """The bits of ``ultralytics.engine.trainer.BaseTrainer`` that finetune reads."""

    def __init__(self, save_dir: Path, *, write_results: bool = True) -> None:
        self.save_dir = save_dir
        self.best = save_dir / "weights" / "best.pt"
        self.last = save_dir / "weights" / "last.pt"
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        self.best.write_bytes(b"weights")
        self.last.write_bytes(b"weights")
        if write_results:
            (save_dir / "results.csv").write_text("epoch,metrics\n0,0.1\n", encoding="utf-8")


class FakeYolo:
    """Record the train arguments and write the Ultralytics output tree."""

    def __init__(self, weights: Path, *, write_results: bool = True) -> None:
        self.weights = weights
        self.write_results = write_results
        self.train_kwargs: dict[str, Any] | None = None
        self.trainer: FakeTrainer | None = None

    def train(self, **kwargs: Any) -> None:
        self.train_kwargs = kwargs
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        self.trainer = FakeTrainer(save_dir, write_results=self.write_results)


class FakeExporter:
    """Record the export call and write a placeholder graph."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, dict[str, Any]]] = []

    def __call__(self, model: Any, output: str | Path, **kwargs: Any) -> Path:
        path = Path(output)
        path.write_bytes(b"onnx")
        self.calls.append((Path(model), path, kwargs))
        return path


def test_defaults_follow_the_board_finetune_recipe() -> None:
    config = FineTuneConfig(weights=Path("w.pt"), data=Path("d.yaml"), out=Path("out"))

    assert config.epochs == 20
    assert config.imgsz == 320
    assert config.batch == 16
    assert config.device == "cpu"
    assert config.lr0 == 0.002
    assert config.mosaic == 0.5
    assert config.degrees == 180.0
    assert config.fliplr == 0.5
    assert (config.hsv_h, config.hsv_s, config.hsv_v) == (0.0, 0.0, 0.3)
    assert config.export_imgsz == (256, 320) == DEFAULT_EXPORT_IMGSZ


def test_train_args_pin_the_recipe(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    args = build_train_args(config)

    assert args["data"] == str(config.data)
    assert (args["epochs"], args["imgsz"], args["batch"]) == (20, 320, 16)
    assert args["device"] == "cpu"
    assert args["lr0"] == 0.002
    assert args["mosaic"] == 0.5
    assert args["close_mosaic"] == 0
    assert args["degrees"] == 180.0
    assert args["fliplr"] == 0.5
    assert (args["hsv_h"], args["hsv_s"], args["hsv_v"]) == (0.0, 0.0, 0.3)
    assert args["pretrained"] is True
    assert args["resume"] is False
    assert args["amp"] is False
    assert args["project"] == str(config.out)
    assert args["name"] == "finetune"


def test_multi_gpu_device_is_passed_as_a_list(tmp_path: Path) -> None:
    assert build_train_args(make_config(tmp_path, device="0,1"))["device"] == [0, 1]


def test_finetune_copies_artifacts_and_exports_the_dpu_graph(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    yolo = FakeYolo(config.weights)
    exporter = FakeExporter()
    loaded: list[Path] = []

    def factory(weights: Path) -> FakeYolo:
        loaded.append(weights)
        return yolo

    result = finetune(config, yolo_factory=factory, exporter=exporter)

    assert loaded == [config.weights]
    assert yolo.train_kwargs is not None
    assert yolo.train_kwargs["data"] == str(config.data)
    assert (config.out / "best.pt").read_bytes() == b"weights"
    assert (config.out / "results.csv").read_text(encoding="utf-8").startswith("epoch")
    assert result.best_pt == config.out / "best.pt"
    assert result.results_csv == config.out / "results.csv"
    assert result.best_onnx == config.out / "best.onnx"
    assert result.best_onnx.read_bytes() == b"onnx"
    assert exporter.calls == [
        (
            config.out / "best.pt",
            config.out / "best.onnx",
            {
                "imgsz": (256, 320),
                "device": "cpu",
            },
        )
    ]


def test_onnx_export_runs_on_the_cpu_for_a_multi_gpu_train(tmp_path: Path) -> None:
    config = make_config(tmp_path, device="0,1")
    exporter = FakeExporter()

    finetune(config, yolo_factory=lambda weights: FakeYolo(weights), exporter=exporter)

    assert exporter.calls[0][2]["device"] == "cpu"


def test_single_gpu_export_keeps_the_device(tmp_path: Path) -> None:
    config = make_config(tmp_path, device="0")
    exporter = FakeExporter()

    finetune(config, yolo_factory=lambda weights: FakeYolo(weights), exporter=exporter)

    assert exporter.calls[0][2]["device"] == "0"


def test_missing_weights_are_rejected(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.weights.unlink()

    with pytest.raises(FileNotFoundError, match="weights not found"):
        finetune(config, yolo_factory=lambda weights: FakeYolo(weights), exporter=FakeExporter())


def test_missing_data_yaml_is_rejected(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.data.unlink()

    with pytest.raises(FileNotFoundError, match="data.yaml not found"):
        finetune(config, yolo_factory=lambda weights: FakeYolo(weights), exporter=FakeExporter())


def test_missing_results_csv_is_reported(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="results.csv"):
        finetune(
            config,
            yolo_factory=lambda weights: FakeYolo(weights, write_results=False),
            exporter=FakeExporter(),
        )


def test_invalid_sizes_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="multiples? of 32"):
        make_config(tmp_path, export_imgsz=(300, 320))
    with pytest.raises(ValueError, match="multiples? of 32"):
        make_config(tmp_path, imgsz=300)
    with pytest.raises(ValueError, match="lr0"):
        make_config(tmp_path, lr0=0.0)


def test_cli_wires_the_arguments(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    exporter = FakeExporter()
    yolo = FakeYolo(config.weights)

    code = main(
        [
            "--weights",
            str(config.weights),
            "--data",
            str(config.data),
            "--out",
            str(config.out),
            "--epochs",
            "3",
            "--imgsz",
            "256",
            "--batch",
            "4",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--name",
            "smoke",
        ],
        yolo_factory=lambda weights: yolo,
        exporter=exporter,
    )

    assert code == 0
    assert yolo.train_kwargs is not None
    assert (yolo.train_kwargs["epochs"], yolo.train_kwargs["imgsz"]) == (3, 256)
    assert yolo.train_kwargs["batch"] == 4
    assert yolo.train_kwargs["seed"] == 7
    assert yolo.train_kwargs["name"] == "smoke"
    assert exporter.calls[0][2]["imgsz"] == (256, 320)
    assert (config.out / "best.pt").is_file()
    assert (config.out / "best.onnx").is_file()
