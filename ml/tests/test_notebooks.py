"""Notebook and kernel-metadata validation (no Kaggle credentials needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = ("01_build_halfbust", "03_train_fp32", "04_variant_study", "05_qat")

#: Kaggle datasets attached to each notebook.
EXPECTED_DATASETS = {
    "01_build_halfbust": ["awsaf49/coco-2017-dataset"],
    "03_train_fp32": ["fabiocabeccia/cpsa-halfbust"],
    "04_variant_study": [
        "fabiocabeccia/cpsa-halfbust",
        "fabiocabeccia/cpsa-checkpoints",
    ],
    "05_qat": ["fabiocabeccia/cpsa-halfbust", "fabiocabeccia/cpsa-checkpoints"],
}

#: Notebooks that train on a GPU (notebook 01 only builds the dataset).
GPU_NOTEBOOKS = ("03_train_fp32", "04_variant_study", "05_qat")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebooks_are_valid_nbformat_json() -> None:
    for stem in NOTEBOOKS:
        path = NOTEBOOK_DIR / f"{stem}.ipynb"
        notebook = _load(path)
        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] >= 5
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        assert notebook["cells"]
        for cell in notebook["cells"]:
            assert cell["cell_type"] in {"code", "markdown"}
            assert isinstance(cell["source"], list)
            assert cell["metadata"] == {}
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []

        # Validate with nbformat when it is installed; the structural checks
        # above keep the default dev extra self-contained.
        try:
            import nbformat
        except ImportError:  # pragma: no cover - optional dependency
            nbformat = None
        if nbformat is not None:
            nbformat.validate(nbformat.reads(json.dumps(notebook), as_version=4))


def test_first_code_cell_pins_and_installs_cpsa_ml() -> None:
    for stem in NOTEBOOKS:
        notebook = _load(NOTEBOOK_DIR / f"{stem}.ipynb")
        first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        source = "".join(first_code["source"])
        assert "COMMIT_SHA" in source
        assert "subdirectory=ml" in source
        assert "github.com/fcabecciaw/CPSA_2026" in source
        assert "pip" in source


@pytest.mark.parametrize(
    ("stem", "entry_point"),
    [
        ("01_build_halfbust", "build_halfbust_dataset"),
        ("03_train_fp32", "train"),
        ("04_variant_study", "train_study"),
        ("05_qat", "quantize_model"),
    ],
)
def test_notebooks_are_thin_wrappers(stem: str, entry_point: str) -> None:
    notebook = _load(NOTEBOOK_DIR / f"{stem}.ipynb")
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert entry_point in source
    assert "def " not in source  # no local reimplementation


def test_notebook_04_uses_the_variant_study_module() -> None:
    notebook = _load(NOTEBOOK_DIR / "04_variant_study.ipynb")
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "from cpsa_ml.variant_study import" in source
    assert "DEFAULT_VARIANTS" in source
    assert "StudyConfig" in source


def test_kernel_metadata_is_complete_and_valid() -> None:
    for stem in NOTEBOOKS:
        folder = NOTEBOOK_DIR / stem
        metadata = _load(folder / "kernel-metadata.json")
        assert metadata["id"].startswith("fabiocabeccia/")
        assert metadata["code_file"] == f"{stem}.ipynb"
        assert metadata["language"] == "python"
        assert metadata["kernel_type"] == "notebook"
        assert metadata["is_private"] is True
        assert metadata["enable_internet"] is True
        assert metadata["dataset_sources"] == EXPECTED_DATASETS[stem]
        assert metadata["competition_sources"] == []
        assert metadata["kernel_sources"] == []
        if stem in GPU_NOTEBOOKS:
            assert metadata["enable_gpu"] is True
            assert metadata["machine_shape"] == "NvidiaTeslaT4"
        else:
            assert metadata["enable_gpu"] is False
            assert "machine_shape" not in metadata
        assert (folder / metadata["code_file"]).is_file()


def test_push_folder_notebook_matches_canonical() -> None:
    for stem in NOTEBOOKS:
        canonical = _load(NOTEBOOK_DIR / f"{stem}.ipynb")
        pushed = _load(NOTEBOOK_DIR / stem / f"{stem}.ipynb")
        assert canonical == pushed
