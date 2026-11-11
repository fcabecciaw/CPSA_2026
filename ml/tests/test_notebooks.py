"""Notebook and kernel-metadata validation (no Kaggle credentials needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = (
    "01_build_halfbust",
    "02_build_fullframe",
    "03_train_fp32",
    "04_variant_study",
    "05_qat",
    "07_hand_pose",
)

#: Kaggle datasets attached to each notebook.
EXPECTED_DATASETS = {
    "01_build_halfbust": ["awsaf49/coco-2017-dataset"],
    "02_build_fullframe": [
        "awsaf49/coco-2017-dataset",
        "fabiocabeccia/cpsa-halfbust",
    ],
    "03_train_fp32": [
        "fabiocabeccia/cpsa-halfbust",
        "fabiocabeccia/cpsa-fullframe",
    ],
    "04_variant_study": [
        "fabiocabeccia/cpsa-halfbust",
        "fabiocabeccia/cpsa-fullframe",
        "fabiocabeccia/cpsa-checkpoints",
    ],
    "05_qat": ["fabiocabeccia/cpsa-halfbust", "fabiocabeccia/cpsa-checkpoints"],
    # Notebook 07 downloads HaGRID and hand-keypoints itself and needs no input.
    "07_hand_pose": [],
}

#: Notebooks that train on a GPU (notebook 01 only builds the dataset).
GPU_NOTEBOOKS = ("03_train_fp32", "04_variant_study", "05_qat", "07_hand_pose")


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
        ("02_build_fullframe", "build_fullframe_dataset"),
        ("03_train_fp32", "train"),
        ("04_variant_study", "train_study"),
        ("05_qat", "quantize_model"),
        ("07_hand_pose", "build_hand_dataset"),
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


def test_notebook_02_builds_the_full_frame_set_with_the_module() -> None:
    notebook = _load(NOTEBOOK_DIR / "02_build_fullframe.ipynb")
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "from cpsa_ml.datasets.fullframe import" in source
    assert "build_fullframe_dataset" in source
    assert "half_train_count" in source
    assert "/kaggle/working/fullframe62" in source


@pytest.mark.parametrize("stem", ["03_train_fp32", "04_variant_study"])
def test_training_notebooks_use_the_full_frame_dataset(stem: str) -> None:
    notebook = _load(NOTEBOOK_DIR / f"{stem}.ipynb")
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert '"cpsa-fullframe" in str(p)' in source  # located by content under /kaggle/input
    assert '"cpsa-halfbust" in str(p)' in source
    assert "full_data=FULL_DATA" in source or "build_mixed_data_yaml" in source


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


def _source(stem: str) -> str:
    notebook = _load(NOTEBOOK_DIR / f"{stem}.ipynb")
    return "".join("".join(cell["source"]) for cell in notebook["cells"])


@pytest.mark.parametrize("stem", ["03_train_fp32", "04_variant_study"])
def test_training_notebooks_expose_the_curation_knobs(stem: str) -> None:
    source = _source(stem)
    # Notebook 04 stages the manifest subset, so its floor is the 256 px of D1.
    assert f"MIN_HEIGHT = {256 if stem == '04_variant_study' else 128}" in source
    assert "HAND_REPEAT = 2" in source
    assert "FACE_REPEAT = 1" in source
    assert "LABELS_ROOT" in source
    assert "cpsa-halfbust-pl" in source
    assert "labels_root=LABELS_ROOT" in source
    assert "EXTRA_DATA" in source
    assert "cpsa-hagrid62" in source
    # Notebook 03 curates explicitly; notebook 04 lets train_study prepare the data.
    if stem == "03_train_fp32":
        assert "curate_train_list" in source


def test_notebook_04_defaults_to_the_urgent_run() -> None:
    source = _source("04_variant_study")
    assert "VARIANTS = [DEFAULT_VARIANTS[2]]" in source  # lrelu 512 -> deploy 384x512
    assert "EPOCHS = 30" in source
    assert "the main run uses 120" in source
    assert "albumentations==" in source  # motion blur is installed best-effort
    assert "min_height=MIN_HEIGHT" in source
    assert "extra_data=EXTRA_DATA" in source


def test_notebook_04_stages_the_manifest_and_the_teacher_archive() -> None:
    source = _source("04_variant_study")
    assert "from cpsa_ml.datasets.staging import" in source
    assert 'include=load_manifest("halfbust62_h256")' in source
    assert 'HAGRID_TAR = next(iter(find_inputs(INPUT, "hagrid62.tar")), None)' in source
    assert 'stage_archive(HAGRID_TAR, Path("/tmp/hagrid62"))' in source
    assert "MIN_HEIGHT = 256" in source
    # The markdown documents the mix, the manifest and notebook 06's attachment.
    assert "16,828" in source and "42,327" in source
    assert "Add Input" in source and "06_teacher_hagrid" in source
    assert "staging: the manifest keeps" in source
    assert "staging: teacher dataset ready" in source
    # The teacher archive is required unless the operator opts out explicitly.
    assert "ALLOW_NO_HAGRID = False" in source
    assert "elif not ALLOW_NO_HAGRID:" in source
    assert "hagrid62.tar is not attached" in source


def test_notebook_07_builds_and_trains_the_hand_model() -> None:
    source = _source("07_hand_pose")
    assert "from cpsa_ml.datasets.hands import" in source
    assert "HagridSource" in source
    assert "UltralyticsHandSource" in source
    assert "USE_ULTRALYTICS" in source  # the CC BY-NC-SA source stays optional
    assert 'schema="hand21"' in source
    assert 'ACTIVATION = "lrelu"' in source
    assert 'config_path("yolov8n-hand-lrelu.yaml")' in source
    assert "imgsz=IMGSZ" in source and "IMGSZ = 192" in source
    assert "CC BY-NC-SA" in source  # the non-commercial warning stays visible
    assert "cpsa_ml.pseudolabel" in source  # recorded frames can extend the set


def test_notebook_03_uses_the_curated_list_in_the_mixed_dataset() -> None:
    source = _source("03_train_fp32")
    assert "curated_train=CURATION.train_list" in source
    assert "val_h160=CURATION.val_h160_list" in source
