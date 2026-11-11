"""Notebook 06 (HaGRID teacher labels on the Kaggle GPU) structure tests.

Mirrors the helpers of ``test_notebooks.py`` and adds the cases specific to
notebook 06: the CUDA preflight, the CPU fallback message, the HaGRID download,
the auto-label command and the shipped archive.  The notebook itself stays a
thin wrapper around ``cpsa_ml.pseudolabel``.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
STEM = "06_teacher_hagrid"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(stem: str = STEM) -> str:
    notebook = _load(NOTEBOOK_DIR / f"{stem}.ipynb")
    return "".join("".join(cell["source"]) for cell in notebook["cells"])


def test_notebook_06_is_valid_nbformat_json() -> None:
    notebook = _load(NOTEBOOK_DIR / f"{STEM}.ipynb")
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["cells"]
    assert [cell["cell_type"] for cell in notebook["cells"]][0] == "markdown"
    assert [cell["cell_type"] for cell in notebook["cells"]][-1] == "markdown"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["source"], list)
        assert cell["metadata"] == {}
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []

    # Validate with nbformat when it is installed; the structural checks above
    # keep the default dev extra self-contained.
    try:
        import nbformat
    except ImportError:  # pragma: no cover - optional dependency
        nbformat = None
    if nbformat is not None:
        nbformat.validate(nbformat.reads(json.dumps(notebook), as_version=4))


def test_first_code_cell_pins_and_installs_cpsa_ml_with_the_gpu_runtime() -> None:
    notebook = _load(NOTEBOOK_DIR / f"{STEM}.ipynb")
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "".join(first_code["source"])
    assert "COMMIT_SHA" in source
    assert "subdirectory=ml" in source
    assert "github.com/fcabecciaw/CPSA_2026" in source
    assert "pip" in source
    assert "cpsa-ml[pseudolabel]" in source
    # The CPU onnxruntime of the extra is removed before the CUDA build goes in.
    assert 'metadata.version("onnxruntime")' in source
    assert "uninstall" in source
    assert "onnxruntime-gpu[cuda,cudnn]" in source
    assert "rtmlib==0.0.16" in source


def test_notebook_06_confirms_cuda_before_labeling() -> None:
    source = _source()
    assert "import onnxruntime as ort" in source
    assert "ort.preload_dlls()" in source
    assert 'assert "CUDAExecutionProvider" in PROVIDERS' in source
    assert "ort.InferenceSession" in source
    assert "model.ir_version = 10" in source  # onnx 1.23 defaults to IR 14
    assert 'providers=["CUDAExecutionProvider"]' in source
    assert 'session_providers[0].startswith("CUDA")' in source
    assert "RuntimeError" in source
    assert "--device cpu --teacher rtmw-l" in source  # the D3 fallback
    # The preflight runs before the labeler builds its own sessions.
    assert source.index('"CUDAExecutionProvider" in PROVIDERS') < source.index('"rtmw-x"')


def test_notebook_06_downloads_the_hagrid_sample_to_tmp() -> None:
    source = _source()
    assert "curl" in source
    assert "-sSL" in source
    assert "--retry" in source
    assert '"-C", "-"' in source
    assert "cj-mills/hagrid-sample-30k-384p" in source
    assert 'Path("/tmp/hagrid")' in source
    assert "for _, _, files in os.walk(HAGRID_ROOT)" in source
    assert "hagrid_30k" in source


def test_notebook_06_labels_through_the_module_with_streamed_output() -> None:
    source = _source()
    assert '"cpsa_ml.pseudolabel"' in source
    assert 'Path("/tmp/hagrid_pl")' in source
    assert 'DEVICE = "cuda"' in source
    assert '"--teacher"' in source and '"rtmw-x"' in source
    assert '"--no-hands"' in source
    assert '"--val-fraction"' in source and '"0.1"' in source
    assert "Popen" in source
    assert "stdout=subprocess.PIPE" in source
    assert "for line in process.stdout" in source


def test_notebook_06_asserts_the_labeled_frame_count() -> None:
    source = _source()
    assert "stats.json" in source
    assert 'for split in ("train", "val")' in source
    assert "labeled >= 25000" in source


def test_notebook_06_renders_samples_and_writes_the_archive() -> None:
    source = _source()
    assert 'Path("/kaggle/working/samples")' in source
    assert "random.Random(0).shuffle(frames)" in source
    assert "frames[:16]" in source
    assert "HALFBUST_62.names" in source
    assert 'Path("/kaggle/working/hagrid62.tar")' in source
    assert 'tarfile.open(TARBALL, "w")' in source
    assert "recursive=False" in source
    assert '{"data.yaml", "images/train", "images/val", "labels/train", "labels/val"}' in source


def test_kernel_metadata_is_complete_and_valid() -> None:
    folder = NOTEBOOK_DIR / STEM
    metadata = _load(folder / "kernel-metadata.json")
    assert metadata["id"].startswith("fabiocabeccia/")
    assert metadata["code_file"] == f"{STEM}.ipynb"
    assert metadata["language"] == "python"
    assert metadata["kernel_type"] == "notebook"
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == []
    assert metadata["competition_sources"] == []
    assert metadata["kernel_sources"] == []
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert (folder / metadata["code_file"]).is_file()


def test_push_folder_notebook_matches_canonical() -> None:
    canonical = NOTEBOOK_DIR / f"{STEM}.ipynb"
    pushed = NOTEBOOK_DIR / STEM / f"{STEM}.ipynb"
    assert canonical.read_bytes() == pushed.read_bytes()


def test_notebook_06_documents_licenses_and_the_notebook_04_handoff() -> None:
    notebook = _load(NOTEBOOK_DIR / f"{STEM}.ipynb")
    markdown = "".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "CC BY-SA 4.0" in markdown
    assert "Apache-2.0" in markdown
    assert "Notebook output" in markdown
    assert "notebook 04" in markdown
    assert "stage_archive" in markdown
    assert "data.yaml" in markdown
    assert "images/{train,val}" in markdown
    assert "labels/{train,val}" in markdown
