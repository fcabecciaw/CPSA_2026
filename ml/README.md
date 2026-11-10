# `cpsa_ml` — vision dataset and training toolchain

`ml/` is a separate Python package (`cpsa-ml`, import name `cpsa_ml`) outside
the `cpsa` wheel. It holds the keypoint schema, the COCO-WholeBody -> YOLO-pose
half-bust builder, DPU-compatible model definitions, the resumable Kaggle
training wrapper, ONNX operator checks and the optional QAT path.

The package is importable from the repository or from a pinned Git commit:

```bash
cd ml
uv sync --extra dev
uv run pytest -q
```

Optional extras:

```bash
uv sync --extra dev --extra qat       # Brevitas QAT
uv sync --extra dev --extra teacher   # rtmlib + onnxruntime (Task 5b)
```

## Pinned versions

| Package | Pin | Why |
|---|---|---|
| Python | `3.10` (`.python-version`) | matches the KV260 runtime and the ruff target |
| `torch` / `torchvision` | `>=2.2` / `>=0.17` | resolved locally from the CPU index; Kaggle's `pip` gets CUDA builds |
| `ultralytics` | `8.4.203` | verified on PyPI; supports the top-level `activation:` YAML key |
| `brevitas` | `0.13.4` | verified `Int8WeightPerTensorFixedPoint`, `Int8ActPerTensorFixedPoint`, `QuantConv2d`, `QuantHardSwish` |
| `onnx` | `>=1.16,<2` | Ultralytics 8.4.203 export range |
| `rtmlib` | `0.0.16` | teacher extra only (Task 5b) |
| `numpy` | `>=1.23` | Ultralytics minimum |
| `pyyaml` | `>=6` | dataset/model YAML |
| `pillow` | `>=10` | image crops and synthetic hand paste |

The code targets Python 3.10 (`ruff target-version = "py310"`), matching the
KV260 runtime. `uv` uses a CPU PyTorch index locally; Kaggle's plain `pip`
install gets the CUDA build needed by the T4 workers.

## Keypoint schemas

`cpsa_ml.schema` defines COCO-WholeBody indices once:

* `COCO17` — COCO body keypoints 0–16.
* `HALFBUST_62` — body 0–10, face `23,39,31,42,47,71,77,74,80`, left hand
  91–111 and right hand 112–132.

`flip_idx` is derived from the names (explicit left/right pairs), so it is an
involution by construction. The face pairs are jaw 0<->16, brow 19<->24,
mouth 48<->54; chin (jaw 8) and lip centres 51/57 self-map.

## Build the half-bust dataset

COCO-WholeBody uses the COCO 2017 images plus
`coco_wholebody_{train,val}_v1.0.json`. The builder keeps persons with visible
head and shoulders, crops from above the head to the hip line (with a 3:4
aspect limit), drops out-of-crop keypoints (`v=0`) and optionally pastes a
seeded synthetic hand over the face.

```bash
uv run python -m cpsa_ml.datasets.halfbust \
  --coco-root /data/coco \
  --wholebody-json /data/annotations/coco_wholebody_train_v1.0.json \
  --split train --schema halfbust62 --out /data/halfbust62 \
  --occlusion-p 0.3 --seed 0
```

Repeat with `--split val` and the val JSON into the same `--out` folder. The
writer emits:

```text
<out>/images/{train,val}/<image_id>_<stem>_a<ann_id>.jpg
<out>/labels/{train,val}/<image_id>_<stem>_a<ann_id>.txt
<out>/data.yaml     # kpt_shape, flip_idx, names (relative split paths)
<out>/schema.json   # indices, names, flip_idx
<out>/stats.json    # per-split counters
```

Each label line is `0 cx cy w h` followed by `K*(x y v)` normalized to the
crop. The synthetic hand is taken from the visible hand keypoints of another
sample; the same seed reproduces both the hand choice and placement.

## Training

`cpsa_ml.train.train` is a thin wrapper around `YOLO(cfg).train`:

```bash
uv run python -m cpsa_ml.train \
  --data /data/halfbust62/data.yaml --epochs 100 --batch 32 \
  --device 0,1 --imgsz 640 --project /kaggle/working/runs --name fp32
```

* Ultralytics trains square letterboxed inputs. Use `imgsz=640` and
  `rect=False` (the wrapper's default); the deployment graph is exported and
  audited at the camera's native `1x3x480x640`.
* `device="0,1"` is converted to `[0, 1]` for Ultralytics DDP on 2x T4.
* `resume=/path/to/last.pt` loads the checkpoint and restores optimizer, epoch
  and training arguments. Keep the original data/model settings when resuming.
* `seed` and `deterministic=True` are passed to Ultralytics.

A CPU smoke run is available but skipped by default:

```bash
CPSA_ML_SLOW=1 uv run pytest -q tests/test_train_smoke.py
```

## ONNX export and DPU operator check

```bash
uv run cpsa-ml-onnx-check \
  --model /kaggle/working/export/halfbust62.pt \
  --output /kaggle/working/export/halfbust62.onnx \
  --imgsz 480 640 --opset 17
```

`cpsa_ml.onnx_check` exports a fixed `1x3x480x640` graph at opset 17 and
splits operators at the last backbone/head `Conv`; the DFL subgraph and pose
decode are counted as post-processing:

* backbone/neck: allow-list only. `Sigmoid`/`Mul` SiLU patterns, `MatMul`,
  `Softmax`, attention, `Tanh`, `Gelu`, `Erf` and anything else outside the
  list fail.
* keypoint/box decode: reported separately as **CPU post-processing allowed**
  (for example the confidence `Sigmoid` in `Pose.kpts_decode`).

The Hard-Swish YAMLs (`configs/yolov8{n,s}-pose-hswish.yaml`) use
`activation: nn.Hardswish()`, which Ultralytics 8.4.203 reads as a top-level
key and applies to every `Conv` block. They use the standard YOLOv8 P3/P4/P5
backbone with no attention modules. The `*-silu.yaml` files are the negative
controls for the operator check.

## Kaggle protocol

Kaggle credentials are not available from the development host; the notebooks
and metadata are validated locally and pushed by the user.

### 1. Upload data once

* Attach the public COCO 2017 Kaggle dataset as
  `/kaggle/input/coco-2017-dataset/coco2017`.
* Create/upload `coco-wholebody-json` once with the two annotation files.
* Build the dataset in notebook 01, then publish `/kaggle/working/halfbust62`
  as a private dataset named `cpsa-halfbust`. Add a
  `dataset-metadata.json` next to the built files, for example
  `{"title": "CPSA half-bust", "id": "YOUR_USER/cpsa-halfbust",
  "licenses": [{"name": "CC-BY-4.0"}]}`, then run
  `kaggle datasets create -p halfbust62` the first time and
  `kaggle datasets version -p halfbust62 -m "..."` afterwards. The helper is
  `ml/scripts/kaggle_push_dataset.sh`.
* Download checkpoints with `kaggle kernels output <kernel> -p <dir>` and
  publish them as the `cpsa-checkpoints` dataset version.

### 2. Push notebooks

Each notebook has a Kaggle CLI folder:

```bash
kaggle kernels push -p ml/notebooks/01_build_halfbust
kaggle kernels push -p ml/notebooks/03_train_fp32
kaggle kernels push -p ml/notebooks/05_qat
```

Before pushing, set:

* `COMMIT_SHA` in the first code cell (the install is
  `pip install "git+https://github.com/fcabecciaw/CPSA_2026@<sha>#subdirectory=ml"`).
* The Kaggle username and dataset slugs in `kernel-metadata.json`
  (`fcabecciaw/...` is a placeholder).
* The `/kaggle/input/...` paths in the notebook parameters.

`kernel-metadata.json` follows the Kaggle CLI format with
`id`, `title`, `code_file`, `language`, `kernel_type`, `is_private`,
`enable_gpu`, `enable_internet`, `dataset_sources`, `competition_sources` and
`kernel_sources`. The accelerator finding: official Kaggle documentation does
not document a GPU-model key, but community-verified CLI reports (Kaggle CLI
issues #490 and #821, August 2026) show that
`"machine_shape": "NvidiaTeslaT4"` selects the UI's **GPU T4 x2** option;
without it `enable_gpu: true` defaults to P100. The newer CLI
`--accelerator NvidiaTeslaT4` flag has mixed reports. If a push still lands on
P100, select **GPU T4 x2** in the Kaggle UI before running; the metadata key is
already present.

### 3. Checkpoints and resume

Notebook 03 writes checkpoints under `/kaggle/working/runs/<name>/weights/`.
After the session, publish that folder as a new private dataset version (for
example `cpsa-checkpoints`). In the next session attach it and set:

```python
RESUME = "/kaggle/input/cpsa-checkpoints/last.pt"
```

Ultralytics resumes from `last.pt`; `best.pt` is for inference/export, not for
continuing an interrupted optimizer state. Kaggle sessions are capped at 12 h,
so checkpoint every few epochs and re-publish at least once per session.

## QAT

`cpsa_ml.qat` probes `pytorch_nndct` and exposes a Brevitas fallback:

```python
from cpsa_ml.qat import try_nndct, quantize_model

print(try_nndct())
qmodel = quantize_model(model)  # QuantConv2d + QuantHardSwish
```

The wrapper is layer-wise, not FX/graph traced: it replaces `Conv2d` with
`QuantConv2d` (weights `Int8WeightPerTensorFixedPoint`) and `Hardswish` with
`QuantHardSwish` (activations `Int8ActPerTensorFixedPoint`), preserving weights.
Batch norms, residual adds and the Ultralytics decode stay in float; the adapted
float weights are then fed to the same Vitis AI 3.5 PTQ flow. Brevitas is never
used for deployment export. `QuantHardTanh` is also available in Brevitas if a
ReLU6-style activation is preferred.

## What was verified locally

No Kaggle credentials were available on the development host, so the Kaggle
runs are left to the user. Locally:

* `uv run --extra dev pytest -q` passes; the slow smoke test and the optional
  Brevitas tests are skipped.
* `CPSA_ML_SLOW=1 uv run --extra dev pytest -q` runs the 1-epoch, 2-image CPU
  smoke test and passes.
* `uv run --extra dev --extra qat pytest -q` passes the Brevitas `QuantConv2d`
  / `QuantHardSwish` tests on a tiny conv net and on the 62-keypoint pose
  model.
* The Hard-Swish ONNX export passes `cpsa_ml.onnx_check` at `1x3x480x640`,
  opset 17; the stock SiLU export fails with `Sigmoid`/`Mul` violations.
* The top-level `activation: nn.Hardswish()` YAML key is honoured by
  Ultralytics 8.4.203: the built model contains Hard-Swish and no SiLU.

## Licences

* Ultralytics is **AGPL-3.0**. It is isolated in `ml/`; trained weights are
  subject to its terms.
* UBody is **research-only**; it is optional and must not ship in the product.
* COCO 2017 and COCO-WholeBody images/annotations keep their original COCO
  terms; check them before redistribution.
* Brevitas is BSD-3-Clause; `rtmlib`/RTMW follow their upstream licences.
