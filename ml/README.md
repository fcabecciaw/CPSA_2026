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
uv sync --extra dev --extra sources   # gdown for the COCO-WholeBody download
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
| `gdown` | `>=5` | `sources` extra only; notebook 01 downloads the COCO-WholeBody annotations from the official Google Drive ids |
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
`coco_wholebody_{train,val}_v1.0.json`; `cpsa_ml.datasets.sources` locates
attached files or downloads the official ones (see the Kaggle protocol).
The builder keeps persons with visible head and shoulders, crops from above
the head to the hip line (with a 3:4 aspect limit), drops out-of-crop
keypoints (`v=0`) and optionally pastes a seeded synthetic hand over the face.

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
* `--activation {hswish,relu,lrelu,silu}` selects
  `configs/yolov8n-pose-<activation>.yaml`: `relu` and `lrelu` are the DPU
  variants of the activation study, `silu` is the negative control.

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

### 1. Build and publish the dataset

Notebook 01 obtains both inputs itself:

* **Images** — attach the public `awsaf49/coco-2017-dataset` (any tree with
  `train2017/` and `val2017/`); `cpsa_ml.datasets.sources.find_coco_root`
  locates it anywhere under `/kaggle/input`.
* **COCO-WholeBody v1.0 annotations** — `wholebody_annotations` uses attached
  `coco_wholebody_<split>_v1.0.json` files when a dataset provides them, and
  otherwise downloads the official Google Drive files
  (https://github.com/jin-s13/COCO-WholeBody) into `/tmp/coco_wholebody` so
  they are not published with the output. The notebook installs the
  `cpsa-ml[sources]` extra for `gdown`; with Internet disabled, attach a
  dataset containing the two JSON files instead. Every file is checked for
  `info.description == "COCO-WholeBody"`.

Then publish `/kaggle/working/halfbust62` as a private dataset named
`cpsa-halfbust`. Add a `dataset-metadata.json` next to the built files, for
example `{"title": "CPSA half-bust", "id": "YOUR_USER/cpsa-halfbust",
"licenses": [{"name": "CC-BY-4.0"}]}`, then run
`kaggle datasets create -p halfbust62` the first time and
`kaggle datasets version -p halfbust62 -m "..."` afterwards. The helper is
`ml/scripts/kaggle_push_dataset.sh`. Download checkpoints with
`kaggle kernels output <kernel> -p <dir>` and publish them as the
`cpsa-checkpoints` dataset version.

### 2. Push notebooks

Each notebook has a Kaggle CLI folder:

```bash
kaggle kernels push -p ml/notebooks/01_build_halfbust
kaggle kernels push -p ml/notebooks/03_train_fp32
kaggle kernels push -p ml/notebooks/04_variant_study
kaggle kernels push -p ml/notebooks/05_qat
```

Before pushing, set:

* `COMMIT_SHA` in the first code cell (notebook 01 installs
  `cpsa-ml[sources]`; the others install the plain package, all from
  `git+https://github.com/fcabecciaw/CPSA_2026@<sha>#subdirectory=ml`).
* The Kaggle username and dataset slugs in `kernel-metadata.json` (the CPSA
  notebooks use `fabiocabeccia/...`).
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
already present in the GPU notebooks.

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

## Variant study

The study measures how activation choice and training/deployment size trade
accuracy for speed on the DPU. Notebook 04 takes a list of
`(activation, train_imgsz, deploy HxW)` rows. The default matrix is a
**proposal pending approval**:

| variant | activation | train imgsz | deploy input |
| --- | --- | ---: | --- |
| `hswish_640` | Hard-Swish | 640 | 480x640 (baseline) |
| `lrelu_640` | LeakyReLU(0.1015625) | 640 | 480x640 |
| `lrelu_512` | LeakyReLU(0.1015625) | 512 | 384x512 |
| `lrelu_384` | LeakyReLU(0.1015625) | 384 | 288x384 |

Every row trains square at its own `train_imgsz`, then validates and exports
`deploy.onnx` at its deployment size, so a smaller row also trains on fewer
pixels. Seed, schedule and augmentations are shared and pinned in
`cpsa_ml.variant_study.AUGMENTATIONS`. Configs are
`configs/yolov8n-pose-{hswish,relu,lrelu}.yaml` (exact names), selected through
`cpsa_ml.models.model_yaml_path`; `relu` remains available for the speed matrix
even though the default accuracy matrix uses `lrelu`.

1. **Train the matrix on Kaggle.** Download the dataset once
   (`kaggle datasets download -d fabiocabeccia/cpsa-halfbust -p ~/data`), push
   notebook 04, then publish its output as a new `cpsa-checkpoints` version:

   ```bash
   kaggle kernels push -p ml/notebooks/04_variant_study
   kaggle kernels output fabiocabeccia/cpsa-04-variant-study -p variant_study
   ```

   Notebook 04 starts every row from the official COCO `yolov8n-pose.pt`,
   copies every shape-matching backbone/neck tensor, keeps the freshly
   initialised 62-keypoint head, logs the transfer in
   `transfer_report.json`, and writes `best.pt`, `last.pt`, `results.csv`,
   `metrics_fp32.json`, `transfer_report.json` and `deploy.onnx` under
   `/kaggle/working/variant_study/<activation>_<train_imgsz>/`. `RESUME` in the
   notebook is a dict keyed by that folder name, for example
   `{"lrelu_512": ".../lrelu_512/last.pt"}`.

2. **Quantize and dump the INT8 head tensors for every row.** Calibration uses
   train images preprocessed exactly like `DpuPoseEstimator`; the dump stores
   the int8 head values, the `.xmodel` `fix_point` of every output and the
   letterbox parameters. The deployment size of each row is recorded as
   `deploy_size` in its `metrics_fp32.json`; for `hswish_640` (480x640):

   ```bash
   tools/vitis/run_vitis.sh python tools/vitis/dump_quant_outputs.py \
     --weights variant_study/hswish_640/best.pt --data ~/data/cpsa-halfbust \
     --calib-split train --test-split val --calib-count 64 --test-count 64 \
     --height 480 --width 640 --threads 1 \
     --out build/variant_study/hswish_640/quant
   ```

   Repeat with `--height 384 --width 512` for `lrelu_512` and
   `--height 288 --width 384` for `lrelu_384` (`lrelu_640` uses 480x640).
   `--artifacts <host deploy bundle>` quantizes from a pre-built bundle instead
   of `best.pt` and adds the container/host reference check.

3. **Score INT8 and FP32 with the same metric code** (person AP50, pose
   mAP50-95 by OKS, per-group PCK@0.1/0.2 face-width normalized, hand and face
   centroid errors, per-group missing-detection rate):

   ```bash
   PYTHONPATH=src python tools/vitis/eval_quant.py \
     --dump build/variant_study/hswish_640/quant/quant_outputs.npz \
     --weights variant_study/hswish_640/best.pt --data ~/data/cpsa-halfbust \
     --test-split val \
     --baseline build/variant_study/hswish_640/eval/metrics_int8.json \
     --out build/variant_study/hswish_640/eval
   ```

   `eval_quant.py` decodes with the runtime decoder
   (`cpsa.vision.pose_decode.QuantizedRawHeadDecoder`), rebuilds the same graph
   in float for `metrics_fp32_host.json`, and writes `metrics_int8.json` plus
   `comparison.json` / `comparison.md`. The baseline is the INT8 metrics file
   of the `hswish_640` row.

4. **Apply the decision rule (proposal) below** to the `decision` block of
   every `comparison.json`. It reports each group's INT8 PCK@0.2fw against the
   baseline and returns `pass`, `fail` or `unknown`.

5. **Record the study.** Keep each row's `comparison.md` with its build
   artifacts and cite the accepted pairing in the study report; the JSON keeps
   the per-group deltas so the choice can be audited later.

**Decision rule (proposal, not a project requirement).** Accept a faster
activation/input pairing only when its INT8 wrist, hand and face PCK@0.2fw are
each within 2 points of the INT8 baseline, that is `delta >= -0.02` against
`hswish_640` 480x640; a missing or zero baseline PCK yields `unknown` instead
of a pass. Person AP50, pose mAP50-95 and the centroid errors are reported next
to the rule for context but do not decide it.

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
