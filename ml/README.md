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
* `HAND_21` — the left-hand WholeBody block 91–111 on its own, i.e. the
  MediaPipe 21-landmark order of the stage-2 hand model. `flip_idx` is the
  identity: a mirrored crop keeps every landmark on the same finger joint.

`flip_idx` is derived from the names (explicit left/right pairs), so it is an
involution by construction. The face pairs are jaw 0<->16, brow 19<->24,
mouth 48<->54; chin (jaw 8) and lip centres 51/57 self-map.

### Keypoint OKS sigmas

Ultralytics 8.4.203 only uses the COCO-17 `OKS_SIGMA` table for a `[17, 3]`
head; for any other `kpt_shape` its `v8PoseLoss` falls back to
`ones(nkpt) / nkpt` unless the dataset YAML declares `kpt_oks_sigmas`
(`PoseTrainer.set_model_attributes` in `models/yolo/pose/train.py`, consumed
by `utils/loss.py`). With `1/62 = 0.016` the loss `1 - exp(-e)` saturates and
its gradient vanishes for a joint that starts far from its target, which is
why a COCO-only `lrelu_512` run fits the face (median error 1-3% of the crop
height) but not shoulders, elbows, wrists or hands (PCK@0.2fw for the wrists
was 0.0).

`cpsa_ml.schema.COCO_WHOLEBODY_SIGMAS` holds the 133 annotation-scale sigmas of
mmpose's `configs/_base_/datasets/coco_wholebody.py` (which cites
COCO-WholeBody `evaluation/myeval_wholebody.py`); `KeypointSchema.oks_sigmas`
selects them through `wholebody_index` and floors them at
`OKS_SIGMA_FLOOR = 0.025`, the smallest COCO-17 sigma (eyes), so the iBUG face
landmarks and the hand joints (down to 0.007 in WholeBody) keep a usable
gradient. Mirrored pairs share the larger value so a horizontal flip cannot
change the loss: the jaw ends are the only asymmetric pair (0.042 vs 0.045).
Every `data.yaml` written by `write_dataset_yaml`, `write_hand_dataset_yaml`
and `build_mixed_data_yaml` now carries the 62- or 21-entry
`kpt_oks_sigmas` list.

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

## Build the full-frame dataset

A model trained only on half-bust crops never sees a standing person, so its
predicted box hugs the keypoints instead of the true person extent — and the
runtime uses that box height divided by the face width to decide `full_body`
vs `half_body`. `cpsa_ml.datasets.fullframe` therefore writes the **original**
COCO images with the true COCO person box:

```bash
uv run python -m cpsa_ml.datasets.fullframe \
  --coco-root /data/coco \
  --wholebody-json /data/annotations/coco_wholebody_train_v1.0.json \
  --split train --schema halfbust62 --out /data/fullframe62 \
  --half-train-root /data/halfbust62 --seed 0
```

Repeat with `--split val` and the val JSON into the same `--out` folder. The
layout is the same as the half-bust set; labels use the same `HALFBUST_62`
mapping and `flip_idx`, because the mapping and writers are imported from
`cpsa_ml.datasets.halfbust` (one implementation).

* **Mix ratio** — the full-frame training split is sized to make full frames
  25% of the mixed set: `n_full = round(0.25 / 0.75 * n_half)`, where `n_half`
  is read from `stats.json` (or pass `--target-count`/`--max-images`). The
  sample is deterministic for a seed. The val split writes **every** qualifying
  image so `val_full` is a fair full-frame measure.
* **Unlabeled people** — a person that fails `--min-visible` (or has a
  degenerate COCO box) gets no label line, matching Ultralytics'
  `convert_coco`; the person is counted in `persons_unlabeled`. An image with
  no labelable person is not written. An image that contains a **crowd**
  annotation is skipped entirely, because a crowd box would teach the model
  that those people are background.
* **No hand-over-face occlusion** — the point of this set is the true person
  extent, so its pixels stay untouched.
* **Framing check** — open `build/fullframe_check.jpg` (or run the builder with
  `--preview`) to confirm boxes hug the feet and heads before publishing.

## Build the hand-crop dataset (stage 2)

The two-stage design trains a dedicated hand model on square hand crops: the
body model finds the person and the hands, and the hand model returns 21
MediaPipe landmarks on a 192 px crop. `cpsa_ml.datasets.hands` builds that set
from the public sources in one pass:

```bash
uv run python -m cpsa_ml.datasets.hands \
  --hagrid-images /data/hagrid/hagrid_30k \
  --hagrid-annotations /data/hagrid/annotations \
  --ultralytics-root /data/hand-keypoints \
  --out /data/hand21 --imgsz 192 --val-ratio 0.05 --seed 0
```

* **Sources** — HaGRIDv2 `annotations_with_landmarks` (normalized hand boxes
  plus 21 normalized MediaPipe landmarks per hand) with the 384p image sample
  (`cj-mills/hagrid-sample-30k-384p`; its image ids match the official
  annotations, only the sample's own gesture JSONs lack landmarks), and the
  Ultralytics hand-keypoints YOLO-pose set (26,768 images, 21 keypoints). Both
  are re-cropped to the same geometry so the sources stay visually consistent.
* **Crop geometry** — the hand box is enlarged 1.6–2.2x, the centre is shifted
  up to 6% and the side jittered by ±10%, the window is kept square, clamped
  inside the image (no padding artefacts) and resized to 192 px with LANCZOS.
  The jitter seed is derived from the sample identity, so a rerun reproduces
  every crop and an interrupted build resumes by skipping existing files.
* **Labels** — class `0` (`hand`), the hand box inside the crop and 21
  keypoints; landmarks outside the crop get `v=0`. `flip_idx` is the identity
  (see above). Left and right hands share the class and the handedness is
  counted in `stats.json` only.
* **HaGRID boxes** — the official gesture boxes do not always contain every
  fingertip, so the crop box is the union of the box and the landmark extent.
* **Splits** — `train`/`val` follow the source split (`val` and `test` go to
  validation) and `--val-ratio` additionally holds out a seeded hash share of
  the training images, so validation is never empty when a source only ships
  training images.
* **Provenance** — `stats.json` records the per-split counters, gesture labels,
  handedness, crop geometry and the source licences; `data.yaml` and
  `schema.json` describe the 21x3 schema.

`configs/yolov8n-hand-lrelu.yaml` is the DPU-friendly model (LeakyReLU 26/256,
`kpt_shape [21, 3]`); `TrainConfig(model_cfg=...)` selects it:

```python
from cpsa_ml.models import config_path
from cpsa_ml.train import TrainConfig, train

train(
    TrainConfig(
        data="/data/hand21/data.yaml",
        schema="hand21",
        activation="lrelu",
        model_cfg=config_path("yolov8n-hand-lrelu.yaml"),
        imgsz=192,
        batch=64,
        device="0,1",
        name="hand21-yolov8n-lrelu",
    )
)
```

Notebook 07 runs the same build plus the training on Kaggle.

### Auto-label the booth recordings

`cpsa_ml.pseudolabel` turns frames recorded with `scripts/record_frames.py`
(or a video) into both datasets with one RTMW teacher pass:

```bash
uv sync --extra pseudolabel  # rtmlib + onnxruntime
uv run python -m cpsa_ml.pseudolabel \
  --frames data/booth/P01_20260901T100000Z --out data/labeled/P01
uv run python -m cpsa_ml.pseudolabel --video clip.mp4 --stride 5 --out data/labeled/clip
```

* `<out>/halfbust` is a full-frame `HALFBUST_62` dataset for the body model:
  YOLOX gives the person box, RTMW-x gives 133 keypoints, points below their
  group threshold or outside the frame get `v=0` and the rest `v=2`, so a
  teacher point is never mistaken for a human annotation.
* `<out>/hand21` is the stage-2 crop set built by the same crop code as the
  public data; the handedness is known here (WholeBody 91–111 left, 112–132
  right) and recorded in `stats.json`.
* A hand block becomes a crop only when at least `--min-hand-keypoints` (10)
  points reach `--hand-point-score` (2.0, stricter than the 1.0 group
  threshold), its extent is at most `--max-hand-fraction` (0.6) of the larger
  person-box side, and its wrist→middle-MCP / middle-MCP→middle-tip ratio is
  anatomically plausible. When a hand is not visible the teacher often fires a
  second block over the torso or the background; on the four HaGRID frames used
  for the local check these gates kept all five real hands (21 confident
  points each) and removed the three hallucinated blocks (at most 7 confident
  points, extents 0.61–1.07 of the person box). The reason counters
  (`low_score_left_hand`, `too_large_left_hand`, ...) are kept in `stats.json`.
* The split is a seeded hash of the relative path and the run is resumable: a
  frame whose label exists is skipped, and `stats.json` is rewritten atomically
  after every chunk. `--min-hand-keypoints` (default 10) and `--hand-score`
  decide which teacher hands become crops.
* The recordings contain faces: keep them local, record only people who agreed,
  and do not publish the frames.

## Teacher labels for HaGRID (notebook 06)

Notebook 06 (`ml/notebooks/06_teacher_hagrid.ipynb`) downloads the HaGRID 30k-384p sample and labels it with the RTMW-x whole-body teacher of `cpsa_ml.pseudolabel` on a Kaggle T4. It writes `/kaggle/working/hagrid62.tar`: an uncompressed `halfbust62` YOLO-pose dataset with `data.yaml`, `images/{train,val}` and `labels/{train,val}` at the archive root (10 % val by seeded hash), plus 16 rendered samples in `/kaggle/working/samples/`.

GPU use:
- The install replaces the CPU `onnxruntime` with `onnxruntime-gpu[cuda,cudnn]`, and the first code cell proves a real CUDA session.
- The labeler runs in its own process, so it repeats the check there. With `--device cuda` it calls `onnxruntime.preload_dlls()` before creating a model and checks `session.get_providers()` on the teacher and the detector. It raises an error instead of silently running about 20x slower on the CPU.
- The log reports frames/s and an ETA for each chunk.

Run it as a committed version (Save & Run All), then in notebook 04 use **Add Input → Notebook output → 06_teacher_hagrid**. Notebook 04 finds `hagrid62.tar`, extracts it with `cpsa_ml.datasets.staging.stage_archive` and mixes it into training through `EXTRA_DATA`. No dataset needs publishing.

If the CUDA preflight fails, label on the local PC instead: `python -m cpsa_ml.pseudolabel --device cpu --teacher rtmw-l --no-hands --val-fraction 0.1`. This is slow (several frames per minute per core).

Licences: HaGRID is a CC BY-SA 4.0 variant, and the derived labels inherit that notice. The RTMW teacher and rtmlib are Apache-2.0.

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

Notebook 03 trains on the **mixed** dataset. `build_mixed_data_yaml` (in
`cpsa_ml.variant_study`) writes a `data.yaml` whose `train` entry is the curated
train list plus the full-frame `images/train` directory, keeps `val` on the
half-bust split (the primary metric Ultralytics uses for early stopping and
`best.pt`), adds `val_full` for the full-frame split and `val_h160` for the
size-filtered half-bust val list. Absolute paths are used because the YAML lives
outside both dataset roots. If the full-frame root is missing, it logs
`WARNING: FULL-FRAME DATASET MISSING` and trains on half-body images only.

### Training-list curation

The demo sees a seated user at 0.45-0.7 m, where hands and face matter most, but
the built crops are small (median 86x115 px) and 77% of them show no hand at all.
`cpsa_ml.datasets.curation.curate_train_list` turns the built dataset into an
explicit Ultralytics train list (`.txt`, one absolute image path per line):

* keep only crops with `height >= min_height` (default 128 px; sizes come from
  the JPEG headers, so 104k crops take a few seconds);
* repeat crops that show a hand with `>= 10` visible keypoints `hand_repeat`
  times (default 2) and, optionally, crops with `>= 5` visible face keypoints
  `face_repeat` times (default 1 = off); a crop that qualifies for both is
  repeated `max()` times, never twice over;
* mix extra `halfbust62` datasets, each `(root, repeat)`, into the same list;
* write `curated_train.txt`, `curated_val_h160.txt` and
  `curated_train_summary.json` (counts before/after, hand and face shares, per
  extra dataset) into the output directory;
* with `labels_root` set, mirror the images into
  `<out>/curated_train_mirror/{images,labels}/train` (one symlinked image
  directory, one symlink per override label) so Ultralytics resolves the Task P
  pseudo-labels from the mirrored tree instead of the read-only dataset. A
  missing override label aborts the build rather than mixing GT and pseudo
  labels.

Measured on the local `halfbust62` build: 103,996 train crops -> 47,983 kept at
`min_height=128` -> 69,901 list entries with `hand_repeat=2`; the hand share of
the kept crops rises from 27% to 46%, the face share from 48% to 64%. The val
split stays the full 4,335 GT crops, with 1,497 of them in the `h >= 160` demo
list. Notebooks 03 and 04 expose `MIN_HEIGHT`, `HAND_REPEAT`, `FACE_REPEAT`,
`LABELS_ROOT` (a dataset named `cpsa-halfbust-pl`, located by content) and
`EXTRA_DATA` (a dataset named `cpsa-hagrid62`, or notebook 06's
`hagrid62.tar`, empty when absent).

Notebook 04 trains on the D1 mix (COCO half-bust crops at least 256 px tall
plus the HaGRID teacher set) and stages only that subset of `cpsa-halfbust`:
`load_manifest("halfbust62_h256")` reads the gzipped package data
`cpsa_ml/data/halfbust62_h256.txt.gz`, which lists 16,828 of the 103,996 train
crops with their labels, all 4,335 val crops and labels, and `data.yaml` (42,327
paths). `stage_dataset(..., include=...)` copies only those paths after listing
the mount with `os.walk`, so Kaggle moves ~42k files instead of 216k and reads
no image header over `/kaggle/input`. Notebook 04's `MIN_HEIGHT = 256` keeps the
curation height filter on the extra datasets. Regenerate the manifest after
rebuilding the dataset:

```bash
python ml/scripts/make_height_manifest.py \
    --root ~/Downloads/_output_/halfbust62 --min-height 256 \
    --out ml/src/cpsa_ml/data/halfbust62_h256.txt.gz
```

The script reads only the JPEG headers, prints the counts (103,996 train crops,
16,828 at >= 256 px) and writes the sorted POSIX relative paths, gzipped without
a timestamp so a rerun on the same build is byte-identical.

### Fine-tuning recipe of the study

`variant_study.AUGMENTATIONS` pins the recipe shared by every row, so a library
default change cannot make the rows incomparable: explicit `optimizer="SGD"`
(`auto` silently switches to AdamW for short runs) with `lr0=0.01`, `lrf=0.01`,
`momentum=0.937`, `weight_decay=0.0005`, `warmup_epochs=3`, `cos_lr=True`,
`degrees=10`, `translate=0.1`, `scale=0.5`, `fliplr=0.5` with the schema
`flip_idx`, `mosaic=1.0`, `mixup=0`, `copy_paste=0`, `close_mosaic=15`, the Pose
loss gain at the Ultralytics default (`pose=12.0`, pinned and not tuned) and
`patience=30`.

* **Motion blur.** Ultralytics 8.4.203 accepts an `augmentations=` train argument
  (a list of albumentations transforms, serialised into the checkpoint), so
  `variant_study.build_albumentations()` passes its own default transform list
  plus `MotionBlur(blur_limit=(3, 9), p=0.15)`; nothing is monkeypatched. Without
  albumentations (or with a release older than 1.4.4, which the defaults need)
  the list is not passed, a `motion blur is SKIPPED` warning is logged and
  `metrics_fp32.json` records `"custom_augmentations": null`, so the run stays
  honest about what it did. Notebook 04 installs `albumentations==1.4.21`
  best-effort (`check=False`).
* **Multi-scale.** Recommendation: **off** (`multi_scale=0.0`, the Ultra
  default). It resizes the training batch every iteration, which costs step time
  on 2x T4, and `scale=0.5` already augments scale on these crops; it was not
  benchmarked on a T4 here, so revisit only when a run has spare time and the
  `h160` metric shows a reason.

Notebook 03 keeps the `cpsa_ml.train` defaults for the baseline Hard-Swish
checkpoint; notebook 04 carries this recipe for the urgency run
(`VARIANTS = [DEFAULT_VARIANTS[2]]`, `EPOCHS = 30`) and the main run
(`DEFAULT_VARIANTS`, `EPOCHS = 120`).

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

### 1. Build and publish the datasets

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

**Notebook 02** then builds the full-frame set from the same COCO images plus
the attached `cpsa-halfbust` dataset (it reads `n_half` from its
`stats.json`): `/kaggle/working/fullframe62` is published the same way as
`cpsa-fullframe`. Attach both `cpsa-halfbust` and `cpsa-fullframe` to notebooks
03/04; a missing `cpsa-fullframe` is loud but not fatal (training falls back to
half-body images only).

**Notebook 06** (`06_teacher_hagrid`) labels the HaGRID 30k-384p sample with
the RTMW-x whole-body teacher on the GPU and writes one
`/kaggle/working/hagrid62.tar`. Notebook 04 attaches that output directly
(**Add Input → Notebook output → `06_teacher_hagrid`**), locates it with
`find_inputs(INPUT, "hagrid62.tar")` and extracts it locally with
`staging.stage_archive`, so no dataset version is published between the two
runs. Its COCO half is the `halfbust62_h256` manifest subset
(`MIN_HEIGHT = 256`), so staging copies ~42k crops plus one tar instead of 216k
small files; see "Training-list curation" for the manifest itself.

**Notebook 07** needs no attached dataset: it downloads the HaGRID 384p sample
and its `annotations_with_landmarks`, plus the Ultralytics hand-keypoints zip
(~2.1 GB) into `/tmp/hand_data`, builds `/kaggle/working/hand21` and trains the
192 px hand model. Publish the crops as `cpsa-hand21` (only if the
non-commercial hand-keypoints source is acceptable; set
`USE_ULTRALYTICS = False` for a HaGRID-only set) and the weights plus metadata
from `/kaggle/working/hand-model` as `cpsa-hand-model`.

### 2. Push notebooks

Each notebook has a Kaggle CLI folder:

```bash
kaggle kernels push -p ml/notebooks/01_build_halfbust
kaggle kernels push -p ml/notebooks/02_build_fullframe
kaggle kernels push -p ml/notebooks/03_train_fp32
kaggle kernels push -p ml/notebooks/04_variant_study
kaggle kernels push -p ml/notebooks/05_qat
kaggle kernels push -p ml/notebooks/07_hand_pose
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

Every row trains square at its own `train_imgsz` on the mixed half-bust +
full-frame set, then validates and exports `deploy.onnx` at its deployment
size, so a smaller row also trains on fewer pixels. Seed, schedule and
augmentations are shared and pinned in `cpsa_ml.variant_study.AUGMENTATIONS`.
Configs are `configs/yolov8n-pose-{hswish,relu,lrelu}.yaml` (exact names),
selected through `cpsa_ml.models.model_yaml_path`; `relu` remains available for
the speed matrix even though the default accuracy matrix uses `lrelu`.

The COCO transfer copies the backbone, neck and the Pose head's box (`cv2`),
class (`cv3`) and `dfl` branches, whose shapes do not depend on the keypoint
count; only the keypoint branch `cv4` is freshly initialised. `metrics_fp32.json`
records `metrics.half` (the half-bust `val` split, the primary metric) and
`metrics.full` (the full-frame `val_full` split), so the framing decision has
evidence on both framings.

1. **Train the matrix on Kaggle.** Download the dataset once
   (`kaggle datasets download -d fabiocabeccia/cpsa-halfbust -p ~/data`), push
   notebook 04, then publish its output as a new `cpsa-checkpoints` version:

   ```bash
   kaggle kernels push -p ml/notebooks/04_variant_study
   kaggle kernels output fabiocabeccia/cpsa-04-variant-study -p variant_study
   ```

   Notebook 04 starts every row from the official COCO `yolov8n-pose.pt`,
   copies every shape-matching backbone/neck tensor plus the Pose head's box,
   class and DFL branches, and re-initialises only the 62-keypoint `cv4`
   branch. It logs the transfer in `transfer_report.json` and writes `best.pt`,
   `last.pt`, `results.csv`, `metrics_fp32.json`, `transfer_report.json` and
   `deploy.onnx` under
   `/kaggle/working/variant_study/<activation>_<train_imgsz>/`. `RESUME` in the
   notebook is a dict keyed by that folder name, for example
   `{"lrelu_512": ".../lrelu_512/last.pt"}`.

2. **Quantize and dump the INT8 head tensors for every row.** Calibration uses
   train images preprocessed exactly like `DpuPoseEstimator`; the dump stores
   the int8 head values, the `.xmodel` `fix_point` of every output and the
   letterbox parameters. Repeat `--test-images NAME=DIR` (and
   `--calib-images`) to put both val splits in one dump; `--calib-count` and
   `--test-count` apply to each directory and the name tags the frames so the
   two sets stay distinguishable even though they share COCO image stems. The
   deployment size of each row is recorded as `deploy_size` in its
   `metrics_fp32.json`; for `hswish_640` (480x640):

   ```bash
   tools/vitis/run_vitis.sh python tools/vitis/dump_quant_outputs.py \
     --weights variant_study/hswish_640/best.pt \
     --calib-images train=~/data/cpsa-halfbust/images/train \
     --calib-images train_full=~/data/cpsa-fullframe/images/train \
     --test-images half=~/data/cpsa-halfbust/images/val \
     --test-images full=~/data/cpsa-fullframe/images/val \
     --calib-count 64 --test-count 64 \
     --height 480 --width 640 --threads 1 \
     --out build/variant_study/hswish_640/quant
   ```

   Repeat with `--height 384 --width 512` for `lrelu_512` and
   `--height 288 --width 384` for `lrelu_384` (`lrelu_640` uses 480x640).
   `--data <root> --calib-split/--test-split` still dumps one split per file.
   `--artifacts <host deploy bundle>` quantizes from a pre-built bundle instead
   of `best.pt` and adds the container/host reference check.

3. **Score INT8 and FP32 with the same metric code** (person AP50, pose
   mAP50-95 by OKS, per-group PCK@0.1/0.2 face-width normalized, hand and face
   centroid errors, per-group missing-detection rate, and framing):

   ```bash
   PYTHONPATH=src python tools/vitis/eval_quant.py \
     --dump build/variant_study/hswish_640/quant/quant_outputs.npz \
     --weights variant_study/hswish_640/best.pt \
     --eval-split half=~/data/cpsa-halfbust/images/val:~/data/cpsa-halfbust/labels/val \
     --eval-split full=~/data/cpsa-fullframe/images/val:~/data/cpsa-fullframe/labels/val \
     --baseline build/variant_study/hswish_640/eval/metrics_int8.json \
     --out build/variant_study/hswish_640/eval
   ```

   `eval_quant.py` decodes with the runtime decoder
   (`cpsa.vision.pose_decode.QuantizedRawHeadDecoder`), rebuilds the same graph
   in float for the FP32 reference, and writes `metrics_int8.json` plus
   `comparison.json` / `comparison.md` for the first split (so the baseline
   workflow is unchanged). The per-split files
   `metrics_int8_<split>.json`/`metrics_fp32_host_<split>.json` and the
   side-by-side `comparison_splits.md` / `comparison_splits.json` show both
   splits next to each other. The `framing.*` rows classify every ground-truth
   person from its COCO box and from its matched prediction box: a full-frame
   split must read mostly `full_body` (and a half-body split mostly
   `half_body`), otherwise the predicted box still does not describe the true
   person extent. The threshold is read from
   `cpsa.vision.subject.SubjectConfig.full_body_ratio`; it is never copied into
   the metric code. The baseline is the INT8 metrics file of the `hswish_640`
   row.

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

* `uv run --extra dev pytest -q` passes: 231 passed, 2 skipped (Brevitas and
  the one-epoch smoke test). The root `tests/unit/test_eval_quant.py` passes
  too.
* A real-data full-frame smoke test ran on five COCO val images and the real
  `coco_wholebody_val_v1.0.json`: one image was skipped for its crowd
  annotation, four images with six labelled people were written, and
  `build/fullframe_check.jpg` shows the GT boxes/heads/hands over the original
  pixels (boxes reach the feet).
* The hand builder ran on the real HaGRIDv2 peace data (the 384p sample plus
  `annotations/train/peace.json`) and wrote 1,529 crops in 3.8 s: train 1,466
  (1,235 `peace` + 231 `no_gesture`, from 1,236 images) and val 63 (55 + 8,
  from 55 images), 23 MB of images and 6.2 MB of labels. `build/hand_check.jpg`
  shows six crops with the hand box and all 21 landmarks on the real pixels.
  The 1,075 peace entries whose `hand_landmarks` list is empty are skipped, and
  the official gesture box is unioned with the landmark extent.
* A 12-crop build from the real Ultralytics hand-keypoints tree (18,777 train /
  7,993 val images) produced valid `data.yaml` (`kpt_shape [21, 3]`, identity
  flip_idx) and 21x3 labels, so both sources share one geometry.
* The teacher auto-labeler ran end-to-end with the real RTMW-x + YOLOX
  checkpoints on four HaGRID peace frames: all four persons got full-frame
  `HALFBUST_62` lines (teacher points `v=2`), and five hand crops were accepted
  while the three hallucinated blocks were rejected as `low_score` (at most 7
  of their 21 landmarks reached the 2.0 crop gate, against 21/21 for every real
  hand). `build/autolabel_check.jpg` shows every accepted crop: a real hand with
  coherent MediaPipe landmarks. Frames, `hand21` dataset and `stats.json` are
  under `build/hand_samples/batch_04/labeled_final/`.
* A 1-epoch CPU smoke run trained the hand config on the 1,529-crop local set:
  `yolov8n-hand-lrelu`, 3,385,074 parameters, 9.7 GFLOPs at 192 px, 46
  iterations in 31 s plus validation, `best.pt`/`last.pt` and `results.csv`
  under `build/hand_runs/smoke_1ep`. The metrics after one epoch are not
  meaningful; the run proves the config, schema and dataset load and train.
* Ultralytics 8.4.203 loads the generated mixed `data.yaml` with a two-root
  `train` list plus `val` and `val_full` (`YOLODataset` only, no training);
  every split reports 62x3 keypoints.
* `CPSA_ML_SLOW=1 uv run --extra dev pytest -q` runs the 1-epoch, 2-image CPU
  smoke test and passes.
* `uv run --extra dev --extra qat pytest -q` passes the Brevitas `QuantConv2d`
  / `QuantHardSwish` tests on a tiny conv net and on the 62-keypoint pose
  model.
* The Hard-Swish ONNX export passes `cpsa_ml.onnx_check` at `1x3x480x640`,
  opset 17; the stock SiLU export fails with `Sigmoid`/`Mul` violations.
* The top-level `activation: nn.Hardswish()` YAML key is honoured by
  Ultralytics 8.4.203: the built model contains Hard-Swish and no SiLU.
* `uv lock --check` passes after adding the `pseudolabel` extra and the
  `cpsa-ml-autolabel` script; `ruff check .` is clean and the touched Python
  files and notebook 07 are formatted.

## Licences

* Ultralytics is **AGPL-3.0**. It is isolated in `ml/`; trained weights are
  subject to its terms.
* HaGRIDv2 (images plus `annotations_with_landmarks`) is **CC BY-SA 4.0
  variant** (attribution, share-alike). Hand crops derived from it inherit the
  share-alike terms.
* The Ultralytics hand-keypoints dataset is **CC BY-NC-SA 4.0 —
  non-commercial**. Mixing it into `hand21` makes the crops non-commercial and
  not redistributable for commercial use; notebook 07 prints the warning and
  `USE_ULTRALYTICS = False` builds a HaGRID-only set instead.
* UBody is **research-only**; it is optional and must not ship in the product.
* COCO 2017 and COCO-WholeBody images/annotations keep their original COCO
  terms; check them before redistribution.
* Brevitas is BSD-3-Clause. `rtmlib` is Apache-2.0; the RTMW whole-body and
  YOLOX person checkpoints are Apache-2.0 via MMPose; ONNX Runtime is MIT.
  The teacher models are downloaded at run time and recorded in `stats.json`.
