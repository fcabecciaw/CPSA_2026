# `tools/vitis` — the KV260 DPU model pipeline

PC-side pipeline that turns the Vitis AI model-zoo **YOLOX-nano** and
**MoveNet** into `.xmodel` files compiled for the KV260 DPU
(`DPUCZDX8G_ISA1_B4096`, fingerprint `0x101000016010407`). No training and no
requantization: both zoo packages are already quantized.

Vitis AI 3.5's `DPUCZDX8G_ISA1_B4096` target alias maps to fingerprint
`0x101000056010407` (the 3.x feature code), which the 2.5-era KV260 B4096
firmware rejects with a fingerprint mismatch. `compile.sh` therefore compiles
against an explicit `{"fingerprint": "0x101000016010407"}` arch file and reads
the fingerprint back from the artifact; override it with
`VITIS_DPU_FINGERPRINT`, or pass a complete arch file with `VITIS_ARCH_JSON`.

The graph is cut before the decode: every stride (8, 16, 32) returns its raw
head tensor with 85 channels, `[tx, ty, tw, th, objectness, 80 class logits]`.
Sigmoid, grid/anchor decode and NMS stay on the CPU, in `cpsa.yolox`.

## Images

| Image | Tag | Purpose |
|---|---|---|
| `xilinx/vitis-ai-pytorch-cpu` | `ubuntu2004-3.5.0.306` | base: Vitis AI 3.5 compiler + `pytorch_nndct` |
| `cpsa-vitis-pytorch` | `3.5.0.306` | the base plus the packages the scripts import |

Base digest:
`sha256:baf43fd9d6e1ea5a2c12b727ac3aafd15ead936172bcd636d7af98b0cca49298`.
`run_vitis.sh` builds the derived image on first use.

## Compiling the detector

```sh
tools/vitis/zoo_yolox_nano.sh            # download, extract, compile, inspect
tools/vitis/zoo_yolox_nano.sh --skip-download
```

The script downloads the float+QAT package
(`pt_yolox-nano_coco_416_416_1G_3.0.zip`, 23.7 MB), compiles the QAT-converted
`qat/convert_qat_results/YOLOX_0_int.xmodel` with the pinned-fingerprint
`compile.sh`, and inspects the artifact.

Outputs (git-ignored): `build/models/zoo/yolox_nano/` (package),
`build/vitis/yolox_nano/compiled/yolox_nano.xmodel`,
`build/vitis/report/inspect_yolox_nano.json`, `build/vitis/logs/zoo_yolox_nano_*.log`.
The script fails unless the artifact reports one DPU subgraph, fingerprint
`0x101000016010407`, input `1x416x416x3`, and the three `85`-channel heads.

Observed: 5 subgraphs, 1 DPU (371 ops), input `1x416x416x3` xint8 `fp=-2`,
outputs `1x{52,26,13}x{52,26,13}x85` xint8 `fp=3`. The package
(`yolox/models/yolo_head_q.py`) applies `sigmoid` to channels `4:` and then
`xy = (t_xy + grid) * stride`, `wh = exp(t_wh) * stride`, so the runtime decodes
with `logits=True`.

`inspect_xmodel.py` prints the same facts for any `.xmodel`:

```sh
tools/vitis/run_vitis.sh python tools/vitis/inspect_xmodel.py model.xmodel
```

## Decoder reference check

`yolox_reference.py` compares the detector's CPU half (`cpsa.yolox`) against the
package's own post-processing on real images. The reference model needs
PyTorch/`pytorch_nndct` (container Python 3.8) while `cpsa` runs on the host, so
it has three modes:

```sh
# 1. Host: letterbox the images exactly as the runtime does (BGR, pad 114).
uv run --no-sync python tools/vitis/yolox_reference.py prepare \
  --images build/vitis/yolox_reference/images --work build/vitis/yolox_reference

# 2. Container: run the package model and its own post-processing.
tools/vitis/run_vitis.sh python tools/vitis/yolox_reference.py dump \
  --model build/models/zoo/yolox_nano/pt_yolox-nano_coco_416_416_1G_3.0 \
  --work build/vitis/yolox_reference

# 3. Host: identical preprocessing, int8 head round-trip, decode and compare.
uv run --no-sync python tools/vitis/yolox_reference.py check \
  --work build/vitis/yolox_reference --out build/vitis/report/yolox_reference.json
```

The reference path is the QAT-converted deployable model
(`qat/convert_qat_results/deployable.pth`), the weights the `int` xmodel was
dumped from. Re-running `pytorch_nndct`'s quantized simulation in the Vitis AI
3.5 container is not possible: the package's `quant_info.json` was produced by
3.0 and 3.5 rejects its node names. Both sides decode the same head tensors
rounded through int8 with the graph's output fix point (`fp=3`), which is what
the board decodes.

`check` requires pixel-identical preprocessing, the same class, IoU >= 0.9 and
no extra or missing boxes; boxes that collapse below 1 px^2 against the frame
edge are excluded on both sides (predictions into the letterbox padding have no
usable area). Observed on 27 COCO images (plus the package demo and two
Ultralytics samples): 177 matched boxes, minimum IoU 0.999987, no failures.

## MoveNet (`src/cpsa/movenet.py`, M1)

```sh
tools/vitis/zoo_movenet.sh            # download, extract, compile, inspect
tools/vitis/zoo_movenet.sh --skip-download
```

`pt_movenet_coco_192_192_0.5G_3.0.zip` (11.4 MB) ships the quantized
`quantized/MoveNet_int.xmodel`, which is recompiled with the same pinned
fingerprint. Outputs (git-ignored): `build/models/zoo/movenet/` (package),
`build/vitis/movenet/compiled/movenet.xmodel`,
`build/vitis/report/inspect_movenet.json`, `build/vitis/logs/zoo_movenet_*.log`.
The script fails unless the artifact reports one DPU subgraph, fingerprint
`0x101000016010407`, a `1x192x192x3` input and the four heads below.

Observed: 6 subgraphs, 1 DPU, input `1x192x192x3` xint8 `fp=6`, four `48x48`
heads at stride 4:

| head | shape | `fix_point` | contents |
|---|---|---|---|
| heatmaps | `1x48x48x17` | 3 | keypoint logits, sigmoid in a CPU subgraph |
| centers | `1x48x48x1` | 3 | person-center logits, sigmoid in a CPU subgraph |
| regs | `1x48x48x34` | 2 | `(dx, dy)` per keypoint from the center cell |
| offsets | `1x48x48x34` | 7 | sub-cell `(dx, dy)` per keypoint |

The input `fix_point` is 6, i.e. the int8 grid covers `[-2, 2)`, while the zoo
preprocessing is a plain RGB square resize with no normalization. Feeding raw
`0..255` saturates every pixel above 2 at `127` and the pose collapses; the
runtime feeds `pixel / 128` (`int8 = round(pixel / 2)`,
`movenet.MOVENET_INPUT_SCALE`). Measured against the package's float model on 24
COCO person crops (keypoints both sides score >= 0.3): `pixel / 128` 2.69 px
mean, `/127.5 - 1` 3.94 px, `/255` 4.26 px, raw 37.87 px. The package's own
reported accuracy uses raw pixels and does not transfer to the compiled grid.

Because the graph-level heatmaps and centers are post-sigmoid but the DPU
subgraph exports logits, `movenet.decode` applies `sigmoid` itself, then the
zoo threshold (`0.1`), the analytic center prior (`1 / (distance + 1.8)`, the
package's `center_weight_origin.npy`), the regression target from the center
cell, the `1 / (distance + 1.8)` heatmap weighting and the offset field.

The runtime is not wired into the pipeline yet (that is M2); `movenet_reference.py`
checks it standalone:

```sh
uv run --no-sync python tools/vitis/movenet_reference.py prepare \
  --images build/vitis/movenet_reference/images --work build/vitis/movenet_reference
tools/vitis/run_vitis.sh python tools/vitis/movenet_reference.py dump \
  --model build/models/zoo/movenet/pt_movenet_coco_192_192_0.5G_3.0 \
  --work build/vitis/movenet_reference
uv run --no-sync python tools/vitis/movenet_reference.py check \
  --work build/vitis/movenet_reference --out build/vitis/report/movenet_reference.json
```

`dump` runs the package's quantized TorchScript model on the dequantized inputs,
inverts the `nn.Sigmoid` the package ends its heatmap and center heads with,
round-trips all four heads through the DPU fix points and decodes them with the
package's own `movenetDecode` (the function `movenet.decode` mirrors).
`movenetDecode` returns no scores, so the tool reads each score from the
package's heatmap at the package's own peak cell and aborts if that peak does not
reproduce the package coordinates. `check` requires pixel-identical
preprocessing and int8 inputs, the same set of keypoints at score >= 0.3, and at
least 20 images with a mean distance <= 2 px on the 192 grid. Observed on 24
COCO person crops: 102 compared keypoints, mean 0.0000 px, max 0.0000 px, no
failures.

## The 60 fps runtime

The showcase loop does not dequantize the three heads and sigmoid every cell:
`cpsa.yolox.QuantizedYoloxDecoder` thresholds the objectness and the best class
in the int8 domain first and runs the float decode only on the survivors, which
is bit-identical to the float path on the dequantized heads for every box above
`conf_thr`. Preprocessing letterboxes with `cv2.resize` + `cv2.copyMakeBorder`
and quantizes through `cv2.LUT` straight into the DPU input view, and
`YoloxDetector.submit`/`wait` keeps two frames in flight so frame N runs on the
DPU while frame N-1 is decoded and drawn.

`scripts/profile_showcase.py` reports the per-stage means (camera conversion,
preprocess, DPU, decode, draw, JPEG) and the achievable loop rate with that
overlap:

```sh
# On the board, with the unit stopped so the DPU is free.
sudo systemctl stop cpsa-showcase
cd ~/cpsa && PYTHONPATH=src python3 scripts/profile_showcase.py \
  --model models/yolox_nano.xmodel --runs 60
```

With `--model` it runs the real graph; without it the DPU stage assumes the
measured 4.7 ms so the CPU stages can be profiled before a deploy. The script
exits 0 when the measurement ran; `--target-ms 16.7` labels the budget line
PASS/OVER. Host numbers are not board numbers: on a laptop the loop is CPU-bound
and the DPU stage is the stand-in.

## On the board

The board (Kria KV260, Vitis AI 3.5 VART built natively) must have the DPU app
loaded (`sudo xmutil loadapp kv260-benchmark-b4096`, done by
`deploy/systemd/cpsa-dpu.service`); `xdputil query` must report
`DPUCZDX8G_ISA1_B4096_0101000016010407` / fingerprint `0x101000016010407`.

If `xdputil query` fails with `cannot open file: /tmp/DPU_0`, remove the stale
`/tmp/DPU_0` lock file.
