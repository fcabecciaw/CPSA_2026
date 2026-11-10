# Vitis AI 3.5 PTQ and KV260 compilation of `HALFBUST_62` pose

Quantize the raw-head YOLOv8-pose deploy graph with `vai_q_pytorch` 3.5 and
compile it for the KV260 (`DPUCZDX8G_ISA1_B4096`, fingerprint
`0x101000016010407`). The reproducible commands, images and rerun steps live in
`tools/vitis/README.md`; this note records what was observed and why the
pieces are shaped the way they are.

Status: results below come from a structural run with randomly initialised
weights (no trained checkpoint yet). The graph, compilation, tensor layout,
fix-points and decoder path are what this run validates; accuracy numbers are
meaningless until the Kaggle `best.pt` is plugged in with `--weights`. The board
latency run is still outstanding (see section 5).

## 1. Tooling

| Item | Value |
|---|---|
| Base image | `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306` |
| Base digest | `sha256:baf43fd9d6e1ea5a2c12b727ac3aafd15ead936172bcd636d7af98b0cca49298` |
| Derived image | `cpsa-vitis-pytorch:3.5.0.306` (base + `ultralytics==8.4.203`, `--no-deps`) |
| Container Python | 3.8.6 (conda env `vitis-ai-pytorch`) |
| Container PyTorch | 1.13.1+cpu |
| `pytorch_nndct` | 3.5.0+60df3f1+torch1.13.1 |
| `vai_c_xir` | `/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/vai_c_xir` |
| KV260 arch file | `/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json` |

### 1.1 Target alias versus fingerprint

The arch file shipped in the 3.5 image contains
`{"target": "DPUCZDX8G_ISA1_B4096"}` rather than an explicit `fingerprint`.
That alias is not stable across Vitis AI releases: the same target maps to
`0x101000016010407` in Vitis AI 2.5 and to `0x101000056010407` from 3.0 on
(feature-code change). The KV260 benchmark B4096 firmware is the 2.5-era
design, so compiling through the alias produces an `.xmodel` that the DPU
runner rejects:

```text
CHECK fingerprint fail! model_fingerprint 0x101000056010407 is un-matched with
actual dpu_fingerprint 0x101000016010407. Please re-compile xmodel.
```

`tools/vitis/compile.sh` therefore writes `{"fingerprint":
"0x101000016010407"}` to `build/vitis/compiled/arch_kv260.json` and compiles
against it. The compiler reports
`Target architecture: DPUCZDX8G_ISA1_B4096_0101000016010407`, and the script
reads `dpu_fingerprint` back from the artifact and fails the build if it is not
the requested value. `VITIS_DPU_FINGERPRINT` overrides the fingerprint and
`VITIS_ARCH_JSON` passes a complete arch file.

### 1.2 Why the container rebuilds the module

`cpsa_ml` targets Python 3.10 (`dataclass(slots=True)`, `zip(strict=True)`), so
the container's Python 3.8 cannot import it. Host TorchScript is also not an
option: TorchScript archives are not loadable by an older PyTorch, and the host
runs PyTorch 2.14 against the container's 1.13.1.

The host therefore exports a bundle (`deploy_state_dict.pt`, `deploy_spec.json`,
`model.yaml` with `kpt_shape: [62, 3]`, and a `deploy_reference.npz` output),
and the container:

1. loads `ml/src/cpsa_ml/deploy.py` **by path** (single source of truth for the
   graph), builds the `PoseModel` from the same YAML with Ultralytics, wraps it
   with `deploy_from_yolo`, and loads the `state_dict` with `strict=True`;
2. runs one forward and requires the output to match `deploy_reference.npz`
   within `1e-4` before quantizing (observed max difference `1e-6` or less).

### 1.3 Thread caps matter

Calibration on the shared build host was pathological with the default 16
threads: ~20 s per frame, and one run was killed after 48 min with no progress
log. The quantized graph is a chain of small CPU ops, so OpenMP thread
synchronisation dominates. Measured with `--time-only` (4 frames, same host):

| `--threads` | float forward | quantized forward |
|---|---|---|
| 16 (torch default) | 6–15 s | ~20 s (killed) |
| 8 | 0.064 s | 1.78 s |
| 1 | 0.164 s | 1.39 s |

`tools/vitis/quantize.py --threads 1` is used for the calibration run and the
script logs every 10 frames with s/frame.

## 2. Deploy graph

`ml/src/cpsa_ml/deploy.py:RawHeadDeploy` wraps the Ultralytics `PoseModel`,
walks the layers up to (not including) the `Pose` head using Ultralytics' own
`f`/`i`/`save` metadata, and returns one tensor per stride:

```text
stride 8 : (1, 251, 60, 80)   [box 64 | class 1 | keypoints 186]
stride 16: (1, 251, 30, 40)
stride 32: (1, 251, 15, 20)
```

`251 = 4 * 16 + 1 + 3 * 62`. No DFL, sigmoid, anchor decode or NMS is in the
graph: those stay in `cpsa.vision.pose_decode.decode_raw_heads`.

## 3. PTQ results

Calibration: 120 frames, 133.4 s total (1.11 s/frame, `--threads 1`); test:
24 frames; whole run 150.7 s. The quantizer compares the `test`-mode outputs
against the float module on the same frames (`build/vitis/report/quant_report.json`):

| Output | Shape (NCHW) | Cosine | Max abs error | Mean abs error |
|---|---|---|---|---|
| `head_stride8` | `(1, 251, 60, 80)` | 0.99992 | 0.0313 | 0.0100 |
| `head_stride16` | `(1, 251, 30, 40)` | 0.99991 | 0.0315 | 0.0102 |
| `head_stride32` | `(1, 251, 15, 20)` | 0.99990 | 0.0312 | 0.0109 |

Outputs saturate no channel at the compiled `fix_point`; the maximum absolute
value is 7.15 against the int8 range of ±8 at `fix_point = 4`.

## 4. Compiled graph

`vai_c_xir` reports `Total device subgraph number 5, DPU subgraph number 1`:

| # | Device | Ops | Content |
|---|---|---|---|
| 0 | USER | 1 | `data-fix` (float input → int8) |
| 1 | DPU | 389 | `conv2d-fix` 72, `const-fix` 144, `eltwise-fix` 69, `hard-sigmoid-fix` 63, `concat-fix` 16, `strided_slice-fix` 16, `download` 3, `pool-fix` 3, `upsample-fix` 2, `upload` 1 |
| 2–4 | CPU | 1 each | `fix2float` (int8 → float per output) |

No layer runs on the CPU: the CPU subgraphs only convert dtypes at the
boundaries. Artifact: `build/vitis/compiled/pose_halfbust62.xmodel`, 6,126,485
bytes, md5 `06cff1d893c0ff239f0f49d96fe103ba`, fingerprint
`0x101000016010407` (`DPUCZDX8G_ISA1_B4096_0101000016010407`).

### 4.1 DPU tensors and fix-points

| Role | Shape (NHWC) | Dtype | `fix_point` |
|---|---|---|---|
| Input | `(1, 480, 640, 3)` | xint8 | 7 |
| Output stride 8 | `(1, 60, 80, 251)` | xint8 | 4 |
| Output stride 16 | `(1, 30, 40, 251)` | xint8 | 4 |
| Output stride 32 | `(1, 15, 20, 251)` | xint8 | 4 |

The DPU subgraph lists its outputs in the order 32, 16, 8. `DpuPoseEstimator`
derives each stride from the tensor shape and calls
`order_outputs_by_stride`, so the runtime order does not matter; the host
check replays that path with the real order and verifies the arrays are
unchanged (`build/vitis/report/inspect_compiled.json`,
`build/vitis/report/decoder_check.json`).

## 5. Decoder compatibility and board status

Host checks (`tools/vitis/verify_deploy.py`, randomly initialised weights):

- The float deploy graph decoded with `decode_raw_heads` matches Ultralytics'
  own decode over 6300 candidates: boxes `2.0e-5` px, keypoints `3.1e-5` px,
  scores `5.5e-10`.
- Quantized test outputs, converted to NHWC int8 with `fix_point = 4` and
  passed through `cpsa.adapters.pose.dpu`, decode to one candidate per cell
  (4800 / 1200 / 300 per frame) with finite values, the same counts as the
  float decode, cosine ≥ 0.99990 and no saturation. `order_outputs_by_stride`
  reproduces the stride order 8/16/32 from the 32/16/8 tensor order.
- Derived bounds from the int8 output step (`2 ** -4`): keypoint offsets are
  exact to 0.5 / 1.0 / 2.0 px at strides 8 / 16 / 32, and the class score
  changes by at most 0.008. The box logits are unaffected for these weights,
  but the random head is degenerate: every box is 120 / 240 / 480 px at stride
  8 / 16 / 32, i.e. the DFL distribution is uniform before quantization. With
  trained weights the box logits will span a wider range and the `fix_point`
  choice must be re-checked on the real checkpoint.

Board run (KV260, Vitis AI 3.5 VART built natively, DPU app
`DPUCZDX8G_ISA1_B4096_0101000016010407` at 300 MHz, `xdputil query` fingerprint
`0x101000016010407`): the runner loaded and executed the model without a
fingerprint error. `scripts/bench_pose.py --layout halfbust62 --frames 200`
(random uint8 frames, warmup 10):

| Run | Mean | Median | p95 | Throughput |
|---|---|---|---|---|
| 200 frames | 289.0 ms | 288.8 ms | 291.5 ms | 3.46 fps |
| 100 frames (repeat) | 288.7 ms | 288.4 ms | 290.8 ms | 3.46 fps |

The DPU is not the bottleneck. Measured on the board:

| Stage | Time per frame |
|---|---|
| DPU runner (`execute_async` + `wait`, int8 in/out) | 25.3 ms (p95 25.4) |
| Python decode (`decode_raw_heads`: DFL softmax, sigmoid, NMS) | 182 ms |
| Input quantization (`quantize_fix_point`) | 44 ms |
| Output dequantization (`dequantize_fix_point`) | 20 ms |
| Total `DpuPoseEstimator.estimate` | 289 ms |

`cProfile` over 20 estimates attributes 63 % of the time to
`decode_raw_heads` (49 ms/frame in `_softmax`, 38 ms/frame in `_sigmoid` over
6300 candidates x 62 keypoints); the DPU itself would sustain ~39 fps. So the
3.5 fps figure is a CPU post-processing cost on the Kria's A53 cores, not a
model or compilation problem. The threshold-first decoder below removes that
cost (30.0 fps at `conf_thr = 0.25`); NMS on many surviving boxes is what
remains. `detections/frame` is 0.0 because the randomly initialised head scores
below `conf_thr = 0.25`; the host check exercises the same decode with
`conf_thr = -1`.

### Fast post-processing: threshold before activation

`cpsa.vision.pose_decode.QuantizedRawHeadDecoder` compares the raw int8 class
logits against the precomputed integer threshold before DFL, sigmoid,
dequantization and grid decode, so only survivors are activated and
dequantized. `DpuPoseEstimator` quantizes the input through a precomputed
256-entry lookup table (`input_quantization_lut` / `quantize_rgb_lut`), and
`letterbox` returns the frame unchanged when it already has the model input
size. `decode_raw_heads` stays as the reference path.

Equivalence is checked at two levels:

- Host: `tests/unit/test_pose_decode.py` compares the fast decoder with
  `decode_raw_heads` on random and golden tensors, `fix_point` values -3/0/4/7,
  one and three classes, all-below-threshold and all-above-threshold inputs, and
  a threshold that falls exactly on a quantization step. Boxes, scores,
  keypoints and keypoint confidences match with `np.array_equal`.
- Board: `DpuPoseEstimator` from the previous commit and the new one run the
  same random frame through `pose_halfbust62.xmodel` at `conf_thr` 0.25, 0.005
  and 0.0001 and return the same persons in the same order (0, 14 and 280
  detections, every box, score, keypoint and keypoint confidence identical).
  Random int8 tensors at each `fix_point` decode identically too, with 124, 130,
  138 and 252 surviving candidates.

The integer cut is `floor(logit(conf_thr) * 2 ** fix_point) - 1`: the trailing
`- 1` absorbs one integer of rounding in the computed cut, so the prefilter can
only ever admit candidates that the reference path might keep, never drop one,
even when the threshold lands exactly on a quantization step. Candidates it
admits that the reference rejects are removed by the unchanged exact
`score > conf_thr` comparison in `_select`, so the decoded persons are
identical. `conf_thr <= 0` disables the prefilter (`-128`) and `conf_thr >= 1`
keeps nothing.

The compiled head has random weights and its class logits are constant per
stride (-114 / -92 / -70 in int8, `fix_point = 4`), so `conf_thr = 0.25` keeps
no candidates and a low threshold keeps whole scales: 0.005 keeps the
300-candidate stride-32 map, 0.0001 keeps all 6300. That makes case (b) an
upper bound on the survivor count, not a trained-model expectation.
`scripts/bench_pose.py --frames 200 --warmup 10` (random frames; "before" is
HEAD 089b419 on the same board):

| Run | Mean | Median | p95 | Throughput | Candidates/frame | Detections/frame |
|---|---|---|---|---|---|---|
| before, `conf_thr=0.25` | 289.5 / 291.7 ms | 289.3 / 291.6 ms | 292.2 / 293.8 ms | 3.4 / 3.5 fps | 6300 decoded | 0 |
| after, `conf_thr=0.25` | 33.31 ms | 33.31 ms | 33.44 ms | 30.0 fps | 0 | 0 |
| before, `conf_thr=0.0001` | 813.6 ms | 812.9 ms | 818.0 ms | 1.2 fps | 6300 decoded | 280 |
| after, `conf_thr=0.0001` | 752.4 ms | 751.7 ms | 757.1 ms | 1.3 fps | 6300 | 280 |
| after, `conf_thr=0.005` | 56.03 ms | 56.00 ms | 56.18 ms | 17.8 fps | 300 | 14 |

Mean per-stage time in ms/frame. The "before" columns use the same stage timers
inserted into the old `estimate`, so they are directly comparable; both columns
of one `conf_thr` come from a single 200-frame process, so they share the
board's state (run-to-run variation is about 1 %):

| Stage | before 0.25 | after 0.25 | before 0.0001 | after 0.0001 | after 0.005 |
|---|---|---|---|---|---|
| preprocess (letterbox + quantize + copy) | 57.2 | 6.8 | 46.8 | 7.9 | 6.9 |
| DPU runner (`execute_async` + `wait`) | 25.4 | 25.4 | 25.4 | 25.4 | 25.4 |
| dequantize + DFL softmax + sigmoid + grid decode | 201.8 (NMS trivial) | 1.0 | 699.1 (incl. ~515 NMS) | 149.2 | 6.9 |
| NMS (separate in the fast path) | - | 0.0 | - | 516.1 | 14.5 |
| unletterbox | 0.0 | 0.0 | 44.9 | 45.0 | 2.2 |
| total `estimate` | 288.7 | 33.3 | 818.2 | 752.4 | 56.0 |

In the old path NMS runs inside `decode_raw_heads`, so its cost is inside the
decode row; `cProfile` splits the 699 ms into about 515 ms of `_nms` and 184 ms
of decode. The old preprocess row is not stable across runs (46.8-59.4 ms over
four runs) because it allocates a full float32 copy of the frame plus int8
temporaries every frame, so allocator and cache state depend on what the
threshold made the decoder allocate; the fast path stays at 6.8-7.9 ms.

At `conf_thr = 0.25` the fast decode costs under 1 ms (threshold scan,
dequantize and decode of the survivors) against 201.8 ms before; the input LUT
costs 6.8 ms against 57.2 ms. At `conf_thr = 0.0001` the new path still removes
550 ms of decode work, but 516 ms of NMS on 6300 survivors remains, so that
frame is DPU-bound plus NMS-bound rather than decode-bound.

A synthetic decoder-only run with exactly 100 surviving candidates costs 4.6-4.8
ms for decode and 16.7-32.8 ms for NMS with 18-37 retained boxes; no threshold
of this randomly initialised model produces that survivor count, so it was
measured on the decoder alone. Combined with the 6.9 ms preprocess, the 25.4 ms
DPU runner and about 3 ms of unletterbox that extrapolates to roughly 70 ms
(14.3 fps) for a frame with 100 candidates and 37 retained boxes — below the
15 fps floor. A trained checkpoint with one or two persons per frame keeps very
few candidates above 0.25 and runs the NMS loop once or twice (1-2 ms), so its
operating point is the 33.3 ms row. The 30 fps proposal is met for the
empty-detection frame (33.3 ms, 30.0 fps); the 15 fps floor is met with 300
survivors (56.0 ms, 17.8 fps). The remaining fixed cost is the 25.4 ms DPU
runner plus the 6.9 ms input LUT; the remaining variable cost is NMS and
`unletterbox` on surviving boxes.

Note on the input LUT: the 256-entry table is applied with
`bytes.translate` on the frame bytes (a C-level 256-byte table lookup), not
`numpy` fancy indexing, because `np.take` on the A53 costs 10.0 ms/frame against
6.9 ms for the byte translation; both were verified bit-identical to
`quantize_fix_point(rgb.astype(np.float32) / 255.0, fix_point)` for every uint8
value and every tested `fix_point`. No new dependency is involved.

Operational note: `show_dpu`/`xdputil query` failed once with
`cannot open file: /tmp/DPU_0` from `file_lock_lnx.cpp`; a stale zero-length
`/tmp/DPU_0` lock caused it and removing the file fixed the query. Not related
to the model.

## 6. Variant speed matrix

Latency does not depend on the weights, so the matrix below uses the seeded
random initialisation of the deploy bundle (`--seed 0`). It compares three
activations at three input sizes, nine variants in total:

| Axis | Values |
|---|---|
| Activation | `hswish` (Hard-Swish, `ml/configs/yolov8n-pose-hswish.yaml`), `relu` (`yolov8n-pose-relu.yaml`), `lrelu` (`nn.LeakyReLU(0.1015625)` = 26/256, the alpha the DPUCZDX8G fuses, `yolov8n-pose-lrelu.yaml`) |
| Input HxW | 480x640 (current), 384x512, 288x384 (multiples of 32, camera 3:4) |

Only speed is measured here. Accuracy needs trained weights and is a separate
measurement: the random head scores below `conf_thr = 0.25`, so every board run
decodes zero candidates and the stage timings exclude NMS work on real
detections (see the NMS cost in section 5).

### 6.1 Build and structural guard

`tools/vitis/variants.sh` runs the whole matrix (or one cell with
`--activations`/`--sizes`):

```sh
tools/vitis/variants.sh --count 40
```

Per cell it exports the deploy bundle at the requested size
(`cpsa_ml.deploy --cfg configs/yolov8n-pose-<act>.yaml --height --width`),
calibrates 40 frames with `--threads 1` (0.30-1.36 s/frame: 12 s for a
288x384 relu variant, 54 s for hswish 480x640), compiles with the pinned
fingerprint, inspects the artifact and fails unless the graph is fully
offloaded. Calibration frames depend only on the input size, so they are
generated once per size and shared by the three activations. Outputs land in
`build/vitis/variants/<act>_<H>x<W>/`, with one row per variant in
`build/vitis/variants/matrix.tsv`.

All nine artifacts passed:

| Check | Result |
|---|---|
| Device subgraphs | 5: one USER `data-fix` (float in to int8), one DPU, three CPU `fix2float` (one per output) |
| Compute on CPU | none: the only ops outside the DPU are `data-fix`/`fix2float` |
| Fingerprint | `0x101000016010407` (`DPUCZDX8G_ISA1_B4096_0101000016010407`) for all nine |
| Activation | `relu`/`lrelu` folded into `conv2d` (`nonlinear` = `RELU`/`LEAKYRELU` on 71 convs, `LEAKYRELU_alpha = 0.1015625`); `hswish` stays as 63 `hard-sigmoid` + 63 `eltwise` mul ops |
| DPU tensors | input `1x{H,W}x3` xint8 `fix_point` 7, outputs xint8 `fix_point` 4, for all nine |
| PTQ vs float | cosine >= 0.99989 on every output of every variant |

`hswish` at 480x640 rebuilds the section 4 artifact (389 DPU ops) and the board
reproduces section 5 (25.3 ms DPU, 33.3 ms end to end), which is the control for
the matrix.

### 6.2 Latency

Board: KV260, DPU app `DPUCZDX8G_ISA1_B4096_0101000016010407` at 300 MHz, one
process per variant, random uint8 frames:

```sh
PYTHONPATH=src python3 scripts/bench_pose.py --xmodel <variant>.xmodel \
  --layout halfbust62 --frames 200 --warmup 10 --conf-thr 0.25 --json
```

This is the threshold-first post-processing of section 5 (`DpuPoseEstimator` +
`QuantizedRawHeadDecoder`), which is why the end-to-end column is DPU-dominated
instead of decode-bound.

| Activation | Input | DPU mean | DPU p95 | end-to-end mean | end-to-end p95 | fps | vs `hswish` 480x640 (DPU / end-to-end) |
|---|---|---|---|---|---|---|---|
| `hswish` | 480x640 | 25.32 ms | 25.42 ms | 33.29 ms | 33.43 ms | 30.0 | 1.00x / 1.00x |
| `hswish` | 384x512 | 16.04 ms | 16.13 ms | 21.32 ms | 21.44 ms | 46.9 | 1.58x / 1.56x |
| `hswish` | 288x384 | 10.25 ms | 10.33 ms | 13.40 ms | 13.50 ms | 74.7 | 2.47x / 2.49x |
| `relu` | 480x640 | 16.99 ms | 17.09 ms | 24.96 ms | 25.06 ms | 40.1 | 1.49x / 1.33x |
| `relu` | 384x512 | 11.09 ms | 11.19 ms | 16.36 ms | 16.46 ms | 61.1 | 2.28x / 2.03x |
| `relu` | 288x384 | 7.39 ms | 7.49 ms | 10.52 ms | 10.65 ms | 95.0 | 3.43x / 3.16x |
| `lrelu` | 480x640 | 16.98 ms | 17.06 ms | 24.97 ms | 25.08 ms | 40.1 | 1.49x / 1.33x |
| `lrelu` | 384x512 | 11.07 ms | 11.16 ms | 16.34 ms | 16.45 ms | 61.2 | 2.29x / 2.04x |
| `lrelu` | 288x384 | 7.34 ms | 7.45 ms | 10.49 ms | 10.60 ms | 95.3 | 3.45x / 3.17x |

A second 200-frame pass over all nine models reproduced every DPU and
end-to-end mean within 0.4 % (`build/vitis/report/variants_board_rerun.jsonl`).

Per-stage mean, ms/frame (`total` is the measured `estimate`; the stages account
for all but 0.05-0.1 ms of Python overhead per frame):

| Activation | Input | preprocess | DPU | decode | NMS | unletterbox | total |
|---|---|---|---|---|---|---|---|
| `hswish` | 480x640 | 6.89 | 25.32 | 0.98 | 0.00 | 0.01 | 33.29 |
| `hswish` | 384x512 | 4.42 | 16.04 | 0.77 | 0.00 | 0.01 | 21.32 |
| `hswish` | 288x384 | 2.48 | 10.25 | 0.59 | 0.00 | 0.01 | 13.40 |
| `relu` | 480x640 | 6.90 | 16.99 | 0.98 | 0.00 | 0.01 | 24.96 |
| `relu` | 384x512 | 4.42 | 11.09 | 0.77 | 0.00 | 0.01 | 16.36 |
| `relu` | 288x384 | 2.48 | 7.39 | 0.59 | 0.00 | 0.01 | 10.52 |
| `lrelu` | 480x640 | 6.93 | 16.98 | 0.98 | 0.00 | 0.01 | 24.97 |
| `lrelu` | 384x512 | 4.43 | 11.07 | 0.77 | 0.00 | 0.01 | 16.34 |
| `lrelu` | 288x384 | 2.49 | 7.34 | 0.59 | 0.00 | 0.01 | 10.49 |

### 6.3 DPU op mix and artifact

| Activation | Input | `conv2d` | `eltwise` | `hard-sigmoid` | total DPU ops | `.xmodel` bytes |
|---|---|---|---|---|---|---|
| `hswish` | 480x640 | 72 | 69 | 63 | 389 | 6,126,505 |
| `hswish` | 384x512 | 72 | 69 | 63 | 389 | 5,919,609 |
| `hswish` | 288x384 | 72 | 69 | 63 | 389 | 5,849,926 |
| `relu` | 480x640 | 80 | 6 | 0 | 271 | 5,799,140 |
| `relu` | 384x512 | 80 | 6 | 0 | 271 | 5,739,501 |
| `relu` | 288x384 | 80 | 6 | 0 | 271 | 5,706,063 |
| `lrelu` | 480x640 | 80 | 6 | 0 | 271 | 5,801,618 |
| `lrelu` | 384x512 | 80 | 6 | 0 | 271 | 5,741,979 |
| `lrelu` | 288x384 | 80 | 6 | 0 | 271 | 5,708,541 |

The op counts do not depend on the input size, only on the activation. `relu` and
`lrelu` have eight more `conv2d` ops and 16 fewer `strided_slice` ops than
`hswish`: `C2f` splits `cv1` into two channel halves (`chunk(2, 1)`), so with the
activation folded into the convolution the compiler emits one conv per half
(`strided_slice_0_16`/`_16_32`) instead of one conv plus a slice. The remaining 6
`eltwise` ops are `C2f` Bottleneck residual adds (backbone `C2f_2`, `C2f_4`,
`C2f_6`, `C2f_8`); the other residual adds do not appear as separate DPU ops.
The 63 `hard-sigmoid` + 63 `eltwise` of `hswish` are its activation, which has no
fused form on this DPU.

### 6.4 Reading

- **The activation is the first lever.** At 480x640, replacing Hard-Swish with
  ReLU or LeakyReLU drops the DPU from 25.32 to 16.99 ms (-33 %) and end to end
  from 33.29 to 24.97 ms (1.33x, 30.0 to 40.1 fps). That is the 63
  `hard-sigmoid` + 63 muls (and the intermediate tensors they need) replaced by
  a fused activation, not a change in convolution work.
- **LeakyReLU(0.1015625) is free against ReLU** on this target: 16.98 vs 16.99 ms
  at 480x640, 11.07 vs 11.09 at 384x512, 7.34 vs 7.39 at 288x384 (0.1-0.7 %).
  Speed gives no reason to prefer one over the other, so the choice is
  accuracy-only.
- **Input size is the larger lever, and the DPU part is close to linear in
  pixels.** 384x512 is 0.64 of the pixels and cuts DPU time 1.58x (`hswish`) and
  1.53x (`relu`/`lrelu`); 288x384 is 0.36 and cuts it 2.47x and 2.30-2.31x. The
  scaling is slightly sublinear because per-op overhead stays fixed.
- **The end-to-end path follows the DPU.** With the threshold-first decoder the
  non-DPU stages are small and also shrink with the input: preprocess (no-op
  letterbox + int8 LUT) 6.9 / 4.4 / 2.5 ms, empty-frame decode 0.98 / 0.77 /
  0.59 ms, NMS and unletterbox ~0. So fps goes 30.0 -> 40.1 -> 61.2 -> 95.3 from
  `hswish` 480x640 to `lrelu` 288x384, a 3.17x end-to-end speedup.
- **A trained frame adds NMS.** These runs decode zero candidates; section 5
  measured 1-2 ms of NMS for a trained frame with one or two persons and 14.5 ms
  with 300 survivors, which is the same absolute cost in every row above.

Speed therefore favours `lrelu` (or `relu`) at the smallest input that keeps the
keypoints usable; the accuracy of those variants has to be measured with trained
checkpoints before choosing.
