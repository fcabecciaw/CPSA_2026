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
model or compilation problem: moving the decode to C++/VART post-processing (or
cutting its candidate count) is the next runtime task. `detections/frame` is
0.0 because the randomly initialised head scores below `conf_thr = 0.25`; the
host check exercises the same decode with `conf_thr = -1`.

Operational note: `show_dpu`/`xdputil query` failed once with
`cannot open file: /tmp/DPU_0` from `file_lock_lnx.cpp`; a stale zero-length
`/tmp/DPU_0` lock caused it and removing the file fixed the query. Not related
to the model.
