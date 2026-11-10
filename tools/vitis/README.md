# `tools/vitis` — KV260 DPU quantize and compile pipeline

Reproducible PC-side pipeline that takes a Hard-Swish YOLOv8-pose model
(`HALFBUST_62`, 62 keypoints, input `1x3x480x640`) through Vitis AI 3.5 PTQ
into an `.xmodel` compiled for the KV260 DPU (`DPUCZDX8G_ISA1_B4096`,
fingerprint `0x101000016010407`).

Vitis AI 3.5's `DPUCZDX8G_ISA1_B4096` target alias maps to fingerprint
`0x101000056010407` (the 3.x feature code), which the 2.5-era KV260 B4096
firmware rejects with a fingerprint mismatch. `compile.sh` therefore compiles
against an explicit `{"fingerprint": "0x101000016010407"}` arch file and reads
the fingerprint back from the artifact; override it with
`VITIS_DPU_FINGERPRINT`, or pass a complete arch file with `VITIS_ARCH_JSON`.

The graph is cut before DFL and the pose decode: every stride (8, 16, 32)
returns its raw head tensor with channels

```text
[ box (4 * reg_max = 64) | class logits (nc) | keypoints (3 * K = 186) ]
```

which is exactly what `cpsa.vision.pose_decode.decode_raw_heads` and
`cpsa.adapters.pose.dpu.DpuPoseEstimator` consume. DFL softmax, sigmoid, anchor
decode and NMS stay on the CPU in `cpsa`.

The deploy graph lives in `ml/src/cpsa_ml/deploy.py` (`RawHeadDeploy`, exports
and tests in `ml/tests/test_deploy.py`).

## Images

| Image | Tag | Purpose |
|---|---|---|
| `xilinx/vitis-ai-pytorch-cpu` | `ubuntu2004-3.5.0.306` | base: Vitis AI 3.5 compiler + `pytorch_nndct` |
| `cpsa-vitis-pytorch` | `3.5.0.306` | the base plus `ultralytics==8.4.203` (no deps) |

Base digest:
`sha256:baf43fd9d6e1ea5a2c12b727ac3aafd15ead936172bcd636d7af98b0cca49298`.

The container Python is 3.8.6 with PyTorch 1.13.1 and `pytorch_nndct` 3.5.0.
`cpsa_ml` targets 3.10, so the container does not import the package: it loads
`ml/src/cpsa_ml/deploy.py` by path, rebuilds the same `PoseModel` from the
exported YAML, and loads the host `state_dict` strictly. `run_vitis.sh` builds
the derived image on first use.

## Full run

```sh
# 1. Export the deploy bundle on the host (ml environment, PyTorch 2.x).
cd ml
uv run python -m cpsa_ml.deploy --schema halfbust62 --seed 0 \
  --out ../build/vitis/artifacts
cd ..

# 2. Calibration frames: camera captures + augmentations + synthetic, letterboxed
#    to 480x640, RGB, [0, 1], saved as (3, H, W) float32 .npy.
uv run python tools/vitis/make_calib.py --count 120 --out build/vitis/calib

# 3. Calibrate, test and export the quantized xmodel (vai_q_pytorch).
#    --threads caps OMP/MKL/torch: one thread is fastest on a busy host.
tools/vitis/run_vitis.sh python tools/vitis/quantize.py \
  --artifacts build/vitis/artifacts --calib build/vitis/calib \
  --out build/vitis/quant --calib-count 120 --test-count 24 --threads 1

# 4. Compile for the KV260 DPU (vai_c_xir).
tools/vitis/run_vitis.sh bash tools/vitis/compile.sh \
  build/vitis/quant/pose_halfbust62.xmodel build/vitis/compiled pose_halfbust62

# 5. Inspect subgraphs, tensors and fix-points.
tools/vitis/run_vitis.sh python tools/vitis/inspect_xmodel.py \
  build/vitis/compiled/pose_halfbust62.xmodel \
  --out build/vitis/report/inspect_compiled.json

# 6. Decoder compatibility (host: deploy graph vs Ultralytics, quantized
#    test outputs through decode_raw_heads with the runtime int8 helpers).
cd ml
PYTHONPATH=../src uv run python ../tools/vitis/verify_deploy.py \
  --artifacts ../build/vitis/artifacts --quant ../build/vitis/quant \
  --inspect ../build/vitis/report/inspect_compiled.json \
  --out ../build/vitis/report/decoder_check.json
```

Outputs land in `build/vitis/` (git-ignored): `artifacts/`, `calib/`,
`quant/`, `compiled/`, `report/`, `logs/`.

Observed for the randomly initialised `HALFBUST_62` model (see
`docs/notes/vitis-compile.md` for details):

| Stage | Result |
|---|---|
| Calibration | 120 frames, 133 s (`--threads 1`), test 24 frames |
| Quantized vs float | cosine 0.99990-0.99992, max abs error 0.031 |
| Compiled graph | 5 subgraphs, 1 DPU (389 ops), no compute on CPU |
| DPU tensors | input `1x480x640x3` int8 `fp=7`; outputs `1x{H,W}x251` int8 `fp=4` |
| Host decoder check | float graph matches Ultralytics within `3.1e-5 px`; int8 outputs decode with the same candidate counts |
| Board | 289 ms/frame end to end, 3.46 fps; DPU runner alone 25.3 ms/frame |

## Running with trained weights

The pipeline takes the Kaggle `best.pt` unchanged:

```sh
cd ml
uv run python -m cpsa_ml.deploy --schema halfbust62 --weights /path/to/best.pt \
  --out ../build/vitis/artifacts
```

`load_deploy_weights` accepts the Ultralytics `{"model": ...}` checkpoint
layout and loads it strictly, so the architecture and the keypoint schema must
match (`halfbust62`, `nc = 1`). The rest of the pipeline is unchanged.

## Calibration data

`make_calib.py` reads the camera captures in
`/home/fabio/cpsa-session-state/scratchpad/cap` (override with `--source`,
repeatable), augments them (flip, brightness, contrast, crop, rotation, noise),
adds seeded synthetic frames, and letterboxes every frame with
`cpsa.vision.pose_decode.letterbox` — the same preprocessing as
`DpuPoseEstimator` and `OnnxPoseEstimator`. Use `--count` for the real run
(100–200 frames); `--height/--width` follow the model input.

## Board

The board (Kria KV260, Vitis AI 3.5 VART built natively, see
`docs/notes/vart-kv260.md`) must have the DPU app loaded
(`sudo xmutil loadapp kv260-benchmark-b4096`); `xdputil query` must report
`DPUCZDX8G_ISA1_B4096_0101000016010407` / fingerprint `0x101000016010407`.

```sh
scp build/vitis/compiled/pose_halfbust62.xmodel kria@10.42.0.83:~/bench/
# On the board:
cd ~/bench && PYTHONPATH=src python3 scripts/bench_pose.py \
  --xmodel pose_halfbust62.xmodel --layout halfbust62 --frames 200
```

`scripts/bench_pose.py` reports mean/median/p95 latency and fps with the
`DpuPoseEstimator` (VART runner + `decode_raw_heads` on the CPU). Measured
here: 289.0 ms mean, 291.5 ms p95, 3.46 fps for the full estimator, and
25.3 ms/frame for the DPU runner alone; the Python decode is ~182 ms of the
total. If `show_dpu`/`xdputil query` fails with
`cannot open file: /tmp/DPU_0`, remove the stale `/tmp/DPU_0` lock file.
