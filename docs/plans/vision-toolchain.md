# CPSA vision data and model toolchain — Implementation Plan

**Status:** READY

**Goal:** Provide everything the video pipeline needs before it can run on the board:

- synchronized RGB-D and wrist-IMU recording with scripted trials;
- camera calibration;
- datasets: a public half-bust training set, teacher pseudo-labels on our recordings, and PTQ calibration frames;
- 3D keypoint trajectory smoothing;
- IMU–vision fusion: clock alignment and contact events;
- an importable Kaggle protocol for training, fine-tuning, and QAT;
- a Vitis AI 3.5 quantize-and-compile flow;
- DPU bring-up on the KV260;
- an offline evaluation that decides whether a factor graph is needed.

**Spec:** design brainstorm of 2026-11-06 (vision role, camera checks, model review, smoothing and data-collection decisions).

**Scope / non-goals**

Out of scope:

- the on-board episode runtime: trigger, pre-roll ring buffer, `VisionReport` into the `Dispatcher`, policy shaping;
- the GUI;
- deploying the Option 1 hand and face models (MediaPipe hand, 3DDFA_V2) on the DPU;
- FINN or any other custom programmable-logic accelerator;
- implementing a factor graph. Only the go/no-go evidence for one is in scope.

Sensor fusion in this plan means clock alignment and event-level fusion. Tight IMU–vision state fusion is deferred to the Task 11 decision.

**Constraints**

Camera:

- The Creative VF0800 is an Intel RealSense F200, used through stock V4L2 on both the PC and the KV260.
- `/dev/video0` gives RGB as YUYV; `/dev/video2` gives depth as `Z16`.
- **Mode:** RGB 640×480 @ 30 fps and depth 640×480 @ 30 fps. This is also the model input size, so frames are not resized.
- Depth extension-unit (XU) settings: unit 6, GUID `a55751a1-f3c5-4a5e-8d5a-6854b8fa2716`. The values are `laser_power=16`, `confidence=1`, `filter=6`. They are volatile and must be set every time the camera is opened.
- RGB uses manual exposure (`auto_exposure=1`). Auto exposure drops the frame rate to 15 fps in dim light.
- On the KV260 the camera must be behind a powered USB hub. Without one, the SD card fails.

Demo and timing:

- The subject is seated at 0.45–0.7 m from the camera.
- Every timestamp is `CLOCK_MONOTONIC` seconds within one process (`time.monotonic()` and V4L2 buffer timestamps).

Accelerator:

- The KV260 has no AI Engines, so it has no NPU. The only accelerator is the DPUCZDX8G with Vitis AI 3.5.
- The DPU target is `DPUCZDX8G_ISA1_B4096_0101000016010407` (fingerprint `0x101000016010407`). It is provided by the `kv260-benchmark-b4096` firmware app.
- Models must be conv-only and use only ReLU, LeakyReLU(26/256), Hard-Swish, or Hard-Sigmoid. SiLU, MatMul, and Softmax are not allowed in the backbone or neck.
- Quantization is INT8, symmetric, with power-of-two scales.

Training:

- Training runs on Kaggle with 2× T4 GPUs and DDP. The TPU v5e-8 is not used because Ultralytics has no XLA path.
- Kaggle limits sessions to 12 h and gives about 30 GPU-hours per week, so training must checkpoint and resume.

PC host:

- Docker 29.8 is available.
- The 6 GB Blackwell laptop GPU is not supported by the CUDA 11.8 Vitis image, so the CPU image `xilinx/vitis-ai-pytorch-cpu` (3.5) is used.
- About 42 GB of disk is free.

Repository:

- Layer rules are enforced by `tests/architecture/test_layers.py`. Every new `cpsa` subpackage needs a `RULES` entry.
- `.xmodel` files must never be tracked (`tests/architecture/test_hygiene.py`).
- ruff line length is 100. mypy runs strict for the non-adapter layers.

Native IMU libraries:

- `libProcessDataWristsQuat.so` and `libPredictPericolosaWristsQuat.so` are aarch64-only.

Labeling:

- No frame-by-frame manual labeling. Labels come from the protocol, IMU peaks, depth, and a teacher model. Recording our own data must stay minimal.

## Approach

### Package layout

| Location | Responsibility | Layer rule |
|---|---|---|
| `src/cpsa/vision/` (new) | Pure numpy: RGB/depth pairing, calibration model and (de)projection, depth lifting, `KeypointTracker` (EKF + RTS), fusion (clock offset, occluder, contacts) | `"vision": (frozenset({"domain"}), frozenset({"numpy"}))` |
| `src/cpsa/domain/model.py` | Add `FaceRegion`, `ContactEvent` (stdlib only) | unchanged |
| `src/cpsa/ports.py` | Add `RgbdFrame`, `FrameSource` | unchanged |
| `src/cpsa/adapters/camera/f200.py` (new) | V4L2 mmap capture of both nodes with buffer timestamps; XU settings via `UVCIOC_CTRL_QUERY` | adapters |
| `src/cpsa/adapters/recording/` (new) | Session writer and reader (format below) | adapters |
| `src/cpsa/recorder/` (new) | Composition root and the `cpsa-record` CLI (record, calibration capture) | same as `board` |
| `src/cpsa/board/replay.py` (new) | `cpsa-replay-imu`: session `imu.csv` → sync → window → native classifier → `tags.jsonl` (aarch64) | board |
| `tools/calibration/` | `charuco_board.py` (exists, untracked), `solve.py` (OpenCV ChArUco) | standalone scripts |
| `ml/` (new, own `pyproject.toml`, package `cpsa_ml`) | Dataset builders, keypoint schema, training/teacher wrappers, QAT, ONNX checks, evaluation, Vitis scripts; `ml/notebooks/*.ipynb` are thin wrappers importing `cpsa_ml` | outside the `cpsa` layers |

### Key decisions

1. **Recording runs on the PC, not the KV260.**
   - The PC has more CPU and disk and no SD-card risk, and it uses the same V4L2 driver.
   - The BlueCoins connect through the PC's Bluetooth. `cpsa-board` must not run on the KV260 at the same time, because a BLE peripheral accepts only one central.
   - Frames are written only inside trial windows, with ±1 s margins. The recorder prints MB/s.
2. **Labels without manual annotation.**
   - Trial windows come from the protocol.
   - Contact times come from IMU acceleration peaks plus depth occluder evidence.
   - Keypoints come from an RTMW-x teacher (Apache-2.0) run through `rtmlib` on Kaggle.
   - The operator only flags bad trials.
3. **Keypoint schema `HALFBUST_62`**, defined once in `cpsa_ml/schema.py` as indices into the 133 COCO-WholeBody keypoints. The COCO-17 schema is used for the Option 1 body model.

   | Group | Points | COCO-WholeBody indices |
   |---|---|---|
   | Body | nose, eyes, ears, shoulders, elbows, wrists | 0–10 |
   | Face | jaw 0, jaw 16, chin (jaw 8) | 23, 39, 31 |
   | Face | brow centres 19 and 24 | 42, 47 |
   | Face | mouth corners 48 and 54 | 71, 77 |
   | Face | lips 51 and 57 | 74, 80 |
   | Left hand | 21 points | 91–111 |
   | Right hand | 21 points | 112–132 |

   Face points use WholeBody index = 23 + iBUG68 index. `flip_idx` is derived from this table and tested for symmetry.
4. **Smoothing.**
   - Each keypoint gets a 3D constant-velocity EKF in camera coordinates.
     - When depth is valid, the measurement is the 3D point.
     - Otherwise it is the 2D pixel, through the projection model.
   - Mahalanobis gating at χ²(3) = 11.34 for 3D and χ²(2) = 9.21 for 2D, both at p = 0.01.
   - Measurement noise scales with detector confidence.
   - **Causal mode** serves the fast verdict; an **RTS smoother** over the episode serves the report.
   - The head is tracked as face anchor points using the same tracker. Rigid head-pose filtering is deferred.
5. **Fusion.**
   - The IMU↔camera clock offset is the lag that maximizes the cross-correlation between IMU wrist acceleration magnitude and the RTS-derived wrist acceleration magnitude, searched over ±0.3 s.
   - Depth occluder test: the fraction of face-box pixels at least 3 cm in front of the face depth.
   - A `ContactEvent` is created when hand–face-anchor distance falls below a threshold. Two optional pieces of evidence are attached to it:
     - an IMU acceleration peak within ±Δ;
     - the occluder flag.
   - Thresholds are config values tuned in Task 11.
6. **Quantization path.**
   - The baseline is always available:
     1. FP32 training on Kaggle;
     2. local Vitis 3.5 Docker `vai_q_pytorch` PTQ with fast finetune;
     3. `vai_c_xir` compile against `arch_kv260_benchmark_b4096.json` (`{"fingerprint": "0x101000016010407"}`, the same file as `fcabecciaw/cpsa-example/files/arch_kv260_benchmark_b4096.json`).
   - QAT is used only if the PTQ accuracy drop exceeds the AC-09 gate:
     1. First choice: `pytorch_nndct` QAT on Kaggle, if the Task 8 install spike passes.
     2. Otherwise: Brevitas QAT with `Int8WeightPerTensorFixedPoint` / `Int8ActPerTensorFixedPoint`, which emulate power-of-two DPU scales. The QAT-adapted float weights then go through the same Vitis PTQ.
   - Brevitas is never used for export.
7. **Kaggle configuration.** The repository `github.com/fcabecciaw/CPSA_2026` is public.
   - **Code:** each notebook enables Internet and installs a pinned commit: `pip install "git+https://github.com/fcabecciaw/CPSA_2026@<commit-sha>#subdirectory=ml"`. The SHA is a notebook parameter so results stay reproducible.
   - **Data:** sessions, built datasets, and checkpoints are Kaggle **datasets**, versioned with the `kaggle` CLI and attached as inputs. Raw COCO 2017 comes from an existing public Kaggle dataset; the COCO-WholeBody JSON is uploaded once.
   - **Notebooks:** stored as `.ipynb` with a `kernel-metadata.json` each, pushed with `kaggle kernels push`. Accelerator: GPU T4 ×2. The exact metadata keys for the accelerator are to be verified in Task 8.
   - **Outputs:** checkpoints are written to `/kaggle/working` and re-published as dataset versions, so the next session resumes from them.
8. **DPU on the KV260** follows `fcabecciaw/cpsa-example` §5.1:
   - `xlnx-firmware-kv260-benchmark-b4096` is installed through apt, after `xlnx-config.sysinit` if the package is not visible;
   - `xmutil loadapp kv260-benchmark-b4096`;
   - `xdputil query` and `show_dpu` are used to verify.

   That repository uses C++ VART. Python VART bindings on Ubuntu 22.04 Kria are not verified (see Task 10).

### Contracts

**`ports.py`**

```python
@dataclass(frozen=True)
class RgbdFrame:
    seq: int
    rgb: NDArray[np.uint8]  # (480, 640, 3), RGB order
    depth: NDArray[np.uint16]  # (480, 640), raw Z16, 0 = invalid
    ts_rgb: float  # CLOCK_MONOTONIC s, V4L2 buffer timestamp
    ts_depth: float


class FrameSource(Protocol):
    def start(self, on_frame: Callable[[RgbdFrame], None]) -> None: ...
    def stop(self) -> None: ...
```

Pairing takes the nearest timestamps within `1 / (2 · fps)`. Unpaired frames are dropped and counted.

**`vision.tracking`**

```python
@dataclass(frozen=True)
class TrackerConfig:
    accel_std: float
    meas_std_px: float
    meas_std_depth_rel: float
    gate_chi2_3d: float = 11.34
    gate_chi2_2d: float = 9.21
    max_missed: int = 15

def smooth(t, obs3d, obs2d, conf, intrinsics, cfg) -> Track
# t (T,), obs3d (T, K, 3) with NaN for missing, obs2d (T, K, 2), conf (T, K)
# Track: pos (T, K, 3), vel (T, K, 3), cov (T, K, 6, 6), valid (T, K)

class CausalTracker:
    def update(self, t, obs3d_k, obs2d_k, conf_k) -> TrackState: ...
```

**`vision.fusion`**

```python
def estimate_clock_offset(imu_t, imu_acc_norm, cam_t, cam_acc_norm,
                          max_lag_s: float = 0.3) -> tuple[float, float]  # (offset_s, peak corr)
def occluder_fraction(depth_m, face_box, face_depth_m, margin_m: float = 0.03) -> float
def detect_contacts(track, anchors, imu, occluder, cfg) -> list[ContactEvent]
```

**`domain/model.py`**

```python
class FaceRegion(Enum):  # EYES, MOUTH, NOSE_CHEEK, FOREHEAD_TEMPLE, EAR, HEAD_TOP

@dataclass(frozen=True)
class ContactEvent:
    t: float
    side: Side
    region: FaceRegion
    min_distance_m: float
    approach_speed_mps: float
    impact_peak_g: float | None
    occluder: bool
```

**Session format, schema v1** (`<root>/<session_id>/`)

| File | Content |
|---|---|
| `session.json` | `schema_version`, pseudonymous `participant_id`, camera modes and XU values, calibration file sha256, BlueCoin names, protocol sha256, notes |
| `calibration.json` | Copy of the calibration in use |
| `frames.csv` | `seq,trial_id,ts_rgb,ts_depth,rgb,depth` |
| `rgb/NNNNNN.jpg` | JPEG, quality 95 |
| `depth/NNNNNN.png` | uint16 PNG, raw Z16 |
| `imu.csv` | `ts,side,kind,v0,v1,v2,v3`, raw units as received (mg, dps, quaternion ×10000) |
| `trials.jsonl` | `trial_id,class,hand,target,cue_ts,start_ts,end_ts,status(ok\|bad),note` |

**`calibration.json` v1**

```json
{
  "schema_version": 1,
  "rgb": {"w": 640, "h": 480, "K": [[...]], "dist": [5 values]},
  "ir": {"w": 640, "h": 480, "K": [[...]], "dist": [5 values]},
  "T_ir_rgb": {"R": [[...]], "t_m": [3 values]},
  "depth": {"scale_m": 0.00003125, "offset_m": 0.0},
  "rms": {"rgb_px": 0.0, "ir_px": 0.0, "stereo_px": 0.0, "depth_mm": 0.0},
  "board": {"squares": [11, 8], "square_mm": 30.0, "marker_mm": 21.0, "dictionary": "DICT_5X5_100"}
}
```

The `vision.calibration` loader validates the file and raises a `ValueError` naming the bad key.

**`config/protocol.yaml`**

- Classes, each with `{hand: L|R|both, target, reps, duration_s}`.
- Dangerous classes: fist to temple/forehead, slap to cheek, finger poke or press on the eye, hand to mouth or bite, scratching the face.
- Confounders: scratching the nose, adjusting glasses, drinking, phone to ear, touching hair, covering a yawn, chin on hand.
- Non-dangerous stereotypies: hand flapping, rocking.
- Idle.
- `seed` and `distances_cm: [45, 55, 65]`.
- Trial order is randomized and deterministic for a given seed.
- Hits are simulated as soft contact. Hard impacts use a foam mannequin head.

**Board config.** `config/board.yaml` gets a required `camera:` section: device paths, modes, XU values, and exposure. It is shared by `cpsa-record` and the future board runtime. The README configuration table must be updated, and so must the copy deployed on the KV260.

## Verification contract

**AC-01 — camera**
- Expected:
  - `F200Source` delivers paired 640×480 RGB and depth at ≥ 29 fps for 60 s.
  - XU values read back as set.
  - Timestamps are monotonic and within ±50 ms of `time.monotonic()`.
- Evidence: a hardware check script on the PC; unit tests for `vision.pairing`.

**AC-02 — calibration**
- Expected:
  - RGB and IR reprojection RMS < 0.5 px.
  - Depth scale and offset fitted from board planes, with residual ≤ 5 mm at 0.4–0.7 m.
  - Re-measuring the 81 cm door target agrees within 2%.
- Evidence: the solve report; a unit test on synthetically projected boards that recovers known intrinsics within 1%.

**AC-03 — recorder**
- Expected:
  - A 10-trial dry run produces a schema-v1 session that the reader loads.
  - Frame drop rate < 1%. Drops are counted and logged.
  - The recorder reports the IMU rate per sensor, which settles the unknown IMU rate.
  - The recorder prints MB/s.
  - Ctrl-C finalizes all files.
- Evidence: an integration test with fake `FrameSource` and `SensorSource` and a writer→reader round trip; a real dry run.

**AC-04 — replay**
- Expected: on the KV260, `cpsa-replay-imu` produces the same tags as live classification of the same rows. It reuses `IMUSynchronizer`, `Windower`, and `ClassificationStage` unchanged.
- Evidence: a test marked `target`, with a short fixture.

**AC-05 — datasets**
- Expected:
  - The `HALFBUST_62` builder produces a YOLO-pose dataset with head-to-waist crops and synthetic hand-over-face occlusions. `flip_idx` passes the symmetry test.
  - Every recorded frame gets teacher pseudo-labels with confidences.
  - The PTQ calibration set has ≥ 500 frames, stratified by session and distance.
- Evidence: `cpsa_ml` unit tests; dataset statistics printed by the notebooks.

**AC-06 — smoothing**
- Expected:
  - On synthetic constant-velocity and impact trajectories with injected depth outliers and gaps, RTS position RMSE is ≤ 50% of the raw RMSE.
  - Outliers are rejected.
  - Gaps up to `max_missed` frames are bridged.
  - Causal lag at an impact stays bounded.
- Evidence: seeded, deterministic unit tests with statistical tolerances.

**AC-07 — fusion**
- Expected:
  - The clock offset is recovered within ±1 frame on synthetic signals with known lag.
  - Occluder and contact detection give the expected result for each case: contact present, contact absent, IMU-only peak.
- Evidence: unit tests.

**AC-08 — Kaggle**
- Expected:
  - Each notebook runs top to bottom on Kaggle T4×2 from a clean session, using only the pinned `cpsa_ml`.
  - Training resumes from a checkpoint dataset.
  - Exported ONNX contains no SiLU, MatMul, or Softmax in the backbone or neck.
- Evidence: notebook run logs; `cpsa_ml.onnx_check` output.

**AC-09 — Vitis**
- Expected:
  - The Docker scripts produce a compiled `.xmodel` for `DPUCZDX8G_ISA1_B4096_0101000016010407`.
  - The compile log shows the compute graph in DPU subgraphs, with only the allowed CPU post-processing outside them.
  - INT8 vs FP32 pose mAP drop is ≤ 3 points. A larger drop triggers the QAT path.
- Evidence: the `vai_c_xir` log; an accuracy comparison report.

**AC-10 — board DPU**
- Expected:
  - `xmutil listapps` shows `kv260-benchmark-b4096` active.
  - `xdputil query` reports fingerprint `0x101000016010407`; `show_dpu` runs without crashing.
  - Benchmark results (latency, fps) for a reference model and for our compiled model.
- Evidence: the benchmark log.

**AC-11 — evaluation**
- Expected: the offline chain (teacher or student keypoints → lifting → smoothing → fusion) over the recorded sessions reports:
  - contact precision and recall against protocol labels;
  - contact timing error;
  - false-positive rate on confounders.

  Factor-graph rule: GO only if RTS impact-timing error > 2 frames or wrist-track loss > 10% of contact frames.
- Evidence: a generated report.

**AC-12 — regression**
- Expected: `uv run pytest`, `uv run ruff check . && uv run ruff format --check .`, and `uv run mypy src` all pass. The layer test includes rules for `vision` and `recorder`.
- Evidence: command output.

## Tasks

### Task 1 — Camera adapter and frame pairing
**Depends on:** none
**Files / responsibilities:**
- `ports.py`: `RgbdFrame`, `FrameSource`.
- `vision/__init__.py`, `vision/pairing.py`.
- `adapters/camera/f200.py`.
- `application/config.py`: `CameraConfig`.
- `config/board.yaml` and the README.
- `tests/architecture/test_layers.py`: `vision` rule.
- `tests/fakes`: `FakeFrameSource`.

**Actions:**
1. Spike `linuxpy` for `Z16` and YUYV capture with buffer timestamps. If it fails, write a minimal ctypes/fcntl V4L2 mmap capture inside `adapters`.
2. Apply XU settings and manual exposure on every open. The probe logic from the brainstorm session (`UVCIOC_CTRL_QUERY`, unit 6, selectors 1–7) is the reference.
3. Run one capture thread per node and pair frames in `vision.pairing`.
4. On disconnect, log the error and raise to the caller of `start`.

**Acceptance:** AC-01, AC-12.

### Task 2 — Calibration capture and solve
**Depends on:** Task 1 and the printed A3 board
**Files / responsibilities:**
- `recorder` subcommand `calib-capture`: synchronized RGB plus IR (`Y10` on `/dev/video2`) with the laser off for IR frames, then depth with the laser on.
- `tools/calibration/solve.py`.
- `vision/calibration.py`: loader, `deproject`, `project`, `depth_to_m`.
- Tests.
- Commit `tools/calibration/charuco_board.py`.

**Actions:**
1. ChArUco detection.
2. Per-camera calibration, then stereo extrinsics.
3. Robust depth scale and offset fit: board plane distance from the board pose vs measured depth over the board pixels.
4. Write `calibration.json`.

**Acceptance:** AC-02.

### Task 3 — Recorder and protocol runner
**Depends on:** Task 1. Task 2 is needed for real sessions, not for the code.
**Files / responsibilities:**
- `adapters/recording/{writer,reader}.py`.
- `recorder/__main__.py`: `cpsa-record record --config --protocol --participant --out`.
- `config/protocol.yaml`.
- `pyproject.toml`: a `recorder` extra (`linuxpy` if chosen, `opencv-python-headless`) and the script entry.
- README section.

**Actions:**
1. One writer thread behind a bounded queue. When the queue overflows, drop frames and count them.
2. On-screen cues plus a beep through `aplay`; check it exists and fall back to the terminal bell.
3. Operator keys: `b` marks a trial bad, `p` pauses, `q` finalizes.
4. IMU via `BlueCoinSource` with the existing `BlueCoinsConfig`.
5. Verify the `board` extra installs on the PC: `bluepy` needs `libglib2.0-dev`, and scanning needs `cap_net_admin`. Document the setup.

**Acceptance:** AC-03.

### Task 4 — IMU replay on the board
**Depends on:** the Task 3 reader
**Files / responsibilities:** `board/replay.py`, the `pyproject.toml` script entry, a target test.
**Acceptance:** AC-04.

### Task 5 — Dataset generation (`ml/`)
**Depends on:** 5a has no dependencies. 5b and 5c depend on the Task 3 format and on Task 2 for 3D.
**Files / responsibilities:**
- `ml/pyproject.toml`.
- `ml/src/cpsa_ml/{schema,teacher}.py`.
- `ml/src/cpsa_ml/datasets/{halfbust,sessions,calib_frames}.py`.
- `ml/notebooks/01_build_halfbust.ipynb`, `02_pseudolabel_sessions.ipynb`.
- `ml/scripts/kaggle_push_dataset.sh`.

**Actions:**
- **5a.** Build COCO-WholeBody (COCO 2017 images plus WholeBody JSON) into YOLO-pose `HALFBUST_62` and COCO-17 variants. Use head-to-waist crops and synthetic hand-over-face pastes. UBody is optional and research-only.
- **5b.** Turn sessions into a trial dataset:
  - RTMW-x pseudo-labels (133 keypoints mapped to `HALFBUST_62`);
  - depth-lifted 3D per keypoint;
  - automatic contact times.
- **5c.** Build the stratified PTQ calibration frame set.
- **Upload.** Push sessions as Kaggle dataset versions. Local copies may then be deleted to respect the 42 GB disk limit.

**Acceptance:** AC-05.

### Task 6 — Trajectory smoothing
**Depends on:** `Intrinsics` from Task 2. Can start against a stub.
**Files / responsibilities:** `vision/tracking.py`, `tests/unit/test_tracking.py`.
**Acceptance:** AC-06.

### Task 7 — Sensor fusion
**Depends on:** Task 6
**Files / responsibilities:**
- `vision/fusion.py`.
- `domain/model.py`: `FaceRegion`, `ContactEvent`.
- Face-anchor to region mapping, from the `HALFBUST_62` face points.
- Tests.

**Acceptance:** AC-07.

### Task 8 — Kaggle training protocol
**Depends on:** 5a. The fine-tuning step also needs 5b.
**Files / responsibilities:**
- `ml/src/cpsa_ml/{models,train,qat,onnx_check}.py`.
- Model YAMLs: YOLOv8-pose with `activation: nn.Hardswish()`, attention-free.
  - `kpt_shape: [17, 3]` for Option 1.
  - `kpt_shape: [62, 3]` for Option 2.
- Notebooks `03_train_fp32`, `04_finetune_pseudolabels`, `05_qat`, each with `kernel-metadata.json`.

**Actions:**
1. DDP training with `device=[0, 1]`. Checkpoints go to `/kaggle/working` and are re-published as dataset versions.
2. Run the `pytorch_nndct` install spike on Kaggle and record the outcome. Brevitas fixed-point QAT is the fallback.
3. Export ONNX and `.pt`. `onnx_check` rejects SiLU, MatMul, and Softmax in the backbone or neck.
4. Verify the `kernel-metadata.json` accelerator keys for T4×2.

**Acceptance:** AC-08.

### Task 9 — Vitis AI 3.5 quantize and compile (PC Docker)
**Depends on:** Task 8. The DPU fingerprint is already known; Task 10 confirms it on the board.
**Files / responsibilities:**
- `ml/vitis/{run_docker.sh,quantize.py,compile.sh,README.md}`.
- `ml/vitis/arch_kv260_benchmark_b4096.json` (`{"fingerprint": "0x101000016010407"}`).

**Actions:**
1. Check free disk, then pull `xilinx/vitis-ai-pytorch-cpu` 3.5.
2. PTQ calibration with fast finetune on the 5c frames.
3. Export the quantized xmodel in test mode.
4. `vai_c_xir --arch ml/vitis/arch_kv260_benchmark_b4096.json`.
5. Evaluate INT8 against FP32.
6. Keep all artifacts out of git.

**Acceptance:** AC-09.

### Task 10 — DPU bring-up on the KV260
**Depends on:** none. Authorized by the user: follow `fcabecciaw/cpsa-example` §5.1.
**Actions (on the board; `kria` user; board at `10.42.0.83`):**
1. Check package visibility with `apt search xlnx-firmware-kv260`. If `xlnx-firmware-kv260-benchmark-b4096` is missing:
   1. `sudo snap install xlnx-config --classic --channel=2.x`
   2. `sudo xlnx-config.sysinit`
   3. `sudo apt update`

   When asked about the modified flash-kernel file, keep the local version.
2. `sudo apt install -y xlnx-firmware-kv260-benchmark-b4096`. Keep `dfx-mgr.service` selected in the restart dialog. Then `sudo systemctl restart dfx-mgrd`.
3. Load the app:
   1. `sudo xmutil unloadapp`
   2. `sudo xmutil loadapp kv260-benchmark-b4096`
   3. `sudo xmutil listapps`
4. `xdputil query` must report fingerprint `0x101000016010407`. Then run `show_dpu`.
5. Run the reference model from `cpsa-example` (`run_all_target.sh kv260`), then the Task 9 model.
6. Check whether Python VART bindings (`import vart, xir`) are available. If not, fall back to a small C++/pybind11 runner or the C++ VART app pattern from `cpsa-example`, and record this as a decision for the future runtime plan.
7. Reboot if apt reports a kernel update. Re-check that `/` is mounted rw and that the camera works through the powered hub.

**Acceptance:** AC-10.

### Task 11 — Offline evaluation and factor-graph decision
**Depends on:** Tasks 5b, 6, and 7. The Task 8 student model is optional.
**Files / responsibilities:** `ml/src/cpsa_ml/evaluate.py`, notebook `06_evaluate`.
**Acceptance:** AC-11.

### Parallel boundaries

- Tasks 1, 5a, 6, and 10 can start immediately and in parallel.
- Task 7 follows Task 6.
- Tasks 2 and 3 follow Task 1.
- Tasks 4, 5b, and 5c follow Task 3.
- Task 8 follows 5a.
- Task 9 follows Task 8.
- Task 11 is last.

## Integration verification

- **Offline chain:** one real session → 5b → Tasks 6 and 7 → Task 11 report.
- **Checks:** `uv run pytest`, ruff, and mypy after every `src/` task.
- **Board:** the target suite on the KV260 for Task 4.
- **Board config:** after the `camera:` section is added to `board.yaml`, `cpsa-board` must still start with the updated config. The copy deployed on the KV260 must be updated too.

## Open decisions / risks

1. **Storage.** Recording writes about 6 MB/s. Five participants × 150 trials × 5 s is about 23 GB. Sessions must be uploaded to Kaggle and removed locally, or written to an external disk.
2. **Licenses.**
   - Ultralytics is AGPL-3.0, isolated in `ml/`. It is fine for academic use, but it affects redistribution of the trained weights.
   - UBody is research-only.
   - The native IMU libraries are under the MATLAB Academic licence, as already documented.
3. **QAT feasibility.** Neither path is proven: `pytorch_nndct` on Kaggle's PyTorch version, or Brevitas FX tracing of Ultralytics models. The PTQ baseline does not depend on either.
4. **Python VART on the board.** Not verified (Task 10, action 6). This affects the future runtime design, not this plan's deliverables.
5. **BLE latency.** IMU timestamps are arrival times, so per-session clock-offset estimation is needed (Task 7). Jitter may limit contact-timing accuracy; Task 11 measures it.
6. **Ethics.** Every participant must give written consent. Check whether the university requires ethics approval.
7. **Breaking config change.** The new required `camera:` section breaks any `board.yaml` that lacks it, including the copy deployed on the KV260, until it is updated.
8. **apt on the board.** `xlnx-config.sysinit` and the firmware install may pull a kernel update and need a reboot. The SD card already had one read error, so back up the board's data before Task 10.
