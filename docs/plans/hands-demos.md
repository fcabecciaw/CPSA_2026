# Hand demos: "Mano olografica" and "Sasso-carta-forbice" — Implementation Plan

**Status:** READY. Execution has two authorization gates, listed in "Open decisions": G1 and G2.

**Goal:** two new supervisor modes on the KV260. Both use one F200 camera pointing down at a table, and one DPU hand model.
- `hologram`: every hand seen from above becomes a glowing 3D hand in the browser. Its 21 joints follow the real fingers.
- `rps`: rock/paper/scissors against the board. A countdown runs, the player shows a gesture, and the board shows its own move and the score.

**Requested by:** the user, 2026-11-11:
- "Right now medical is fully deprecated and must be kept as is."
- "prepare the codebase for the hologram, meaning that we probably need the notebook so that we can train immediately";
- "prepare a rock/paper/scissor game which should be very easy to do given the hagrid dataset. We just need a classifier for that, and probably there's already one we can reuse";
- "we work in a new worktree".

**Supersedes:** `plan-hand3d-training.md`. That plan's own recordings and IR labelling are dropped: the user has no time to prepare a dataset.

**Non-goals:**
- **Medical mode stays as it is:** no behaviour change. The only shared edits are additive: new values in the mode list, new message types, new mode buttons.
- No IMU, no actuators.
- No RGB-to-depth calibration.
- No new web runtime dependency (no three.js).

**Constraints:**
- **Deadline:** booth on Friday 2026-11-13. The model must train on Thursday 2026-11-12.
- **Hardware:** KV260 with the DPU B4096 and Vitis AI 3.5; the F200's depth node is `/dev/video3` today (match the camera by name, not by number).
- **Training:** Kaggle 2×T4. Notebooks install `cpsa-ml` from GitHub at a pinned commit (`COMMIT_SHA`), so the code must be pushed before the training run.
- **Labels:** web UI text in Italian, as in the current dashboard.

## Repository evidence the design relies on

- **`cpsa.vision.pose_decode`:** `decode_raw_heads` and `QuantizedRawHeadDecoder` work for any keypoint count `n_kpt` and derive the class count from the channel count. `DpuPoseEstimator(model_path, layout, conf_thr, iou_thr)` (`src/cpsa/adapters/pose/dpu.py`) takes `len(layout)` as `n_kpt`. A `HAND21` layout is therefore enough on the board.
- **Supervisor modes:** `cpsa.supervisor.contract.Mode` is a frozen interface: `start(ctx)`, `stop(timeout_s)`, `health()` with keys `HEALTH_KEYS`, and `ModeStartError(code)` with the closed list `ERROR_CODES`. `supervisor_main.main` builds `modes = {"medical": ...}` and handles `mode.set`. `ModeName = Literal["idle", "lab", "medical"]`. The mode lists in `docs/protocol/v1/{hello,status,mode.set,mode.progress}.schema.json` repeat these names.
- **Mode buttons:** `web/src/main.ts` already has them (`elements.modeButtons`, `data-mode`) and dispatches messages by `type` (`medical.*`).
- **Camera:** `cpsa.adapters.camera.f200._open_video(..., fourcc)` opens a node with any fourcc through linuxpy.
- **`INZI` format, verified on the board on 2026-11-11:** 921,600 bytes per frame (640×480×3), interleaved per pixel. Bytes 0–1 are depth (Z16, little-endian) and byte 2 is IR, 8-bit. IR and depth are pixel-aligned.
- **Depth scale:** about 1/32 mm per unit (`docs`/memory; the earlier 2-point fit gave 0.0276 mm/unit).
- **IR test (2026-11-11):** a model trained only on RGB (RTMW) finds hands and plausible fingers on F200 IR frames once their contrast is boosted with CLAHE.
- **Training data:** `ml/src/cpsa_ml/datasets/hands.py`:
  - `iter_hagrid_samples` and `iter_ultralytics_samples` yield `HandSample(image_path, image_size, box, keypoints (21,3), source, origin_split, label, handedness)`;
  - for HaGRID, `label` is the per-hand gesture name (`fist`, `palm`, `peace`, `no_gesture`, …);
  - `_hagrid_hands` skips hands without valid landmarks.
- **Notebook 07** (`ml/notebooks/07_hand_pose.ipynb`) downloads the HaGRID 30k 384p sample, the HaGRIDv2 `annotations_with_landmarks`, and Ultralytics `hand-keypoints.zip` (about 2.1 GB). It builds 192 px crops and trains `yolov8n` with lrelu on 2×T4.
- **Notebook tests:** `ml/tests/test_notebooks.py` checks the `NOTEBOOKS` list, a pinned `COMMIT_SHA`, the thin-wrapper entry points, and `kernel-metadata.json`. `ml/scripts/sync_notebooks.py` copies each notebook into its Kaggle folder.
- **No gesture classifier exists** in the repo. `ml/scripts/gesture_demo.py` only draws the recorder's prompt animations.
- **OpenCV:** the board venv installs the `vision` extra, which includes `opencv-python-headless`. `cpsa_ml` does not depend on `cpsa`, so a shared constant must be duplicated and pinned by tests on both sides.

## Approach

### Model (hand detector + 21 keypoints)

- **Architecture:** YOLOv8n-pose with LeakyReLU, as in notebook 07. One class (`hand`), `kpt_shape [21, 3]`, `flip_idx` = identity.
- **Input size:** train at `imgsz=320` (square letterbox); export and compile at **256×320 (H×W)**.
- **Training data:** the whole frame, not crops. Each sample is a window around all the hands of one source image:
  - window aspect 5:4;
  - window size chosen so that the largest hand spans 25–60% of the window width, matching a hand seen from above at 50–70 cm;
  - window resized to 320×256;
  - **every** hand inside the window is labelled. A source image that has a hand without valid landmarks is skipped, so the model never learns to ignore an unlabelled hand.
- **Look like the IR image** — every training image gets, in order:
  1. converted to grayscale and replicated to 3 channels;
  2. random gamma 0.6–1.6 and contrast 0.6–1.4;
  3. Gaussian noise σ 0–8;
  4. Gaussian blur with probability 0.3;
  5. the **same CLAHE** as at runtime (min-max normalize, then `cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))`), with probability 0.8.
- **Training settings:** `hsv_h=0`, `hsv_s=0`, `hsv_v=0.3`, `degrees=180` (hands from above at any angle), `fliplr=0.5`, `mosaic=1.0`, `scale=0.5`, `epochs=60`, `batch=64`, `device="0,1"`.

### Runtime chain (board), shared by both modes

```
IrDepthSource(INZI 640x480@30) -> ir_to_input (CLAHE, 3ch) -> DpuPoseEstimator(HAND21, 256x320)
   -> lift_keypoints (depth median 5x5 -> metres -> IR pinhole) -> HandFrame -> mode callback
```

- **Nominal IR intrinsics** until a calibration exists: `fx = fy = 430`, `cx = 320`, `cy = 240` (from the F200 depth FOV of about 73°×59°). They are configurable. A hologram tolerates the error.
- **Depth model:** `scale_m = 3.125e-5`, `offset_m = 0`; configurable.

### Mode behaviour

**`hologram`:** publishes `hologram.hands` at the camera rate, at most 30 Hz.

**`rps`:** runs the same chain, plus `RpsClassifier` on each hand, plus the game state machine below. It publishes `rps.state` on each change, and at 10 Hz during the countdown.
- `waiting`: no hand.
- `ready`: one hand visible for 0.5 s.
- `countdown`: 3 → 2 → 1, 0.7 s per step.
- `shoot`: majority vote of the classifier over the 0.4 s after "1"; the board picks its move uniformly at random (`random.SystemRandom`).
- `result`: win, lose or draw, shown for 3 s, then back to `ready`.
- **Unclear gesture:** if the winning class is `none`, or its share of the votes is below 0.6, the result is `unclear` and the score stays the same.
- **Score reset:** after 20 s with no hand.
- **Multiple hands:** the largest hand plays.

### Gesture classifier (rock / paper / scissors / none)

A small MLP over the 21 keypoints in 2D, trained on the HaGRIDv2 **landmark annotations only**: the annotations JSON already holds about 1M hands with landmarks and gesture labels, so no images are needed. It trains on a CPU in minutes, on this PC.

**Class mapping:**

| class | HaGRID labels |
|---|---|
| `rock` | `fist` |
| `paper` | `palm`, `stop`, `stop_inverted` |
| `scissors` | `peace`, `peace_inverted` |
| `none` | every other label, `no_gesture` included |

Caution: HaGRID's own `rock` label is the "horns" sign, so it maps to `none`. `none` is subsampled to at most 2× the largest other class.

**Features (`rps_features`, v1):**
1. subtract the wrist;
2. rotate so that wrist → middle-finger knuckle (joint 9) points up;
3. divide by the wrist → joint-9 distance;
4. flatten the 21 (x, y) pairs into 42 values.

**Augmentation:** mirror x with probability 0.5 (covers left and right hands and the back view seen from above), rotate ±20°, scale ±10%, noise σ 0.02.

**Network:** 42 → 64 → 64 → 4, ReLU, softmax. Exported to `.npz`. The runtime is pure NumPy.

**Baseline:** a rule-based classifier (fingertip-to-wrist distance against knuckle-to-wrist distance, per finger) is reported next to the MLP in the evaluation. The plan keeps the MLP unless the rules score higher.

### Web

- Two new views, chosen by `status.mode`, plus "Ologramma" and "Sasso-carta-forbice" mode buttons. The medical view is unchanged.
- **Hologram view:** canvas 2D with a hand-written perspective projection:
  - virtual camera tilted 30° and slowly orbiting;
  - cyan glow (`shadowBlur`), joints as spheres, bones as lines, a faint grid on the table plane, a scanline overlay;
  - a joint without depth uses the hand's median depth.
- **RPS view:**
  - countdown "Sasso… carta… forbice!";
  - the player's gesture (icon and label) beside the board's move;
  - the result banner and the score;
  - a small IR skeleton preview from `kpts2d`.
- `mock.ts` replays both new message types, so the views can be built without the board.

## Frozen interfaces

All are additive. Each is pinned by the named test.

**F-A — `cpsa.vision.irdepth` (decode, constants) + `cpsa.adapters.camera.irdepth.ir_to_input` (cv2 CLAHE; moved to adapters because the `vision` layer may import only NumPy).** Pinned by `tests/unit/test_irdepth.py` (golden 2×2 `INZI` buffer; CLAHE parameters).

```python
INZI_BYTES_PER_PIXEL = 3
CLAHE_CLIP = 3.0; CLAHE_TILES = (8, 8)
def decode_inzi(data: bytes, width: int, height: int) -> tuple[NDArray[np.uint8], NDArray[np.uint16]]
    # (ir[H,W], depth_raw[H,W]); ValueError if len(data) != width*height*3
def ir_to_input(ir: NDArray[np.uint8]) -> NDArray[np.uint8]
    # min-max normalize (flat image -> zeros), cv2 CLAHE(CLAHE_CLIP, CLAHE_TILES), stack to (H,W,3)
```

`ml/src/cpsa_ml/datasets/handframe.py` duplicates `CLAHE_CLIP` and `CLAHE_TILES`. `ml/tests/test_handframe.py` asserts the same values.

**F-B — `cpsa.vision.hand3d`.** Pinned by `tests/unit/test_hand3d.py`.

```python
@dataclass(frozen=True) class Hand3D:
    score: float; box: tuple[float, float, float, float]        # IR pixels
    kpts2d: NDArray[np.float32]   # (21, 3) u, v, conf
    kpts3d: NDArray[np.float32]   # (21, 4) x, y, z metres (IR camera frame), valid 0/1
@dataclass(frozen=True) class HandFrame:
    seq: int; t: float; width: int; height: int
    hands: tuple[Hand3D, ...]; timing: Mapping[str, float]      # decode_ms, preprocess_ms, dpu_ms, post_ms, lift_ms
def lift_keypoints(kpts2d, depth_raw, intrinsics: Intrinsics, depth: DepthModel, window: int = 5) -> NDArray[np.float32]
    # median of nonzero raw depth in window around round(u,v); none -> valid 0, xyz 0
```

`Intrinsics` and `DepthModel` are reused from `cpsa.vision.calibration`.

**F-C — `cpsa.application.hand_perception.HandPerception`** (moved from `vision` to satisfy `tests/architecture/test_layers.py`; the constructor takes a required keyword `preprocess` callable, and `cpsa.board.hand_modes` passes `ir_to_input`). Pinned by `tests/application/test_hand_perception.py` with a fake source and a fake estimator.

```python
HandPerception(source: IrDepthSource-like, estimator: PoseEstimator, intrinsics: Intrinsics,
               depth: DepthModel, max_fps: float = 30.0)
    .start(on_frame: Callable[[HandFrame], None]) -> None    # own thread; drops frames when busy (latest wins)
    .stop(timeout_s: float) -> bool
    .health() -> Mapping[str, Any]                            # camera, fps, dpu_ms
```

**F-D — `cpsa.vision.rps_classifier`.** Pinned by `tests/unit/test_rps_classifier.py`, using the parity vector stored in the `.npz`.

```python
RPS_CLASSES = ("rock", "paper", "scissors", "none")
def rps_features(kpts2d: NDArray[np.float32]) -> NDArray[np.float32]   # (42,), v1 as above
class RpsClassifier:
    @classmethod def load(cls, path: Path) -> RpsClassifier
    def predict(self, kpts2d) -> tuple[str, float]           # (class, probability)
```

`.npz` keys: `w1 b1 w2 b2 w3 b3 feature_version classes test_input (N,21,3) test_proba (N,4)`.

`ml/src/cpsa_ml/gestures.py` duplicates `rps_features`. The runtime test asserts `predict_proba(test_input) ≈ test_proba` (atol 1e-5), which proves both sides compute the same features.

**F-E — protocol v1 additions.** New schemas live in `docs/protocol/v1/` with examples in `examples/`. The mode enums gain `"hologram"` and `"rps"`.

```jsonc
// hologram.hands
{ "seq": int, "t": float, "width": 640, "height": 480,
  "hands": [ { "score": float, "box": [x1,y1,x2,y2],
               "kpts2d": [[u,v,c] x21], "kpts3d": [[x,y,z] | null x21] } ],
  "timing": { "dpu_ms": float, "total_ms": float } }
// rps.state
{ "phase": "waiting"|"ready"|"countdown"|"shoot"|"result",
  "count": 3|2|1|null, "player": "rock"|"paper"|"scissors"|"none"|null,
  "player_p": float|null, "board": "rock"|"paper"|"scissors"|null,
  "outcome": "win"|"lose"|"draw"|"unclear"|null,
  "score": { "player": int, "board": int, "draws": int },
  "hand": { "kpts2d": [[u,v,c] x21] } | null }
```

Python builders live in `src/cpsa/protocol/hands.py` (`hologram_hands(frame)`, `rps_state(...)`). The TypeScript types go in `web/src/protocol.ts`.

## Verification contract

| ID | Expected | Evidence |
|---|---|---|
| AC-01 | **Dataset builder:** one 320×256 window per source image, gray 3-channel (all channels equal), every hand inside labelled, images with an unlabelled hand skipped, deterministic for a seed, IR-look and CLAHE applied as specified | `ml/tests/test_handframe.py` |
| AC-02 | **Notebook `08_hand_topdown`** passes the notebook tests (nbformat, pinned `COMMIT_SHA`, thin wrapper over `build_handframe_dataset`, metadata, sync) | `ml/tests/test_notebooks.py` |
| AC-03 | **Kaggle run:** training converges (val pose mAP50 rising, no NaN); `best.pt` and ONNX at 256×320 in the output | `results.csv`, output zip |
| AC-04 | **INT8 xmodel:** compiles with 1 DPU subgraph, input 1×256×320×3; FP32 vs INT8 hand PCK@0.2 reported on the val split; overlay sheet on the 50 IR frames captured on 2026-11-11 is plausible | `compile.log`, `eval_quant` table, contact sheet seen by the user |
| AC-05 | **Classifier:** accuracy ≥ 0.95 on the HaGRID **test** split for the 4 classes (target); confusion matrix and rule baseline reported; parity vector in the `.npz` | `cpsa_ml.gestures` report JSON |
| AC-06 | **Runtime units:** `decode_inzi` golden; `ir_to_input` shape and constants; `lift_keypoints` (valid median, all-invalid window, edge clipping); `HAND21` layout (21 names, 20 bones); `RpsClassifier` parity | `tests/unit/test_irdepth.py`, `test_hand3d.py`, `test_layouts.py -k hand21`, `test_rps_classifier.py` |
| AC-07 | **`HandPerception`:** the stream delivers `HandFrame`s from fakes; latest frame wins under load; stops within the timeout | `tests/application/test_hand_perception.py` |
| AC-08 | **Modes:** `hologram` and `rps` satisfy the `Mode` contract (start/stop/health keys; `camera_missing` and `model_missing` raise `ModeStartError`) | `tests/board/test_hand_modes.py` |
| AC-09 | **RPS state machine:** waiting → ready → countdown 3·2·1 → shoot → result → ready; majority vote; `unclear` when the vote share is < 0.6 or the class is `none`; score reset after 20 s; win/lose/draw table | `tests/unit/test_rps_game.py` (fake clock, fixed random source) |
| AC-10 | **Protocol:** new schemas validate the examples; the builders' output validates; mode enums include `hologram` and `rps` in the Python and TypeScript copies | `tests/unit/test_protocol_hands.py`, `tests/contract/` schema tests |
| AC-11 | **Medical regression:** medical behaviour unchanged; all existing tests pass unmodified, except the additive mode-enum and status tests | full gate |
| AC-12 | **Web:** typecheck and build pass; with the mock, the hologram view renders a moving 3D hand and the RPS view plays a round | `npm run typecheck && npm run build`; screenshot via `npm run dev` + mock |
| AC-13 | **Board, hologram:** at least 25 fps, end-to-end latency ≤ 100 ms, the 3D hand follows the fingers | dashboard Tempi line, user observation |
| AC-14 | **Board, rps:** 10 rounds by the user, with at least 8 gestures recognized correctly | user observation |

## Tasks

The order of priority is T1, then the Kaggle launch, then everything else. T1, T2, T3 and T5 run in parallel. T4 uses the frozen F-B, F-C and F-D with fakes, so it can run in parallel too. T6 and T7 come after training.

### T1 — full-frame IR-look hand dataset builder + notebook 08 (top priority)

**Depends on:** G1 (the base contains the medical work).

**Files:**
- new `ml/src/cpsa_ml/datasets/handframe.py`, which reuses `iter_hagrid_samples`, `iter_ultralytics_samples` and `HagridSource`/`UltralyticsHandSource` from `hands.py` without modifying them;
- new `ml/tests/test_handframe.py`;
- new `ml/notebooks/08_hand_topdown.ipynb` plus the folder `ml/notebooks/08_hand_topdown/` (copy of the notebook and `kernel-metadata.json`; id `fabiocabeccia/cpsa-hand-topdown`, GPU, internet on);
- `ml/scripts/sync_notebooks.py` and `ml/tests/test_notebooks.py`: add `08_hand_topdown` to `NOTEBOOKS`, `GPU_NOTEBOOKS`, the no-input map, and the thin-wrapper map (`build_handframe_dataset`).

**Requirements:** "we probably need the notebook so that we can train immediately"; "grayscale" training; full-frame top-down hand model.

**Actions:**
1. `HandFrameConfig(out, imgsz_hw=(256, 320), hand_frac=(0.25, 0.6), clahe_p=0.8, seed=0, max_images=None, val_ratio=0.05, overwrite=False)`.
2. `build_handframe_dataset(config, *, hagrid=None, ultralytics=None) -> dict` (stats):
   - group the samples by `image_path`;
   - drop an image if its HaGRID entry has more boxes than valid hands (needs a count of the raw boxes; add a small helper that reads the raw entry, not a change to `_hagrid_hands`);
   - pick the window, the IR-look transform and the CLAHE;
   - write YOLO-pose labels for every hand whose centre is inside the window; clip the keypoints and set `v=0` outside;
   - write `data.yaml` (`kpt_shape [21, 3]`, `flip_idx` = identity, `names {0: hand}`), `schema.json` and `stats.json` (with the licences, copied from `hands.py`).
3. Keep the source splits as they are (train → train; val/test → val), plus the deterministic `val_ratio`.
4. **Notebook 08**, built from notebook 07's cells:
   1. pin and install;
   2. download (same URLs);
   3. build with `build_handframe_dataset`;
   4. train with `TrainConfig(schema="hand21", variant="n", activation="lrelu", model_cfg=config_path("yolov8n-hand-lrelu.yaml"), imgsz=320, epochs=60, batch=64, device="0,1", extra={hsv_h: 0, hsv_s: 0, hsv_v: 0.3, degrees: 180, fliplr: 0.5, mosaic: 1.0, scale: 0.5})`;
   5. export `best.pt` plus `cpsa_ml.onnx_check.export_onnx(best, ..., imgsz=(256, 320))` into `/kaggle/working/hand-topdown-model`, together with the dataset `stats.json`.
   
   If `export_onnx` does not accept an `(h, w)` pair, extend it compatibly, with a test in `ml/tests/test_onnx_check.py`.
5. Run `python ml/scripts/sync_notebooks.py`.

**Acceptance:** AC-01, AC-02.

**Verification:**
- `cd ml && .venv/bin/python -m pytest -q tests/test_handframe.py tests/test_notebooks.py tests/test_hands_dataset.py tests/test_onnx_check.py`;
- a local smoke build on 50 images of the hand-keypoints set, producing a contact sheet of the labels.

### T2 — rock/paper/scissors classifier (training side)

**Depends on:** G1. Runs in parallel with T1, on different files.

**Files:** new `ml/src/cpsa_ml/gestures.py` (CLI `python -m cpsa_ml.gestures --annotations DIR --out models/rps_mlp.npz --report OUT.json`), new `ml/tests/test_gestures.py`.

**Requirements:** "a rock/paper/scissor game … given the hagrid dataset. We just need a classifier".

**Actions:**
1. Read the HaGRIDv2 `annotations_with_landmarks` JSONs (train/val/test) directly.
2. Map the labels as in Approach, compute the features (F-D v1), apply the augmentation, train the MLP with torch on the CPU (seeded), and early-stop on val.
3. Export the `.npz` with the F-D keys, including 16 parity samples.
4. Write the report: test accuracy, confusion matrix, the rule baseline.
5. Download `annotations.zip` (the URL is in notebook 07) to `build/data/hagrid_annotations/`, run once locally, and copy the result to `models/rps_mlp.npz` (small, committed).

**Acceptance:** AC-05; `.npz` < 100 KB.

**Verification:** `cd ml && .venv/bin/python -m pytest -q tests/test_gestures.py`, plus the report JSON.

### T3 — runtime core: INZI source, preprocessing, 3D lift, HAND21 layout, perception loop

**Depends on:** G1 (the base contains the medical work).

**Files:**
- new `src/cpsa/vision/irdepth.py` (F-A decode and constants; `ir_to_input` lives in `src/cpsa/adapters/camera/irdepth.py`);
- new `src/cpsa/vision/hand3d.py` (F-B);
- new `src/cpsa/application/hand_perception.py` (F-C);
- `src/cpsa/vision/layouts.py`: add `HAND21` (MediaPipe names `wrist, thumb_cmc…pinky_tip`, the 20 bones) to `LAYOUTS`; existing layouts unchanged;
- new `IrDepthSource` in `src/cpsa/adapters/camera/f200.py` or a new `irdepth.py` next to it: depth node chosen by the card-name match plus the `INZI` format; 640×480 at 30 fps; XU settings reused from `F200Source` (`confidence=1`, `filter=6`); delivers `(seq, t, ir, depth_raw)` from one thread;
- tests: `tests/unit/test_irdepth.py`, `tests/unit/test_hand3d.py`, `tests/unit/test_layouts.py` (hand21 cases), `tests/application/test_hand_perception.py`, `tests/contract/test_irdepth_source.py` (fake linuxpy stream, as in `test_uvc_source.py`).

**Acceptance:** AC-06, AC-07.

**Verification:** `uv run --no-sync pytest -q tests/unit/test_irdepth.py tests/unit/test_hand3d.py tests/unit/test_layouts.py tests/application/test_hand_perception.py tests/contract/test_irdepth_source.py`.

### T4 — modes, game, protocol, supervisor wiring, config

**Depends on:** F-B, F-C and F-D (frozen), so it uses fakes until T3 lands. Run it after T3 if the workers would conflict on `layouts.py`; T4 does not edit that file.

**Files:**
- new `src/cpsa/board/hand_modes.py` (`HologramMode`, `RpsMode`, both on one `HandPerception`, built from the config);
- new `src/cpsa/vision/rps_game.py` (pure state machine; injected clock and random source);
- new `src/cpsa/vision/rps_classifier.py` (F-D runtime);
- new `src/cpsa/protocol/hands.py` plus the schemas and examples (F-E);
- edits to `src/cpsa/supervisor/contract.py` (`ModeName` gains `hologram` and `rps`), the 4 mode-enum schemas, and `src/cpsa/protocol/system.py` if it lists modes;
- `src/cpsa/board/supervisor_main.py`: register the `hologram` and `rps` modes when the config section `hands` exists;
- `src/cpsa/application/config.py`: a new `HandsConfig` (optional `hands:` section with `model.path`, `model.input_height=256`, `model.input_width=320`, `conf_thr`, `iou_thr`, `camera.match`, `intrinsics {fx, fy, cx, cy}`, `depth {scale_m, offset_m}`, `rps.classifier_path`, `rps` timings);
- `config/board.demo.yaml`: add the `hands:` section; medical keys untouched;
- `deploy/deploy.sh`: copy `models/rps_mlp.npz`, and the hand xmodel when the new `--hand-model PATH` option is given;
- tests: `tests/unit/test_rps_game.py`, `tests/unit/test_rps_classifier.py`, `tests/unit/test_protocol_hands.py`, `tests/board/test_hand_modes.py`, the config tests.

**Acceptance:** AC-08, AC-09, AC-10, AC-11.

**Verification:** `uv run --no-sync pytest -q tests/unit/test_rps_game.py tests/unit/test_rps_classifier.py tests/unit/test_protocol_hands.py tests/board/test_hand_modes.py tests/application/test_config.py tests/supervisor tests/contract`.

### T5 — web views

**Depends on:** F-E (frozen). Runs in parallel.

**Files:**
- `web/src/protocol.ts`: types plus the mode values;
- new `web/src/hologram.ts` and `web/src/rps.ts`;
- `web/src/main.ts`: dispatch `hologram.hands` and `rps.state`; switch the visible view by `status.mode`;
- `web/index.html`: two mode buttons and two view containers;
- `web/src/styles.css`;
- `web/src/labels.ts`: Italian labels;
- `web/src/mock.ts`: synthetic streams for both modes, such as an animated hand and an RPS round loop.

The medical view's markup and code are unchanged except for the view switch.

**Acceptance:** AC-12.

**Verification:** `cd web && npm run typecheck && npm run build`, plus a mock screenshot.

### T6 — quantize, compile, evaluate the hand model

**Depends on:** T1 and the Kaggle training output.

**Actions:**
1. Extract the output zip to `build/models/hand_topdown/`.
2. Run `tools/vitis/run_vitis.sh python tools/vitis/dump_quant_outputs.py --weights build/models/hand_topdown/best.pt --calib-images train=<built val images dir> --test-images half=<built val images dir> --calib-count 120 --test-count 64 --height 256 --width 320 --threads 1 --out build/vitis/hand_topdown/quant`.
   - The calibration set is IR-look images from the val split, plus the 50 IR frames captured on 2026-11-11, passed through `ir_to_input` and saved as PNG.
   - Check whether `dump_quant_outputs.py` assumes 62 keypoints; if it does, add an `--n-kpt` option or read it from the weights, with a test.
3. Compile: `tools/vitis/run_vitis.sh bash tools/vitis/compile.sh build/vitis/hand_topdown/quant/<name>.xmodel build/vitis/hand_topdown/compiled hand21_topdown`.
4. Evaluate on the host with `ml/.venv` and `eval_quant.py`. If it only knows the half-bust schema, add `hand21` PCK@0.2 of the hand box.
5. Make an overlay contact sheet on the 50 IR frames.

**Acceptance:** AC-04.

**Verification:** compile log, eval table, and the sheet shown to the user.

### T7 — board integration test

**Depends on:** T3, T4, T5, T6, and an authorized deploy.

**Actions:**
1. `deploy/deploy.sh --model build/vitis/hagrid_lrelu512/compiled/pose_halfbust62.xmodel --hand-model build/vitis/hand_topdown/compiled/hand21_topdown.xmodel`. The medical model flag must stay, or the COCO model returns.
2. Mount the camera top-down.
3. Switch to `hologram`, then to `rps`, from the dashboard.

**Acceptance:** AC-13, AC-14.

## Integration verification

Run once on the integrated commit, through `python3 "$HOME/agent-harness/tools/run_check.py"`:

```
uv run --no-sync pytest -q && uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy src
cd ml && .venv/bin/python -m pytest -q
cd web && npm run typecheck && npm run build
```

Also run:
- the medical regression: the existing medical tests unchanged, plus a medical-mode start on the board after deploy (dashboard shows the skeleton);
- the cross-task check: the `.npz` from T2 loads in the T4 `RpsClassifier`, and the parity vector passes.

## Timeline

| when | what |
|---|---|
| Wed evening | T1 + T2 + T3 + T5 in parallel; T1 integrated first, then G2 → Kaggle launch |
| Wed night / Thu morning | Kaggle training (my estimate is about 2 h for 60 epochs at 320 on 2×T4; not measured); T4 |
| Thu late morning | T6 |
| Thu afternoon | T7; tune `conf_thr`, the intrinsics and the RPS timings |

## Open decisions / risks

**Gates:**
- **G1 — base (done 2026-11-11):** the camera-only medical work (tracker, video, single analyzer, Kalman K1, demo config) is committed and merged to `main`. The hands worktree must contain it: if it was branched from an older `main`, merge `origin/main` into it first.
- **G2 — push for Kaggle:** notebook 08 installs `cpsa-ml` from GitHub at `COMMIT_SHA`, so the T1 commit must be pushed before the Kaggle run. Ask the user before that push, unless the session prompt authorizes it.

**Risks:**
- **R1:** gray RGB photos differ from F200 IR images (skin brighter, more noise, projector speckle). Mitigations: the IR-look augmentation and CLAHE at training and runtime. AC-04's sheet on real IR frames shows the result before the board test. Fallback: fine-tune on a few hundred IR frames labelled by RTMW (the test showed it works on IR).
- **R2:** HaGRID and hand-keypoints are mostly frontal views, not from above. Mitigations: `degrees=180` and the flip augmentation. The back of the hand seen from above looks like a mirrored palm view in 2D.
- **R3:** the nominal intrinsics and depth scale are not calibrated. The 3D hand may be stretched or offset; the scale is acceptable for a hologram. Calibrate later with `tools/calibration/charuco_board.py` on IR `Y10`.
- **R4:** licences. The Ultralytics hand-keypoints set is CC BY-NC-SA 4.0 (non-commercial, share-alike); HaGRID is CC BY-SA 4.0. Both are acceptable for an academic open day, but the model inherits the share-alike terms. `stats.json` records the licences.
- **R5:** the DPU and the camera serve one mode at a time. A mode switch releases the camera and the DPU runner; the supervisor already serializes start and stop.
- **R6:** if `cpsa_ml.onnx_check.export_onnx`, `dump_quant_outputs.py` or `eval_quant.py` assume a square input or 62 keypoints, T1 and T6 grow by small, test-pinned extensions. This is not verified yet; the tasks name the check.
