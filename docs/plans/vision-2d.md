# 2D Vision Pipeline — Implementation Plan

**Status:** READY
**Goal:** When the IMU flags a stereotypy, the board analyzes the pre-roll and live RGB frames with one half-body pose model. A 2D tracker smooths the keypoints, and the vision features gate, re-grade and describe the episode. The result goes into actuation, the event log and the evidence clips.
**Spec:** `docs/specs/vision-2d-rasd.md` (FR-001..FR-025, NFR-001..NFR-006, AC-001..AC-018).
**Supersedes:** the depth and calibration path of `docs/plans/vision-toolchain.md`. Its 3D code (`cpsa.vision.calibration`, `tracking`, `fusion`, `pairing`) stays as level 3 groundwork. Do not delete or rewrite it.
**Scope / non-goals:**
- Out of scope: depth, calibration, articulated or IMU state fusion, object detection, GUI, multi-person tracking.
- Out of scope here: model training (Kaggle, run by the user) and DPU compilation (Vitis Docker).
- This plan delivers the runtime that consumes a pose model.
**Constraints:**
- Python 3.10, line length 100, mypy strict for `cpsa.domain`, `imu`, `ports`, `application` and `vision`.
- Layer rules in `tests/architecture/test_layers.py`:
  - `vision`: domain + numpy only.
  - `ports`: domain + vision.
  - `application`: domain, imu, ports, vision + numpy/yaml.
  - `adapters`: anything.
- Routine tests are deterministic and need no hardware.
- No commits.

## Approach

**Pipeline per episode:**

```
camera ──► VideoSource ──► pre-roll ring (raw frames) ──┐
                                                         ▼
IMU Detection ─► Dispatcher ─► VisionVerifier.verify ─► EpisodeAnalyzer
                                                          │  decode ─► PoseEstimator (DPU/ONNX) ─► PoseFrame
                                                          │  KeypointTracker2D (subject, framing, skeleton, KF) ─► TrackedFrame
                                                          │  features/contacts/handshape/oscillation
                                                          ├─► stage 1: VisionAssessment (verdict, severity, features)
                                                          └─► stage 2: EpisodeReport (+ EvidenceSink, event log)
```

**Key decisions:**

1. **One pose model.** Its keypoint layout is configured, so the same runtime runs:
   - `HALFBUST_62` (body 0-10, 9 face points, 2×21 hand points) — the target model;
   - `COCO17` as a stand-in, where hand shape is always `unknown`.
   - The full-body vs half-body framing comes from the person-box height divided by face width. No lower-body keypoints are needed.
2. **2D Kalman tracker.**
   - One constant-acceleration Kalman filter per keypoint. State per keypoint: x, y, vx, vy, ax, ay.
   - A chi-square gate rejects outliers.
   - A prediction horizon, then the keypoint is marked lost.
   - An RTS smoother runs over the episode history, for the stage-2 report.
   - Motion and measurement models sit behind protocols (NFR-001). A level 3 articulated EKF can replace them without touching the consumers.
   - A particle filter is not needed: the motion is near-linear at 30 fps.
3. **Level 2 skeleton.**
   - Reference bone lengths in face-widths: robust median over confident frames, kept per session.
   - The plausibility check removes a measurement before the KF update (FR-010).
   - Foreshortening ratio is current forearm length divided by the reference.
4. **Normalization.**
   - Face width comes from the tracker (`TrackedFrame.face_width_px`), with a configurable source and a stabilizing median.
   - Every feature is in fw, fw/s or fw/s².
5. **Synchronous core, threaded shell.**
   - `EpisodeAnalyzer` (pure, `cpsa.vision.episode`) consumes `PoseFrame`s in timestamp order. It answers `assess(...)` and `report(...)` from frame timestamps only.
   - `VisionVerifier` (`cpsa.application`) owns threads, the pre-roll, the deadline budget and the failure handling.
   - Replay drives the same analyzer, so replay and live give equal outputs (FR-025).
6. **Two-stage authority (FR-015..FR-017, FR-020).**
   - Each `verify(detection)` returns a stage-1 `VisionAssessment` built from the frames up to `detection.window_end_ts + fast_after_s`, within `fast_budget_s`. `fast_budget_s` must be less than `dispatch.verification_deadline_s`.
   - The dispatcher maps the result:
     - `CONFIRMED` with a severity → policy uses the severity;
     - `UNDECIDED` → the IMU tag;
     - `REJECTED` → no actuation.
   - Follow-up detections in an active episode get the episode's latest severity. Once the report says `active == False`, a framed subject gives `REJECTED`, which stops the follow-ups.
7. **Cheap capture.**
   - The pre-roll keeps raw YUYV bytes.
   - `VideoFrame.decode` converts only analyzed or saved frames, with `cv2.cvtColor` when OpenCV is available and the numpy path otherwise. This removes the 34 ms/frame conversion from the capture thread.

## Frozen interfaces (already in the worktree — Task 0, done by the coordinator)

- `src/cpsa/domain/model.py`:
  - enums `HandShape`, `StereotypyLabel`;
  - `VisionAssessment(verdict, severity: Tag|None, features: Mapping[str,float], episode_id: str|None)`;
  - `Decision` gains `severity: Tag|None = None` (the tag that drove the policy) and `assessment: VisionAssessment|None = None`.
- `src/cpsa/vision/types.py`:
  - `FramingState`, `KeypointStatus`, `KeypointLayout` (`index`, `has`, `hand(side)`);
  - `PersonPose`, `PoseFrame`, `VideoFrame` (lazy `decode`);
  - `TrackedFrame`, `HandFaceSample`, `Contact`, `EpisodeReport`.
- `src/cpsa/ports.py`:
  - `Verifier.verify -> VisionAssessment | Verdict` (a bare `Verdict` counts as an assessment with no severity);
  - `VideoSource`, `PoseEstimator`, `EvidenceSink`.
- Layer rules updated as listed above.

Changing any of these needs the coordinator: workers use `ask`.

**Function contracts that wave 2 consumes.** Wave 1 workers must provide them exactly. Each module may add private helpers and its own config dataclass with documented defaults.

| Module | Public API |
|---|---|
| `cpsa.vision.layouts` | `COCO17: KeypointLayout`, `HALFBUST_62: KeypointLayout`, `get_layout(name: str) -> KeypointLayout` |
| `cpsa.vision.pose_decode` | `decode_ultralytics(output: NDArray, n_kpt: int, conf_thr: float, iou_thr: float) -> list[PersonPose]` (in model-input pixels, shape `(1, 5+3K, N)`); `decode_raw_heads(outputs: Sequence[NDArray], strides: Sequence[int], n_kpt: int, conf_thr: float, iou_thr: float, reg_max: int = 16) -> list[PersonPose]` (YOLOv8 pose head before DFL/decoding, NHWC float per scale); `letterbox(rgb, size) -> (img, scale, pad)`; `unletterbox(persons, scale, pad) -> list[PersonPose]` |
| `cpsa.vision.tracker2d` | `Tracker2DConfig`; protocols `MotionModel`, `MeasurementModel`; `KeypointTracker2D(cfg, layout, motion=None, measurement=None)` with `.update(frame: PoseFrame) -> TrackedFrame`, `.reset() -> None`, `.smoothed() -> list[TrackedFrame]` (RTS over all frames since the last reset) |
| `cpsa.vision.skeleton` | `SkeletonConfig`; `SkeletonModel(cfg, layout)` with `.observe(xy, conf, face_width_px) -> NDArray[np.bool_]` (mask of accepted keypoints), `.reference_fw -> Mapping[str, float]`, `.foreshortening(xy) -> Mapping[Side, float or None]` |
| `cpsa.vision.subject` | `SubjectConfig` (rule: `largest`, `central`); `select_subject(persons, width, height, cfg) -> PersonPose or None`; `framing_state(person or None, layout, face_width_px, cfg) -> FramingState` |
| `cpsa.vision.handface` | `FeatureConfig`; `hand_face_samples(frame: TrackedFrame, layout, cfg) -> tuple[HandFaceSample, ...]` |
| `cpsa.vision.handshape` | `HandShapeConfig`; `classify_hand_shape(xy: (21,2), conf: (21,), cfg) -> HandShape` |
| `cpsa.vision.contacts2d` | `ContactConfig2D`; `detect_contacts(frames: Sequence[TrackedFrame], layout, cfg: AnalysisConfig) -> list[Contact]` |
| `cpsa.vision.oscillation` | `periodicity(t, signal, band_hz: tuple[float, float]) -> tuple[float or None, float]` (frequency, strength in [0,1]); `detect_rocking(frames, layout, cfg) -> bool`; `detect_flapping(frames, layout, cfg) -> bool` |
| `cpsa.vision.assessment` | `AnalysisConfig` (nests the feature, contact, hand-shape and oscillation configs, plus severity rules and activity window); `severity_of(contact, cfg) -> Tag`; `assess_fast(frames, t_from, t_to, layout, cfg) -> VisionAssessment` (`episode_id=None`); `build_report(episode_id, t_start, frames, layout, cfg, now, imu: IMUSignal or None = None) -> EpisodeReport` |

## Verification contract

| AC | Expected | Evidence |
|---|---|---|
| AC-001 | No inference while idle. The pre-roll is bounded. Analysis includes pre-detection frames. The episode ends after the idle time. | `tests/application/test_vision_verifier.py` with fake source, fake estimator and fake clock |
| AC-002 | Framing correct for synthetic half- and full-body persons. Half-body keeps upper-body confidences. | `tests/unit/test_subject.py`. The real-frame check is deferred to the trained model. |
| AC-003 | Two persons give one subject per rule. | `tests/unit/test_subject.py` |
| AC-004 | KF beats raw differencing (position and velocity RMSE) on a noisy synthetic trajectory. Short dropout gives PREDICTED; long dropout gives LOST. | `tests/unit/test_tracker2d.py` |
| AC-005 | An injected bone-breaking keypoint is excluded. Reference lengths are stable. Foreshortening is below 1 for a shortened forearm. | `tests/unit/test_skeleton.py` |
| AC-006 | Features are equal within tolerance when the same motion is scaled 0.5× and 2×. | `tests/unit/test_handface.py` |
| AC-007 | Synthetic contacts give the right side and region. Fist, point and open-palm hand fixtures are classified. | `tests/unit/test_contacts2d.py`, `test_handshape.py`. Labeled real clips deferred (Q-03). |
| AC-008 | Synthetic episodes give CONFIRMED, REJECTED and UNDECIDED. The verdict returns within `fast_budget_s`. | `tests/unit/test_assessment.py`, `tests/application/test_vision_verifier.py` |
| AC-009 | Fist to eye under NON_DANGEROUS actuates DANGEROUS. Slow cheek touch under DANGEROUS actuates NON_DANGEROUS. UNDECIDED uses the IMU tag. REJECTED gives no actuation. | `tests/unit/test_assessment.py` (severity) + `tests/application/test_dispatcher.py` (policy) |
| AC-010 | Count and frequency of repeated contacts are correct. Rocking and flapping are labeled. Labels alone never raise severity. | `tests/unit/test_oscillation.py`, `test_assessment.py` |
| AC-011 | Follow-ups use the latest severity. REJECTED after activity ends. | `tests/application/test_vision_verifier.py` |
| AC-012 | Keyframe and clip are written. Log records carry `episode_id`. Retention purge works. | `tests/contract/test_evidence.py`, `tests/adapters/test_event_log.py` (or the existing event-log test file) |
| AC-013 | Estimator raise or source failure gives UNDECIDED, a logged error and no crash. The stub path is unchanged. | `tests/application/test_vision_verifier.py`, `tests/application/test_dispatcher.py` |
| AC-014 | Replay output equals live output for the same frames. | `tests/application/test_vision_replay_equivalence.py` |
| AC-015 | A test-only motion model plugs into `KeypointTracker2D` with no consumer change. | `tests/unit/test_tracker2d.py` |
| AC-016 | DPU pose throughput on the KV260 is at least Q-02. | **Blocked:** needs the trained Hard-Swish model compiled by Vitis. `scripts/bench_pose.py` is delivered now. |
| AC-017 | Demo metrics meet Q-03. | **Blocked:** needs recorded, labeled sessions. |
| AC-018 | The injected clock offset is recovered in the report. | `tests/unit/test_assessment.py` |

**Gate command:** `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src`

## Tasks

### Task 0 — Frozen contracts (DONE, coordinator)
Domain types, `cpsa.vision.types`, ports and layer rules. The gate passes: 393 tests.

### Task A — Pose I/O (wave 1)
**Owns:**
- `src/cpsa/vision/layouts.py`, `src/cpsa/vision/pose_decode.py`
- `src/cpsa/adapters/pose/{__init__,onnx,dpu}.py`
- `scripts/bench_pose.py`
- `pyproject.toml` (extra `vision = ["onnxruntime", "opencv-python-headless"]`) and `uv.lock`
- tests `tests/unit/test_layouts.py`, `tests/unit/test_pose_decode.py`, `tests/contract/test_pose_onnx.py`

**Actions:**
1. **Layouts.**
   - Build `COCO17` and `HALFBUST_62` with names identical to `ml/src/cpsa_ml/schema.py`. Do not import `cpsa_ml`.
   - Face regions:
     - eyes, brows → `EYES`;
     - nose, jaws → `NOSE_CHEEK`;
     - mouth corners, lips, chin → `MOUTH`;
     - ears → `EAR`.
   - Hands use 21-point order.
   - Edges cover the body, face outline and finger chains.
   - Add a sync test: when `cpsa_ml` is importable, `HALFBUST_62.names` must equal `cpsa_ml.schema.HALFBUST_62.names`; otherwise the test skips with a reason.
2. **Decoders.** Pure numpy, tested on synthetic tensors:
   - the Ultralytics decoded layout;
   - the raw YOLOv8-pose head: DFL box, sigmoid class, keypoints `(kpt*2 + grid - 0.5... )` per the Ultralytics `Pose.kpts_decode`; document the exact formula;
   - NMS.
3. **`OnnxPoseEstimator(model_path, layout, input_size, conf_thr, iou_thr)`.**
   - Uses onnxruntime CPU.
   - Pipeline: letterbox, then normalize to [0,1] NCHW, then `decode_ultralytics`, then unletterbox.
4. **`DpuPoseEstimator(xmodel_path, layout, conf_thr, iou_thr)`.**
   - Uses the `xir` and `vart` Python bindings, imported lazily.
   - Finds the DPU subgraph.
   - Converts input to int8 with `fix_point` scaling, and output with `2**-fix_point`.
   - Uses `decode_raw_heads`.
   - Unit-test the quantize/dequantize helpers and the output-tensor ordering by stride, without vart.
5. **`scripts/bench_pose.py`.** Runs N inferences on random or real frames with either estimator and prints the mean and p95 latency and the fps.

**Acceptance:** decoder round-trip tests (encode a known person → decode → same box and keypoints within 1 px); layout tests; ONNX contract test skips cleanly without a model file (path set by env `CPSA_POSE_ONNX`).

### Task B — Tracking, skeleton, subject (wave 1)
**Owns:**
- `src/cpsa/vision/tracker2d.py`, `skeleton.py`, `subject.py`
- tests `tests/unit/test_tracker2d.py`, `test_skeleton.py`, `test_subject.py`

**Actions:**
1. **Subject selection and framing** per the contracts.
   - Full-body if `box_height / face_width_px >= full_body_ratio` (default documented, for example 9).
   - `face_not_visible` if fewer than 2 face anchors (eyes, nose, ears) have conf ≥ threshold.
   - `hands_not_visible` if neither wrist is confident.
   - Lower-body keypoints never enter any confidence.
2. **Face width.** Sources:
   - `ears` (ear-to-ear);
   - `eyes` (inter-ocular × `eye_to_face_ratio`, default 2.2);
   - `jaw` (jaw left-to-right, HALFBUST only);
   - `auto`: first available of jaw, ears, eyes.
   - Stabilize with a median over the last `face_width_window` frames.
3. **`SkeletonModel`.**
   - References in fw for `shoulder_width` and `{left,right}_{upper_arm,forearm}`.
   - Collected from frames where both joints have conf ≥ `ref_conf`, as a median of up to `ref_frames` samples. Frozen after `ref_min_frames` samples.
   - `observe` marks a joint as rejected when a bone through it is longer than `max_ratio` × reference (default 1.4). There is no lower bound, because foreshortening shortens bones.
   - When both joints of a bone are suspect, prefer to reject the lower-confidence one.
4. **Constant-acceleration KF.**
   - `MotionModel` protocol: `dim`, `transition(dt)`, `process_noise(dt)`.
   - `MeasurementModel` protocol: `observe_matrix()`, `noise(conf)`.
   - Defaults: CA model with white-jerk noise; position measurement with variance scaled by 1/conf.
   - Chi-square gate.
   - `max_predict_frames` horizon, then LOST.
   - Re-initialize a LOST keypoint on its next accepted measurement.
   - Variable `dt` from timestamps.
   - Store filtered and predicted states so `smoothed()` can run RTS. A model must expose its linear `F` and `Q` for that, or `smoothed()` returns the filtered frames.
5. **`KeypointTracker2D.update`.** Runs: select subject → framing → face width → skeleton mask → KF update or predict → `TrackedFrame` (with foreshortening).

**Acceptance:** AC-002, AC-003, AC-004, AC-005, AC-015. All arrays are float64, and the tests are seeded.

### Task C — Features, contacts, hand shape, oscillation, assessment (wave 1)
**Owns:**
- `src/cpsa/vision/handface.py`, `handshape.py`, `contacts2d.py`, `oscillation.py`, `assessment.py`
- the matching tests in `tests/unit/`

Build `TrackedFrame` fixtures directly. Do not depend on the Task B code. Reuse `cpsa.vision.fusion.estimate_clock_offset` and `IMUSignal` for FR-023.

**Actions:**
1. **`hand_face_samples`.**
   - Hand point: the centroid of the confident hand keypoints, else the wrist.
   - Face reference point: the centroid of the confident face anchors.
   - Distance: from the hand point to the nearest confident face anchor, in fw.
   - Closing speed and acceleration: from the tracked velocity and acceleration projected on the hand→face direction.
   - Approach angle: between the hand velocity and the hand→face direction.
   - TTC: distance / closing speed, when closing.
   - Occlusion fraction: the share of face anchors that are not MEASURED or have conf below threshold.
   - Region: from the nearest anchor. A point above the brow line by `forehead_margin_fw` maps to `FOREHEAD_TEMPLE`.
   - Return `None` fields when the face width is unknown.
2. **`classify_hand_shape`.**
   - Per finger: the extension ratio (tip-to-wrist distance / MCP-to-wrist distance), plus the thumb-tip to index-tip distance.
   - Fist: all fingers curled.
   - Point/pinch: index extended and others curled, or thumb and index tips close.
   - Open palm: at least 4 fingers extended.
   - Anything else, or low confidence: `UNKNOWN`.
3. **`detect_contacts`.**
   - Hysteresis on distance: `contact_enter_fw` and `contact_exit_fw`. Occlusion ≥ `occlusion_contact` with the hand within `occlusion_near_fw` also counts.
   - Minimum duration.
   - Peak closing speed in the `approach_window_s` before onset.
   - Deceleration: the drop in closing speed over the onset frames, divided by dt.
   - Hand shape: majority over the contact frames.
   - Head recoil: the largest displacement of the face-anchor centroid in `recoil_window_s` after onset, relative to the onset position, in fw, plus its peak speed.
4. **`oscillation`.**
   - `periodicity`: detrended autocorrelation, peak in the band, strength = normalized peak.
   - Rocking: periodicity of the shoulder-midpoint horizontal position (fw) in `rocking_band_hz` (default 0.5–2 Hz) with strength ≥ threshold and amplitude ≥ `rocking_min_amp_fw`.
   - Flapping: periodicity of wrist position relative to the shoulders in `flapping_band_hz` (default 2–6 Hz), with the wrist near or above shoulder height.
5. **`assessment`.**
   - `severity_of`: `DANGEROUS` if any of:
     - region is in `dangerous_regions` (default EYES);
     - hand shape is in `dangerous_shapes` (default FIST, POINT_PINCH) with peak closing speed ≥ `shape_speed_fwps`;
     - peak closing speed ≥ `strike_speed_fwps`;
     - decel ≥ `strike_decel_fwps2`;
     - head recoil ≥ `recoil_fw`.
     - Otherwise `NON_DANGEROUS`.
   - `assess_fast`:
     - CONFIRMED if a contact starts within `[t_from, t_to]`, or there is an approach with distance ≤ `approach_fw` and closing speed ≥ `approach_speed_fwps`. Severity = max over contacts.
     - REJECTED only if ≥ `min_framed_fraction` of the window frames are HALF_BODY or FULL_BODY with the face and both visible wrists, and no hand came within `contact_exit_fw`.
     - Else UNDECIDED.
     - `features` has `min_distance_fw`, `peak_closing_speed_fwps`, `decel_fwps2`, `occlusion_max`, `head_recoil_fw`, `region_code`, `hand_shape_code`, `framed_fraction` (codes documented).
   - `build_report`:
     - contacts; count; repetition frequency from the periodicity of the distance signal; inter-contact CV;
     - speed trend: linear slope of the contact peak speeds;
     - peak severity;
     - labels: rocking and flapping never change severity;
     - `active`: activity seen within `activity_window_s` of `now`;
     - clock offset and consistency via `estimate_clock_offset` (camera wrist acceleration magnitude in fw/s² vs the IMU), when `imu` is given.

**Acceptance:** AC-006, AC-007, AC-008 (logic), AC-009 (severity), AC-010, AC-018.

### Task D — Dispatcher, policy, event log (wave 1)
**Owns:**
- `src/cpsa/application/dispatcher.py`, `src/cpsa/domain/policy.py`, `src/cpsa/domain/gating.py`
- `src/cpsa/adapters/verifier_stub.py`, `src/cpsa/adapters/event_log.py`
- their tests

**Actions:**
1. Normalize the verifier output: a bare `Verdict` becomes `VisionAssessment(verdict)`. Any other type fails open, as today.
2. Effective severity:
   - CONFIRMED with a severity in {NON_DANGEROUS, DANGEROUS} → that severity;
   - otherwise → the detection tag.
   - `policy.actuations_for(effective)`.
   - `Decision.severity` = the effective severity, set only when actuated; `Decision.assessment` = the assessment (`None` on timeout or error).
3. `StubVerifier` returns `VisionAssessment(Verdict.UNDECIDED)`.
4. The event-log JSON adds `severity` and `assessment` (verdict, severity, features, episode_id), with `null` when absent. Existing keys are unchanged.

**Acceptance:** AC-009 (policy), AC-013 (stub regression, timeout and error fail-open unchanged), AC-012 (log fields).

### Task E — Episode runtime, capture, evidence, config, wiring, replay (wave 2; after A–D)
**Owns:**
- `src/cpsa/vision/episode.py`, `src/cpsa/application/vision_verifier.py`
- `src/cpsa/adapters/camera/uvc.py`, `src/cpsa/adapters/evidence.py`
- `src/cpsa/application/config.py` (`VisionConfig`) and `config/board.yaml` (`vision:` section), `tests/application/test_config.py`
- `src/cpsa/board/__main__.py` (wiring), `src/cpsa/board/vision_replay.py`
- `pyproject.toml` script `cpsa-replay-vision`
- README section
- tests listed in the verification contract for AC-001, AC-011..AC-014

**Actions:**
1. **`EpisodeAnalyzer(layout, tracker_cfg, analysis_cfg)`.**
   - `start(episode_id, t)`, `push(pose_frame)`, `assess(t_from, t_to) -> VisionAssessment` (fills `episode_id`), `report(now, imu=None) -> EpisodeReport`, `frames`.
   - `active(now)` and `end()`. Inputs in timestamp order only.
2. **`VisionVerifier(source: VideoSource, estimator: PoseEstimator, evidence: EvidenceSink | None, cfg, clock, layout)`.**
   - The capture callback appends to a pre-roll deque bounded by `preroll_s`.
   - `verify(detection)`: if no episode is active, start one and hand the pre-roll snapshot (frames with `t >= window_end_ts - lookback_s`) to the analysis thread. The analysis thread decodes, estimates and pushes, and keeps up with live frames. Under backlog it skips frames with `analysis_stride`, so it never runs slower than real time.
   - Wait until the analyzed time is at least `window_end_ts + fast_after_s` or `fast_budget_s` has elapsed. Then return `analyzer.assess(...)`.
   - Any estimator or source error → log once per episode, return UNDECIDED.
   - The episode ends when `analyzer.active(now)` is false and no detection has arrived for `idle_timeout_s`. At the end: write the report (JSONL via the logger `cpsa.vision`), evidence (`clip_s` around the peak-severity contact, keyframe), and purge.
   - No pose inference outside episodes.
3. **`UvcVideoSource(device, width, height, fps, exposure)`.**
   - RGB-only, with linuxpy and own `select` + `raw_read` as in `f200.py`.
   - Frames keep raw YUYV bytes; `decode` uses `cv2.cvtColor(..., COLOR_YUV2RGB_YUYV)` when available, else `f200.yuyv_to_rgb`.
   - Timestamps are the V4L2 buffer `CLOCK_MONOTONIC`.
4. **`FileEvidenceSink(base_path, retention_days)`.**
   - Writes under `<base>/<YYYY-MM-DD>/<episode_id>/`: `clip.mp4` (cv2 `mp4v`, MJPG AVI fallback) with skeleton overlay from `layout.edges`, plus `keyframe.jpg`.
   - `purge` removes day folders older than the retention.
5. **Config.**
   - `VisionConfig(enabled, layout, model (onnx or xmodel path and kind), camera (device, width, height, fps, exposure), preroll_s, lookback_s, fast_after_s, fast_budget_s, idle_timeout_s, analysis_stride, evidence (enabled, base_path, clip_s, retention_days), tracker, analysis)`.
   - Every field is validated.
   - `fast_budget_s < dispatch.verification_deadline_s` is enforced.
   - In `board.yaml`, `vision.enabled: false` by default, so the current board behavior is unchanged until a model exists.
6. **Board wiring.** When `vision.enabled`, build the source, the estimator (`dpu` or `onnx`), the evidence sink and `VisionVerifier`. Otherwise `StubVerifier`.
7. **`cpsa-replay-vision <session> [--onnx|--xmodel] [--detections file]`.**
   - Runs the same `EpisodeAnalyzer` over recorded RGB frames (session format v1).
   - Detections come from a JSONL file or from the session's trial annotations.
   - Prints the JSONL assessments and reports.
   - The AC-014 test runs the same fake frames through `VisionVerifier` (fake clock) and through the replay function, and compares the outputs.

**Acceptance:** AC-001, AC-011, AC-012, AC-013, AC-014; the gate is green.

## Integration verification
- Full gate on the integrated tree.
- Architecture test green with the new modules.
- A Codex review of the plan (before wave 1) and of the integrated change.
- Board smoke test when the board is reachable:
  - `cpsa-replay-vision` with an ONNX COCO17 stand-in on a short recorded clip;
  - `scripts/bench_pose.py --onnx` CPU timing.
  - Both are informational. The DPU result is AC-016, which is blocked.

## Open decisions / risks
- Q-01 to Q-05 are carried as config defaults: `fast_budget_s` 0.8 within the 1.0 s deadline, and `subject.rule: largest`.
- The thresholds are provisional until there are recorded sessions (AC-017).
- AC-016 is blocked on Kaggle Hard-Swish training plus Vitis compile (`docs/plans/vision-toolchain.md` T8/T9).
- **Raw-head decode correctness:** the decoder is written now, but the compiled graph's output cut is confirmed only at T9. The adapter is isolated so a fix stays local.
- **Pre-roll catch-up:** analysis must catch up with backlog in real time. The stride covers this; actual timing is measured with AC-016.

## Amendments after plan review (Codex, task_0c0c431c8e44): these take precedence over the sections above

**P-01 — acceleration uncertainty (FR-007).** `TrackedFrame` now has `acc_std` `(K, 2)`. Task B fills it. AC-004 also checks that `acc_std` is finite and positive.

**P-02 — severity without a contact (FR-016).** Task C adds `severity_of_approach(samples: Sequence[HandFaceSample], hand_shape: HandShape, cfg) -> Tag`. It applies the same rules as `severity_of` to the features available before contact: nearest region, hand shape, peak closing speed. Rules on missing features (decel, recoil) count as not met.
- `assess_fast` always sets a severity when `CONFIRMED`:
  - with contacts: the maximum over contacts;
  - approach-only: `severity_of_approach`.
- Tests cover an approach-only `CONFIRMED` under both IMU grades, through Task D's policy test (a fake verifier returning each severity).

**P-03 — coverage and freshness (FR-015).** `assess(t_from, t_to)` returns `UNDECIDED` unless both hold:
- (a) analyzed frames cover `[t_from, t_to]`: the first frame is at or before `t_from + max_gap_s`, the last at or after `t_to - max_gap_s`, and no gap between consecutive analyzed frames exceeds `max_gap_s` (default 0.2 s);
- (b) the source is healthy: no `on_error` received, and the last captured frame is younger than `stale_after_s` (default 0.5 s).

`REJECTED` additionally needs (a) over the whole window. `features` reports `coverage_end_t` and `coverage_ok`.
- Tests: a backlogged estimator (coverage incomplete at the budget) gives `UNDECIDED`, not `REJECTED`. An unplugged source (`on_error`, or no frames) gives `UNDECIDED`.

**P-04 — episode follow-ups and detections inside the cooldown (FR-020, AC-011).**
- New port `DetectionObserver.observe(detection)` in `src/cpsa/ports.py`, already added.
- Task D: `Dispatcher.handle` calls `observe(detection)` for **every** stereotypy detection, before `should_evaluate`, when the verifier also implements `DetectionObserver` (`isinstance` check). It is guarded by try/except and logs on error. The verify, cooldown and actuation logic is otherwise unchanged. `StubVerifier` does not implement it, so stub behavior is identical.
- Task E: `observe` starts or extends the episode, resetting the idle timer.
- `EpisodeAnalyzer.assess(t_from, t_to, now)` with episode state:
  1. If the window has a contact or approach → `CONFIRMED`, severity from the window (P-02).
  2. Else if the episode report is `active` at `now` (activity within `activity_window_s`) → `CONFIRMED` with the **latest** contact's severity in the episode; `features.from_episode = 1`.
  3. Else, with coverage and framing met → `REJECTED`, which stops follow-ups.
  4. Else → `UNDECIDED`.
- AC-011 is tested at dispatcher level in Task E, with the real `Dispatcher`, a fake clock, cooldown, and `VisionVerifier` over fake frames. It covers: detection during cooldown extends the episode; a follow-up after cooldown gets the latest severity (upgrade and downgrade cases); a follow-up after inactivity gets `REJECTED` and no actuation.

**P-05 — threading contract for `VisionVerifier` (Task E).**
- **Capture thread** (source callback): appends `VideoFrame` to a lock-protected pre-roll deque bounded by `preroll_s`. If an episode is active, it also puts the frame on a bounded analysis queue (`queue_max` frames; drop-oldest is counted and logged).
- **Analysis thread**, the single owner of `EpisodeAnalyzer`: consumes the queue, applies frame selection (P-06), decodes, estimates and pushes. After each push it publishes an immutable snapshot (`analyzed_until_t`, the assessment inputs) under a `threading.Condition`.
- **`verify`**: never touches the analyzer directly. It sends a request (`t_from`, `t_to`, deadline) to the analysis thread and waits on the condition until the result is posted or the absolute deadline `start + fast_budget_s` passes. On deadline it returns `UNDECIDED` with `features.timeout = 1`.
- **`observe`**: posts an "extend/start episode" message to the analysis thread and does not wait.
- **Errors**: `on_error` sets a health flag read by the analysis thread. An estimator exception gives `UNDECIDED` for pending requests, logged once per episode, and the episode continues.
- **`start()` and `stop()`**: `stop` sets a stop event, closes the source and joins the threads with a timeout. It never waits on a hung `estimate` call: the analysis thread is a daemon thread and is abandoned after the join timeout, with a warning logged.
- Add `start()` and `stop()` to `VisionVerifier`. Task E calls them from the board runtime lifecycle (`src/cpsa/application/runtime.py` is now owned by Task E).

**P-06 — deterministic frame selection (FR-025).**
- Live and replay select analyzed frames with the same pure rule `select_frames(frames, analysis_fps)` (in `cpsa.vision.episode`): keep the first frame at or after each grid point `t0 + k / analysis_fps`, where `t0` is the episode start. The selection depends on timestamps only, never on backlog.
- If live cannot keep up, coverage stays incomplete and the verdict is `UNDECIDED` (P-03). Nothing is skipped silently.
- Every live decision logs `t_from`, `t_to` and `coverage_end_t`. Replay recomputes the same assessment by calling `assess` with `t_to` truncated to the recorded `coverage_end_t`, when given in the detections file.
- AC-014 test: run the same frames through live with estimator delays of 0, half the frame time, and 2× the frame time. Then run replay with the logged cutoffs. The verdicts, features and reports must be identical.

**P-07 — IMU feed for FR-023 (Task E).**
- `VisionVerifier.on_imu_sample(side, kind, values, ts)` keeps a bounded per-side deque (`imu_buffer_s`) of ACC magnitudes in g. Raw values are in mg, so divide by 1000, as `IMUSignal` documents.
- `build_runtime` tees the `SensorSource` samples to it when vision is enabled (Task E owns `runtime.py` and `board/__main__.py`).
- The report computes the offset and consistency per side against that side's tracked wrist, and keeps the side with the higher correlation.
- Replay builds `IMUSignal` per side from the session `imu.csv`.
- AC-018 is tested end to end: replay of a synthetic session with a known injected IMU time shift reports it within one frame time.

**P-08 — recorded-session acceptance kept as blocked gates.**
- AC-007 (labeled clips) and AC-008 (recorded sessions) keep their original wording.
- Task E adds `tests/target/test_vision_sessions.py`, marked `target`. It reads the labeled sessions from env `CPSA_LABELED_SESSIONS` (session dirs plus a `labels.jsonl` with contact side, region, hand shape and expected verdict), runs replay, and asserts per label. It skips with a reason when the env is unset.
- Status stays BLOCKED until the sessions are recorded and labeled (owner: user and coordinator). The synthetic tests remain.

**P-09 (accepted) — session-level skeleton.** `KeypointTracker2D.reset()` clears the Kalman states and the RTS history but keeps the `SkeletonModel` references (session-long, FR-009). `reset_session()` clears both. Task B tests two episodes in one session.

**P-10 (accepted) — independent decoder oracle.**
- Task A pins the decode convention to Ultralytics 8.4.203, the version in `ml/pyproject.toml`.
- Generate golden fixtures using `ml/.venv`: random raw head outputs go through the upstream `Pose` head decode and are saved as `tests/fixtures/pose/yolov8_pose_raw_golden.npz`. Commit the generator under `scripts/make_pose_golden.py`.
- `decode_raw_heads` must match the fixtures. The test loads only the `.npz`, so the root env needs no Ultralytics.
- Add fix_point tests: nonzero `fix_point`, saturation at -128/127, and output order by stride.

## Amendment 2 (re-review task_63c10886d7df): replaces the P-04 rules 2–4 and the P-06 replay rule above

**P-04 (revised) — episode grade state.**
- `EpisodeAnalyzer` keeps `latest_grade`: the time and severity of the latest hand-to-face event in the episode. An event is a contact (`severity_of`) or a qualifying approach (`severity_of_approach`). Rocking and flapping never set `latest_grade` (FR-019).
- `assess(t_from, t_to, now)`, in order:
  1. Coverage or health not met (P-03) → `UNDECIDED`.
  2. The window has a contact or approach → `CONFIRMED`, severity from the window. This also updates `latest_grade`.
  3. `latest_grade` exists and `now - latest_grade.t <= activity_window_s` → `CONFIRMED` with `latest_grade.severity`; `features.from_episode = 1`.
  4. Activity in the episode is rocking or flapping only (labels active within `activity_window_s`, no recent `latest_grade`) → `UNDECIDED` (`features.activity_only = 1`). The IMU tag then drives actuation, and vision neither raises nor rejects.
  5. Framing condition of FR-015 met over the whole covered window → `REJECTED`.
  6. Else → `UNDECIDED`.
- Tests (Task E, analyzer level and dispatcher-level AC-011) cover:
  - an approach-only episode, where the follow-up gets the approach severity;
  - a rocking-only episode, where the follow-up is `UNDECIDED` and the IMU tag is used;
  - contact then inactivity, where the result is `REJECTED`.

**P-06 (revised) — replay reproduces the live analysis exactly.**
- **Trace.** The analysis thread records, per episode, the ordered list of frames actually pushed into the analyzer (`seq`, `t`), after frame selection and any drop-oldest losses. For every assessment it also records:
  - the requested `t_from` and `t_to`, and `now`;
  - the index in that pushed list at which the result was computed;
  - `timeout` (deadline passed before a result);
  - `source_healthy`.
- **Where the trace goes.** It is written at episode end as `<evidence base>/<day>/<episode_id>/trace.jsonl`, or under the log dir when evidence is disabled, and referenced from the episode-report log record.
- **Replay modes** (`cpsa-replay-vision`):
  - **audit** (`--trace trace.jsonl`): push exactly the traced frames from the recorded session, by `seq`, in order. Re-evaluate each assessment with the same requested bounds and `now`, at the same push index, with the traced `timeout` and `source_healthy` values. A traced `timeout` gives `UNDECIDED` with the same features as live. Reports are built from the same pushed history, so they match exactly.
  - **ideal** (default): apply `select_frames` to all session frames with no timing effects, for development and threshold tuning. No equivalence claim is made for this mode.
- The earlier "truncate `t_to` to `coverage_end_t`" rule is withdrawn.
- **AC-014 test.** Compare live and audit-mode replay over the same frames for:
  - zero estimator delay;
  - a slow estimator that triggers drop-oldest;
  - a stalled estimator that hits the deadline (`timeout`);
  - a source `on_error`.
  
  Verdicts, severities, features and the final `EpisodeReport` must be identical.
