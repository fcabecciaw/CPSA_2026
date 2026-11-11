# Medical demo slice for Thursday morning — Implementation Plan

**Status:** READY
**Goal:** By Thursday 2026-11-12 morning the KV260 boots into the medical demo with no manual steps:
- the BlueCoin IMUs trigger the vision verifier;
- a halfbust62 body model, trained on Kaggle and quantized to INT8, runs on the DPU;
- the actuators respond;
- a browser on the booth laptop shows a live medical dashboard over the shared WebSocket protocol.

**Spec:**
- `docs/specs/system-rasd.md`: FR-101, FR-201..203, FR-401..403, FR-501..504, FR-506, NFR-01, NFR-02, NFR-04, NFR-05, NFR-07.
- `docs/specs/interface-rasd.md`: IF-101..104, IF-201..205, IF-301..304, IF-451..454, IF-601, IF-701..704.
- `docs/specs/vision-2d-rasd.md`: FR-001..025, NFR-003, NFR-004.
- User decisions of 2026-11-11 (see Decisions below).

**Scope:**
- An optimized Kaggle training path for the body model, on a new data mix.
- PTQ and compile with the real weights.
- A minimal shared server layer: the supervisor, an HTTP and WebSocket server, and the protocol schemas.
- The medical mode as an adapter over that layer.
- The medical dashboard frontend.
- Board deployment as a boot service.

**Non-goals for this slice:**
- The model lab: gestures, the scene, Three.js, `lab.*` topics.
- The stage-2 hand model (notebook 07) and its runtime integration. D8 supersedes them; notebook 07 is no longer on any planned path.
- The shared perception service (IF-702, IF-801..805); see D4.
- The camera preview (IF-512, IF-305).
- QAT and Brevitas.
- Full-frame data (notebook 02).
- COCO pseudo-labels (`cpsa-halfbust-pl`).
- Notebook 03.
- Revising the specs for Option C.
- Threshold tuning on booth recordings. It starts after the Thursday-morning test.

**Constraints:**
- Board: KV260, Ubuntu 22.04, Python 3.10, VART 3.5, DPU fingerprint `0x101000016010407`, `kv260-benchmark-b4096` app.
- No internet at runtime (NFR-07). The frontend is bundled locally.
- Kaggle session: 2× T4, 4 vCPU, 12 h cap. Internet is on for installs.
- `/kaggle/input` is a network mount that costs about 160 ms per small-file operation per thread. This was measured: curation managed 100 crops/s with 16 threads.
- Host PC: 16 cores. Its GPU (RTX PRO 500 Blackwell, 6 GB) is not usable by the `ml` venv (CPU torch) or by the Vitis AI 3.5 Docker image (CPU-only).
- `cpsa-board` stays as it is, as the headless fallback.
- Layer rules in `tests/architecture/test_layers.py` must hold, extended for the new packages.

## Decisions

- **D1. Training data (approved by the user, 2026-11-11).** Two sources are mixed:
  - **HaGRID 30k-384p, labeled by the RTMW-x whole-body teacher** (`cpsa_ml.pseudolabel`) on the Kaggle GPU. These are people facing the camera with large hands, many of them near the face, at the deploy resolution.
  - **COCO half-bust crops at least 256 px tall**, with human labels and the hand-over-face occlusion.

  Measured on the current `cpsa-halfbust` (a 3,000-crop sample plus a full header pass): the median crop height is 118 px. Only 16,828 of 103,996 train crops (16%) are at least 256 px tall, and 59% of those show a hand with at least 10 visible keypoints.

  Why the mix, and why not HaGRID alone (approved by the user, 2026-11-11):
  - HaGRID's own labels are hand boxes, a gesture class, and 21 MediaPipe hand landmarks per hand. It has no face, shoulder or elbow keypoints and no person box.
  - The medical pipeline needs those keypoints: hand-to-face distance and the face region touched (FR-013, FR-014), rocking from the shoulders, and the framing state (FR-005). So a model trained on raw HaGRID labels cannot serve it.
  - The teacher adds the body and face keypoints. HaGRID contributes the demo framing (people facing the camera at 0.5–4 m) and large, clear hands.
  - HaGRID's MediaPipe hand landmarks are not merged in. They are auto-annotated too, so they are no more reliable than the teacher's hands on large hands, and merging adds work before Thursday.
  - The COCO crops of at least 256 px stay in the mix for two reasons. Their labels are human-made. They also carry 22,898 synthetic hand-over-face occlusions (the medical contact case), which HaGRID rarely shows: classes such as `call`, `mute` and `hand_heart` bring the hand near the face, but rarely onto it.
  - For the long main run after Thursday, the natural upgrade is the 120k HaGRID sample.
- **D2. Quantization stays on Vitis AI 3.5 `vai_q_pytorch` PTQ, on the CPU.**
  - Brevitas is not used. Its exported graphs (QONNX or QCDQ ONNX) have no import path into `vai_c_xir` for DPUCZDX8G, so it would add a step, not replace one. This is an assumption from the Vitis AI 3.5 toolchain, and nothing below depends on it.
  - A GPU is not needed. PTQ with real-size inputs was measured at 120 calibration frames in 133 s with `--threads 1` (`docs/notes/vitis-compile.md`).
  - QAT or `fast_finetune` is used only if AC-04 fails.
- **D3. GPU use:** Kaggle T4s for the teacher (ONNX Runtime CUDA) and for training (DDP). The fallback when ONNX Runtime CUDA fails on Kaggle is the local CPU with `--teacher rtmw-l`.
- **D4. Perception service deferred.** The medical mode keeps today's arrangement: `VisionVerifier` owns the camera and the DPU (`src/cpsa/board/__main__.py:92-131`).
  - This deviates from IF-702 and IF-801..805. The lab plan must add the shared service later, with no change to the protocol.
  - `status.perception.fps` and `dpu_ms` MAY be `null` in this slice.
- **D5. Server library: `aiohttp`.** It gives one port for HTTP and WebSocket (IF-101) and runs on asyncio. Wheels for linux-aarch64 cp310 exist. A `server` extra is added to `pyproject.toml`.
- **D6. New composition root `cpsa-supervisor`.** It starts the server, the supervisor and the default mode from config. Only the modes `idle` and `medical` exist in this slice. `cpsa-board` is unchanged.
- **D7. Frontend:** Vite + TypeScript, no UI framework, no Three.js in this slice (stack per `PRODUCT.md`). It is built on the PC into `web/dist` and served by the board.
- **D8. Lab gesture route, after Thursday (approved by the user, 2026-11-11).** The lab reuses the medical body model and the gesture layer of `ai-forever/dynamic_gestures` (https://github.com/ai-forever/dynamic_gestures). See "Follow-up: lab gesture route".
  - It replaces the stage-2 hand-keypoint model (notebook 07, Option C) and most of the planned pinch and fingertip rules.
  - It also replaces `dynamic_gestures`' own hand detector. The body model's wrist and hand keypoints give the hand box, which leaves the crop classifier as the only extra model on the DPU.
  - Nothing in this slice depends on D8. The medical tasks below stay as written, and `src/cpsa/vision` and the protocol stay compatible with it.

## Approach

### Training track (critical path; Kaggle runs overnight)
1. **Staging becomes cheap on the slow mount** (`ml/src/cpsa_ml/datasets/staging.py`):
   - list files with `os.walk` (no per-file `stat`);
   - optional `include` manifest, so only listed relative paths are copied;
   - tar sources are extracted to local disk;
   - 64 copy threads by default;
   - progress during listing.
2. **A manifest of the COCO crops at least 256 px tall** is generated once from the local copy of the dataset (`~/Downloads/_output_/halfbust62`) by a new script. It ships inside `cpsa_ml` as package data, gzipped. Kaggle then copies about 42k files (16,828 train images and their labels, plus all val) instead of 216k, with no header reads on the mount.
3. **New notebook 06** labels HaGRID 30k with the teacher on the GPU and writes one `hagrid62.tar` to `/kaggle/working`. Notebook 04 attaches that notebook output directly, so the user doesn't have to publish a dataset, and copies it as one large file.
4. **Notebook 04** uses:
   - the manifest and `MIN_HEIGHT = 256`;
   - `EXTRA_DATA` = the extracted `hagrid62`, which gives the `val_extra1` metric;
   - `VARIANTS = [DEFAULT_VARIANTS[2]]` (`lrelu_512`, deploy 384×512) and `EPOCHS = 30`;
   - a per-epoch wall-time printout.

### Model track (Thursday early morning)
1. `best.pt` goes through PTQ, then compile, then INT8-vs-FP32 eval, with existing tools at 384×512. The calibration images mix HaGRID val frames (demo-like) and COCO val.
2. The `.xmodel` is copied to the board.

### App track (parallel with training, Wednesday afternoon and evening)
```
browser (web/dist)  ⇄  aiohttp /ws, /, /assets, /health   [adapters/web]
                          ▲ Publisher port (thread-safe)
supervisor (mode state machine, status 1 Hz)               [supervisor]
   └─ MedicalMode  ──► discover BlueCoins ► load classifier ► build_runtime(gui=WebGui)
                         WebGui = Gui + VisionObserver ► cpsa.protocol builders ► Publisher
```

New packages and files:
- `src/cpsa/protocol/`: a pure layer. It holds the envelope and the message builders from domain and vision types.
- `docs/protocol/v1/*.schema.json` and `docs/protocol/v1/examples/*.json`.
- `src/cpsa/supervisor/`: a pure layer. It holds the mode lifecycle and the status assembly.
- `src/cpsa/adapters/web/`: the aiohttp server.
- `src/cpsa/adapters/gui_web.py`.
- `src/cpsa/board/medical_mode.py` and `src/cpsa/board/supervisor_main.py`.
- `web/`.
- `deploy/` (systemd units, deploy script) and `config/board.demo.yaml`.

### Frozen interfaces (written by the coordinator before tasks 4–7 start; changes need coordinator approval)

**F1 — `Publisher` port** (`src/cpsa/ports.py`):
```python
class Publisher(Protocol):
    def state(
        self, type: str, data: Mapping[str, Any]
    ) -> None: ...  # latest-wins per type, thread-safe, never blocks
    def event(
        self, type: str, data: Mapping[str, Any]
    ) -> None: ...  # FIFO, thread-safe, never blocks
```
The envelope (`v`, `seq`, `t`, `wall`) is added by the server per connection (IF-102). Callers pass only `type` and `data`.

**F2 — `VisionObserver` port** (`src/cpsa/ports.py`), plus a `VisionVerifier.__init__(..., observer: VisionObserver | None = None)` keyword argument with default `None`, which keeps today's behavior:
```python
class VisionObserver(Protocol):
    def on_episode(self, episode_id: str, t_start: float, active: bool) -> None: ...
    def on_tracked(self, tracked: TrackedFrame) -> None: ...
    def on_report(self, report: EpisodeReport) -> None: ...
```
- It is called from the verifier's worker thread.
- `on_episode`: when an episode starts (`active=True`) and in `_finish_episode` (`active=False`).
- `on_tracked`: once per analyzed frame (`_analyze_frame`).
- `on_report`: in `_finish_episode`, after the report is written.
- An exception from the observer is logged and swallowed. It never changes a verdict.

**F3 — `Runtime.imu_status(now: float) -> Literal["connected", "lost"]`**
- `connected` when both sides delivered a sample within the last 1.0 s; otherwise `lost`.
- It is fed by `Runtime._on_sample`.
- `searching` is reported by the medical mode while BlueCoin discovery runs.

**F4 — Supervisor mode contract** (`src/cpsa/supervisor/`):
```python
class ModeStartError(Exception):
    code: str  # one of IF-304 codes
    message: str


@dataclass(frozen=True)
class ModeContext:
    publisher: Publisher
    config: Config  # cpsa.application.config.Config
    stop_event: threading.Event


class Mode(Protocol):
    name: str  # "medical"

    def start(self, ctx: ModeContext) -> None: ...  # blocking until running; raises ModeStartError
    def stop(self, timeout_s: float) -> bool: ...  # True when all devices are released
    def health(self) -> Mapping[str, Any]: ...  # keys: imu, camera, model, fps, dpu_ms, errors
```
The supervisor runs `start` and `stop` on a worker thread, never on the asyncio loop.

**F5 — Protocol v1 schemas** (`docs/protocol/v1/`). One JSON Schema for the envelope, and one per type:
- `hello`, `status`, `mode.progress`, `error`, `pong`;
- `medical.state`, `medical.decision`, `medical.episode`, `medical.hands`;
- the commands `mode.set` and `ping`.

Fields are exactly those listed in IF-102, IF-201, IF-301..304 and IF-451..454. `medical.hands.data` = `{layout, w, h, keypoints: [[x, y, conf], ...], subject_box}`, in frame pixel coordinates. Each type has at least one example JSON file under `examples/`.

**F6 — Config additions** (`src/cpsa/application/config.py` and `config/board.yaml`):
```yaml
server:
  host: 0.0.0.0
  port: 8080
  static_dir: web/dist
  idle_timeout_s: 30
supervisor:
  default_mode: medical   # idle | medical
```

## Verification contract

- **AC-01 [D1, training speed]:** On Kaggle, notebook 04 lists and copies the manifest subset and the `hagrid62.tar` with visible progress, and epoch 1 starts within 20 min of Run All.
  - Evidence: the user's notebook log with staging times and the epoch-1 wall time.
  - Locally: `ml` unit tests of the staging listing, manifest filtering and tar extraction.
- **AC-02 [D1, teacher]:** Notebook 06 confirms that `CUDAExecutionProvider` is active before labeling, labels at least 25,000 HaGRID frames with a person, writes `hagrid62.tar` with `data.yaml`, `images/{train,val}` and `labels/{train,val}` in the halfbust62 schema, and saves 16 rendered samples.
  - Evidence: the notebook output, `stats.json`, the rendered PNGs, and a local CPU smoke test of the same code path on 50 frames.
- **AC-03 [FR-004, FR-005]:** The `lrelu_512` run finishes 30 epochs. `metrics_fp32.json` holds the half-body val, `h160` and `val_extra1` (HaGRID) sections, and `deploy.onnx` is 384×512.
  - Evidence: the run outputs.
- **AC-04 [NFR-003]:** The compiled xmodel has fingerprint `0x101000016010407`. INT8 wrist, hand and face PCK@0.2fw on HaGRID val stay within 2 points of FP32 (the README "Variant study" rule).
  - Evidence: `inspect_xmodel.py` and `eval_quant.py` outputs.
- **AC-05 [NFR-003, AC-016 of vision RASD]:** On the board, `DpuPoseEstimator.estimate` at 384×512 with the trained weights reaches at least 30 fps (mean over 200 frames). At most, `analysis_fps: 10` is kept.
  - Evidence: a timing log from the board.
- **AC-06 [IF-101..104, IF-201..205, FR-503, FR-504, FR-506]:**
  - Every message the server emits validates against its schema.
  - A connect yields `hello`, then the full snapshot.
  - `ping` returns `pong` with the same `data.id`.
  - `seq` increases strictly per connection.
  - A client that never reads does not delay the others: states are coalesced, and the client is disconnected when its event queue overflows.
  - Four concurrent clients are served.
  - Evidence: `tests/contract/test_protocol_schemas.py` and `tests/adapters/test_web_server.py` (aiohttp test client).
- **AC-07 [IF-301..304, IF-701..703, FR-201..203]:**
  - `mode.set medical` gives `mode.progress` in the order stopping, releasing, starting, ready.
  - A second `mode.set` during a switch gives `error{code: busy}`.
  - A failed start (no BlueCoins) leaves `mode = idle`, `mode_state = failed` and `errors = ["imu_not_found"]`.
  - `status` is sent at 1 Hz and on change.
  - Evidence: `tests/supervisor/test_supervisor.py` with fake modes.
- **AC-08 [FR-401, FR-402, FR-403, IF-451..454, IF-601, IF-704]:**
  - Each `DecisionRecord` yields one `medical.decision` event.
  - The end of an episode yields `medical.episode`.
  - Tracked frames yield `medical.hands` at no more than 15 Hz.
  - `medical.state` is sent at 1 Hz with `imu` ∈ {searching, connected, lost}.
  - The medical runtime's behavior is unchanged: all existing tests pass, and `VisionVerifier` with `observer=None` behaves identically (`tests/application/test_vision_verifier.py`, `test_vision_replay_equivalence.py`).
  - Evidence: new unit tests for `WebGui`, the observer hooks and `imu_status`, plus the existing suites.
- **AC-09 [FR-501, NFR-07, PRODUCT.md]:**
  - `npm --prefix web run build` produces `web/dist` with no external URLs (a grep for `http` in `dist` finds only same-origin references).
  - The dashboard renders the snapshot and updates live: IMU L/R state, current tag, episode active, the decision feed, the latest episode summary with contacts, and a 2D skeleton canvas from `medical.hands`.
  - It reconnects automatically with backoff.
  - It works in current Chrome and Firefox.
  - Evidence: the build output, the grep, and a manual check against `cpsa-supervisor --fake-medical` on the PC.
- **AC-10 [NFR-05, NFR-04]:**
  - After a board reboot, the DPU app is loaded and `cpsa-supervisor` runs with the medical mode, reachable at `http://10.42.0.83:8080/`.
  - `kill -9` of the supervisor gets it restarted within 10 s.
  - Evidence: `systemctl status`, `curl /health`, and a timing of the restart.
- **AC-11 [end to end, manual, Thursday]:** With the BlueCoins worn, a hand-to-face movement gives an IMU detection, then a vision verdict, then an actuation, and the dashboard shows the decision and the episode.
  - Evidence: the user's test, the event log, and the episode report JSON.

## Tasks

Parallel groups:
- **Group A**, start immediately: tasks 1, 2 and 4a.
- **Group B**, after 4a: tasks 5, 6, 7 and 8.
- **Group C**, after the Kaggle run: task 9.

The coordinator pins and pushes the notebooks after tasks 1 and 2 are integrated, because the user needs the pushed notebooks to start Kaggle.

### Task 1 — Fast staging with a manifest, and notebook 04 on the new mix
**Depends on:** none.

**Files:**
- `ml/src/cpsa_ml/datasets/staging.py`;
- new `ml/scripts/make_height_manifest.py`;
- new package data `ml/src/cpsa_ml/data/halfbust62_h256.txt.gz`, with the `pyproject` package-data entry;
- `ml/notebooks/04_variant_study.ipynb` and `ml/notebooks/04_variant_study/04_variant_study.ipynb` (keep them identical);
- `ml/notebooks/04_variant_study/kernel-metadata.json`;
- new `ml/tests/test_staging.py`;
- the notebook-04 cases in `ml/tests/test_notebooks.py`;
- `ml/README.md` ("Kaggle protocol" and "Training-list curation" sections only).

**Requirements:**
- User: "We need optimized code to train the halfbust model."
- D1: "COCO half-bust crops at least 256 px tall … HaGRID 30k … labeled by the RTMW-x teacher."

**Interfaces:**
- `stage_dataset(source, target, workers=64, include: Iterable[str] | None = None) -> Path`:
  - `include` holds POSIX paths relative to `source`;
  - a listed path that is missing raises `FileNotFoundError` naming the first 3 missing paths;
  - `data.yaml` is always copied.
- New `stage_archive(archive: Path, target: Path) -> Path`:
  - extracts a `.tar` with `tarfile` (filter `"data"`) and returns the dataset root (the folder holding `data.yaml`);
  - skips the work when the target already holds the completion marker.

**Actions:**
1. Replace `rglob("*")` + `is_file()` with `os.walk` and print the file count during listing.
2. Add `include` and `stage_archive`.
3. Write `make_height_manifest.py`. It takes `--root ~/Downloads/_output_/halfbust62 --min-height 256` and writes the sorted relative paths of qualifying train images, their label files, all val images and labels, and `data.yaml`. Gzip the result.
4. Generate the manifest. Expect 16,828 train images.
5. Update notebook 04:
   - locate `hagrid62.tar` with `find_inputs(INPUT, "hagrid62.tar")`;
   - stage the halfbust dataset with the manifest and the archive with `stage_archive`;
   - set `MIN_HEIGHT = 256` and `EXTRA_DATA = [(hagrid_root, 1)]` when the archive is present;
   - keep `VARIANTS = [DEFAULT_VARIANTS[2]]` and `EPOCHS = 30`;
   - print a wall-time line per epoch (an Ultralytics `on_train_epoch_end` callback in `variant_study` or in the notebook);
   - leave `LABELS_ROOT` logic untouched.
6. Pin `COMMIT_SHA` to the placeholder value the notebook tests expect. The coordinator re-pins after the merge.

**Acceptance:**
- Unit tests prove that listing performs no per-file `stat` (monkeypatch `Path.is_file` and `os.stat` to fail inside the listing), that manifest filtering and missing-path errors work, that tar extraction and the reuse marker work, and that the notebook-04 structure holds.
- The manifest line counts match the local dataset.

**Verification:**
- `cd ml && uv run pytest -q tests/test_staging.py tests/test_notebooks.py tests/test_curation.py tests/test_variant_study.py`
- `uv run ruff check src tests scripts`

### Task 2 — Notebook 06: HaGRID teacher labels on the Kaggle GPU
**Depends on:** none. It runs in parallel with Task 1. It must not edit the notebook-04 cases in `test_notebooks.py`; it adds its own cases.

**Files:**
- new `ml/notebooks/06_teacher_hagrid.ipynb` and `ml/notebooks/06_teacher_hagrid/{06_teacher_hagrid.ipynb, kernel-metadata.json}`;
- the notebook-06 cases in `ml/tests/test_notebooks.py`;
- `ml/src/cpsa_ml/pseudolabel.py`, only if the CUDA device string or the progress output needs a fix;
- `ml/README.md`, a new "Teacher labels for HaGRID" subsection only.

**Requirements:** D1 and D3: "Kaggle T4s for the teacher (ONNX Runtime CUDA)."

**Actions:**
1. Install `cpsa-ml` at `COMMIT_SHA`, `rtmlib==0.0.16` and `onnxruntime-gpu[cuda,cudnn]`.
2. Call `onnxruntime.preload_dlls()` when it is available.
3. Assert that `"CUDAExecutionProvider" in ort.get_available_providers()` and run one dummy session on CUDA. On failure, stop with a clear message naming the fallback: the local CPU with `--teacher rtmw-l`.
4. Download `https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip` (URL from notebook 07) to `/tmp` with `curl -C -`, and extract it.
5. Run `python -m cpsa_ml.pseudolabel --frames <hagrid_30k root> --out /tmp/hagrid_pl --device cuda --no-hands --teacher rtmw-x --val-fraction 0.1`. It is resumable and prints progress per chunk. Verify the device string rtmlib accepts.
6. Render 16 random labeled frames with keypoints to `/kaggle/working/samples/`.
7. Write `/kaggle/working/hagrid62.tar` from `/tmp/hagrid_pl/halfbust` (uncompressed, with `data.yaml` at the archive root) and print its size and the train/val frame counts.
8. `kernel-metadata.json`: GPU on, Internet on, no dataset sources.

**Acceptance:**
- The notebook-structure tests pass.
- A local smoke test runs the same pseudolabel command on 50 HaGRID frames with `--device cpu` (the `video-pipeline/ml/.venv` has `rtmlib`) and yields a loadable halfbust62 dataset that `stage_archive` (Task 1) accepts once integrated.

**Verification:**
- `cd ml && uv run pytest -q tests/test_notebooks.py tests/test_pseudolabel.py`
- the smoke-test command and its output, recorded in the task result.

### Task 4a — Frozen contracts (coordinator)
**Depends on:** none.

**Files:**
- `src/cpsa/ports.py` (F1, F2);
- `src/cpsa/supervisor/__init__.py` and `contract.py` (F4);
- `docs/protocol/v1/*.schema.json` and `examples/*.json` (F5);
- `tests/architecture/test_layers.py`, with new rules:
  - `protocol`: imports domain and vision; no third-party modules;
  - `supervisor`: imports domain, ports, protocol, application; no third-party modules;
  - `adapters` may import `protocol`;
- `pyproject.toml`: the `server = ["aiohttp>=3.9"]` extra, and `jsonschema` in `dev`.

**Acceptance:** the layer tests pass, and every example validates against its schema.

**Verification:**
- `uv run pytest -q tests/architecture tests/contract`
- `uv run mypy src`

### Task 5 — Web server adapter, supervisor and the `cpsa-supervisor` entry point
**Depends on:** 4a.

**Files:**
- `src/cpsa/adapters/web/server.py`;
- `src/cpsa/supervisor/supervisor.py`;
- `src/cpsa/board/supervisor_main.py`;
- the `cpsa-supervisor` script in `pyproject.toml`;
- F6 config in `src/cpsa/application/config.py` and `config/board.yaml`;
- `tests/adapters/test_web_server.py`, `tests/supervisor/test_supervisor.py`, `tests/application/test_config.py`;
- `README.md`, a new "Supervisor and web UI" section.

**Requirements:** IF-101..104, IF-201..205, IF-301..304, IF-601 and IF-701..703, quoted in `docs/specs/interface-rasd.md` §A–C, F and G.

**Actions:**
1. Build the aiohttp app with `/` (static `index.html`), `/assets/*`, `/health` (JSON with mode, mode_state and uptime) and `/ws`.
2. Give each client a sender task with:
   - a latest-wins dict for states;
   - a bounded event deque (256), with disconnect on overflow;
   - the envelope stamped per connection, with `time.monotonic()` and ISO UTC.
3. Implement `Publisher` with thread-safe calls through `loop.call_soon_threadsafe`.
4. Send `hello` and the snapshot on connect, and close idle connections after `idle_timeout_s`.
5. Implement the supervisor state machine with modes `idle` and `medical`, run on a worker thread:
   - `status` at 1 Hz and on change, built from `Mode.health()`;
   - `mode.set` handling and the busy error.
6. `supervisor_main`:
   - load the config;
   - start the server, then the default mode;
   - stop cleanly on SIGINT and SIGTERM.
7. Add `--fake-medical`: a fake `Mode` that replays `docs/protocol/v1/examples` messages on a loop, so the frontend can be developed on the PC.

**Acceptance:** AC-06 and AC-07.

**Verification:**
- `uv run pytest -q tests/adapters/test_web_server.py tests/supervisor tests/application/test_config.py tests/contract`

### Task 6 — Medical mode: WebGui, observer hooks and IMU status
**Depends on:** 4a. It runs in parallel with Task 5 and talks to Task 5 only through F1 and F4.

**Files:**
- `src/cpsa/protocol/medical.py` (builders: `decision(record)`, `episode(report)`, `hands(tracked, layout)`, `state(...)`);
- `src/cpsa/adapters/gui_web.py` (`WebGui`: `Gui` + `VisionObserver`, with `medical.hands` throttled to 15 Hz);
- `src/cpsa/application/vision_verifier.py` (F2 hooks only);
- `src/cpsa/application/runtime.py` (F3);
- `src/cpsa/board/medical_mode.py` (the `Mode` implementation, reusing `board_adapters`, `discover_bluecoins` and `build_runtime` from `src/cpsa/board/__main__.py`, with no behavior change to `cpsa-board`);
- tests under `tests/unit/test_protocol_medical.py`, `tests/adapters/test_gui_web.py`, `tests/application/test_runtime.py` and `tests/application/test_vision_verifier.py`.

**Requirements:**
- FR-401: "In medical mode, the existing runtime behavior SHALL be unchanged."
- IF-704: "The medical mode wraps the existing runtime (`build_runtime` and the dispatcher) without changing its behavior. Its `Gui` port implementation SHALL be the publisher for IF-451..454."
- IF-451..454 (as in `interface-rasd.md` §E).

**Actions:**
1. Add the F2 call sites, with observer exceptions logged and swallowed.
2. Add F3.
3. Write the builders, so the payload fields match the F5 schemas.
4. Write `WebGui`:
   - `publish(record)` sends a `medical.decision` event and updates `last_decision`;
   - the observer callbacks send `medical.episode` and `medical.hands`, and update the episode field of `medical.state`.
5. `MedicalMode.start`:
   - set imu to `searching`;
   - discover the BlueCoins (failure raises `ModeStartError("imu_not_found")`);
   - load the classifier (failure raises `internal`);
   - build the verifier with `observer=web_gui`; a vision startup failure raises `model_missing` or `camera_missing`, chosen by the cause;
   - `build_runtime(gui=web_gui)`, then `runtime.start()`.
6. `MedicalMode.stop` calls `runtime.stop()`. `health()` reports imu, camera, model and errors.

**Acceptance:** AC-08. The full existing suite passes unchanged.

**Verification:**
- `uv run pytest -q tests/application tests/unit/test_protocol_medical.py tests/adapters/test_gui_web.py tests/contract`

### Task 7 — Medical dashboard frontend
**Depends on:** 4a (schemas and examples). It needs Task 5's `--fake-medical` for the manual check; until that is integrated, it develops against a mock that replays `docs/protocol/v1/examples`.

**Files:** `web/` (`package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/**`) and `web/README.md`.

**Requirements:**
- FR-402 and FR-403 (as in `system-rasd.md`).
- FR-501, FR-504 and NFR-07.
- IF-103: clients ignore unknown types and fields.
- `PRODUCT.md` (platform, stack, users, principles, accessibility).

**Actions:**
1. Write a WebSocket client: same-origin `/ws`, reconnect with backoff, `ping` every 10 s, a typed message model generated by hand from the F5 schemas, and unknown types ignored.
2. Build the views:
   - connection and mode bar;
   - IMU L/R state;
   - current tag;
   - episode active;
   - decision feed (latest 20);
   - latest episode summary with its contact list;
   - a 2D canvas skeleton from `medical.hands`, using the halfbust62 bone list.
3. Bundle fonts locally and keep runtime CDNs out.
4. Make `npm run build` produce `web/dist`.

**Acceptance:** AC-09.

**Verification:**
- `npm --prefix web ci && npm --prefix web run build && npm --prefix web run typecheck`
- `grep -rEo 'https?://[^"'"'"' )]+' web/dist | sort -u` returns no external hosts.

### Task 8 — Board deployment as a boot service
**Depends on:** 5 and 6 integrated. It can be prepared in parallel.

**Files:**
- `deploy/systemd/cpsa-dpu.service` (oneshot: `xmutil unloadapp; xmutil loadapp kv260-benchmark-b4096`);
- `deploy/systemd/cpsa-supervisor.service` (`After=cpsa-dpu.service bluetooth.service`, `Restart=always`, `RestartSec=3`, the `kria` user, `cpsa-supervisor --config config/board.demo.yaml`);
- `deploy/deploy.sh`;
- `config/board.demo.yaml`;
- `README.md`, a new "Demo deployment" section.

**Requirements:** NFR-05 "the supervisor restarts on a crash (service manager)"; NFR-04; FR-201 "start automatically at boot".

**Actions:**
1. Write `deploy.sh`: rsync the repo (excluding `ml/`, `build/` and `.venv`) plus `web/dist` and the model file to `kria@10.42.0.83:~/cpsa`, then run `uv sync --extra board --extra server` on the board.
   - If the board has no internet, it first runs `uv pip download` on the PC for linux-aarch64 cp310 wheels and installs from them offline.
   - Then it installs and enables the units.
2. Write `board.demo.yaml`:
   - `vision.enabled: true`, `model.kind: xmodel`, `model.path: models/pose_halfbust62.xmodel`;
   - `input_width: 512`, `input_height: 384`;
   - the BlueCoin names `CPSA_L2` and `CPSA_R2`;
   - `server` and `supervisor` per F6.
3. Wednesday evening: a bring-up on the board with the existing random-weights xmodel (`video-pipeline/build/vitis/compiled/pose_halfbust62.xmodel`, compiled at 480×640, so use a temporary 640×480 input) and the BlueCoins, to prove the plumbing. Then a reboot test.

**Acceptance:** AC-10, plus the plumbing part of AC-11 (IMU detection to dashboard decision) with random weights.

**Verification:** `systemctl is-active cpsa-dpu cpsa-supervisor`, `curl -s http://10.42.0.83:8080/health`, the restart timing after `kill -9`, and the board journal.

### Task 9 — Quantize, compile and deploy the trained model
**Depends on:** AC-03 (the Kaggle run finished; the user downloads `variant_study/lrelu_512/` and `hagrid62.tar`), plus Task 8.

**Files:** none new. If needed, a `tools/vitis/README.md` runbook entry for the 384×512 trained-model command line.

**Actions:**
1. Run `tools/vitis/run_vitis.sh python tools/vitis/dump_quant_outputs.py --weights <best.pt> --height 384 --width 512 --threads 1 --calib-count 120 --test-count 64`, with calibration and test images from HaGRID val and COCO val, writing to `--out build/vitis/trained/quant`.
2. Compile with `tools/vitis/compile.sh`.
3. Check the fingerprint with `inspect_xmodel.py`.
4. Run `eval_quant.py` for INT8 vs FP32 on the HaGRID val and COCO val splits.
5. Copy the xmodel to the board, restore the 512×384 input in the config, and restart the service.
6. Measure fps (AC-05).
7. Do a quick live check with the user (AC-11).

**Acceptance:** AC-04, AC-05 and AC-11.

**Verification:** the tool outputs listed above, plus the board timing log.

## Integration verification

Run once by the coordinator after the integration of Group B:
```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src
cd ml && uv run pytest -q && uv run ruff check .
npm --prefix web ci && npm --prefix web run build && npm --prefix web run typecheck
```

Cross-task checks:
- `cpsa-supervisor --fake-medical --config config/board.yaml` on the PC, with the dashboard open in Chrome and Firefox (AC-09).
- `cpsa-board --config config/board.yaml` behaves as before (FR-401).
- The architecture layer tests pass with the new packages.

## Timeline (Wednesday 2026-11-11, from about 12:00)

| When | Who | What |
|---|---|---|
| Now | User | Stop the running notebook 04 session |
| Now to about 13:30 | Agents | Tasks 1, 2 and 4a; coordinator pins and pushes |
| About 13:30 | User | Import and run notebook 06 (about 1 h) |
| About 14:30 | User | Run notebook 04 with notebook 06's output attached (30 epochs, an estimated 2–4 h, CPU-bound on 4 vCPU) |
| Afternoon | Agents | Tasks 5, 6 and 7 |
| Evening | Agents + user (board, BlueCoins) | Task 8 bring-up with random weights |
| Thursday early morning | Agents | Task 9; the user tests the demo |

## Follow-up: lab gesture route (D8; not part of the Thursday slice)

This section is planning input for the lab plan. It is not executed with tasks 1–9. It lists the reuse design and the checks that decide it.

**Pipeline per frame (lab mode, continuous perception):**
1. The shared body model (the same `.xmodel` as medical, system FR-101) gives the person, the face, the wrists and the coarse 21-point hands.
2. A hand box is derived per side from the body model's hand keypoints, falling back to the elbow-to-wrist direction plus the face width when the hand points have low confidence. A square crop is cut from the full-resolution 640×480 frame.
3. The `dynamic_gestures` crop classifier (128×128 input, ONNX upstream; preprocessing: RGB, mean 127, std 128, NCHW) runs on the DPU and gives a static gesture class per hand: fist, palm, grabbing, point and so on (the HaGRID classes).
4. `dynamic_gestures`' OC-SORT tracking and action controller turn the tracked boxes and classes into dynamic events: zoom, drag-and-drop, swipes, click, tap. This is pure logic on the board.
5. Our own rules on top of the body keypoints and the gesture classes cover the lab gestures upstream lacks: explode (both palms moving apart), reassemble (both fists moving together) and reset (fist held still).

**Checks that gate the route (run first in the lab plan):**
- **L-1 DPU compile:** the classifier's architecture is not stated upstream. Export it, then run the existing ONNX operator check (`cpsa_ml.onnx_check`) and Vitis AI 3.5 PTQ and compile for `0x101000016010407`.
  - Pass: it compiles fully onto the DPU, and its INT8 accuracy on HaGRID val crops stays within 2 points of FP32.
  - Fail (for example, attention blocks): retrain a small CNN classifier (a MobileNet- or ResNet18-class model) on HaGRID hand crops, about 1 GPU hour on Kaggle, with the same class list.
- **L-2 Crop match:** upstream trained the classifier on crops from its own detector boxes. Our boxes come from body keypoints, so the size and offset can differ.
  - Measure the classifier's accuracy on HaGRID val crops cut from teacher keypoints the way our runtime would cut them.
  - Accept if it is within 3 points of crops from ground-truth boxes. Otherwise fine-tune the classifier on keypoint-derived crops.
- **L-3 Gesture set and FR-301:**
  - Map the upstream events to the lab actions: zoom gives scale, drag-and-drop gives grab and rotate, swipe gives the next or previous model, click gives select.
  - Keep our own rules for explode, reassemble and reset.
  - Revise `docs/specs/system-rasd.md` FR-301 (and Q-05 there) to the final gesture set.
  - Revise FR-004 of `docs/specs/vision-2d-rasd.md` and FR-101 of `docs/specs/system-rasd.md` for one body model plus a gesture classifier, instead of "one pose model".
- **L-4 Licenses:** the `dynamic_gestures` code is "a variant of Apache License, Version 2.0" and HaGRID is "a variant of Creative Commons Attribution-ShareAlike 4.0". Read both variants (the `license/` folders) and record the result in the asset and model manifests before any public release. Both are acceptable for the open-day demo.
- **L-5 Vendoring:** decide whether to vendor the action controller and OC-SORT under `src/cpsa/` with attribution or to depend on the package. The code must sit in the pure layer (no I/O) and be adapted to our box and class types.

## Open decisions and risks

- **ONNX Runtime CUDA on Kaggle:** the Kaggle image ships torch cu130. `onnxruntime-gpu` uses CUDA 12 libraries pulled as pip wheels.
  - Mitigation: Task 2 step 3 checks it first.
  - Fallback: the local CPU (16 cores) with `rtmw-l`, which takes a few hours and delays training to the evening.
- **Kaggle epoch time is unmeasured:** the data loader is likely CPU-bound on 4 vCPU. If epoch 1 takes more than 8 min, cut to 20 epochs (cosine LR adapts) rather than miss Thursday.
- **Teacher label quality:** RTMW-x labels become training targets with `v = 2`. The rendered samples (AC-02) are the only check before training.
- **The board's Python environment and internet** are unknown since the refactor. Task 8 covers both cases.
- **The IMU path has not run on real BlueCoins since the refactor.** The code is a near-verbatim port of `fcf3ec9:sensors/BLE/bluecoin.py`, and the user accepted this. Task 8's bring-up is the first hardware run.
- **D4 deviation:** the lab will need the perception service, which moves camera ownership out of `VisionVerifier`. That is planned later and does not change the protocol.
- **Licenses:** HaGRID is CC BY-SA 4.0 and RTMW is Apache-2.0. Both are fine for a university demo. Check ShareAlike before any weight release.
