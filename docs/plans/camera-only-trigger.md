# Camera-only medical trigger — Implementation Plan

**Status:** READY
**Goal:** The medical demo runs on the KV260 with **no BlueCoins**. The camera watches continuously. An episode, with a decision, an optional actuation, the event log and the dashboard, starts when vision sees at least 2 hand-to-face contacts within 10 s, or rocking/flapping sustained for at least 3 s. The IMU-triggered mode stays unchanged and selectable.

**Spec:**
- `docs/specs/vision-2d-rasd.md`, "Revision 2026-11-11" (FR-026..FR-032, NFR-007, AC-019..AC-026), plus the existing FR-014, FR-016, FR-019, FR-020 and FR-003;
- `docs/specs/system-rasd.md` and `docs/specs/interface-rasd.md`, "Revision 2026-11-11".

**Base:** HEAD `58e587b` plus the uncommitted T10–T12 work (cv2 letterbox, camera `auto`, command logging, connection log, deploy persistence) and the spec revisions. The tree is dirty; do not lose those changes.

**Scope / non-goals:**
- No new detection algorithms: reuse `EpisodeAnalyzer`, `detect_contacts`, `detect_rocking`/`detect_flapping` and the severity rules.
- No changes to verdict math, thresholds or the IMU path.
- No lab mode.
- No shared perception service (plan D4 of `medical-demo-slice.md` still applies).

**Constraints:**
- Must fit tonight (2026-11-11). The demo is Friday 2026-11-13.
- KV260 frame cost after T10 is about 21–25 ms (DPU 18.5 ms); `analysis_fps` defaults to 10.
- NFR-02: the camera-only medical mode must install and run **without** the BLE extras. No module-level `blue_st_sdk` or `mbientlab` import on that path.
- The layer rules in `tests/architecture/test_layers.py` hold: `cpsa.vision` uses numpy only; the application layer has no I/O.

## Approach

**Vision acts as the classifier.** With `medical.trigger: vision`:
1. The medical mode skips BlueCoin discovery and the native classifier, and builds the runtime with a no-op sensor source.
2. `VisionVerifier` runs in **continuous** mode:
   - every captured frame is offered for analysis;
   - while no episode is active, frames on the `analysis_fps` grid go to a **watch** path: pose, then a rolling watch `EpisodeAnalyzer`, then `VisionTrigger`.
3. When `VisionTrigger` fires, the verifier calls an injected `on_detection(Detection)`, which puts it into the dispatcher's `LatestSlot`, exactly where the IMU `ClassificationStage` puts its detections today.
4. From there the existing flow is untouched:
   - the dispatcher calls `observe` (the episode starts with the pre-roll lookback) and `verify` (the stage-1 assessment gives verdict and severity, FR-016/FR-017);
   - then cooldown, actuation policy, event log and `Gui`;
   - the episode ends on idle (FR-003); FR-020 follow-ups go through the same path.
5. While an episode is active, each new contact or a still-active oscillation makes the trigger emit follow-up detections; the dispatcher cooldown rate-limits them.
6. When the episode ends, the trigger emits one `NON_STEREOTIPY` detection, so the tag on the dashboard resets, and the refractory period starts.

### Frozen interfaces (do not change without coordinator approval)

**F7 — `cpsa.vision.trigger`** (new, pure, numpy and stdlib only):

```python
@dataclass(frozen=True)
class TriggerConfig:
    min_contacts: int = 2  # N (FR-028a)
    contact_window_s: float = 10.0  # T
    oscillation_min_s: float = 3.0  # D (FR-028b)
    refractory_s: float = 5.0  # FR-029; the config loader defaults it to dispatch.cooldown_s


@dataclass(frozen=True)
class TriggerFire:
    t: float  # monotonic frame time of the firing frame
    severity: Tag  # FR-016 contact severity; NON_DANGEROUS for an oscillation-only fire
    reason: Literal["contacts", "oscillation", "clear"]  # "clear" carries Tag.NON_STEREOTIPY


class VisionTrigger:
    def __init__(self, cfg: TriggerConfig) -> None: ...
    def contact(self, t_start: float, severity: Tag) -> None: ...  # once per NEW contact event
    def oscillation(
        self, t: float, active: bool
    ) -> None: ...  # per analyzed frame (rocking or flapping)
    def poll(self, t: float, episode_active: bool) -> TriggerFire | None: ...
    def episode_ended(
        self, t: float
    ) -> TriggerFire: ...  # returns the "clear" fire; starts the refractory period
```

Semantics:
- **Start** (`episode_active=False`, not refractory):
  - fire `contacts` when at least `min_contacts` contact starts fall within the last `contact_window_s`; the severity is the max of those contacts;
  - else fire `oscillation` when `active` has been continuously true for at least `oscillation_min_s`.
- **Follow-up** (`episode_active=True`): fire on every new contact (its severity), or once per `oscillation_min_s` of continued oscillation.
- **Refractory:** no start fire within `refractory_s` after `episode_ended`.
- **Dedupe:** each contact counts once.
- Pinned by `tests/unit/test_vision_trigger.py` (AC-021, AC-022 cases).

**F8 — `VisionVerifier` keyword arguments** (all optional; defaults keep today's behavior, AC-026):

```python
VisionVerifier(..., continuous: bool = False,
               trigger: VisionTrigger | None = None,
               on_detection: Callable[[Detection], None] | None = None)
```

- With `continuous=True` both `trigger` and `on_detection` are required; otherwise raise `ValueError`.
- `on_detection` is called from the analysis thread and must not block. Exceptions are logged and swallowed.
- The fired `Detection` is `Detection(tag=fire.severity, timestamp=clock.now(), window_end_ts=fire.t, features=())`.
- A `clear` fire gives `tag=Tag.NON_STEREOTIPY`.

**F9 — Config** (`cpsa.application.config`, `config/board.yaml` commented defaults, `config/board.demo.yaml`):

```yaml
medical:
  trigger: imu # imu | vision
  vision_trigger:
    min_contacts: 2
    contact_window_s: 10.0
    oscillation_min_s: 3.0
    refractory_s: null # null -> dispatch.cooldown_s
```

A missing `medical` section means `trigger: imu`. Validate the ranges, and raise `ConfigError` on bad values.

**F10 — Runtime builder** (`cpsa.application.runtime`):

```python
def build_vision_runtime(config: Config, *, verifier: VisionVerifier, actuators: Sequence[Actuator],
                         sink: EventSink, gui: Gui, clock: Clock) -> tuple[Runtime, Callable[[Detection], None]]
```

- It returns the runtime and the `on_detection` callable that feeds its `LatestSlot`.
- It uses the same `Dispatcher` as `build_runtime`, and a no-op `SensorSource` (a private class in the application layer, no I/O).
- `Runtime.imu_status` is not used in vision mode; the medical mode reports `"n/a"`.

**F11 — Protocol:** `medical.state.trigger` (`"imu"` | `"vision"`) is already in `docs/protocol/v1/medical.state.schema.json` (coordinator). In vision mode, `imu` is `"n/a"` in `medical.state` and `status`.

## Verification contract

| AC | Requirement | Expected | Evidence |
|---|---|---|---|
| AC-019 | FR-026 | `trigger: vision`, no BlueCoin powered: medical reaches `running`, `status.imu == "n/a"`, and no `blue_st_sdk` import happens | `tests/board/test_medical_mode.py` (fakes, an import guard via a `sys.modules` check); board run |
| AC-020 | FR-027, NFR-007 | Over 60 s with no trigger, analyzed frames ≥ 0.9 × `analysis_fps` × 60 on the KV260 | Unit test of continuous offering in `tests/application/test_vision_verifier.py`; board measurement (V4) |
| AC-021 | FR-028 | 2 contacts in 10 s start an episode; 1 does not; 2 contacts 12 s apart do not; 3 s of flapping starts one; 2 s does not | `tests/unit/test_vision_trigger.py`; a replayed contact sequence in `test_vision_verifier.py` |
| AC-022 | FR-029 | The episode ends after the idle time; no start within the refractory period; a start after it | `test_vision_trigger.py` and `test_vision_verifier.py` |
| AC-023 | FR-030 | A fire yields exactly one `medical.decision` with the vision tag and verdict `CONFIRMED`, an event-log line, and an actuation when one is present; the end yields a `NON_STEREOTIPY` tag change | `tests/application/test_runtime.py` (vision runtime, fakes); board |
| AC-024 | FR-031 | No actuator: AC-023 holds without the actuation | Same suite |
| AC-025 | FR-032 | Camera missing with `trigger: vision` gives `camera_missing`/`camera_error` in `status.errors`, and the UI keeps serving | `test_medical_mode.py`; board unplug test |
| AC-026 | regression | `trigger: imu` is unchanged | The full existing suite, unchanged and green |

## Tasks

**Parallelism:** V1 and V3 can run in parallel. V2 starts after F7/F8 exist; V2 may stub them from the frozen signatures and integrate once V1 lands. V4 runs after V1–V3 are integrated.

### Task V1 — VisionTrigger and continuous VisionVerifier
**Depends on:** none.

**Files:**
- `src/cpsa/vision/trigger.py` (new);
- `src/cpsa/application/vision_verifier.py` (continuous and watch path only);
- `src/cpsa/vision/episode.py`, only if a read-only accessor for new contact events is needed (for example `contact_events -> tuple[tuple[float, Tag], ...]`). Read-only; no grading changes.
- Tests: `tests/unit/test_vision_trigger.py` (new) and new test functions in `tests/application/test_vision_verifier.py`.

**Requirements:**
- FR-027: "WHILE `trigger = vision` and the mode runs, the system SHALL run pose inference and subject tracking continuously on the analyzed-frame grid (`analysis_fps`), with no external detection."
- FR-028: "start an episode WHEN either (a) at least `N` hand-to-face contacts (FR-014 definition) occur within `T` seconds; or (b) rocking or hand flapping (FR-019) is detected continuously for at least `D` seconds."
- FR-029: "an episode SHALL continue while contacts or oscillation persist, and SHALL end after the idle time of FR-003 … a new episode SHALL NOT start within a configurable refractory period …"
- FR-030 (the detection part): the tag comes from vision; contacts use FR-016 severity; oscillation-only gives `NON_DANGEROUS`.

**Actions:**
1. Implement F7 exactly.
2. In `VisionVerifier`, with `continuous=True`:
   - `_on_frame` always offers frames;
   - when no episode is active, frames on the `analysis_fps` grid go to a watch method: `estimator.estimate`, then a dedicated watch `EpisodeAnalyzer` (started once with a synthetic id, restarted after each real episode and at least every 5 minutes, to bound memory), then feeding `trigger.contact(...)` for each new contact event and `trigger.oscillation(t, rocking or flapping)` from `detect_rocking`/`detect_flapping` over the watch frames of the last `oscillation_min_s` seconds;
   - call `trigger.poll` and forward any fire through `on_detection` as F8 specifies;
   - during an episode, keep today's `_analyze_frame`, and additionally feed the trigger from the episode analyzer's events for follow-ups;
   - in `_finish_episode`, call `trigger.episode_ended(now)` and forward the clear fire.
3. Continuous mode must not change behavior when `continuous=False`: the existing tests stay unchanged and green.
4. Watch-path inference errors use the existing source-health and `episode_error` logging. Make sure one bad frame does not stop the watch.

**Acceptance:** AC-021, AC-022, the unit part of AC-020, and AC-026.

**Verification:**
- `uv run --no-sync pytest -q tests/unit/test_vision_trigger.py tests/application/test_vision_verifier.py tests/application/test_vision_replay_equivalence.py tests/architecture`
- ruff and mypy on your files.

### Task V2 — Medical mode wiring, config, runtime builder, WebGui field
**Depends on:** F7/F8 (stub them until V1 lands).

**Files:**
- `src/cpsa/application/config.py` (F9);
- `config/board.yaml` (commented defaults) and `config/board.demo.yaml` (`trigger: vision` with the defaults);
- `src/cpsa/application/runtime.py` (F10);
- `src/cpsa/board/medical_mode.py` (vision branch);
- `src/cpsa/board/__main__.py` (keyword seams only, if needed; `cpsa-board` stays IMU-only and unchanged);
- `src/cpsa/adapters/gui_web.py` (the `trigger` field in `medical.state`; `imu "n/a"` in vision mode);
- `src/cpsa/protocol/medical.py` (the `state(...)` builder gains `trigger`);
- tests: `tests/application/test_config.py`, `tests/application/test_runtime.py`, `tests/board/test_medical_mode.py`, `tests/adapters/test_gui_web.py`, `tests/unit/test_protocol_medical.py`.

**Requirements:**
- FR-026: "WITH `trigger = vision`, it SHALL start and run with no BlueCoin or other IMU present, and SHALL NOT scan for or connect to IMUs."
- FR-030: "… SHALL go through the same cooldown, actuation policy, event log and `Gui` publishing as IMU decisions."
- FR-031: no actuator, everything else continues.
- FR-032: camera or DPU failure gives an error code in status; the supervisor stays operable.
- IF-451 revision: `medical.state.trigger`.

**Actions:**
1. Implement F9 and F10.
2. `MedicalMode.start` with `trigger: vision`:
   - imu stays `"n/a"`; no discovery; no classifier load; no BLE imports (lazy imports only on the IMU branch);
   - build the verifier with `continuous=True`, a `VisionTrigger(TriggerConfig(...))`, and the `on_detection` returned by `build_vision_runtime`;
   - camera and DPU failures map to the existing `ModeStartError` codes.
3. `health()` reports imu `"n/a"` in vision mode.
4. `WebGui` and the protocol builder include `trigger`. Validate against the F11 schema.

**Acceptance:** AC-019, AC-023, AC-024, AC-025 and AC-026.

**Verification:**
- `uv run --no-sync pytest -q tests/application tests/board tests/adapters tests/unit/test_protocol_medical.py tests/contract tests/architecture`
- ruff and mypy.

### Task V3 — Dashboard shows the trigger
**Depends on:** F11 (already in place).

**Files:** `web/src/protocol.ts`, `web/src/state.ts`, `web/src/views.ts`, `web/src/labels.ts`, `web/src/mock.ts`, `web/index.html` and `web/src/styles.css`, as needed.

**Requirements:**
- IF-451 revision: `medical.state.trigger` is `"imu"` or `"vision"`; unknown fields are ignored;
- PRODUCT.md: the UI is in Italian.

**Actions:**
- Parse the optional `trigger` field.
- With `vision`, show "Attivazione: telecamera" in the live-status area and render the IMU tile as "Non usato (solo telecamera)", not as an error.
- With `imu`, the display is unchanged.
- Add both variants to the mock.

**Acceptance:** builds, typechecks, no external URLs; the mock shows both variants.

**Verification:** `npm --prefix web ci && npm --prefix web run build && npm --prefix web run typecheck`, then the dist URL grep.

### Task V4 — Full-pipeline board test with the provisional (COCO-only) model
**Depends on:** V1–V3 integrated and the full gate green. The coordinator runs this together with the user. It is not a code task.

**Steps:**
1. **Quantization drop** (already measured on the host): INT8 vs FP32 PCK@0.2fw on 64 COCO h≥160 val crops (`build/vitis/coco_lrelu512/eval/comparison.md`): face 0.81 vs 0.89; shoulders 0.15 vs 0.19; elbows 0.08 vs 0.05; wrists 0.00 vs 0.00; hands about 0.01–0.02. Report the drop per group and note that the FP32 model itself is weak on wrists and hands (the sigma defect, fixed in `d2accc5`).
2. **Deploy** with `deploy/deploy.sh --model build/vitis/coco_lrelu512/compiled/pose_halfbust62.xmodel`, using the board config `trigger: vision`, camera `auto`, MetaMotion `enable: false` (or true if the user wants it). Confirm that the T11 post-sync patches applied.
3. **Throughput and drops** (60 s, no trigger):
   - analyzed frames per second (AC-020);
   - DPU ms and total ms per frame (T10 timing);
   - the verifier's `dropped` frame counter (queue drops), the camera fps and the camera timestamp offset;
   - CPU load (`top -b -n 3`);
   - `/health` responsiveness (curl latency).
4. **End-to-end episode:**
   - the user, seated at 0.5–0.7 m, taps the face 3 times within 10 s. Expect an episode, a `medical.decision` (vision tag, `CONFIRMED`), an event-log line, the dashboard episode and contacts, and the connection log;
   - then 3–5 s of flapping;
   - then a single nose scratch, which must not start an episode;
   - with the COCO-only model, wrist accuracy is low, so record misses as model limits, not plumbing failures.
5. **Camera unplug test** (AC-025): unplug, check status errors, replug, then `mode.set medical` to recover.
6. **Report:** a table of measured values against the targets, plus the observed issues.

## Integration verification

After V1–V3, the coordinator runs once:

```bash
uv run --no-sync pytest -q && uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy src
npm --prefix web ci && npm --prefix web run build && npm --prefix web run typecheck
```

Then a scoped `$review` and `$verify` (Sonnet 5.5 high via Orca) covering T10–T12 and V1–V3. Then the user approves the commit, and V4 runs.

## Open decisions / risks

- **Model quality:** the COCO-only model has wrist PCK 0.0, so contacts may be missed and the vision trigger may not fire in V4. That is expected until the HaGRID model (notebook 04 with `kpt_oks_sigmas`) arrives. V4 then validates plumbing, throughput and drops; the trigger itself is validated with the HaGRID model.
- **CPU load:** continuous 10 fps inference plus web, on 4 A53 cores. If AC-020 fails, lower `analysis_fps` to 8 and record it.
- **Watch-analyzer memory:** bounded by periodic restarts (V1 step 2).
- **Booth tuning:** Q-06 (N, T, D) and Q-07 (refractory), during the rehearsal.
