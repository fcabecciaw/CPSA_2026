# CPSA_2026

CPSA detects motor stereotypies from two wrist-worn IMUs and answers them with
haptic, audio or light feedback. It runs as a single Python process on a Kria
KV260 (aarch64, Ubuntu 22.04) and is fully installable and testable on any host
with in-memory fakes.

```text
 BlueCoin L ─┐
             ├─> synchronizer ─> windower ─> classification ─> latest-detection slot
 BlueCoin R ─┘   (skew/stale)    (150 / 75)  (native libs)          │
                                                                    v
                         event log <── dispatcher ──> vision verifier (deadline, fail-open)
                         GUI       <──     │
                                           └──> policy ──> MetaMotion / speaker / LED strip
```

1. **Acquisition.** Two STMicroelectronics BlueCoin boards stream acceleration
   (mg), angular rate (dps) and compact quaternions (×10000) over BLE.
2. **Synchronization.** Left and right samples are combined into one 20-value row
   only when their timestamps differ by at most `sync.max_skew_ms`; the older wrist
   is dropped otherwise, and a wrist waiting longer than `sync.stale_ms` is discarded.
3. **Windowing.** Rows are grouped into windows of 150 rows every 75 rows. Both
   values are fixed by the trained model and the native library interface.
4. **Classification.** Each window is preprocessed (mg to m/s², quaternion frame
   alignment) and passed to two vendored native libraries: feature extraction
   (150 × 20 → 18 features) and a decision tree (18 features → tag). Window #1 is
   discarded and window #2 only calibrates the extractor; every later window
   produces a detection with one of the tags `NO_CLASS` (0), `NON_DANGEROUS` (1),
   `DANGEROUS` (2), `NON_STEREOTIPY` (3).
5. **Hand-off.** Detections go through a single-slot hand-off: a new detection
   replaces one not yet processed, so classification never waits for the dispatcher.
6. **Verification gate.** For tag 1 or 2, when the cooldown allows, the dispatcher
   asks the vision verifier for a verdict (`CONFIRMED`, `REJECTED`, `UNDECIDED`)
   within `dispatch.verification_deadline_s`. Only `REJECTED` within the deadline
   prevents actuation; `CONFIRMED`, `UNDECIDED`, a timeout or a verifier error all
   actuate (fail-open). With the shipped `vision.enabled: false` the stub verifier
   always answers `UNDECIDED`, so actuation depends on the IMU tag alone. Enabling
   vision replaces it with the 2D pose episode analyzer (see
   [Vision verification](#vision-verification)).
7. **Actuation.** The policy fires every available actuator: a mild response for
   tag 1, a strong one for tag 2. Same-tag detections are not actuated again until
   `dispatch.cooldown_s` has elapsed; a tag change resets the cooldown.
8. **Logging.** Every tag change and every actuation decision is written to the
   event log and published to the GUI (a stub that renders nothing).

## Architecture

The package `cpsa` is layered; each layer may import only the layers listed for it.
`tests/architecture/test_layers.py` enforces this rule on every test run.

| Package | Responsibility | May import |
|---|---|---|
| `cpsa.domain` | Pure rules: model types, cooldown and verdict gate, actuation policy | stdlib |
| `cpsa.imu` | Synchronization, windowing, preprocessing | `domain`, numpy |
| `cpsa.vision` | Pure 2D analysis: layouts, pose decode, tracking, features, episode assessment | `domain`, numpy |
| `cpsa.ports` | `typing.Protocol` contracts between application and adapters | `domain`, `vision`, numpy |
| `cpsa.application` | Configuration, classification stage, slot, dispatcher, runtime wiring, vision verifier | `domain`, `imu`, `ports`, `vision`, numpy, yaml |
| `cpsa.adapters` | Native libraries, BLE sensors, camera/pose/evidence adapters, actuators, logs, verifier stub, null GUI | `domain`, `ports`, `vision`, any library |
| `cpsa.recorder` | `cpsa-record` entry point: scripted session capture | `domain`, `imu`, `ports`, `application`, `adapters`, `vision` |
| `cpsa.board` | `cpsa-board` entry point: builds the real adapters and runs | everything |

Only `cpsa.adapters` and `cpsa.board` may use `ctypes` or the hardware libraries
(`blue_st_sdk`, `bluepy`, `mbientlab`, `flux_led`, `playsound`, `dbus_fast`).

### Ports

| Port | Implementations | Contract |
|---|---|---|
| `SensorSource` | `adapters.bluecoin.source.BlueCoinSource` | `discover(timeout_s) -> set[str]`, `start(on_sample)`, `stop()` |
| `FrameSource` | `adapters.camera.f200.F200Source` | `start(on_frame)`, `stop()`; delivers paired `RgbdFrame`s with `CLOCK_MONOTONIC` timestamps |
| `VideoSource` | `adapters.camera.uvc.UvcVideoSource` | `start(on_frame, on_error)`, `stop()`; RGB-only UVC frames with `CLOCK_MONOTONIC` timestamps; a failure after start is reported once through `on_error` |
| `FeatureExtractor` | `adapters.native.classifier.NativeFeatureExtractor` | `(150, 20)` preprocessed window → `(18,)` features |
| `Classifier` | `adapters.native.classifier.NativeClassifier` | features → `Tag`; raises `ClassificationError` on an unknown code |
| `PoseEstimator` | `adapters.pose.onnx.OnnxPoseEstimator`, `adapters.pose.dpu.DpuPoseEstimator` | RGB image → persons; raises on a model or accelerator failure |
| `Verifier` | `adapters.verifier_stub.StubVerifier` by default; `application.vision_verifier.VisionVerifier` when `vision.enabled` | detection → `Verdict`; may block or raise. The vision verifier also implements `DetectionObserver` |
| `EvidenceSink` | `adapters.evidence.FileEvidenceSink` | `write_episode(episode_id, frames, fps, keyframe)`, `purge(now)` |
| `Actuator` | `adapters.actuators.{metamotion,speaker,led_strip}` | `id`, `kind`, `execute(params)`, `stop()`; never raises for a disconnected device |
| `EventSink` | `adapters.event_log.JsonlEventSink` | `write(DecisionRecord)` |
| `Gui` | `adapters.gui_null.NullGui` | `start()`, `publish(DecisionRecord)`, `stop()` |
| `Clock` | `adapters.clock.SystemClock` | `monotonic()`, `now()` |

### Replacing a component

- **Vision verifier:** with `vision.enabled: false` the runner uses `StubVerifier`.
  With `vision.enabled: true`, `_build_verifier()` in
  `src/cpsa/board/__main__.py` is the single construction site of
  `VisionVerifier` over `UvcVideoSource`, the configured pose estimator and
  `FileEvidenceSink`. To use a different analyzer, implement `cpsa.ports.Verifier`
  (optionally with `cpsa.ports.DetectionObserver` and `start`/`stop` methods) and
  return it there; the runtime starts and stops a shipped `VisionVerifier` and
  tees IMU samples to it. The dispatcher already enforces the deadline and
  fail-open behavior; a verifier may block or raise. A verifier that hangs keeps
  the single verification worker busy, so later verifications time out and fail
  open; it does not delay shutdown.
- **GUI:** implement `cpsa.ports.Gui` and construct it instead of `NullGui()` in
  the same function. `publish` is called from the dispatcher thread for every
  event-log record.
- **Policy:** change `actuations_for()` in `src/cpsa/domain/policy.py`. It maps a
  tag and the available actuator ids (`meta_<mac>`, `speaker_<mac>`, `led_<ip>`)
  to a list of actuations.

| Actuator | Tag 1 (mild) | Tag 2 (strong) |
|---|---|---|
| MetaMotion (`meta_`) | `duty=40, duration=700` ms | `duty=100, duration=900` ms |
| Speaker (`speaker_`) | `stereotipia_non_pericolosa_1.mp3` | `stereotipia_pericolosa_1.mp3` |
| LED strip (`led_`) | white, intensity 50 | red, intensity 100 |

The LED strip turns itself off after `actuators.led_strip.duration_ms`. Speaker
files are audio assets bundled in `cpsa.adapters.assets.audio`; an unknown name
plays `missing_audio_file.mp3` and logs a warning.

## Installation

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

Development host (any architecture, no hardware):

```bash
uv sync --extra dev
```

KV260 board:

```bash
uv sync --extra board          # add --extra dev to run the test suites on the board
```

The `board` extra installs `blue-st-sdk`, `bluepy`, `metawear`, `flux_led`,
`playsound` and `dbus-fast`. Board prerequisites:

- **System packages:** BlueZ and build tools, for example
  `sudo apt install bluetooth bluez libbluetooth-dev libglib2.0-dev libboost-all-dev build-essential libcap2-bin`.
- **BLE permissions:** add the user to the `bluetooth` group
  (`sudo usermod -aG bluetooth <user>`, then reboot) and give the bluepy helper raw
  socket capabilities:
  `sudo setcap "cap_net_raw+eip cap_net_admin+eip" .venv/lib/python3.*/site-packages/bluepy/bluepy-helper`.
- **blue-st-sdk on Python 3.10:** if it fails with `collections.MutableMapping`,
  change `collections.MutableMapping` to `collections.abc.MutableMapping` in
  `blue_st_sdk/utils/dict_put_single_element.py`.
- **MetaWear:** on the KV260 the release build of `warble` (a `metawear`
  dependency) can overflow a buffer. Rebuild
  [PyWarble](https://github.com/mbientlab/PyWarble) in debug mode: clone it with
  `--recurse-submodules`, add `CONFIG=debug` to the `make` command in `setup.py`,
  change the library path from `dist/release` to `dist/debug`, and install it with
  `uv pip install .` in place of the released `warble`.
- **Hardware:** a BLE adapter, two BlueCoins with the names in the configuration,
  and optionally a MetaMotion, a paired Bluetooth speaker and a flux_led Wi-Fi LED
  strip on the same network.

## Running

```bash
uv run cpsa-board --config config/board.yaml
```

Stop with `Ctrl+C` or `kill -TERM <pid>`: the GUI, dispatcher, sensors, the vision
verifier when enabled, and the actuators are stopped in this order and the
process exits with code 0.

| Exit code | Meaning |
|---|---|
| 0 | Orderly stop after SIGINT or SIGTERM |
| 1 | Unexpected error (logged with traceback) |
| 2 | Startup failure: invalid configuration (the message names the key), `board` or `vision` extra missing, a pose model missing when vision is enabled, a BlueCoin not found after the configured retries (the log names it), or the native libraries cannot be loaded (for example on x86) |

With no actuator enabled or found, detection, verification and logging continue.

## Recording sessions

`cpsa-record record` captures synchronized RGB-D frames and, unless `--no-imu` is
given, the two wrist IMUs during a scripted protocol of cued trials. Recording runs
on the PC, not on the KV260: the PC has the disk and CPU, and a BlueCoin accepts
only one BLE central, so `cpsa-board` must not run at the same time.

```bash
uv sync --extra camera --extra recorder   # add --extra board for the BlueCoin stack
uv run cpsa-record record \
  --config config/board.yaml \
  --protocol config/protocol.yaml \
  --participant P01 \
  --out ~/Desktop/CPSA_sessions \
  [--calibration calibration.json] [--no-imu] [--queue 90]
```

| Flag | Meaning |
|---|---|
| `--config` | Board YAML; camera modes and XU values are copied into `session.json` |
| `--protocol` | Scripted trial YAML (see below) |
| `--participant` | Pseudonymous id matching `^[A-Za-z0-9_-]{1,32}$`; the session id is `<YYYYmmdd-HHMMSS>_<participant>` |
| `--out` | Session root; a new `<session_id>/` directory is created inside it |
| `--calibration` | `calibration.json` to copy into the session (calibration capture is deferred) |
| `--no-imu` | Record frames only; the BlueCoin module is not imported and `imu.enabled` is `false` |
| `--queue` | Bounded frame queue size in frames (default 90) |

`config/protocol.yaml` lists the initial dangerous classes, confounders,
non-dangerous stereotypies and idle. `reps` is per hand and `hand: both` expands
into one `L` and one `R` trial, so the shipped file is a full session of about
35 minutes across the three distance blocks. For a short dry run set `reps: 1` and
keep one distance (for example `distances_cm: [55]`). Trial order is shuffled from
`seed`; the same seed and file always produce the same order. The protocol runs once
per distance block and the runner prompts the operator to set the distance before
each block. The safety instruction "simulate gently; no real hits; use the foam
head for impacts" is shown at the start.

During a session the runner shows the block/trial counter, the instruction, a
countdown and `GO` with a short cue tone (`aplay` when available, otherwise the
terminal bell). Keys: `b` marks the current or last trial bad, `p` pauses/resumes,
`q` finishes and finalizes. `Ctrl+C` and `SIGTERM` finalize cleanly too. Progress
(MB/s, frames, queue drops and IMU samples) is printed every 5 s; the final summary
reports the measured IMU sample rate per wrist and kind, which settles the unknown
BlueCoin rate. A missing BlueCoin after the configured retries exits with code 2
and names it.

Frames are written only inside the trial window `cue_ts - pre_s` to
`end_ts + post_s`; a pre-roll ring keeps the frames captured just before the cue.
IMU samples are recorded continuously for the whole session. If the frame queue
fills, the frame is dropped, counted in `frames_dropped_queue` and never blocks the
capture callback.

Session layout (schema v1):

```text
<root>/<session_id>/
  session.json        # schema version, ids, camera/XU, calibration sha256, protocol sha256+seed, counters, host
  calibration.json    # copy, only when --calibration was given
  frames.csv          # seq,trial_id,ts_rgb,ts_depth,rgb,depth
  rgb/NNNNNN.jpg      # JPEG quality 95, RGB order
  depth/NNNNNN.png    # uint16 PNG, raw Z16
  imu.csv             # ts,side,kind,v0,v1,v2,v3 (header only with --no-imu)
  trials.jsonl        # one JSON object per trial
```

All timestamps are `CLOCK_MONOTONIC` seconds. `trials.jsonl` fields are
`trial_id`, `class`, `hand`, `target`, `cue_ts`, `start_ts`, `end_ts`, `status`
(`ok` or `bad`), `note` and `distance_cm`; `distance_cm` is a schema-v1 field added
by the protocol runner so a dataset can stratify by camera distance.
`session.json` records the synchronizer settings under `imu.max_skew_ms` and
`imu.stale_ms` so replay behaves like the live run.

Exit codes: 0 after an orderly stop, 1 on an unexpected error, 2 for an invalid
configuration/protocol/participant/calibration, missing camera dependencies, a
missing BlueCoin after the configured retries, or an existing session directory.

## Replaying IMU on the board

`cpsa-replay-imu` feeds a recorded session's `imu.csv` rows in timestamp order
through the same `IMUSynchronizer` -> `Windower` -> `ClassificationStage` chain as
the live runtime, using the vendored native extractor and classifier. The clock
follows the recorded timestamps, so skew and stale handling match a live session.

```bash
uv run cpsa-replay-imu ~/Desktop/CPSA_sessions/20261106-103005_P01 --out tags.jsonl
```

The native libraries are aarch64-only; on any other host the command exits with
code 2 and says so. The output has one JSON object per detection:

```json
{"window_end_ts": 12345.678, "tag": "DANGEROUS", "timestamp": "2026-11-06T10:31:02.123456+01:00"}
```

Exit codes: 0 on success, 1 on an unexpected replay error, 2 for an invalid session
or a host without the native libraries.

## Vision verification

`vision.enabled: false` is the shipped default: `cpsa-board` builds
`StubVerifier`, never opens the camera and behaves exactly as before. With
`vision.enabled: true`, it builds `VisionVerifier` over an RGB-only UVC camera and
answers every tag-1/tag-2 detection with a 2D pose episode.

```bash
uv sync --extra vision --extra camera   # add --extra board on the KV260
uv run cpsa-board --config config/board.yaml
```

`vision.model.kind` selects the pose backend: `onnx` runs an exported model on the
CPU (the development stand-in) and `xmodel` runs a compiled DPU model on the
KV260. Export the model from `ml/` (see `ml/README.md`) and point
`vision.model.path` at it; no model weights are shipped. A missing runtime or
model file stops startup with exit code 2.

The pipeline is:

1. **Capture.** One UVC source delivers undecoded `CLOCK_MONOTONIC` frames for the
   whole session. A pre-roll ring bounded by `vision.preroll_s` keeps the frames
   just before a detection; an `on_error`, or a last frame older than
   `vision.stale_after_s`, marks the source unhealthy.
2. **Episode.** A detection starts an episode and later stereotypy detections
   extend it, including detections inside the dispatch cooldown. The frames from
   `window_end_ts - lookback_s` and the live frames are analyzed on one worker
   thread; no pose inference runs outside episodes.
3. **Selection.** Live and replay keep the first frame at or after each grid point
   `t0 + k / vision.analysis_fps`. Selection depends only on timestamps; if the
   worker falls behind, `vision.queue_max` drops the oldest frame and the
   assessment reports incomplete coverage instead of silently skipping.
4. **Assessment.** The analyzer tracks one subject and its keypoints, classifies
   hand-face contacts, hand shape, rocking and flapping, and returns
   `CONFIRMED` with a severity when the window has an event, or the episode's
   latest severity within `vision.analysis.activity_window_s`; `UNDECIDED` for a
   timeout, an unhealthy source, incomplete coverage, an estimator error or
   rocking/flapping only; `REJECTED` when the covered window is fully framed and
   has no event and no recent grade. `verify` waits at most
   `vision.fast_budget_s`, which must stay below
   `dispatch.verification_deadline_s`.
5. **Report and evidence.** At episode end the analyzer writes the episode report
   to the log tree, a `trace.jsonl` for audit replay, and, when evidence is
   enabled, a clip and keyframe around the peak-severity contact. Skeleton edges
   come from `vision.layout` (`HALFBUST_62` or `COCO17`). Evidence day folders
   older than `vision.evidence.retention_days` are purged; `0` keeps only today.
   The event log record carries the same `episode_id` in
   `decision.assessment.episode_id`.

Outputs:

```text
<vision.evidence.base_path>/<YYYY-MM-DD>/<episode_id>/clip.mp4      # MJPG .avi fallback
<vision.evidence.base_path>/<YYYY-MM-DD>/<episode_id>/keyframe.jpg  # skeleton overlay
<vision.evidence.base_path>/<YYYY-MM-DD>/<episode_id>/trace.jsonl   # pushed frames + assessments
<log.base_path>/<YYYY-MM-DD>/vision/reports.jsonl                   # one line per episode
<log.base_path>/<YYYY-MM-DD>/vision/<episode_id>/trace.jsonl        # only when evidence is disabled
```

The report also goes to `system.log` as `episode_report <json>`; its `trace`
field names the audit file for that episode.

### Replaying vision episodes

`cpsa-replay-vision` runs the same analyzer over a recorded session (schema v1):

```bash
uv run cpsa-replay-vision ~/Desktop/CPSA_sessions/20261106-103005_P01 \
  --onnx models/pose_halfbust62.onnx [--detections detections.jsonl] [--out vision.jsonl]
```

Without `--trace` it runs **ideal** replay: `select_frames` over all session
frames with no timing effects, for development and threshold tuning. The
session's `trials.jsonl` supplies the detection bounds unless `--detections`
gives one JSON object per line with `window_end_ts` (optional `tag` and
`timestamp`). No equivalence claim is made for this mode.

With `--trace <episode>/trace.jsonl` it runs **audit** replay: it pushes exactly
the traced frames, at the same push index, and re-evaluates every assessment with
the traced bounds, `now`, `timeout`, `error` and `source_healthy` flags. Verdicts,
features and the episode report are identical to the live run (FR-025, AC-014),
provided the same frames, model, layout and vision configuration are replayed.
IMU signals for the report come from the session `imu.csv`. The trace is JSON
Lines with one `episode`, `frame` and `assessment` record per line.

The output has one JSON object per assessment plus the episode report. Exit
codes: 0 on success, 1 on an unexpected replay error, 2 for an invalid session or
configuration, or a host without the pose model/runtime.

### Blocked vision gates

Two acceptance groups need hardware and recorded data:

- **AC-016 / AC-017 (DPU rate and demo-session metrics):** blocked until the
  KV260 and a compiled pose model are available; `scripts/bench_pose.py --onnx`
  only gives the CPU timing.
- **AC-007 / AC-008 (labeled clips and recorded sessions):**
  `tests/target/test_vision_sessions.py` is ready but skips until
  `CPSA_LABELED_SESSIONS` points at a labeled session tree (`labels.jsonl`); the
  test docstring documents the schema. Synthetic tests cover the same logic
  meanwhile.

## Configuration

`config/board.yaml` is the reference configuration. Every key is required and
unknown keys are rejected; an invalid file stops startup with exit code 2.

| Key | Type | Meaning |
|---|---|---|
| `log.base_path` | path | Root of the log folders; `~` is expanded |
| `log.level` | `DEBUG`…`CRITICAL` | Level of the console and system log |
| `bluecoins.left`, `bluecoins.right` | string | BLE names of the left and right BlueCoin |
| `bluecoins.scan_timeout_s` | > 0 | Duration of each discovery scan |
| `bluecoins.discovery_retries` | integer ≥ 0 | Extra scans before startup fails |
| `bluecoins.retry_interval_s` | > 0 | Wait between scans |
| `sync.max_skew_ms` | > 0 | Max left/right timestamp difference to combine samples |
| `sync.stale_ms` | > 0 | Discard a wrist's samples older than this |
| `dispatch.cooldown_s` | > 0 | Minimum time between actuations for the same tag |
| `dispatch.verification_deadline_s` | > 0 | Time the verifier gets before failing open. The shipped 1.0 s is a placeholder to be measured with a real verifier |
| `actuators.retry.scan_timeout_s` | > 0 | Scan duration for MetaMotion and LED devices |
| `actuators.retry.fast_attempts` | integer ≥ 0 | Fast reconnection attempts after a disconnection |
| `actuators.retry.interval_s` | > 0 | Wait between fast attempts |
| `actuators.retry.sleep_s` | > 0 | Wait between slow attempts (speaker, LED strip) |
| `actuators.metamotion.enable` | bool | Scan for and use MetaMotion devices |
| `actuators.speaker.enable`, `actuators.speaker.mac` | bool, string | Use the Bluetooth speaker with this MAC address |
| `actuators.led_strip.enable` | bool | Scan for and use flux_led LED strips |
| `actuators.led_strip.duration_ms` | integer > 0 | LED auto-off delay |
| `camera.rgb_device`, `camera.depth_device` | path | V4L2 nodes for RGB (YUYV) and depth (`Z16 `) |
| `camera.width`, `camera.height` | integer > 0 | Capture size in pixels; frames are not resized (640×480) |
| `camera.fps` | > 0 | Capture rate; RGB/depth pair tolerance is `1 / (2 · fps)` |
| `camera.rgb_exposure` | integer 83…10000 | Manual `exposure_time_absolute`; auto exposure is disabled |
| `camera.xu.laser_power` | integer 0…16 | Depth extension-unit laser power |
| `camera.xu.accuracy` | integer 1…3 | Depth extension-unit accuracy |
| `camera.xu.filter` | integer 0…7 | Depth extension-unit filter |
| `camera.xu.confidence` | integer 0…15 | Depth extension-unit confidence threshold |
| `vision.enabled` | bool | Run the 2D pose verifier instead of the stub |
| `vision.layout` | `HALFBUST_62` or `COCO17` | Keypoint layout of the pose model |
| `vision.model.kind` | `onnx` or `xmodel` | Pose backend: CPU ONNX export or KV260 DPU model |
| `vision.model.path` | path | Model file (not shipped) |
| `vision.model.input_width`, `vision.model.input_height` | integer > 0 | Network input size in pixels |
| `vision.model.conf_thr` | 0…1 | Minimum person score |
| `vision.model.iou_thr` | 0…1 | NMS IoU threshold |
| `vision.camera.device` | path | V4L2 node of the RGB-only UVC camera |
| `vision.camera.width`, `vision.camera.height` | integer > 0 | Capture size in pixels |
| `vision.camera.fps` | > 0 | Capture rate |
| `vision.camera.exposure` | integer 1…10000 | Manual `exposure_time_absolute` |
| `vision.preroll_s` | > 0 | Pre-roll ring length before a detection |
| `vision.lookback_s` | > 0 | Window start relative to `window_end_ts` |
| `vision.fast_after_s` | ≥ 0 | Live analysis past `window_end_ts` |
| `vision.fast_budget_s` | > 0 | Stage-1 verifier budget; must be < `dispatch.verification_deadline_s` |
| `vision.idle_timeout_s` | > 0 | End an episode after this long without a detection |
| `vision.stale_after_s` | > 0 | Age at which the last camera frame makes the source unhealthy |
| `vision.analysis_fps` | > 0 | Analyzed-frame grid |
| `vision.queue_max` | integer > 0 | Bounded analysis queue; oldest frame dropped when full |
| `vision.imu_buffer_s` | > 0 | Per-side ACC buffer used by the episode report |
| `vision.evidence.enabled` | bool | Write clips, keyframes and traces |
| `vision.evidence.base_path` | path | Root of the evidence tree |
| `vision.evidence.clip_s` | > 0 | Clip length centred on the peak-severity contact |
| `vision.evidence.retention_days` | integer ≥ 0 | Day folders kept; `0` keeps only today |
| `vision.tracker.*` | numbers | Tracker, subject and skeleton thresholds (see `config/board.yaml`) |
| `vision.analysis.*` | numbers | Contact, hand-shape, oscillation and severity thresholds (see `config/board.yaml`) |

A disabled actuator kind is never imported, scanned or started.

## Logs

Both logs are grouped by local date under `log.base_path`:

```text
<log.base_path>/<YYYY-MM-DD>/system.log     # Python logging output
<log.base_path>/<YYYY-MM-DD>/events.jsonl   # one JSON object per line
<log.base_path>/<YYYY-MM-DD>/vision/reports.jsonl   # vision episodes only
```

An event record is written for every tag change and every actuation decision:

```json
{
  "timestamp": "2026-11-06T10:30:00.123456+01:00",
  "window_end_ts": 12345.678,
  "tag": "DANGEROUS",
  "tag_changed": true,
  "decision": {
    "verdict": "CONFIRMED",
    "reason": "CONFIRMED",
    "actuated": true,
    "actuations": [{"actuator_id": "meta_D3:E7:79:45:1F:E1", "params": {"duty": 100, "duration": 900}}],
    "severity": "NON_DANGEROUS",
    "assessment": {
      "verdict": "CONFIRMED",
      "severity": "NON_DANGEROUS",
      "features": {"contact_count": 2.0, "peak_closing_speed_fwps": 4.1},
      "episode_id": "2026-11-06T10-29-58Z-7f3a"
    }
  }
}
```

| Field | Meaning |
|---|---|
| `timestamp` | Local time the record was produced (ISO 8601) |
| `window_end_ts` | Monotonic timestamp (s) of the last row of the classified window |
| `tag` | IMU tag name |
| `tag_changed` | Whether the tag differs from the previous detection |
| `decision` | `null` when the detection was not evaluated (non-stereotypy tag or cooldown) |
| `decision.verdict` | Verdict used: `CONFIRMED`, `REJECTED` or `UNDECIDED` (also for timeout and error) |
| `decision.reason` | `CONFIRMED`, `UNDECIDED`, `TIMEOUT`, `VERIFIER_ERROR` or `REJECTED` |
| `decision.actuated` | Whether at least one actuator was triggered |
| `decision.actuations` | Actuator ids and parameters sent |
| `decision.severity` | Tag that drove the policy: the vision severity for a confirmed assessment that carries one, else the IMU tag; `null` when nothing actuated |
| `decision.assessment` | Stage-1 vision assessment (`verdict`, `severity`, `features`, `episode_id`); `null` on timeout, verifier error or when there is no decision |

## Tests

| Suite | Command | Runs on | Prerequisites |
|---|---|---|---|
| Routine | `uv run pytest` | Any host, including x86 CI | `uv sync --extra dev`; no hardware, no `board` extra, no native libraries |
| Target | `uv run pytest -m target` | KV260 (aarch64) only | `uv sync --extra dev --extra board --extra vision --extra camera`; the golden fixture `tests/target/fixtures/classifier_golden.npz`; the labeled session tree in `CPSA_LABELED_SESSIONS` |

The routine suite contains unit tests (`tests/unit`: domain, IMU, vision),
application tests with fakes (`tests/application`), contract tests for adapters
with vendor libraries stubbed (`tests/contract`) and architecture tests
(`tests/architecture`: layer rule and repository hygiene). Shared fakes for every
port are in `tests/fakes`. Vision evidence and camera contract tests skip without
OpenCV or `linuxpy`; the ONNX contract test skips unless `CPSA_POSE_ONNX` points
at an exported model.

The target suite replays recorded IMU rows through the native libraries and
requires features and tags identical to the baseline implementation. The fixture
must be captured on a KV260 at the baseline commit `fcf3ec9`; the procedure and
the current status are in `tests/target/fixtures/README.md`. Until it is captured
the target test is skipped. The labeled vision sessions test
(`tests/target/test_vision_sessions.py`) skips until `CPSA_LABELED_SESSIONS`
points at a session tree with its `labels.jsonl`, as long as AC-007 and AC-008
remain blocked.

Static checks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

mypy is strict for `domain`, `imu`, `ports`, `application`, `vision` and
`recorder`.

Hardware adapters are covered by contract tests only; check them on the board:
both BlueCoins discovered and windows classified after window #2 in `system.log`;
one BlueCoin off gives exit code 2 naming it; a stereotypy triggers the MetaMotion
(and speaker/LED when enabled) with an `events.jsonl` record showing verdict
`UNDECIDED`, reason `UNDECIDED`, `actuated: true`; `kill -TERM <pid>` stops the
process with exit code 0. With `vision.enabled: true`, also check that the camera
opens, the episode leaves a report, clip/keyframe and trace, and that
`cpsa-replay-vision --trace` reproduces the same verdicts.

## Native libraries

The feature extractor and classifier are opaque MATLAB Coder libraries built for
aarch64. They cannot be rebuilt from this repository. Provenance, sha256 hashes,
C signatures and licence terms are in
[`src/cpsa/adapters/native/lib/PROVENANCE.md`](src/cpsa/adapters/native/lib/PROVENANCE.md);
`tests/contract/test_provenance.py` checks the hashes.

## License

MIT for the Python code. The native libraries are generated under a MATLAB
Academic License (see `PROVENANCE.md`).
