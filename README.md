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
   actuate (fail-open). The shipped verifier is a stub that always answers
   `UNDECIDED`, so actuation currently depends on the IMU tag alone.
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
| `cpsa.ports` | `typing.Protocol` contracts between application and adapters | `domain`, numpy |
| `cpsa.application` | Configuration, classification stage, slot, dispatcher, runtime wiring | `domain`, `imu`, `ports`, numpy, yaml |
| `cpsa.adapters` | Native libraries, BLE sensors, actuators, logs, verifier stub, null GUI | `domain`, `ports`, any library |
| `cpsa.board` | `cpsa-board` entry point: builds the real adapters and runs | everything |

Only `cpsa.adapters` and `cpsa.board` may use `ctypes` or the hardware libraries
(`blue_st_sdk`, `bluepy`, `mbientlab`, `flux_led`, `playsound`, `dbus_fast`).

### Ports

| Port | Implementations | Contract |
|---|---|---|
| `SensorSource` | `adapters.bluecoin.source.BlueCoinSource` | `discover(timeout_s) -> set[str]`, `start(on_sample)`, `stop()` |
| `FeatureExtractor` | `adapters.native.classifier.NativeFeatureExtractor` | `(150, 20)` preprocessed window → `(18,)` features |
| `Classifier` | `adapters.native.classifier.NativeClassifier` | features → `Tag`; raises `ClassificationError` on an unknown code |
| `Verifier` | `adapters.verifier_stub.StubVerifier` | detection → `Verdict`; may block or raise |
| `Actuator` | `adapters.actuators.{metamotion,speaker,led_strip}` | `id`, `kind`, `execute(params)`, `stop()`; never raises for a disconnected device |
| `EventSink` | `adapters.event_log.JsonlEventSink` | `write(DecisionRecord)` |
| `Gui` | `adapters.gui_null.NullGui` | `start()`, `publish(DecisionRecord)`, `stop()` |
| `Clock` | `adapters.clock.SystemClock` | `monotonic()`, `now()` |

### Replacing a component

- **Vision verifier:** implement `cpsa.ports.Verifier` in a new adapter module and
  construct it instead of `StubVerifier()` in `board_adapters()` in
  `src/cpsa/board/__main__.py`, its single construction site. The dispatcher
  already enforces the deadline and fail-open behavior; a verifier may block or
  raise. A verifier that hangs keeps the single verification worker busy, so later
  verifications time out and fail open.
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

Stop with `Ctrl+C` or `kill -TERM <pid>`: the GUI, dispatcher, sensors and
actuators are stopped in this order and the process exits with code 0.

| Exit code | Meaning |
|---|---|
| 0 | Orderly stop after SIGINT or SIGTERM |
| 1 | Unexpected error (logged with traceback) |
| 2 | Startup failure: invalid configuration (the message names the key), `board` extra missing, a BlueCoin not found after the configured retries (the log names it), or the native libraries cannot be loaded (for example on x86) |

With no actuator enabled or found, detection, verification and logging continue.

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

A disabled actuator kind is never imported, scanned or started.

## Logs

Both logs are grouped by local date under `log.base_path`:

```text
<log.base_path>/<YYYY-MM-DD>/system.log     # Python logging output
<log.base_path>/<YYYY-MM-DD>/events.jsonl   # one JSON object per line
```

An event record is written for every tag change and every actuation decision:

```json
{
  "timestamp": "2026-11-06T10:30:00.123456+01:00",
  "window_end_ts": 12345.678,
  "tag": "DANGEROUS",
  "tag_changed": true,
  "decision": {
    "verdict": "UNDECIDED",
    "reason": "UNDECIDED",
    "actuated": true,
    "actuations": [{"actuator_id": "meta_D3:E7:79:45:1F:E1", "params": {"duty": 100, "duration": 900}}]
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

## Tests

| Suite | Command | Runs on | Prerequisites |
|---|---|---|---|
| Routine | `uv run pytest` | Any host, including x86 CI | `uv sync --extra dev`; no hardware, no `board` extra, no native libraries |
| Target | `uv run pytest -m target` | KV260 (aarch64) only | `uv sync --extra dev --extra board`; the golden fixture `tests/target/fixtures/classifier_golden.npz` |

The routine suite contains unit tests (`tests/unit`: domain, IMU), application
tests with fakes (`tests/application`), contract tests for adapters with vendor
libraries stubbed (`tests/contract`) and architecture tests (`tests/architecture`:
layer rule and repository hygiene). Shared fakes for every port are in
`tests/fakes`.

The target suite replays recorded IMU rows through the native libraries and
requires features and tags identical to the baseline implementation. The fixture
must be captured on a KV260 at the baseline commit `fcf3ec9`; the procedure and
the current status are in `tests/target/fixtures/README.md`. Until it is captured
the target test is skipped.

Static checks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

mypy is strict for `domain`, `imu`, `ports` and `application`.

Hardware adapters are covered by contract tests only; check them on the board:
both BlueCoins discovered and windows classified after window #2 in `system.log`;
one BlueCoin off gives exit code 2 naming it; a stereotypy triggers the MetaMotion
(and speaker/LED when enabled) with an `events.jsonl` record showing verdict
`UNDECIDED`, reason `UNDECIDED`, `actuated: true`; `kill -TERM <pid>` stops the
process with exit code 0.

## Native libraries

The feature extractor and classifier are opaque MATLAB Coder libraries built for
aarch64. They cannot be rebuilt from this repository. Provenance, sha256 hashes,
C signatures and licence terms are in
[`src/cpsa/adapters/native/lib/PROVENANCE.md`](src/cpsa/adapters/native/lib/PROVENANCE.md);
`tests/contract/test_provenance.py` checks the hashes.

## License

MIT for the Python code. The native libraries are generated under a MATLAB
Academic License (see `PROVENANCE.md`).
