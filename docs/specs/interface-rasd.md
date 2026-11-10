# CPSA_2026 shared interfaces: Requirements Analysis and Specification

**Status:** READY (the frontend fields are reserved; Q-items are nonblocking)
**Type:** Interface specification, derived from `docs/specs/system-rasd.md`.
**Scope:**
1. The client-server protocol between the Kria and the browser.
2. The supervisor mode lifecycle.
3. The perception service contract shared by the lab and medical modes.
4. The 3D asset manifest format.

Frontend visuals are out of scope; the fields they need are reserved here.
**Parent requirements:** system-rasd FR-101..103, FR-201..205, FR-301..306, FR-401..403, FR-501..506, FR-601..602, FR-701, NFR-01..07.
**Baseline evidence:**
- `src/cpsa/vision/types.py`: `TrackedFrame`, `PoseFrame`, `KeypointLayout`, `FramingState`.
- `src/cpsa/domain/model.py`: `Tag`, `Verdict`, `VisionAssessment`, `DecisionRecord`, `HandShape`, `FaceRegion`.
- `src/cpsa/ports.py`: `VideoSource`, `PoseEstimator`, `Gui`.
- `src/cpsa/application/vision_verifier.py`: episode-mode perception.

## Purpose and boundaries
**Goal:** Fix the contracts that let the backend (supervisor, perception, lab, medical) and the frontend be built in parallel, and tested independently, before Friday 2026-11-13.

**Actors:**
- The supervisor (server).
- Display clients: the lab view and the medical dashboard, in a browser.
- The operator, who sends commands from a client.
- The lab and medical runtimes, which consume perception.
- The asset authors, who write the manifests.

**Non-goals:** visual design, authentication, multiple users, internet access, binary or video encoding beyond the optional preview (IF-512).

## Design decisions (stated here because they shape the contract)
- **D1. State model:** per-topic *full state* messages plus discrete *event* messages.
  - Each state message fully replaces the client's copy of that topic.
  - There are no JSON patches.
  - Reason: the topics are small (under ~4 KB). A full state per message removes patch-ordering bugs and makes reconnects trivial.
- **D2. Kria is authoritative:** clients only render the state they receive and send commands. Gesture interpretation and the scene state live on the Kria (system FR-302).
- **D3. Machine-readable schema:** the protocol is defined by JSON Schema files in the repo. The server validates its outgoing messages against them in tests.

## Requirements

### A. Transport and envelope
- **IF-101** (MUST): The server SHALL serve HTTP and a WebSocket endpoint on one configurable port. The WebSocket path is `/ws`; the static UI is at `/`; the assets are under `/assets/`; the health endpoint is `GET /health`.
- **IF-102** (MUST): Every WebSocket message SHALL be a UTF-8 JSON object with this envelope:

  | Field | Type | Meaning |
  |---|---|---|
  | `v` | int | Protocol major version, currently 1 |
  | `type` | string | Dotted message type |
  | `seq` | int | Per-connection, strictly increasing, starting at 1 |
  | `t` | float | Kria `CLOCK_MONOTONIC` seconds |
  | `wall` | string | ISO-8601 UTC |
  | `data` | object | The payload |

- **IF-103** (MUST): Compatible additions (new optional fields, new types) SHALL NOT change `v`. Removals, renames or a change of meaning SHALL increment `v`.
  - Clients SHALL ignore unknown types and unknown fields.
  - Fields prefixed with `x_` are reserved for frontend experiments and SHALL be ignored by the server.
- **IF-104** (MUST): A WebSocket message SHALL be at most 64 KB. Server-to-client message rates per type are bounded by IF-6xx.

### B. Connection lifecycle
- **IF-201** (MUST): On connect, the server SHALL first send `hello` with:
  - `server_version`;
  - `protocol_version`;
  - `capabilities` (the list of topic names);
  - `modes` (the available modes);
  - `models` (the lab model ids and display names).
- **IF-202** (MUST): Immediately after `hello`, the server SHALL send a full snapshot: the current state message of every topic that applies to the current mode (`status`, plus `lab.scene` and `lab.hands` or `medical.state`). After that it SHALL send only state updates and events.
- **IF-203** (MUST): Clients MAY send `ping`. The server SHALL answer `pong` with the same `data.id`. The server SHALL close connections that stay idle with no reads for longer than a configurable timeout.
- **IF-204** (MUST): Losing a connection SHALL NOT change any server state. A reconnecting client gets IF-201 and IF-202 again.
- **IF-205** (MUST): The server SHALL support at least 4 concurrent clients. A slow client SHALL NOT delay the others or perception. Its state messages are coalesced (drop to the latest); events are queued up to a bound, then the client is disconnected.

### C. Supervisor topics and commands
- **IF-301** (MUST): `status` state (1 Hz, and on any change). It carries:
  - `mode` (`idle | lab | medical`);
  - `mode_state` (`stopped | starting | running | stopping | failed`);
  - `perception` with:
    - `fps` and `dpu_ms` (means over the last second);
    - `camera` (`ok | missing | error`);
    - `model` (the xmodel or onnx id);
  - `imu` (`n/a | searching | connected | lost`; medical only);
  - `errors` (a list of active error codes);
  - `uptime_s`.
- **IF-302** (MUST): The command `mode.set {mode}`.
  - The server SHALL answer with the event `mode.progress`, with `step` in `stopping | releasing | starting | ready | failed`, in order, with `failed` carrying `error`.
  - Only one switch can be in progress at a time. A second `mode.set` during a switch SHALL get `error {code: "busy"}`.
  - Setting the mode that is already active SHALL return `ready` immediately.
- **IF-303** (MUST): A failed start SHALL leave `mode = idle`, `mode_state = failed`, and the error in `status.errors`, until the next successful `mode.set`.
- **IF-304** (MUST): Error codes form a closed, documented list (extended only by protocol additions):
  `camera_missing`, `camera_error`, `model_missing`, `dpu_error`, `imu_not_found`, `asset_invalid`, `busy`, `bad_command`, `internal`.
  - The `error` event carries `code`, `message`, and `ref_seq` when it answers a command.
- **IF-305** (MUST): The command `preview.set {on: bool}` toggles the optional preview (IF-512). It applies in both modes.

### D. Lab topics and commands
- **IF-401** (MUST): `lab.hands` state, at every perception frame up to 30 Hz. It carries:
  - `user_present` (bool);
  - `framing` (a `FramingState` value);
  - `hands`: a list of at most 2 entries, each with:
    - `side` (`left | right`);
    - `cursor`: `[x, y]` in interaction space, normalized to [-1, 1] and anchored to the shoulders;
    - `image`: `[u, v]` normalized to [0, 1] in camera space;
    - `shape` (a `HandShape` value);
    - `pinch` (bool);
    - `pinch_strength` (0..1);
    - `confidence` (0..1);
    - `tracked` (bool);
  - optional `skeleton`: keypoint `[u, v, conf]` triples in the layout order, with the layout id.
- **IF-402** (MUST): `lab.scene` state, at most 30 Hz and on change. It carries:
  - `model_id`;
  - `rotation` (a quaternion `[x, y, z, w]`);
  - `scale` (a factor, clamped to the manifest's limits);
  - `explode` (0..1);
  - `selected_part` (a part id or null);
  - `grab`: null or `{part_id, offset:[x,y,z], hand}`;
  - `part_offsets`: an object of part id to `[x,y,z]` in model space;
  - `attract` (bool: no user present).
- **IF-403** (MUST): Discrete `lab.gesture` events with `name`:
  `pinch_start`, `pinch_end`, `grab_start`, `grab_end`, `scale`, `rotate`, `explode`, `reassemble`, `swipe_next`, `swipe_prev`, `reset`, `select`.
  - Each carries `hand` or `hands` and, where it applies, `part_id`.
  - These events exist only for frontend feedback (effects, sounds). The state is in `lab.scene`.
- **IF-404** (MUST): Lab commands:
  - `lab.reset`;
  - `lab.model.set {model_id}` (operator override);
  - `lab.part.select {part_id | null}` (operator or debug).
  
  An invalid id gives `error {code: "bad_command"}`.
- **IF-405** (MUST): The scene state SHALL be reproducible: the same sequence of perception frames and commands produces the same sequence of `lab.scene` states.

### E. Medical topics
- **IF-451** (MUST): `medical.state` state (1 Hz, and on change). It carries:
  - `imu` (as IF-301);
  - `tag` (the current IMU `Tag`);
  - `episode`: null or `{episode_id, t_start, active}`;
  - `last_decision` (as IF-452, or null).
- **IF-452** (MUST): The `medical.decision` event, one per `DecisionRecord`. It carries:
  - `tag`;
  - `tag_changed`;
  - `verdict`;
  - `reason`;
  - `severity` (the effective severity);
  - `vision`: `{verdict, severity, episode_id, features}` or null;
  - `actuations`: a list of `{actuator_id, kind}`.

  The field values come directly from the existing domain types.
- **IF-453** (MUST): The `medical.episode` event when an episode report changes or ends. It carries:
  - `episode_id`;
  - `active`;
  - `contact_count`;
  - `repetition_hz`;
  - `peak_severity`;
  - `labels`;
  - `contacts`: a list of `{side, region, t_start, duration_s, hand_shape}`.
- **IF-454** (SHOULD): During an active episode, the server SHALL send `medical.hands` states with the same shape as `lab.hands`, at up to 15 Hz, for the skeleton overlay.

### F. Rates and preview
- **IF-601** (MUST): Maximum rates: `lab.hands` and `lab.scene` 30 Hz each; `medical.hands` 15 Hz; `status` and `medical.state` 1 Hz plus on change. Events are sent as they occur.
- **IF-512** (SHOULD): The preview is served as MJPEG over HTTP at `/preview.mjpg` (at most 320×240, at most 15 fps), not over the WebSocket. When `preview.set` is off, the endpoint returns 404 or an empty stream.

### G. Supervisor lifecycle contract (in-process)
- **IF-701** (MUST): A mode SHALL be a component with `start(context)`, `stop(timeout)` and `health()`.
  - `context` gives it the perception service, a publisher for its topics, and its config section.
  - `stop` SHALL release all its subscriptions and devices within the timeout, or report failure.
- **IF-702** (MUST): The supervisor SHALL be the only owner of the camera and the DPU (the perception service). Modes SHALL access perception only through IF-8xx.
- **IF-703** (MUST): The state machine for each mode switch is `running → stopping → stopped → (devices released) → starting → running | failed`. The progress is published per IF-302.
- **IF-704** (MUST): The medical mode wraps the existing runtime (`build_runtime` and the dispatcher) without changing its behavior. Its `Gui` port implementation SHALL be the publisher for IF-451..454.

### H. Perception service contract (in-process, shared)
- **IF-801** (MUST): The perception service SHALL deliver `PerceptionFrame` objects. Each one carries:
  - `seq`, `t`;
  - the layout id;
  - `tracked`: a `TrackedFrame` (the existing type);
  - `hands`: per side, `{shape, pinch_strength, confidence}`, computed in the shared core;
  - `framing`;
  - `stats`: `{fps, dpu_ms}`.
- **IF-802** (MUST): The service SHALL offer a continuous subscription (`subscribe(callback, max_rate_hz)` / `unsubscribe`) for the lab, and episode access for the medical verifier.
  - The existing `VisionVerifier` behavior and its pre-roll, deadline and replay contracts are unchanged (vision-2d RASD FR-001..025).
- **IF-803** (MUST): Callbacks SHALL run off the capture thread. A slow subscriber SHALL get the latest frame only (frames are dropped and counted), and SHALL NOT slow perception.
- **IF-804** (MUST): The service SHALL accept a replay or synthetic source in place of the camera and DPU (system FR-701), with identical `PerceptionFrame` output for identical input.
- **IF-805** (MUST): Pinch and hand-shape computation SHALL live in the shared pure core, not in the lab, so the medical dashboard can reuse it.

### I. Asset manifest
- **IF-901** (MUST): Each model is a folder `web/assets/models/<model_id>/` containing a single `model.glb` and a `manifest.json`. The manifest fields:

  | Field | Meaning |
  |---|---|
  | `manifest_version` | 1 |
  | `id` | model id |
  | `name` | display name |
  | `description` | text |
  | `file` | `"model.glb"` |
  | `units` | `"m"` |
  | `default` | `{scale, rotation: [x,y,z,w], position: [x,y,z]}` |
  | `scale_limits` | `[min, max]` |
  | `parts` | list, see IF-902 |
  | `license` | `{name, url}` |
  | `source` | `{author, url}` |
  | `attribution` | text |

- **IF-902** (MUST): Each entry in `parts` carries:
  - `id`;
  - `node`: the glTF node name;
  - `name`;
  - `description`;
  - `explode`: `{dir: [x,y,z] unit, distance}` in model units;
  - `pick`: `{center: [x,y,z], radius}` in model units.
  
  A model SHALL have at least 3 parts.
- **IF-903** (MUST): A validator SHALL check:
  - the manifest against its JSON Schema;
  - that every `node` exists in `model.glb`;
  - that part ids are unique;
  - that the license fields are present;
  - that the `.glb` file is at most a configured size (default 20 MB).
  
  An invalid model is excluded, `asset_invalid` is reported, and the other models still load.
- **IF-904** (MUST): The server SHALL read only the manifests (no geometry). `hello.models` SHALL be generated from the valid manifests.

### J. Specification artefacts
- **IF-951** (MUST): The protocol SHALL be documented in `docs/protocol/` as a human-readable `README.md` plus JSON Schema files for the envelope, every message type, and the manifest.
- **IF-952** (MUST): A reference mock server (no camera, no DPU) SHALL emit schema-valid traffic from a synthetic or recorded stream, so the frontend can be developed against it.

### Nonfunctional
- **NFR-I1** (MUST): JSON encoding and validation SHALL NOT reduce lab perception below 30 fps on the KV260.
- **NFR-I2** (MUST): The protocol and supervisor code SHALL be testable on a PC with no hardware.

## Acceptance criteria

**Transport and connection**
- **AC-I01** [IF-101, IF-102, IF-951]: Every message the server emits in tests validates against the envelope and type schemas. `GET /health` returns 200 with the mode.
- **AC-I02** [IF-103]: A client fixture that ignores unknown types and fields keeps working when the server adds a new optional field. Messages containing `x_` fields are accepted.
- **AC-I03** [IF-201, IF-202, IF-204]: A client connecting mid-session receives `hello`, then the full snapshot, then updates. After a disconnect and reconnect, it receives a fresh snapshot equal to the current state.
- **AC-I04** [IF-203, IF-205]:
  - `ping` gets a matching `pong`.
  - With 4 clients, one of which stalls, the others keep the 30 Hz `lab.hands` rate.
  - The stalled client gets the latest states only, and is disconnected after the event bound.

**Supervisor**
- **AC-I05** [IF-301, IF-302, IF-303, IF-703]:
  - `mode.set` produces the ordered progress steps.
  - A concurrent `mode.set` gets `busy`.
  - A failed start (the camera removed in a test fake) ends in `idle/failed` with `camera_missing`.
  - Setting the active mode returns `ready` at once.
- **AC-I06** [IF-304]: Every error emitted in tests uses a code from the closed list. Error answers to a command carry `ref_seq`.
- **AC-I07** [IF-701, IF-702]: After `stop`, a mode holds no subscriptions or devices, verified with fakes. Only the supervisor opens the camera and the DPU.

**Lab**
- **AC-I08** [IF-401, IF-402, IF-403, IF-405]:
  - A synthetic frame sequence yields the expected `lab.hands` and `lab.scene` states and `lab.gesture` events.
  - Running it twice gives identical outputs.
- **AC-I09** [IF-404]: The lab commands change the scene as specified. An unknown `model_id` or `part_id` gives `bad_command`.

**Medical**
- **AC-I10** [IF-451, IF-452, IF-453, IF-454, IF-704]:
  - A replayed medical session produces `medical.state`, `medical.decision` and `medical.episode` messages that match the runtime's `DecisionRecord` and `EpisodeReport` values.
  - The existing medical tests stay green.

**Rates and preview**
- **AC-I11** [IF-601, IF-512, NFR-I1]:
  - In a timing test, the measured rates do not exceed the limits.
  - The preview toggles on and off.
  - On the board, lab perception stays at or above 30 fps with 2 clients connected and the preview on.

**Perception service**
- **AC-I12** [IF-801, IF-802, IF-803, IF-804, IF-805]:
  - The continuous subscription delivers `PerceptionFrame` objects.
  - A slow subscriber drops frames without slowing the producer.
  - Replay and synthetic sources give identical frames for identical inputs.
  - Pinch and hand shape come from the shared core.

**Assets**
- **AC-I13** [IF-901, IF-902, IF-903, IF-904]:
  - The 3 real models validate.
  - A corrupted manifest (a missing node, a duplicate id, no license) is rejected with `asset_invalid` while the others load.
  - `hello.models` lists only the valid models.

**Mock and hardware**
- **AC-I14** [IF-952, NFR-I2]: The mock server runs on a PC, emits schema-valid traffic, and supports every command.

## Assumptions and open decisions
**Assumptions:**
- Clients are trusted (booth LAN only); there is no auth.
- 30 Hz is enough for smooth rendering, with interpolation on the client.
- The glTF node names in the chosen models are stable.

**Open decisions:**
- **Q-I1** [IF-401]: The exact interaction-space mapping (shoulder-anchored box size and offset). It is tuned in the frontend step; the field semantics stay fixed.
- **Q-I2** [IF-403]: The final gesture names, confirmed in the frontend step. Additions are compatible.
- **Q-I3** [IF-512]: Whether to keep the MJPEG preview at all. It is optional and does not block anything.

## Readiness assessment
**Status:** READY. The frontend fields are reserved (`x_` fields, gesture events, compatible additions), so the frontend step cannot break v1.

**Blockers:** None.

**Nonblocking risks:**
- The 30 Hz full-state messages on the A53 CPU need NFR-I1 checked on the board.
- glTF node naming depends on the asset sourcing.
