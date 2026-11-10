# CPSA_2026 system: Requirements Analysis and Specification

**Status:** READY (open decisions are nonblocking)
**Type:** System-level specification. It covers the new holographic model lab and the restructuring the medical demo needs to coexist with it.
**Scope:** The whole repository: the shared perception code, the medical stereotypy demo, the holographic model lab, the supervisor and web layer, recording and replay, and the ML and Vitis toolchains. The visual design of the frontend is specified separately (next step).
**Baseline:** `main` after PR #3 (merge `e4d58ca`), plus branch `fcabecciaw/video-pipeline` at `23119c6`.
**Sources:**
- Stakeholder decisions on 2026-11-10:
  - The university open day is Friday 2026-11-13. Both demos are kept.
  - The model lab offers free exploration only. The models are a satellite, a heart and an engine. No robotic-arm content.
  - Seated user directly in front of the camera.
  - The Kria does all the processing and communicates with a web page over Ethernet.
  - One repository.
  - Modes switch from the web page.
  - One shared web UI that also shows the medical demo live.
  - Approved: a quick training run, LeakyReLU at 384×512, about 30 epochs.
- Repository evidence:
  - `README.md` "Architecture" (layer table);
  - `tests/architecture/test_layers.py`;
  - `src/cpsa/ports.py` (`Gui`, `VideoSource`, `PoseEstimator`, `Verifier`, `DetectionObserver`);
  - `src/cpsa/application/vision_verifier.py`;
  - `src/cpsa/board/__main__.py`;
  - `pyproject.toml` (scripts and extras);
  - `docs/specs/vision-2d-rasd.md`;
  - `docs/notes/vitis-compile.md`.

## Purpose and boundaries

**Goal:** One KV260 system runs two demos on the same perception core, chosen from a web page. Both must be presentable to high-school visitors at the open day.

**Actors:**

| Actor | Role |
|---|---|
| Visitor (seated student) | Uses gestures in the model lab, or is the subject of the medical demo |
| Booth operator (researcher) | Switches modes, watches status, restarts |
| Kria KV260 | Camera, DPU pose, all processing, web server |
| Display client | A laptop browser on the booth display, connected over Ethernet. It renders the UI and does no perception |
| BlueCoin IMUs and actuators | Medical mode only |
| Developer | Recording, replay, training, quantization |

**In scope:**
- the shared perception core;
- the medical runtime (existing) under the supervisor;
- the lab runtime: gestures and scene interaction;
- the supervisor with mode switching;
- the web server and the client-server protocol;
- the medical live dashboard;
- the 3D model assets and their metadata;
- deployment on the board;
- the repository layout and layer rules;
- the ML and Vitis toolchains as they exist.

**Non-goals:**
- the frontend's visual design (next step);
- timed games, scores and leaderboards;
- robotic-arm content;
- multiple simultaneous users;
- rendering on the Kria;
- internet connectivity;
- level-3 articulated tracking;
- running both demos at once.

## Current behavior and requested changes

**Current, verified in the repo:**
- `cpsa` is layered: `domain`, `imu`, `vision`, `ports`, `application`, `adapters`, then the composition roots `board` and `recorder`.
- The only entry points are `cpsa-board` (medical runtime), `cpsa-record`, `cpsa-replay-imu` and `cpsa-replay-vision`.
- Perception (camera to pose to tracker) is only reachable through the episode-driven `VisionVerifier`, which runs inference only during IMU-triggered episodes.
- The `Gui` port exists, but the board wires `NullGui`.
- There is no web server, no frontend and no supervisor.
- `ml/` (cpsa_ml) and `tools/vitis/` are separate toolchains.

**ADDED:**
- the supervisor service and mode switching;
- the web server, the client-server protocol and static frontend hosting;
- a continuous perception stream;
- gesture recognition;
- the lab scene interaction state;
- the 3D model assets and their part metadata;
- the medical dashboard data feed through the `Gui` port;
- the board service install.

**MODIFIED:**
- The medical runtime becomes startable and stoppable by the supervisor as well as by `cpsa-board`, which keeps working standalone.
- The perception core becomes usable in both an episode mode (medical) and a continuous mode (lab).

**REMOVED:** none.

## Requirements

### Functional: shared perception core
- **FR-101** (MUST): The system SHALL provide one perception pipeline: RGB capture, DPU pose, subject selection, 2D tracking and hand shape. Both demos SHALL use it, with the same model and the same code.
- **FR-102** (MUST): The perception pipeline SHALL support an episode mode (pre-roll plus triggered inference, as today) and a continuous mode (inference on every selected frame while the mode is active).
- **FR-103** (MUST): At any time, the camera and the DPU SHALL be held by at most one active mode.

### Functional: supervisor and modes
- **FR-201** (MUST): A supervisor service on the Kria SHALL start automatically at boot and run exactly one mode at a time: `lab`, `medical` or `idle`.
- **FR-202** (MUST): WHEN a mode change is requested from the web UI, the supervisor SHALL stop the current mode, release the camera and DPU, start the requested mode, and report progress and the result to all connected clients.
- **FR-203** (MUST): WHEN a mode fails to start (camera missing, model missing, IMUs not found), the supervisor SHALL report the failure reason to clients and stay operable: idle or the previous mode, never crashed.
- **FR-204** (MUST): The supervisor SHALL expose its health to clients: current mode, perception fps, DPU latency, camera status, IMU status (medical) and errors.
- **FR-205** (SHOULD): The default mode at boot SHALL be configurable.

### Functional: model lab
- **FR-301** (MUST): The lab SHALL recognize these gestures of the selected (seated) user from the pose keypoints:
  - one-hand pinch and drag;
  - two-hand pinch spread or close;
  - both palms pushed apart (explode);
  - fists brought together (reassemble);
  - pinch on a part (select or grab);
  - swipe (next or previous model);
  - fist hold (reset).
- **FR-302** (MUST): The lab SHALL maintain the scene interaction state on the Kria:
  - the current model;
  - model rotation and scale;
  - the explode level;
  - the selected part;
  - per-part offsets when a part is grabbed;
  - hand cursors.
- **FR-303** (MUST): The lab SHALL offer the satellite, heart and engine models. Each one is decomposable into named parts with descriptive information.
- **FR-304** (MUST): WHEN finger keypoints are unreliable (low confidence), interactions SHALL degrade to open/closed-hand and wrist-based equivalents instead of stalling.
- **FR-305** (MUST): WHEN no user is detected for a configurable time, the scene SHALL return to an attract state: an idle animation and the default model.
- **FR-306** (MUST): Gesture recognition SHALL apply hysteresis and smoothing, so that a held pose does not toggle grab and release or select and deselect.

### Functional: medical dashboard
- **FR-401** (MUST): In medical mode, the existing runtime behavior SHALL be unchanged: IMU detection, vision verifier, dispatcher, actuators and logs (`docs/specs/vision-2d-rasd.md` FR-001 to FR-025).
- **FR-402** (MUST): The medical runtime SHALL publish to the web UI, through the `Gui` port:
  - live status: IMU connection, current tag, episode active;
  - decision records: verdict, severity, actuations;
  - episode summaries.
- **FR-403** (SHOULD): During an episode, the dashboard SHALL show the tracked skeleton and the hand-to-face contacts.

### Functional: web layer and communication
- **FR-501** (MUST): The Kria SHALL serve the web frontend (static files and model assets) over HTTP. A display client SHALL need only a browser and the Kria's address.
- **FR-502** (MUST): The Kria SHALL push state to clients over a bidirectional real-time channel (WebSocket). Message families:
  - `hello` and version handshake;
  - supervisor `status` and health;
  - `mode` change progress;
  - lab `scene` state;
  - lab `hands` (cursors, gesture events);
  - `medical` status, decisions and episodes;
  - `error`.

  Clients send only `command`s: mode change, reset, and operator actions.
- **FR-503** (MUST): Every message SHALL carry a protocol version, a type, a monotonic sequence number and a Kria timestamp. Clients SHALL ignore unknown types.
- **FR-504** (MUST): Clients SHALL be stateless renderers. WHEN a client connects or reconnects, it SHALL receive a full state snapshot, then incremental updates. Losing the client SHALL NOT affect the Kria's state.
- **FR-505** (SHOULD): The Kria SHALL offer an optional low-resolution camera preview stream with skeleton overlay (mirror effect), which can be switched off.
- **FR-506** (MUST): The protocol SHALL be specified in a versioned document in the repo, and validated by tests on the server side.

### Functional: assets
- **FR-601** (MUST): Each 3D model SHALL ship with a manifest:
  - id and display name;
  - per part: id, name, description, the explode direction or offset, and a pick volume;
  - default scale and orientation;
  - license and source attribution.

  The Kria SHALL use only the manifest. The geometry is used only by the browser.
- **FR-602** (MUST): The model geometry and the manifest SHALL be consistent: every manifest part maps to a node in the model file. A check SHALL detect any mismatch.

### Functional: development and replay
- **FR-701** (MUST): The lab SHALL run from recorded sessions or synthetic keypoint streams, with no camera or DPU, so that the frontend and gestures can be developed and tested on a PC.
- **FR-702** (MUST): The medical replay tools (`cpsa-replay-imu`, `cpsa-replay-vision`) SHALL keep working.

### Nonfunctional
- **NFR-01** (MUST) **Separation of concerns:**
  - Pure logic (perception analysis, gestures, scene state, protocol encoding) SHALL be free of I/O and hardware imports, and testable without hardware.
  - Hardware, network and filesystem SHALL live in adapters.
  - Composition roots SHALL be the only places that wire them together.
  - These rules SHALL be enforced by `tests/architecture/test_layers.py`, extended for the new packages.
- **NFR-02** (MUST) **Isolation of the two demos:**
  - The lab code SHALL NOT import the medical runtime or the IMU code, and the medical code SHALL NOT import the lab code.
  - Both MAY import the shared core.
  - The medical demo SHALL install and run without the lab's dependencies, and the lab without the BLE and actuator dependencies.
- **NFR-03** (MUST) **Lab performance:**
  - perception at least 30 fps;
  - gesture to rendered response within the budget of Q-01;
  - no visible stutter on the display client over Ethernet.
- **NFR-04** (MUST) **Platform:** the board code SHALL run on the KV260 with Ubuntu 22.04, Python 3.10, VART 3.5 and the DPU fingerprint `0x101000016010407`. Display clients SHALL work in current Chrome and Firefox without plugins.
- **NFR-05** (MUST) **Robustness for a full open day:**
  - no restart needed in normal use;
  - automatic recovery from a browser disconnect or a camera re-plug;
  - the supervisor restarts on a crash (service manager).
- **NFR-06** (MUST) **Mode switch** completes within the time of Q-02.
- **NFR-07** (MUST) **Local only:** the system SHALL work with no internet connection. Web assets SHALL be served locally, with no CDN at runtime.
- **NFR-08** (SHOULD) **Asset licensing:** every model asset SHALL have a license that permits public demonstration, recorded in its manifest.

## Repository organization (stakeholder-requested characterization)

These are requirements on observable structure. `$plan` decides the exact module names inside each area.

**Areas:**

| Area | Responsibility | Notes |
|---|---|---|
| Shared core (`src/cpsa/domain`, `vision`, `ports`) | Pure perception analysis, types, contracts | Already exists. Gains the pure gesture and scene-interaction logic, or a sibling pure package. |
| Shared perception service | Camera to pose to tracker, in episode and continuous modes | Now embedded in the medical `VisionVerifier`; becomes reusable |
| Medical demo (`imu`, `application`, `board`) | IMU detection, dispatcher, vision verifier, actuators | Behavior unchanged |
| Model lab | Gesture interpretation, scene state, lab runtime | New |
| Supervisor and web | Mode lifecycle, HTTP and WebSocket server, protocol | New; the only boot entry point on the board |
| Adapters (`src/cpsa/adapters`) | Camera, DPU/ONNX, BLE, actuators, logs, evidence, recording, web transport | Gains the web transport adapter |
| Frontend (`web/`) | Static browser app: shared shell, lab view, medical dashboard; the 3D model assets and manifests | New; no build step needed on the board |
| Toolchains (`ml/`, `tools/vitis/`, `tools/calibration/`) | Dataset, training, quantization, compile | Already exist; unchanged |
| Docs (`docs/specs`, `docs/plans`, `docs/notes`), plus the protocol spec | Requirements, plans, notes | Gains the protocol document |

**Entry points:**
- the existing `cpsa-board`, `cpsa-record`, `cpsa-replay-imu` and `cpsa-replay-vision`;
- new: a supervisor entry point, which is the boot service;
- new: a lab development entry point that runs from a synthetic or recorded stream.

**Dependency extras:**
- the medical extras (`board`, `camera`) stay;
- the lab and web dependencies go in their own extra;
- `vision` stays shared.

## Interfaces and data

- **Client-server protocol:** WebSocket, JSON, versioned (FR-502 to FR-504). The exact schema goes in the protocol document (FR-506).
- **HTTP:** static frontend and assets, plus a health endpoint.
- **Network:** a direct Ethernet link from the laptop to the Kria (the current `10.42.0.x` setup), on a configurable port.
- **Configuration:** `config/board.yaml` gains supervisor, lab and web sections, validated like the existing ones.
- **Asset manifests:** per model, alongside its geometry (FR-601).

## Acceptance criteria

**Perception and supervisor**

- **AC-101** [FR-101, FR-102]: The same xmodel and perception code serve both modes. In lab mode, inference runs on every selected frame. In medical mode, it runs only during episodes (the existing AC-001 still holds).
- **AC-102** [FR-103, FR-201, FR-202]:
  - Given the lab running, when the operator switches to medical from the web UI, the lab stops, the camera and DPU are released, and medical starts.
  - Every client sees the progress and the final mode.
  - The reverse switch works the same way.
- **AC-103** [FR-203]: With no IMUs reachable, a switch to medical reports "IMU not found", and the supervisor stays responsive in idle or in the previous mode.
- **AC-104** [FR-204, FR-205]: The status shows the mode, fps, DPU latency and device states. The configured default mode starts at boot.

**Model lab**

- **AC-301** [FR-301, FR-306]:
  - Every gesture is recognized on synthetic keypoint sequences, plus recorded seated sequences once available.
  - A held pinch does not toggle.
  - Each gesture has a negative case: a similar but different motion does not trigger it.
- **AC-302** [FR-302]: Scene state transitions (rotate, scale, explode, reassemble, select, grab, next model, reset) follow the gesture events deterministically.
- **AC-303** [FR-303, FR-601, FR-602]: All three models load with their manifests, and the consistency check passes.
- **AC-304** [FR-304]: With the finger confidences forced low, the fallback interactions still rotate, scale and explode the model.
- **AC-305** [FR-305]: After the configured idle time with no user, the scene returns to the attract state.

**Medical dashboard**

- **AC-401** [FR-401, FR-702]: The existing medical tests and replay tools pass unchanged.
- **AC-402** [FR-402, FR-403]: During a replayed medical session, the dashboard feed delivers the IMU status, decisions and episode summaries, with the skeleton during episodes.

**Web layer**

- **AC-501** [FR-501, NFR-07]: With the internet off, a laptop on the Ethernet link opens the Kria address, the UI loads, and the models render.
- **AC-502** [FR-502, FR-503, FR-504, FR-506]:
  - A client that connects mid-session receives a snapshot, then updates.
  - Killing and reopening the browser resumes correctly.
  - Unknown message types are ignored.
  - The protocol document matches the server's tests.
- **AC-503** [FR-505]: The preview can be turned on and off, and it does not reduce lab perception below 30 fps.

**Development**

- **AC-701** [FR-701]: The lab and the frontend run end to end on a PC from a synthetic or recorded stream, with no camera or DPU.

**Nonfunctional**

- **AC-901** [NFR-01, NFR-02]:
  - The architecture test enforces the new layers and the lab/medical isolation.
  - The medical demo installs without the lab extra.
  - The lab installs without the BLE and actuator extras.
- **AC-902** [NFR-03, NFR-04]: On the KV260 with the trained model, the lab sustains at least 30 fps perception, and the measured gesture-to-render latency meets Q-01.
- **AC-903** [NFR-05]:
  - A 2-hour soak runs with periodic browser reconnects and one camera re-plug, with no manual restart.
  - A killed supervisor is restarted by the service manager.
- **AC-904** [NFR-06]: The mode switch time meets Q-02.
- **AC-905** [NFR-08]: Every model manifest names its license and source.

## Assumptions and open decisions

**Assumptions:**
- One booth display, driven by a laptop browser on direct Ethernet.
- One seated user at 0.45–0.7 m.
- The trained model (quick run) is available by Thursday.
- Suitable satellite, heart and engine models exist with separable parts and permissive licenses. NASA 3D Resources is likely for the satellite; the others need a search.

**Open decisions:**

| ID | Requirement | Question | Proposal |
|---|---|---|---|
| Q-01 | NFR-03 | Gesture-to-render latency budget | ≤ 100 ms |
| Q-02 | NFR-06 | Mode switch time budget | ≤ 10 s |
| Q-03 | FR-303 | Exact model files and part lists | Decided in the frontend step |
| Q-04 | FR-505 | Preview on by default in the lab? | On, small inset |
| Q-05 | FR-301 | Final gesture set and mapping | Revisited in the frontend step |

## Readiness assessment

**Status:** READY for `$plan` of the backend and system structure.

**Blockers:** None.

**Nonblocking risks:**
- **Friday deadline:** the full scope (supervisor, lab, dashboard, protocol, assets) is ambitious for 3 days. `$plan` should mark a minimum viable cut: supervisor with mode switch, lab with the core gestures, a basic dashboard.
- **Model availability:** finger-keypoint quality depends on the quick training run.
- **Assets:** finding licensed, separable models.
