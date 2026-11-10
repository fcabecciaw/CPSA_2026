# 2D Vision Pipeline: Requirements Analysis and Specification

**Status:** READY (open decisions are nonblocking)
**Type:** New capability, plus a behavioral change to the verification path
**Scope:** The on-board vision analysis that the IMU detections trigger: pose, 2D tracking, features, the two-stage verdict and report, and evidence. The GUI is out of scope.
**Baseline:** branch `fcabecciaw/video-pipeline` at base `58903e3` plus the uncommitted vision toolchain work.
**Sources:**
- Stakeholder decisions (2026-11-09):
  - 2D only.
  - Level 1 point-kinematics tracker plus level 2 items.
  - Modular, so level 3 (articulated model plus IMU fusion) can be added later.
  - Level 3 elements enter only where strictly needed, for example Kalman smoothing.
  - Scope adds hand shape, other stereotypies and evidence clips. Object-in-hand suppression is excluded.
  - Two-stage timing.
  - Re-grade in both directions.
- Code: `src/cpsa/ports.py` (`Verifier`), `src/cpsa/application/dispatcher.py`, `src/cpsa/domain/{model,gating,policy}.py`, `src/cpsa/vision/tracking.py`, `config/board.yaml` (`verification_deadline_s`).

## Purpose and boundaries

**Goal:** Use the camera to check, grade and describe the stereotypy episodes the IMU detects. "Dangerous" means a hand moving toward the face with self-harm intent.

**Actors:**
- Monitored subject (seated, single person).
- IMU classifier (BlueCoin).
- Dispatcher and actuators.
- Caregiver or operator, through the logs and evidence.
- Developer, through replay of recorded sessions.

**In scope:**
- Analysis triggered by the IMU, with a RAM pre-roll.
- One half-body pose model on the DPU.
- 2D per-keypoint tracking with smoothing.
- Per-person bone lengths, plausibility checks and the forearm foreshortening hint.
- Kinematic and contact features.
- Hand shape.
- Detection of rocking and hand flapping.
- Fast verdict with re-grade, then the episode report.
- Evidence clips.
- Event logging.
- Replay on recorded sessions.

**Non-goals:**
- Use of depth.
- Camera calibration as a prerequisite.
- The articulated kinematic model and IMU state fusion (level 3).
- Object-in-hand suppression.
- Vision starting an episode without an IMU trigger.
- The GUI.
- Tracking more than one person.

## Current behavior and requested changes

**Current:**
- The `Dispatcher` sends each stereotypy `Detection` to `Verifier.verify(detection) -> Verdict` with a deadline (`verification_deadline_s: 1.0`, a placeholder).
- On timeout or error it fails open.
- Only `REJECTED` blocks actuation.
- Actuation strength comes from the IMU `Tag` alone (`policy.actuations_for`).
- `StubVerifier` always returns `UNDECIDED`.
- The existing `CausalTracker` is a 3D EKF that needs camera intrinsics.

**ADDED:**
- Vision analysis runtime, pose inference, 2D tracker, features, hand shape, episode report, stereotypy labels, evidence clips.

**MODIFIED:**
- The verification result carries a vision severity and features, as well as the `Verdict`.
- Actuation severity follows the vision severity when the verdict is `CONFIRMED`.
- Follow-up actuations within an episode follow the episode state.
- Event-log records include the vision fields.
- **Compatibility:**
  - A verifier that returns `UNDECIDED` (the stub) behaves exactly as today.
  - Fail-open on timeout or error is kept.

**REMOVED from the required path (code stays as level 3 groundwork):**
- Depth lifting.
- The depth occluder test.
- 3D contact distances in metres.
- Calibration as a dependency.

## Requirements

### Functional

**Trigger and capture**

- **FR-001** (MUST): WHILE the board runs, the system SHALL keep the most recent camera frames in a RAM pre-roll buffer of configurable length. It SHALL NOT run pose inference outside an active episode.
- **FR-002** (MUST): WHEN the dispatcher evaluates a stereotypy detection, the system SHALL start or extend an episode and analyze the pre-roll frames plus live frames.
- **FR-003** (MUST): WHEN no stereotypy detection and no hand-to-face or stereotypy activity occurs for a configurable idle time, the system SHALL end the episode and stop inference.

**Pose**

- **FR-004** (MUST): The system SHALL estimate the subject's keypoints with confidences using one deep-learning pose model. The model is fine-tuned on half-body images. It covers:
  - face anchors: eyes, nose, ears, brows, mouth, jaw, chin;
  - shoulders, elbows, wrists;
  - 21 keypoints per hand, which is enough to classify hand shape.
  - (Amended: hips are not required. Rocking uses the shoulders. The `HALFBUST_62` schema has no hips.)
- **FR-005** (MUST): For each frame the system SHALL report a framing state: `full_body`, `half_body`, `face_not_visible`, `hands_not_visible` or `no_person`. Missing lower-body keypoints in a half-body frame SHALL NOT lower the person's confidence or the upper-body keypoint confidences.
- **FR-006** (MUST): WHEN more than one person is detected, the system SHALL analyze one subject, chosen by a rule set in configuration.

**Tracking**

- **FR-007** (MUST): The system SHALL produce, per tracked keypoint and frame:
  - a filtered 2D position, velocity and acceleration;
  - their uncertainties;
  - a status: measured, predicted, or lost.
- **FR-008** (MUST): WHEN a keypoint is missing or rejected, the tracker SHALL predict it for up to a configurable number of frames, then mark it lost.
- **FR-009** (MUST): The system SHALL estimate per-person reference bone lengths (shoulder width, upper arms, forearms) in face-width units from confident frames. It SHALL keep them for the session.
- **FR-010** (MUST): The system SHALL flag and exclude keypoint measurements that break the configured bone-length bounds.
- **FR-011** (MUST): The system SHALL report a forearm foreshortening ratio per side: the observed forearm length divided by the reference length.

**Features**

- **FR-012** (MUST): The system SHALL express lengths in face-widths and speeds in face-widths per second. The face-width source is set in configuration.
- **FR-013** (MUST): For each hand, the system SHALL compute:
  - hand-to-face distance;
  - closing speed;
  - peak closing speed before contact;
  - deceleration at contact;
  - approach angle;
  - time to contact;
  - occluded face-keypoint fraction.
- **FR-014** (MUST): The system SHALL detect hand-to-face contacts. For each contact it SHALL report:
  - side;
  - target zone (`FaceRegion`);
  - start and duration;
  - hand shape: `fist`, `point_pinch`, `open_palm` or `unknown`;
  - head displacement and velocity after contact (recoil).

**Fast verdict (stage 1)**

- **FR-015** (MUST): WITHIN the verification deadline, the system SHALL return one of:
  - `CONFIRMED`: a contact or an approach matching hand-to-face behavior;
  - `REJECTED`: subject framed, face and hands visible, and no hand within the contact threshold during the analysis window;
  - `UNDECIDED`: anything else, including camera or model failure.
- **FR-016** (MUST): WITH a `CONFIRMED` verdict, the system SHALL return a vision severity, `NON_DANGEROUS` or `DANGEROUS`. It is derived from target zone, hand shape, peak closing speed, deceleration and head recoil, using configurable rules.
- **FR-017** (MUST): WHEN the verdict is `CONFIRMED`, actuation SHALL use the vision severity. This can upgrade or downgrade the IMU tag.
  - WHEN the verdict is `UNDECIDED`, actuation SHALL use the IMU tag.
  - WHEN the verdict is `REJECTED`, there SHALL be no actuation.

**Episode report (stage 2)**

- **FR-018** (MUST): During and at the end of an episode, the system SHALL produce an episode report:
  - start and end;
  - contact count and repetition frequency;
  - variability of the interval between contacts;
  - amplitude or speed trend;
  - peak severity;
  - stereotypy labels from {`hand_to_face`, `rocking`, `hand_flapping`}.
- **FR-019** (MUST): The system SHALL detect rocking from oscillation of the shoulder midpoint, and hand flapping from wrist oscillation near the shoulders. These labels SHALL NOT raise severity by themselves.
- **FR-020** (MUST): WHILE an episode is active, follow-up actuations SHALL use the latest episode severity. WHEN the episode state shows no hand-to-face or stereotypy activity, follow-up actuations SHALL stop.

**Evidence and logging**

- **FR-021** (MUST): For each episode, the system SHALL save:
  - a keyframe at peak severity;
  - a short clip with a skeleton overlay.
  - Clip length and retention are configurable.
- **FR-022** (MUST): The event log SHALL record, per decision:
  - verdict;
  - vision severity;
  - the stage 1 features;
  - the episode report reference.
- **FR-023** (SHOULD): The system SHALL estimate the camera–IMU clock offset and the IMU–wrist motion consistency for each episode, and report them. They SHALL NOT change the verdict.

**Robustness and replay**

- **FR-024** (MUST): WHEN the camera or DPU fails, the system SHALL return `UNDECIDED`, log the failure, and keep the IMU-only behavior.
- **FR-025** (MUST): The same analysis SHALL run on recorded sessions and produce the same features, verdicts and reports as on live input for the same frames.

### Nonfunctional

- **NFR-001** (MUST): The tracker SHALL keep its motion and measurement models swappable. A later articulated joint-angle state and IMU measurements must be addable without changing the feature, verdict or report consumers.
- **NFR-002** (MUST): The stage 1 verdict SHALL be produced within the configured verification deadline (Q-01).
- **NFR-003** (MUST): Pose inference SHALL run on the KV260 DPU (`DPUCZDX8G` B4096, Vitis AI 3.5, INT8). Throughput during an episode SHALL be at least the rate set in Q-02.
- **NFR-004** (MUST): The pipeline SHALL need only an RGB stream. It works with the F200 RGB or any UVC RGB camera, with no depth and no calibration.
- **NFR-005** (MUST): Evidence SHALL stay on the board's local storage and follow the configured retention.
- **NFR-006** (SHOULD): Detection quality on recorded demo sessions SHALL meet the targets set in Q-03.

## Interfaces, data, and compatibility

- **Verification result:**
  - Contents: `Verdict`, an optional vision severity (`Tag`), and a feature set.
  - It replaces the bare `Verdict` return, but must keep the dispatcher's deadline and fail-open contract.
  - `StubVerifier` semantics are kept.
- **Decision and event-log records:** add the effective severity, vision severity, features and episode-report reference. Old records stay readable.
- **Configuration (`config/board.yaml`):**
  - pre-roll length
  - idle timeout
  - subject-selection rule
  - prediction horizon
  - bone-length bounds
  - face-width source
  - contact threshold
  - severity rules
  - oscillation bands
  - clip length and retention
  - deadline and inference rate
- **Recorded-session input:** session format v1 from `src/cpsa/adapters/recording/` (FR-025).

## Acceptance criteria

**Trigger and capture**

- **AC-001** [FR-001, FR-002, FR-003]:
  - Given an idle board, no inference runs and the pre-roll stays at its configured length.
  - When a stereotypy detection arrives, the analysis includes frames from before the detection.
  - After the idle time with no activity, inference stops.

**Pose**

- **AC-002** [FR-004, FR-005]: On half-body and full-body test frames, the framing state is correct. Half-body frames keep upper-body confidences comparable to full-body frames.
- **AC-003** [FR-006]: With two people in view, exactly one subject is analyzed, as the configured rule selects.

**Tracking**

- **AC-004** [FR-007, FR-008]:
  - On a synthetic trajectory with noise, the filtered positions and velocities beat raw differencing.
  - A dropout shorter than the horizon is predicted.
  - A longer dropout is marked lost.
- **AC-005** [FR-009, FR-010, FR-011]:
  - An injected keypoint that breaks bone length is excluded.
  - The reference lengths stay stable across a session.
  - The foreshortening ratio drops when the forearm points toward the camera.

**Features**

- **AC-006** [FR-012, FR-013]: The features are unchanged when the same motion is scaled in image size (seated distance changes).
- **AC-007** [FR-014]: On labeled clips, contacts give the correct side and target zone, and the hand shape is right for fist, point and open palm.

**Fast verdict**

- **AC-008** [FR-015, NFR-002]: On recorded sessions:
  - hand-to-face episodes give `CONFIRMED`;
  - hands visible away from the face give `REJECTED`;
  - face or hands not visible gives `UNDECIDED`;
  - all within the deadline.
- **AC-009** [FR-016, FR-017]:
  - A fist to the eye under an IMU `NON_DANGEROUS` tag actuates as `DANGEROUS`.
  - A slow touch to the cheek under `DANGEROUS` actuates as `NON_DANGEROUS`.
  - `UNDECIDED` actuates as the IMU tag.
  - `REJECTED` does not actuate.

**Episode report**

- **AC-010** [FR-018, FR-019]:
  - Repeated contacts give the correct count and frequency.
  - Rocking and flapping sequences are labeled.
  - Those labels alone never raise severity.
- **AC-011** [FR-020]: In an episode, follow-ups use the latest severity and stop after activity ends.

**Evidence, logging and robustness**

- **AC-012** [FR-021, FR-022, NFR-005]: Each episode leaves a keyframe, a clip and log entries that cross-reference each other. Evidence older than the retention period is removed.
- **AC-013** [FR-024]:
  - A camera that is unplugged or a DPU that fails gives `UNDECIDED`, a logged error, and IMU-only actuation, with no crash.
  - A regression check shows the stub verifier behavior is unchanged.
- **AC-014** [FR-025]: Replay and live processing of the same frames give identical outputs.

**Nonfunctional**

- **AC-015** [NFR-001]: A second motion model can be swapped in (in a test) with no change to the consumers.
- **AC-016** [NFR-003, NFR-004]: On the KV260 with an RGB-only source, DPU pose inference reaches the Q-02 rate.
- **AC-017** [NFR-006]: Demo-session metrics meet Q-03.
- **AC-018** [FR-023]: The reported clock offset matches a known injected offset in replay.

## Assumptions and open decisions

**Assumptions:**
- One seated subject, framed head to waist, with the camera about 30–45° off frontal.
- The IMU is on the wrist.
- Rocking and flapping are non-dangerous stereotypies.

**Q-01** [NFR-002, FR-015]: What is the stage 1 verdict deadline (the placeholder is 1.0 s)? Proposal: measure it on the board, then fix it.

**Q-02** [NFR-003]: What is the minimum inference rate during an episode? Proposal: 30 fps. 15 fps is the floor for the speed and deceleration features.

**Q-03** [NFR-006]: What are the demo quality targets (verdict agreement and contact recall on labeled recorded sessions)? This needs recorded sessions first.

**Q-04** [FR-021, NFR-005]: What evidence retention period and subject consent apply?

**Q-05** [FR-006]: Which subject-selection rule: largest, most central, or the best IMU–wrist correlation?

## Readiness assessment

**Status:** READY
**Blockers:** None.
**Nonblocking risks:**
- Q-03 cannot be checked until sessions are recorded.
- Hand-shape accuracy depends on the fine-tuned model's hand keypoints.
- The depth and calibration tasks of `docs/plans/vision-toolchain.md` are off the required path.
