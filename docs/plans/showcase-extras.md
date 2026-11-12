# Showcase extras: MoveNet skeleton room + cognitive robot face — Implementation Plan

**Status:** READY (2026-11-12)
**Goal:** on top of the YOLOX-nano showcase (`docs/plans/showcase-only.md`):
1. a dashboard switch between **YOLO** (objects) and **MoveNet** (17-keypoint skeleton drawn inside a stylized
   simulated 3D room);
2. a **rule-based cognitive layer** (ACT-R style: perception → working memory → day memory → production rules with
   conflict resolution) that speaks Italian sentences through the browser voice and drives a **holographic robot
   face** with expressions and gaze.
**Spec (user, 2026-11-12):** "We provide a functionality in the dashboard to switch between yolo and movenet. With
movenet we show the skeleton overlap in a simulated room. At the same time we fix the functionalities of the
cognitive/chat bot architecture. We decide some phrases triggered by some situations, and we let an ai voice talk. In
the dashboard we have a robotic face, styled like an holografic face, that has some kind of expressions etc." /
"I want as many rules as possible for the rule based cognitive architecture".
**Constraints:** the 60 fps YOLO path must not regress; the cognitive layer runs in the browser (Chrome on Ubuntu
24.04 laptop; audio goes to a Bluetooth speaker paired to that laptop); no LLM, no API key; MoveNet from the Vitis AI
3.0 zoo package (`pt_movenet_coco_192_192_0.5G_3.0.zip`, quantized), compiled for our DPU (fingerprint
`0x101000016010407`; the zoo prebuilt `movenet_ntd_pt-zcu102_zcu104_kv260-r3.0.0` targets another fingerprint).

## Frozen interfaces

### F-E1 — protocol additions (WebSocket, envelope unchanged)

- Client → server `model.set` `{ "model": "yolox" | "movenet" }`; unknown model → `error` `bad_command`.
- `status` gains `"active_model": "yolox" | "movenet"`.
- Server → client `pose.state` (only while `movenet` is active, <= 30 Hz):
  `{ "model": "movenet", "width": 640, "height": 480, "fps": float|null, "dpu_ms": float|null, "total_ms": float|null,
     "score": float, "keypoints": [[x, y, score] x 17] }` — source-image pixels, COCO-17 order (nose, left_eye,
  right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
  left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle). `score` = mean keypoint score; when no person,
  `score` < 0.2 and the keypoints may be stale.
- `showcase.state` unchanged and sent only while `yolox` is active.

### F-E2 — MoveNet runtime (`src/cpsa/movenet.py`, new file)

`DpuMovenet(xmodel_path)` with `submit(rgb) -> token`, `wait(token) -> MovenetPose`, `stage_ms` (preprocess, dpu,
decode), mirroring `cpsa.yolox` detector conventions; `MovenetPose(keypoints: NDArray (17, 3) float32 in source
pixels, score: float)`; pure helpers `preprocess(rgb) -> model input` and `decode(heads) -> MovenetPose` testable
without VART.

### F-E3 — cognitive layer (web, TypeScript)

- `perception.ts`: debounced world model from `showcase.state` / `pose.state`: a class is present when seen in >= 6 of
  the last 10 states, absent after >= 2 s unseen; person boxes for gaze.
- Events: `appear(cls)`, `disappear(cls)`, `count(person, n_from, n_to)`, `first_seen(cls)`, `return(cls, absence_s)`,
  `dwell(cls, s)`, `together(a, b)`, `idle(s)`, `milestone(n)`, `pose(kind)` (arms_up, wave, t_pose, crouch, hands_on_head,
  lean), `mode(model)`, `clock(hour)`.
- `memory.ts`: working memory (last 10 events, focus, last utterance); day memory per class (first/last seen,
  times, total dwell) + totals (people seen, max at once), persisted in `localStorage` (try/catch) per calendar day.
- `rules.ts` engine: every 250 ms collect matching rules not in cooldown; utility = priority + novelty − habituation
  (ACT-R-like decay of how often that class was commented) + small noise; fire one; global gap >= 4 s between
  utterances; while speaking only priority >= 80 queues (max 1 queued); variants never repeat the previous one.
- `rulebook.ts` (data): rules `{ id, when(event, memory) -> bool, priority, cooldown_s, per_object_cooldown_s?,
  max_per_day?, expression, gaze, phrases: string[] (slots {oggetto} {n} {ora}) }`; target >= 70 rules / >= 150
  phrases in Italian, including one rule per COCO class.
- `voice.ts`: Web Speech API, prefer a voice whose lang starts with `it` (prefer "Google italiano"), fallback any;
  rate/pitch slightly robotic; exposes speaking state and word boundaries for the mouth; a mute toggle.
- `face.ts`: SVG/canvas holographic robot face (cyan glow, scanlines): eyes with blink, gaze (x/y toward the largest
  person, idle wander), expressions neutral/happy/surprised/curious/sleepy/thinking, mouth bar animated while speaking;
  speech bubble with the current sentence.

### F-E4 — corner cases the cognitive layer MUST handle (each one gets a vitest case)

User requirement (2026-11-12): "avoid any concurrent firing, wrong ones etc. For instance, if more than a person is
detected, then we dont let it speak n times, just once for everyone. If a single person is identified then we handled
it differently etc. Identify all possible corners and possible problems and handle them accordingly."

**People**
1. *Several people arrive together* (0→3): one collective line ("Ciao a tutti!"), never one per person. Person-count
   changes are settled for 1.5 s and the rule sees only the final stable count.
2. *Single vs group*: separate rule families — exactly 1 person (direct "tu"), 2 people ("voi due"), 3–4 ("voi"), >= 5
   crowd mode (only group-level lines; object comments suppressed; longer gaps, >= 8 s).
3. *Count flicker* (2↔3 from detection noise): a new count must hold >= 1.5 s (hysteresis); otherwise no event.
4. *Short absence* (occlusion, person leaves < 10 s): no new greeting; a "welcome back" only after >= 30 s away.
5. *No identities*: there is no re-identification; never claim "you were here before" for a person; day totals are
   approximate (sum of positive stable-count steps) and lines say "circa".
6. *Arrival during speech*: the greeting is queued (priority >= 80), re-validated before speaking (still someone
   there?) and dropped if older than 3 s.

**Objects**
7. *Several objects at once*: one sentence about the most salient one (highest novelty/priority), optionally naming a
   second ("vedo un telefono e una tazza"); never a list of everything.
8. *Same class multiple instances*: correct Italian plural/article/gender per class ("il telefono", "la tazza",
   "le forbici" plural-only, "l'ombrello", "due tazze") from a per-class grammar table (all 80 classes).
9. *Scenery*: objects already present in the first 30 s after start/reconnect, or continuously present > 2 min (chair,
   tv, dining table…) become scenery and are never commented unless they disappear >= 30 s and come back.
10. *False positives*: speech needs score >= 0.5, stable presence, and a minimum box area; implausible indoor classes
    (elephant, giraffe, zebra, bear, airplane, train, boat, cow, horse, sheep) need score >= 0.7 for >= 2 s and use a
    doubtful line ("sembra una giraffa… ma forse mi sbaglio").
11. *Objects that belong to people* (tie, handbag, backpack, phone in hand) are fine; never comment on bodies,
    clothes style, age, gender or looks — objects and actions only.
12. *Sharp objects* (knife, scissors): gentle line, no alarm, low frequency.

**Speech**
13. *One voice at a time*: one utterance in flight; at most 1 queued (priority >= 80); global gap >= 4 s (>= 8 s in crowd
    mode); every queued item re-checks its rule condition just before speaking and expires after 3 s.
14. *No repetition*: per-rule cooldown, per-object cooldown, habituation (ACT-R-like decay), variant never equal to the
    previous one for that rule, `max_per_day` for milestones and facts.
15. *Chrome autoplay*: speech starts only after a user click — the dashboard shows "Attiva la voce" once; state kept.
16. *Voices load asynchronously* (`onvoiceschanged`); prefer `it-*` ("Google italiano", needs internet), fallback to any
    Italian voice, else show a muted badge (no crash).
17. *Stuck speech* (Chrome bug on long utterances): sentences <= ~12 words; watchdog cancels after 8 s.
18. *Several dashboard tabs*: only the tab with voice enabled speaks (`?voice=1` or the toggle); others stay silent.
19. *Mute/volume*: a mute toggle in the top bar; muted still animates the face and bubble.

**Lifecycle**
20. *Start/reload/reconnect warm-up*: for 3 s after a (re)connect the world model is seeded silently (no appear events).
21. *WebSocket loss / camera error*: one line ("Ho perso la vista!"), worried face, no other rules until data resumes.
22. *Mode switch* YOLO↔MoveNet: announce once; reset perception without disappear storms; object/people-count rules
    only in YOLO, pose rules only in MoveNet (single person only — MoveNet has no count).
23. *Day rollover* at midnight: day memory resets; reload keeps the same day's memory (`localStorage`).
24. *Idle*: "where are you?" only after >= 15 s with nobody; facts about itself at most once per 60 s, never in crowd
    mode; "sei ancora lì?" if one person stays >= 25 s with nothing new.

**Pose (MoveNet)**
25. Gestures must be held/confirmed: arms up >= 0.5 s; wave = >= 2 wrist oscillations within 1.5 s with amplitude
    >= 15% of shoulder width; per-gesture cooldown 10 s; keypoints with score < 0.3 ignored; no pose rule when the
    pose score < 0.3.

**Face**
26. Gaze target = the largest person box, switched only if another box is >= 30% larger for >= 1 s; smoothed; idle
    wander when nobody.
27. Expression held for the utterance + 1.5 s, then decays to neutral (sleepy after 60 s idle).

## Verification contract

- **AC-E1** MoveNet compiled: fingerprint `0x101000016010407`, 1 DPU subgraph, input 192x192; reference check: on >= 20
  images with people, our decode matches the package's reference keypoints (mean distance <= 2 px at 192 scale for
  keypoints with score >= 0.3). Unit tests for preprocess/decode.
- **AC-E2** `model.set` switches models in <= 200 ms without restarting; YOLO path still 60 fps on the board; MoveNet
  path >= 30 fps on the board.
- **AC-E3** web: switch buttons YOLO/MoveNet; skeleton room view renders `pose.state`; face + voice + rules run with the
  mock (scripted scenario: empty → person appears → phone → two people → leave → idle) and speak in Italian; engine unit
  tests (vitest) for debounce, events, cooldowns, conflict resolution, no-repeat, persistence.
- **AC-E4** board + laptop: user hears Italian sentences on the Bluetooth speaker reacting to real scenes; face follows
  the nearest person.

## Tasks

- **M1 — MoveNet model and runtime** (backend new files only): `tools/vitis/zoo_movenet.sh`, `tools/vitis/movenet_reference.py`,
  `src/cpsa/movenet.py`, `tests/test_movenet.py`. AC-E1.
- **C1 — cognitive layer + face + voice** (web only; `web/src/cognitive/*`, `web/src/face.ts`, `web/src/voice.ts`,
  layout integration in the showcase view, mock scenario, vitest setup). AC-E3 (YOLO part; pose rules driven by the
  mock's `pose.state`).
- **M2 — MoveNet in the app + room view** (after M1 and C1): backend `model.set`, `pose.state`, `status.active_model`,
  both runners preloaded in process A; web YOLO/MoveNet switch + skeleton room (perspective room, floor grid, glow;
  style of the deleted `web/src/hologram.ts`, available in git history at `fded0e7`). AC-E2, AC-E3.
- **K-board** (coordinator): deploy, measure, user test. AC-E2, AC-E4.

## Round 2 (user feedback 2026-11-12, approved)

User (verbatim, abridged): "I dont like the face. First of all it's half, not full face. plus, i dont want a front
face, a 2/3 third view positioned on top right would be much cooler. Plus the face should look like [a digital face
made of luminous points] with animated expressions and mouth. [...] I prefer something like an animated wave [a
point-cloud sphere]. [...] the voice is very clearly fake. Cant we use something more human? [...] when it's idle, it
says some random facts or quick poetry. Next, the interactions with human should be more cohesive. For example when a
human waves, smiles, etc." Approved design: hybrid particle sphere <-> face, local human TTS, scenes, MediaPipe smile
detection on the laptop. "The random facts are especially interesting."

- **R2-F (avatar, web):** one three.js particle system: idle = breathing point-cloud wave sphere (cyan→violet, like the
  user's reference); a person engages = particles flow into a 3D head (2/3 view, top right, full face) with ARKit
  blendshapes (expressions: neutral, happy/smile, surprised, curious, thinking, sleepy, worried; blink; idle
  micro-motion) and a mouth driven by the voice audio (WebAudio analyser → jawOpen/visemes); leaving = back to sphere.
  Head mesh: three.js `examples/models/gltf/facecap.glb` if its licence allows reuse (check the three.js repo), else a
  TalkingHead (met4citizen, MIT) sample avatar head with ARKit blendshapes, else a procedural head.
- **R2-V (voice, laptop):** local TTS HTTP service (`tools/tts/`): Kokoro (`if_sara`, `im_nicola`) and XTTS-v2 (Italian),
  disk cache keyed by (engine, voice, text), pre-generation of all fixed phrases; the browser plays the WAV via WebAudio
  (amplitude feeds the mouth) with Web Speech as the fallback.
- **R2-D (dialogue, web):** scene state machine Attract → Engage → Play → Farewell with continuity (refers to what
  happened); idle content = curated Italian facts and short original poems (`web/src/cognitive/content/`), never
  repeated within a day; gesture/face events from pose (MoveNet, board) and from MediaPipe Face Landmarker running in
  the laptop browser on `/video.mjpeg` (smile, mouth open, nod, head shake, looking at the camera); F-E4 corner cases
  still hold.
- **R2-G (board):** MoveNet runs every frame alongside YOLOX (DPU ~4.7 + ~2–3 ms) so gestures work in both views;
  `pose.state` always published; the YOLO/MoveNet switch only changes the view.
