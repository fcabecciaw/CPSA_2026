/**
 * Dev-only WebSocket stand-in (`/?mock=1`).
 *
 * It replays the protocol v1 examples from `docs/protocol/v1/examples` and
 * animates a plausible HALFBUST_62 pose so the views can be built before the
 * server exists. The module is imported with a dynamic `import()` behind
 * `import.meta.env.DEV`, so Vite removes it from the production bundle.
 *
 * Both activation sources of the medical mode are available: `?mock=1` replays
 * the `imu` variant, `?mock=1&trigger=vision` the camera-only one, whose
 * `medical.state` reports `trigger: "vision"` and `imu: "n/a"` (F11) and whose
 * connection log has no BlueCoin lines, because no IMU is scanned (FR-026).
 */

import type { SocketLike } from "./client";
import {
  MODES,
  type Decision,
  type EpisodePayload,
  type HandsPayload,
  type HologramHandsPayload,
  type ImuState,
  type Keypoint,
  type LogPayload,
  type MedicalStatePayload,
  type Mode,
  type PerceptionTiming,
  type Point3D,
  type RpsGesture,
  type RpsMove,
  type RpsOutcome,
  type RpsPhase,
  type RpsStatePayload,
  type ServerMessage,
  type StatusPayload,
  type Tag,
  type TriggerSource,
} from "./protocol";

/** IF-451 revision: `?mock=1&trigger=vision` selects the camera-only variant. */
function requestedTrigger(): TriggerSource {
  const requested = new URLSearchParams(window.location.search).get("trigger");
  return requested === "vision" ? "vision" : "imu";
}

const TRIGGER: TriggerSource = requestedTrigger();

/**
 * `?mock=1&mode=hologram` (or `rps`) starts directly in a hand-demo mode, so a
 * screenshot does not have to click through the switch.
 */
function requestedMode(): Mode {
  const requested = new URLSearchParams(window.location.search).get("mode");
  return (MODES as readonly string[]).includes(requested ?? "")
    ? (requested as Mode)
    : "medical";
}

const INITIAL_MODE: Mode = requestedMode();

export const MOCK_BADGE =
  "Dati di esempio (mock): replay di docs/protocol/v1/examples, nessun server connesso." +
  (TRIGGER === "vision"
    ? " Variante solo telecamera (variante IMU: ?mock=1)."
    : " Variante IMU (variante solo telecamera: ?mock=1&trigger=vision).");

const FIXTURES = import.meta.glob("../../docs/protocol/v1/examples/*.json", {
  eager: true,
  import: "default",
}) as Record<string, unknown>;

type MessageOf<T extends ServerMessage["type"]> = Extract<ServerMessage, { type: T }>;

function fixture<T extends ServerMessage["type"]>(type: T): MessageOf<T> {
  const path = `../../docs/protocol/v1/examples/${type}.json`;
  const raw = FIXTURES[path];
  if (raw === undefined) {
    throw new Error(`Fixture mock mancante: ${path}`);
  }
  return structuredClone(raw) as MessageOf<T>;
}

const WIDTH = 640;
const HEIGHT = 480;
const POSE_PERIOD_HZ = 1.4;
const EPISODE_DURATION_MS = 22_000;
const EPISODE_PAUSE_MS = 5_000;

/** Sasso-carta-forbice timings, the same shape as the real state machine. */
const RPS_READY_MS = 600;
const RPS_STEP_MS = 700;
const RPS_SHOOT_MS = 400;
const RPS_RESULT_MS = 3_000;

/** IR camera guess, used only to draw the mock hands' 2D points. */
const MOCK_FOCAL_PX = 500;

const IMU_LOG_LINES: readonly LogPayload[] = [
  { level: "info", source: "supervisor", message: "Modalità medical avviata" },
  { level: "info", source: "camera", message: "telecamera aperta" },
  { level: "info", source: "bluecoin:CPSA_L2", message: "[left:CPSA_L2] Connected successfully" },
  { level: "info", source: "bluecoin:CPSA_R2", message: "[right:CPSA_R2] Connected successfully" },
  { level: "info", source: "metamotion", message: "[MetaMotion AA:BB:CC:DD:EE:FF] Connected successfully" },
  { level: "warning", source: "bluecoin:CPSA_R2", message: "[right:CPSA_R2] Connection failed, retrying..." },
  { level: "info", source: "bluecoin:CPSA_R2", message: "[right:CPSA_R2] Reconnected successfully" },
  { level: "error", source: "camera", message: "camera frame callback failed" },
  { level: "info", source: "camera", message: "telecamera riaperta" },
];

/** Camera-only variant: no IMU is scanned, so there are no BlueCoin lines (FR-026). */
const VISION_LOG_LINES: readonly LogPayload[] = [
  { level: "info", source: "supervisor", message: "Modalità medical avviata, attivazione da telecamera" },
  { level: "info", source: "camera", message: "telecamera aperta" },
  { level: "info", source: "vision", message: "analisi continua, 10 fps" },
  { level: "info", source: "vision", message: "nessun contatto negli ultimi 10 s" },
  { level: "warning", source: "camera", message: "frame scartato: coda di analisi piena" },
  { level: "error", source: "camera", message: "camera frame callback failed" },
  { level: "info", source: "camera", message: "telecamera riaperta" },
];

const LOG_LINES = TRIGGER === "vision" ? VISION_LOG_LINES : IMU_LOG_LINES;

let sequence = 0;
let monotonic = 2000;
let frameSeq = 0;

function message<T extends ServerMessage["type"]>(
  type: T,
  data: MessageOf<T>["data"],
): ServerMessage {
  sequence += 1;
  monotonic += 0.05;
  return {
    v: 1,
    type,
    seq: sequence,
    t: monotonic,
    wall: new Date().toISOString(),
    data,
  } as ServerMessage;
}

export function createMockSocket(_url: string): SocketLike {
  const timers = new Set<ReturnType<typeof setTimeout>>();
  const intervals = new Set<ReturnType<typeof setInterval>>();

  let closed = false;
  let mode: Mode = INITIAL_MODE;
  let switching = false;
  let episodeNumber = 0;
  let episodeId = "";
  let episodeActive = false;
  let episodeStart = 0;
  let tag: Tag = "DANGEROUS";
  let lastDecision: Decision = fixture("medical.decision").data;
  let decisionIndex = 0;
  let logIndex = 0;
  let startedAt = Date.now();
  let hologramSeq = 0;

  // Sasso-carta-forbice state (T5): the same shape the real state machine has,
  // replayed so the view can be built without the board.
  let rpsPhase: RpsPhase = "waiting";
  let rpsSince = Date.now();
  let rpsCount = 3;
  let rpsRound = 0;
  let rpsPlayer: RpsGesture | null = null;
  let rpsPlayerP: number | null = null;
  let rpsBoard: RpsMove | null = null;
  let rpsOutcome: RpsOutcome | null = null;
  const rpsScore = { player: 0, board: 0, draws: 0 };

  const socket: SocketLike = {
    onopen: null,
    onmessage: null,
    onclose: null,
    onerror: null,
    send: (data: string) => handleClient(data),
    close: () => stop(),
  };

  const later = (ms: number, run: () => void): void => {
    const id = setTimeout(() => {
      timers.delete(id);
      if (!closed) {
        run();
      }
    }, ms);
    timers.add(id);
  };

  const every = (ms: number, run: () => void): void => {
    const id = setInterval(() => {
      if (!closed) {
        run();
      }
    }, ms);
    intervals.add(id);
  };

  function emit(msg: ServerMessage): void {
    if (!closed) {
      socket.onmessage?.({ data: JSON.stringify(msg) });
    }
  }

  function stop(): void {
    closed = true;
    for (const id of timers) {
      clearTimeout(id);
    }
    for (const id of intervals) {
      clearInterval(id);
    }
    timers.clear();
    intervals.clear();
  }

  function sendError(
    code: "busy" | "bad_command",
    text: string,
    refSeq: number | null = null,
  ): void {
    emit(message("error", { code, message: text, ref_seq: refSeq }));
  }

  function sendStatus(): void {
    const status: StatusPayload = fixture("status").data;
    status.mode = mode;
    // Idle and lab run nothing; the three demo modes run a camera pipeline.
    const vision = mode === "medical" || mode === "hologram" || mode === "rps";
    status.mode_state = switching ? "starting" : vision ? "running" : "stopped";
    const fps = vision ? 29.6 + Math.sin(Date.now() / 900) * 0.8 : null;
    status.perception.fps = fps;
    status.perception.dpu_ms = vision ? 17.8 : null;
    status.perception.camera = "ok";
    status.perception.model = vision
      ? mode === "medical"
        ? "pose_halfbust62"
        : "pose_hand21_int8"
      : null;
    // F13: plausible rolling-window numbers, so the "Tempi" line and the FPS
    // caption can be built before the server reports them.
    status.perception.timing = vision && fps !== null ? mockTiming(fps) : null;
    status.imu = mode === "medical" && TRIGGER === "imu" ? imuState() : "n/a";
    status.errors = [];
    status.uptime_s = 125.4 + (Date.now() - startedAt) / 1000;
    emit(message("status", status));
  }

  function mockTiming(fps: number): PerceptionTiming {
    const drift = Math.sin(Date.now() / 1300);
    const round = (value: number): number => Math.round(value * 10) / 10;
    return {
      analyzed_fps: round(fps),
      camera_fps: 30,
      decode_ms: round(3.1 + drift * 0.3),
      preprocess_ms: round(2.2 + drift * 0.2),
      dpu_ms: round(17.8 + drift * 0.4),
      postprocess_ms: round(4.4 + drift * 0.3),
      pose_ms: round(25.4 + drift * 0.6),
      track_ms: round(1.1 + drift * 0.1),
      latency_ms: round(61.2 + drift * 2.5),
      dropped: 0,
    };
  }

  function imuState(): ImuState {
    const elapsed = (Date.now() - startedAt) / 1000;
    if (elapsed > 40 && elapsed < 46) {
      return "lost";
    }
    if (elapsed >= 46 && elapsed < 52) {
      return "searching";
    }
    return "connected";
  }

  function episodeRef(): MedicalStatePayload["episode"] {
    return episodeActive
      ? { episode_id: episodeId, t_start: 1234.5 + episodeNumber, active: true }
      : null;
  }

  function sendMedicalState(): void {
    if (mode !== "medical") {
      return;
    }
    const state: MedicalStatePayload = {
      imu: TRIGGER === "vision" ? "n/a" : imuState(),
      tag,
      episode: episodeRef(),
      last_decision: lastDecision,
      trigger: TRIGGER,
    };
    emit(message("medical.state", state));
  }

  function startEpisode(): void {
    episodeNumber += 1;
    episodeId = `ep-20261112-${String(episodeNumber).padStart(4, "0")}`;
    episodeActive = true;
    episodeStart = Date.now();
    tag = "DANGEROUS";

    const episode: EpisodePayload = {
      episode_id: episodeId,
      active: true,
      contact_count: 0,
      repetition_hz: POSE_PERIOD_HZ,
      peak_severity: null,
      labels: ["hand_to_face"],
      contacts: [],
    };
    emit(message("medical.episode", episode));
    sendMedicalState();
    later(EPISODE_DURATION_MS, endEpisode);
  }

  function endEpisode(): void {
    if (!episodeActive) {
      return;
    }
    episodeActive = false;
    if (mode !== "medical") {
      // Leaving medical mode: the server drops the medical states instead of
      // replaying a final summary (V-04), so the dashboard resets its tiles.
      return;
    }
    const summary: EpisodePayload = fixture("medical.episode").data;
    summary.episode_id = episodeId;
    summary.active = false;
    emit(message("medical.episode", summary));
    sendMedicalState();
    later(EPISODE_PAUSE_MS, () => {
      if (mode === "medical" && !episodeActive) {
        startEpisode();
      }
    });
  }

  function sendHands(): void {
    if (mode !== "medical" || !episodeActive) {
      return;
    }
    emit(message("medical.hands", pose(Date.now() - episodeStart)));
  }

  // --- Hologram (T5): an animated hand at the camera rate -------------------

  function sendHologram(): void {
    if (mode !== "hologram") {
      return;
    }
    hologramSeq += 1;
    emit(message("hologram.hands", hologramFrame(hologramSeq)));
  }

  // --- Sasso-carta-forbice (T5): the round loop ----------------------------

  function startRpsRound(): void {
    rpsPhase = "ready";
    rpsSince = Date.now();
    rpsCount = 3;
    rpsPlayer = null;
    rpsPlayerP = null;
    rpsBoard = null;
    rpsOutcome = null;
  }

  function resetRpsGame(): void {
    rpsRound = 0;
    rpsScore.player = 0;
    rpsScore.board = 0;
    rpsScore.draws = 0;
    startRpsRound();
  }

  function tickRps(): void {
    if (mode !== "rps") {
      return;
    }
    advanceRps(Date.now());
    emitRps();
  }

  function advanceRps(now: number): void {
    const elapsed = now - rpsSince;
    if (rpsPhase === "ready" && elapsed >= RPS_READY_MS) {
      rpsPhase = "countdown";
      rpsCount = 3;
      rpsSince = now;
      return;
    }
    if (rpsPhase === "countdown" && elapsed >= RPS_STEP_MS) {
      if (rpsCount > 1) {
        rpsCount -= 1;
        rpsSince = now;
        return;
      }
      rpsPhase = "shoot";
      rpsSince = now;
      return;
    }
    if (rpsPhase === "shoot" && elapsed >= RPS_SHOOT_MS) {
      settleRound();
      rpsPhase = "result";
      rpsSince = now;
      return;
    }
    if (rpsPhase === "result" && elapsed >= RPS_RESULT_MS) {
      startRpsRound();
    }
  }

  function settleRound(): void {
    rpsRound += 1;
    const gestures: readonly RpsMove[] = ["rock", "paper", "scissors"];
    // Every fifth round the classifier recognises nothing, so the "unclear"
    // branch is visible in the mock too and the score stays put.
    const gesture: RpsGesture =
      rpsRound % 5 === 0 ? "none" : (gestures[(rpsRound - 1) % gestures.length] ?? "rock");
    rpsPlayer = gesture;
    if (gesture === "none") {
      rpsPlayerP = 0.38;
      rpsBoard = null;
      rpsOutcome = "unclear";
      return;
    }
    rpsPlayerP = 0.92 - (rpsRound % 3) * 0.05;
    const board = gestures[Math.floor(Math.random() * gestures.length)] ?? "rock";
    rpsBoard = board;
    rpsOutcome = rpsResult(gesture, board);
    if (rpsOutcome === "win") {
      rpsScore.player += 1;
    } else if (rpsOutcome === "lose") {
      rpsScore.board += 1;
    } else {
      rpsScore.draws += 1;
    }
  }

  function emitRps(): void {
    const shown: RpsGesture | null =
      rpsPhase === "ready" || rpsPhase === "countdown" || rpsPhase === "shoot"
        ? "paper"
        : rpsPhase === "result" && rpsPlayer !== null && rpsPlayer !== "none"
          ? rpsPlayer
          : null;
    const state: RpsStatePayload = {
      phase: rpsPhase,
      count: rpsPhase === "countdown" ? rpsCount : null,
      player: rpsPhase === "result" ? rpsPlayer : null,
      player_p: rpsPhase === "result" ? rpsPlayerP : null,
      board: rpsPhase === "result" ? rpsBoard : null,
      outcome: rpsPhase === "result" ? rpsOutcome : null,
      score: { ...rpsScore },
      hand: shown === null ? null : { kpts2d: rpsKeypoints(shown, Date.now()) },
    };
    emit(message("rps.state", state));
  }

  function sendDecision(): void {
    if (mode !== "medical") {
      return;
    }
    const decisions = decisionVariants();
    const decision = structuredClone(decisions[decisionIndex % decisions.length]);
    decisionIndex += 1;
    tag = decision.tag;
    lastDecision = decision;
    emit(message("medical.decision", decision));
    sendMedicalState();
  }

  function sendLog(): void {
    const line = LOG_LINES[logIndex % LOG_LINES.length];
    logIndex += 1;
    emit(message("log", line));
  }

  function handleClient(raw: string): void {
    let value: unknown;
    try {
      value = JSON.parse(raw);
    } catch {
      sendError("bad_command", "JSON non valido");
      return;
    }
    if (typeof value !== "object" || value === null) {
      sendError("bad_command", "comando non valido");
      return;
    }
    const command = value as { type?: unknown; data?: unknown; seq?: unknown };
    const refSeq = typeof command.seq === "number" ? command.seq : null;
    if (command.type === "ping") {
      const data = (command.data ?? {}) as { id?: unknown };
      emit(message("pong", { id: data.id }));
      return;
    }
    if (command.type === "mode.set") {
      const data = (command.data ?? {}) as { mode?: unknown };
      const requested = data.mode;
      if (typeof requested !== "string" || !(MODES as readonly string[]).includes(requested)) {
        sendError("bad_command", `modalità sconosciuta: ${String(requested)}`, refSeq);
        return;
      }
      if (switching) {
        sendError("busy", "a mode switch is already in progress", refSeq);
        return;
      }
      switchMode(requested as Mode);
      return;
    }
    sendError("bad_command", `comando sconosciuto: ${String(command.type)}`, refSeq);
  }

  function switchMode(next: Mode): void {
    if (next === mode && !switching) {
      emit(message("mode.progress", { mode: next, step: "ready", error: null }));
      return;
    }
    switching = true;
    emit(message("mode.progress", { mode: next, step: "stopping", error: null }));
    sendStatus();

    later(350, () => {
      emit(message("mode.progress", { mode: next, step: "releasing", error: null }));
    });

    later(700, () => {
      switching = false;
      mode = next;
      emit(message("mode.progress", { mode: next, step: "ready", error: null }));
      sendStatus();
      if (next === "medical") {
        if (!episodeActive) {
          startEpisode();
        }
      } else if (episodeActive) {
        endEpisode();
      }
      if (next === "rps") {
        resetRpsGame();
      }
    });
  }

  function snapshot(): void {
    emit(fixture("hello"));
    sendStatus();
    const progress = fixture("mode.progress");
    progress.data = { mode, step: "ready", error: null };
    emit(progress);
    // Recent history, as the server replays it after the state snapshot.
    for (const line of LOG_LINES.slice(0, 5)) {
      emit(message("log", line));
    }
    logIndex = 5;
    if (mode === "medical") {
      startEpisode();
      // `medical.state` seeds the feed with its last decision; start the cycle
      // after it so the first visible rows differ.
      decisionIndex = 1;
    }
    if (mode === "rps") {
      resetRpsGame();
    }
  }

  later(10, () => {
    socket.onopen?.();
    snapshot();
    every(50, sendHands);
    every(33, sendHologram);
    every(100, tickRps);
    every(1000, sendMedicalState);
    every(2000, sendStatus);
    every(5000, sendDecision);
    every(9000, sendLog);
  });

  return socket;
}

function decisionVariants(): Decision[] {
  const base = fixture("medical.decision").data;
  return [
    base,
    {
      tag: "NON_STEREOTIPY",
      tag_changed: true,
      verdict: "REJECTED",
      reason: "REJECTED",
      severity: "NON_STEREOTIPY",
      vision: {
        verdict: "REJECTED",
        severity: "NON_STEREOTIPY",
        episode_id: null,
        features: { contact_count: 0, repetition_hz: null },
      },
      actuations: [],
    },
    {
      tag: "NON_DANGEROUS",
      tag_changed: true,
      verdict: "UNDECIDED",
      reason: "UNDECIDED",
      severity: "NON_DANGEROUS",
      vision: {
        verdict: "UNDECIDED",
        severity: "NON_DANGEROUS",
        episode_id: null,
        features: { contact_count: 1, repetition_hz: 0.4 },
      },
      actuations: [],
    },
    {
      tag: "DANGEROUS",
      tag_changed: true,
      verdict: "CONFIRMED",
      reason: "CONFIRMED",
      severity: "DANGEROUS",
      vision: {
        verdict: "CONFIRMED",
        severity: "DANGEROUS",
        episode_id: null,
        features: { contact_count: 3, repetition_hz: 1.6 },
      },
      actuations: base.actuations,
    },
    {
      tag: "NO_CLASS",
      tag_changed: true,
      verdict: null,
      reason: "TIMEOUT",
      severity: null,
      vision: null,
      actuations: [],
    },
  ];
}

function pose(elapsedMs: number): HandsPayload {
  const points: Keypoint[] = Array.from({ length: 62 }, () => [0, 0, 0]);
  const set = (index: number, x: number, y: number, conf = 0.97): void => {
    points[index] = [x, y, conf];
  };

  // Body (COCO-17 points 0..10) and face anchors (11..19).
  set(0, 320, 140);
  set(1, 306, 130);
  set(2, 334, 130);
  set(3, 282, 142, 0.42);
  set(4, 358, 142);
  set(5, 238, 330);
  set(6, 402, 330);
  set(7, 210, 408);
  set(8, 430, 408);
  set(9, 252, 300);
  set(10, 388, 300);

  set(11, 288, 172);
  set(12, 352, 172);
  set(13, 320, 198);
  set(14, 296, 116);
  set(15, 344, 116);
  set(16, 304, 176);
  set(17, 336, 176);
  set(18, 320, 170);
  set(19, 320, 183);

  const phase = (elapsedMs / 1000) * 2 * Math.PI * POSE_PERIOD_HZ;
  const rightWave = 0.5 + 0.5 * Math.sin(phase);
  const leftWave = 0.5 + 0.5 * Math.sin(phase + Math.PI);

  // The right hand rises to the mouth; the left hand reaches the cheek.
  writeHand(points, 41, 410 - 74 * rightWave, 318 - 122 * rightWave, -0.45, 0.97);
  writeHand(points, 20, 230 + 68 * leftWave, 318 - 112 * leftWave, 0.45, 0.97);

  return {
    layout: "HALFBUST_62",
    w: WIDTH,
    h: HEIGHT,
    keypoints: points,
    subject_box: [150, 60, 490, 470],
    // F14: stands in for the analyzed frame counter the real stream reports.
    seq: frameSeq++,
  };
}

const HAND_LOCAL: readonly (readonly [number, number])[] = [
  [0, 18],
  [-10, 10],
  [-16, 4],
  [-19, -2],
  [-20, -8],
  [-8, 2],
  [-9, -8],
  [-10, -15],
  [-11, -21],
  [-2, 0],
  [-2, -11],
  [-2, -19],
  [-2, -26],
  [5, 1],
  [7, -9],
  [8, -16],
  [9, -22],
  [11, 4],
  [14, -4],
  [16, -10],
  [17, -15],
];

const LOW_CONFIDENCE = new Set([3, 8, 14, 19]);

function writeHand(
  points: Keypoint[],
  offset: number,
  centerX: number,
  centerY: number,
  angle: number,
  baseConfidence: number,
): void {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  HAND_LOCAL.forEach(([x, y], index) => {
    const rotatedX = x * cos - y * sin;
    const rotatedY = x * sin + y * cos;
    const confidence = LOW_CONFIDENCE.has(index) ? 0.3 : baseConfidence;
    points[offset + index] = [centerX + rotatedX, centerY + rotatedY, confidence];
  });
}

/**
 * hand21 in "hand units": wrist at the origin, fingers towards -v, thumb on the
 * left. Shared by the two hand demos, which need a plausible 21-joint shape
 * without a board.
 */
const HAND21_LOCAL: readonly (readonly [number, number])[] = [
  [0, 0],
  // thumb: cmc, mcp, ip, tip
  [-12, -6],
  [-20, -18],
  [-26, -30],
  [-30, -40],
  // index: mcp, pip, dip, tip
  [-16, -44],
  [-18, -60],
  [-19, -72],
  [-19, -83],
  // middle
  [-5, -48],
  [-6, -66],
  [-6, -79],
  [-6, -91],
  // ring
  [6, -46],
  [7, -64],
  [7, -76],
  [7, -87],
  // pinky
  [16, -40],
  [18, -54],
  [18, -66],
  [19, -75],
];

/** Finger chains after the wrist: thumb, index, middle, ring, pinky. */
const HAND21_FINGERS: readonly (readonly number[])[] = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];

/** Per-finger extension: 0 curled into the palm, 1 fully stretched. */
const GESTURE_EXTENSION: Record<RpsGesture, readonly number[]> = {
  rock: [0.1, 0.05, 0, 0, 0],
  paper: [0.9, 1, 1, 1, 1],
  scissors: [0.2, 1, 1, 0.05, 0],
  none: [0.4, 0.55, 0.3, 0.45, 0.65],
};

/** Who beats whom: the board plays the same three gestures. */
const BEATS: Record<RpsMove, RpsMove> = {
  rock: "scissors",
  paper: "rock",
  scissors: "paper",
};

function rpsResult(player: RpsMove, board: RpsMove): RpsOutcome {
  if (player === board) {
    return "draw";
  }
  return BEATS[player] === board ? "win" : "lose";
}

/** Curled fingers stay near their knuckle and fold towards the palm. */
function localHand(extension: readonly number[]): Array<readonly [number, number]> {
  const points: Array<readonly [number, number]> = HAND21_LOCAL.map(([x, y]) => [x, y]);
  HAND21_FINGERS.forEach((chain, finger) => {
    const knuckle = chain[0] === undefined ? undefined : points[chain[0]];
    if (knuckle === undefined) {
      return;
    }
    const amount = extension[finger] ?? 0;
    chain.forEach((index, step) => {
      const target = HAND21_LOCAL[index];
      if (step === 0 || target === undefined) {
        return;
      }
      const reach = 0.15 + 0.85 * amount;
      const fold = 1 - amount;
      points[index] = [
        knuckle[0] + (target[0] - knuckle[0]) * reach,
        knuckle[1] + (target[1] - knuckle[1]) * reach + (fold * 9 * step) / 3,
      ];
    });
  });
  return points;
}

/** IR pixels of the mock hand, as the classifier would see it. */
function rpsKeypoints(gesture: RpsGesture, nowMs: number): Keypoint[] {
  const sway = Math.sin(nowMs / 380) * 2.5;
  return localHand(GESTURE_EXTENSION[gesture]).map(([x, y]) => [
    WIDTH / 2 + x * 1.9 + sway,
    HEIGHT * 0.62 + y * 1.9,
    0.96,
  ]);
}

/**
 * One `hologram.hands` frame: the hand opens and closes over seven seconds, so
 * the 3D view has something to follow. The pinky tip loses its depth every
 * fifth frame, which is what the view must survive with the median-depth rule.
 */
function hologramFrame(seq: number): HologramHandsPayload {
  const now = Date.now();
  // The hand opens and half-closes over seven seconds: a moving 3D hand whose
  // fingers stay readable, which a full fist would not.
  const wave = 0.5 + 0.5 * Math.cos(((now % 7_000) / 7_000) * Math.PI * 2);
  const extension = 0.55 + 0.45 * wave;
  const curl = 1 - wave;
  const points = localHand([0.55 + 0.35 * extension, extension, extension, extension, extension]);
  const sway = Math.sin(now / 1_500) * 0.006;

  const metres = points.map(([x, y]) => ({
    x: x * 0.0016 + sway,
    y: y * 0.0016,
    // The palm plane is turned a little towards the camera, and a half-closed
    // hand brings the fingertips closer, so the projection has real depth.
    z: 0.45 - x * 0.0007 - Math.max(0, -y) * 0.00025 - curl * 0.012,
  }));

  const kpts2d: Keypoint[] = metres.map((point) => [
    WIDTH / 2 + (point.x / point.z) * MOCK_FOCAL_PX,
    HEIGHT / 2 + (point.y / point.z) * MOCK_FOCAL_PX,
    0.96,
  ]);
  const kpts3d = metres.map((point, index) =>
    index === 20 && seq % 5 === 0 ? null : ([point.x, point.y, point.z] as Point3D),
  );

  const us = kpts2d.map((point) => point[0]);
  const vs = kpts2d.map((point) => point[1]);
  const drift = Math.sin(now / 1_300);

  return {
    seq,
    t: now / 1000,
    width: WIDTH,
    height: HEIGHT,
    hands: [
      {
        score: 0.94,
        box: [Math.min(...us), Math.min(...vs), Math.max(...us), Math.max(...vs)],
        kpts2d,
        kpts3d,
      },
    ],
    timing: {
      dpu_ms: Math.round((17.8 + drift * 0.4) * 10) / 10,
      total_ms: Math.round((61.2 + drift * 2.5) * 10) / 10,
    },
  };
}
