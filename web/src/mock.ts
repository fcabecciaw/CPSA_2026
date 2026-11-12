/**
 * Dev-only WebSocket stand-in (`/?mock=1`).
 *
 * It replays the protocol v1 `hello` and `status` examples and synthesises the
 * two hand streams, so the views can be built and reviewed without the board.
 * The module is imported with a dynamic `import()` behind `import.meta.env.DEV`,
 * so Vite removes it from the production bundle.
 *
 * Query parameters, used for deterministic screenshots:
 *   `mode=hologram|rps|showcase|idle`  mode the mock starts in (default
 *   `hologram`);
 *   `phase=waiting|ready|countdown|shoot|result`  freezes the RPS round;
 *   `count=3|2|1` and `outcome=win|lose|draw|unclear`  refine that freeze;
 *   `objects=0`  leaves the showcase pane on its invitation (nothing in view);
 *   `video=0`  leaves the camera pane on its placeholder.
 */

import type { SocketLike } from "./client";
import { HAND21_BONES } from "./hologram";
import {
  MODES,
  type HologramHandsPayload,
  type Keypoint,
  type Mode,
  type Point3D,
  type RpsGesture,
  type RpsMove,
  type RpsOutcome,
  type RpsPhase,
  type RpsStatePayload,
  type ServerMessage,
  type ShowcaseObject,
  type ShowcaseStatePayload,
  type StatusPayload,
} from "./protocol";
import type { VideoSource } from "./video";

const PARAMS = new URLSearchParams(window.location.search);

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

function requestedMode(): Mode {
  const requested = PARAMS.get("mode") ?? "";
  return requested === "rps" || requested === "showcase" || requested === "idle"
    ? requested
    : "hologram";
}

const INITIAL_MODE: Mode = requestedMode();

/** The mode the mock socket runs, read by the fake camera picture too. */
let mockMode: Mode = INITIAL_MODE;

export const MOCK_BADGE = "Dati di esempio (mock): nessun server connesso.";

/** Sasso-carta-forbice timings, the same shape as the real state machine. */
const RPS_READY_MS = 600;
const RPS_STEP_MS = 700;
const RPS_SHOOT_MS = 400;
const RPS_RESULT_MS = 3_000;
const RPS_TICK_MS = 100;
/** F-S2: `showcase.state` is published at most at 10 Hz. */
const SHOWCASE_TICK_MS = 100;

/** The mock plays this cycle, so a screenshot covers win, lose and draw. */
const RPS_ROUNDS: readonly { player: RpsMove; board: RpsMove; outcome: RpsOutcome }[] = [
  { player: "paper", board: "rock", outcome: "win" },
  { player: "rock", board: "paper", outcome: "lose" },
  { player: "scissors", board: "scissors", outcome: "draw" },
];

/** One object of the mock showcase scene: where it stands and how sure it is. */
interface SceneObject {
  label: string;
  classId: number;
  box: readonly [number, number, number, number];
  score: number;
}

/**
 * The mock scene: two people behind a table with a bottle, a cup, a book and a
 * phone. The same coordinates drive the fake RGB picture and the detection
 * boxes, so the boxes sit on the objects they name.
 */
const SHOWCASE_SCENE: readonly SceneObject[] = [
  { label: "persona", classId: 0, box: [48, 30, 236, 424], score: 0.94 },
  { label: "persona", classId: 0, box: [252, 116, 372, 424], score: 0.88 },
  { label: "bottiglia", classId: 39, box: [520, 256, 582, 404], score: 0.82 },
  { label: "tazza", classId: 41, box: [428, 344, 502, 412], score: 0.78 },
  { label: "libro", classId: 73, box: [168, 376, 288, 418], score: 0.64 },
  { label: "cellulare", classId: 67, box: [330, 384, 372, 418], score: 0.57 },
];

const FORCED_PHASE = phaseParam();
const FORCED_COUNT = countParam();
const FORCED_OUTCOME = outcomeParam();

let sequence = 0;
let monotonic = 2000;

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
  let startedAt = Date.now();
  let hologramSeq = 0;

  let rpsPhase: RpsPhase = FORCED_PHASE ?? "waiting";
  let rpsSince = Date.now();
  let rpsCount = FORCED_COUNT ?? 3;
  let rpsRound = 0;
  let rpsPlayer: RpsGesture | null = null;
  let rpsBoard: RpsMove | null = null;
  let rpsOutcome: RpsOutcome | null = null;
  const rpsScore = { player: 0, board: 0, draws: 0 };

  if (FORCED_PHASE === "shoot") {
    rpsPlayer = "paper";
  }
  if (FORCED_PHASE === "result") {
    const round: { player: RpsMove; board: RpsMove; outcome: RpsOutcome } = {
      player: "paper",
      board: "rock",
      outcome: FORCED_OUTCOME ?? "win",
    };
    rpsPlayer = round.player;
    rpsBoard = round.board;
    rpsOutcome = round.outcome;
    award(rpsOutcome);
  }

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

  function sendError(code: "busy" | "bad_command", text: string, refSeq: number | null): void {
    emit(message("error", { code, message: text, ref_seq: refSeq }));
  }

  function sendStatus(): void {
    const status: StatusPayload = fixture("status").data;
    status.mode = mode;
    const camera = mode === "hologram" || mode === "rps" || mode === "showcase";
    status.mode_state = switching ? "starting" : camera ? "running" : "stopped";
    const drift = Math.sin(Date.now() / 900);
    status.perception.camera = camera ? "ok" : null;
    if (mode === "showcase") {
      status.perception.fps = 29.7 + drift * 0.6;
      status.perception.dpu_ms = 4.1 + drift * 0.2;
      status.perception.model = "yolox_nano";
    } else {
      status.perception.fps = camera ? 29.6 + drift * 0.8 : null;
      status.perception.dpu_ms = camera ? 17.8 + drift * 0.3 : null;
      status.perception.model = camera ? "pose_hand21_int8" : null;
    }
    // The streamed modes publish no F13 timing block: the dashboard reads fps
    // and latency from the streams themselves.
    status.perception.timing = null;
    status.imu = "n/a";
    status.errors = [];
    status.uptime_s = 125.4 + (Date.now() - startedAt) / 1000;
    emit(message("status", status));
  }

  function sendHologram(): void {
    if (mode !== "hologram") {
      return;
    }
    hologramSeq += 1;
    emit(message("hologram.hands", hologramFrame(hologramSeq)));
  }

  function sendShowcase(): void {
    if (mode !== "showcase") {
      return;
    }
    emit(message("showcase.state", showcaseState(Date.now())));
  }

  function tickRps(): void {
    if (mode !== "rps") {
      return;
    }
    const now = Date.now();
    if (FORCED_PHASE === null) {
      advanceRps(now);
    }
    emit(message("rps.state", rpsState(now)));
  }

  function advanceRps(now: number): void {
    const elapsed = now - rpsSince;
    if (rpsPhase === "waiting" && elapsed >= RPS_READY_MS) {
      enter("ready", now);
    } else if (rpsPhase === "ready" && elapsed >= RPS_READY_MS) {
      rpsCount = 3;
      enter("countdown", now);
    } else if (rpsPhase === "countdown" && elapsed >= RPS_STEP_MS) {
      if (rpsCount > 1) {
        rpsCount -= 1;
        rpsSince = now;
      } else {
        enter("shoot", now);
      }
    } else if (rpsPhase === "shoot" && elapsed >= RPS_SHOOT_MS) {
      const round = RPS_ROUNDS[rpsRound % RPS_ROUNDS.length] ?? RPS_ROUNDS[0]!;
      rpsRound += 1;
      rpsPlayer = round.player;
      rpsBoard = round.board;
      rpsOutcome = round.outcome;
      award(round.outcome);
      enter("result", now);
    } else if (rpsPhase === "result" && elapsed >= RPS_RESULT_MS) {
      rpsPlayer = null;
      rpsBoard = null;
      rpsOutcome = null;
      enter("ready", now);
    }
  }

  function enter(phase: RpsPhase, now: number): void {
    rpsPhase = phase;
    rpsSince = now;
  }

  function award(outcome: RpsOutcome | null): void {
    if (outcome === "win") {
      rpsScore.player += 1;
    } else if (outcome === "lose") {
      rpsScore.board += 1;
    } else if (outcome === "draw") {
      rpsScore.draws += 1;
    }
  }

  function rpsState(now: number): RpsStatePayload {
    const gesture: RpsGesture =
      rpsPhase === "countdown" || rpsPhase === "waiting" || rpsPhase === "ready"
        ? "none"
        : (rpsPlayer ?? "none");
    const hand = rpsPhase === "waiting" ? null : { kpts2d: rpsKeypoints(gesture, now) };
    return {
      phase: rpsPhase,
      count: rpsPhase === "countdown" ? rpsCount : null,
      player: rpsPhase === "result" ? rpsPlayer : null,
      player_p: rpsPhase === "result" ? 0.87 : null,
      board: rpsPhase === "result" ? rpsBoard : null,
      outcome: rpsPhase === "result" ? rpsOutcome : null,
      score: { ...rpsScore },
      hand,
    };
  }

  /**
   * The detections of the mock scene at `now`: boxes drift a pixel or two and
   * the scores breathe, so the panel shows change without a camera.
   */
  function showcaseState(now: number): ShowcaseStatePayload {
    const objects = showcaseObjects(now);
    const grouped = new Map<string, number>();
    for (const object of objects) {
      grouped.set(object.label, (grouped.get(object.label) ?? 0) + 1);
    }
    const counts = [...grouped.entries()]
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count || compareText(a.label, b.label));
    const drift = Math.sin(now / 900);
    return {
      model: "yolox_nano",
      width: WIDTH,
      height: HEIGHT,
      fps: round(29.7 + drift * 0.6, 1),
      dpu_ms: round(4.1 + drift * 0.2, 1),
      total_ms: round(21.0 + drift * 1.4, 1),
      objects,
      counts,
    };
  }

  function handleClient(raw: string): void {
    let command: unknown;
    try {
      command = JSON.parse(raw);
    } catch {
      sendError("bad_command", "comando non JSON", null);
      return;
    }
    const envelope = command as {
      type?: unknown;
      seq?: unknown;
      data?: { id?: unknown; mode?: unknown };
    };
    const refSeq = typeof envelope.seq === "number" ? envelope.seq : null;

    if (envelope.type === "ping") {
      emit(message("pong", { id: envelope.data?.id ?? null }));
      return;
    }
    if (envelope.type !== "mode.set") {
      return;
    }

    const requested = envelope.data?.mode;
    if (
      typeof requested !== "string" ||
      !(MODES as readonly string[]).includes(requested)
    ) {
      sendError("bad_command", "modalità sconosciuta", refSeq);
      return;
    }
    const next = requested as Mode;
    if (next === mode) {
      emit(message("mode.progress", { mode: next, step: "ready", error: null }));
      return;
    }

    switching = true;
    mode = next;
    mockMode = next;
    emit(message("mode.progress", { mode: next, step: "starting", error: null }));
    sendStatus();
    later(350, () => {
      switching = false;
      emit(message("mode.progress", { mode: next, step: "ready", error: null }));
      if (next === "rps") {
        rpsPhase = "waiting";
        rpsSince = Date.now();
        rpsPlayer = null;
        rpsBoard = null;
        rpsOutcome = null;
      }
      sendStatus();
    });
  }

  later(10, () => {
    socket.onopen?.();
    emit(message("hello", fixture("hello").data));
    sendStatus();
    every(33, sendHologram);
    every(RPS_TICK_MS, tickRps);
    every(SHOWCASE_TICK_MS, sendShowcase);
    every(2000, sendStatus);
  });

  return socket;
}

/** Detections of the mock scene, score-descending as the server sends them. */
function showcaseObjects(now: number): ShowcaseObject[] {
  if (PARAMS.get("objects") === "0") {
    return [];
  }
  return SHOWCASE_SCENE.map((scene, index) => {
    const sway = Math.sin(now / 1_100 + index * 0.9) * 1.5;
    const breath = Math.sin(now / 830 + index * 1.7) * 0.02;
    const [x1, y1, x2, y2] = scene.box;
    const score = Math.min(0.99, Math.max(0.35, scene.score + breath));
    return {
      label: scene.label,
      class_id: scene.classId,
      score: round(score, 2),
      box: [x1 + sway, y1, x2 + sway, y2] as [number, number, number, number],
    };
  }).sort((a, b) => b.score - a.score);
}

function round(value: number, digits: number): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** IR pixels of the mock hand, as the classifier would see it. */
function rpsKeypoints(gesture: RpsGesture, nowMs: number): Keypoint[] {
  const sway = Math.sin(nowMs / 380) * 3;
  const scale = 3;
  return localHand(GESTURE_EXTENSION[gesture]).map(([x, y]) => [
    WIDTH / 2 + x * scale + sway,
    HEIGHT * 0.64 + y * scale,
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

  const focal = 500;
  const kpts2d: Keypoint[] = metres.map((point) => [
    WIDTH / 2 + (point.x / point.z) * focal,
    HEIGHT / 2 + (point.y / point.z) * focal,
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

function phaseParam(): RpsPhase | null {
  const value = PARAMS.get("phase");
  const phases: readonly RpsPhase[] = ["waiting", "ready", "countdown", "shoot", "result"];
  return value !== null && (phases as readonly string[]).includes(value)
    ? (value as RpsPhase)
    : null;
}

function countParam(): number | null {
  const value = Number(PARAMS.get("count"));
  return value === 3 || value === 2 || value === 1 ? value : null;
}

function outcomeParam(): RpsOutcome | null {
  const value = PARAMS.get("outcome");
  const outcomes: readonly RpsOutcome[] = ["win", "lose", "draw", "unclear"];
  return value !== null && (outcomes as readonly string[]).includes(value)
    ? (value as RpsOutcome)
    : null;
}

/**
 * Dev-only stand-in for `/video.mjpeg`: the IR-looking hand of the two hand
 * demos, and the RGB room of the showcase, so the panes can be reviewed without
 * a camera. Clearly a mock: the dashboard shows the mock badge while it runs.
 */
export function createMockVideo(image: HTMLImageElement, note: HTMLElement): VideoSource {
  if (PARAMS.get("video") === "0") {
    // Same pane, no picture: the placeholder stays, for comparing the layout.
    return { start: () => undefined, stop: () => undefined };
  }

  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const context = canvas.getContext("2d");
  let timer: number | null = null;

  const draw = (): void => {
    if (context === null) {
      return;
    }
    const now = Date.now();
    if (mockMode === "showcase") {
      drawShowcaseFrame(context, now);
    } else {
      drawIrFrame(context, now);
    }
    image.hidden = false;
    note.hidden = true;
    image.src = canvas.toDataURL("image/jpeg", 0.72);
  };

  return {
    start(): void {
      if (timer !== null) {
        return;
      }
      draw();
      timer = window.setInterval(draw, 125);
    },
    stop(): void {
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
      image.removeAttribute("src");
      image.hidden = true;
      note.hidden = false;
    },
  };
}


/** The IR-looking hand the two hand demos describe, with its bones lit. */
function drawIrFrame(context: CanvasRenderingContext2D, now: number): void {
  const background = context.createRadialGradient(
    WIDTH * 0.5,
    HEIGHT * 0.4,
    40,
    WIDTH * 0.5,
    HEIGHT * 0.5,
    WIDTH * 0.8,
  );
  background.addColorStop(0, "#2a2f33");
  background.addColorStop(0.55, "#191d21");
  background.addColorStop(1, "#0b0e11");
  context.fillStyle = background;
  context.fillRect(0, 0, WIDTH, HEIGHT);

  // Faint table edge, a little brighter than the wall behind it.
  context.fillStyle = "rgba(255, 255, 255, 0.04)";
  context.fillRect(0, HEIGHT * 0.72, WIDTH, HEIGHT * 0.28);

  const points = rpsKeypoints("paper", now);
  const jitter = Math.sin(now / 210) * 1.2;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.shadowColor = "rgba(226, 238, 244, 0.75)";
  context.shadowBlur = 22;
  context.strokeStyle = "#b9c4ca";
  context.lineWidth = 34;
  for (const [first, second] of HAND21_BONES) {
    const from = points[first];
    const to = points[second];
    if (from === undefined || to === undefined) {
      continue;
    }
    context.beginPath();
    context.moveTo(from[0] + jitter, from[1]);
    context.lineTo(to[0] + jitter, to[1]);
    context.stroke();
  }

  const palm = [0, 1, 5, 9, 13, 17]
    .map((index) => points[index])
    .filter((point): point is Keypoint => point !== undefined);
  if (palm.length >= 3) {
    context.fillStyle = "#b9c4ca";
    context.beginPath();
    const first = palm[0]!;
    context.moveTo(first[0], first[1]);
    for (const point of palm.slice(1)) {
      context.lineTo(point[0], point[1]);
    }
    context.closePath();
    context.fill();
  }
  context.shadowBlur = 0;
}

/**
 * The RGB room of the showcase mock: the scene the detections name, with the
 * same boxes and Italian labels the board draws on the stream.
 */
function drawShowcaseFrame(context: CanvasRenderingContext2D, now: number): void {
  const background = context.createLinearGradient(0, 0, 0, HEIGHT);
  background.addColorStop(0, "#242b31");
  background.addColorStop(0.62, "#171c20");
  background.addColorStop(1, "#0c0f11");
  context.fillStyle = background;
  context.fillRect(0, 0, WIDTH, HEIGHT);

  // Two people standing behind the table, face on.
  drawPerson(context, 140, 82, 34, "#6c7883");
  drawPerson(context, 310, 166, 26, "#5f6a74");

  // The table: a lit top surface and a darker front face.
  context.fillStyle = "#35312c";
  context.fillRect(0, 352, WIDTH, 68);
  context.fillStyle = "#211d19";
  context.fillRect(0, 420, WIDTH, 60);
  context.fillStyle = "rgba(255, 255, 255, 0.07)";
  context.fillRect(0, 352, WIDTH, 2);

  drawBottle(context);
  drawCup(context);
  drawBook(context);
  drawPhone(context);
  drawShowcaseOverlay(context, showcaseObjects(now));
}

function drawPerson(
  context: CanvasRenderingContext2D,
  x: number,
  headY: number,
  headRadius: number,
  color: string,
): void {
  context.fillStyle = color;
  context.beginPath();
  context.arc(x, headY, headRadius, 0, Math.PI * 2);
  context.fill();

  const shoulderY = headY + headRadius + 6;
  const halfWidth = headRadius * 2.2;
  context.beginPath();
  context.moveTo(x - halfWidth * 0.55, shoulderY);
  context.quadraticCurveTo(x - halfWidth, shoulderY + 34, x - halfWidth, HEIGHT);
  context.lineTo(x + halfWidth, HEIGHT);
  context.quadraticCurveTo(x + halfWidth, shoulderY + 34, x + halfWidth * 0.55, shoulderY);
  context.closePath();
  context.fill();
}

function drawBottle(context: CanvasRenderingContext2D): void {
  // Box [520, 256, 582, 404]: neck above the body, cap on top.
  context.fillStyle = "#5b7d6a";
  context.fillRect(528, 296, 46, 108);
  context.fillStyle = "#4b6a59";
  context.fillRect(539, 256, 25, 40);
  context.fillStyle = "#2f3b34";
  context.fillRect(537, 252, 29, 8);
}

function drawCup(context: CanvasRenderingContext2D): void {
  // Box [428, 344, 502, 412]: a mug with its handle on the right.
  context.fillStyle = "#cfd6da";
  context.fillRect(434, 348, 56, 60);
  context.beginPath();
  context.strokeStyle = "#cfd6da";
  context.lineWidth = 6;
  context.arc(496, 378, 14, -Math.PI / 2, Math.PI / 2);
  context.stroke();
  context.fillStyle = "#e6ebee";
  context.fillRect(434, 348, 56, 6);
}

function drawBook(context: CanvasRenderingContext2D): void {
  // Box [168, 376, 288, 418]: a closed book lying flat.
  context.fillStyle = "#8a5f4a";
  context.fillRect(172, 380, 112, 34);
  context.fillStyle = "rgba(255, 255, 255, 0.18)";
  context.fillRect(176, 384, 104, 3);
  context.fillStyle = "#e8e1d6";
  context.fillRect(276, 382, 8, 30);
}

function drawPhone(context: CanvasRenderingContext2D): void {
  // Box [330, 384, 372, 418]: a phone on the table, screen up.
  context.fillStyle = "#1b1f24";
  context.fillRect(333, 388, 36, 26);
  context.fillStyle = "#3b6a86";
  context.fillRect(336, 391, 30, 20);
}

/** The board's overlay: a 2 px box and a label chip for each detection. */
function drawShowcaseOverlay(
  context: CanvasRenderingContext2D,
  objects: readonly ShowcaseObject[],
): void {
  context.lineWidth = 2;
  context.strokeStyle = "rgba(79, 214, 255, 0.92)";
  context.font = "600 13px 'Ubuntu Mono', ui-monospace, 'DejaVu Sans Mono', monospace";
  context.textBaseline = "alphabetic";
  for (const object of objects) {
    const [x1, y1, x2, y2] = object.box;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const text = `${object.label} ${Math.round(object.score * 100)}%`;
    const chipWidth = context.measureText(text).width + 10;
    const chipHeight = 19;
    const chipY = y1 >= chipHeight + 6 ? y1 - chipHeight - 4 : y1 + 4;
    context.fillStyle = "rgba(4, 8, 15, 0.74)";
    context.fillRect(x1 - 1, chipY, chipWidth, chipHeight);
    context.fillStyle = "#dff6ff";
    context.fillText(text, x1 + 4, chipY + 14);
  }
}
