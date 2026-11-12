/**
 * Dev-only WebSocket stand-in (`/?mock=1`).
 *
 * It speaks the F-K1 protocol and nothing else: `hello` on connect, `status`
 * at 1 Hz, `showcase.state` at 10 Hz and a `pong` for every `ping`; any other
 * command gets `error` `bad_command`, exactly as the server does. It also
 * synthesises the RGB picture the detections name, so the dashboard can be
 * built and reviewed without the board. The module is imported with a dynamic
 * `import()` behind `import.meta.env.DEV`, so Vite removes it from the
 * production bundle.
 *
 * F-E1 `model.set` is accepted and pins the published model (`yolox` or
 * `movenet`); without it the scripted timeline alternates the two.
 *
 * Query parameters, used for deterministic screenshots:
 *   `at=<s>`     fast-forwards the scripted scenario by that many seconds;
 *   `objects=0`  leaves the showcase pane on its invitation (nothing in view);
 *   `video=0`    leaves the camera pane on its placeholder.
 */

import type { SocketLike } from "./client";
import type { FaceCue } from "./cognitive/facecues";
import {
  type CalibrationValues,
  type ModelName,
  type PoseStatePayload,
  type ServerMessage,
  type ShowcaseObject,
  type ShowcaseStatePayload,
  type StatusPayload,
} from "./protocol";
import type { VideoSource } from "./video";

const PARAMS = new URLSearchParams(window.location.search);

const WIDTH = 640;
const HEIGHT = 480;

/**
 * One clock for the scripted scenario, shared by the socket and the fake
 * picture. `?at=<seconds>` fast-forwards it, for deterministic screenshots.
 */
const AT_SECONDS = Number(PARAMS.get("at") ?? "0");
const STARTED_AT = Date.now() - (Number.isFinite(AT_SECONDS) ? AT_SECONDS : 0) * 1_000;

/** F-E1 `model.set`: when the dashboard asks, the mock stops alternating. */
let requestedModel: ModelName | null = PARAMS.get("model") === "movenet"
  ? "movenet"
  : PARAMS.get("model") === "yolox"
    ? "yolox"
    : null;

/**
 * The board runs one model at a time: a swap reports `switching`, publishes no
 * frames while it unloads and loads, and takes about half a second.
 */
let switchingModel: ModelName | null = null;

export function scenarioSeconds(now = Date.now()): number {
  return (now - STARTED_AT) / 1_000;
}

export const MOCK_BADGE = "Dati di esempio (mock): nessun server connesso.";

/** F-S2: `showcase.state` is published at most at 10 Hz. */
const SHOWCASE_TICK_MS = 100;
/** F-E1: `pose.state` is published at most at 30 Hz. */
const POSE_TICK_MS = 33;
/** F-K1: `status` at 1 Hz. */
const STATUS_TICK_MS = 1_000;

/** Scripted scenario (seconds), used by the screenshots and the demo. */
export const SCRIPT = {
  personAt: 5,
  phoneAt: 12,
  cupAt: 19,
  groupAt: 25,
  leaveAt: 34,
  returnAt: 68,
  movenetAt: 76,
  armsUpAt: 80,
  waveAt: 84,
  endAt: 92,
  loopAt: 96,
} as const;

/** One object of the mock showcase scene: where it stands and how sure it is. */
interface SceneObject {
  label: string;
  classId: number;
  box: readonly [number, number, number, number];
  score: number;
  /** The picture of this object, so boxes and pixels stay together. */
  draw: (context: CanvasRenderingContext2D) => void;
}

function sceneObject(
  label: string,
  classId: number,
  box: readonly [number, number, number, number],
  score: number,
  draw: (context: CanvasRenderingContext2D) => void,
): SceneObject {
  return { label, classId, box, score, draw };
}

const PERSON = sceneObject("persona", 0, [48, 30, 236, 424], 0.94, (context) => {
  drawPerson(context, 140, 82, 34, "#6c7883");
});
const PERSON_2 = sceneObject("persona", 0, [252, 116, 372, 424], 0.88, (context) => {
  drawPerson(context, 310, 166, 26, "#5f6a74");
});
const PERSON_3 = sceneObject("persona", 0, [408, 156, 516, 424], 0.9, (context) => {
  drawPerson(context, 460, 200, 22, "#57626c");
});
const PHONE = sceneObject("cellulare", 67, [330, 384, 372, 418], 0.82, drawPhone);
const CUP = sceneObject("tazza", 41, [428, 344, 502, 412], 0.78, drawCup);
const BOTTLE = sceneObject("bottiglia", 39, [520, 256, 582, 404], 0.8, drawBottle);

/**
 * The scripted scenario: empty → one person → phone → cup → three people →
 * leave → idle → return → MoveNet arms up and wave. The same timeline drives
 * the detections, the fake picture and `pose.state`.
 */
export function scriptedObjects(seconds: number): SceneObject[] {
  const t = ((seconds % SCRIPT.loopAt) + SCRIPT.loopAt) % SCRIPT.loopAt;
  if (PARAMS.get("objects") === "0") {
    return [];
  }
  if (t < SCRIPT.personAt || t >= SCRIPT.leaveAt) {
    return t >= SCRIPT.returnAt && t < SCRIPT.movenetAt ? [PERSON] : [];
  }
  if (t >= SCRIPT.groupAt) {
    return [PERSON, PERSON_2, PERSON_3, PHONE, CUP, BOTTLE];
  }
  const objects: SceneObject[] = [PERSON];
  if (t >= SCRIPT.phoneAt) {
    objects.push(PHONE);
  }
  if (t >= SCRIPT.cupAt) {
    objects.push(CUP, BOTTLE);
  }
  return objects;
}

/** Is the mock streaming the MoveNet segment? */
export function scriptedMovenet(seconds: number): boolean {
  const t = ((seconds % SCRIPT.loopAt) + SCRIPT.loopAt) % SCRIPT.loopAt;
  return t >= SCRIPT.movenetAt && t < SCRIPT.endAt;
}

/**
 * Frames of the MoveNet segment: arms up for four seconds, then the wave.
 * COCO-17 order, source pixels of the 640×480 picture.
 */
export function scriptedKeypoints(seconds: number): Array<[number, number, number]> {
  const t = ((seconds % SCRIPT.loopAt) + SCRIPT.loopAt) % SCRIPT.loopAt;
  const waving = t >= SCRIPT.waveAt;
  const waveX = 424 + Math.sin((t - SCRIPT.waveAt) * 7) * 26;
  const wristY = waving ? 44 : 58;
  const points: Array<[number, number, number]> = [
    [320, 82, 0.94], // nose
    [308, 76, 0.92],
    [332, 76, 0.92],
    [292, 84, 0.85],
    [348, 84, 0.85],
    [278, 142, 0.93], // shoulders
    [362, 142, 0.93],
    [252, 108, 0.9], // elbows: up, above the shoulders
    [388, 108, 0.9],
    [238, wristY, 0.88], // wrists: up
    [waving ? waveX : 402, wristY, 0.88],
    [286, 292, 0.9], // hips
    [354, 292, 0.9],
    [290, 372, 0.86],
    [350, 372, 0.86],
    [292, 442, 0.82],
    [348, 442, 0.82],
  ];
  return points.map(([x, y, score]) => [round(x, 1), round(y, 1), score]);
}

function poseState(seconds: number): PoseStatePayload {
  const keypoints = scriptedKeypoints(seconds);
  const score = keypoints.reduce((sum, point) => sum + point[2], 0) / keypoints.length;
  const drift = Math.sin(Date.now() / 900);
  return {
    model: "movenet",
    width: WIDTH,
    height: HEIGHT,
    fps: round(29.8 + drift * 0.3, 1),
    dpu_ms: round(7.4 + drift * 0.3, 1),
    total_ms: round(24.6 + drift * 0.9, 1),
    score: round(score, 2),
    keypoints,
  };
}



let sequence = 0;
let monotonic = 2000;

function message<T extends ServerMessage["type"]>(
  type: T,
  data: Extract<ServerMessage, { type: T }>["data"],
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
  const startedAt = STARTED_AT;
  /** The debug panel's calibration, as `config/showcase.yaml` ships it. */
  const yamlCalibration: CalibrationValues = {
    zoom: 1.3,
    zoom_center: [0.5, 0.5],
    gain: 200,
    exposure: 100,
    saturation: 110,
  };
  let calibration: CalibrationValues = { ...yamlCalibration };

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

  function activeMovenet(): boolean {
    return requestedModel === null ? scriptedMovenet(scenarioSeconds()) : requestedModel === "movenet";
  }

  function status(): StatusPayload {
    const drift = Math.sin(Date.now() / 900);
    const movenet = activeMovenet();
    return {
      state: switchingModel === null ? "running" : "switching",
      camera: "ok",
      model: movenet ? "movenet" : "yolox_nano",
      fps: round((movenet ? 29.8 : 59.6) + drift * 0.5, 1),
      dpu_ms: round((movenet ? 7.4 : 4.1) + drift * 0.2, 1),
      errors: [],
      uptime_s: round(125.4 + (Date.now() - startedAt) / 1000, 1),
      calibration: { ...calibration, zoom_center: [...calibration.zoom_center] },
    };
  }

  function showcaseState(now: number): ShowcaseStatePayload {
    const objects = showcaseObjects(scenarioSeconds(), now);
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
      fps: round(59.6 + drift * 0.5, 1),
      dpu_ms: round(4.1 + drift * 0.2, 1),
      total_ms: round(15.8 + drift * 0.7, 1),
      objects,
      counts,
    };
  }

  function handleClient(raw: string): void {
    let command: unknown;
    try {
      command = JSON.parse(raw);
    } catch {
      emit(message("error", { code: "bad_command", message: "comando non JSON", ref_seq: null }));
      return;
    }
    const envelope = command as {
      type?: unknown;
      seq?: unknown;
      data?: { id?: unknown; model?: unknown } & Partial<CalibrationValues>;
    };
    const refSeq = typeof envelope.seq === "number" ? envelope.seq : null;

    if (envelope.type === "ping") {
      emit(message("pong", { id: envelope.data?.id ?? null }));
      return;
    }
    if (envelope.type === "model.set") {
      const model = envelope.data?.model;
      if (model !== "yolox" && model !== "movenet") {
        emit(
          message("error", {
            code: "bad_command",
            message: `modello sconosciuto: ${String(model)}`,
            ref_seq: refSeq,
          }),
        );
        return;
      }
      // The board swaps its single model: no frames for ~0.5 s, then the new
      // view starts. Emitting `switching` lets the dashboard say so.
      switchingModel = model;
      emit(message("status", status()));
      later(500, () => {
        if (closed) {
          return;
        }
        requestedModel = model;
        switchingModel = null;
        emit(message("status", status()));
      });
      return;
    }
    if (envelope.type === "calibration.set") {
      // Trusts the dashboard's ranges: the board validates for real.
      calibration = { ...calibration, ...(envelope.data ?? {}) } as CalibrationValues;
      emit(message("status", status()));
      return;
    }
    if (envelope.type === "calibration.reset") {
      calibration = { ...yamlCalibration };
      emit(message("status", status()));
      return;
    }
    // Anything else is answered as the board does.
    emit(
      message("error", {
        code: "bad_command",
        message: `comando non supportato: ${String(envelope.type)}`,
        ref_seq: refSeq,
      }),
    );
  }

  later(10, () => {
    socket.onopen?.();
    emit(message("hello", { server_version: "mock", protocol_version: 1 }));
    emit(message("status", status()));
    // One clock: `pose.state` at 30 Hz during the MoveNet segment, otherwise
    // `showcase.state` at 10 Hz, as the board does.
    let lastShowcase = Number.NEGATIVE_INFINITY;
    let lastPose = Number.NEGATIVE_INFINITY;
    every(POSE_TICK_MS, () => {
      if (switchingModel !== null) {
        return;
      }
      const now = Date.now();
      const seconds = scenarioSeconds();
      if (activeMovenet()) {
        if (now - lastPose >= POSE_TICK_MS) {
          lastPose = now;
          emit(message("pose.state", poseState(seconds)));
        }
        return;
      }
      if (now - lastShowcase >= SHOWCASE_TICK_MS) {
        lastShowcase = now;
        emit(message("showcase.state", showcaseState(now)));
      }
    });
    every(STATUS_TICK_MS, () => emit(message("status", status())));
  });

  return socket;
}

/** Detections of the scripted scene, score-descending as the server sends them. */
function showcaseObjects(seconds: number, now: number): ShowcaseObject[] {
  return scriptedObjects(seconds)
    .map((scene, index) => {
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
    })
    .sort((a, b) => b.score - a.score);
}

function round(value: number, digits: number): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Dev-only stand-in for `/video.mjpeg`: the RGB room of the showcase, with the
 * boxes and Italian labels the board draws on the stream. Clearly a mock: the
 * dashboard shows the mock badge while it runs.
 */
/**
 * Scripted face cues for the mock (the drawn room has no real faces): the
 * MediaPipe tracker cannot detect anything in the fake picture, so the mock
 * plays the same smiles and nods a person would show.
 */
export function createMockFaceCues(onCue: (cue: FaceCue) => void): VideoSource {
  const schedule: Array<{ at: number; kind: FaceCue["kind"] }> = [
    { at: SCRIPT.personAt + 1.5, kind: "looking_at_me" },
    { at: SCRIPT.phoneAt + 1, kind: "mouth_open" },
    { at: SCRIPT.cupAt + 1, kind: "smile" },
    { at: SCRIPT.groupAt + 1, kind: "smile" },
    { at: SCRIPT.movenetAt + 1.5, kind: "nod" },
    { at: SCRIPT.waveAt + 2, kind: "head_shake" },
  ];
  let timer: number | null = null;
  const fired = new Set<string>();
  const tick = (): void => {
    const raw = scenarioSeconds() % SCRIPT.loopAt;
    const loop = Math.floor(scenarioSeconds() / SCRIPT.loopAt);
    for (const cue of schedule) {
      const key = `${loop}:${cue.at}`;
      if (raw >= cue.at && !fired.has(key)) {
        fired.add(key);
        onCue({
          kind: cue.kind,
          score: 0.8,
          faces: 1,
          at: Date.now(),
        });
      }
    }
    if (fired.size > 64) {
      fired.clear();
    }
  };
  return {
    start(): void {
      if (timer !== null) {
        return;
      }
      tick();
      timer = window.setInterval(tick, 200);
    },
    stop(): void {
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
    },
  };
}

export function createMockVideo(image: HTMLImageElement, note: HTMLElement): VideoSource {  if (PARAMS.get("video") === "0") {
    // Same pane, no picture: the placeholder stays, for comparing the layout.
    return { start: () => undefined, stop: () => undefined };
  }

  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const context = canvas.getContext("2d");
  let timer: number | null = null;
  let encoding = false;
  let lastUrl: string | null = null;

  // The picture is handed to the <img> as a blob URL: `toBlob` encodes off the
  // main thread, while `toDataURL` would stall the dashboard's frame for tens
  // of milliseconds every time the fake camera produces a frame.
  const draw = (): void => {
    if (context === null || encoding) {
      return;
    }
    drawShowcaseFrame(context, scenarioSeconds());
    image.hidden = false;
    note.hidden = true;
    encoding = true;
    canvas.toBlob(
      (blob) => {
        encoding = false;
        if (blob === null) {
          return;
        }
        const url = URL.createObjectURL(blob);
        image.src = url;
        if (lastUrl !== null) {
          URL.revokeObjectURL(lastUrl);
        }
        lastUrl = url;
      },
      "image/jpeg",
      0.7,
    );
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
      if (lastUrl !== null) {
        URL.revokeObjectURL(lastUrl);
        lastUrl = null;
      }
    },
  };
}

/**
 * The RGB room of the showcase mock: the scene the detections name, with the
 * same boxes and Italian labels the board draws on the stream.
 */
function drawShowcaseFrame(context: CanvasRenderingContext2D, seconds: number): void {
  const background = context.createLinearGradient(0, 0, 0, HEIGHT);
  background.addColorStop(0, "#242b31");
  background.addColorStop(0.62, "#171c20");
  background.addColorStop(1, "#0c0f11");
  context.fillStyle = background;
  context.fillRect(0, 0, WIDTH, HEIGHT);

  if (scriptedMovenet(seconds)) {
    drawTable(context);
    drawPoseFigure(context, scriptedKeypoints(seconds));
    return;
  }

  const objects = scriptedObjects(seconds);
  for (const object of objects) {
    if (object.label === "persona") {
      object.draw(context);
    }
  }

  drawTable(context);
  for (const object of objects) {
    if (object.label !== "persona") {
      object.draw(context);
    }
  }
  drawShowcaseOverlay(context, showcaseObjects(seconds, Date.now()));
}

function drawTable(context: CanvasRenderingContext2D): void {
  // The table: a lit top surface and a darker front face.
  context.fillStyle = "#35312c";
  context.fillRect(0, 352, WIDTH, 68);
  context.fillStyle = "#211d19";
  context.fillRect(0, 420, WIDTH, 60);
  context.fillStyle = "rgba(255, 255, 255, 0.07)";
  context.fillRect(0, 352, WIDTH, 2);
}

/** The MoveNet segment: the skeleton the board would draw over the person. */
function drawPoseFigure(
  context: CanvasRenderingContext2D,
  keypoints: ReadonlyArray<readonly [number, number, number]>,
): void {
  const edges: ReadonlyArray<readonly [number, number]> = [
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 6],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [0, 5],
    [0, 6],
  ];
  context.lineWidth = 5;
  context.lineCap = "round";
  context.strokeStyle = "rgba(79, 214, 255, 0.85)";
  for (const [a, b] of edges) {
    const from = keypoints[a];
    const to = keypoints[b];
    if (from === undefined || to === undefined || from[2] < 0.3 || to[2] < 0.3) {
      continue;
    }
    context.beginPath();
    context.moveTo(from[0], from[1]);
    context.lineTo(to[0], to[1]);
    context.stroke();
  }
  context.fillStyle = "#dff6ff";
  for (const [x, y, score] of keypoints) {
    if (score < 0.3) {
      continue;
    }
    context.beginPath();
    context.arc(x, y, 4.5, 0, Math.PI * 2);
    context.fill();
  }
  context.font = "600 15px 'Ubuntu Mono', ui-monospace, 'DejaVu Sans Mono', monospace";
  context.fillStyle = "rgba(79, 214, 255, 0.9)";
  context.fillText("movenet · postura", 20, 34);
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
