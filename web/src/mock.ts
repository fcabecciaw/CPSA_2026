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
 * Query parameters, used for deterministic screenshots:
 *   `objects=0`  leaves the showcase pane on its invitation (nothing in view);
 *   `video=0`    leaves the camera pane on its placeholder.
 */

import type { SocketLike } from "./client";
import {
  type ServerMessage,
  type ShowcaseObject,
  type ShowcaseStatePayload,
  type StatusPayload,
} from "./protocol";
import type { VideoSource } from "./video";

const PARAMS = new URLSearchParams(window.location.search);

const WIDTH = 640;
const HEIGHT = 480;

export const MOCK_BADGE = "Dati di esempio (mock): nessun server connesso.";

/** F-S2: `showcase.state` is published at most at 10 Hz. */
const SHOWCASE_TICK_MS = 100;
/** F-K1: `status` at 1 Hz. */
const STATUS_TICK_MS = 1_000;

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
  const startedAt = Date.now();

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

  function status(): StatusPayload {
    const drift = Math.sin(Date.now() / 900);
    return {
      state: "running",
      camera: "ok",
      model: "yolox_nano",
      fps: round(59.6 + drift * 0.5, 1),
      dpu_ms: round(4.1 + drift * 0.2, 1),
      errors: [],
      uptime_s: round(125.4 + (Date.now() - startedAt) / 1000, 1),
    };
  }

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
      data?: { id?: unknown };
    };
    const refSeq = typeof envelope.seq === "number" ? envelope.seq : null;

    if (envelope.type === "ping") {
      emit(message("pong", { id: envelope.data?.id ?? null }));
      return;
    }
    // F-K1: `ping` is the only command the protocol accepts.
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
    every(SHOWCASE_TICK_MS, () => emit(message("showcase.state", showcaseState(Date.now()))));
    every(STATUS_TICK_MS, () => emit(message("status", status())));
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

/**
 * Dev-only stand-in for `/video.mjpeg`: the RGB room of the showcase, with the
 * boxes and Italian labels the board draws on the stream. Clearly a mock: the
 * dashboard shows the mock badge while it runs.
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
    drawShowcaseFrame(context, Date.now());
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
