/**
 * Canvas skeleton for the HALFBUST_62 layout (FR-403).
 *
 * The bone list is transcribed from `src/cpsa/vision/layouts.py`:
 * COCO-17 body points 0..10, nine face anchors 11..19, then the 21-point
 * left hand 20..40 and right hand 41..61. Camera pixels are mirrored so the
 * display reads like a mirror for the person in front of the camera.
 */

import type { HandsPayload, Keypoint } from "./protocol";

export const LAYOUT_NAME = "HALFBUST_62";
export const EXPECTED_POINTS = 62;

/** F14 stream endpoint, same origin, served by the dashboard's aiohttp server. */
export const VIDEO_URL = "/video.mjpeg";

/** Copy shown while no stream is running (contract: F14 web consumer). */
export const VIDEO_PLACEHOLDER = "Video non disponibile";

/** Retry backoff of the `<img>` stream: 2 s, then doubling, capped at 10 s. */
const RETRY_BASE_MS = 2_000;
const RETRY_MAX_MS = 10_000;

/** Used only until the first frame reveals the real camera size. */
const FALLBACK_WIDTH = 640;
const FALLBACK_HEIGHT = 480;

type Edge = readonly [number, number];

const BODY_EDGES: readonly Edge[] = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 0],
  [6, 0],
];

const FACE_EDGES: readonly Edge[] = [
  [11, 13],
  [13, 12],
  [11, 16],
  [12, 17],
  [16, 17],
  [16, 18],
  [18, 17],
  [16, 19],
  [19, 17],
  [13, 19],
  [14, 15],
  [14, 1],
  [15, 2],
  [11, 3],
  [12, 4],
];

const HAND_EDGES: readonly Edge[] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
];

const LEFT_HAND_OFFSET = 20;
const RIGHT_HAND_OFFSET = 41;

const EDGES: readonly Edge[] = [
  ...BODY_EDGES,
  ...FACE_EDGES,
  ...HAND_EDGES.map(([a, b]): Edge => [a + LEFT_HAND_OFFSET, b + LEFT_HAND_OFFSET]),
  ...HAND_EDGES.map(([a, b]): Edge => [a + RIGHT_HAND_OFFSET, b + RIGHT_HAND_OFFSET]),
];

const COLORS = {
  background: "#070d17",
  body: "#8fd0ff",
  face: "#ffd479",
  hands: "#8ce3b0",
  subject: "#5c6f8c",
} as const;

const HIDDEN_ALPHA = 0;

export class SkeletonView {
  private readonly canvas: HTMLCanvasElement;
  private readonly wrap: HTMLElement;
  private readonly video: HTMLImageElement;
  private readonly videoNote: HTMLElement;
  private readonly empty: HTMLElement;
  private readonly meta: HTMLElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly observer: ResizeObserver | null;
  private readonly onWindowResize = (): void => this.scheduleDraw();
  private drawFrame: number | null = null;
  private metaKey: string | null = null;

  private hands: HandsPayload | null = null;

  /** The stream is wanted (medical mode, camera ok, no mock) and not stopped. */
  private videoWanted = false;
  /** A stream is attached and its frames paint under the canvas. */
  private videoLive = false;
  private videoAttempt = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly onVideoLoad = (): void => {
    if (!this.videoWanted) {
      return;
    }
    // A fresh frame is flowing: restart the backoff and drop the placeholder.
    this.videoAttempt = 0;
    this.videoNote.hidden = true;
    this.scheduleDraw();
  };

  private readonly onVideoError = (): void => {
    if (!this.videoWanted) {
      return;
    }
    this.videoAttempt += 1;
    const delay = Math.min(RETRY_BASE_MS * 2 ** (this.videoAttempt - 1), RETRY_MAX_MS);
    this.video.hidden = true;
    this.videoLive = false;
    this.videoNote.textContent = `${VIDEO_PLACEHOLDER} · nuovo tentativo tra ${Math.round(delay / 1000)} s`;
    this.videoNote.hidden = false;
    this.scheduleDraw();
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
    }
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      if (this.videoWanted) {
        this.attachVideo();
      }
    }, delay);
  };

  constructor(
    canvas: HTMLCanvasElement,
    empty: HTMLElement,
    meta: HTMLElement,
    video: HTMLImageElement,
    videoNote: HTMLElement,
  ) {
    const context = canvas.getContext("2d");
    if (context === null) {
      throw new Error("Canvas 2D non disponibile in questo browser");
    }
    this.canvas = canvas;
    this.wrap = canvas.parentElement ?? canvas;
    this.empty = empty;
    this.meta = meta;
    this.video = video;
    this.videoNote = videoNote;
    this.context = context;

    this.video.addEventListener("load", this.onVideoLoad);
    this.video.addEventListener("error", this.onVideoError);

    this.observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(() => this.scheduleDraw());
    this.observer?.observe(canvas.parentElement ?? canvas);
    window.addEventListener("resize", this.onWindowResize);

    this.updateMeta();
    this.draw();
  }

  setHands(hands: HandsPayload | null): void {
    this.hands = hands;
    this.updateMeta();
    this.draw();
  }

  /**
   * F14: show the camera stream under the skeleton. The caller decides
   * availability (medical mode with `perception.camera == "ok"`, never in the
   * mock); this view owns the retry backoff and the placeholder copy.
   */
  setVideo(available: boolean): void {
    if (available === this.videoWanted) {
      return;
    }
    this.videoWanted = available;
    if (available) {
      this.videoAttempt = 0;
      this.attachVideo();
    } else {
      this.detachVideo();
    }
  }

  private attachVideo(): void {
    this.videoLive = true;
    this.video.hidden = false;
    this.videoNote.hidden = true;
    // The retry counter only busts the cache between attempts; the stream
    // itself ignores the query string.
    this.video.src = this.videoAttempt === 0 ? VIDEO_URL : `${VIDEO_URL}?retry=${this.videoAttempt}`;
    this.draw();
  }

  private detachVideo(): void {
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.videoAttempt = 0;
    this.videoLive = false;
    this.video.removeAttribute("src");
    this.video.hidden = true;
    this.videoNote.textContent = VIDEO_PLACEHOLDER;
    this.videoNote.hidden = false;
    this.draw();
  }

  destroy(): void {
    this.observer?.disconnect();
    window.removeEventListener("resize", this.onWindowResize);
    this.video.removeEventListener("load", this.onVideoLoad);
    this.video.removeEventListener("error", this.onVideoError);
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    if (this.drawFrame !== null) {
      cancelAnimationFrame(this.drawFrame);
      this.drawFrame = null;
    }
  }

  /** Defer a redraw so a ResizeObserver callback never mutates layout in place. */
  private scheduleDraw(): void {
    if (this.drawFrame !== null) {
      return;
    }
    this.drawFrame = requestAnimationFrame(() => {
      this.drawFrame = null;
      this.draw();
    });
  }

  private updateMeta(): void {
    const hands = this.hands;

    if (hands === null) {
      if (this.metaKey === "none") {
        return;
      }
      this.metaKey = "none";
      this.showEmpty("In attesa di episodio…");
      this.meta.textContent = "Nessun frame ricevuto.";
      this.canvas.setAttribute("aria-label", "Scheletro non disponibile: in attesa di dati.");
      return;
    }

    if (hands.layout !== LAYOUT_NAME) {
      const key = `layout:${hands.layout}`;
      if (this.metaKey === key) {
        return;
      }
      this.metaKey = key;
      this.showEmpty(`Layout non supportato: ${hands.layout}`);
      this.meta.textContent = `Ricevuto ${hands.layout}; atteso ${LAYOUT_NAME}.`;
      this.canvas.setAttribute("aria-label", `Scheletro non disponibile: layout ${hands.layout}.`);
      return;
    }

    const visible = hands.keypoints.filter((point) => (point[2] ?? 0) > 0).length;
    const key = `ok:${hands.layout}:${hands.w}:${hands.h}:${visible}`;
    if (this.metaKey === key) {
      return;
    }
    this.metaKey = key;

    if (visible === 0) {
      this.showEmpty("In attesa di episodio…");
    } else {
      this.empty.hidden = true;
    }

    this.meta.textContent = `${hands.layout} · ${hands.w}×${hands.h} px · ${visible}/${EXPECTED_POINTS} punti visibili`;
    this.canvas.setAttribute(
      "aria-label",
      `Scheletro ${hands.layout} specchiato: ${visible} punti visibili su ${EXPECTED_POINTS}.`,
    );
  }

  private showEmpty(text: string): void {
    this.empty.hidden = false;
    this.empty.textContent = text;
  }

  private draw(): void {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) {
      return;
    }

    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(rect.width * ratio);
    const pixelHeight = Math.round(rect.height * ratio);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }

    const context = this.context;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);
    this.wrap.classList.toggle("has-video", this.videoLive);
    if (!this.videoLive) {
      // Over the stream the canvas must stay transparent, so the camera frame
      // under it is visible and only the bones are painted.
      context.fillStyle = COLORS.background;
      context.fillRect(0, 0, rect.width, rect.height);
    }

    const hands = this.hands;
    this.layoutVideo(hands, rect);
    if (hands === null || hands.layout !== LAYOUT_NAME || hands.keypoints.length === 0) {
      return;
    }

    // Over the camera stream the whole frame is shown (no subject zoom), so the
    // bones and the mirrored image share one letterbox and stay aligned.
    const view = this.videoLive ? fullView(hands.w, hands.h) : viewFor(hands);
    const { scale, offsetX, offsetY } = frameFor(view, rect);
    const mapX = (x: number): number => offsetX + (hands.w - x - view.x) * scale;
    const mapY = (y: number): number => offsetY + (y - view.y) * scale;
    const unit = Math.min(rect.width, rect.height);

    const box = hands.subject_box;
    if (box !== null) {
      const left = Math.min(mapX(box[0]), mapX(box[2]));
      const right = Math.max(mapX(box[0]), mapX(box[2]));
      const top = Math.min(mapY(box[1]), mapY(box[3]));
      const bottom = Math.max(mapY(box[1]), mapY(box[3]));
      context.strokeStyle = COLORS.subject;
      context.globalAlpha = 0.55;
      context.lineWidth = Math.max(1, unit * 0.0025);
      context.setLineDash([unit * 0.012, unit * 0.01]);
      context.strokeRect(left, top, right - left, bottom - top);
      context.setLineDash([]);
      context.globalAlpha = 1;
    }

    const keypoints = hands.keypoints;
    const lineWidth = Math.max(1.5, unit * 0.0035);

    for (const [a, b] of EDGES) {
      const first = keypoints[a];
      const second = keypoints[b];
      if (first === undefined || second === undefined) {
        continue;
      }
      const alpha = Math.min(alphaFor(first), alphaFor(second));
      if (alpha === HIDDEN_ALPHA) {
        continue;
      }
      context.globalAlpha = alpha;
      context.strokeStyle = colorFor(Math.max(a, b));
      context.lineWidth = lineWidth;
      context.beginPath();
      context.moveTo(mapX(first[0]), mapY(first[1]));
      context.lineTo(mapX(second[0]), mapY(second[1]));
      context.stroke();
    }

    for (let index = 0; index < keypoints.length; index += 1) {
      const point = keypoints[index];
      if (point === undefined) {
        continue;
      }
      const alpha = alphaFor(point);
      if (alpha === HIDDEN_ALPHA) {
        continue;
      }
      context.globalAlpha = alpha;
      context.fillStyle = colorFor(index);
      context.beginPath();
      context.arc(mapX(point[0]), mapY(point[1]), radiusFor(index, unit), 0, Math.PI * 2);
      context.fill();
    }

    context.globalAlpha = 1;
  }

  /**
   * Places the camera frame under the canvas with the exact mapping the bones
   * use: same letterbox (`frameFor`) and the same horizontal mirroring
   * (`scaleX(-1)` around the frame's own centre). Keypoints in camera pixels
   * therefore land on the person in the image. Without a pose yet, the whole
   * camera frame is used at the image's natural size.
   */
  private layoutVideo(hands: HandsPayload | null, rect: DOMRect): void {
    if (!this.videoLive) {
      return;
    }
    const pose = hands !== null && hands.layout === LAYOUT_NAME ? hands : null;
    const width =
      pose !== null
        ? pose.w
        : this.video.naturalWidth > 0
          ? this.video.naturalWidth
          : FALLBACK_WIDTH;
    const height =
      pose !== null
        ? pose.h
        : this.video.naturalHeight > 0
          ? this.video.naturalHeight
          : FALLBACK_HEIGHT;
    const { scale, offsetX, offsetY } = frameFor(fullView(width, height), rect);

    this.video.style.width = `${width * scale}px`;
    this.video.style.height = `${height * scale}px`;
    this.video.style.left = `${offsetX}px`;
    this.video.style.top = `${offsetY}px`;
  }
}

interface ViewWindow {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Frame {
  scale: number;
  offsetX: number;
  offsetY: number;
}

/** Letterbox the camera-pixel window inside the CSS box, centred. */
function frameFor(view: ViewWindow, rect: { width: number; height: number }): Frame {
  const scale = Math.min(rect.width / view.width, rect.height / view.height);
  return {
    scale,
    offsetX: (rect.width - view.width * scale) / 2,
    offsetY: (rect.height - view.height * scale) / 2,
  };
}

/**
 * Frame the subject box when the server sends one, so the skeleton fills a
 * booth screen instead of sitting in the middle of the 640×480 camera frame.
 * The mapping stays in camera pixels; this only chooses the visible window.
 */
function fullView(width: number, height: number): ViewWindow {
  return { x: 0, y: 0, width, height };
}

function viewFor(hands: HandsPayload): ViewWindow {
  const full: ViewWindow = { x: 0, y: 0, width: hands.w, height: hands.h };
  const box = hands.subject_box;
  if (box === null) {
    return full;
  }

  const boxWidth = Math.abs(box[2] - box[0]);
  const boxHeight = Math.abs(box[3] - box[1]);
  if (boxWidth < 1 || boxHeight < 1) {
    return full;
  }

  const width = Math.min(hands.w, Math.max(boxWidth * 1.1, hands.w * 0.25));
  const height = Math.min(hands.h, Math.max(boxHeight * 1.1, hands.h * 0.25));
  const centerX = (box[0] + box[2]) / 2;
  const centerY = (box[1] + box[3]) / 2;

  return {
    x: clamp(centerX - width / 2, 0, hands.w - width),
    y: clamp(centerY - height / 2, 0, hands.h - height),
    width,
    height,
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function alphaFor(point: Keypoint): number {
  const confidence = point[2];
  if (confidence >= 0.5) {
    return 1;
  }
  if (confidence >= 0.2) {
    return 0.45;
  }
  if (confidence > 0) {
    return 0.18;
  }
  return HIDDEN_ALPHA;
}

function colorFor(index: number): string {
  if (index >= LEFT_HAND_OFFSET) {
    return COLORS.hands;
  }
  if (index >= 11) {
    return COLORS.face;
  }
  return COLORS.body;
}

function radiusFor(index: number, unit: number): number {
  if (index >= LEFT_HAND_OFFSET) {
    return Math.max(1.5, unit * 0.004);
  }
  if (index >= 11) {
    return Math.max(1.8, unit * 0.0048);
  }
  return Math.max(2.2, unit * 0.006);
}
