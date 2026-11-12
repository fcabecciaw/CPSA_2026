/**
 * Sasso-carta-forbice view (T5): the hand keypoints, the countdown and the
 * result, as one booth-facing panel.
 *
 * The skeleton is canvas 2D, drawn from `rps.state.hand.kpts2d` and scaled up
 * so the booth can read the hand from a distance; the countdown and the result
 * are DOM text over the canvas.
 */

import { HAND21_BONES } from "./hologram";
import {
  RPS_COUNTDOWN_WORDS,
  RPS_GESTURE_LABEL,
  RPS_LABELS,
  RPS_OUTCOME_LABEL,
} from "./labels";
import type { Keypoint, RpsStatePayload } from "./protocol";

const COLORS = {
  background: "#050b14",
  bone: "rgba(126, 232, 255, 0.62)",
  joint: "#c8f6ff",
  glow: "rgba(56, 214, 255, 0.55)",
} as const;

/** Empty slot shown before a gesture is known. */
const NO_GESTURE = "—";

/** The hand spans this share of the pane's shorter side. */
const HAND_SHARE = 0.6;
/** The hand sits a little below centre, under the countdown and result. */
const HAND_CENTRE = 0.56;

/** How loudly the callout reads: a hint, a countdown numeral, or the result. */
type Emphasis = "calm" | "countdown" | "shoot" | "result";

export interface RpsElements {
  count: HTMLElement;
  player: HTMLElement;
  board: HTMLElement;
  duel: HTMLElement;
  canvas: HTMLCanvasElement;
  empty: HTMLElement;
}

export class RpsView {
  private readonly elements: RpsElements;
  private readonly context: CanvasRenderingContext2D;
  private readonly observer: ResizeObserver | null;

  private state: RpsStatePayload | null = null;
  private active = false;

  constructor(elements: RpsElements) {
    const context = elements.canvas.getContext("2d");
    if (context === null) {
      throw new Error("Canvas 2D non disponibile in questo browser");
    }
    this.elements = elements;
    this.context = context;

    this.observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => this.draw());
    this.observer?.observe(elements.canvas.parentElement ?? elements.canvas);
    window.addEventListener("resize", this.onWindowResize);

    this.render();
  }

  setState(state: RpsStatePayload | null): void {
    this.state = state;
    this.render();
  }

  /** Draws only while the panel is visible; the mode switch calls this. */
  setActive(active: boolean): void {
    this.active = active;
    if (active) {
      this.draw();
    }
  }

  destroy(): void {
    this.observer?.disconnect();
    window.removeEventListener("resize", this.onWindowResize);
  }

  private readonly onWindowResize = (): void => this.draw();

  private render(): void {
    const { count, player, board, duel, empty } = this.elements;
    const state = this.state;
    const view = describe(state);

    count.textContent = view.text;
    count.dataset.emphasis = view.emphasis;
    if (view.outcome === "none") {
      delete count.dataset.outcome;
    } else {
      count.dataset.outcome = view.outcome;
    }

    const gesturePlayer = state?.player ?? null;
    const gestureBoard = state?.board ?? null;
    player.textContent = `${RPS_LABELS.player}: ${
      gesturePlayer === null ? NO_GESTURE : RPS_GESTURE_LABEL[gesturePlayer]
    }`;
    board.textContent = `${RPS_LABELS.board}: ${
      gestureBoard === null ? NO_GESTURE : RPS_GESTURE_LABEL[gestureBoard]
    }`;
    duel.hidden = gesturePlayer === null && gestureBoard === null;

    const points = state?.hand?.kpts2d ?? [];
    empty.hidden = visibleAlpha(points) > 0;
    this.elements.canvas.setAttribute(
      "aria-label",
      `${RPS_LABELS.player} ${player.textContent}, ${RPS_LABELS.board} ${board.textContent}, ${view.text}`,
    );
    this.draw();
  }

  private draw(): void {
    const canvas = this.elements.canvas;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) {
      return;
    }

    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(rect.width * ratio);
    const pixelHeight = Math.round(rect.height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }

    const context = this.context;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);
    context.fillStyle = COLORS.background;
    context.fillRect(0, 0, rect.width, rect.height);

    const points = this.state?.hand?.kpts2d ?? [];
    const mapper = this.active ? handMapper(points, rect) : null;
    if (mapper === null) {
      return;
    }

    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = Math.max(2, mapper.span * 0.014);
    context.strokeStyle = COLORS.bone;
    for (const [first, second] of HAND21_BONES) {
      const from = points[first];
      const to = points[second];
      if (from === undefined || to === undefined) {
        continue;
      }
      const alpha = Math.min(pointAlpha(from), pointAlpha(to));
      if (alpha <= 0) {
        continue;
      }
      context.globalAlpha = alpha;
      context.beginPath();
      context.moveTo(mapper.x(from[0]), mapper.y(from[1]));
      context.lineTo(mapper.x(to[0]), mapper.y(to[1]));
      context.stroke();
    }

    context.globalAlpha = 1;
    context.fillStyle = COLORS.joint;
    context.shadowColor = COLORS.glow;
    context.shadowBlur = mapper.span * 0.05;
    const radius = Math.max(3, mapper.span * 0.022);
    for (const point of points) {
      const alpha = pointAlpha(point);
      if (alpha <= 0) {
        continue;
      }
      context.globalAlpha = alpha;
      context.beginPath();
      context.arc(mapper.x(point[0]), mapper.y(point[1]), radius, 0, Math.PI * 2);
      context.fill();
    }
    context.shadowBlur = 0;
    context.globalAlpha = 1;
  }
}

interface Callout {
  text: string;
  emphasis: Emphasis;
  outcome: "none" | "win" | "lose" | "draw" | "unclear";
}

/** Countdown word, result banner or invitation, and how large to show it. */
function describe(state: RpsStatePayload | null): Callout {
  if (state === null || state.phase === "waiting") {
    return { text: RPS_LABELS.waiting, emphasis: "calm", outcome: "none" };
  }
  switch (state.phase) {
    case "ready":
      return { text: RPS_LABELS.ready, emphasis: "calm", outcome: "none" };
    case "countdown": {
      const word =
        state.count === null
          ? RPS_LABELS.ready
          : (RPS_COUNTDOWN_WORDS[3 - state.count] ?? RPS_LABELS.ready);
      return { text: word, emphasis: "countdown", outcome: "none" };
    }
    case "shoot":
      return { text: RPS_LABELS.shoot, emphasis: "shoot", outcome: "none" };
    case "result":
      return state.outcome === null
        ? { text: RPS_LABELS.shoot, emphasis: "shoot", outcome: "none" }
        : {
            text: RPS_OUTCOME_LABEL[state.outcome],
            emphasis: "result",
            outcome: state.outcome,
          };
  }
}

/** Confidence of the most visible keypoint; `0` when the hand is not visible. */
function visibleAlpha(points: Keypoint[]): number {
  return points.reduce((best, point) => Math.max(best, pointAlpha(point)), 0);
}

function pointAlpha(point: Keypoint): number {
  const confidence = point[2] ?? 0;
  return confidence >= 0.5 ? 1 : confidence >= 0.2 ? 0.5 : confidence > 0 ? 0.2 : 0;
}

interface HandMapper {
  /** Pixel length the hand spans after the zoom, for line widths. */
  span: number;
  x: (u: number) => number;
  y: (v: number) => number;
}

/**
 * Zooms the visible keypoints into the pane. The booth reads the hand, not the
 * camera frame, so the skeleton is not drawn in IR pixels: it fills
 * `HAND_SHARE` of the shorter side and sits below the countdown.
 */
function handMapper(
  points: Keypoint[],
  rect: { width: number; height: number },
): HandMapper | null {
  const visible = points.filter((point) => pointAlpha(point) > 0);
  if (visible.length === 0) {
    return null;
  }

  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (const point of visible) {
    minU = Math.min(minU, point[0]);
    maxU = Math.max(maxU, point[0]);
    minV = Math.min(minV, point[1]);
    maxV = Math.max(maxV, point[1]);
  }

  const span = Math.max(maxU - minU, maxV - minV, 1) * 1.15;
  const scale = (Math.min(rect.width, rect.height) * HAND_SHARE) / span;
  const centreU = (minU + maxU) / 2;
  const centreV = (minV + maxV) / 2;
  const centreX = rect.width / 2;
  const centreY = rect.height * HAND_CENTRE;
  return {
    span: span * scale,
    x: (u: number) => centreX + (u - centreU) * scale,
    y: (v: number) => centreY + (v - centreV) * scale,
  };
}
