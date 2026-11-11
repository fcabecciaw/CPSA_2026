/**
 * Sasso-carta-forbice view (T5): `rps.state` as a booth-facing game panel.
 *
 * The countdown, the player's gesture, the board's move, the result banner and
 * the score are plain DOM; the small IR skeleton preview is canvas 2D, drawn
 * from the same 21 points the classifier saw (`hand.kpts2d`).
 */

import { el } from "./dom";
import { HAND21_BONES } from "./hologram";
import {
  RPS_COUNTDOWN_WORDS,
  RPS_GESTURE_ICON,
  RPS_GESTURE_LABEL,
  RPS_LABELS,
  RPS_OUTCOME_LABEL,
  RPS_PHASE_LABEL,
} from "./labels";
import type { Keypoint, RpsScore, RpsStatePayload } from "./protocol";

const COLORS = {
  background: "#070d17",
  bone: "rgba(143, 208, 255, 0.6)",
  joint: "#bff4ff",
} as const;

/** Empty slot shown before a gesture is known. */
const NO_GESTURE = "—";

export interface RpsElements {
  count: HTMLElement;
  phase: HTMLElement;
  playerIcon: HTMLElement;
  playerLabel: HTMLElement;
  boardIcon: HTMLElement;
  boardLabel: HTMLElement;
  score: HTMLElement;
  canvas: HTMLCanvasElement;
  empty: HTMLElement;
}

type ScoreKey = keyof RpsScore;

export class RpsView {
  private readonly elements: RpsElements;
  private readonly context: CanvasRenderingContext2D;
  private readonly observer: ResizeObserver | null;
  private readonly scoreValues = new Map<ScoreKey, HTMLElement>();

  private state: RpsStatePayload | null = null;

  constructor(elements: RpsElements) {
    const context = elements.canvas.getContext("2d");
    if (context === null) {
      throw new Error("Canvas 2D non disponibile in questo browser");
    }
    this.elements = elements;
    this.context = context;

    const rows: Array<[string, ScoreKey]> = [
      [RPS_LABELS.player, "player"],
      [RPS_LABELS.board, "board"],
      [RPS_LABELS.draws, "draws"],
    ];
    for (const [label, key] of rows) {
      const value = el("span", "rps-score-value", "0");
      const item = el("li", "rps-score-item");
      item.append(el("span", "rps-score-label", label), value);
      elements.score.append(item);
      this.scoreValues.set(key, value);
    }

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

  destroy(): void {
    this.observer?.disconnect();
    window.removeEventListener("resize", this.onWindowResize);
  }

  private readonly onWindowResize = (): void => this.draw();

  private render(): void {
    const { count, phase, playerIcon, playerLabel, boardIcon, boardLabel, empty } = this.elements;
    const state = this.state;

    if (state === null) {
      count.textContent = RPS_LABELS.waiting;
      phase.textContent = "";
      playerIcon.textContent = RPS_GESTURE_ICON.none;
      playerLabel.textContent = NO_GESTURE;
      boardIcon.textContent = RPS_GESTURE_ICON.none;
      boardLabel.textContent = NO_GESTURE;
      delete count.dataset.outcome;
      empty.hidden = false;
      this.renderScore(null);
      this.draw();
      return;
    }

    count.textContent = countText(state);
    phase.textContent = RPS_PHASE_LABEL[state.phase];
    if (state.outcome === null) {
      delete count.dataset.outcome;
    } else {
      // The callout doubles as the result banner, tinted by the outcome.
      count.dataset.outcome = state.outcome;
    }

    const player = state.player;
    playerIcon.textContent = player === null ? RPS_GESTURE_ICON.none : RPS_GESTURE_ICON[player];
    playerLabel.textContent = player === null ? NO_GESTURE : RPS_GESTURE_LABEL[player];

    const board = state.board;
    boardIcon.textContent = board === null ? RPS_GESTURE_ICON.none : RPS_GESTURE_ICON[board];
    boardLabel.textContent = board === null ? NO_GESTURE : RPS_GESTURE_LABEL[board];

    empty.hidden = state.hand !== null && visibleCount(state.hand.kpts2d) > 0;
    this.renderScore(state.score);
    // The canvas label spells out the round: the icons and the count are
    // decorative to a screen reader, which only reads this summary.
    this.elements.canvas.setAttribute(
      "aria-label",
      `${RPS_LABELS.preview}: ${count.textContent}, ` +
        `${playerLabel.textContent} ${RPS_LABELS.versus} ${boardLabel.textContent}.`,
    );
    this.draw();
  }

  private renderScore(score: RpsScore | null): void {
    for (const [key, node] of this.scoreValues) {
      node.textContent = String(score === null ? 0 : score[key]);
    }
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
    const mapper = previewMapper(points, rect);
    if (mapper === null) {
      return;
    }

    const unit = Math.min(rect.width, rect.height);
    context.lineWidth = Math.max(1, unit * 0.012);
    context.strokeStyle = COLORS.bone;
    for (const [first, second] of HAND21_BONES) {
      const from = points[first];
      const to = points[second];
      if (from === undefined || to === undefined) {
        continue;
      }
      const alpha = Math.min(visibleAlpha(from), visibleAlpha(to));
      if (alpha === 0) {
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
    for (const point of points) {
      const alpha = visibleAlpha(point);
      if (alpha === 0) {
        continue;
      }
      context.globalAlpha = alpha;
      context.beginPath();
      context.arc(mapper.x(point[0]), mapper.y(point[1]), Math.max(1.5, unit * 0.014), 0, Math.PI * 2);
      context.fill();
    }
    context.globalAlpha = 1;
  }
}

/** Countdown word for the current step; the phase copy everywhere else. */
function countText(state: RpsStatePayload): string {
  switch (state.phase) {
    case "waiting":
      return RPS_LABELS.waiting;
    case "ready":
      return RPS_LABELS.ready;
    case "countdown":
      return state.count === null
        ? RPS_LABELS.ready
        : (RPS_COUNTDOWN_WORDS[3 - state.count] ?? RPS_LABELS.ready);
    case "shoot":
      return RPS_LABELS.shoot;
    case "result":
      return state.outcome === null ? RPS_LABELS.shoot : RPS_OUTCOME_LABEL[state.outcome];
  }
}

function visibleCount(points: Keypoint[]): number {
  return points.filter((point) => visibleAlpha(point) > 0).length;
}

function visibleAlpha(point: Keypoint): number {
  const confidence = point[2] ?? 0;
  return confidence >= 0.5 ? 1 : confidence >= 0.2 ? 0.5 : confidence > 0 ? 0.2 : 0;
}

interface PointMapper {
  x: (u: number) => number;
  y: (v: number) => number;
}

/**
 * Fits the visible joints into the canvas with a 20 % margin, mirrored like the
 * medical skeleton so the preview reads like a mirror to the player.
 */
function previewMapper(
  points: Keypoint[],
  rect: { width: number; height: number },
): PointMapper | null {
  const visible = points.filter((point) => visibleAlpha(point) > 0);
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

  const centreU = (minU + maxU) / 2;
  const centreV = (minV + maxV) / 2;
  const span = Math.max(maxU - minU, maxV - minV, 1) * 1.2;
  const scale = Math.min(rect.width, rect.height) / span;

  return {
    x: (u: number) => rect.width / 2 - (u - centreU) * scale,
    y: (v: number) => rect.height / 2 + (v - centreV) * scale,
  };
}
