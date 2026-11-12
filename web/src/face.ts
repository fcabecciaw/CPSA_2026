/**
 * Gaze and expression control of the dashboard avatar (F-E4 #26/#27).
 *
 * The SVG holographic face of round 1 is gone — R2-F replaced it with the
 * particle avatar in `avatar/` — but the two DOM-free controllers it owned are
 * still the contract of the dashboard: `GazeController` follows the largest
 * person box with 30 %/1 s hysteresis and smoothing and wanders when nobody is
 * there, `ExpressionController` holds the expression of an utterance for 1.5 s
 * and falls asleep after a minute alone. They are unit-tested here.
 */

import type { Box } from "./cognitive/perception";
import type { Expression } from "./cognitive/rules";

export interface GazePoint {
  x: number;
  y: number;
}

export interface GazeOptions {
  /** A box must be this much larger to steal the gaze… */
  switchMargin?: number;
  /** …and stay so for this long. */
  holdMs?: number;
  /** Smoothing rate, per second. */
  smoothing?: number;
}

/** Largest-box gaze with hysteresis and smoothing (F-E4 #26). */
export class GazeController {
  private readonly switchMargin: number;
  private readonly holdMs: number;
  private readonly smoothing: number;
  private current: GazePoint = { x: 0, y: 0 };
  private target: GazePoint = { x: 0, y: 0 };
  private targetArea = 0;
  private candidate: { area: number; since: number } | null = null;
  private lastAt: number | null = null;

  constructor(options: GazeOptions = {}) {
    this.switchMargin = options.switchMargin ?? 1.3;
    this.holdMs = options.holdMs ?? 1_000;
    this.smoothing = options.smoothing ?? 6;
  }

  update(
    boxes: readonly Box[],
    frame: { width: number; height: number },
    now: number,
    wander = boxes.length === 0,
  ): GazePoint {
    const dt = this.lastAt === null ? 0 : Math.min(0.5, Math.max(0, (now - this.lastAt) / 1000));
    this.lastAt = now;

    if (boxes.length === 0) {
      this.candidate = null;
      this.targetArea = 0;
      this.target = {
        x: Math.sin(now / 2_400) * 0.55,
        y: Math.cos(now / 3_300) * 0.35,
      };
      if (wander) {
        this.current = this.smooth(this.current, this.target, dt);
      }
      return this.current;
    }

    const largest = boxes.reduce(
      (best, box) => (area(box) > area(best) ? box : best),
      boxes[0],
    );
    const largestArea = area(largest);
    const center = centerOf(largest, frame);

    if (this.targetArea === 0) {
      this.targetArea = largestArea;
      this.target = center;
      this.candidate = null;
    } else if (largestArea >= this.targetArea * this.switchMargin) {
      if (this.candidate === null) {
        this.candidate = { area: largestArea, since: now };
      } else if (now - this.candidate.since >= this.holdMs) {
        this.targetArea = largestArea;
        this.target = center;
        this.candidate = null;
      }
    } else {
      this.candidate = null;
    }

    this.current = this.smooth(this.current, this.target, dt);
    return this.current;
  }

  reset(): void {
    this.current = { x: 0, y: 0 };
    this.target = { x: 0, y: 0 };
    this.targetArea = 0;
    this.candidate = null;
    this.lastAt = null;
  }

  private smooth(from: GazePoint, to: GazePoint, dt: number): GazePoint {
    const factor = Math.min(1, dt * this.smoothing);
    return {
      x: from.x + (to.x - from.x) * factor,
      y: from.y + (to.y - from.y) * factor,
    };
  }
}

function area(box: Box): number {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

function centerOf(box: Box, frame: { width: number; height: number }): GazePoint {
  const cx = (box[0] + box[2]) / 2;
  const cy = (box[1] + box[3]) / 2;
  return {
    x: Math.max(-1, Math.min(1, (cx / Math.max(1, frame.width) - 0.5) * 2)),
    y: Math.max(-1, Math.min(1, (cy / Math.max(1, frame.height) - 0.5) * 2)),
  };
}

/** Expression hold machine (F-E4 #27). */
export class ExpressionController {
  private readonly holdMs: number;
  private readonly sleepyAfterMs: number;
  private expression: Expression = "neutral";
  private holdUntil = 0;
  private speaking = false;

  constructor(options: { holdMs?: number; sleepyAfterMs?: number } = {}) {
    this.holdMs = options.holdMs ?? 1_500;
    this.sleepyAfterMs = options.sleepyAfterMs ?? 60_000;
  }

  get current(): Expression {
    return this.expression;
  }

  beginUtterance(expression: Expression, _now: number): void {
    this.speaking = true;
    this.expression = expression;
    this.holdUntil = Number.POSITIVE_INFINITY;
  }

  endUtterance(now: number): void {
    this.speaking = false;
    this.holdUntil = now + this.holdMs;
  }

  /** Direct hold, for connection loss and mode changes. */
  hold(expression: Expression, now: number, holdMs = this.holdMs): void {
    if (this.speaking) {
      return;
    }
    this.expression = expression;
    this.holdUntil = now + holdMs;
  }

  tick(now: number, idleS: number): Expression {
    if (this.speaking) {
      return this.expression;
    }
    if (now < this.holdUntil) {
      return this.expression;
    }
    this.expression = idleS * 1000 >= this.sleepyAfterMs ? "sleepy" : "neutral";
    return this.expression;
  }
}
