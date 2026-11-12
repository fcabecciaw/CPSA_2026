/**
 * The two numbers of the top bar: how fast the board analyses frames and how
 * long one frame takes.
 *
 * The hologram mode publishes a `timing` block with every frame, so its rate
 * and its total time come from the frame arrivals themselves; the
 * Sasso-carta-forbice mode publishes no timing, so it reads the `status`
 * snapshot, exactly as the medical dashboard did; the showcase mode reports
 * its own rate and frame time in `showcase.state`.
 */

import type { Mode, ShowcaseStatePayload, StatusPayload } from "./protocol";

export interface Readout {
  fps: number | null;
  latencyMs: number | null;
}

const WINDOW_MS = 2_000;

const EMPTY: Readout = { fps: null, latencyMs: null };

export class MetricsTracker {
  private arrivals: number[] = [];
  private hologramLatency: number | null = null;
  private statusFps: number | null = null;
  private statusLatency: number | null = null;
  private showcaseFps: number | null = null;
  private showcaseLatency: number | null = null;

  /** One `hologram.hands` frame: when it arrived and its total processing time. */
  frame(receivedAt: number, totalMs: number): void {
    this.arrivals.push(receivedAt);
    this.hologramLatency = totalMs;
    const cutoff = receivedAt - WINDOW_MS;
    while (this.arrivals.length > 0 && (this.arrivals[0] ?? 0) < cutoff) {
      this.arrivals.shift();
    }
  }

  /** The latest `status` snapshot; `null` clears the readout with the connection. */
  status(status: StatusPayload | null): void {
    const perception = status?.perception;
    if (perception === undefined || perception === null) {
      this.statusFps = null;
      this.statusLatency = null;
      return;
    }
    this.statusFps = perception.timing?.analyzed_fps ?? perception.fps;
    this.statusLatency = perception.timing?.latency_ms ?? perception.dpu_ms;
  }

  /** One `showcase.state`: the detector reports its own rate and frame time. */
  showcase(state: ShowcaseStatePayload): void {
    this.showcaseFps = state.fps;
    this.showcaseLatency = state.total_ms ?? state.dpu_ms;
  }

  /** Forgets every sample; used when the server leaves a streamed mode. */
  reset(): void {
    this.arrivals = [];
    this.hologramLatency = null;
    this.showcaseFps = null;
    this.showcaseLatency = null;
  }

  read(mode: Mode | null): Readout {
    if (mode === "hologram") {
      return { fps: this.hologramFps(), latencyMs: this.hologramLatency };
    }
    if (mode === "rps") {
      return { fps: this.statusFps, latencyMs: this.statusLatency };
    }
    if (mode === "showcase") {
      return { fps: this.showcaseFps, latencyMs: this.showcaseLatency };
    }
    return EMPTY;
  }

  private hologramFps(): number | null {
    const first = this.arrivals[0];
    const last = this.arrivals[this.arrivals.length - 1];
    if (first === undefined || last === undefined || this.arrivals.length < 2 || last <= first) {
      return null;
    }
    return ((this.arrivals.length - 1) * 1000) / (last - first);
  }
}
