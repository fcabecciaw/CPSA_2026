/**
 * The two numbers of the top bar: how fast the board analyses frames and how
 * long one frame takes.
 *
 * The detector reports its own rate and frame time in `showcase.state`
 * (`fps`, `total_ms`, falling back to `dpu_ms`); until the first frame arrives,
 * and while the process is not running, the `status` snapshot is the source.
 */

import type { ShowcaseStatePayload, StatusPayload } from "./protocol";

export interface Readout {
  fps: number | null;
  latencyMs: number | null;
}

export class MetricsTracker {
  private statusFps: number | null = null;
  private statusLatency: number | null = null;
  private showcaseFps: number | null = null;
  private showcaseLatency: number | null = null;

  /** The latest `status` snapshot. */
  status(status: StatusPayload | null): void {
    this.statusFps = status?.fps ?? null;
    this.statusLatency = status?.dpu_ms ?? null;
    if (status === null || status.state !== "running") {
      this.showcaseFps = null;
      this.showcaseLatency = null;
    }
  }

  /** One `showcase.state`: the detector reports its own rate and frame time. */
  showcase(state: ShowcaseStatePayload): void {
    this.showcaseFps = state.fps;
    this.showcaseLatency = state.total_ms ?? state.dpu_ms;
  }

  read(): Readout {
    if (this.showcaseFps === null && this.showcaseLatency === null) {
      return { fps: this.statusFps, latencyMs: this.statusLatency };
    }
    return {
      fps: this.showcaseFps ?? this.statusFps,
      latencyMs: this.showcaseLatency ?? this.statusLatency,
    };
  }
}
