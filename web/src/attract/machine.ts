/**
 * The attract loop's state machine, DOM-free.
 *
 * The page opens in `loop`: the "Come funziona" scenes play one after the
 * other, forever, until the operator activates the demo. `activate` switches
 * to `demo`; `returnToLoop` goes back to the first scene. A pinned scene (the
 * dev screenshots) never advances on its own.
 */

export type AttractPhase = "loop" | "demo";

export interface AttractMachineOptions {
  /** How long each scene stays on screen, in milliseconds. */
  durations: readonly number[];
  now: number;
  /** `demo` skips the loop at load (`?attract=0`). */
  start?: AttractPhase;
  /** Hold this scene (0-based) and never advance: dev screenshots. */
  pinnedScene?: number | null;
}

export interface AttractSnapshot {
  phase: AttractPhase;
  scene: number;
  sceneStartedAt: number;
}

export class AttractMachine {
  private readonly durations: readonly number[];
  private readonly pinned: number | null;
  private currentPhase: AttractPhase;
  private currentScene: number;
  private startedAt: number;

  constructor(options: AttractMachineOptions) {
    if (options.durations.length === 0) {
      throw new Error("the attract loop needs at least one scene");
    }
    this.durations = options.durations;
    const pinned = options.pinnedScene ?? null;
    this.pinned =
      pinned === null || !Number.isInteger(pinned)
        ? null
        : Math.min(this.durations.length - 1, Math.max(0, pinned));
    this.currentPhase = options.start ?? "loop";
    this.currentScene = this.pinned ?? 0;
    this.startedAt = options.now;
  }

  get phase(): AttractPhase {
    return this.currentPhase;
  }

  get scene(): number {
    return this.currentScene;
  }

  get sceneCount(): number {
    return this.durations.length;
  }

  get isPinned(): boolean {
    return this.pinned !== null;
  }

  snapshot(): AttractSnapshot {
    return {
      phase: this.currentPhase,
      scene: this.currentScene,
      sceneStartedAt: this.startedAt,
    };
  }

  durationOf(scene: number): number {
    return this.durations[scene] ?? 0;
  }

  /** 0..1 through the current scene; 0 while the demo runs. */
  progress(now: number): number {
    if (this.currentPhase !== "loop") {
      return 0;
    }
    const duration = this.durationOf(this.currentScene);
    return duration <= 0 ? 1 : Math.min(1, Math.max(0, (now - this.startedAt) / duration));
  }

  /**
   * Advance past every scene whose time is up. Returns true when the scene
   * changed. A long pause (a hidden tab) skips scenes rather than replaying
   * them one per tick.
   */
  tick(now: number): boolean {
    if (this.currentPhase !== "loop" || this.pinned !== null) {
      return false;
    }
    let changed = false;
    let guard = this.durations.length * 4;
    while (now - this.startedAt >= this.durationOf(this.currentScene) && guard > 0) {
      this.startedAt += Math.max(1, this.durationOf(this.currentScene));
      this.currentScene = (this.currentScene + 1) % this.durations.length;
      changed = true;
      guard -= 1;
    }
    if (guard === 0) {
      // Away for many loops: restart the current scene from now.
      this.startedAt = now;
    }
    return changed;
  }

  /** The operator starts the demo. Returns true when the phase changed. */
  activate(): boolean {
    if (this.currentPhase === "demo") {
      return false;
    }
    this.currentPhase = "demo";
    return true;
  }

  /** Back to the loop, from its first scene (or the pinned one). */
  returnToLoop(now: number): boolean {
    if (this.currentPhase === "loop") {
      return false;
    }
    this.currentPhase = "loop";
    this.currentScene = this.pinned ?? 0;
    this.startedAt = now;
    return true;
  }

  /** The operator skips ahead or back (arrow keys) while the loop runs. */
  step(delta: number, now: number): boolean {
    if (this.currentPhase !== "loop" || delta === 0) {
      return false;
    }
    const count = this.durations.length;
    this.currentScene = (((this.currentScene + delta) % count) + count) % count;
    this.startedAt = now;
    return true;
  }
}
