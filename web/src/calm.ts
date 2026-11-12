/**
 * Calm view of the objects panel (coordinator feedback, 2026-11-12).
 *
 * `showcase.state` arrives at 10 Hz even with stabilized backend tracks, so the
 * panel would still flicker. This DOM-free filter throttles the visible text to
 * one refresh every 500 ms, rounds scores to 5 %, keeps a stable order (class,
 * then first-seen), holds a disappeared item for 1 s before dropping it and
 * commits a count only after it has held for 500 ms. The UI only has to render
 * the returned snapshot.
 */

import type { ShowcaseStatePayload } from "./protocol";

export interface CalmObject {
  /** Stable identity across frames, from IoU matching. */
  key: string;
  label: string;
  classId: number;
  /** Score in percent, rounded to the nearest 5. */
  scorePercent: number;
  score: number;
  box: [number, number, number, number];
  firstSeenAt: number;
  /** No longer detected, within the 1 s hold. */
  leaving: boolean;
}

export interface CalmCount {
  label: string;
  classId: number;
  count: number;
}

export interface CalmView {
  objects: CalmObject[];
  counts: CalmCount[];
  meta: { model: string; width: number; height: number } | null;
  updatedAt: number;
}

export interface CalmOptions {
  refreshMs?: number;
  countHoldMs?: number;
  leaveHoldMs?: number;
  /** Minimum overlap to consider two boxes the same instance. */
  matchIoU?: number;
}

interface Entry {
  key: string;
  label: string;
  classId: number;
  box: [number, number, number, number];
  score: number;
  firstSeenAt: number;
  lastSeenAt: number;
  matched: boolean;
  /** Absent from the latest frame, inside the leave hold. */
  leaving: boolean;
}

interface PendingCount {
  value: number;
  since: number;
  committed: number;
}

const DEFAULTS: Required<CalmOptions> = {
  refreshMs: 500,
  countHoldMs: 500,
  leaveHoldMs: 1_000,
  matchIoU: 0.3,
};

/** Rounded to the nearest 5 %, for a display that does not jitter. */
export function scorePercent(score: number): number {
  const percent = Math.round((Math.min(1, Math.max(0, score)) * 100) / 5) * 5;
  return Math.min(100, Math.max(0, percent));
}

function iou(a: readonly number[], b: readonly number[]): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (intersection === 0) {
    return 0;
  }
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / (areaA + areaB - intersection);
}

export class CalmShowcase {
  private readonly options: Required<CalmOptions>;
  private entries = new Map<string, Entry>();
  private readonly pendingCounts = new Map<string, PendingCount>();
  private classIds = new Map<string, number>();
  private nextInstance = 0;
  private lastRefreshAt = Number.NEGATIVE_INFINITY;
  private current: CalmView | null = null;
  private lastState: ShowcaseStatePayload | null = null;

  constructor(options: CalmOptions = {}) {
    this.options = { ...DEFAULTS, ...options };
  }

  get view(): CalmView | null {
    return this.current;
  }

  /** Ingest one 10 Hz frame; returns a snapshot only when 500 ms passed. */
  update(state: ShowcaseStatePayload, now: number): CalmView | null {
    this.lastState = state;
    this.trackObjects(state, now);
    this.trackCounts(state, now);
    return this.refresh(now);
  }

  /** Time only: matures counts and drops held items even without frames. */
  tick(now: number): CalmView | null {
    if (this.lastState === null) {
      return null;
    }
    this.trackCounts(this.lastState, now);
    return this.refresh(now);
  }

  reset(): void {
    this.entries = new Map();
    this.pendingCounts.clear();
    this.lastRefreshAt = Number.NEGATIVE_INFINITY;
    this.current = null;
    this.lastState = null;
  }

  private trackObjects(state: ShowcaseStatePayload, now: number): void {
    for (const object of state.objects) {
      this.classIds.set(object.label, object.class_id);
    }
    for (const entry of this.entries.values()) {
      entry.matched = false;
    }

    const incoming = [...state.objects].sort((a, b) => b.score - a.score);
    for (const object of incoming) {
      let best: Entry | null = null;
      let bestIoU = this.options.matchIoU;
      for (const entry of this.entries.values()) {
        if (entry.matched || entry.label !== object.label) {
          continue;
        }
        const overlap = iou(entry.box, object.box);
        if (overlap >= bestIoU) {
          best = entry;
          bestIoU = overlap;
        }
      }
      if (best !== null) {
        best.box = [...object.box] as [number, number, number, number];
        best.score = object.score;
        best.lastSeenAt = now;
        best.matched = true;
      } else {
        const classId = object.class_id;
        const key = `${classId}:${this.nextInstance}`;
        this.nextInstance += 1;
        this.entries.set(key, {
          key,
          label: object.label,
          classId,
          box: [...object.box] as [number, number, number, number],
          score: object.score,
          firstSeenAt: now,
          lastSeenAt: now,
          matched: true,
          leaving: false,
        });
      }
    }

    for (const [key, entry] of this.entries) {
      entry.leaving = !entry.matched;
      if (now - entry.lastSeenAt > this.options.leaveHoldMs) {
        this.entries.delete(key);
      }
    }
  }

  private trackCounts(state: ShowcaseStatePayload, now: number): void {
    const observed = new Map<string, number>();
    if (state.counts.length > 0) {
      for (const entry of state.counts) {
        observed.set(entry.label, entry.count);
      }
    } else {
      for (const object of state.objects) {
        observed.set(object.label, (observed.get(object.label) ?? 0) + 1);
      }
    }
    const labels = new Set([...observed.keys(), ...this.pendingCounts.keys()]);
    for (const label of labels) {
      const value = observed.get(label) ?? 0;
      const pending = this.pendingCounts.get(label);
      if (pending === undefined) {
        this.pendingCounts.set(label, { value, since: now, committed: value });
      } else if (pending.value !== value) {
        pending.value = value;
        pending.since = now;
      } else if (pending.committed !== value && now - pending.since >= this.options.countHoldMs) {
        pending.committed = value;
      }
      if (this.pendingCounts.get(label)?.committed === 0 && !observed.has(label)) {
        this.pendingCounts.delete(label);
      }
    }
  }

  private refresh(now: number): CalmView | null {
    if (now - this.lastRefreshAt < this.options.refreshMs) {
      return null;
    }
    this.lastRefreshAt = now;
    const objects: CalmObject[] = [...this.entries.values()]
      .map((entry) => ({
        key: entry.key,
        label: entry.label,
        classId: entry.classId,
        scorePercent: scorePercent(entry.score),
        score: entry.score,
        box: entry.box,
        firstSeenAt: entry.firstSeenAt,
        leaving: entry.leaving,
      }))
      .sort(
        (a, b) =>
          a.classId - b.classId ||
          a.firstSeenAt - b.firstSeenAt ||
          (a.key < b.key ? -1 : a.key > b.key ? 1 : 0),
      );

    const counts: CalmCount[] = [...this.pendingCounts.entries()]
      .filter(([, pending]) => pending.committed > 0)
      .map(([label, pending]) => ({
        label,
        classId: this.classIds.get(label) ?? 999,
        count: pending.committed,
      }))
      .sort((a, b) => a.classId - b.classId || (a.label < b.label ? -1 : 1));

    this.current = {
      objects,
      counts,
      meta:
        this.lastState === null
          ? null
          : {
              model: this.lastState.model,
              width: this.lastState.width,
              height: this.lastState.height,
            },
      updatedAt: now,
    };
    return this.current;
  }
}
