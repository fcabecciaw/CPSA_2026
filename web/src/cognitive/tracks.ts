/**
 * A tiny IoU tracker for person and face boxes (C3 arrivals), pure and
 * DOM-free.
 *
 * Boxes are `[x0, y0, x1, y1]` normalised to the full frame. Every sample is
 * matched greedily to the live tracks by IoU (best pair first); unmatched
 * boxes start new tracks and a track unseen for `maxMissMs` is dropped. Each
 * track carries the "at the stand" gate with hysteresis: it enters when its
 * size reaches the threshold and leaves only below 80 % of it, so a person on
 * the edge of the threshold does not blink in and out.
 */

import type { NormBox } from "./facecrop";

export interface Track {
  readonly id: number;
  box: NormBox;
  bornAt: number;
  lastSeen: number;
  hits: number;
  /** The gated size of the last sample (box height or face width). */
  size: number;
  /** Close enough to count (gate with hysteresis). */
  atStand: boolean;
  /** When the track last entered the stand, `null` while in the background. */
  standSince: number | null;
  /** Already greeted (or seeded, or the same person as a greeted track). */
  greeted: boolean;
}

export interface TrackerOptions {
  /** Minimum IoU to continue a track. */
  minIou: number;
  /** A track unseen for this long is gone. */
  maxMissMs: number;
  /** Leave the stand below this fraction of the threshold. */
  hysteresis: number;
}

export const DEFAULT_TRACKER: TrackerOptions = { minIou: 0.2, maxMissMs: 1_500, hysteresis: 0.8 };

export function iou(a: NormBox, b: NormBox): number {
  const x0 = Math.max(a[0], b[0]);
  const y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]);
  const y1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union =
    (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union <= 0 ? 0 : inter / union;
}

export function centreInside(inner: NormBox, outer: NormBox): boolean {
  const x = (inner[0] + inner[2]) / 2;
  const y = (inner[1] + inner[3]) / 2;
  return x >= outer[0] && x <= outer[2] && y >= outer[1] && y <= outer[3];
}

export class BoxTracker {
  readonly options: TrackerOptions;
  private tracks: Track[] = [];
  private nextId = 1;

  constructor(
    /** The gated size of a box: its height for people, its width for faces. */
    private readonly sizeOf: (box: NormBox) => number,
    options: Partial<TrackerOptions> = {},
  ) {
    this.options = { ...DEFAULT_TRACKER, ...options };
  }

  get live(): readonly Track[] {
    return this.tracks;
  }

  clear(): void {
    this.tracks = [];
  }

  /**
   * One sample of boxes; returns the tracks dropped by this call (unseen for
   * `maxMissMs`), so the caller can remember who was greeted.
   */
  update(boxes: readonly NormBox[], now: number, threshold: number): Track[] {
    const pairs: Array<{ track: Track; box: number; score: number }> = [];
    this.tracks.forEach((track) => {
      boxes.forEach((box, index) => {
        const score = iou(track.box, box);
        if (score >= this.options.minIou) {
          pairs.push({ track, box: index, score });
        }
      });
    });
    pairs.sort((a, b) => b.score - a.score);
    const usedTracks = new Set<Track>();
    const usedBoxes = new Set<number>();
    for (const pair of pairs) {
      if (usedTracks.has(pair.track) || usedBoxes.has(pair.box)) {
        continue;
      }
      usedTracks.add(pair.track);
      usedBoxes.add(pair.box);
      this.see(pair.track, boxes[pair.box]!, now, threshold);
    }
    boxes.forEach((box, index) => {
      if (usedBoxes.has(index)) {
        return;
      }
      const track: Track = {
        id: this.nextId++,
        box,
        bornAt: now,
        lastSeen: now,
        hits: 0,
        size: 0,
        atStand: false,
        standSince: null,
        greeted: false,
      };
      this.tracks.push(track);
      this.see(track, box, now, threshold);
    });
    return this.prune(now);
  }

  /** Drop the tracks unseen for `maxMissMs` (the source may have stopped). */
  prune(now: number): Track[] {
    const dropped = this.tracks.filter((track) => now - track.lastSeen > this.options.maxMissMs);
    if (dropped.length > 0) {
      this.tracks = this.tracks.filter((track) => !dropped.includes(track));
    }
    return dropped;
  }

  private see(track: Track, box: NormBox, now: number, threshold: number): void {
    track.box = box;
    track.lastSeen = now;
    track.hits += 1;
    track.size = this.sizeOf(box);
    const atStand = track.atStand
      ? track.size >= threshold * this.options.hysteresis
      : track.size >= threshold;
    if (atStand && !track.atStand) {
      track.standSince = now;
    } else if (!atStand) {
      track.standSince = null;
    }
    track.atStand = atStand;
  }
}
