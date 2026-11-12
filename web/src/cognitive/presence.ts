/**
 * Presence and arrivals (C3): who is at the stand, who just arrived, and
 * whether everybody has left.
 *
 * Three sources, each gated by distance so people in the background never
 * count:
 *
 * - **persona** — YOLOX person boxes, followed by an IoU tracker; a person is
 *   "at the stand" when the box is at least `minBoxHeight` of the frame high;
 * - **volto** — laptop MediaPipe face boxes, followed the same way; at the
 *   stand when the face is at least `minFaceWidth` of the frame wide;
 * - **scheletro** — a MoveNet pose with a score of at least 0.3 (no box: it
 *   only counts when no person or face track is at the stand).
 *
 * **Arrivals come from tracks, not from counts.** A newcomer is a track that
 * has been at the stand for `confirmMs` (0.6 s) and was never greeted; a face
 * whose centre is inside a person box is the same person. Newcomers that
 * confirm within `batchWindowMs` of each other get one collective greeting
 * with their number; the greetings are at least `greetingCooldownMs` (8 s)
 * apart, and newcomers confirmed during the cooldown are batched into the
 * next greeting if they are still there. Nobody is greeted twice: a track
 * lost and found again within `ghostMs` at the same place keeps its flag.
 *
 * **The goodbye** needs everybody gone: no gated detection from any source
 * for `absenceMs` (3 s), a greeting earlier in the visit, none said yet in the
 * visit, at most one per `farewellCooldownMs` (60 s). A partial departure
 * (2 → 1) changes nothing.
 *
 * The two size thresholds live here as module state with a persisted setter,
 * so a calibration panel can tune them on site.
 *
 * Pure and DOM-free apart from the optional `localStorage` persistence.
 */

import type { NormBox } from "./facecrop";
import { BoxTracker, centreInside, iou, type Track } from "./tracks";

// --- thresholds -------------------------------------------------------------

export interface PresenceThresholds {
  /** `presence.min_box_height`: person box height / frame height. */
  minBoxHeight: number;
  /** `presence.min_face_width`: face box width / frame width. */
  minFaceWidth: number;
}

export const DEFAULT_PRESENCE_THRESHOLDS: Readonly<PresenceThresholds> = Object.freeze({
  minBoxHeight: 0.35,
  minFaceWidth: 0.08,
});

export const PRESENCE_THRESHOLDS_KEY = "cpsa.presence.thresholds.v1";

interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function browserStorage(): StorageLike | null {
  try {
    const storage = (globalThis as { localStorage?: StorageLike }).localStorage;
    return storage ?? null;
  } catch {
    return null;
  }
}

function sanitize(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(1, Math.max(0, value))
    : fallback;
}

/** Read the persisted thresholds, the defaults when missing or invalid. */
export function loadPresenceThresholds(
  storage: StorageLike | null = browserStorage(),
): PresenceThresholds {
  let saved: Partial<PresenceThresholds> = {};
  try {
    const raw = storage?.getItem(PRESENCE_THRESHOLDS_KEY) ?? null;
    if (raw !== null) {
      saved = JSON.parse(raw) as Partial<PresenceThresholds>;
    }
  } catch {
    saved = {};
  }
  return {
    minBoxHeight: sanitize(saved.minBoxHeight, DEFAULT_PRESENCE_THRESHOLDS.minBoxHeight),
    minFaceWidth: sanitize(saved.minFaceWidth, DEFAULT_PRESENCE_THRESHOLDS.minFaceWidth),
  };
}

let currentThresholds: PresenceThresholds = loadPresenceThresholds();

/** The live thresholds (read by the trackers at every sample). */
export function presenceThresholds(): Readonly<PresenceThresholds> {
  return currentThresholds;
}

/** Set (and persist) the thresholds, e.g. from a calibration panel. */
export function setPresenceThresholds(
  next: Partial<PresenceThresholds>,
  storage: StorageLike | null = browserStorage(),
): Readonly<PresenceThresholds> {
  currentThresholds = {
    minBoxHeight: sanitize(next.minBoxHeight, currentThresholds.minBoxHeight),
    minFaceWidth: sanitize(next.minFaceWidth, currentThresholds.minFaceWidth),
  };
  try {
    storage?.setItem(PRESENCE_THRESHOLDS_KEY, JSON.stringify(currentThresholds));
  } catch {
    // Private mode or quota: the value still applies to this session.
  }
  return currentThresholds;
}

// --- fusion -----------------------------------------------------------------

export type PresenceSourceName = "yolo" | "face" | "pose";

export interface PresenceOptions {
  /** Everybody gone: no gated detection from any source for this long. */
  absenceMs: number;
  /** A track at the stand for this long is a confirmed newcomer. */
  confirmMs: number;
  /** Newcomers confirmed within this window share one greeting. */
  batchWindowMs: number;
  /** Greetings at least this far apart. */
  greetingCooldownMs: number;
  /** At most one goodbye in this window. */
  farewellCooldownMs: number;
  /** A greeted track lost and found again at the same place within this. */
  ghostMs: number;
  /** A pose sample older than this no longer counts. */
  sourceTtlMs: number;
  /** A detection seen within this is "visible now" (readout, goodbye gate). */
  visibleMs: number;
  /** Minimum MoveNet pose score that counts as a person. */
  poseMinScore: number;
  /** Size thresholds; the module-level, persisted ones by default. */
  thresholds: () => Readonly<PresenceThresholds>;
}

export const DEFAULT_PRESENCE_OPTIONS: PresenceOptions = {
  absenceMs: 3_000,
  confirmMs: 600,
  batchWindowMs: 600,
  greetingCooldownMs: 8_000,
  farewellCooldownMs: 60_000,
  ghostMs: 3_000,
  sourceTtlMs: 1_000,
  visibleMs: 500,
  poseMinScore: 0.3,
  thresholds: presenceThresholds,
};

/** One tracked person or face, for the debug readout. */
export interface TrackedPerson {
  kind: "persona" | "volto";
  id: number;
  /** Box height (persona) or face width (volto), as a fraction of the frame. */
  size: number;
  atStand: boolean;
  greeted: boolean;
}

export interface PresenceSnapshot {
  /** Somebody is at the stand (absent only after `absenceMs` of nothing). */
  present: boolean;
  /** Gated detections seen just now, per source. */
  sources: Record<PresenceSourceName, boolean>;
  /** Gated faces visible now. */
  faces: number;
  /** People at the stand: max(gated person tracks, gated faces). */
  people: number;
  /** While present but nothing is seen: for how long. */
  missingMs: number | null;
  /** A greeting was issued in this visit. */
  greeted: boolean;
  /** Confirmed newcomers waiting for their greeting (window or cooldown). */
  waiting: number;
  thresholds: Readonly<PresenceThresholds>;
  tracks: TrackedPerson[];
}

interface Ghost {
  kind: "persona" | "volto";
  box: NormBox;
  until: number;
}

export class PresenceFusion {
  readonly options: PresenceOptions;

  private readonly persons = new BoxTracker((box) => box[3] - box[1]);
  private readonly faces = new BoxTracker((box) => box[2] - box[0]);
  private ghosts: Ghost[] = [];
  private pose = { positive: false, at: Number.NEGATIVE_INFINITY };
  private poseSince: number | null = null;
  private poseGreeted = false;
  private present = false;
  private lastPositiveAt: number | null = null;
  private visitGreeted = false;
  private goodbyeSaid = false;
  private lastFarewellAt: number | null = null;
  private lastGreetingAt: number | null = null;
  private batchStart: number | null = null;
  private seedUntil = Number.NEGATIVE_INFINITY;
  private issued = 0;
  private waiting = 0;
  private lastNow = 0;

  constructor(options: Partial<PresenceOptions> = {}) {
    this.options = { ...DEFAULT_PRESENCE_OPTIONS, ...options };
  }

  get isPresent(): boolean {
    return this.present;
  }

  /** One YOLOX sample: the person boxes, normalised to the full frame. */
  observePersons(boxes: readonly NormBox[], now: number): void {
    const dropped = this.persons.update(boxes, now, this.options.thresholds().minBoxHeight);
    this.remember("persona", dropped, now);
    if (this.persons.live.some((track) => track.lastSeen === now && track.atStand)) {
      this.lastPositiveAt = now;
    }
    this.update(now);
  }

  /** One laptop Face Landmarker sample: the face boxes, normalised. */
  observeFaces(boxes: readonly NormBox[], now: number): void {
    const dropped = this.faces.update(boxes, now, this.options.thresholds().minFaceWidth);
    this.remember("volto", dropped, now);
    if (this.faces.live.some((track) => track.lastSeen === now && track.atStand)) {
      this.lastPositiveAt = now;
    }
    this.update(now);
  }

  /** One MoveNet `pose.state`: its overall score. */
  observePose(score: number, now: number): void {
    this.pose = { positive: score >= this.options.poseMinScore, at: now };
    if (this.pose.positive) {
      this.lastPositiveAt = now;
    }
    this.update(now);
  }

  /**
   * Whoever confirms before `until` is already there (start, reconnect, mode
   * switch): greeted silently, never as a newcomer (F-E4 #20/#22).
   */
  seed(until: number): void {
    this.seedUntil = Math.max(this.seedUntil, until);
  }

  /** Newcomers to greet now, consumed: 0 most of the time. */
  takeArrivals(): number {
    const issued = this.issued;
    this.issued = 0;
    return issued;
  }

  /** Kept for the agent: a greeting is tracked when it is issued. */
  noteGreeting(_now: number): void {
    this.visitGreeted = true;
  }

  /** A goodbye was spoken: none again in this visit nor for 60 s. */
  noteFarewell(now: number): void {
    this.goodbyeSaid = true;
    this.lastFarewellAt = now;
  }

  /** Whether the goodbye may be said right now. */
  farewellAllowed(now: number): boolean {
    if (!this.visitGreeted || this.goodbyeSaid || this.visibleFaces(now) > 0) {
      return false;
    }
    return (
      this.lastFarewellAt === null ||
      now - this.lastFarewellAt >= this.options.farewellCooldownMs
    );
  }

  /** Advance to `now`: stale tracks, presence, arrivals. */
  update(now: number): PresenceSnapshot {
    this.lastNow = Math.max(this.lastNow, now);
    this.remember("persona", this.persons.prune(now), now);
    this.remember("volto", this.faces.prune(now), now);
    this.ghosts = this.ghosts.filter((ghost) => ghost.until > now);
    const poseFresh = this.pose.positive && now - this.pose.at <= this.options.sourceTtlMs;

    const positive =
      this.lastPositiveAt !== null && now - this.lastPositiveAt < this.options.absenceMs;
    if (positive && !this.present) {
      // A new visit: its own greeting and its own goodbye.
      this.present = true;
      this.visitGreeted = false;
      this.goodbyeSaid = false;
    } else if (!positive && this.present) {
      this.present = false;
      this.batchStart = null;
      this.poseGreeted = false;
      this.poseSince = null;
      // Everybody left: whoever comes next is a newcomer.
      this.ghosts = [];
    }

    this.arrivals(now, poseFresh);
    return this.snapshotFor(now, poseFresh);
  }

  snapshot(now: number = this.lastNow): PresenceSnapshot {
    const poseFresh = this.pose.positive && now - this.pose.at <= this.options.sourceTtlMs;
    return this.snapshotFor(now, poseFresh);
  }

  // --- internals -------------------------------------------------------------

  private arrivals(now: number, poseFresh: boolean): void {
    const persons = this.persons.live.filter((track) => track.atStand);
    const faces = this.faces.live.filter((track) => track.atStand);
    const confirmed = (track: Track): boolean =>
      track.standSince !== null && now - track.standSince >= this.options.confirmMs && track.hits >= 3;
    const pending: Track[] = [];

    for (const person of persons) {
      if (person.greeted || !confirmed(person)) {
        continue;
      }
      // Same person as a greeted face inside the box, or back at the same place.
      const knownFace = faces.some((face) => face.greeted && centreInside(face.box, person.box));
      if (knownFace || this.matchGhost("persona", person.box)) {
        person.greeted = true;
        continue;
      }
      pending.push(person);
    }
    for (const face of faces) {
      if (face.greeted || !confirmed(face)) {
        continue;
      }
      const owner = persons.find((person) => centreInside(face.box, person.box));
      if (owner !== undefined) {
        // Not a separate person: it shares the person's greeting.
        face.greeted = owner.greeted;
        continue;
      }
      if (this.matchGhost("volto", face.box)) {
        face.greeted = true;
        continue;
      }
      pending.push(face);
    }

    // The skeleton has no box: it is a person only when nothing else is.
    let posePending = false;
    if (poseFresh) {
      this.poseSince ??= now;
      if (persons.length > 0 || faces.length > 0) {
        this.poseGreeted = true;
      } else if (!this.poseGreeted && now - this.poseSince >= this.options.confirmMs) {
        posePending = true;
      }
    } else {
      this.poseSince = null;
    }

    const count = pending.length + (posePending ? 1 : 0);
    if (count === 0) {
      this.batchStart = null;
      this.waiting = 0;
      return;
    }
    if (now < this.seedUntil) {
      this.markGreeted(pending, faces, posePending);
      this.batchStart = null;
      this.waiting = 0;
      return;
    }
    this.batchStart ??= now;
    this.waiting = count;
    const windowDone = now - this.batchStart >= this.options.batchWindowMs;
    const cooled =
      this.lastGreetingAt === null ||
      now - this.lastGreetingAt >= this.options.greetingCooldownMs;
    if (!windowDone || !cooled) {
      return;
    }
    this.markGreeted(pending, faces, posePending);
    this.issued += count;
    this.lastGreetingAt = now;
    this.visitGreeted = true;
    this.batchStart = null;
    this.waiting = 0;
  }

  private markGreeted(pending: Track[], faces: readonly Track[], pose: boolean): void {
    for (const track of pending) {
      track.greeted = true;
      // The faces inside a greeted person box are that person.
      for (const face of faces) {
        if (centreInside(face.box, track.box)) {
          face.greeted = true;
        }
      }
    }
    if (pose) {
      this.poseGreeted = true;
    }
  }

  private remember(kind: Ghost["kind"], dropped: readonly Track[], now: number): void {
    for (const track of dropped) {
      if (track.greeted) {
        this.ghosts.push({ kind, box: track.box, until: now + this.options.ghostMs });
      }
    }
  }

  private matchGhost(kind: Ghost["kind"], box: NormBox): boolean {
    const index = this.ghosts.findIndex(
      (ghost) =>
        ghost.kind === kind && (iou(ghost.box, box) >= 0.2 || centreInside(box, ghost.box)),
    );
    if (index < 0) {
      return false;
    }
    this.ghosts.splice(index, 1);
    return true;
  }

  private visible(tracks: readonly Track[], now: number): Track[] {
    return tracks.filter(
      (track) =>
        track.atStand && track.hits >= 3 && now - track.lastSeen <= this.options.visibleMs,
    );
  }

  private visibleFaces(now: number): number {
    return this.visible(this.faces.live, now).length;
  }

  private snapshotFor(now: number, poseFresh: boolean): PresenceSnapshot {
    const persons = this.visible(this.persons.live, now);
    const faces = this.visible(this.faces.live, now);
    const sources = {
      yolo: this.persons.live.some(
        (track) => track.atStand && now - track.lastSeen <= this.options.visibleMs,
      ),
      face: this.faces.live.some(
        (track) => track.atStand && now - track.lastSeen <= this.options.visibleMs,
      ),
      pose: poseFresh,
    };
    let people = Math.max(persons.length, faces.length);
    if (people === 0 && (poseFresh || this.present)) {
      people = this.present ? 1 : 0;
    }
    const any = sources.yolo || sources.face || sources.pose;
    const tracks: TrackedPerson[] = [
      ...this.persons.live.map((track) => ({
        kind: "persona" as const,
        id: track.id,
        size: track.size,
        atStand: track.atStand,
        greeted: track.greeted,
      })),
      ...this.faces.live.map((track) => ({
        kind: "volto" as const,
        id: track.id,
        size: track.size,
        atStand: track.atStand,
        greeted: track.greeted,
      })),
    ];
    return {
      present: this.present,
      sources,
      faces: faces.length,
      people,
      missingMs:
        this.present && !any && this.lastPositiveAt !== null ? now - this.lastPositiveAt : null,
      greeted: this.visitGreeted,
      waiting: this.waiting,
      thresholds: this.options.thresholds(),
      tracks,
    };
  }
}
