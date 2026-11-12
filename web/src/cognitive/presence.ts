/**
 * Presence fusion (C3): "is anybody there?" from every sensor at once.
 *
 * A single detector loses a person all the time: YOLOX misses "persona" when
 * somebody stands very close (only a face and a torso), YOLOX does not run at
 * all in the Scheletro view, and the laptop face tracker misses a turned head.
 * The fused state says a person is present when ANY source sees one:
 *
 * - **yolo** — the stable YOLOX "persona" track (debounced by `Perception`);
 * - **face** — at least one face from the laptop MediaPipe Face Landmarker;
 * - **pose** — a MoveNet skeleton with a score of at least 0.3.
 *
 * Absence is the opposite: ALL sources negative for `absenceMs` (5 s). A person
 * coming back within `visitGapMs` (15 s) of the last sighting continues the
 * same visit: no new greeting, no second goodbye. The goodbye itself is
 * allowed only after a greeting happened in this visit, at most once per
 * `farewellCooldownMs` (60 s) and never while a face is visible.
 *
 * Pure and DOM-free: the agent feeds it, the tests drive it with numbers.
 */

export type PresenceSourceName = "yolo" | "face" | "pose";

export interface PresenceOptions {
  /** All sources negative for this long before the person is gone. */
  absenceMs: number;
  /** A return within this gap after the last sighting is the same visit. */
  visitGapMs: number;
  /** At most one goodbye in this window. */
  farewellCooldownMs: number;
  /** A face or pose sample older than this no longer counts (tracker stopped). */
  sourceTtlMs: number;
  /** Minimum MoveNet pose score that counts as a person. */
  poseMinScore: number;
}

export const DEFAULT_PRESENCE_OPTIONS: PresenceOptions = {
  absenceMs: 5_000,
  visitGapMs: 15_000,
  farewellCooldownMs: 60_000,
  sourceTtlMs: 1_000,
  poseMinScore: 0.3,
};

export interface PresenceSnapshot {
  /** The fused state: somebody is in front of ARI. */
  present: boolean;
  /** Live state of every source (after the freshness check). */
  sources: Record<PresenceSourceName, boolean>;
  /** Faces in the last laptop sample (0 when stale). */
  faces: number;
  /** People for the greetings: YOLOX when it sees them, else the faces. */
  people: number;
  /** While present but every source is negative: how long it has been. */
  missingMs: number | null;
  /** The current presence continues an earlier visit (no new greeting). */
  sameVisit: boolean;
  /** A greeting was spoken in this visit. */
  greeted: boolean;
}

interface SourceSample {
  positive: boolean;
  at: number;
}

export class PresenceFusion {
  readonly options: PresenceOptions;

  private yolo: SourceSample = { positive: false, at: Number.NEGATIVE_INFINITY };
  private yoloPeople = 0;
  private face: SourceSample = { positive: false, at: Number.NEGATIVE_INFINITY };
  private faceCount = 0;
  private pose: SourceSample = { positive: false, at: Number.NEGATIVE_INFINITY };
  private present = false;
  private lastPositiveAt: number | null = null;
  private visitStartedAt: number | null = null;
  private continued = false;
  private greeted = false;
  private lastFarewellAt: number | null = null;
  private lastNow = 0;

  constructor(options: Partial<PresenceOptions> = {}) {
    this.options = { ...DEFAULT_PRESENCE_OPTIONS, ...options };
  }

  /**
   * The stable YOLOX person track and the settled YOLOX people count. Only
   * the track says "present": the settled count is itself held by this fusion
   * and would otherwise keep the person here forever.
   */
  observeYolo(personTrack: boolean, people: number, now: number): void {
    this.yolo = { positive: personTrack, at: now };
    this.yoloPeople = Math.max(0, people);
    this.update(now);
  }

  /** One laptop Face Landmarker sample: the number of faces in it. */
  observeFaces(count: number, now: number): void {
    this.faceCount = Math.max(0, count);
    this.face = { positive: count > 0, at: now };
    this.update(now);
  }

  /** One MoveNet `pose.state`: its overall score. */
  observePose(score: number, now: number): void {
    this.pose = { positive: score >= this.options.poseMinScore, at: now };
    this.update(now);
  }

  /** A greeting was spoken: the goodbye of this visit is now allowed. */
  noteGreeting(_now: number): void {
    this.greeted = true;
  }

  /** A goodbye was spoken: none again for `farewellCooldownMs`. */
  noteFarewell(now: number): void {
    this.lastFarewellAt = now;
  }

  /** Whether the goodbye may be said right now. */
  farewellAllowed(now: number): boolean {
    if (!this.greeted || this.facesVisible(now)) {
      return false;
    }
    return (
      this.lastFarewellAt === null ||
      now - this.lastFarewellAt >= this.options.farewellCooldownMs
    );
  }

  get isPresent(): boolean {
    return this.present;
  }

  /** Advance the fused state to `now` (sources may have gone stale). */
  update(now: number): PresenceSnapshot {
    this.lastNow = Math.max(this.lastNow, now);
    const sources = this.sources(now);
    const any = sources.yolo || sources.face || sources.pose;
    if (any) {
      if (!this.present) {
        const gap = this.lastPositiveAt === null ? Number.POSITIVE_INFINITY : now - this.lastPositiveAt;
        if (this.visitStartedAt !== null && gap <= this.options.visitGapMs) {
          this.continued = true;
        } else {
          this.visitStartedAt = now;
          this.continued = false;
          this.greeted = false;
        }
        this.present = true;
      }
      this.lastPositiveAt = now;
    } else if (
      this.present &&
      this.lastPositiveAt !== null &&
      now - this.lastPositiveAt >= this.options.absenceMs
    ) {
      this.present = false;
    }
    return this.snapshotFor(now, sources);
  }

  snapshot(now: number = this.lastNow): PresenceSnapshot {
    return this.snapshotFor(now, this.sources(now));
  }

  private snapshotFor(
    now: number,
    sources: Record<PresenceSourceName, boolean>,
  ): PresenceSnapshot {
    const faces = sources.face ? this.faceCount : 0;
    let people = 0;
    if (sources.yolo && this.yoloPeople > 0) {
      people = this.yoloPeople;
    } else if (faces > 0) {
      people = faces;
    } else if (this.present) {
      // Held through a short gap, or seen by the skeleton only.
      people = Math.max(1, sources.pose ? 1 : this.yoloPeople);
    }
    const any = sources.yolo || sources.face || sources.pose;
    return {
      present: this.present,
      sources,
      faces,
      people,
      missingMs:
        this.present && !any && this.lastPositiveAt !== null ? now - this.lastPositiveAt : null,
      sameVisit: this.present && this.continued,
      greeted: this.greeted,
    };
  }

  private facesVisible(now: number): boolean {
    return this.fresh(this.face, now) && this.faceCount > 0;
  }

  private sources(now: number): Record<PresenceSourceName, boolean> {
    return {
      yolo: this.fresh(this.yolo, now),
      face: this.fresh(this.face, now) && this.faceCount > 0,
      pose: this.fresh(this.pose, now),
    };
  }

  private fresh(sample: SourceSample, now: number): boolean {
    return sample.positive && now - sample.at <= this.options.sourceTtlMs;
  }
}
