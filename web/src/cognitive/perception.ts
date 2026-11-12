/**
 * Debounced world model of the cognitive layer (F-E3 `perception.ts`).
 *
 * `showcase.state` arrives at ≤ 10 Hz and the rules must not see every
 * flicker. A class is present when it appears in at least 6 of the last 10
 * states and absent after 2 s unseen; the number of people is settled for 1.5 s
 * before a `count` event exists, so a 0→3 arrival is one collective event
 * (F-E4 #1) and a noisy 2↔3 count never fires (F-E4 #3). A class that was
 * already there at start/reconnect, or present for two minutes, is scenery and
 * is never commented (F-E4 #9); implausible indoor classes need a high score
 * for two seconds and produce a doubtful line (F-E4 #10).
 */

import type { PoseStatePayload, ShowcaseObject, ShowcaseStatePayload } from "../protocol";
import type { CognitiveEvent, PoseKind, WorldModelName } from "./events";

export type Box = [number, number, number, number];

export interface PerceptionOptions {
  enterFrames: number;
  windowFrames: number;
  absenceMs: number;
  countSettleMs: number;
  warmupMs: number;
  sceneryContinuousMs: number;
  sceneryReturnMs: number;
  minScore: number;
  minAreaRatio: number;
  implausibleScore: number;
  implausibleHoldMs: number;
  returnMs: number;
  dwellThresholds: readonly number[];
  idleThresholds: readonly number[];
  personDwellThresholds: readonly number[];
  poseHoldMs: number;
  poseCooldownMs: number;
  waveWindowMs: number;
  waveAmplitudeRatio: number;
  keypointMinScore: number;
  poseMinScore: number;
}

export const DEFAULT_PERCEPTION_OPTIONS: PerceptionOptions = {
  // ~0.4 s at the board's ~10 Hz showcase.state: quick, still debounced (T1).
  enterFrames: 4,
  windowFrames: 6,
  absenceMs: 1_500,
  countSettleMs: 800,
  warmupMs: 3_000,
  sceneryContinuousMs: 120_000,
  sceneryReturnMs: 30_000,
  minScore: 0.5,
  minAreaRatio: 0.0015,
  implausibleScore: 0.7,
  implausibleHoldMs: 2_000,
  returnMs: 30_000,
  dwellThresholds: [10, 30, 60, 120],
  idleThresholds: [15, 60, 180],
  personDwellThresholds: [25, 60, 120],
  poseHoldMs: 500,
  poseCooldownMs: 10_000,
  waveWindowMs: 1_500,
  waveAmplitudeRatio: 0.15,
  keypointMinScore: 0.3,
  poseMinScore: 0.3,
};

/** Classes a living room should almost never contain (F-E4 #10). */
export const IMPLAUSIBLE_INDOOR: ReadonlySet<string> = new Set([
  "elefante",
  "giraffa",
  "zebra",
  "orso",
  "aereo",
  "treno",
  "barca",
  "mucca",
  "cavallo",
  "pecora",
]);

/** Dangerous objects get a gentle, low-frequency line (F-E4 #12). */
export const SHARP_CLASSES: ReadonlySet<string> = new Set(["coltello", "forbici"]);

interface Track {
  label: string;
  frames: boolean[];
  present: boolean;
  count: number;
  score: number;
  box: Box | null;
  area: number;
  firstSeenAt: number | null;
  presentSince: number | null;
  lastSeenAt: number;
  lostAt: number | null;
  scenery: boolean;
  doubtful: boolean;
  /** Already present when the session started (F-E4 #9). */
  seeded: boolean;
  dwellEmitted: Set<number>;
}

interface Observation {
  label: string;
  count: number;
  score: number;
  box: Box;
  area: number;
  doubtful: boolean;
}

interface PoseHold {
  since: number | null;
  emitted: boolean;
}

function newTrack(label: string, now: number): Track {
  return {
    label,
    frames: [],
    present: false,
    count: 0,
    score: 0,
    box: null,
    area: 0,
    firstSeenAt: null,
    presentSince: null,
    lastSeenAt: now,
    lostAt: null,
    scenery: false,
    doubtful: false,
    seeded: false,
    dwellEmitted: new Set(),
  };
}

function areaOf(box: Box): number {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

export class Perception {
  readonly options: PerceptionOptions;
  private readonly tracks = new Map<string, Track>();
  private readonly implausibleSince = new Map<string, number>();
  private readonly pairEmitted = new Set<string>();
  private readonly poseHolds = new Map<PoseKind, PoseHold>();
  private readonly poseCooldown = new Map<PoseKind, number>();
  private poseSamples: Array<{ t: number; x: number }> = [];

  private mode: WorldModelName = "yolox";
  private sessionStartedAt: number | null = null;
  private warmupUntil = 0;
  private stablePeople = 0;
  private pendingPeople: { count: number; since: number } | null = null;
  private lastPeopleClearAt: number | null = null;
  private lastPeopleAt: number | null = null;
  private personPresentSince: number | null = null;
  private readonly idleEmitted = new Set<number>();
  private largestPerson: Box | null = null;
  private rawPeople = 0;
  private posePerson = false;

  constructor(options: Partial<PerceptionOptions> = {}) {
    this.options = { ...DEFAULT_PERCEPTION_OPTIONS, ...options };
  }

  get world(): {
    mode: WorldModelName;
    people: number;
    rawPeople: number;
    crowd: boolean;
    largestPerson: Box | null;
    /** MoveNet sees a body: MoveNet has no count, only presence (F-E4 #22). */
    posePerson: boolean;
  } {
    return {
      mode: this.mode,
      people: this.stablePeople,
      rawPeople: this.rawPeople,
      crowd: this.stablePeople >= 5,
      largestPerson: this.largestPerson,
      posePerson: this.posePerson,
    };
  }

  /** Read-only track of a class, for the face and the tests. */
  track(label: string): Readonly<Track> | null {
    return this.tracks.get(label) ?? null;
  }

  /** Labels currently present, for the engine's pre-speech revalidation. */
  presentLabels(): string[] {
    return [...this.tracks.values()]
      .filter((track) => track.present)
      .map((track) => track.label);
  }

  /**
   * Forget the world without emitting a single `disappear`: a mode switch or a
   * reconnect must not produce a storm (F-E4 #20/#22).
   */
  reset(now: number, mode: WorldModelName = this.mode): void {
    this.tracks.clear();
    this.implausibleSince.clear();
    this.pairEmitted.clear();
    this.poseHolds.clear();
    this.poseCooldown.clear();
    this.poseSamples = [];
    this.mode = mode;
    this.sessionStartedAt = now;
    this.warmupUntil = now + this.options.warmupMs;
    this.stablePeople = 0;
    this.pendingPeople = null;
    this.lastPeopleClearAt = null;
    this.lastPeopleAt = now;
    this.personPresentSince = null;
    this.idleEmitted.clear();
    this.largestPerson = null;
    this.rawPeople = 0;
    this.posePerson = false;
  }

  /** Feed one `showcase.state`; returns the events it produced. */
  update(state: ShowcaseStatePayload, now: number): CognitiveEvent[] {
    if (this.sessionStartedAt === null) {
      this.reset(now, state.model === "movenet" ? "movenet" : "yolox");
    }
    if (this.mode !== "yolox" || state.model === "movenet") {
      return [];
    }
    const events: CognitiveEvent[] = [];
    const frameArea = Math.max(1, state.width * state.height);
    const observations = this.qualify(state.objects, frameArea, now);
    const warmup = now < this.warmupUntil;

    for (const observation of observations.values()) {
      if (!this.tracks.has(observation.label)) {
        this.tracks.set(observation.label, newTrack(observation.label, now));
      }
    }

    for (const track of this.tracks.values()) {
      const observation = observations.get(track.label) ?? null;
      const hit = observation !== null;
      track.frames.push(hit);
      if (track.frames.length > this.options.windowFrames) {
        track.frames.shift();
      }
      if (hit && observation !== null) {
        track.lastSeenAt = now;
        track.count = observation.count;
        track.score = observation.score;
        track.box = observation.box;
        track.area = observation.area;
        track.doubtful = observation.doubtful;
        if (track.firstSeenAt === null) {
          track.firstSeenAt = now;
          track.seeded = warmup;
        }
      }

      if (!track.present) {
        const hits = track.frames.filter(Boolean).length;
        if (hit && hits >= this.options.enterFrames) {
          const absence = track.lostAt === null ? null : now - track.lostAt;
          this.appear(track, now, absence, events, warmup);
        }
      } else if (now - track.lastSeenAt >= this.options.absenceMs) {
        track.present = false;
        track.lostAt = now;
        if (!warmup && this.mode === "yolox") {
          events.push({
            type: "disappear",
            label: track.label,
            absentS: (now - track.lastSeenAt) / 1000,
            now,
          });
        }
        this.dropPairs(track.label);
      }
    }

    const people = this.countPeople(state.objects, frameArea);
    this.rawPeople = people.count;
    this.largestPerson = people.largest;
    if (warmup) {
      this.stablePeople = this.rawPeople;
      this.pendingPeople = null;
      if (this.rawPeople > 0) {
        this.lastPeopleAt = now;
        this.personPresentSince ??= now;
        this.clearPersonDwell();
      }
    } else {
      this.settlePeople(now, events);
    }

    this.promoteScenery(now);
    return events;
  }

  /** Feed one `pose.state`; returns confirmed gestures only (F-E4 #25). */
  updatePose(pose: PoseStatePayload, now: number): CognitiveEvent[] {
    if (pose.score < this.options.poseMinScore) {
      this.poseHolds.clear();
      this.poseSamples = [];
      this.posePerson = false;
      return [];
    }
    const points = pose.keypoints.map(([x, y, score]) =>
      score >= this.options.keypointMinScore ? ([x, y] as const) : null,
    );
    const nose = points[0];
    const leftShoulder = points[5];
    const rightShoulder = points[6];
    if (leftShoulder === null || rightShoulder === null || nose === null) {
      this.poseHolds.clear();
      this.poseSamples = [];
      this.posePerson = false;
      return [];
    }
    this.posePerson = true;
    const width = Math.max(1, Math.abs(rightShoulder[0] - leftShoulder[0]));
    const shoulderMidX = (leftShoulder[0] + rightShoulder[0]) / 2;
    const shoulderMidY = (leftShoulder[1] + rightShoulder[1]) / 2;
    const leftWrist = points[9];
    const rightWrist = points[10];
    const leftHip = points[11];
    const rightHip = points[12];

    const raised = (
      wrist: readonly [number, number] | null,
      shoulder: readonly [number, number],
    ): boolean => wrist !== null && wrist[1] < shoulder[1] - width * 0.05;
    const bothRaised = raised(leftWrist, leftShoulder) && raised(rightWrist, rightShoulder);
    const armsUp =
      bothRaised &&
      leftWrist !== null &&
      rightWrist !== null &&
      Math.abs(leftWrist[0] - nose[0]) >= width * 0.6 &&
      Math.abs(rightWrist[0] - nose[0]) >= width * 0.6;
    const handsOnHead =
      bothRaised &&
      leftWrist !== null &&
      rightWrist !== null &&
      Math.hypot(leftWrist[0] - nose[0], leftWrist[1] - nose[1]) < width &&
      Math.hypot(rightWrist[0] - nose[0], rightWrist[1] - nose[1]) < width;
    const tPose =
      leftWrist !== null &&
      rightWrist !== null &&
      Math.abs(leftWrist[1] - leftShoulder[1]) < width * 0.2 &&
      Math.abs(rightWrist[1] - rightShoulder[1]) < width * 0.2 &&
      Math.abs(leftWrist[0] - leftShoulder[0]) > width * 1.1 &&
      Math.abs(rightWrist[0] - rightShoulder[0]) > width * 1.1;
    const hipMidY =
      leftHip !== null && rightHip !== null ? (leftHip[1] + rightHip[1]) / 2 : null;
    const hipMidX =
      leftHip !== null && rightHip !== null ? (leftHip[0] + rightHip[0]) / 2 : null;
    const crouch = hipMidY !== null && hipMidY - shoulderMidY < width * 0.9;
    const lean = hipMidX !== null && Math.abs(shoulderMidX - hipMidX) > width * 0.4;

    const events: CognitiveEvent[] = [];
    this.holdGesture("arms_up", armsUp && !handsOnHead, now, pose.score, events);
    this.holdGesture("hands_on_head", handsOnHead, now, pose.score, events);
    this.holdGesture("t_pose", tPose, now, pose.score, events);
    this.holdGesture("crouch", crouch, now, pose.score, events);
    this.holdGesture("lean", lean, now, pose.score, events);
    this.wave(now, points, width, pose.score, events);
    return events;
  }

  /** Time-driven events: dwell milestones, idle, pairs, scenery promotion. */
  tick(now: number): CognitiveEvent[] {
    if (this.sessionStartedAt === null) {
      return [];
    }
    const events: CognitiveEvent[] = [];
    this.promoteScenery(now);

    for (const track of this.tracks.values()) {
      if (
        !track.present ||
        track.scenery ||
        track.presentSince === null ||
        track.label === "persona" ||
        this.stablePeople >= 5
      ) {
        continue;
      }
      for (const threshold of this.options.dwellThresholds) {
        if (now - track.presentSince >= threshold * 1000 && !track.dwellEmitted.has(threshold)) {
          track.dwellEmitted.add(threshold);
          events.push({
            type: "dwell",
            label: track.label,
            seconds: threshold,
            count: track.count,
            now,
          });
        }
      }
    }

    events.push(...this.together(now));
    events.push(...this.idle(now));
    return events;
  }

  // --- internals -----------------------------------------------------------

  private qualify(
    objects: readonly ShowcaseObject[],
    frameArea: number,
    now: number,
  ): Map<string, Observation> {
    const grouped = new Map<
      string,
      { label: string; count: number; score: number; box: Box; area: number; best: number }
    >();
    for (const object of objects) {
      if (object.score < this.options.minScore) {
        continue;
      }
      const area = areaOf(object.box) / frameArea;
      if (area < this.options.minAreaRatio) {
        continue;
      }
      const current = grouped.get(object.label);
      if (current === undefined) {
        grouped.set(object.label, {
          label: object.label,
          count: 1,
          score: object.score,
          box: object.box,
          area,
          best: object.score,
        });
      } else {
        current.count += 1;
        current.best = Math.max(current.best, object.score);
        if (object.score > current.score) {
          current.score = object.score;
          current.box = object.box;
          current.area = area;
        }
      }
    }

    const observations = new Map<string, Observation>();
    for (const [label, observation] of grouped) {
      if (!IMPLAUSIBLE_INDOOR.has(label)) {
        observations.set(label, {
          label,
          count: observation.count,
          score: observation.score,
          box: observation.box,
          area: observation.area,
          doubtful: false,
        });
        continue;
      }
      // False-positive gate: a high score held for two seconds (F-E4 #10).
      if (observation.best < this.options.implausibleScore) {
        this.implausibleSince.delete(label);
        continue;
      }
      const since = this.implausibleSince.get(label) ?? now;
      this.implausibleSince.set(label, since);
      if (now - since < this.options.implausibleHoldMs) {
        continue;
      }
      observations.set(label, {
        label,
        count: observation.count,
        score: observation.score,
        box: observation.box,
        area: observation.area,
        doubtful: true,
      });
    }
    return observations;
  }

  private appear(
    track: Track,
    now: number,
    absence: number | null,
    events: CognitiveEvent[],
    warmup: boolean,
  ): void {
    track.present = true;
    track.presentSince = now;
    track.firstSeenAt ??= now;
    track.lostAt = null;
    track.dwellEmitted.clear();

    if (absence !== null && absence >= this.options.sceneryReturnMs) {
      // A genuine return makes even a scenery class commentable again (#9).
      track.scenery = false;
      track.seeded = false;
    } else if (track.seeded) {
      track.scenery = true;
    }

    if (warmup || this.mode !== "yolox") {
      return;
    }
    events.push({
      type: "appear",
      label: track.label,
      count: track.count,
      score: track.score,
      area: track.area,
      scenery: track.scenery,
      doubtful: track.doubtful,
      now,
    });
    if (
      !track.scenery &&
      absence !== null &&
      absence >= this.options.returnMs &&
      track.label !== "persona"
    ) {
      events.push({
        type: "return",
        label: track.label,
        absenceS: absence / 1000,
        count: track.count,
        now,
      });
    }
  }

  /** Promote a class present for two minutes to scenery (F-E4 #9). */
  private promoteScenery(now: number): void {
    for (const track of this.tracks.values()) {
      if (
        track.present &&
        !track.scenery &&
        track.presentSince !== null &&
        now - track.presentSince >= this.options.sceneryContinuousMs
      ) {
        track.scenery = true;
      }
    }
  }

  private countPeople(
    objects: readonly ShowcaseObject[],
    frameArea: number,
  ): { count: number; largest: Box | null } {
    let count = 0;
    let largest: Box | null = null;
    let largestArea = 0;
    for (const object of objects) {
      if (object.label !== "persona" || object.score < this.options.minScore) {
        continue;
      }
      const area = areaOf(object.box) / frameArea;
      if (area < this.options.minAreaRatio * 0.5) {
        continue;
      }
      count += 1;
      if (area > largestArea) {
        largestArea = area;
        largest = object.box;
      }
    }
    return { count, largest };
  }

  private settlePeople(now: number, events: CognitiveEvent[]): void {
    if (this.rawPeople === this.stablePeople) {
      this.pendingPeople = null;
      return;
    }
    if (this.pendingPeople === null || this.pendingPeople.count !== this.rawPeople) {
      this.pendingPeople = { count: this.rawPeople, since: now };
      return;
    }
    if (now - this.pendingPeople.since < this.options.countSettleMs) {
      return;
    }
    const from = this.stablePeople;
    const to = this.rawPeople;
    this.stablePeople = to;
    this.pendingPeople = null;

    let quiet = false;
    let returnAfterS: number | null = null;
    if (from === 0 && to > 0) {
      const absence = this.lastPeopleClearAt === null ? null : now - this.lastPeopleClearAt;
      if (absence !== null) {
        quiet = absence < this.options.returnMs;
        if (!quiet) {
          returnAfterS = absence / 1000;
        }
      }
      this.personPresentSince ??= now;
      this.clearPersonDwell();
    } else if (to === 0) {
      this.lastPeopleClearAt = now;
      this.personPresentSince = null;
      this.idleEmitted.clear();
    }
    if (to > 0) {
      this.lastPeopleAt = now;
    }
    events.push({ type: "count", from, to, quiet, returnAfterS, now });
  }

  private clearPersonDwell(): void {
    for (const threshold of this.options.personDwellThresholds) {
      this.idleEmitted.delete(-threshold);
    }
  }

  private dropPairs(label: string): void {
    for (const key of [...this.pairEmitted]) {
      if (key.split("|").includes(label)) {
        this.pairEmitted.delete(key);
      }
    }
  }

  private together(now: number): CognitiveEvent[] {
    const events: CognitiveEvent[] = [];
    const present = [...this.tracks.values()].filter(
      (track) =>
        track.present &&
        !track.scenery &&
        track.presentSince !== null &&
        now - track.presentSince >= 2_000 &&
        track.label !== "persona",
    );
    const candidates = present.map((track) => track.label);
    if (this.stablePeople === 1) {
      candidates.push("persona");
    }
    candidates.sort();
    for (let i = 0; i < candidates.length; i += 1) {
      for (let j = i + 1; j < candidates.length; j += 1) {
        const a = candidates[i];
        const b = candidates[j];
        const key = `${a}|${b}`;
        if (this.pairEmitted.has(key)) {
          continue;
        }
        this.pairEmitted.add(key);
        events.push({
          type: "together",
          a,
          b,
          countA: a === "persona" ? this.stablePeople : (this.tracks.get(a)?.count ?? 1),
          countB: b === "persona" ? this.stablePeople : (this.tracks.get(b)?.count ?? 1),
          now,
        });
        if (events.length >= 3) {
          return events;
        }
      }
    }
    return events;
  }

  private idle(now: number): CognitiveEvent[] {
    const events: CognitiveEvent[] = [];
    if (this.stablePeople === 0) {
      const since = this.lastPeopleAt ?? this.sessionStartedAt ?? now;
      const idleS = (now - since) / 1000;
      for (const threshold of this.options.idleThresholds) {
        if (idleS >= threshold && !this.idleEmitted.has(threshold)) {
          this.idleEmitted.add(threshold);
          events.push({ type: "idle", seconds: threshold, now });
        }
      }
      return events;
    }
    if (this.personPresentSince !== null) {
      const personS = (now - this.personPresentSince) / 1000;
      for (const threshold of this.options.personDwellThresholds) {
        if (personS >= threshold && !this.idleEmitted.has(-threshold)) {
          this.idleEmitted.add(-threshold);
          events.push({
            type: "dwell",
            label: "persona",
            seconds: threshold,
            count: this.stablePeople,
            now,
          });
        }
      }
    }
    return events;
  }

  private holdGesture(
    kind: PoseKind,
    condition: boolean,
    now: number,
    score: number,
    events: CognitiveEvent[],
  ): void {
    const hold = this.poseHolds.get(kind) ?? { since: null, emitted: false };
    this.poseHolds.set(kind, hold);
    if (!condition) {
      hold.since = null;
      hold.emitted = false;
      return;
    }
    hold.since ??= now;
    if (
      !hold.emitted &&
      now - hold.since >= this.options.poseHoldMs &&
      now >= (this.poseCooldown.get(kind) ?? 0)
    ) {
      hold.emitted = true;
      this.poseCooldown.set(kind, now + this.options.poseCooldownMs);
      events.push({ type: "pose", kind, score, now });
    }
  }

  private wave(
    now: number,
    points: ReadonlyArray<readonly [number, number] | null>,
    width: number,
    score: number,
    events: CognitiveEvent[],
  ): void {
    const leftShoulder = points[5];
    const rightShoulder = points[6];
    const leftWrist = points[9];
    const rightWrist = points[10];
    if (leftShoulder === null || rightShoulder === null) {
      this.poseSamples = [];
      return;
    }
    let wrist: readonly [number, number] | null = null;
    if (rightWrist !== null && rightWrist[1] < rightShoulder[1] - width * 0.05) {
      wrist = rightWrist;
    } else if (leftWrist !== null && leftWrist[1] < leftShoulder[1] - width * 0.05) {
      wrist = leftWrist;
    }
    if (wrist === null) {
      this.poseSamples = [];
      return;
    }
    this.poseSamples.push({ t: now, x: wrist[0] });
    this.poseSamples = this.poseSamples.filter(
      (sample) => now - sample.t <= this.options.waveWindowMs,
    );
    if (this.poseSamples.length < 3) {
      return;
    }
    const xs = this.poseSamples.map((sample) => sample.x);
    const range = Math.max(...xs) - Math.min(...xs);
    if (range < width * this.options.waveAmplitudeRatio) {
      return;
    }
    let reversals = 0;
    let direction = 0;
    for (let index = 1; index < xs.length; index += 1) {
      const delta = xs[index] - xs[index - 1];
      if (Math.abs(delta) < 1) {
        continue;
      }
      const next = delta > 0 ? 1 : -1;
      if (direction !== 0 && next !== direction) {
        reversals += 1;
      }
      direction = next;
    }
    if (reversals >= 2 && now >= (this.poseCooldown.get("wave") ?? 0)) {
      this.poseCooldown.set("wave", now + this.options.poseCooldownMs);
      this.poseSamples = [];
      events.push({ type: "pose", kind: "wave", score, now });
    }
  }
}
