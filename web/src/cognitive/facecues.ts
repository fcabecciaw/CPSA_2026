/**
 * Smile, mouth, nod, head-shake and "looking at me" detectors (R2-D).
 *
 * Pure and DOM-free: `FaceCueDetector` consumes synthetic or real
 * `FaceObservation`s (MediaPipe blendshapes + head angles) and returns at most
 * one cue per stream, with holds, hysteresis and cooldowns. `facepipeline.ts`
 * feeds it from the laptop browser; the vitest suite feeds it synthetic
 * blendshape streams.
 *
 * The head angles are relative to a short calibration at the start of the
 * session, so the detectors work with any camera angle: a nod is a swing in
 * pitch, a head shake a swing in yaw, "looking at me" is staying near the
 * calibrated centre.
 *
 * Smiles are relative too (C3): every face keeps its own neutral baseline, the
 * running median of max(mouthSmileLeft, mouthSmileRight) over the last ~3 s
 * while it is not smiling. A smile is the baseline + 0.20 (or 0.55 absolute)
 * held for 0.3 s, so a face with a high neutral does not fire by itself and a
 * small, shy smile on a low neutral still does.
 */

import type { FaceCueKind } from "./events";

/** One face in one sampled frame. */
export interface FaceObservation {
  /** MediaPipe blendshape name ("mouthSmileLeft", "jawOpen"…) → confidence. */
  blendshapes: Record<string, number>;
  /** Head yaw in degrees, 0 when looking at the camera. */
  yaw: number;
  /** Head pitch in degrees, 0 when level with the camera. */
  pitch: number;
  /**
   * Face box `[x0, y0, x1, y1]` normalised to the full frame, to follow the
   * same face across samples. Without it the face index is the identity.
   */
  box?: readonly [number, number, number, number];
}

export interface FaceFrame {
  now: number;
  faces: readonly FaceObservation[];
}

export interface FaceCue {
  kind: FaceCueKind;
  /** Confidence of the cue in [0, 1]. */
  score: number;
  /** Faces seen in the frame. */
  faces: number;
  at: number;
}

export interface FaceCueOptions {
  /** A smile is this much above the face's neutral baseline… */
  smileDelta: number;
  /** …or at least this, whatever the baseline. */
  smileAbsolute: number;
  smileHoldMs: number;
  /** The smile resets this much below its threshold (hysteresis). */
  smileReleaseDelta: number;
  /** Window of the neutral baseline (running median). */
  smileBaselineMs: number;
  /** Neutral samples needed before the baseline is trusted. */
  smileBaselineMinSamples: number;
  /** A face unseen for longer than this loses its baseline and hold. */
  smileFaceGapMs: number;
  /** Max centre distance (normalised) to follow a face between samples. */
  smileMatchDistance: number;
  jawOpenThreshold: number;
  mouthHoldMs: number;
  /** Swing amplitude that counts as a nod / head shake, degrees. */
  nodAmplitude: number;
  shakeAmplitude: number;
  /** Window where the swing must complete. */
  swingWindowMs: number;
  /** Max distance from the calibrated centre to count as "looking at me". */
  lookYaw: number;
  lookPitch: number;
  lookingHoldMs: number;
  /** Calibration time at the start, in which no cue can fire. */
  calibrateMs: number;
  cooldowns: Record<FaceCueKind, number>;
}

const DEFAULTS: FaceCueOptions = {
  smileDelta: 0.2,
  smileAbsolute: 0.55,
  smileHoldMs: 300,
  smileReleaseDelta: 0.1,
  smileBaselineMs: 3_000,
  smileBaselineMinSamples: 5,
  smileFaceGapMs: 500,
  smileMatchDistance: 0.2,
  jawOpenThreshold: 0.5,
  mouthHoldMs: 300,
  nodAmplitude: 6,
  shakeAmplitude: 8,
  swingWindowMs: 1_500,
  lookYaw: 12,
  lookPitch: 12,
  lookingHoldMs: 1_200,
  calibrateMs: 400,
  cooldowns: {
    smile: 6_000,
    mouth_open: 12_000,
    nod: 12_000,
    head_shake: 12_000,
    looking_at_me: 30_000,
  },
};

/** The live smile reading of the strongest face, for the debug readout. */
export interface SmileReadout {
  value: number;
  threshold: number;
  /** `null` until enough neutral samples were seen (absolute threshold only). */
  baseline: number | null;
}

/** One face followed across samples, with its own neutral baseline. */
interface SmileTrack {
  index: number;
  centre: { x: number; y: number } | null;
  lastSeen: number;
  neutral: Array<{ t: number; v: number }>;
  since: number | null;
  fired: boolean;
  readout: SmileReadout;
}

function smileOf(face: FaceObservation): number {
  return Math.max(face.blendshapes.mouthSmileLeft ?? 0, face.blendshapes.mouthSmileRight ?? 0);
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]!
    : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

interface SwingSample {
  t: number;
  value: number;
}

/** One axis of head motion, with hysteresis, used for nod and head shake. */
class AxisSwing {
  private readonly samples: SwingSample[] = [];

  constructor(
    private readonly amplitude: number,
    private readonly windowMs: number,
  ) {}

  reset(): void {
    this.samples.length = 0;
  }

  /**
   * Add a sample; returns the swing amplitude when the axis completed an
   * out-and-back movement inside the window, `null` otherwise.
   */
  push(now: number, value: number): number | null {
    this.samples.push({ t: now, value });
    while (this.samples.length > 0 && now - this.samples[0]!.t > this.windowMs) {
      this.samples.shift();
    }
    if (this.samples.length < 3) {
      return null;
    }
    let direction = 0;
    let extreme = this.samples[0]!.value;
    let reversals = 0;
    let lowest = extreme;
    let highest = extreme;
    for (const sample of this.samples) {
      lowest = Math.min(lowest, sample.value);
      highest = Math.max(highest, sample.value);
      if (direction === 0) {
        if (Math.abs(sample.value - extreme) >= this.amplitude) {
          direction = Math.sign(sample.value - extreme);
          extreme = sample.value;
        }
      } else if (direction > 0) {
        if (sample.value > extreme) {
          extreme = sample.value;
        } else if (extreme - sample.value >= this.amplitude) {
          reversals += 1;
          direction = -1;
          extreme = sample.value;
        }
      } else if (sample.value < extreme) {
        extreme = sample.value;
      } else if (sample.value - extreme >= this.amplitude) {
        reversals += 1;
        direction = 1;
        extreme = sample.value;
      }
    }
    if (reversals < 1 || highest - lowest < this.amplitude * 2) {
      return null;
    }
    this.samples.length = 0;
    return highest - lowest;
  }
}

export class FaceCueDetector {
  readonly options: FaceCueOptions;

  private readonly nod: AxisSwing;
  private readonly shake: AxisSwing;
  private lastFired = new Map<FaceCueKind, number>();
  private smileTracks: SmileTrack[] = [];
  private smileReadout: SmileReadout | null = null;
  private mouthSince: number | null = null;
  private mouthFired = false;
  private lookingSince: number | null = null;
  private lookingFired = false;
  private baselineYaw: number | null = null;
  private baselinePitch: number | null = null;
  private firstFrameAt: number | null = null;
  private calibrated = false;
  private calibrationSamples = 0;

  constructor(options: Partial<FaceCueOptions> = {}) {
    this.options = { ...DEFAULTS, ...options };
    this.nod = new AxisSwing(this.options.nodAmplitude, this.options.swingWindowMs);
    this.shake = new AxisSwing(this.options.shakeAmplitude, this.options.swingWindowMs);
  }

  /** The strongest face's smile against its threshold, `null` with no face. */
  get smile(): SmileReadout | null {
    return this.smileReadout;
  }

  /** Forget holds and calibration (e.g. the camera restarted). */
  reset(): void {
    this.nod.reset();
    this.shake.reset();
    this.smileTracks = [];
    this.smileReadout = null;
    this.mouthSince = null;
    this.mouthFired = false;
    this.lookingSince = null;
    this.lookingFired = false;
    this.baselineYaw = null;
    this.baselinePitch = null;
    this.firstFrameAt = null;
    this.calibrated = false;
  }

  update(frame: FaceFrame): FaceCue[] {
    // Smile tracks survive a short dropout (a missed sample is not a new face).
    const smileCandidates = this.updateSmiles(frame);
    if (frame.faces.length === 0) {
      this.mouthSince = null;
      this.mouthFired = false;
      this.lookingSince = null;
      this.lookingFired = false;
      return [];
    }
    const faces = frame.faces;
    this.firstFrameAt ??= frame.now;
    if (!this.calibrated) {
      if (this.baselineYaw === null || this.baselinePitch === null) {
        this.baselineYaw = 0;
        this.baselinePitch = 0;
        this.calibrationSamples = 0;
      }
      this.calibrationSamples += 1;
      // A short average, then the baseline is frozen: a nod must not move it.
      const weight = 1 / this.calibrationSamples;
      this.baselineYaw += (faces[0]!.yaw - this.baselineYaw) * weight;
      this.baselinePitch += (faces[0]!.pitch - this.baselinePitch) * weight;
      if (frame.now - this.firstFrameAt < this.options.calibrateMs) {
        return [];
      }
      this.calibrated = true;
    }

    const cues: FaceCue[] = [];
    const jawOpen = Math.max(...faces.map((face) => face.blendshapes.jawOpen ?? 0));
    const primary = faces[0]!;
    const pitch = primary.pitch - (this.baselinePitch ?? 0);
    const yaw = primary.yaw - (this.baselineYaw ?? 0);

    if (smileCandidates.length > 0 && this.ready("smile", frame.now)) {
      // One cue for everybody smiling at once; each face fires once per smile.
      let score = 0;
      for (const track of smileCandidates) {
        track.fired = true;
        score = Math.max(score, track.readout.value);
      }
      cues.push({ kind: "smile", score: Math.min(1, score), faces: faces.length, at: frame.now });
    }

    const mouthCue = this.hold(
      "mouth_open",
      frame.now,
      jawOpen,
      this.options.jawOpenThreshold,
      this.options.jawOpenThreshold * 0.6,
      this.options.mouthHoldMs,
      this.mouthSince,
      this.mouthFired,
      (since, fired) => {
        this.mouthSince = since;
        this.mouthFired = fired;
      },
    );
    if (mouthCue !== null) {
      cues.push({
        kind: "mouth_open",
        score: mouthCue,
        faces: faces.length,
        at: frame.now,
      });
    }

    const nod = this.nod.push(frame.now, pitch);
    if (nod !== null && this.ready("nod", frame.now)) {
      cues.push({
        kind: "nod",
        score: Math.min(1, nod / (this.options.nodAmplitude * 4)),
        faces: faces.length,
        at: frame.now,
      });
    }
    const shake = this.shake.push(frame.now, yaw);
    if (shake !== null && this.ready("head_shake", frame.now)) {
      cues.push({
        kind: "head_shake",
        score: Math.min(1, shake / (this.options.shakeAmplitude * 4)),
        faces: faces.length,
        at: frame.now,
      });
    }

    const looking =
      Math.abs(yaw) <= this.options.lookYaw && Math.abs(pitch) <= this.options.lookPitch;
    if (!looking) {
      this.lookingSince = null;
      this.lookingFired = false;
    } else if (this.lookingSince === null) {
      this.lookingSince = frame.now;
    } else if (
      !this.lookingFired &&
      frame.now - this.lookingSince >= this.options.lookingHoldMs &&
      this.ready("looking_at_me", frame.now)
    ) {
      this.lookingFired = true;
      const distance = Math.max(
        Math.abs(yaw) / this.options.lookYaw,
        Math.abs(pitch) / this.options.lookPitch,
      );
      cues.push({
        kind: "looking_at_me",
        score: Math.max(0, 1 - distance),
        faces: faces.length,
        at: frame.now,
      });
    }

    return cues;
  }

  /**
   * Follow every face, update its neutral baseline and its smile hold; returns
   * the faces whose smile has been held long enough and has not fired yet.
   */
  private updateSmiles(frame: FaceFrame): SmileTrack[] {
    const now = frame.now;
    const options = this.options;
    this.smileTracks = this.smileTracks.filter(
      (track) => now - track.lastSeen <= options.smileFaceGapMs,
    );
    const unmatched = new Set(this.smileTracks);
    const candidates: SmileTrack[] = [];
    let strongest: SmileReadout | null = null;
    frame.faces.forEach((face, index) => {
      const centre =
        face.box === undefined
          ? null
          : { x: (face.box[0] + face.box[2]) / 2, y: (face.box[1] + face.box[3]) / 2 };
      let track: SmileTrack | null = null;
      let best = Number.POSITIVE_INFINITY;
      for (const candidate of unmatched) {
        if (centre === null || candidate.centre === null) {
          if (centre === null && candidate.centre === null && candidate.index === index) {
            track = candidate;
            break;
          }
          continue;
        }
        const distance = Math.hypot(candidate.centre.x - centre.x, candidate.centre.y - centre.y);
        if (distance <= options.smileMatchDistance && distance < best) {
          best = distance;
          track = candidate;
        }
      }
      if (track === null) {
        track = {
          index,
          centre,
          lastSeen: now,
          neutral: [],
          since: null,
          fired: false,
          readout: { value: 0, threshold: options.smileAbsolute, baseline: null },
        };
        this.smileTracks.push(track);
      } else {
        unmatched.delete(track);
      }
      track.index = index;
      track.centre = centre;
      track.lastSeen = now;

      const value = smileOf(face);
      // The last few neutral samples outlive the window: a long smile keeps the
      // baseline it started from instead of falling back to the absolute one.
      while (
        track.neutral.length > options.smileBaselineMinSamples &&
        now - track.neutral[0]!.t > options.smileBaselineMs
      ) {
        track.neutral.shift();
      }
      const baseline =
        track.neutral.length >= options.smileBaselineMinSamples
          ? median(track.neutral.map((sample) => sample.v))
          : null;
      const threshold =
        baseline === null
          ? options.smileAbsolute
          : Math.min(baseline + options.smileDelta, options.smileAbsolute);
      track.readout = { value, threshold, baseline };

      if (value < threshold - options.smileReleaseDelta) {
        track.since = null;
        track.fired = false;
      }
      if (value >= threshold) {
        track.since ??= now;
        if (!track.fired && now - track.since >= options.smileHoldMs) {
          candidates.push(track);
        }
      } else if (track.since === null && !track.fired) {
        // Only a face at rest teaches the baseline: a held smile never does.
        track.neutral.push({ t: now, v: value });
      }
      if (strongest === null || value - threshold > strongest.value - strongest.threshold) {
        strongest = track.readout;
      }
    });
    this.smileReadout = strongest;
    return candidates;
  }

  /** Shared hold/hysteresis logic of the two blendshape cues. */
  private hold(
    kind: FaceCueKind,
    now: number,
    value: number,
    threshold: number,
    release: number,
    holdMs: number,
    since: number | null,
    fired: boolean,
    set: (since: number | null, fired: boolean) => void,
  ): number | null {
    if (value < release) {
      set(null, false);
      return null;
    }
    if (value < threshold) {
      return null;
    }
    if (since === null) {
      set(now, false);
      return null;
    }
    if (fired || now - since < holdMs) {
      return null;
    }
    if (!this.ready(kind, now)) {
      return null;
    }
    set(since, true);
    return Math.min(1, value);
  }

  private ready(kind: FaceCueKind, now: number): boolean {
    const last = this.lastFired.get(kind);
    const cooldown = this.options.cooldowns[kind];
    if (last !== undefined && now - last < cooldown) {
      return false;
    }
    this.lastFired.set(kind, now);
    return true;
  }
}
