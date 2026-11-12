/**
 * The moods of the particle sphere (R2-F, round 3).
 *
 * The user dropped the 3D head: the avatar is one breathing point-cloud wave
 * sphere, and it has to carry the character on its own. Each dashboard
 * expression is a *mood* — a set of surface parameters rather than blendshape
 * weights:
 *
 * - `wave`/`speed`/`bounce`: how alive the surface is (happy bounces, sleepy
 *   breathes slowly, worried ripples fast and small);
 * - `radius`/`flatten`: surprised puffs up quickly, sleepy sags;
 * - `swirl`: thinking turns slowly with a visible twist;
 * - `jitter`: worried shivers;
 * - `tint`/`brightness`: the colour of the mood;
 * - `attention`: how strongly the cloud bulges and glows toward the person.
 *
 * `MoodAnimator` blends every parameter with a time constant, so a change of
 * mood reads as the sphere *feeling* different rather than switching costume.
 */

import type { Expression } from "../cognitive/rules";
import { EXPRESSIONS } from "../cognitive/rules";

export interface MoodPose {
  /** Surface wave amplitude, fraction of the radius. */
  wave: number;
  /** Wave and breathing speed multiplier. */
  speed: number;
  /** Extra breathing amplitude. */
  bounce: number;
  /** Radius of the rest sphere, world units. */
  radius: number;
  /** Vertical squash: 1 keeps the sphere round, < 1 flattens it. */
  flatten: number;
  /** Torsional twist at the top of the sphere, radians. */
  swirl: number;
  /** Per-point shimmer, world units. */
  jitter: number;
  /** Additive colour of the mood. */
  tint: [number, number, number];
  /** Global brightness multiplier. */
  brightness: number;
  /** How much the cloud bulges toward the person (0..1). */
  attention: number;
  /** Lean of the whole cloud, radians. */
  tilt: number;
}

/**
 * The seven moods, in the same order and with the same names as the cognitive
 * layer's expressions, so `main.ts` keeps calling `setExpression`.
 */
export const MOODS: Record<Expression, MoodPose> = {
  neutral: {
    wave: 0.05,
    speed: 1,
    bounce: 0.012,
    radius: 0.43,
    flatten: 1,
    swirl: 0,
    jitter: 0,
    tint: [0, 0, 0],
    brightness: 1,
    attention: 0.35,
    tilt: 0,
  },
  happy: {
    wave: 0.078,
    speed: 1.45,
    bounce: 0.032,
    radius: 0.44,
    flatten: 1,
    swirl: 0,
    jitter: 0,
    tint: [0.1, 0.3, 0.18],
    brightness: 1.12,
    attention: 0.45,
    tilt: 0.02,
  },
  surprised: {
    wave: 0.092,
    speed: 1.25,
    bounce: 0.014,
    radius: 0.465,
    flatten: 1,
    swirl: 0,
    jitter: 0,
    tint: [0.22, 0.26, 0.36],
    brightness: 1.14,
    attention: 0.4,
    tilt: 0,
  },
  curious: {
    wave: 0.06,
    speed: 1.05,
    bounce: 0.014,
    radius: 0.435,
    flatten: 1,
    swirl: 0.12,
    jitter: 0,
    tint: [0.16, 0.05, 0.3],
    brightness: 1.03,
    attention: 0.95,
    tilt: 0.12,
  },
  thinking: {
    wave: 0.048,
    speed: 0.62,
    bounce: 0.01,
    radius: 0.425,
    flatten: 1,
    swirl: 0.5,
    jitter: 0,
    tint: [0.08, 0.16, 0.34],
    brightness: 0.96,
    attention: 0.5,
    tilt: 0.05,
  },
  sleepy: {
    wave: 0.05,
    speed: 0.4,
    bounce: 0.02,
    radius: 0.445,
    flatten: 0.9,
    swirl: 0,
    jitter: 0,
    tint: [0.02, 0.02, 0.12],
    brightness: 0.68,
    attention: 0.25,
    tilt: -0.04,
  },
  worried: {
    wave: 0.038,
    speed: 2.1,
    bounce: 0.006,
    radius: 0.425,
    flatten: 1,
    swirl: 0,
    jitter: 0.007,
    tint: [0.16, 0.01, 0.05],
    brightness: 1.05,
    attention: 0.5,
    tilt: 0,
  },
};

export const MOOD_NAMES = EXPRESSIONS;

/** The numeric parameters the animator blends; `tint` is the vector. */
const NUMERIC_MOOD_KEYS = [
  "wave",
  "speed",
  "bounce",
  "radius",
  "flatten",
  "swirl",
  "jitter",
  "brightness",
  "attention",
  "tilt",
] as const satisfies ReadonlyArray<Exclude<keyof MoodPose, "tint">>;

export function isExpression(value: string): value is Expression {
  return (EXPRESSIONS as readonly string[]).includes(value);
}

/** The frame inputs the sphere needs besides the mood itself. */
export interface MoodChannels {
  /** Voice level 0..1, already smoothed by the energy driver. */
  speaking: number;
  /** Attention point in -1..1, from the gaze controller. */
  gaze: { x: number; y: number };
  /** 0 alone, 1 with a person engaged. */
  presence: number;
}

export interface MoodFrame {
  pose: MoodPose;
  channels: MoodChannels;
  /** Decaying 0..1 impulse: surprised puffs up when the mood begins. */
  pulse: number;
}

const BLEND_RATE = 4.5;
const PULSE_SECONDS = 0.45;

/** Smoothly blends the mood parameters and owns the surprised impulse. */
export class MoodAnimator {
  private readonly reducedMotion: boolean;
  private current: MoodPose;
  private target: MoodPose;
  private pulse = 0;
  private expression: Expression = "neutral";

  constructor(options: { reducedMotion?: boolean } = {}) {
    this.reducedMotion = options.reducedMotion === true;
    this.current = { ...MOODS.neutral, tint: [...MOODS.neutral.tint] };
    this.target = { ...MOODS.neutral, tint: [...MOODS.neutral.tint] };
  }

  get mood(): Expression {
    return this.expression;
  }

  setExpression(expression: Expression): void {
    if (expression === this.expression) {
      return;
    }
    if (expression === "surprised") {
      this.pulse = 1;
    }
    this.expression = expression;
    this.target = { ...MOODS[expression], tint: [...MOODS[expression].tint] };
  }

  /** Jump to a mood without blending: used to pin a frame for screenshots. */
  snap(expression: Expression): void {
    this.expression = expression;
    this.target = { ...MOODS[expression], tint: [...MOODS[expression].tint] };
    this.current = { ...MOODS[expression], tint: [...MOODS[expression].tint] };
    this.pulse = expression === "surprised" ? 1 : 0;
  }

  update(dtSeconds: number, channels: MoodChannels): MoodFrame {
    const dt = Math.min(0.1, Math.max(0, dtSeconds));
    const blend = 1 - Math.exp(-dt * BLEND_RATE);
    const motion = this.reducedMotion ? 0.3 : 1;
    for (const key of NUMERIC_MOOD_KEYS) {
      this.current[key] += (this.target[key] - this.current[key]) * blend;
    }
    for (let axis = 0; axis < 3; axis += 1) {
      this.current.tint[axis] += (this.target.tint[axis] - this.current.tint[axis]) * blend;
    }
    this.pulse = Math.max(0, this.pulse - dt / PULSE_SECONDS);

    const pose: MoodPose = {
      ...this.current,
      tint: [...this.current.tint],
      wave: this.current.wave * motion,
      bounce: this.current.bounce * motion,
      swirl: this.current.swirl * motion,
      jitter: this.current.jitter * motion,
      speed: this.current.speed * (this.reducedMotion ? 0.55 : 1),
    };
    return { pose, channels, pulse: this.pulse * motion };
  }
}
