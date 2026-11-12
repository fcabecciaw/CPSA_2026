/**
 * Face-cue detectors on synthetic blendshape and head-pose streams (R2-D).
 *
 * `FaceCueDetector` is the pure part of the MediaPipe Face Landmarker pipeline:
 * the browser only adds frames, the tests drive it with numbers.
 */

import { describe, expect, it } from "vitest";

import {
  FaceCueDetector,
  type FaceCue,
  type FaceCueOptions,
  type FaceFrame,
  type FaceObservation,
} from "./facecues";

function face(overrides: Partial<FaceObservation> = {}): FaceObservation {
  return { blendshapes: {}, yaw: 0, pitch: 0, ...overrides };
}

function frame(now: number, faces: readonly FaceObservation[] = [face()]): FaceFrame {
  return { now, faces };
}

function options(overrides: Partial<FaceCueOptions> = {}): Partial<FaceCueOptions> {
  return { calibrateMs: 400, ...overrides };
}

function cueKinds(cues: readonly FaceCue[]): string[] {
  return cues.map((cue) => cue.kind);
}

/** Neutral frames for the calibration window; returns the next timestamp. */
function calibrate(detector: FaceCueDetector, from = 0, step = 50): number {
  let now = from;
  for (; now <= from + 500; now += step) {
    detector.update(frame(now));
  }
  return now;
}

describe("smile — held, released and cooldown", () => {
  it("needs 0.3 s above the threshold before it fires, then fires once per smile", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const smile = face({ blendshapes: { mouthSmileLeft: 0.8, mouthSmileRight: 0.7 } });
    const fired: FaceCue[] = [];
    for (; now <= 3_000; now += 100) {
      fired.push(...detector.update(frame(now, [smile])));
    }
    expect(cueKinds(fired).filter((kind) => kind === "smile")).toEqual(["smile"]);
    expect(fired[0]!.score).toBeGreaterThan(0.5);
    expect(fired[0]!.at).toBeGreaterThanOrEqual(550 + 300);
  });

  it("does not fire on a spike or below the threshold", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    expect(detector.update(frame(now, [face({ blendshapes: { mouthSmileLeft: 0.9 } })])).length).toBe(0);
    now += 100;
    // The neutral baseline is 0 here: 0.15 is below baseline + 0.20.
    for (let step = 0; step < 10; step += 1, now += 100) {
      expect(
        detector.update(frame(now, [face({ blendshapes: { mouthSmileLeft: 0.15 } })])).length,
      ).toBe(0);
    }
  });

  it("respects the smile cooldown and then fires again", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const smile = face({ blendshapes: { mouthSmileLeft: 0.9 } });
    const neutral = face();
    while (now < 1_500) {
      detector.update(frame(now, [smile]));
      now += 100;
    }
    const fired: FaceCue[] = [];
    // Release, then smile again after 9 s: the 8 s cooldown is over.
    for (; now < 9_500; now += 100) {
      detector.update(frame(now, [neutral]));
    }
    for (; now < 11_000; now += 100) {
      fired.push(...detector.update(frame(now, [smile])));
    }
    expect(cueKinds(fired)).toEqual(["smile"]);
  });
});

describe("mouth_open — surprise", () => {
  it("needs 0.3 s of jawOpen above 0.5", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const open = face({ blendshapes: { jawOpen: 0.8 } });
    const fired: FaceCue[] = [];
    for (; now < 1_200; now += 100) {
      fired.push(...detector.update(frame(now, [open])));
    }
    expect(cueKinds(fired)).toEqual(["mouth_open"]);
  });
});

describe("nod and head shake — swings around the calibrated centre", () => {
  it("detects a nod from the pitch swing", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const pitches = [0, 8, 16, 8, 0];
    const fired: FaceCue[] = [];
    for (const pitch of pitches) {
      fired.push(...detector.update(frame(now, [face({ pitch })])));
      now += 100;
    }
    expect(cueKinds(fired)).toEqual(["nod"]);
    expect(fired[0]!.score).toBeGreaterThan(0);
  });

  it("detects a head shake from the yaw swing", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const yaws = [0, -12, -24, -12, 0];
    const fired: FaceCue[] = [];
    for (const yaw of yaws) {
      fired.push(...detector.update(frame(now, [face({ yaw })])));
      now += 100;
    }
    expect(cueKinds(fired)).toEqual(["head_shake"]);
  });

  it("ignores a small wobble and respects the 12 s cooldown", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const fired: FaceCue[] = [];
    for (const pitch of [0, 2, 4, 2, 0, 2, 4, 2, 0]) {
      fired.push(...detector.update(frame(now, [face({ pitch })])));
      now += 100;
    }
    expect(fired).toEqual([]);

    // A real nod, then another one inside the cooldown: only the first speaks.
    for (const pitch of [0, 10, 20, 10, 0]) {
      fired.push(...detector.update(frame(now, [face({ pitch })])));
      now += 100;
    }
    for (const pitch of [0, 10, 20, 10, 0]) {
      fired.push(...detector.update(frame(now, [face({ pitch })])));
      now += 100;
    }
    expect(cueKinds(fired)).toEqual(["nod"]);
  });
});

describe("looking_at_me — held near the calibrated centre", () => {
  it("does not fire while the head is turned", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const fired: FaceCue[] = [];
    for (let step = 0; step < 10; step += 1, now += 100) {
      fired.push(...detector.update(frame(now, [face({ yaw: 25 })])));
    }
    expect(cueKinds(fired)).not.toContain("looking_at_me");
  });

  it("fires after 1.2 s of looking at the camera", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const fired: FaceCue[] = [];
    for (let step = 0; step < 20; step += 1, now += 100) {
      fired.push(...detector.update(frame(now, [face()])));
    }
    expect(cueKinds(fired).filter((kind) => kind === "looking_at_me")).toEqual([
      "looking_at_me",
    ]);
  });
});

describe("several faces and empty frames", () => {
  it("takes the strongest smile even when it comes from the second face", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const faces = [face(), face({ blendshapes: { mouthSmileRight: 0.85 } })];
    const fired: FaceCue[] = [];
    for (let step = 0; step < 15; step += 1, now += 100) {
      fired.push(...detector.update(frame(now, faces)));
    }
    expect(cueKinds(fired).filter((kind) => kind === "smile")).toEqual(["smile"]);
    expect(fired[0]!.faces).toBe(2);
  });

  it("resets holds when the face disappears", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const smile = face({ blendshapes: { mouthSmileLeft: 0.9 } });
    detector.update(frame(now, [smile]));
    now += 100;
    detector.update(frame(now, []));
    now += 100;
    const fired: FaceCue[] = [];
    for (let step = 0; step < 6; step += 1, now += 100) {
      fired.push(...detector.update(frame(now, [smile])));
    }
    expect(cueKinds(fired)).toEqual(["smile"]);
  });
});

describe("smile — per-face neutral baseline (C3)", () => {
  const smiles = (cues: readonly FaceCue[]): string[] =>
    cueKinds(cues).filter((kind) => kind === "smile");

  /** A slightly noisy stream around `level`, deterministic. */
  function wobble(level: number, step: number): number {
    return level + [0, 0.03, -0.02, 0.04, -0.03][step % 5]!;
  }

  function run(
    detector: FaceCueDetector,
    from: number,
    to: number,
    level: number,
  ): { cues: FaceCue[]; now: number } {
    const cues: FaceCue[] = [];
    let now = from;
    for (let step = 0; now < to; step += 1, now += 100) {
      const value = wobble(level, step);
      cues.push(
        ...detector.update(
          frame(now, [face({ blendshapes: { mouthSmileLeft: value, mouthSmileRight: value - 0.02 } })]),
        ),
      );
    }
    return { cues, now };
  }

  it("does not fire on a high neutral, and fires on that person's smile", () => {
    const detector = new FaceCueDetector(options());
    // A face whose resting mouth reads ~0.42: the old fixed 0.5 was a coin toss.
    const neutral = run(detector, 0, 4_000, 0.42);
    expect(smiles(neutral.cues)).not.toContain("smile");
    expect(detector.smile?.baseline).toBeCloseTo(0.42, 1);
    expect(detector.smile?.threshold).toBeCloseTo(0.55, 2);
    const smile = run(detector, neutral.now, neutral.now + 1_000, 0.75);
    expect(smiles(smile.cues)).toEqual(["smile"]);
  });

  it("fires on a shy smile over a low neutral, never on the neutral itself", () => {
    const detector = new FaceCueDetector(options());
    const neutral = run(detector, 0, 4_000, 0.05);
    expect(smiles(neutral.cues)).not.toContain("smile");
    expect(detector.smile?.threshold).toBeCloseTo(0.25, 1);
    // 0.33 never reached the old 0.5 threshold.
    const smile = run(detector, neutral.now, neutral.now + 1_000, 0.33);
    expect(smiles(smile.cues)).toEqual(["smile"]);
    expect(smile.cues[0]!.at - neutral.now).toBeGreaterThanOrEqual(300);
  });

  it("keeps a held smile out of the baseline", () => {
    const detector = new FaceCueDetector(options());
    const neutral = run(detector, 0, 3_000, 0.05);
    const held = run(detector, neutral.now, neutral.now + 5_000, 0.6);
    expect(smiles(held.cues)).toEqual(["smile"]);
    expect(detector.smile?.baseline).toBeCloseTo(0.05, 1);
  });

  it("keeps one baseline per face, followed by its box", () => {
    const detector = new FaceCueDetector(options());
    const left = (value: number): FaceObservation =>
      face({ blendshapes: { mouthSmileLeft: value }, box: [0.1, 0.2, 0.2, 0.35] });
    const right = (value: number): FaceObservation =>
      face({ blendshapes: { mouthSmileLeft: value }, box: [0.7, 0.2, 0.8, 0.35] });
    let now = 0;
    const cues: FaceCue[] = [];
    // Left rests at 0.45, right at 0.05; the order of the faces flips.
    for (let step = 0; now < 4_000; step += 1, now += 100) {
      const faces = step % 2 === 0 ? [left(0.45), right(0.05)] : [right(0.05), left(0.45)];
      cues.push(...detector.update(frame(now, faces)));
    }
    expect(smiles(cues)).not.toContain("smile");
    // The right face smiles shyly (0.3): only its own low baseline sees it.
    for (const end = now + 800; now < end; now += 100) {
      cues.push(...detector.update(frame(now, [left(0.45), right(0.3)])));
    }
    expect(smiles(cues)).toEqual(["smile"]);
  });
});
