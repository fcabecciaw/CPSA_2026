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
  it("needs 0.4 s above 0.5 before it fires, then fires once per smile", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    const smile = face({ blendshapes: { mouthSmileLeft: 0.8, mouthSmileRight: 0.7 } });
    const fired: FaceCue[] = [];
    for (; now <= 3_000; now += 100) {
      fired.push(...detector.update(frame(now, [smile])));
    }
    expect(cueKinds(fired).filter((kind) => kind === "smile")).toEqual(["smile"]);
    expect(fired[0]!.score).toBeGreaterThan(0.5);
    expect(fired[0]!.at).toBeGreaterThanOrEqual(550 + 400);
  });

  it("does not fire on a spike or below the threshold", () => {
    const detector = new FaceCueDetector(options());
    let now = calibrate(detector);
    expect(detector.update(frame(now, [face({ blendshapes: { mouthSmileLeft: 0.9 } })])).length).toBe(0);
    now += 100;
    for (let step = 0; step < 10; step += 1, now += 100) {
      expect(
        detector.update(frame(now, [face({ blendshapes: { mouthSmileLeft: 0.45 } })])).length,
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
