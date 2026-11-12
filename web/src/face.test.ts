/**
 * Gaze hysteresis and expression hold of the holographic face.
 */

import { describe, expect, it } from "vitest";

import { ExpressionController, GazeController, type GazePoint } from "./face";
import type { Box } from "./cognitive/perception";

const FRAME = { width: 640, height: 480 };
const A: Box = [100, 100, 200, 200];
const B: Box = [420, 100, 580, 300];
const SLIGHTLY_BIGGER: Box = [100, 100, 200, 215];

function run(
  controller: GazeController,
  boxes: readonly Box[],
  from: number,
  to: number,
  step = 100,
): GazePoint {
  let point: GazePoint = { x: 0, y: 0 };
  for (let now = from; now <= to; now += step) {
    point = controller.update(boxes, FRAME, now);
  }
  return point;
}

describe("F-E4 #26 — gaze follows the largest person with hysteresis", () => {
  it("switches only when another box is 30 % larger for a second", () => {
    const controller = new GazeController();
    controller.update([A], FRAME, 0);
    const atA = run(controller, [A], 100, 1_000);
    expect(atA.x).toBeLessThan(-0.3);

    const slightly = run(controller, [A, SLIGHTLY_BIGGER], 1_100, 4_000);
    expect(slightly.x).toBeLessThan(-0.3);

    const held = run(controller, [A, B], 4_100, 5_000);
    expect(held.x).toBeLessThan(-0.2);
    const switched = run(controller, [A, B], 5_100, 7_000);
    expect(switched.x).toBeGreaterThan(0.2);
  });

  it("smooths instead of jumping and wanders when nobody is there", () => {
    const controller = new GazeController();
    controller.update([A], FRAME, 0);
    run(controller, [A], 100, 2_000);
    const before = controller.update([B], FRAME, 2_000);
    const after = controller.update([B], FRAME, 2_050);
    expect(Math.abs(after.x - before.x)).toBeLessThan(0.5);
    expect(after.x).not.toBe(before.x);

    const idle = new GazeController();
    const first = idle.update([], FRAME, 0);
    const later = idle.update([], FRAME, 4_000);
    expect(later).not.toEqual(first);
    expect(Math.abs(later.x)).toBeLessThanOrEqual(1);
    expect(Math.abs(later.y)).toBeLessThanOrEqual(1);
  });
});

describe("F-E4 #27 — expression hold and sleepy idle", () => {
  it("holds the utterance expression for 1.5 s, then decays to neutral", () => {
    const expressions = new ExpressionController();
    expect(expressions.current).toBe("neutral");
    expressions.beginUtterance("happy", 0);
    expect(expressions.tick(10_000, 0)).toBe("happy");
    expressions.endUtterance(2_000);
    expect(expressions.tick(3_000, 0)).toBe("happy");
    expect(expressions.tick(3_600, 0)).toBe("neutral");
  });

  it("falls asleep after 60 s alone and wakes on an utterance", () => {
    const expressions = new ExpressionController();
    expect(expressions.tick(100_000, 61)).toBe("sleepy");
    expressions.beginUtterance("surprised", 100_000);
    expect(expressions.tick(100_100, 61)).toBe("surprised");
    expressions.endUtterance(100_100);
    expect(expressions.tick(102_000, 61)).toBe("sleepy");
  });
});
