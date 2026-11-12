/** Attract loop: loop → activate → demo → back to the loop. */

import { describe, expect, it } from "vitest";

import { AttractMachine } from "./machine";

const DURATIONS = [10_000, 10_000, 10_000];

describe("AttractMachine", () => {
  it("opens in the loop on the first scene", () => {
    const machine = new AttractMachine({ durations: DURATIONS, now: 0 });
    expect(machine.phase).toBe("loop");
    expect(machine.scene).toBe(0);
    expect(machine.progress(5_000)).toBeCloseTo(0.5);
  });

  it("advances scene by scene and wraps around forever", () => {
    const machine = new AttractMachine({ durations: DURATIONS, now: 0 });
    expect(machine.tick(9_999)).toBe(false);
    expect(machine.tick(10_000)).toBe(true);
    expect(machine.scene).toBe(1);
    expect(machine.tick(20_000)).toBe(true);
    expect(machine.scene).toBe(2);
    expect(machine.tick(30_000)).toBe(true);
    expect(machine.scene).toBe(0);
    expect(machine.phase).toBe("loop");
  });

  it("keeps the rhythm when a tick comes late", () => {
    const machine = new AttractMachine({ durations: DURATIONS, now: 0 });
    machine.tick(25_000);
    expect(machine.scene).toBe(2);
    expect(machine.progress(25_000)).toBeCloseTo(0.5);
  });

  it("survives a very long pause without spinning", () => {
    const machine = new AttractMachine({ durations: DURATIONS, now: 0 });
    machine.tick(10_000_000);
    expect(machine.scene).toBeGreaterThanOrEqual(0);
    expect(machine.scene).toBeLessThan(3);
    expect(machine.tick(10_000_001)).toBe(false);
  });

  it("stops the loop when the demo is activated, and resumes from scene one", () => {
    const machine = new AttractMachine({ durations: DURATIONS, now: 0 });
    machine.tick(15_000);
    expect(machine.scene).toBe(1);

    expect(machine.activate()).toBe(true);
    expect(machine.phase).toBe("demo");
    expect(machine.activate()).toBe(false);
    // No scene changes while the demo runs.
    expect(machine.tick(100_000)).toBe(false);
    expect(machine.progress(100_000)).toBe(0);

    expect(machine.returnToLoop(200_000)).toBe(true);
    expect(machine.phase).toBe("loop");
    expect(machine.scene).toBe(0);
    expect(machine.returnToLoop(200_001)).toBe(false);
    expect(machine.tick(209_999)).toBe(false);
    expect(machine.tick(210_000)).toBe(true);
    expect(machine.scene).toBe(1);
  });

  it("can start in the demo (?attract=0)", () => {
    const machine = new AttractMachine({ durations: DURATIONS, now: 0, start: "demo" });
    expect(machine.phase).toBe("demo");
    expect(machine.tick(50_000)).toBe(false);
  });

  it("holds a pinned scene for screenshots", () => {
    const machine = new AttractMachine({ durations: DURATIONS, now: 0, pinnedScene: 2 });
    expect(machine.scene).toBe(2);
    expect(machine.tick(1_000_000)).toBe(false);
    expect(machine.scene).toBe(2);
    machine.activate();
    machine.returnToLoop(5);
    expect(machine.scene).toBe(2);
    expect(new AttractMachine({ durations: DURATIONS, now: 0, pinnedScene: 9 }).scene).toBe(2);
  });

  it("lets the operator step through the scenes", () => {
    const machine = new AttractMachine({ durations: DURATIONS, now: 0 });
    expect(machine.step(-1, 1_000)).toBe(true);
    expect(machine.scene).toBe(2);
    expect(machine.step(1, 2_000)).toBe(true);
    expect(machine.scene).toBe(0);
    expect(machine.progress(2_000)).toBe(0);
    machine.activate();
    expect(machine.step(1, 3_000)).toBe(false);
  });

  it("refuses an empty loop", () => {
    expect(() => new AttractMachine({ durations: [], now: 0 })).toThrow();
  });
});
