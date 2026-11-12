/**
 * The calm objects panel: 5 % scores, 500 ms text, stable order, 1 s leave
 * hold and count hysteresis (coordinator feedback, 2026-11-12).
 */

import { describe, expect, it } from "vitest";

import { CalmShowcase, scorePercent } from "./calm";
import type { ShowcaseObject, ShowcaseStatePayload } from "./protocol";

function object(
  label: string,
  classId: number,
  box: [number, number, number, number],
  score: number,
): ShowcaseObject {
  return { label, class_id: classId, score, box };
}

function state(objects: ShowcaseObject[], counts?: ShowcaseStatePayload["counts"]): ShowcaseStatePayload {
  const grouped = new Map<string, number>();
  for (const entry of objects) {
    grouped.set(entry.label, (grouped.get(entry.label) ?? 0) + 1);
  }
  return {
    model: "yolox_nano",
    width: 640,
    height: 480,
    fps: 60,
    dpu_ms: 4,
    total_ms: 16,
    objects,
    counts:
      counts ?? [...grouped.entries()].map(([label, count]) => ({ label, count })),
  };
}

const cup = object("tazza", 41, [100, 300, 180, 380], 0.93);
const book = object("libro", 73, [200, 300, 300, 380], 0.78);

describe("calm objects panel", () => {
  it("rounds scores to 5 %", () => {
    expect(scorePercent(0.93)).toBe(95);
    expect(scorePercent(0.92)).toBe(90);
    expect(scorePercent(0.78)).toBe(80);
    expect(scorePercent(0.03)).toBe(5);
    expect(scorePercent(0.02)).toBe(0);
    expect(scorePercent(-1)).toBe(0);
    expect(scorePercent(2)).toBe(100);
  });

  it("refreshes the visible text at most every 500 ms", () => {
    const calm = new CalmShowcase();
    expect(calm.update(state([cup]), 0)).not.toBeNull();
    expect(calm.update(state([cup, book]), 250)).toBeNull();
    expect(calm.update(state([cup, book]), 500)).not.toBeNull();
    expect(calm.tick(600)).toBeNull();
    expect(calm.tick(1_000)).not.toBeNull();
  });

  it("keeps a stable order by class and first-seen, never by score", () => {
    const calm = new CalmShowcase();
    calm.update(state([book, cup]), 0);
    expect(calm.view?.objects.map((entry) => entry.label)).toEqual(["tazza", "libro"]);

    calm.update(state([{ ...book, score: 0.2 }, { ...cup, score: 0.1 }]), 600);
    expect(calm.view?.objects.map((entry) => entry.label)).toEqual(["tazza", "libro"]);

    const twoCups = new CalmShowcase();
    twoCups.update(state([object("tazza", 41, [0, 0, 50, 50], 0.9)]), 0);
    twoCups.update(
      state([object("tazza", 41, [0, 0, 50, 50], 0.9), object("tazza", 41, [300, 0, 350, 50], 0.95)]),
      600,
    );
    const keys = twoCups.view?.objects.map((entry) => entry.key) ?? [];
    expect(keys).toHaveLength(2);
    expect(keys[0]).toBe("41:0");

    twoCups.update(
      state([object("tazza", 41, [0, 0, 50, 50], 0.2), object("tazza", 41, [300, 0, 350, 50], 0.99)]),
      1_200,
    );
    expect(twoCups.view?.objects.map((entry) => entry.key)).toEqual(keys);
  });

  it("holds a disappeared object for one second before dropping it", () => {
    const calm = new CalmShowcase();
    calm.update(state([cup]), 0);
    expect(calm.view?.objects).toHaveLength(1);
    calm.update(state([]), 600);
    expect(calm.view?.objects[0]?.leaving).toBe(true);
    calm.update(state([]), 1_000);
    expect(calm.view?.objects).toHaveLength(1);
    calm.update(state([]), 1_500);
    expect(calm.view?.objects).toHaveLength(0);
  });

  it("commits a count only after the new value held for 500 ms", () => {
    const calm = new CalmShowcase();
    calm.update(state([cup]), 0);
    expect(calm.view?.counts).toEqual([{ label: "tazza", classId: 41, count: 1 }]);

    calm.update(state([cup, { ...cup, box: [300, 300, 380, 380] }]), 600);
    expect(calm.view?.counts[0]?.count).toBe(1);
    calm.update(state([cup, { ...cup, box: [300, 300, 380, 380] }]), 1_100);
    expect(calm.view?.counts[0]?.count).toBe(2);

    calm.update(state([]), 1_600);
    expect(calm.view?.counts[0]?.count).toBe(2);
    calm.update(state([]), 2_100);
    expect(calm.view?.counts).toHaveLength(0);
  });
});
