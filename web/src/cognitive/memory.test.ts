/**
 * Working and day memory: approximate totals, persistence and midnight.
 */

import { describe, expect, it } from "vitest";

import type { CognitiveEvent } from "./events";
import { CognitiveMemory, dayKey } from "./memory";
import { memoryStorage } from "./testkit";

function count(from: number, to: number, now: number): CognitiveEvent {
  return { type: "count", from, to, quiet: false, returnAfterS: null, now };
}

describe("F-E4 #5 — no identities, approximate day totals", () => {
  it("sums only the positive stable-count steps", () => {
    const memory = new CognitiveMemory({ storage: null });
    const now = new Date(2026, 10, 12, 10, 0).getTime();
    memory.rollover(now);
    memory.observe(
      [count(0, 1, now), count(1, 3, now), count(3, 2, now), count(2, 0, now), count(0, 2, now)],
      now,
    );
    expect(memory.day.totals.peopleSeen).toBe(5);
    expect(memory.day.totals.maxPeople).toBe(3);
    // Only the count is stored, never a person record.
    expect(Object.keys(memory.day.classes)).toEqual(["persona"]);
    expect(JSON.stringify(memory.day)).not.toMatch(/name|identity|volto/i);
  });

  it("counts objects and comments in the day", () => {
    const memory = new CognitiveMemory({ storage: null });
    const now = new Date(2026, 10, 12, 10, 0).getTime();
    memory.rollover(now);
    memory.observe(
      [
        {
          type: "appear",
          label: "tazza",
          count: 2,
          score: 0.9,
          area: 0.02,
          scenery: false,
          doubtful: false,
          now,
        },
        {
          type: "appear",
          label: "sedia",
          count: 1,
          score: 0.9,
          area: 0.1,
          scenery: true,
          doubtful: false,
          now,
        },
      ],
      now,
    );
    expect(memory.day.totals.objectsSeen).toBe(2);
    memory.noteRuleFired("object.tazza", "Ecco due tazze.", "tazza", now);
    expect(memory.day.totals.rulesFired).toBe(1);
    expect(memory.day.classes.tazza?.comments).toBe(1);
    expect(memory.lastPhrase("object.tazza")).toBe("Ecco due tazze.");
  });
});

describe("F-E4 #23 — day rollover and localStorage", () => {
  it("keeps the same day across reloads and resets at midnight", () => {
    const storage = memoryStorage();
    const morning = new Date(2026, 10, 12, 9, 30).getTime();
    const first = new CognitiveMemory({ storage });
    expect(first.rollover(morning)).toBe(true);
    first.observe([count(0, 2, morning)], morning);
    first.noteRuleFired("person.two", "Ciao a tutti e due!", "persona", morning);
    expect(first.day.date).toBe("2026-11-12");

    const reloaded = new CognitiveMemory({ storage });
    expect(reloaded.rollover(new Date(2026, 10, 12, 18, 0).getTime())).toBe(false);
    expect(reloaded.day.date).toBe("2026-11-12");
    expect(reloaded.day.totals.peopleSeen).toBe(2);
    expect(reloaded.lastPhrase("person.two")).toBe("Ciao a tutti e due!");

    const nextDay = new Date(2026, 10, 13, 0, 10).getTime();
    expect(reloaded.rollover(nextDay)).toBe(true);
    expect(reloaded.day.date).toBe("2026-11-13");
    expect(reloaded.day.totals.peopleSeen).toBe(0);
    expect(reloaded.lastPhrase("person.two")).toBeNull();
  });

  it("survives a storage that throws (private mode)", () => {
    const memory = new CognitiveMemory({
      storage: {
        getItem: () => {
          throw new Error("blocked");
        },
        setItem: () => {
          throw new Error("quota");
        },
        removeItem: () => undefined,
      },
    });
    const now = Date.now();
    expect(() => memory.rollover(now)).not.toThrow();
    expect(() => memory.observe([count(0, 1, now)], now)).not.toThrow();
    expect(dayKey(now)).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});
