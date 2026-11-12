/**
 * Production-rule engine: conflict resolution, queueing, cooldowns, variants
 * and habituation. The F-E4 corners are named in the tests.
 */

import { describe, expect, it } from "vitest";

import type { CognitiveEvent } from "./events";
import { CognitiveMemory } from "./memory";
import { RULEBOOK } from "./rulebook";
import { CognitiveEngine, type Rule } from "./rules";

function memoryFor(people = 1, crowd = people >= 5): CognitiveMemory {
  const memory = new CognitiveMemory({ storage: null });
  memory.setFocus({ mode: "yolox", people, crowd });
  return memory;
}

function appear(label: string, now = 0): CognitiveEvent {
  return {
    type: "appear",
    label,
    count: 1,
    score: 0.9,
    area: 0.02,
    scenery: false,
    doubtful: false,
    now,
  };
}

function countEvent(from: number, to: number, now = 0): CognitiveEvent {
  return { type: "count", from, to, quiet: false, returnAfterS: null, now };
}

function selfEvent(now = 0): CognitiveEvent {
  return { type: "self", field: "fps", value: 59.6, now };
}

describe("F-E4 #6 — arrival during speech", () => {
  it("queues a priority ≥ 80 greeting, revalidates it and drops it after 3 s", () => {
    const memory = memoryFor(1);
    const engine = new CognitiveEngine(RULEBOOK, {
      rng: () => 0.5,
      gapMs: 1_000,
      queueExpiryMs: 3_000,
    });
    engine.consider([countEvent(0, 1)], memory, 0);
    expect(engine.tick(0, memory)?.ruleId).toBe("person.single");
    expect(engine.speaking).toBe(true);

    memory.setFocus({ people: 3 });
    engine.consider([countEvent(1, 3)], memory, 500);
    expect(engine.queueLength).toBe(1);
    expect(engine.tick(1_000, memory)).toBeNull();

    engine.endSpeech();
    expect(engine.tick(1_500, memory)?.ruleId).toBe("person.small_group");

    // Same scenario, but the people are gone before the queue is drained.
    const dropped = new CognitiveEngine(RULEBOOK, {
      rng: () => 0.5,
      gapMs: 1_000,
    });
    const gone = memoryFor(1);
    dropped.consider([countEvent(0, 1)], gone, 0);
    expect(dropped.tick(0, gone)).not.toBeNull();
    gone.setFocus({ people: 3 });
    dropped.consider([countEvent(1, 3)], gone, 500);
    gone.setFocus({ people: 0 });
    dropped.endSpeech();
    expect(dropped.tick(1_500, gone)).toBeNull();

    // And a queued line older than 3 s is stale.
    const stale = new CognitiveEngine(RULEBOOK, {
      rng: () => 0.5,
      gapMs: 1_000,
    });
    const still = memoryFor(1);
    stale.consider([countEvent(0, 1)], still, 0);
    expect(stale.tick(0, still)).not.toBeNull();
    still.setFocus({ people: 3 });
    stale.consider([countEvent(1, 3)], still, 1_000);
    stale.endSpeech();
    expect(stale.tick(4_500, still)).toBeNull();
  });
});

describe("F-E4 #7/#13 — one sentence about the most salient object", () => {
  it("picks one object out of three and keeps one utterance in flight", () => {
    const memory = memoryFor(1);
    const engine = new CognitiveEngine(RULEBOOK, { rng: () => 0.5 });
    engine.consider([appear("bottiglia"), appear("tazza"), appear("libro")], memory, 0);
    const utterance = engine.tick(0, memory);
    expect(utterance?.ruleId).toMatch(/^object\./);
    expect(utterance?.text).not.toMatch(/bottiglia.*tazza.*libro/i);
    expect(engine.tick(500, memory)).toBeNull();
    expect(engine.speaking).toBe(true);
    engine.endSpeech();
  });

  it("queues nothing below priority 80 and at most one high-priority line", () => {
    const memory = memoryFor(1);
    const engine = new CognitiveEngine(RULEBOOK, { rng: () => 0.5, gapMs: 1_000 });
    engine.consider([countEvent(0, 1)], memory, 0);
    expect(engine.tick(0, memory)?.ruleId).toBe("person.single");

    engine.consider([appear("tazza", 100)], memory, 100);
    expect(engine.queueLength).toBe(0);

    memory.setFocus({ people: 2 });
    engine.consider([countEvent(1, 2, 200)], memory, 200);
    engine.consider([appear("libro", 300)], memory, 300);
    expect(engine.queueLength).toBe(1);
    engine.endSpeech();
    expect(engine.tick(1_200, memory)?.ruleId).toBe("person.two");
  });
});

describe("F-E4 #13 — global gap between utterances", () => {
  it("waits 2 s, then speaks again", () => {
    const memory = memoryFor(1);
    const engine = new CognitiveEngine(RULEBOOK, { rng: () => 0.5 });
    engine.consider([appear("tazza")], memory, 0);
    expect(engine.tick(0, memory)).not.toBeNull();
    engine.endSpeech();

    engine.consider([appear("libro", 1_000)], memory, 1_000);
    expect(engine.tick(1_999, memory)).toBeNull();
    engine.consider([appear("libro", 2_001)], memory, 2_001);
    expect(engine.tick(2_001, memory)).not.toBeNull();
  });
});

function testRule(overrides: Partial<Rule> = {}): Rule {
  return {
    id: "test.rule",
    category: "object",
    when: () => true,
    priority: 70,
    cooldown_s: 0,
    expression: "curious",
    phrases: ["A", "B"],
    ...overrides,
  };
}

function fire(engine: CognitiveEngine, memory: CognitiveMemory, event: CognitiveEvent, now: number) {
  engine.consider([event], memory, now);
  const utterance = engine.tick(now, memory);
  if (utterance !== null) {
    engine.endSpeech();
  }
  return utterance;
}

describe("F-E4 #14 — no repetition", () => {
  it("honours rule and per-object cooldowns", () => {
    const memory = memoryFor(1);
    const engine = new CognitiveEngine([testRule({ cooldown_s: 10 })], { rng: () => 0.5 });
    expect(fire(engine, memory, appear("tazza"), 0)).not.toBeNull();
    expect(fire(engine, memory, appear("tazza"), 5_000)).toBeNull();
    expect(fire(engine, memory, appear("tazza"), 10_001)).not.toBeNull();

    const perObject = memoryFor(1);
    const engine2 = new CognitiveEngine(
      [testRule({ cooldown_s: 0, per_object_cooldown_s: 60 })],
      { rng: () => 0.5 },
    );
    expect(fire(engine2, perObject, appear("tazza"), 0)).not.toBeNull();
    expect(fire(engine2, perObject, appear("tazza"), 5_000)).toBeNull();
    expect(fire(engine2, perObject, appear("libro"), 5_000)).not.toBeNull();
  });

  it("stops at max_per_day", () => {
    const memory = memoryFor(1);
    const engine = new CognitiveEngine([testRule({ max_per_day: 1 })], { rng: () => 0.5 });
    expect(fire(engine, memory, appear("tazza"), 0)).not.toBeNull();
    expect(fire(engine, memory, appear("libro"), 5_000)).toBeNull();
  });

  it("never repeats the previous variant", () => {
    const memory = memoryFor(1);
    const engine = new CognitiveEngine([testRule()], { rng: () => 0.5 });
    const first = fire(engine, memory, appear("tazza"), 0);
    const second = fire(engine, memory, appear("tazza"), 5_000);
    const third = fire(engine, memory, appear("tazza"), 10_000);
    expect(first?.text).toBe("A");
    expect(second?.text).toBe("B");
    expect(third?.text).toBe("A");
  });

  it("decays habituation and novelty", () => {
    const memory = memoryFor(1);
    memory.noteRuleFired("test.rule", "A", "tazza", 0);
    memory.noteRuleFired("test.rule", "A", "tazza", 1_000);
    memory.noteRuleFired("test.rule", "A", "tazza", 2_000);
    const fresh = memory.habituation("tazza", 2_000);
    expect(fresh).toBeGreaterThan(0);
    expect(memory.habituation("tazza", 200_000)).toBeLessThan(fresh);
    expect(memory.novelty("tazza", 2_000)).toBeLessThan(memory.novelty("mela", 2_000));
  });
});

describe("F-E4 #24 — self facts are rare and never in crowd mode", () => {
  it("respects the 60 s cooldown and the crowd rule", () => {
    const memory = memoryFor(0, false);
    const engine = new CognitiveEngine(RULEBOOK, { rng: () => 0.5 });
    expect(fire(engine, memory, selfEvent(0), 0)?.ruleId).toBe("self.fps");
    expect(fire(engine, memory, selfEvent(30_000), 30_000)).toBeNull();
    expect(fire(engine, memory, selfEvent(61_000), 61_000)?.ruleId).toBe("self.fps");

    const crowdMemory = memoryFor(6, true);
    const crowded = new CognitiveEngine(RULEBOOK, { rng: () => 0.5 });
    expect(fire(crowded, crowdMemory, selfEvent(0), 0)).toBeNull();
  });
});

describe("conflict resolution", () => {
  it("prefers the highest utility and ignores model-specific rules", () => {
    const memory = memoryFor(1);
    const engine = new CognitiveEngine(
      [
        testRule({ id: "test.low", priority: 20 }),
        testRule({ id: "test.high", priority: 90 }),
      ],
      { rng: () => 0.5 },
    );
    engine.consider([appear("tazza")], memory, 0);
    expect(engine.tick(0, memory)?.ruleId).toBe("test.high");

    const poseMemory = memoryFor(0, false);
    poseMemory.setFocus({ mode: "yolox" });
    const modelEngine = new CognitiveEngine([testRule({ id: "test.pose", model: "movenet" })], {
      rng: () => 0.5,
    });
    modelEngine.consider([appear("tazza")], poseMemory, 0);
    expect(modelEngine.tick(0, poseMemory)).toBeNull();
  });
});
