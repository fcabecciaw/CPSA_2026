/**
 * Debouncing, events and pose confirmation. Each test names the F-E4 corner it
 * protects.
 */

import { describe, expect, it } from "vitest";

import { CognitiveMemory } from "./memory";
import { Perception } from "./perception";
import { RULEBOOK, RULEBOOK_STATS } from "./rulebook";
import { CognitiveEngine } from "./rules";
import {
  armsUp,
  detection,
  person,
  pose,
  showcase,
  type SkeletonOverrides,
} from "./testkit";

function run(
  perception: Perception,
  from: number,
  to: number,
  state: () => ReturnType<typeof showcase>,
  step = 100,
): ReturnType<Perception["update"]> {
  const events: ReturnType<Perception["update"]> = [];
  for (let now = from; now <= to; now += step) {
    events.push(...perception.update(state(), now));
  }
  return events;
}

function countEvents(
  events: ReturnType<Perception["update"]>,
): Array<Extract<ReturnType<Perception["update"]>[number], { type: "count" }>> {
  return events.filter((event) => event.type === "count");
}

function engineMemory(people: number, crowd = people >= 5): CognitiveMemory {
  const memory = new CognitiveMemory({ storage: null });
  memory.setFocus({ mode: "yolox", people, crowd });
  return memory;
}

describe("F-E4 #1 — several people arrive together", () => {
  it("settles 0→3 into one collective count event", () => {
    const perception = new Perception();
    perception.reset(0);
    run(perception, 0, 4_000, () => showcase([]));
    const events = run(perception, 4_000, 8_000, () =>
      showcase([person(20), person(220), person(420)]),
    );
    const counts = countEvents(events);
    expect(counts).toHaveLength(1);
    expect(counts[0]).toMatchObject({ from: 0, to: 3 });

    const memory = engineMemory(3);
    const engine = new CognitiveEngine(RULEBOOK, { rng: () => 0.5 });
    engine.consider(counts, memory, counts[0].now);
    const utterance = engine.tick(counts[0].now, memory);
    expect(utterance?.ruleId).toBe("person.small_group");
    expect(utterance?.text).toMatch(/Siete in 3/);
    engine.endSpeech();
    expect(engine.tick(30_000, memory)).toBeNull();
  });
});

describe("F-E4 #2 — single vs group vs crowd", () => {
  it("has separate person families with the right address", () => {
    const memory1 = engineMemory(1);
    const memory2 = engineMemory(2);
    const memory4 = engineMemory(4);
    const memory6 = engineMemory(6, true);
    const single = RULEBOOK.find((rule) => rule.id === "person.single");
    const two = RULEBOOK.find((rule) => rule.id === "person.two");
    const group = RULEBOOK.find((rule) => rule.id === "person.small_group");
    const crowd = RULEBOOK.find((rule) => rule.id === "person.crowd");
    const event = (from: number, to: number) =>
      ({ type: "count", from, to, quiet: false, returnAfterS: null, now: 0 }) as const;

    expect(single?.when(event(0, 1), memory1)).toBe(true);
    expect(single?.phrases.join(" ")).toMatch(/Ti vedo/);
    expect(two?.when(event(0, 2), memory2)).toBe(true);
    expect(two?.phrases.join(" ")).toMatch(/tutti e due|voi due/);
    expect(group?.when(event(0, 4), memory4)).toBe(true);
    expect(crowd?.when(event(0, 6), memory6)).toBe(true);
    expect(single?.when(event(0, 6), memory6)).toBe(false);
  });

  it("suppresses object comments in crowd mode", () => {
    const memory = engineMemory(6, true);
    const engine = new CognitiveEngine(RULEBOOK, { rng: () => 0.5 });
    engine.consider(
      [
        {
          type: "appear",
          label: "bottiglia",
          count: 1,
          score: 0.9,
          area: 0.02,
          scenery: false,
          doubtful: false,
          now: 0,
        },
      ],
      memory,
      0,
    );
    expect(engine.tick(0, memory)).toBeNull();
  });

  it("waits 8 s between utterances in crowd mode", () => {
    const memory = engineMemory(6, true);
    const engine = new CognitiveEngine(RULEBOOK, { rng: () => 0.5 });
    const count = {
      type: "count",
      from: 5,
      to: 6,
      quiet: false,
      returnAfterS: null,
      now: 0,
    } as const;
    engine.consider([count], memory, 0);
    expect(engine.tick(0, memory)).not.toBeNull();
    engine.endSpeech();
    engine.consider([count], memory, 4_500);
    expect(engine.tick(4_500, memory)).toBeNull();
    engine.consider([count], memory, 8_100);
    expect(engine.tick(8_100, memory)).not.toBeNull();
  });
});

describe("F-E4 #3 — count flicker 2↔3", () => {
  it("ignores a count that does not hold for 1.5 s", () => {
    const perception = new Perception();
    perception.reset(0);
    run(perception, 0, 3_000, () => showcase([person(20), person(220)]));
    const flicker: ReturnType<Perception["update"]> = [];
    for (let now = 3_000; now <= 8_000; now += 100) {
      const third = (now / 100) % 2 === 0;
      flicker.push(
        ...perception.update(
          showcase(third ? [person(20), person(220), person(420)] : [person(20), person(220)]),
          now,
        ),
      );
    }
    expect(countEvents(flicker)).toHaveLength(0);

    const stable = run(perception, 8_100, 10_500, () =>
      showcase([person(20), person(220), person(420)]),
    );
    const counts = countEvents(stable);
    expect(counts).toHaveLength(1);
    expect(counts[0]).toMatchObject({ from: 2, to: 3 });
  });
});

describe("F-E4 #4 — short absence and welcome back", () => {
  it("does not greet a return under 30 s and welcomes after 30 s", () => {
    const perception = new Perception();
    perception.reset(0);
    run(perception, 0, 4_000, () => showcase([]));
    run(perception, 4_000, 8_000, () => showcase([person(20)]));
    run(perception, 8_000, 14_000, () => showcase([]));
    const shortReturn = run(perception, 14_000, 18_000, () => showcase([person(20)]));
    const shortCount = countEvents(shortReturn).at(-1);
    expect(shortCount).toMatchObject({ from: 0, to: 1, quiet: true, returnAfterS: null });

    run(perception, 18_000, 20_000, () => showcase([]));
    run(perception, 20_000, 60_000, () => showcase([]));
    const longReturn = run(perception, 60_000, 65_000, () => showcase([person(20)]));
    const longCount = countEvents(longReturn).at(-1);
    expect(longCount?.quiet).toBe(false);
    expect(longCount?.returnAfterS).toBeGreaterThanOrEqual(30);
  });
});

describe("F-E4 #9 — scenery", () => {
  it("never comments an object that was already there at start", () => {
    const perception = new Perception();
    perception.reset(0);
    const events = run(perception, 0, 10_000, () =>
      showcase([detection("sedia", 56, [400, 200, 560, 460], 0.9)]),
    );
    expect(events.filter((event) => event.type === "appear")).toHaveLength(0);
    expect(perception.track("sedia")?.scenery).toBe(true);
    expect(perception.tick(60_000).filter((event) => event.type === "dwell")).toHaveLength(0);
  });

  it("promotes an object present for two minutes, then accepts it back after 30 s away", () => {
    const perception = new Perception();
    perception.reset(0);
    run(perception, 0, 4_000, () => showcase([]));
    const chair = () => showcase([detection("sedia", 56, [400, 200, 560, 460], 0.9)]);
    run(perception, 4_000, 130_000, chair, 500);
    expect(perception.track("sedia")?.scenery).toBe(true);
    expect(perception.tick(130_000).filter((event) => event.type === "dwell")).toHaveLength(0);

    run(perception, 130_500, 175_000, () => showcase([]), 500);
    const back = run(perception, 175_000, 180_000, chair);
    const appear = back.find((event) => event.type === "appear");
    expect(appear).toMatchObject({ label: "sedia", scenery: false });
    expect(back.some((event) => event.type === "return" && event.label === "sedia")).toBe(true);
  });
});

describe("F-E4 #10 — false positives and doubtful classes", () => {
  it("needs score, area, stability and a high held score for implausible classes", () => {
    const perception = new Perception();
    perception.reset(0);
    run(perception, 0, 4_000, () => showcase([]));

    const lowScore = run(perception, 4_000, 7_000, () =>
      showcase([detection("giraffa", 23, [200, 100, 400, 400], 0.6)]),
    );
    expect(lowScore.filter((event) => event.type === "appear")).toHaveLength(0);

    const doubtful = run(perception, 7_100, 12_000, () =>
      showcase([detection("giraffa", 23, [200, 100, 400, 400], 0.75)]),
    );
    const giraffe = doubtful.find((event) => event.type === "appear");
    expect(giraffe).toMatchObject({ label: "giraffa", doubtful: true });

    const tiny = run(perception, 12_000, 16_000, () =>
      showcase([detection("cellulare", 67, [100, 100, 103, 103], 0.99)]),
    );
    expect(tiny.filter((event) => event.type === "appear")).toHaveLength(0);

    const weakPhone = run(perception, 16_000, 20_000, () =>
      showcase([detection("cellulare", 67, [300, 300, 400, 440], 0.4)]),
    );
    expect(weakPhone.filter((event) => event.type === "appear")).toHaveLength(0);

    const memory = engineMemory(1);
    if (giraffe?.type === "appear") {
      expect(RULEBOOK_STATS.ruleForClass("giraffa")?.when(giraffe, memory)).toBe(false);
      expect(RULEBOOK.find((rule) => rule.id === "class.doubtful")?.when(giraffe, memory)).toBe(
        true,
      );
    }
  });
});

describe("F-E4 #20 — silent warm-up after start/reconnect", () => {
  it("seeds the world for 3 s and only speaks about what appears later", () => {
    const perception = new Perception();
    perception.reset(0);
    const warm = run(perception, 0, 2_900, () =>
      showcase([person(20), detection("bottiglia", 39, [520, 250, 580, 400], 0.9)]),
    );
    expect(warm).toHaveLength(0);
    const after = run(perception, 3_000, 5_000, () =>
      showcase([person(20), detection("bottiglia", 39, [520, 250, 580, 400], 0.9)]),
    );
    expect(after.filter((event) => event.type === "appear" && event.label === "bottiglia")).toHaveLength(
      0,
    );
    const later = run(perception, 5_000, 6_500, () =>
      showcase([
        person(20),
        detection("bottiglia", 39, [520, 250, 580, 400], 0.9),
        detection("tazza", 41, [420, 340, 500, 410], 0.9),
      ]),
    );
    expect(later.some((event) => event.type === "appear" && event.label === "tazza")).toBe(true);
  });
});

describe("F-E4 #22 — mode switch resets without a disappear storm", () => {
  it("drops the world silently", () => {
    const perception = new Perception();
    perception.reset(0);
    run(perception, 0, 5_000, () =>
      showcase([person(20), detection("bottiglia", 39, [520, 250, 580, 400], 0.9)]),
    );
    perception.reset(6_000, "movenet");
    expect(perception.update(showcase([person(20)]), 6_100)).toHaveLength(0);
    const events = perception.update(showcase([person(20)]), 6_200);
    expect(events.filter((event) => event.type === "disappear")).toHaveLength(0);
    expect(perception.world.mode).toBe("movenet");
  });
});

describe("F-E4 #24 — idle and long stays", () => {
  it("emits idle milestones after 15, 60 and 180 s with nobody", () => {
    const perception = new Perception();
    perception.reset(0);
    const idle = (now: number) =>
      perception.tick(now).filter((event) => event.type === "idle");
    expect(idle(16_000).map((event) => event.seconds)).toEqual([15]);
    expect(idle(61_000).map((event) => event.seconds)).toEqual([60]);
    expect(idle(181_000).map((event) => event.seconds)).toEqual([180]);
  });

  it("asks 'sei ancora lì?' after 25 s with one person", () => {
    const perception = new Perception();
    perception.reset(0);
    run(perception, 0, 30_000, () => showcase([person(20)]));
    const dwell = perception
      .tick(30_000)
      .find((event) => event.type === "dwell" && event.label === "persona");
    expect(dwell).toMatchObject({ seconds: 25, count: 1 });
    const memory = engineMemory(1);
    expect(
      RULEBOOK.find((rule) => rule.id === "person.stay_25")?.when(dwell as never, memory),
    ).toBe(true);
  });
});

function poseEvents(
  perception: Perception,
  from: number,
  to: number,
  overrides: () => SkeletonOverrides,
  score = 0.85,
): ReturnType<Perception["updatePose"]> {
  const events: ReturnType<Perception["updatePose"]> = [];
  for (let now = from; now <= to; now += 100) {
    events.push(...perception.updatePose(pose(overrides(), score), now));
  }
  return events;
}

describe("F-E4 #25 — confirmed gestures only", () => {
  it("requires arms up for 0.5 s and respects the 10 s gesture cooldown", () => {
    const perception = new Perception();
    const first = poseEvents(perception, 0, 600, () => armsUp());
    expect(first.filter((event) => event.type === "pose" && event.kind === "arms_up")).toHaveLength(1);

    expect(poseEvents(perception, 700, 5_000, () => armsUp())).toHaveLength(0);
    poseEvents(perception, 5_100, 5_200, () => ({}));
    expect(poseEvents(perception, 5_300, 5_900, () => armsUp())).toHaveLength(0);
    const later = poseEvents(perception, 10_800, 11_500, () => armsUp());
    expect(later.some((event) => event.type === "pose" && event.kind === "arms_up")).toBe(true);
  });

  it("confirms a wave with two oscillations of enough amplitude", () => {
    const xs = [370, 400, 430, 400, 370, 400, 430];
    const manual = new Perception();
    const events: ReturnType<Perception["updatePose"]> = [];
    xs.forEach((x, index) => {
      events.push(...manual.updatePose(pose({ 10: [x, 150, 0.9] }), index * 100));
    });
    expect(events.some((event) => event.type === "pose" && event.kind === "wave")).toBe(true);
  });

  it("ignores low-confidence keypoints and a low pose score", () => {
    const perception = new Perception();
    const xs = [370, 400, 430, 400, 370];
    const lowKeypoint = new Perception();
    const events: ReturnType<Perception["updatePose"]> = [];
    xs.forEach((x, index) => {
      events.push(
        ...lowKeypoint.updatePose(pose({ 10: [x, 150, 0.2] }), index * 100),
      );
    });
    expect(events).toHaveLength(0);
    expect(poseEvents(perception, 0, 1_000, () => armsUp(), 0.2)).toHaveLength(0);
  });
});
