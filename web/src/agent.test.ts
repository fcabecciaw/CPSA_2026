/**
 * Lifecycle corners at the agent level: lost vision, mode switch and the
 * silent warm-up. The rule subset keeps these tests focused on the wiring.
 */

import { describe, expect, it, vi } from "vitest";

import { CognitiveAgent, HEARTBEAT_MS } from "./cognitive/agent";
import { CognitiveMemory } from "./cognitive/memory";
import { Perception } from "./cognitive/perception";
import { RULEBOOK } from "./cognitive/rulebook";
import { CognitiveEngine, type Rule } from "./cognitive/rules";
import { SceneMachine } from "./cognitive/scene";
import { armsUp, detection, person, pose, showcase } from "./cognitive/testkit";
import type { StatusPayload } from "./protocol";

function pick(...ids: string[]): Rule[] {
  return ids.map((id) => {
    const rule = RULEBOOK.find((entry) => entry.id === id);
    if (rule === undefined) {
      throw new Error(`regola mancante: ${id}`);
    }
    return rule;
  });
}

function status(overrides: Partial<StatusPayload> = {}): StatusPayload {
  return {
    state: "running",
    camera: "ok",
    model: "yolox_nano",
    fps: 59.6,
    dpu_ms: 4.1,
    errors: [],
    uptime_s: 120,
    ...overrides,
  };
}

function makeAgent(rules: Rule[]) {
  const spoken: string[] = [];
  const ended = vi.fn();
  const memory = new CognitiveMemory({ storage: null });
  const engine = new CognitiveEngine(rules, { rng: () => 0.5 });
  const agent = new CognitiveAgent({
    engine,
    memory,
    perception: new Perception(),
    speak: (text, handlers) => {
      spoken.push(text);
      handlers.onStart?.();
      handlers.onEnd?.("ended");
      ended();
    },
  });
  return { agent, spoken, memory, engine };
}

describe("F-E4 #21 — lost vision", () => {
  it("says it once, stays quiet, then announces the recovery", () => {
    const { agent, spoken } = makeAgent(
      pick("vision.lost", "vision.back", "object.bottiglia"),
    );
    const now = 1_000;
    const lost = agent.onConnection(false, now);
    expect(lost.map((event) => event.type)).toEqual(["vision"]);
    expect(spoken).toEqual(["Ops, non vedo più niente!"]);

    expect(agent.onConnection(false, now + 100)).toEqual([]);
    expect(agent.onStatus(status({ camera: "error" }), now + 200)).toEqual([]);
    expect(spoken).toHaveLength(1);

    agent.tick(now + 500);
    expect(spoken).toHaveLength(1);

    const resumed = agent.onShowcase(
      showcase([detection("bottiglia", 39, [520, 250, 580, 400], 0.9)]),
      now + 1_000,
    );
    expect(resumed.filter((event) => event.type === "appear")).toHaveLength(0);
    expect(spoken).toHaveLength(1);
    // The queued recovery expires 2 s after it was captured and the 2 s gap is
    // over at once: 3 s after the loss is the last moment it can still be said.
    agent.tick(now + 3_000);
    expect(spoken.at(-1)).toBe("Ci vedo di nuovo!");
  });
});

describe("F-E4 #22 — mode switch", () => {
  it("announces the switch once, resets silently and keeps families apart", () => {
    const { agent, spoken, engine, memory } = makeAgent(
      pick("mode.movenet", "mode.yolox", "object.bottiglia", "pose.arms_up"),
    );
    agent.onShowcase(showcase([person(20)]), 0);

    agent.onPose(pose(armsUp()), 1_000);
    expect(spoken).toEqual(["Ora guardo i movimenti: muoviti un po'!"]);
    expect(memory.working.focus.mode).toBe("movenet");

    for (let now = 6_000; now <= 7_000; now += 100) {
      agent.onPose(pose(armsUp()), now);
    }
    expect(spoken.at(-1)).toMatch(/Braccia in alto/);

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
          now: 8_000,
        },
      ],
      memory,
      8_000,
    );
    expect(engine.tick(8_000, memory)).toBeNull();

    agent.onShowcase(showcase([person(20)]), 12_000);
    expect(spoken.at(-1)).toBe("Torno a guardare gli oggetti.");
    expect(spoken.filter((text) => text === "Ora guardo i movimenti: muoviti un po'!")).toHaveLength(1);
  });
});

/** `showcase.state` arrives at most 10 Hz from the board (`src/cpsa/pipeline.py`). */
const STATE_PERIOD_MS = 100;
/** A cached WAV: HTTP round trip on loopback, file read and decode. */
const CACHED_WAV_MS = 60;

describe("T1 — responsiveness", () => {
  it("speaks an object shown to an idle ARI within 1 s of the first detection", () => {
    const spokenAt: Array<{ text: string; at: number }> = [];
    let clock = 0;
    const memory = new CognitiveMemory({ storage: null });
    const agent = new CognitiveAgent({
      engine: new CognitiveEngine(RULEBOOK, { rng: () => 0.5 }),
      memory,
      perception: new Perception(),
      scene: new SceneMachine({ rng: () => 0.5 }),
      speak: (text, handlers) => {
        spokenAt.push({ text, at: clock });
        handlers.onStart?.();
        handlers.onEnd?.("ended");
      },
    });
    const cup = detection("tazza", 41, [300, 200, 380, 300], 0.9);
    const shownAt = 6_000;
    // The real cadence: frames at 10 Hz, the heartbeat at HEARTBEAT_MS.
    for (clock = 0; clock <= shownAt + 2_000; clock += HEARTBEAT_MS) {
      if (clock % STATE_PERIOD_MS === 0) {
        agent.onShowcase(showcase(clock >= shownAt ? [cup] : []), clock);
      }
      agent.tick(clock);
    }
    const line = spokenAt.find((entry) => /tazza/i.test(entry.text));
    expect(line).toBeDefined();
    expect(line!.at + CACHED_WAV_MS - shownAt).toBeLessThanOrEqual(1_000);
  });

  it("hands the line waiting for the gap to the voice before it is spoken", () => {
    const prefetched: Array<{ texts: string[]; at: number }> = [];
    const spokenAt: Array<{ text: string; at: number }> = [];
    let clock = 0;
    const agent = new CognitiveAgent({
      engine: new CognitiveEngine(pick("object.tazza", "person.single"), { rng: () => 0.5 }),
      memory: new CognitiveMemory({ storage: null }),
      perception: new Perception(),
      speak: (text, handlers) => {
        spokenAt.push({ text, at: clock });
        handlers.onEnd?.("ended");
      },
      prefetch: (texts) => prefetched.push({ texts: [...texts], at: clock }),
    });
    const cup = detection("tazza", 41, [300, 200, 380, 300], 0.9);
    for (clock = 0; clock <= 12_000; clock += HEARTBEAT_MS) {
      const objects = clock >= 6_800 ? [cup, person(20)] : clock >= 6_000 ? [cup] : [];
      agent.onShowcase(showcase(objects), clock);
      agent.tick(clock);
    }
    const said = spokenAt.find((entry) => /ciao/i.test(entry.text));
    expect(said).toBeDefined();
    // The greeting was queued inside the 2 s gap after the cup line: the voice
    // got it at once, before the gap released it, and only once.
    const warmed = prefetched.filter((entry) => entry.texts.includes(said!.text));
    expect(warmed).toHaveLength(1);
    expect(warmed[0].at).toBeLessThan(said!.at);
  });
});

