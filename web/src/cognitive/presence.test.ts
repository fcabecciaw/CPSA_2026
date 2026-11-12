/**
 * Presence and arrivals (C3) at the agent level. ARI says hello to every
 * newcomer — a person or face track at the stand for 0.6 s that was never
 * greeted — with one collective greeting for people arriving together and at
 * least 8 s between greetings (newcomers in between are batched). People in
 * the background never count. The goodbye needs everybody gone for 3 s, once
 * per visit; a partial departure says nothing. Plus the smile reaction that
 * waits in the queue slot while ARI is speaking.
 */

import { afterEach, describe, expect, it } from "vitest";

import { CognitiveAgent, HEARTBEAT_MS } from "./agent";
import type { NormBox } from "./facecrop";
import { CognitiveMemory } from "./memory";
import { Perception } from "./perception";
import {
  DEFAULT_PRESENCE_THRESHOLDS,
  PRESENCE_THRESHOLDS_KEY,
  PresenceFusion,
  loadPresenceThresholds,
  presenceThresholds,
  setPresenceThresholds,
} from "./presence";
import { RULEBOOK } from "./rulebook";
import { CognitiveEngine, type SceneLine, type Utterance } from "./rules";
import { SMILE_PRIORITY, SceneMachine } from "./scene";
import { keypoints, memoryStorage, person, pose, showcase } from "./testkit";
import type { ShowcaseObject } from "../protocol";

interface Said {
  at: number;
  utterance: Utterance;
}

function makeAgent() {
  const said: Said[] = [];
  let clock = 0;
  const agent = new CognitiveAgent({
    engine: new CognitiveEngine(RULEBOOK, { rng: () => 0.5 }),
    memory: new CognitiveMemory({ storage: null }),
    perception: new Perception(),
    scene: new SceneMachine({ rng: () => 0.5 }),
    speak: (_text, handlers) => {
      handlers.onStart?.();
      handlers.onEnd?.("ended");
    },
    onUtterance: (utterance) => said.push({ at: clock, utterance }),
  });
  const setClock = (now: number): void => {
    clock = now;
  };
  return { agent, said, setClock };
}

type Rig = ReturnType<typeof makeAgent>;

/** What the board and the laptop see at a given time. */
interface View {
  people?: ShowcaseObject[];
  faces?: NormBox[];
}

/** A visitor at the stand (box 75 % of the frame high) at pixel column `x`. */
const near = (x: number): ShowcaseObject => person(x, 80, 120, 360);
/** Somebody walking by in the background (box 21 % of the frame high). */
const far = (x: number): ShowcaseObject => person(x, 200, 40, 100);
/** A face 10 % of the frame wide (at the stand) at normalised column `x`. */
const face = (x: number, width = 0.1): NormBox => [x, 0.2, x + width, 0.33];

/** Drive the agent at the real cadence: 10 Hz frames, faces and heartbeat. */
function run(rig: Rig, from: number, to: number, view: (now: number) => View): number {
  let now = from;
  for (; now < to; now += HEARTBEAT_MS) {
    rig.setClock(now);
    const seen = view(now);
    rig.agent.onShowcase(showcase(seen.people ?? []), now);
    rig.agent.onFaces(seen.faces ?? [], now);
    rig.agent.tick(now);
  }
  return now;
}

const empty = (): View => ({});
const goodbyes = (said: readonly Said[]): Said[] =>
  said.filter((entry) => entry.utterance.sceneKind === "goodbye");
/** Every hello, from the scene or (never, with arrivals) from the rulebook. */
const HELLO_RULES = new Set([
  "person.single",
  "person.two",
  "person.small_group",
  "person.crowd",
  "person.return",
  "idle.wake",
]);
const greetings = (said: readonly Said[]): Said[] =>
  said.filter(
    (entry) =>
      entry.utterance.sceneKind === "greeting" || HELLO_RULES.has(entry.utterance.ruleId),
  );

/** The silent warm-up (people there at start are seeded), then an empty stand. */
function warm(rig: Rig): number {
  return run(rig, 0, 4_000, empty);
}

afterEach(() => {
  setPresenceThresholds({ ...DEFAULT_PRESENCE_THRESHOLDS }, null);
});

describe("C3 arrivals — every newcomer gets a hello", () => {
  it("greets a newcomer who joins a person already there", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, 14_000, () => ({ people: [near(40)] }));
    expect(greetings(rig.said)).toHaveLength(1);
    const joinAt = now;
    now = run(rig, now, now + 6_000, () => ({ people: [near(40), near(420)] }));
    const said = greetings(rig.said);
    expect(said).toHaveLength(2);
    expect(said[1]!.utterance.ruleId).toMatch(/^scene\.greeting\.(single|return)$/);
    // Confirmed after 0.6 s, batched for 0.6 s more: well under 2 s.
    expect(said[1]!.at - joinAt).toBeLessThan(2_000);
    expect(goodbyes(rig.said)).toHaveLength(0);
  });

  it("greets the newcomer of a swap (one leaves while another arrives)", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, 14_000, () => ({ people: [near(40)] }));
    now = run(rig, now, now + 6_000, () => ({ people: [near(440)] }));
    expect(greetings(rig.said)).toHaveLength(2);
    expect(goodbyes(rig.said)).toHaveLength(0);
  });

  it("greets two people arriving together once, as a pair", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, now + 10_000, (t) => ({
      // The second one shows up 0.3 s later: same window.
      people: t >= 4_300 ? [near(40), near(420)] : [near(40)],
    }));
    const said = greetings(rig.said);
    expect(said).toHaveLength(1);
    expect(said[0]!.utterance.ruleId).toBe("scene.greeting.pair");
  });

  it("greets three people arriving together with their number", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, now + 10_000, () => ({ people: [near(20), near(250), near(480)] }));
    const said = greetings(rig.said);
    expect(said).toHaveLength(1);
    expect(said[0]!.utterance.text).toMatch(/tre/);
  });

  it("batches newcomers who arrive during the 8 s cooldown, if still there", () => {
    const rig = makeAgent();
    let now = warm(rig);
    // A at 4 s; B at 6 s and C at 7 s, both inside the cooldown of A's hello.
    now = run(rig, now, 30_000, (t) => ({
      people: [near(20), ...(t >= 6_000 ? [near(250)] : []), ...(t >= 7_000 ? [near(480)] : [])],
    }));
    const said = greetings(rig.said);
    expect(said).toHaveLength(2);
    expect(said[1]!.at - said[0]!.at).toBeGreaterThanOrEqual(8_000);
    expect(said[1]!.utterance.ruleId).toBe("scene.greeting.pair");
  });

  it("does not greet a newcomer who left before the cooldown ended", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, 30_000, (t) => ({
      people: [near(20), ...(t >= 6_000 && t < 8_000 ? [near(420)] : [])],
    }));
    expect(greetings(rig.said)).toHaveLength(1);
  });

  it("greets a person and their face once (the face is inside the person box)", () => {
    const rig = makeAgent();
    let now = warm(rig);
    // Person box x 40–160 px of 640: the face centre at 0.15 is inside it.
    now = run(rig, now, 20_000, () => ({ people: [near(40)], faces: [face(0.1)] }));
    expect(greetings(rig.said)).toHaveLength(1);
  });

  it("does not greet a person YOLOX loses for a moment", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, 40_000, (t) => ({
      // 2 s gaps every 6 s: the track is lost and found at the same place.
      people: t % 6_000 < 4_000 ? [near(40)] : [],
    }));
    expect(greetings(rig.said)).toHaveLength(1);
    expect(goodbyes(rig.said)).toHaveLength(0);
  });
});

describe("C3 arrivals — only people at the stand count", () => {
  it("ignores small people and faces in the background", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, 30_000, () => ({
      people: [far(20), far(300), far(500)],
      faces: [face(0.5, 0.04)],
    }));
    expect(greetings(rig.said)).toHaveLength(0);
    expect(rig.agent.presence.isPresent).toBe(false);
    expect(rig.agent.perception.world.people).toBe(0);
    expect(rig.agent.memory.working.focus.people).toBe(0);
    const snapshot = rig.agent.presence.snapshot(now);
    expect(snapshot.tracks.filter((track) => track.kind === "persona")).toHaveLength(3);
    expect(snapshot.tracks.every((track) => !track.atStand)).toBe(true);

    // One of them walks up to the stand: now a newcomer.
    now = run(rig, now, now + 5_000, () => ({ people: [far(20), near(300)] }));
    expect(greetings(rig.said)).toHaveLength(1);
    expect(rig.agent.memory.working.focus.people).toBe(1);
  });

  it("counts max(gated people, gated faces)", () => {
    const fusion = new PresenceFusion();
    for (let now = 0; now <= 1_000; now += 100) {
      fusion.observePersons([[0.1, 0.1, 0.3, 0.9]], now);
      fusion.observeFaces([face(0.1), face(0.6), face(0.8, 0.03)], now);
    }
    // One person box, two faces at the stand (the third face is far away).
    expect(fusion.update(1_000)).toMatchObject({ present: true, people: 2, faces: 2 });
  });

  it("persists the thresholds and gates with the new values", () => {
    const storage = memoryStorage();
    setPresenceThresholds({ minBoxHeight: 0.5, minFaceWidth: 0.12 }, storage);
    expect(presenceThresholds()).toEqual({ minBoxHeight: 0.5, minFaceWidth: 0.12 });
    expect(JSON.parse(storage.values.get(PRESENCE_THRESHOLDS_KEY)!)).toEqual({
      minBoxHeight: 0.5,
      minFaceWidth: 0.12,
    });
    expect(loadPresenceThresholds(storage)).toEqual({ minBoxHeight: 0.5, minFaceWidth: 0.12 });
    expect(loadPresenceThresholds(null)).toEqual(DEFAULT_PRESENCE_THRESHOLDS);

    // A 0.42 box is now in the background.
    const fusion = new PresenceFusion();
    for (let now = 0; now <= 1_000; now += 100) {
      fusion.observePersons([[0.1, 0.3, 0.3, 0.72]], now);
    }
    expect(fusion.update(1_000).present).toBe(false);
    expect(fusion.snapshot(1_000).thresholds.minBoxHeight).toBe(0.5);
  });
});

describe("C3 goodbye — only when everybody has left", () => {
  it("says one goodbye 3 s after the last person leaves", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, 14_000, () => ({ people: [near(40), near(420)], faces: [face(0.1)] }));
    const leftAt = now;
    now = run(rig, now, now + 30_000, empty);
    const said = goodbyes(rig.said);
    expect(said).toHaveLength(1);
    expect(said[0]!.at - leftAt).toBeGreaterThanOrEqual(2_800);
    expect(said[0]!.at - leftAt).toBeLessThan(4_000);
  });

  it("says nothing when only some of them leave (2 → 1)", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, 14_000, () => ({ people: [near(40), near(420)] }));
    const before = rig.said.length;
    now = run(rig, now, now + 20_000, () => ({ people: [near(40)] }));
    const after = rig.said.slice(before);
    expect(goodbyes(after)).toHaveLength(0);
    expect(greetings(after)).toHaveLength(0);
  });

  it("never says goodbye while a face is visible, even if YOLOX loses the person", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, 14_000, () => ({ people: [near(40)], faces: [face(0.1)] }));
    // Too close for YOLOX now: only the face.
    now = run(rig, now, now + 30_000, () => ({ faces: [face(0.1)] }));
    expect(goodbyes(rig.said)).toHaveLength(0);
    expect(greetings(rig.said)).toHaveLength(1);
  });

  it("says one goodbye per visit, at most one per 60 s", () => {
    const rig = makeAgent();
    let now = warm(rig);
    now = run(rig, now, 14_000, () => ({ people: [near(40)] }));
    now = run(rig, now, now + 10_000, empty);
    expect(goodbyes(rig.said)).toHaveLength(1);
    // A new visitor 10 s later: greeted, then leaves within 60 s of the goodbye.
    now = run(rig, now, now + 10_000, () => ({ people: [near(300)] }));
    expect(greetings(rig.said)).toHaveLength(2);
    now = run(rig, now, now + 10_000, empty);
    expect(goodbyes(rig.said)).toHaveLength(1);
  });

  it("leaves quietly when nobody was greeted (seeded at start)", () => {
    const rig = makeAgent();
    // Somebody is already there at start: seeded silently, never greeted.
    let now = run(rig, 0, 8_000, () => ({ people: [near(40)], faces: [face(0.1)] }));
    expect(greetings(rig.said)).toHaveLength(0);
    now = run(rig, now, now + 20_000, empty);
    expect(goodbyes(rig.said)).toHaveLength(0);
    expect(rig.agent.presence.isPresent).toBe(false);
  });
});

describe("C3 arrivals — Scheletro view", () => {
  it("greets a MoveNet pose even when its keypoints are weak", () => {
    const rig = makeAgent();
    // A real pose score, shoulders below the keypoint threshold: the old rule
    // (posePerson) saw nobody.
    const weak = {
      ...pose({}, 0.35),
      keypoints: keypoints({ 5: [280, 200, 0.1], 6: [360, 200, 0.1] }),
    };
    let now = 0;
    for (; now < 15_000; now += HEARTBEAT_MS) {
      rig.setClock(now);
      rig.agent.onPose(now < 5_000 ? pose({}, 0.1) : weak, now);
      rig.agent.tick(now);
    }
    expect(rig.agent.perception.world.posePerson).toBe(false);
    expect(rig.agent.presence.snapshot(now).sources.pose).toBe(true);
    expect(rig.agent.presence.isPresent).toBe(true);
    expect(rig.agent.scene.scene).not.toBe("attract");
    expect(greetings(rig.said)).toHaveLength(1);
  });
});

describe("C3 smile — queued while ARI is speaking", () => {
  it("waits in the queue slot and is spoken when the sentence ends", () => {
    const memory = new CognitiveMemory({ storage: null });
    const engine = new CognitiveEngine([], { rng: () => 0.5 });
    const machine = new SceneMachine({ rng: () => 0 });
    machine.reset(0, true);
    const world = { people: 1, crowd: false, posePerson: false, mode: "yolox" as const };
    const talk: SceneLine = {
      ruleId: "scene.prompt.yolox",
      scene: "play",
      kind: "prompt",
      text: "Mostrami un oggetto e provo a indovinare cos'è.",
      expression: "curious",
      gaze: "person",
      priority: 80,
    };
    engine.offerScene(talk, memory, 0);
    expect(engine.tick(0, memory)?.text).toBe(talk.text);
    expect(engine.speaking).toBe(true);

    machine.observe([{ type: "face", kind: "smile", score: 0.8, faces: 1, now: 1_500 }]);
    const smile = machine.step(1_500, memory, world);
    expect(smile?.ruleId).toMatch(/^scene\.smile/);
    expect(smile?.priority).toBe(SMILE_PRIORITY);
    engine.offerScene(smile!, memory, 1_500);
    expect(engine.queueLength).toBe(1);

    engine.endSpeech();
    const next = engine.tick(3_000, memory);
    expect(next?.ruleId).toBe(smile!.ruleId);
  });
});

describe("C3 smile — during Engage (greeting and self-introduction)", () => {
  /** Scene + engine with sentences of `sentenceMs`; a smile at `smileAt`. */
  function engage(sentenceMs: number, smileAt: number): Array<{ ruleId: string; at: number }> {
    const memory = new CognitiveMemory({ storage: null });
    const engine = new CognitiveEngine([], { rng: () => 0.5 });
    const machine = new SceneMachine({ rng: () => 0 });
    machine.reset(0);
    const world = { people: 1, crowd: false, posePerson: false, mode: "yolox" as const };
    const spoken: Array<{ ruleId: string; at: number }> = [];
    let endAt: number | null = null;
    for (let now = 1_000; now <= 30_000; now += HEARTBEAT_MS) {
      if (now === smileAt) {
        machine.observe([{ type: "face", kind: "smile", score: 0.8, faces: 1, now }]);
      }
      if (endAt !== null && now >= endAt) {
        engine.endSpeech();
        endAt = null;
      }
      const line = machine.step(now, memory, world);
      if (line !== null) {
        engine.offerScene(line, memory, now);
      }
      const utterance = engine.tick(now, memory);
      if (utterance !== null) {
        machine.noteSpoken(utterance, memory, now);
        spoken.push({ ruleId: utterance.ruleId, at: now });
        endAt = now + sentenceMs;
      }
    }
    expect(machine.scene).toBe("play");
    return spoken;
  }

  it("never displaces the self-introduction from the queue slot", () => {
    // A 4 s greeting, a smile near its end: the smile line holds the voice
    // when the next Engage step would come; the intro must still be said.
    const ids = engage(4_000, 4_500).map((entry) => entry.ruleId);
    expect(ids.slice(0, 2).map((id) => id.split(".").slice(0, 2).join("."))).toEqual([
      "scene.greeting",
      "scene.smile",
    ]);
    expect(ids.filter((id) => id === "scene.intro")).toHaveLength(1);
    expect(ids.filter((id) => id.startsWith("scene.prompt."))).toHaveLength(1);
  });

  it("reacts right after the greeting, then keeps one intro and one invitation", () => {
    // 1.5 s sentences; the visitor smiles while ARI is still greeting.
    const spoken = engage(1_500, 1_500);
    const ids = spoken.map((entry) => entry.ruleId);
    expect(ids[0]).toMatch(/^scene\.greeting\./);
    expect(ids[1]).toMatch(/^scene\.smile/);
    // Right after the greeting ended, not after a full Engage step.
    expect(spoken[1]!.at - spoken[0]!.at).toBeLessThanOrEqual(2_000);
    expect(ids.filter((id) => id.startsWith("scene.greeting."))).toHaveLength(1);
    expect(ids.filter((id) => id === "scene.intro")).toHaveLength(1);
    expect(ids.filter((id) => id.startsWith("scene.prompt."))).toHaveLength(1);
    expect(ids.indexOf("scene.intro")).toBeLessThan(
      ids.findIndex((id) => id.startsWith("scene.prompt.")),
    );
  });
});
