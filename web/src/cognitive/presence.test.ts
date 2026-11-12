/**
 * Presence fusion (C3) at the agent level: a person is there while YOLOX, a
 * laptop face or a MoveNet skeleton sees them; the goodbye needs 5 s with no
 * source at all, a greeting earlier in the visit, and at most one per 60 s; a
 * return within 15 s is the same visit. Plus the smile reaction that waits in
 * the queue slot while ARI is speaking.
 */

import { describe, expect, it } from "vitest";

import { CognitiveAgent, HEARTBEAT_MS } from "./agent";
import { CognitiveMemory } from "./memory";
import { Perception } from "./perception";
import { PresenceFusion } from "./presence";
import { RULEBOOK } from "./rulebook";
import { CognitiveEngine, type SceneLine, type Utterance } from "./rules";
import { SMILE_PRIORITY, SceneMachine } from "./scene";
import { keypoints, person, pose, showcase } from "./testkit";

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

/** What each source sees at a given time. */
interface Scene {
  yolo: boolean;
  faces: number;
}

/** Drive the agent at the real cadence: 10 Hz showcase/faces, 10 Hz heartbeat. */
function run(
  rig: ReturnType<typeof makeAgent>,
  from: number,
  to: number,
  scene: (now: number) => Scene,
): number {
  let now = from;
  for (; now < to; now += HEARTBEAT_MS) {
    rig.setClock(now);
    const seen = scene(now);
    rig.agent.onShowcase(showcase(seen.yolo ? [person(260)] : []), now);
    rig.agent.onFaces(seen.faces, now);
    rig.agent.tick(now);
  }
  return now;
}

const goodbyes = (said: readonly Said[]): Said[] =>
  said.filter((entry) => entry.utterance.sceneKind === "goodbye");
const greetings = (said: readonly Said[]): Said[] =>
  said.filter(
    (entry) =>
      entry.utterance.sceneKind === "greeting" || entry.utterance.ruleId.startsWith("person."),
  );

/** Empty stand for the warm-up, then a visitor who gets greeted. */
function arriveAndGreet(rig: ReturnType<typeof makeAgent>): number {
  let now = run(rig, 0, 5_000, () => ({ yolo: false, faces: 0 }));
  now = run(rig, now, 12_000, () => ({ yolo: true, faces: 1 }));
  expect(greetings(rig.said).length).toBeGreaterThan(0);
  expect(rig.agent.presence.snapshot(now).greeted).toBe(true);
  return now;
}

describe("C3 presence — YOLOX flicker with a face in view", () => {
  it("never says goodbye while the face is visible", () => {
    const rig = makeAgent();
    let now = arriveAndGreet(rig);
    const greeted = greetings(rig.said).length;
    // Very close to the camera: YOLOX sees "persona" 1 s out of 4 at best,
    // the laptop sees the face throughout.
    now = run(rig, now, 60_000, (t) => ({ yolo: t % 4_000 < 1_000, faces: 1 }));
    expect(goodbyes(rig.said)).toHaveLength(0);
    expect(rig.said.some((entry) => /a presto|alla prossima/i.test(entry.utterance.text))).toBe(
      false,
    );
    expect(rig.agent.scene.scene).not.toBe("farewell");
    // No second greeting either: the YOLOX count never dropped to 0.
    expect(greetings(rig.said)).toHaveLength(greeted);
    expect(rig.agent.presence.isPresent).toBe(true);
  });

  it("does not say goodbye when YOLOX loses the person for 4 s and no face is seen", () => {
    const rig = makeAgent();
    let now = arriveAndGreet(rig);
    now = run(rig, now, now + 4_000, () => ({ yolo: false, faces: 0 }));
    now = run(rig, now, now + 2_000, () => ({ yolo: true, faces: 0 }));
    expect(goodbyes(rig.said)).toHaveLength(0);
  });
});

describe("C3 presence — all sources absent", () => {
  it("says one goodbye after 5 s with nobody, and only one in 60 s", () => {
    const rig = makeAgent();
    let now = arriveAndGreet(rig);
    const leftAt = now;
    now = run(rig, now, now + 30_000, () => ({ yolo: false, faces: 0 }));
    const said = goodbyes(rig.said);
    expect(said).toHaveLength(1);
    // The persona track needs 1.5 s to drop, then 5 s with no source at all.
    expect(said[0]!.at - leftAt).toBeGreaterThanOrEqual(5_000);
    expect(said[0]!.at - leftAt).toBeLessThan(9_000);

    // Back after 30 s (a new visit, greeted again), gone again within 60 s of
    // the first goodbye: no second goodbye.
    now = run(rig, now, now + 8_000, () => ({ yolo: true, faces: 1 }));
    now = run(rig, now, now + 15_000, () => ({ yolo: false, faces: 0 }));
    expect(goodbyes(rig.said)).toHaveLength(1);
  });

  it("leaves quietly when nobody was greeted in this visit", () => {
    const rig = makeAgent();
    // Somebody is already there at start: seeded silently, never greeted.
    let now = run(rig, 0, 8_000, () => ({ yolo: true, faces: 1 }));
    expect(greetings(rig.said)).toHaveLength(0);
    now = run(rig, now, now + 20_000, () => ({ yolo: false, faces: 0 }));
    expect(goodbyes(rig.said)).toHaveLength(0);
    expect(rig.agent.presence.isPresent).toBe(false);
  });
});

describe("C3 presence — a return within 15 s is the same visit", () => {
  it("does not greet again after a 10 s absence, and greets after 20 s", () => {
    const rig = makeAgent();
    let now = arriveAndGreet(rig);
    const before = greetings(rig.said).length;
    now = run(rig, now, now + 10_000, () => ({ yolo: false, faces: 0 }));
    now = run(rig, now, now + 8_000, () => ({ yolo: true, faces: 1 }));
    expect(greetings(rig.said)).toHaveLength(before);
    expect(rig.agent.scene.scene).toBe("play");

    // Away for 25 s: a new visit, a new greeting.
    now = run(rig, now, now + 25_000, () => ({ yolo: false, faces: 0 }));
    now = run(rig, now, now + 8_000, () => ({ yolo: true, faces: 1 }));
    expect(greetings(rig.said).length).toBeGreaterThan(before);
  });
});

describe("C3 presence — Scheletro view", () => {
  it("is present with a MoveNet pose even when its keypoints are weak", () => {
    const rig = makeAgent();
    // A real pose score, shoulders below the keypoint threshold: the old rule
    // (posePerson) saw nobody.
    const weak = { ...pose({}, 0.35), keypoints: keypoints({ 5: [280, 200, 0.1], 6: [360, 200, 0.1] }) };
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
    expect(greetings(rig.said).length).toBeGreaterThan(0);
  });

  it("counts the faces for the greeting when YOLOX is not running", () => {
    const fusion = new PresenceFusion();
    fusion.observePose(0.8, 0);
    fusion.observeFaces(2, 0);
    expect(fusion.update(0)).toMatchObject({ present: true, people: 2 });
    fusion.observeFaces(0, 100);
    expect(fusion.update(100)).toMatchObject({ present: true, people: 1 });
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
