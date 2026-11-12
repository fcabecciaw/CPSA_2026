/**
 * The dialogue scenes (R2-D): Attract → Engage → Play → Farewell, continuity,
 * no-repetition of the idle content, greeting families, questions answered by
 * a nod, and the one-voice contract with the rule engine. Each `describe`
 * names the requirement it protects.
 */

import { describe, expect, it } from "vitest";

import { FACTS } from "./content/facts";
import { POEMS } from "./content/poems";
import type { CognitiveEvent } from "./events";
import { CognitiveMemory } from "./memory";
import { CognitiveEngine, type SceneLine, type Utterance } from "./rules";
import { SceneMachine, type SceneWorld } from "./scene";

function world(overrides: Partial<SceneWorld> = {}): SceneWorld {
  return {
    people: 0,
    crowd: false,
    posePerson: false,
    mode: "yolox",
    ...overrides,
  };
}

function memory(): CognitiveMemory {
  return new CognitiveMemory({ storage: null });
}

function utterance(line: SceneLine, now: number): Utterance {
  return {
    ruleId: line.ruleId,
    text: line.text,
    expression: line.expression,
    gaze: line.gaze,
    event: {
      type: "scene",
      scene: line.scene,
      kind: line.kind,
      text: line.text,
      now,
    },
    objectKey: line.objectKey ?? null,
    priority: line.priority,
    utility: line.priority,
    now,
    scene: line.scene,
    sceneKind: line.kind,
  };
}

function say(
  machine: SceneMachine,
  line: SceneLine,
  day: CognitiveMemory,
  now: number,
): void {
  machine.noteSpoken(utterance(line, now), day, now);
}

describe("Attract — facts and poems, never twice a day", () => {
  it("alternates fact and poem and keeps facts first", () => {
    const machine = new SceneMachine({ rng: () => 0, attractMinMs: 0, attractMaxMs: 0 });
    machine.reset(0);
    const day = memory();
    const kinds: string[] = [];
    for (let step = 0; step < 5; step += 1) {
      const line = machine.step(step * 1_000, day, world());
      expect(line).not.toBeNull();
      kinds.push(line!.kind);
      say(machine, line!, day, step * 1_000);
    }
    // Fact, poem, fact, invitation (every fourth idle line), poem.
    expect(kinds).toEqual(["fact", "poem", "fact", "invitation", "poem"]);
  });

  it("uses every fact and poem once, then stays quiet", () => {
    const machine = new SceneMachine({ rng: () => 0.4, attractMinMs: 0, attractMaxMs: 0 });
    machine.reset(0);
    const day = memory();
    const seen = new Set<string>();
    const kinds = new Map<string, number>();
    for (let step = 0; step < 400; step += 1) {
      const line = machine.step(step * 1_000, day, world());
      if (line === null) {
        continue;
      }
      if (line.kind === "fact" || line.kind === "poem") {
        expect(seen.has(line.ruleId), `ripetuto: ${line.ruleId}`).toBe(false);
        seen.add(line.ruleId);
        kinds.set(line.kind, (kinds.get(line.kind) ?? 0) + 1);
      }
      say(machine, line, day, step * 1_000);
    }
    expect(kinds.get("fact")).toBe(FACTS.length);
    expect(kinds.get("poem")).toBe(POEMS.length);
    expect(seen.size).toBe(FACTS.length + POEMS.length);
  });

  it("says the invitation only now and then", () => {
    const machine = new SceneMachine({ rng: () => 0.1, attractMinMs: 0, attractMaxMs: 0 });
    machine.reset(0);
    const day = memory();
    let invitations = 0;
    for (let step = 0; step < 12; step += 1) {
      const line = machine.step(step * 1_000, day, world());
      if (line?.kind === "invitation") {
        invitations += 1;
        expect(line.text).toMatch(/Avvicinati|Vieni più vicino|ti racconto/);
      }
      if (line !== null) {
        say(machine, line, day, step * 1_000);
      }
    }
    expect(invitations).toBe(3);
  });
});

describe("Engage — one greeting adapted to the number of people", () => {
  function engage(people: number, crowd = people >= 5): SceneLine {
    const machine = new SceneMachine({ rng: () => 0, engageStepMs: 100 });
    machine.reset(0);
    const line = machine.step(1_000, memory(), world({ people, crowd }));
    expect(line).not.toBeNull();
    return line!;
  }

  it("addresses one person directly, two people together, groups as a group", () => {
    const single = engage(1);
    expect(single.ruleId).toBe("scene.greeting.single");
    expect(single.text).toMatch(/ti|Benvenuto/i);
    expect(single.expression).toBe("happy");

    const pair = engage(2);
    expect(pair.ruleId).toBe("scene.greeting.pair");
    expect(pair.text).toMatch(/voi due|Siete arrivati in due|coppia/i);

    const group = engage(4);
    expect(group.ruleId).toBe("scene.greeting.group");
    expect(group.text).toMatch(/Quattro|4|tutti|gruppo/);

    const crowd = engage(7);
    expect(crowd.ruleId).toBe("scene.greeting.crowd");
    expect(crowd.text).toMatch(/tutti|folla|Benvenuti/);
  });

  it("greets once, then introduces itself the first time, then invites", () => {
    const machine = new SceneMachine({ rng: () => 0, engageStepMs: 1_000 });
    machine.reset(0);
    const day = memory();
    const present = world({ people: 1 });
    const greeting = machine.step(1_000, day, present);
    expect(greeting?.kind).toBe("greeting");
    say(machine, greeting!, day, 1_000);

    // Not before the step gap.
    expect(machine.step(1_500, day, present)).toBeNull();
    const intro = machine.step(2_000, day, present);
    expect(intro?.kind).toBe("intro");
    expect(intro?.text).toMatch(/ARI|Piacere/i);
    say(machine, intro!, day, 2_000);

    const prompt = machine.step(3_100, day, present);
    expect(prompt?.kind).toBe("prompt");
    expect(prompt?.text).toMatch(/oggetto|salutarmi|gesto/);
    say(machine, prompt!, day, 3_100);
    expect(machine.scene).toBe("play");
  });

  it("skips the self-introduction after the first time today", () => {
    const machine = new SceneMachine({ rng: () => 0, engageStepMs: 0 });
    machine.reset(0);
    const day = memory();
    day.rule("scene.intro").count = 1;
    const present = world({ people: 1 });
    say(machine, machine.step(1_000, day, present)!, day, 1_000);
    const prompt = machine.step(1_000, day, present);
    expect(prompt?.kind).toBe("prompt");
    expect(machine.scene).toBe("play");
  });

  it("asks for a wave in MoveNet and for an object in YOLO", () => {
    const movenet = new SceneMachine({ rng: () => 0, engageStepMs: 0 });
    movenet.reset(0);
    const day = memory();
    day.rule("scene.intro").count = 1;
    const pose = world({ people: 0, posePerson: true, mode: "movenet" });
    say(movenet, movenet.step(1_000, day, pose)!, day, 1_000);
    const prompt = movenet.step(1_000, day, pose);
    expect(prompt?.text).toMatch(/salutarmi|mano|gesto/i);
  });
});

describe("Farewell — one goodbye, then Attract again", () => {
  it("says goodbye once and waits ~10 s before the idle content", () => {
    const machine = new SceneMachine({ rng: () => 0, attractMinMs: 5_000, attractMaxMs: 5_000 });
    machine.reset(0, true);
    const day = memory();
    const empty = world();
    const goodbye = machine.step(1_000, day, empty);
    expect(goodbye?.kind).toBe("goodbye");
    expect(machine.scene).toBe("farewell");
    say(machine, goodbye!, day, 1_000);

    expect(machine.step(10_500, day, empty)).toBeNull();
    expect(machine.scene).toBe("farewell");

    expect(machine.step(11_001, day, empty)).toBeNull();
    expect(machine.scene).toBe("attract");
    const firstFact = machine.step(16_002, day, empty);
    expect(firstFact).not.toBeNull();
    expect(firstFact?.kind).toBe("fact");
  });

  it("does not greet a short absence, and greets a new arrival", () => {
    const machine = new SceneMachine({ rng: () => 0, farewellMs: 10_000 });
    machine.reset(0, true);
    const day = memory();
    machine.step(1_000, day, world());
    // Back after 3 s: straight to Play, no second greeting.
    const back = machine.step(4_000, day, world({ people: 1 }));
    expect(back).toBeNull();
    expect(machine.scene).toBe("play");

    // A long absence: the return greeting family is allowed.
    const later = new SceneMachine({ rng: () => 0 });
    later.reset(0);
    const day2 = memory();
    later.observe([
      { type: "count", from: 0, to: 1, quiet: false, returnAfterS: 45, now: 500 } as CognitiveEvent,
    ]);
    const greeting = later.step(1_000, day2, world({ people: 1 }));
    expect(greeting?.ruleId).toBe("scene.greeting.return");
  });
});

describe("Play — continuity, smiles and yes/no questions", () => {
  it("refers back to the object it named before", () => {
    const machine = new SceneMachine({ rng: () => 0.2 });
    machine.reset(0, true);
    const day = memory();
    const phone = machine.step(0, day, world({ people: 1 }));
    expect(phone).toBeNull();
    // The rulebook names the phone; the scene remembers it.
    machine.noteSpoken(
      {
        ruleId: "object.cellulare",
        text: "Ecco il cellulare: notifiche in arrivo.",
        expression: "curious",
        gaze: "object",
        event: { type: "appear", label: "cellulare", count: 1, score: 0.9, area: 0.02, scenery: false, doubtful: false, now: 0 },
        objectKey: "cellulare",
        priority: 58,
        utility: 60,
        now: 0,
      },
      day,
      0,
    );
    machine.observe([
      { type: "appear", label: "tazza", count: 1, score: 0.9, area: 0.02, scenery: false, doubtful: false, now: 5_000 },
    ]);
    const line = machine.step(5_000, day, world({ people: 1 }));
    expect(line?.kind).toBe("continuity");
    expect(line?.text).toBe("Prima il telefono, ora una tazza: pausa caffè?");
    expect(line?.expression).toBe("curious");
  });

  it("reacts to a smile, once per cooldown, and varies the repetition", () => {
    const machine = new SceneMachine({ rng: () => 0 });
    machine.reset(0, true);
    const day = memory();
    const present = world({ people: 1 });
    machine.observe([{ type: "face", kind: "smile", score: 0.9, faces: 1, now: 2_000 }]);
    const first = machine.step(2_000, day, present);
    expect(first?.ruleId).toBe("scene.smile");
    expect(first?.text).toMatch(/sorriso/i);
    expect(first?.expression).toBe("happy");

    machine.observe([{ type: "face", kind: "smile", score: 0.9, faces: 1, now: 5_000 }]);
    expect(machine.step(5_000, day, present)).toBeNull();

    machine.observe([{ type: "face", kind: "smile", score: 0.9, faces: 1, now: 12_000 }]);
    const again = machine.step(12_000, day, present);
    expect(again?.ruleId).toBe("scene.smile.again");
  });

  it("answers a yes/no question with the matching nod or shake", () => {
    const machine = new SceneMachine({
      rng: () => 0,
      questionAfterMs: 0,
      questionIdleMs: 0,
    });
    machine.reset(0, true);
    const day = memory();
    const present = world({ people: 1 });
    const question = machine.step(1_000, day, present);
    expect(question?.kind).toBe("question");
    expect(question?.text).toBe("Ti piace la tecnologia?");
    say(machine, question!, day, 1_000);
    expect(machine.pendingQuestion).toBe("Ti piace la tecnologia?");

    machine.observe([{ type: "face", kind: "nod", score: 0.8, faces: 1, now: 3_000 }]);
    const answer = machine.step(3_000, day, present);
    expect(answer?.kind).toBe("answer");
    expect(answer?.text).toBe("Lo sapevo!");
    expect(machine.pendingQuestion).toBeNull();

    const second = new SceneMachine({
      rng: () => 0,
      questionAfterMs: 0,
      questionIdleMs: 0,
    });
    second.reset(0, true);
    const day2 = memory();
    const asked = second.step(1_000, day2, present);
    say(second, asked!, day2, 1_000);
    second.observe([{ type: "face", kind: "head_shake", score: 0.8, faces: 1, now: 3_000 }]);
    const denied = second.step(3_000, day2, present);
    expect(denied?.kind).toBe("answer");
    expect(denied?.text).toMatch(/Rispetto la tua scelta|Meglio la semplicità|resto comunque simpatico/);
  });

  it("says 'che bel sorriso' only in Play, not while nobody is there", () => {
    const machine = new SceneMachine({ rng: () => 0 });
    machine.reset(0);
    const day = memory();
    machine.observe([{ type: "face", kind: "smile", score: 0.9, faces: 1, now: 2_000 }]);
    expect(machine.step(2_000, day, world())).toBeNull();
  });
});

describe("F-E4 #13 — scenes and rules share the one voice", () => {
  it("never speaks two scene lines inside the 4 s gap", () => {
    const engine = new CognitiveEngine([], { rng: () => 0.5 });
    const day = memory();
    const machine = new SceneMachine({ rng: () => 0.5, attractMinMs: 0, attractMaxMs: 0 });
    machine.reset(0);

    const first = machine.step(0, day, world());
    expect(first).not.toBeNull();
    engine.offerScene(first!, day, 0);
    expect(engine.tick(0, day)?.text).toBe(first!.text);

    const second = machine.step(100, day, world());
    expect(second).not.toBeNull();
    engine.offerScene(second!, day, 100);
    expect(engine.tick(100, day)).toBeNull();
    expect(engine.tick(3_999, day)).toBeNull();
    engine.endSpeech();
    expect(engine.tick(3_999, day)).toBeNull();

    engine.offerScene(second!, day, 4_000);
    expect(engine.tick(4_000, day)?.text).toBe(second!.text);
  });
});
