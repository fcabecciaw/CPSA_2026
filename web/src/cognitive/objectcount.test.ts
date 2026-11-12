/**
 * `memory.object_count`: the day's repeated sightings are counted in spoken
 * Italian, with gender/number agreement and no "first time" object phrase.
 */

import { describe, expect, it } from "vitest";

import {
  objectCountSlots,
  ordinalWord,
  renderPhrase,
} from "./grammar";
import { CognitiveMemory } from "./memory";
import { Perception } from "./perception";
import { RULEBOOK, allPhrases } from "./rulebook";
import { CognitiveEngine } from "./rules";
import { showcase, detection } from "./testkit";

const counting = RULEBOOK.find((rule) => rule.id === "memory.object_count");

function phrases(label: string, count: number): string[] {
  if (counting === undefined) {
    throw new Error("memory.object_count mancante");
  }
  const slots = objectCountSlots(label, count);
  return counting.phrases.map((phrase) => renderPhrase(phrase, slots));
}

describe("object count — Italian agreement", () => {
  it("agrees the ordinal with the class gender", () => {
    expect(phrases("cellulare", 3)).toContain("Terzo cellulare di oggi.");
    expect(phrases("tazza", 3)).toContain("Terza tazza di oggi.");
    expect(phrases("libro", 5)).toContain("Quinto libro di oggi.");
    expect(phrases("libro", 5)).toContain("Quinto libro della giornata.");
    expect(phrases("mela", 4)).toContain("Quarta mela di oggi.");
    expect(phrases("auto", 4)).toContain("Quarta auto di oggi.");
    expect(phrases("pianta", 2)).toContain("Seconda pianta di oggi.");
    expect(phrases("tv", 2)).toContain("Seconda tv di oggi.");
  });

  it("keeps elided and impure articles correct", () => {
    expect(phrases("ombrello", 3)).toContain("Terzo ombrello di oggi.");
    expect(phrases("ombrello", 3)).toContain("Ecco l'ombrello numero tre.");
    expect(phrases("arancia", 2)).toContain("Ecco l'arancia numero due.");
    expect(phrases("zaino", 2)).toContain("Ecco lo zaino numero due.");
    expect(phrases("sci", 2)).toContain("Secondo sci di oggi.");
    expect(phrases("sci", 2)).toContain("Ecco lo sci numero due.");
  });

  it("uses the plural for totals and the singular for the numbered object", () => {
    expect(phrases("cellulare", 3)).toContain("E siamo a ben tre cellulari!");
    expect(phrases("tazza", 4)).toContain("Già quattro tazze oggi!");
    expect(phrases("bottiglia", 6)).toContain("Ecco la bottiglia numero sei.");
    expect(phrases("bottiglia", 6)).toContain("Sesta bottiglia di oggi.");
    expect(phrases("libro", 2)).toContain("Ecco il libro numero due.");
    expect(phrases("forbici", 3)).toContain("Ancora forbici? Siamo a quota tre.");
    expect(phrases("forbici", 3)).toContain("Terzo paio di forbici di oggi.");
    expect(phrases("forbici", 3)).toContain("Ecco il paio di forbici numero tre.");
  });

  it("spells 2..10 and falls back to digits above ten", () => {
    expect(ordinalWord(2, "m")).toBe("secondo");
    expect(ordinalWord(10, "f")).toBe("decima");
    expect(ordinalWord(11, "m")).toBe("11°");
    expect(ordinalWord(12, "f")).toBe("12ª");
    expect(phrases("tazza", 12)).toContain("E siamo a ben 12 tazze!");
    expect(phrases("cellulare", 12)).toContain("Ecco il cellulare numero 12.");
    expect(phrases("forbici", 11)).toContain("11° paio di forbici di oggi.");
  });
});

describe("memory.object_count — first sighting and repetition", () => {
  it("has no 'prima volta' object phrase anywhere in the book", () => {
    expect(counting).toBeDefined();
    expect(RULEBOOK.some((rule) => rule.id === "memory.first_object")).toBe(false);
    const objectText = RULEBOOK.filter((rule) => rule.category === "object")
      .flatMap((rule) => [...rule.phrases])
      .join(" ")
      .toLowerCase();
    expect(objectText).not.toMatch(/prima volta|prima vista|novità di oggi/);
    expect(allPhrases().join(" ").toLowerCase()).not.toMatch(
      /prima volta.*oggetto|novità di oggi/,
    );
  });

  it("matches only from the second stable appearance and never in crowd", () => {
    if (counting === undefined) {
      throw new Error("memory.object_count mancante");
    }
    const memory = new CognitiveMemory({ storage: null });
    memory.setFocus({ mode: "yolox", people: 1, crowd: false });
    const appear = {
      type: "appear" as const,
      label: "tazza",
      count: 1,
      score: 0.9,
      area: 0.02,
      scenery: false,
      doubtful: false,
      now: 0,
    };
    memory.observe([appear], 0);
    expect(counting.when(appear, memory)).toBe(false);

    memory.observe([{ ...appear, now: 30_000 }], 30_000);
    expect(counting.when({ ...appear, now: 30_000 }, memory)).toBe(true);

    memory.setFocus({ crowd: true, people: 5 });
    expect(counting.when({ ...appear, now: 30_000 }, memory)).toBe(false);

    memory.setFocus({ crowd: false, people: 1 });
    expect(counting.when({ ...appear, scenery: true, now: 30_000 }, memory)).toBe(false);
    expect(counting.when({ ...appear, doubtful: true, now: 30_000 }, memory)).toBe(false);
    expect(counting.when({ ...appear, label: "persona", now: 30_000 }, memory)).toBe(false);
  });

  it("keeps the class rule for the first sighting and picks the count later", () => {
    const memory = new CognitiveMemory({ storage: null });
    memory.setFocus({ mode: "yolox", people: 1, crowd: false });
    const engine = new CognitiveEngine(RULEBOOK, {
      rng: () => 0.5,
      gapMs: 4_000,
    });
    const appear = (now: number) => ({
      type: "appear" as const,
      label: "cellulare",
      count: 1,
      score: 0.9,
      area: 0.02,
      scenery: false,
      doubtful: false,
      now,
    });

    memory.observe([appear(0)], 0);
    engine.consider([appear(0)], memory, 0);
    const first = engine.tick(0, memory);
    expect(first?.ruleId).toBe("object.cellulare");
    expect(first?.text).not.toMatch(/prima volta/i);

    engine.endSpeech();
    memory.observe([appear(90_000)], 90_000);
    engine.consider([appear(90_000)], memory, 90_000);
    const second = engine.tick(90_000, memory);
    expect(second?.ruleId).toBe("memory.object_count");
    expect(second?.text).toBe("Secondo cellulare di oggi.");

    engine.endSpeech();
    memory.observe([appear(150_000)], 150_000);
    engine.consider([appear(150_000)], memory, 150_000);
    const third = engine.tick(150_000, memory);
    expect(third?.ruleId).toBe("memory.object_count");
    expect(third?.text).toMatch(/^Terzo cellulare (di oggi|della giornata)\.$/);
  });
});

describe("day count — stable appearances only", () => {
  it("counts one per stable appearance and ignores flicker and the return event", () => {
    const perception = new Perception();
    const memory = new CognitiveMemory({ storage: null });
    memory.setFocus({ mode: "yolox", people: 1, crowd: false });
    perception.reset(0);

    let now = 0;
    const tazza = () => showcase([detection("tazza", 41, [80, 80, 200, 200])]);
    const feed = (objects: () => ReturnType<typeof showcase>, until: number) => {
      while (now < until) {
        const events = perception.update(objects(), now);
        memory.observe(events, now);
        now += 100;
      }
    };

    feed(() => showcase([]), 4_000);
    feed(tazza, 5_000);
    expect(memory.day.classes.tazza?.times).toBe(1);

    // One missing frame is flicker: the track never becomes absent.
    feed(() => showcase([]), 5_100);
    feed(tazza, 6_000);
    expect(memory.day.classes.tazza?.times).toBe(1);

    // A real absence, then a new stable appearance: it counts again.
    feed(() => showcase([]), 9_000);
    feed(tazza, 10_000);
    expect(memory.day.classes.tazza?.times).toBe(2);

    // `return` belongs to the same appearance and must not count twice.
    const before = memory.day.classes.tazza?.times ?? 0;
    memory.observe(
      [
        {
          type: "return",
          label: "tazza",
          absenceS: 35,
          count: 1,
          now: 40_000,
        },
      ],
      40_000,
    );
    expect(memory.day.classes.tazza?.times).toBe(before);
  });
});
