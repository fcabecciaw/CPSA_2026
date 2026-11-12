/**
 * The TTS speech catalog (T1): every pre-renderable line, rendered with the
 * real grammar, unique, without slots and in pregenerate priority order.
 */

import { describe, expect, it } from "vitest";

import { FACTS } from "./content/facts";
import { SPECIFIC_CLASSES } from "./rulebook";
import { SPEECH_GROUPS, speechCatalog } from "./speechcatalog";

describe("T1 — speech catalog", () => {
  const catalog = speechCatalog();
  const texts = catalog.map((line) => line.text);

  it("is unique, slot-free and ordered scene, indoor, counts, rest", () => {
    expect(new Set(texts).size).toBe(texts.length);
    expect(texts.filter((text) => /\{\w+\}/.test(text))).toEqual([]);
    const order = catalog.map((line) => SPEECH_GROUPS.indexOf(line.group));
    expect(order).toEqual([...order].sort((a, b) => a - b));
  });

  it("covers the scenes, facts, object variants, counts and greetings", () => {
    expect(SPECIFIC_CLASSES).toHaveLength(34);
    for (const fact of FACTS) {
      expect(texts).toContain(fact.text);
    }
    expect(texts).toContain("Psst! Sì, dico proprio a te. Vieni a dare un'occhiata!");
    expect(texts).toContain("Vedo la tazza. Caffè o tè?");
    expect(texts).toContain("Ecco due tazze! Peccato, io il caffè non posso berlo.");
    expect(texts).toContain("Mi sembra di vedere un'auto… possibile?");
    expect(texts).toContain("Terza tazza di oggi.");
    expect(texts).toContain("Già otto tazze oggi!");
    expect(texts).toContain("Ciao a tutti! Siete in 3.");
    expect(texts).toContain("Ciao a tutti! Siete in otto.");
    expect(texts).toContain("Ancora il libro? Non me lo aspettavo!");
    const counts = catalog.filter((line) => line.group === "counts");
    expect(new Set(counts.map((line) => line.source))).toEqual(new Set(["memory.object_count"]));
  });
});
