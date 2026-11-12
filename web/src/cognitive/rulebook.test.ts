/**
 * Coverage of the rulebook and the grammar table. Each `describe` names the
 * F-E4 corner it protects.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  COCO_IT_LABELS,
  articleFor,
  nounPhrase,
  renderPhrase,
  wordCount,
} from "./grammar";
import {
  RULEBOOK,
  RULEBOOK_STATS,
  allPhrases,
} from "./rulebook";
import type { Rule } from "./rules";

function pythonLabels(): string[] {
  const path = fileURLToPath(new URL("../../../src/cpsa/yolox.py", import.meta.url));
  const source = readFileSync(path, "utf8");
  const block = /COCO80_IT\s*:[^=]*=\s*\(([\s\S]*?)\n\)/.exec(source);
  if (block === null) {
    throw new Error("COCO80_IT non trovato in src/cpsa/yolox.py");
  }
  return [...block[1].matchAll(/"([^"]+)"/g)].map((match) => match[1]);
}

describe("F-E4 #8 — Italian grammar for all 80 COCO classes", () => {
  it("covers exactly the server's COCO80_IT table, in order", () => {
    const fromPython = pythonLabels();
    expect(fromPython).toHaveLength(80);
    expect([...COCO_IT_LABELS]).toEqual(fromPython);
  });

  it("renders the article, gender, plural and plural-only classes", () => {
    expect(nounPhrase("cellulare")).toBe("il cellulare");
    expect(nounPhrase("tazza")).toBe("la tazza");
    expect(nounPhrase("ombrello")).toBe("l'ombrello");
    expect(nounPhrase("auto")).toBe("l'auto");
    expect(nounPhrase("zaino")).toBe("lo zaino");
    expect(nounPhrase("sci")).toBe("lo sci");
    expect(nounPhrase("sci", 2)).toBe("due sci");
    expect(nounPhrase("forbici")).toBe("le forbici");
    expect(nounPhrase("tazza", 2)).toBe("due tazze");
    expect(nounPhrase("forbici", 3)).toBe("tre forbici");
    expect(nounPhrase("persona", 2)).toBe("due persone");
    expect(articleFor("mela", 2)).toBe("le");
  });

  it("has a class rule for every COCO label", () => {
    for (const label of COCO_IT_LABELS) {
      const rule = RULEBOOK.find((entry) => entry.classes?.includes(label));
      expect(rule, `nessuna regola per ${label}`).toBeDefined();
    }
  });

  it("keeps the phrase slots renderable and short", () => {
    const allowed =
      /^\{(oggetto|unoggetto|oggetto2|n|n2|ora|minuti|secondi|fps|dpu|totale|altro|Ordinale|singolare|conteggio|plurale|oggettoNumero|numero|volte)\}$/;
    for (const phrase of allPhrases()) {
      expect(wordCount(phrase)).toBeLessThanOrEqual(12);
      for (const slot of phrase.matchAll(/\{[^}]+\}/g)) {
        expect(slot[0], `slot sconosciuto in "${phrase}"`).toMatch(allowed);
      }
    }
    expect(renderPhrase("Vedo {oggetto}: {n}.", { oggetto: "due tazze", n: "2" })).toBe(
      "Vedo due tazze: 2.",
    );
  });
});

describe("rulebook size and families (AC-E3)", () => {
  it("has at least 70 rules and 150 Italian phrases", () => {
    expect(RULEBOOK_STATS.rules).toBeGreaterThanOrEqual(70);
    expect(RULEBOOK_STATS.phrases).toBeGreaterThanOrEqual(150);
    expect(RULEBOOK_STATS.uniquePhrases).toBeGreaterThanOrEqual(150);
  });

  it("has the expected families", () => {
    for (const rule of RULEBOOK) {
      expect(rule.id).toMatch(/^[a-z]+(\.[a-z0-9_]+)+$/);
    }
    const categories = new Set(RULEBOOK.map((rule) => rule.category));
    for (const category of [
      "person",
      "object",
      "pair",
      "memory",
      "dwell",
      "idle",
      "self",
      "milestone",
      "clock",
      "mode",
      "pose",
      "vision",
    ]) {
      expect(categories.has(category as Rule["category"]), category).toBe(true);
    }
  });
});

describe("F-E4 #11 — no comments on bodies, age, gender or looks", () => {
  const forbidden = [
    "uomo",
    "donna",
    "ragazzo",
    "ragazza",
    "signore",
    "signora",
    "giovane",
    "anziano",
    "anziana",
    "vecchio",
    "vecchia",
    "capelli",
    "occhi",
    "sorriso",
    "aspetto",
    "barba",
    "trucco",
    "maglia",
    "maglietta",
    "vestito",
    "pantaloni",
    "gonna",
    "età",
  ];

  it("never uses appearance words", () => {
    for (const phrase of allPhrases()) {
      const lower = phrase.toLowerCase();
      for (const word of forbidden) {
        expect(lower.includes(word), `"${word}" in "${phrase}"`).toBe(false);
      }
    }
  });

  it("talks only about objects and actions in object rules", () => {
    for (const rule of RULEBOOK.filter((entry) => entry.category === "object")) {
      for (const phrase of rule.phrases) {
        expect(phrase.toLowerCase()).not.toMatch(/\btu\b|\bvoi\b/);
      }
    }
  });
});

describe("F-E4 #12 — sharp objects are gentle and rare", () => {
  for (const label of ["coltello", "forbici"]) {
    it(`keeps ${label} quiet and calm`, () => {
      const rule = RULEBOOK_STATS.ruleForClass(label);
      expect(rule).toBeDefined();
      expect(rule?.priority).toBeLessThanOrEqual(35);
      expect(rule?.cooldown_s).toBeGreaterThanOrEqual(60);
      const text = (rule?.phrases ?? []).join(" ").toLowerCase();
      expect(text).toMatch(/calma|piano/);
      expect(text).not.toMatch(/pericolo|attento|fermati|allarme/);
    });
  }
});

describe("F-E4 #5/#24 — approximate day totals and self facts", () => {
  it("speaks about people in the day with 'circa'", () => {
    const rule = RULEBOOK.find((entry) => entry.id === "memory.people_total");
    expect(rule).toBeDefined();
    expect(rule?.phrases.join(" ")).toMatch(/circa/i);
    expect(rule?.max_per_day).toBeDefined();
  });

  it("caps self facts to once a minute and out of crowd mode", () => {
    for (const rule of RULEBOOK.filter((entry) => entry.category === "self")) {
      expect(rule.cooldown_s).toBeGreaterThanOrEqual(60);
      expect(rule.max_per_day).toBeDefined();
      expect(rule.when({
        type: "self",
        field: "fps",
        value: 60,
        now: 0,
      }, {
        working: { focus: { mode: "yolox", people: 5, crowd: true } },
      } as never)).toBe(false);
    }
  });
});
