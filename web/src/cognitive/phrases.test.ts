/**
 * The words ARI says (P1 rewrite): no claim to recognise a person, no gendered
 * address and no gendered self-description, and every object line correct
 * Italian once its slot is filled, for singular, plural, feminine, elided and
 * plural-only classes.
 */

import { describe, expect, it } from "vitest";

import { FACTS } from "./content/facts";
import { QUESTIONS } from "./content/questions";
import {
  COCO_IT_LABELS,
  indefinitePhrase,
  nounPhrase,
  renderPhrase,
  wordCount,
} from "./grammar";
import { RULEBOOK, SPECIFIC_CLASSES, classLines } from "./rulebook";
import { SCENE_PHRASES } from "./scene";

const OBJECT_LABELS = COCO_IT_LABELS.filter((label) => label !== "persona");

function rulePhrases(filter: (id: string, category: string) => boolean): string[] {
  return RULEBOOK.filter((rule) => filter(rule.id, rule.category)).flatMap((rule) => [
    ...rule.phrases,
  ]);
}

/** Lines said to the people in front of the camera. */
const PERSON_DIRECTED = [
  ...Object.entries(SCENE_PHRASES)
    .filter(([family]) => family !== "FOLLOW_UPS" && family !== "CONTINUITY")
    .flatMap(([, phrases]) => [...phrases]),
  ...rulePhrases((id, category) =>
    ["person", "idle", "pose", "milestone", "clock", "mode"].includes(category) ||
    id.startsWith("person.") ||
    id === "pair.persona_cellulare",
  ),
  ...QUESTIONS.flatMap((question) => [question.text, ...question.yes, ...question.no]),
];

/** Everything ARI can say, self-descriptions included. */
const EVERYTHING = [
  ...Object.values(SCENE_PHRASES).flatMap((phrases) => [...phrases]),
  ...RULEBOOK.flatMap((rule) => [...rule.phrases]),
  ...QUESTIONS.flatMap((question) => [question.text, ...question.yes, ...question.no]),
  ...FACTS.map((fact) => fact.text),
];

describe("no person recognition (the system has no identity)", () => {
  it("never says a person is back or remembered", () => {
    const forbidden =
      /\b(eccoti|rieccoti|bentornat\w*|tornat[oaie]|di nuovo qui|ti ricordi|ti riconosco|ti ricordo|ancora tu|ti aspettavo)\b|ciao di nuovo/i;
    for (const phrase of PERSON_DIRECTED) {
      expect(phrase, phrase).not.toMatch(forbidden);
    }
  });

  it("greets without 'again' in every greeting family", () => {
    for (const family of [
      "GREETINGS_SINGLE",
      "GREETINGS_RETURN",
      "GREETINGS_PAIR",
      "GREETINGS_GROUP",
      "GREETINGS_CROWD",
    ] as const) {
      for (const phrase of SCENE_PHRASES[family]) {
        expect(phrase, phrase).not.toMatch(/di nuovo|ancora|un'altra volta/i);
      }
    }
  });
});

describe("gender-neutral address and self-description", () => {
  const gendered = [
    "benvenuto", "benvenuta",
    "tornato", "tornata",
    "sorpreso", "sorpresi", "stupito", "stupita",
    "pronto", "pronta", "pronti", "pronte",
    "contento", "contenta", "contenti",
    "arrivato", "arrivata", "arrivati",
    "sveglio",
    "simpatico", "simpatica",
    "bravo", "brava",
    "sicuro", "sicura",
    "seduto", "seduta",
    "nascosto", "nascosta",
    "abbassato", "abbassata",
    "accovacciato", "accovacciata",
    "annoiato", "annoiata",
    "emozionato", "emozionata",
    "curioso", "curiosa",
    "attento", "attenta",
    "acceso", "accesa",
    "amico", "amica", "amici", "amiche",
    "velocissimo",
  ];

  // The user gave ARI a feminine voice in this one line on purpose (2026-11-12); visitors are still always
  // addressed in neutral Italian, and every other line stays neutral.
  const userApproved = new Set(["Fermati un attimo, sono simpatica eh!"]);

  it("has no gendered adjective or participle in anything ARI says", () => {
    for (const phrase of EVERYTHING.filter((line) => !userApproved.has(line))) {
      const words = phrase.toLowerCase().split(/[^a-zà-ù']+/);
      for (const word of gendered) {
        expect(words.includes(word), `"${word}" in "${phrase}"`).toBe(false);
      }
      // "stanco/a" as an adjective; "non mi stanco" is the verb and stays.
      expect(phrase, phrase).not.toMatch(/(?<!\b(mi|ti|si|ci|vi) )\bstanc[oa]\b/i);
    }
  });

  it("keeps 'Benvenuti' for groups only", () => {
    const single = [
      ...SCENE_PHRASES.GREETINGS_SINGLE,
      ...SCENE_PHRASES.GREETINGS_RETURN,
      ...SCENE_PHRASES.WAVE_FIRST,
      ...SCENE_PHRASES.WAVE_AGAIN,
      ...SCENE_PHRASES.INTRO,
      ...SCENE_PHRASES.PROMPTS_YOLOX,
      ...SCENE_PHRASES.PROMPTS_MOVENET,
      ...SCENE_PHRASES.INVITATIONS,
      ...rulePhrases((id) =>
        ["person.single", "person.return", "person.stay_25", "person.stay_120", "idle.wake"].includes(id),
      ),
    ];
    for (const phrase of single) {
      expect(phrase, phrase).not.toMatch(/benvenut/i);
    }
  });
});

describe("scene families: size and length", () => {
  // Sizes follow the user's review of the list (2026-11-12): fewer pair/group lines and a 2-line intro.
  it("has varied greetings, an intro and 3-6 attract nudges", () => {
    for (const family of [
      "GREETINGS_SINGLE",
      "GREETINGS_PAIR",
      "GREETINGS_GROUP",
      "GREETINGS_CROWD",
    ] as const) {
      expect(SCENE_PHRASES[family].length, family).toBeGreaterThanOrEqual(2);
    }
    expect(SCENE_PHRASES.INTRO.length).toBeGreaterThanOrEqual(2);
    for (const intro of SCENE_PHRASES.INTRO) {
      expect(intro).toMatch(/ARI/);
      expect(intro).toMatch(/oggetti/);
      expect(intro).toMatch(/sorrisi/);
    }
    expect(SCENE_PHRASES.INVITATIONS.length).toBeGreaterThanOrEqual(3);
    expect(SCENE_PHRASES.INVITATIONS.length).toBeLessThanOrEqual(6);
  });

  it("keeps every spoken scene and question line short", () => {
    const lines = [
      ...Object.values(SCENE_PHRASES).flatMap((phrases) => [...phrases]),
      ...QUESTIONS.flatMap((question) => [question.text, ...question.yes, ...question.no]),
    ];
    for (const line of lines) {
      expect(wordCount(line), line).toBeLessThanOrEqual(12);
    }
  });
});

describe("object lines render as correct Italian", () => {
  const render = (phrase: string, label: string, count: number): string =>
    renderPhrase(phrase, {
      oggetto: nounPhrase(label, count),
      unoggetto: indefinitePhrase(label, count),
    });

  it("gives 2-4 specific lines to the classes likely at an indoor stand", () => {
    const stand = [
      "cellulare", "tazza", "bottiglia", "calice", "libro", "portatile", "tastiera",
      "mouse", "telecomando", "forbici", "orsacchiotto", "banana", "mela", "arancia",
      "zaino", "borsa", "ombrello", "orologio", "sedia", "tv", "vaso", "pianta",
      "cravatta", "valigia", "spazzolino", "coltello", "forchetta", "cucchiaio",
      "panino", "pizza", "torta", "ciambella", "palla", "frisbee",
    ];
    for (const label of stand) {
      expect(SPECIFIC_CLASSES, label).toContain(label);
      const lines = classLines(label as (typeof OBJECT_LABELS)[number]);
      expect(lines.length, label).toBeGreaterThanOrEqual(2);
      expect(lines.length, label).toBeLessThanOrEqual(4);
    }
  });

  it("never puts the slot where article, gender or number would matter", () => {
    for (const label of OBJECT_LABELS) {
      for (const phrase of classLines(label)) {
        expect(phrase, phrase).toMatch(/\{(un)?oggetto\}/);
        expect(phrase, phrase).not.toMatch(/^\{/);
        expect(phrase, phrase).not.toMatch(
          /\{(un)?oggetto\} (è|sono|sta|stanno|resta|si|ha|non)\b/,
        );
        expect(phrase.toLowerCase(), phrase).not.toMatch(/prima volta|prima vista|novità di oggi/);
        expect(phrase.toLowerCase(), phrase).not.toMatch(/\btu\b|\bvoi\b/);
      }
    }
  });

  it("renders every class line for one and for two objects", () => {
    for (const label of OBJECT_LABELS) {
      for (const phrase of classLines(label)) {
        for (const count of [1, 2]) {
          const text = render(phrase, label, count);
          expect(text, text).not.toMatch(/[{}]/);
          expect(text, text).toMatch(/^[A-ZÈ]/);
          expect(text, text).toMatch(/[.!?…]$/);
          expect(text, text).not.toMatch(/ {2}|\b(l|un|dell|all|dall|nell|sull)'\s/i);
          // Never an unmerged preposition and article ("di il", "su la").
          expect(text, text).not.toMatch(/\b(di|a|da|in|su) (il|lo|la|i|gli|le|l')\b/i);
          expect(wordCount(text), text).toBeLessThanOrEqual(12);
        }
      }
    }
  });

  it("uses the right article for feminine, elided, 'lo' and plural-only nouns", () => {
    const rendered = (label: (typeof OBJECT_LABELS)[number], count = 1) =>
      classLines(label).map((phrase) => render(phrase, label, count));
    expect(rendered("arancia")).toContain("Vedo l'arancia. Vitamina C in arrivo!");
    expect(rendered("zaino")).toContain("Vedo lo zaino. Si va a lezione?");
    expect(rendered("forbici")).toContain("Vedo le forbici. Piano piano, mi raccomando.");
    expect(rendered("tazza", 2)).toContain("Vedo due tazze. Caffè o tè?");
    expect(rendered("ombrello")).toContain("Vedo l'ombrello. Piove a Cagliari? Che rarità!");
    // Generic classes take the indefinite article.
    expect(rendered("auto")).toContain("Se non sbaglio, vedo un'auto.");
    expect(rendered("skateboard")).toContain("Se non sbaglio, vedo uno skateboard.");
    expect(rendered("panchina", 3)).toContain("Mi sembra di vedere tre panchine. Dico bene?");
    expect(rendered("aereo")).toContain("Se non sbaglio, vedo un aereo.");
    expect(indefinitePhrase("forbici")).toBe("un paio di forbici");
  });

  it("renders the doubtful, return and dwell families for every class", () => {
    const families = RULEBOOK.filter((rule) =>
      ["class.doubtful", "memory.object_return", "dwell.object_30", "scene.follow_up"].includes(rule.id),
    );
    expect(families).toHaveLength(3);
    for (const rule of families) {
      for (const phrase of rule.phrases) {
        expect(phrase, phrase).not.toMatch(/^\{/);
        expect(phrase, phrase).not.toMatch(/\{(un)?oggetto\} (è|sono|sta|stanno|resta|si|ha|non)\b/);
        for (const label of OBJECT_LABELS) {
          for (const count of [1, 2]) {
            const text = render(phrase, label, count);
            expect(text, text).not.toMatch(/[{}]|\b(l|un)'\s/i);
          }
        }
      }
    }
    for (const phrase of SCENE_PHRASES.FOLLOW_UPS) {
      expect(phrase, phrase).not.toMatch(/^\{|\{oggetto\} (è|sono|sta|si|ha)\b/);
    }
  });
});
