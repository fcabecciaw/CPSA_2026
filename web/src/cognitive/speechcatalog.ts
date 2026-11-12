/**
 * Every sentence ARI can say that is known ahead of time, rendered with the
 * real grammar (`grammar.ts`, `rules.ts`), so the local TTS can synthesize and
 * cache all of them before the demo (T1: "everything cached").
 *
 * Fixed lines come verbatim from the scenes, the questions, the facts and the
 * rulebook. Slot lines are expanded the way the engine fills them at run time:
 *
 * - object lines (class rule, doubtful, return, dwell, follow-up) for every
 *   COCO class, singular and with 2–3 instances;
 * - counting lines (`memory.object_count`) for counts 2..8 of the indoor-stand
 *   classes (the ones with their own lines);
 * - people-count greetings for 3..8 people, in both renderings (scene: "tre",
 *   rule: "3");
 * - the clock greetings for every hour their rule accepts.
 *
 * Lines whose slot is an open number (people seen today, fps, DPU latency,
 * uptime) are left to live synthesis. The entries are in pregenerate priority
 * order: `scene`, `indoor`, `counts`, `rest`. `web/scripts/export-speech.mjs`
 * writes them to JSON for `cpsa-tts pregenerate --catalog`.
 */

import { FACTS } from "./content/facts";
import { QUESTIONS } from "./content/questions";
import type { CognitiveEvent } from "./events";
import {
  COCO_IT_LABELS,
  numberWord,
  objectCountSlots,
  renderPhrase,
  type PhraseSlots,
} from "./grammar";
import { RULEBOOK, SPECIFIC_CLASSES } from "./rulebook";
import { slotsFor, type Rule } from "./rules";
import { SCENE_PHRASES } from "./scene";

export type SpeechGroup = "scene" | "indoor" | "counts" | "rest";

export interface SpeechLine {
  text: string;
  group: SpeechGroup;
  /** Where the line comes from: a rule id, `scene.<FAMILY>`, `fact.<id>`… */
  source: string;
}

export const SPEECH_GROUPS: readonly SpeechGroup[] = ["scene", "indoor", "counts", "rest"];

/** Instances rendered for the object lines ("il libro", "due libri", "tre libri"). */
export const OBJECT_COUNTS = [1, 2, 3] as const;
/** Day counts rendered for the counting lines ("Secondo libro" … "otto libri"). */
export const DAY_COUNTS = [2, 3, 4, 5, 6, 7, 8] as const;
/** People rendered for the group greetings. */
export const PEOPLE_COUNTS = [3, 4, 5, 6, 7, 8] as const;

/** Object-family rules that render `{oggetto}`/`{unoggetto}` for any class. */
const OBJECT_FAMILIES = ["class.doubtful", "memory.object_return", "dwell.object_30"] as const;
const SLOT_RE = /\{\w+\}/;

const OBJECT_LABELS: readonly string[] = COCO_IT_LABELS.filter((label) => label !== "persona");
const INDOOR = new Set(SPECIFIC_CLASSES);

function event(type: "appear" | "return" | "dwell", label: string, count: number): CognitiveEvent {
  const now = 0;
  if (type === "appear") {
    return { type, label, count, score: 0.9, area: 0.02, scenery: false, doubtful: false, now };
  }
  if (type === "return") {
    return { type, label, count, absenceS: 60, now };
  }
  return { type, label, count, seconds: 30, now };
}

function eventTypeOf(ruleId: string): "appear" | "return" | "dwell" {
  if (ruleId === "memory.object_return") {
    return "return";
  }
  if (ruleId === "dwell.object_30") {
    return "dwell";
  }
  return "appear";
}

function render(rule: Rule, slots: PhraseSlots): string[] {
  return rule.phrases.map((phrase) => renderPhrase(phrase, slots));
}

function ruleById(id: string): Rule {
  const rule = RULEBOOK.find((entry) => entry.id === id);
  if (rule === undefined) {
    throw new Error(`speech catalog: missing rule ${id}`);
  }
  return rule;
}

/** The object lines of one class: its rule and the shared object families. */
function objectLines(label: string): Array<[string, string]> {
  const lines: Array<[string, string]> = [];
  const own = RULEBOOK.find(
    (rule) => rule.id.startsWith("object.") && rule.classes?.includes(label),
  );
  const rules = [...(own === undefined ? [] : [own]), ...OBJECT_FAMILIES.map(ruleById)];
  for (const count of OBJECT_COUNTS) {
    for (const rule of rules) {
      const slots = slotsFor(event(eventTypeOf(rule.id), label, count));
      for (const text of render(rule, slots)) {
        lines.push([text, rule.id]);
      }
    }
    // The scene follow-up names the object the same way (`nounPhrase`).
    const slots = slotsFor(event("appear", label, count));
    for (const template of SCENE_PHRASES.FOLLOW_UPS) {
      lines.push([renderPhrase(template, slots), "scene.follow_up"]);
    }
  }
  return lines;
}

/** `memory.object_count` for one class and the day counts. */
function countLines(label: string): Array<[string, string]> {
  const rule = ruleById("memory.object_count");
  return DAY_COUNTS.flatMap((times) =>
    render(rule, { ...slotsFor(event("appear", label, 1)), ...objectCountSlots(label, times) }).map(
      (text): [string, string] => [text, rule.id],
    ),
  );
}

/** Every fixed line: scenes, questions, facts and the slot-free rulebook phrases. */
function sceneLines(): Array<[string, string]> {
  const lines: Array<[string, string]> = [];
  for (const [family, phrases] of Object.entries(SCENE_PHRASES)) {
    for (const phrase of phrases) {
      if (!SLOT_RE.test(phrase)) {
        lines.push([phrase, `scene.${family}`]);
      }
    }
  }
  // Group greetings: the scene says "tre", the rule says "3" (`it()`).
  for (const people of PEOPLE_COUNTS) {
    for (const template of SCENE_PHRASES.GREETINGS_GROUP) {
      lines.push([template.replaceAll("{n}", numberWord(people)), "scene.greeting.group"]);
    }
    const rule = ruleById("person.small_group");
    const slots = slotsFor({ type: "count", from: 0, to: people, quiet: false, returnAfterS: null, now: 0 });
    for (const text of render(rule, slots)) {
      lines.push([text, rule.id]);
    }
  }
  for (const question of QUESTIONS) {
    lines.push([question.text, `question.${question.id}`]);
    for (const answer of [...question.yes, ...question.no]) {
      lines.push([answer, `question.${question.id}`]);
    }
  }
  for (const fact of FACTS) {
    lines.push([fact.text, `fact.${fact.id}`]);
  }
  for (const rule of RULEBOOK) {
    if (rule.id.startsWith("object.")) {
      continue;
    }
    for (const phrase of rule.phrases) {
      if (!SLOT_RE.test(phrase)) {
        lines.push([phrase, rule.id]);
      }
    }
    if (rule.category === "clock") {
      for (let hour = 0; hour < 24; hour += 1) {
        const clock: CognitiveEvent = { type: "clock", hour, now: 0 };
        let matches = false;
        try {
          matches = rule.when(clock, undefined as never);
        } catch {
          matches = false;
        }
        if (matches) {
          for (const text of render(rule, slotsFor(clock))) {
            lines.push([text, rule.id]);
          }
        }
      }
    }
  }
  return lines;
}

/** Every pre-renderable sentence, unique, in pregenerate priority order. */
export function speechCatalog(): SpeechLine[] {
  const groups: Array<[SpeechGroup, Array<[string, string]>]> = [
    ["scene", sceneLines()],
    ["indoor", OBJECT_LABELS.filter((label) => INDOOR.has(label)).flatMap(objectLines)],
    ["counts", OBJECT_LABELS.filter((label) => INDOOR.has(label)).flatMap(countLines)],
    [
      "rest",
      [
        // Classes without their own lines: still named when they show up.
        ...OBJECT_LABELS.filter((label) => !INDOOR.has(label)).flatMap(objectLines),
        // Slot-free rule phrases of the class rules not covered above.
        ...RULEBOOK.filter((rule) => rule.id.startsWith("object."))
          .flatMap((rule) => rule.phrases)
          .filter((phrase) => !SLOT_RE.test(phrase))
          .map((phrase): [string, string] => [phrase, "object"]),
      ],
    ],
  ];
  const seen = new Set<string>();
  const catalog: SpeechLine[] = [];
  for (const [group, lines] of groups) {
    for (const [text, source] of lines) {
      if (text === "" || SLOT_RE.test(text) || seen.has(text)) {
        continue;
      }
      seen.add(text);
      catalog.push({ text, group, source });
    }
  }
  return catalog;
}
