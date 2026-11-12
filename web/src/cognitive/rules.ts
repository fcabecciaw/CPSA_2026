/**
 * Production-rule engine of the cognitive layer (F-E3 `rules.ts`).
 *
 * One utterance at a time, at most one queued (priority ≥ 80), re-validated
 * just before it is spoken and dropped after 3 s; a global gap between
 * utterances (4 s, 8 s in crowd mode); per-rule and per-object cooldowns;
 * ACT-R-like novelty and habituation in the utility; variants never repeat the
 * previous line of the same rule (F-E4 #6, #13, #14). The engine is pure: the
 * caller gives it the clock and a random source.
 */

import type { CognitiveEvent } from "./events";
import { objectKeyOf } from "./events";
import { nounPhrase, renderPhrase, type PhraseSlots } from "./grammar";
import type { CognitiveMemory } from "./memory";
import type { WorldModelName } from "./events";

/** The seven faces the hologram can wear (F-E3). */
export const EXPRESSIONS = [
  "neutral",
  "happy",
  "surprised",
  "curious",
  "sleepy",
  "thinking",
  "worried",
] as const;
export type Expression = (typeof EXPRESSIONS)[number];

/** Where the hologram looks while it talks. */
export type GazeTarget = "person" | "object" | "wander";

export type RuleCategory =
  | "person"
  | "object"
  | "pair"
  | "memory"
  | "dwell"
  | "idle"
  | "self"
  | "milestone"
  | "clock"
  | "mode"
  | "pose"
  | "vision"
  | "special";

export interface Rule {
  id: string;
  /** COCO classes the rule speaks about, for the per-class coverage test. */
  classes?: readonly string[];
  category: RuleCategory;
  when: (event: CognitiveEvent, memory: CognitiveMemory) => boolean;
  priority: number;
  cooldown_s: number;
  per_object_cooldown_s?: number;
  /** Milestones and facts stop after this many times today. */
  max_per_day?: number;
  expression: Expression;
  gaze?: GazeTarget;
  phrases: readonly string[];
  /** Only while this model is active (F-E4 #22). */
  model?: WorldModelName;
}

export interface Utterance {
  ruleId: string;
  text: string;
  expression: Expression;
  gaze: GazeTarget;
  event: CognitiveEvent;
  objectKey: string | null;
  priority: number;
  /** Utility that won the conflict resolution, for the "mente" strip. */
  utility: number;
  /** Engine clock when the utterance was chosen. */
  now: number;
}

interface Candidate {
  rule: Rule;
  event: CognitiveEvent;
  objectKey: string | null;
  utility: number;
}

interface Queued {
  candidate: Candidate;
  at: number;
}

export interface EngineOptions {
  /** [0, 1) random source, injected so tests are deterministic. */
  rng?: () => number;
  /** Global gap between utterances (F-E4 #13). */
  gapMs?: number;
  crowdGapMs?: number;
  queueExpiryMs?: number;
  pendingExpiryMs?: number;
  /** Minimum priority that may wait in the single queue slot. */
  queuePriority?: number;
  /** Utility noise amplitude. */
  noise?: number;
}

const DEFAULTS: Required<EngineOptions> = {
  rng: Math.random,
  gapMs: 4_000,
  crowdGapMs: 8_000,
  queueExpiryMs: 3_000,
  pendingExpiryMs: 1_000,
  queuePriority: 80,
  noise: 1,
};

function it(value: number, digits = 0): string {
  return new Intl.NumberFormat("it-IT", {
    maximumFractionDigits: digits,
  }).format(value);
}

function two(value: number): string {
  return String(Math.trunc(value)).padStart(2, "0");
}

/** Slots of a phrase, filled from the event and the day (F-E3). */
export function slotsFor(event: CognitiveEvent): PhraseSlots {
  switch (event.type) {
    case "appear":
    case "first_seen":
    case "return":
      return {
        oggetto: nounPhrase(event.label, event.count),
        n: it(event.count),
      };
    case "disappear":
      return { oggetto: nounPhrase(event.label, 1) };
    case "dwell":
      return {
        oggetto: nounPhrase(event.label, event.count),
        n: it(event.count),
        secondi: it(event.seconds),
        minuti: it(Math.round(event.seconds / 60)),
      };
    case "count":
      return {
        oggetto: nounPhrase("persona", event.to),
        n: it(event.to),
        n2: it(event.from),
      };
    case "together":
      return {
        oggetto: nounPhrase(event.a, event.countA),
        oggetto2: nounPhrase(event.b, event.countB),
        altro: nounPhrase(event.b, event.countB),
        n: it(event.countA),
      };
    case "idle":
      return {
        secondi: it(event.seconds),
        minuti: it(Math.max(1, Math.round(event.seconds / 60))),
      };
    case "milestone":
      return { n: it(event.value) };
    case "pose":
      return { n: it(Math.round(event.score * 100)) };
    case "mode":
      return {};
    case "clock":
      return { ora: two(event.hour), n: it(event.hour) };
    case "self":
      return {
        fps: it(event.value, 1),
        dpu: it(event.value, 1),
        minuti: it(event.value),
        n: it(event.value),
      };
    case "vision":
      return {};
  }
}

/** Render one phrase for a rule and event, honouring the no-repeat variant. */
export function renderRulePhrase(
  rule: Rule,
  event: CognitiveEvent,
  memory: CognitiveMemory,
): string {
  const previous = memory.lastPhrase(rule.id);
  const slots: PhraseSlots = {
    ...slotsFor(event),
    totale: it(memory.day.totals.peopleSeen),
  };
  const rendered = rule.phrases.map((phrase) => renderPhrase(phrase, slots));
  const fresh = rendered.filter((text) => text !== previous);
  const pool = fresh.length > 0 ? fresh : rendered;
  const index = Math.min(pool.length - 1, Math.floor(memory.rule(rule.id).count % pool.length));
  return pool[index];
}

export class CognitiveEngine {
  readonly rules: readonly Rule[];
  private readonly options: Required<EngineOptions>;
  private pending: (Candidate & { seenAt: number }) | null = null;
  private queue: Queued | null = null;
  private inFlight: Utterance | null = null;
  private lastSpokeAt = Number.NEGATIVE_INFINITY;

  constructor(rules: readonly Rule[], options: EngineOptions = {}) {
    this.rules = rules;
    this.options = { ...DEFAULTS, ...options };
  }

  get speaking(): boolean {
    return this.inFlight !== null;
  }

  get queueLength(): number {
    return this.queue === null ? 0 : 1;
  }

  /** The rule that is being spoken, for the "mente" strip. */
  get current(): Utterance | null {
    return this.inFlight;
  }

  /** Collect the matching rules for a batch of events; keep the best one. */
  consider(events: readonly CognitiveEvent[], memory: CognitiveMemory, now: number): void {
    const crowd = memory.working.focus.crowd;
    for (const event of events) {
      for (const rule of this.rules) {
        if (rule.model !== undefined && rule.model !== memory.working.focus.mode) {
          continue;
        }
        if (crowd && rule.category === "object") {
          continue;
        }
        if (!this.canFire(rule, event, memory, now)) {
          continue;
        }
        let matched = false;
        try {
          matched = rule.when(event, memory);
        } catch {
          matched = false;
        }
        if (!matched) {
          continue;
        }
        const key = objectKeyOf(event);
        const label = key !== null && !key.includes("+") ? key : null;
        const novelty = label === null ? 2 : memory.novelty(label, now);
        const habituation = label === null ? 0 : memory.habituation(label, now);
        const utility =
          rule.priority +
          novelty -
          habituation +
          this.options.rng() * this.options.noise;
        const candidate: Candidate = { rule, event, objectKey: key, utility };

        if (this.speaking || now - this.lastSpokeAt < this.gap(memory)) {
          this.enqueue(candidate, now);
        } else if (this.pending === null || candidate.utility > this.pending.utility) {
          this.pending = { ...candidate, seenAt: now };
        }
      }
    }
  }

  /** Choose the next utterance, if the gap and the queue allow one. */
  tick(now: number, memory: CognitiveMemory): Utterance | null {
    if (this.speaking) {
      return null;
    }
    if (now - this.lastSpokeAt < this.gap(memory)) {
      return null;
    }

    while (this.queue !== null) {
      const queued = this.queue;
      this.queue = null;
      if (now - queued.at > this.options.queueExpiryMs) {
        continue;
      }
      if (!this.stillValid(queued.candidate, memory, now)) {
        continue;
      }
      return this.speak(queued.candidate, memory, now);
    }

    const pending = this.pending;
    this.pending = null;
    const expiry = Math.max(this.options.pendingExpiryMs, this.gap(memory));
    if (pending === null || now - pending.seenAt > expiry) {
      return null;
    }
    if (!this.stillValid(pending, memory, now)) {
      return null;
    }
    return this.speak(pending, memory, now);
  }

  /** Called when the voice finished (or aborted) the utterance. */
  endSpeech(): void {
    this.inFlight = null;
  }

  /** Forget pending and queued speech, e.g. when the vision is lost. */
  clear(): void {
    this.pending = null;
    this.queue = null;
  }

  /**
   * Stop the current utterance immediately (vision lost, mode switch): the
   * face is reset, the engine must not keep it in flight forever.
   */
  interrupt(): void {
    this.inFlight = null;
    this.clear();
  }

  private gap(memory: CognitiveMemory): number {
    return memory.working.focus.crowd ? this.options.crowdGapMs : this.options.gapMs;
  }

  private canFire(
    rule: Rule,
    event: CognitiveEvent,
    memory: CognitiveMemory,
    now: number,
  ): boolean {
    const firedAt = memory.lastFiredAt(rule.id);
    if (firedAt !== null && now - firedAt < rule.cooldown_s * 1000) {
      return false;
    }
    if (rule.max_per_day !== undefined && memory.fireCount(rule.id) >= rule.max_per_day) {
      return false;
    }
    if (rule.per_object_cooldown_s !== undefined) {
      const key = objectKeyOf(event);
      if (key !== null) {
        const last = memory.perObjectAt(rule.id, key);
        if (last !== null && now - last < rule.per_object_cooldown_s * 1000) {
          return false;
        }
      }
    }
    return true;
  }

  private stillValid(candidate: Candidate, memory: CognitiveMemory, now: number): boolean {
    if (candidate.rule.model !== undefined && candidate.rule.model !== memory.working.focus.mode) {
      return false;
    }
    if (memory.working.focus.crowd && candidate.rule.category === "object") {
      return false;
    }
    const present = memory.working.focus.present;
    if (
      present !== undefined &&
      candidate.objectKey !== null &&
      !candidate.objectKey.includes("+") &&
      (candidate.rule.category === "object" ||
        candidate.rule.category === "memory" ||
        candidate.rule.category === "dwell") &&
      !present.includes(candidate.objectKey)
    ) {
      return false;
    }
    if (!this.canFire(candidate.rule, candidate.event, memory, now)) {
      return false;
    }
    try {
      return candidate.rule.when(candidate.event, memory);
    } catch {
      return false;
    }
  }

  private enqueue(candidate: Candidate, now: number): void {
    if (candidate.rule.priority < this.options.queuePriority) {
      return;
    }
    if (this.queue === null || candidate.utility > this.queue.candidate.utility) {
      this.queue = { candidate, at: now };
    }
  }

  private speak(candidate: Candidate, memory: CognitiveMemory, now: number): Utterance {
    const text = renderRulePhrase(candidate.rule, candidate.event, memory);
    const utterance: Utterance = {
      ruleId: candidate.rule.id,
      text,
      expression: candidate.rule.expression,
      gaze: candidate.rule.gaze ?? "object",
      event: candidate.event,
      objectKey: candidate.objectKey,
      priority: candidate.rule.priority,
      utility: candidate.utility,
      now,
    };
    this.inFlight = utterance;
    this.lastSpokeAt = now;
    memory.noteRuleFired(candidate.rule.id, text, candidate.objectKey, now);
    return utterance;
  }
}
