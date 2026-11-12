/**
 * Working memory and day memory of the cognitive layer (F-E3 `memory.ts`).
 *
 * Working memory keeps the last ten events, the current world focus and the
 * last utterance; day memory keeps one record per COCO class and the day's
 * approximate totals, persisted in `localStorage` under the calendar day it
 * belongs to (F-E4 #23). There is no identity record anywhere: a person is only
 * ever a count (F-E4 #5).
 */

import type { CognitiveEvent, WorldModelName } from "./events";

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface DayClassMemory {
  /** First observation of the class today, dashboard clock. */
  firstSeenAt: number;
  lastSeenAt: number;
  /** Number of separate appearances today. */
  times: number;
  /** Longest dwell seen today, seconds. */
  dwellS: number;
  comments: number;
  /** Timestamps of the comments, oldest first (capped), for habituation. */
  commentTimes: number[];
}

export interface DayRuleMemory {
  count: number;
  lastFiredAt: number | null;
  lastPhrase: string | null;
  /** Last fire per object key, for `per_object_cooldown_s`. */
  perObject: Record<string, number>;
}

export interface DayTotals {
  /** Sum of the positive stable-count steps: approximate by design (#5). */
  peopleSeen: number;
  maxPeople: number;
  objectsSeen: number;
  rulesFired: number;
}

/** Idle content already spoken today, so Attract never repeats itself (#14). */
export interface DayIdleMemory {
  facts: string[];
  questions: string[];
}

export interface DayMemory {
  /** Local calendar day, `YYYY-MM-DD`. */
  date: string;
  classes: Record<string, DayClassMemory>;
  rules: Record<string, DayRuleMemory>;
  totals: DayTotals;
  idle: DayIdleMemory;
}

/** What the rules see of the world right now, refreshed every tick. */
export interface WorldFocus {
  mode: WorldModelName;
  /** Stable number of people (F-E4 #1/#3). */
  people: number;
  crowd: boolean;
  /** Labels present in the world model, when the agent knows them. */
  present?: readonly string[];
}

export interface UtteranceRecord {
  ruleId: string;
  text: string;
  at: number;
}

export interface WorkingMemory {
  /** Last ten events, oldest first. */
  events: CognitiveEvent[];
  focus: WorldFocus;
  lastUtterance: UtteranceRecord | null;
  lastActivityAt: number;
}

export interface MemoryOptions {
  storage?: StorageLike | null;
  storageKey?: string;
}

export const STORAGE_KEY = "cpsa.cognitive.day.v1";
const MAX_EVENTS = 10;
const MAX_COMMENT_TIMES = 20;
/** Habituation half-life: a class commented a while ago matters less. */
const HABITUATION_HALF_LIFE_MS = 180_000;

/** Local `YYYY-MM-DD`, so midnight is the user's midnight (F-E4 #23). */
export function dayKey(now: number): string {
  const date = new Date(now);
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}

function emptyClass(now: number): DayClassMemory {
  return {
    firstSeenAt: now,
    lastSeenAt: now,
    times: 0,
    dwellS: 0,
    comments: 0,
    commentTimes: [],
  };
}

export function emptyDay(date: string): DayMemory {
  return {
    date,
    classes: {},
    rules: {},
    totals: { peopleSeen: 0, maxPeople: 0, objectsSeen: 0, rulesFired: 0 },
    idle: { facts: [], questions: [] },
  };
}

function defaultStorage(): StorageLike | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

export class CognitiveMemory {
  readonly working: WorkingMemory = {
    events: [],
    focus: { mode: "yolox", people: 0, crowd: false },
    lastUtterance: null,
    lastActivityAt: 0,
  };

  private readonly storage: StorageLike | null;
  private readonly storageKey: string;
  private dayMemory: DayMemory;

  constructor(options: MemoryOptions = {}) {
    this.storage =
      options.storage === undefined ? defaultStorage() : options.storage;
    this.storageKey = options.storageKey ?? STORAGE_KEY;
    this.dayMemory = this.read() ?? emptyDay("");
  }

  get day(): DayMemory {
    return this.dayMemory;
  }

  /** Start a new day when midnight passed; returns true when it did. */
  rollover(now: number): boolean {
    const date = dayKey(now);
    if (this.dayMemory.date === date) {
      return false;
    }
    this.dayMemory = emptyDay(date);
    this.working.lastUtterance = null;
    this.save();
    return true;
  }

  /** Record events in working memory and in the day's counters. */
  observe(events: readonly CognitiveEvent[], now: number): void {
    for (const event of events) {
      this.working.events.push(event);
      if (this.working.events.length > MAX_EVENTS) {
        this.working.events.shift();
      }
      this.observeDay(event, now);
      this.working.lastActivityAt = now;
    }
    if (events.length > 0) {
      this.save();
    }
  }

  private observeDay(event: CognitiveEvent, now: number): void {
    const { totals } = this.dayMemory;
    switch (event.type) {
      case "appear": {
        const cls = this.class(event.label, now);
        // One stable appearance per `appear`: `perception.ts` already ignores
        // flicker (6 of 10 frames) and only re-emits after a real absence.
        cls.times += 1;
        cls.lastSeenAt = now;
        if (!event.scenery && event.label !== "persona") {
          totals.objectsSeen += event.count;
        }
        return;
      }
      case "return": {
        // Same appearance as the `appear` above, so it must not double count.
        this.class(event.label, now).lastSeenAt = now;
        return;
      }
      case "count": {
        // Approximate on purpose: positive stable steps, no identity (#5).
        totals.peopleSeen += Math.max(0, event.to - event.from);
        totals.maxPeople = Math.max(totals.maxPeople, event.to);
        const cls = this.class("persona", now);
        cls.lastSeenAt = now;
        if (event.to > 0 && cls.times === 0) {
          cls.times = 1;
        }
        return;
      }
      case "dwell": {
        const cls = this.class(event.label, now);
        cls.dwellS = Math.max(cls.dwellS, event.seconds);
        cls.lastSeenAt = now;
        return;
      }
      case "disappear": {
        this.class(event.label, now).lastSeenAt = now;
        return;
      }
      default:
        return;
    }
  }

  /** Day record of a class, created on first use. */
  class(label: string, now: number): DayClassMemory {
    const existing = this.dayMemory.classes[label];
    if (existing !== undefined) {
      return existing;
    }
    const created = emptyClass(now);
    this.dayMemory.classes[label] = created;
    return created;
  }

  rule(id: string): DayRuleMemory {
    const existing = this.dayMemory.rules[id];
    if (existing !== undefined) {
      return existing;
    }
    const created: DayRuleMemory = {
      count: 0,
      lastFiredAt: null,
      lastPhrase: null,
      perObject: {},
    };
    this.dayMemory.rules[id] = created;
    return created;
  }

  /** Remember one fired rule, its rendered phrase and the object it named. */
  noteRuleFired(
    ruleId: string,
    phrase: string,
    objectKey: string | null,
    now: number,
  ): void {
    const rule = this.rule(ruleId);
    rule.count += 1;
    rule.lastFiredAt = now;
    rule.lastPhrase = phrase;
    if (objectKey !== null) {
      rule.perObject[objectKey] = now;
      if (!objectKey.includes("+")) {
        const cls = this.class(objectKey, now);
        cls.comments += 1;
        cls.commentTimes.push(now);
        if (cls.commentTimes.length > MAX_COMMENT_TIMES) {
          cls.commentTimes.shift();
        }
      }
    }
    this.dayMemory.totals.rulesFired += 1;
    this.working.lastUtterance = { ruleId, text: phrase, at: now };
    this.save();
  }

  lastPhrase(ruleId: string): string | null {
    return this.dayMemory.rules[ruleId]?.lastPhrase ?? null;
  }

  fireCount(ruleId: string): number {
    return this.dayMemory.rules[ruleId]?.count ?? 0;
  }

  /** Was this idle fact/question already said today? (F-E4 #14) */
  idleUsed(kind: keyof DayIdleMemory, id: string): boolean {
    return this.dayMemory.idle[kind].includes(id);
  }

  /** Remember one idle line for the rest of the day. */
  markIdleUsed(kind: keyof DayIdleMemory, id: string): void {
    const used = this.dayMemory.idle[kind];
    if (!used.includes(id)) {
      used.push(id);
      this.save();
    }
  }

  lastFiredAt(ruleId: string): number | null {
    return this.dayMemory.rules[ruleId]?.lastFiredAt ?? null;
  }

  perObjectAt(ruleId: string, key: string): number | null {
    return this.dayMemory.rules[ruleId]?.perObject[key] ?? null;
  }

  /**
   * ACT-R-like habituation: how saturated a class already is. Grows with every
   * comment and decays with a three-minute half-life, so a repetitive class
   * loses utility but a fresh one still wins (F-E4 #14).
   */
  habituation(label: string, now: number): number {
    const cls = this.dayMemory.classes[label];
    if (cls === undefined) {
      return 0;
    }
    let total = 0;
    for (const at of cls.commentTimes) {
      const age = Math.max(0, now - at);
      total += Math.exp((-Math.LN2 * age) / HABITUATION_HALF_LIFE_MS);
    }
    return Math.min(6, total * 1.5);
  }

  /** Novelty bonus of a class: new today, or rarely seen (F-E4 #14). */
  novelty(label: string, now: number): number {
    const cls = this.dayMemory.classes[label];
    // Comments count as exposures too, so a class that keeps coming back and
    // being named loses its novelty even between detections.
    const exposures = cls === undefined ? 0 : cls.times + cls.comments;
    if (cls === undefined || exposures === 0) {
      return 6;
    }
    const age = Math.max(0, now - cls.firstSeenAt);
    const freshness = 2 * Math.exp(-age / 600_000);
    return Math.max(0, 4 - Math.log2(1 + exposures)) + freshness;
  }

  setFocus(focus: Partial<WorldFocus>): void {
    this.working.focus = { ...this.working.focus, ...focus };
  }

  save(): void {
    if (this.storage === null) {
      return;
    }
    try {
      this.storage.setItem(this.storageKey, JSON.stringify(this.dayMemory));
    } catch {
      // Private mode or quota: the session still works, it just forgets.
    }
  }

  private read(): DayMemory | null {
    if (this.storage === null) {
      return null;
    }
    try {
      const raw = this.storage.getItem(this.storageKey);
      if (raw === null) {
        return null;
      }
      const parsed = JSON.parse(raw) as Partial<DayMemory>;
      if (typeof parsed.date !== "string" || parsed.classes === undefined) {
        return null;
      }
      return {
        date: parsed.date,
        classes: parsed.classes,
        rules: parsed.rules ?? {},
        totals: {
          peopleSeen: parsed.totals?.peopleSeen ?? 0,
          maxPeople: parsed.totals?.maxPeople ?? 0,
          objectsSeen: parsed.totals?.objectsSeen ?? 0,
          rulesFired: parsed.totals?.rulesFired ?? 0,
        },
        // A day written before the scenes existed has no idle record.
        idle: {
          facts: parsed.idle?.facts ?? [],
          questions: parsed.idle?.questions ?? [],
        },
      };
    } catch {
      return null;
    }
  }
}
