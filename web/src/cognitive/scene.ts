/**
 * The four scenes of the dialogue (R2-D, on top of the production-rule engine):
 *
 * - **Attract** — nobody: a fact every 20–40 s, never repeated within the
 *   day, with a short nudge to come closer every fourth line.
 * - **Engage** — someone arrives: one greeting adapted to the number of people
 *   (single / pair / small group / crowd, F-E4 #1/#2), a one-line
 *   self-introduction the first time, then an invitation to interact.
 * - **Play** — react to objects, gestures and smiles with continuity (it
 *   refers back to the previous thing it named), asks playful yes/no questions
 *   and answers a nod or a head shake.
 * - **Farewell** — people leave: one goodbye, then back to Attract after ~10 s.
 *
 * The machine only *decides what to say and when*: every line is offered to
 * `CognitiveEngine.offerScene`, which keeps the single voice, the global gap,
 * the one queue slot and the 3 s expiry. That is why scenes and rulebook rules
 * can never talk over each other (F-E4 #13).
 */

import { FACTS } from "./content/facts";
import { QUESTIONS } from "./content/questions";
import type {
  CognitiveEvent,
  FaceCueKind,
  PoseKind,
  SceneLineKind,
  SceneName,
  WorldModelName,
} from "./events";
import { nounPhrase, numberWord } from "./grammar";
import type { CognitiveMemory } from "./memory";
import type { Expression, GazeTarget, SceneLine, Utterance } from "./rules";

/** What the scene machine needs to know about the world, once per heartbeat. */
export interface SceneWorld {
  /** Stable number of people from the YOLO model (0 while MoveNet is active). */
  people: number;
  crowd: boolean;
  /** A person in the MoveNet skeleton (MoveNet has no count, F-E4 #22). */
  posePerson: boolean;
  mode: WorldModelName;
}

export interface SceneSnapshot {
  name: SceneName;
  question: string | null;
  nextIdleInMs: number | null;
}

export interface SceneOptions {
  /** [0, 1) random source, injected so tests are deterministic. */
  rng?: () => number;
  attractMinMs?: number;
  attractMaxMs?: number;
  /** Back to Attract after the goodbye. */
  farewellMs?: number;
  /** Between the greeting, the self-introduction and the invitation. */
  engageStepMs?: number;
  /** With one person and nothing new, ask a yes/no question after this. */
  questionAfterMs?: number;
  /** Minimum silence before a question. */
  questionIdleMs?: number;
  /** Wait this long for the nod / head shake. */
  answerWindowMs?: number;
  /** Refer back to an object mentioned within this window. */
  continuityMs?: number;
  followUpMinMs?: number;
  followUpMaxMs?: number;
}

const DEFAULTS: Required<SceneOptions> = {
  rng: Math.random,
  attractMinMs: 20_000,
  attractMaxMs: 40_000,
  farewellMs: 10_000,
  engageStepMs: 5_000,
  questionAfterMs: 12_000,
  questionIdleMs: 10_000,
  answerWindowMs: 12_000,
  continuityMs: 90_000,
  followUpMinMs: 8_000,
  followUpMaxMs: 12_000,
};

/** Stimuli collected between two heartbeats, reacted to one per line. */
type Stimulus =
  | { type: "pose"; kind: PoseKind; at: number }
  | { type: "face"; kind: FaceCueKind; at: number }
  | { type: "appear"; label: string; count: number; at: number };

/*
 * Everything the scenes say. Lines addressed to people are gender-neutral
 * ("Che piacere vederti", "Benvenuti" only for groups) and never claim to
 * recognise someone: the system has no identity, so no "di nuovo" or
 * "sei tornato" about a person (F-E4 #5).
 */
const GREETINGS = {
  single: [
    "Ciao! Che piacere vederti.",
    "Ehi, ciao! Vieni pure avanti.",
    "Ciao! Finalmente un po' di compagnia.",
    "Oh, una visita! Ciao!",
  ],
  /** Someone arrives after the stand was empty for a while: about ARI, not them. */
  return: [
    "Finalmente qualcuno! Ciao.",
    "Oh, compagnia! Ciao!",
    "Ciao! Cominciavo ad annoiarmi.",
    "Ciao! Qui era tutto troppo tranquillo.",
  ],
  pair: [
    "Ciao a tutti e due!",
    "Ehi, siete in due! Ciao.",
    "Ciao a voi due! Venite pure avanti.",
    "Doppia visita! Ciao a voi due.",
  ],
  group: [
    "Ciao a tutti! Siete in {n}.",
    "Che bel gruppetto! Ciao a tutti.",
    "Ciao a voi {n}! Venite pure avanti.",
    "Ehi, siete in {n}! Benvenuti.",
  ],
  crowd: [
    "Ciao a tutti! Quanta gente!",
    "Benvenuti! C'è posto per tutti.",
    "Wow, che folla! Ciao a tutti.",
    "Quante persone! Benvenuti al nostro stand.",
  ],
} as const;

const INTRO = [
  "Piacere, sono ARI! Riconosco oggetti, gesti e sorrisi.",
  "Mi chiamo ARI: so riconoscere oggetti, movimenti e sorrisi.",
  "Io sono ARI, e con questa telecamera vedo oggetti, gesti e sorrisi.",
] as const;

const PROMPTS = {
  yolox: [
    "Mostrami un oggetto e provo a indovinare cos'è.",
    "Hai qualcosa in tasca? Fammelo vedere!",
    "Prova con un oggetto: una tazza, un libro, un telefono.",
    "Metti un oggetto davanti alla telecamera, vediamo se indovino.",
  ],
  movenet: [
    "Prova a salutarmi con la mano!",
    "Alza le braccia, vediamo se ti seguo.",
    "Fammi un gesto: prova a fare la T con le braccia!",
    "Muovi le braccia: io ti disegno lo scheletro!",
  ],
} as const;

/** Attract nudges, every fourth idle line. */
const INVITATIONS = [
  "Avvicinati, ti mostro cosa vedo.",
  "Vieni più vicino, non mordo!",
  "C'è qualcuno? Passa a salutarmi!",
  "Hai un oggetto in tasca? Vieni a mostrarmelo.",
  "Fermati un attimo, ti racconto cosa vedo.",
  "Psst! Sì, dico proprio a te. Vieni a giocare?",
] as const;

const GOODBYES = [
  "Ciao, a presto!",
  "Grazie della visita!",
  "Ciao ciao! Io resto qui.",
  "Alla prossima, buona giornata!",
  "Ciao! È stato bello.",
] as const;

const WAVE_FIRST = [
  "Ciao! Ti saluto anch'io.",
  "Ciao ciao! Ricambio il saluto.",
  "Ehi, ciao! Che bel saluto.",
] as const;

const WAVE_AGAIN = [
  "Un altro saluto? Ciao ciao!",
  "Ciao ancora! Non mi stanco mai.",
  "Ricambio, sempre volentieri!",
] as const;

const SMILES = [
  "Che bel sorriso!",
  "Un sorriso! Mi hai messo di buon umore.",
  "Sorridi? Allora sorrido anch'io.",
] as const;

const SMILES_AGAIN = [
  "Un altro sorriso! È contagioso.",
  "Ancora un sorriso? Così mi vizi.",
  "Sorrisi a raffica: che giornata!",
] as const;

const MOUTH_OPEN = [
  "Oh! Che sorpresa!",
  "Bocca aperta? Ho detto qualcosa di strano?",
  "Uno sbadiglio? Spero di non annoiarti!",
] as const;

const LOOKING = [
  "Mi stai guardando? Io guardo te.",
  "Gara di sguardi? Chi ride perde!",
  "Guardi proprio nella telecamera. Ciao!",
] as const;

const NODS = [
  "Sì? Allora siamo d'accordo.",
  "Un bel sì! Ricevuto.",
  "Annuisci? Mi fa piacere.",
] as const;

const SHAKES = [
  "No? Va bene, ne prendo nota.",
  "Un no deciso! Ricevuto.",
  "Scuoti la testa? Allora cambio argomento.",
] as const;

/** The same object, still in view a few seconds after ARI named it. */
const FOLLOW_UPS = [
  "Ancora {oggetto}? Non me lo aspettavo!",
  "Sempre {oggetto}! Non mi stanco di guardare.",
  "Ancora {oggetto}? Sta diventando la mia cosa preferita.",
] as const;

/**
 * Continuity: one object after another, only for pairs that tell a small
 * story together. Any other pair says nothing (no generic "prima X, ora Y").
 */
const CONTINUITY: Record<string, readonly string[]> = {
  "cellulare+tazza": ["Telefono giù, tazza su: pausa caffè?"],
  "tazza+cellulare": ["Finito il caffè, si torna al telefono?"],
  "portatile+tazza": ["Dal computer alla tazza: pausa meritata!"],
  "tazza+portatile": ["Caffè finito? Si torna al computer."],
  "tazza+libro": ["Una tazza e poi un libro: pomeriggio perfetto."],
  "libro+tazza": ["Un libro e una tazza: manca solo il divano."],
  "cellulare+libro": ["Dal telefono al libro: approvo!"],
  "libro+cellulare": ["Dal libro al telefono? Solo una sbirciatina, eh."],
  "banana+mela": ["Banana, mela… manca solo l'arancia!"],
  "mela+arancia": ["Mela e arancia: macedonia in arrivo?"],
  "pizza+torta": ["Prima la pizza, poi la torta: che festa!"],
  "zaino+ombrello": ["Zaino e ombrello: si parte con ogni tempo!"],
  "tastiera+mouse": ["Tastiera e mouse: manca solo lo schermo!"],
  "mouse+tastiera": ["Mouse e tastiera: qui si programma!"],
};

/** Every scene line, by family (for the tests and the phrase report). */
export const SCENE_PHRASES = {
  GREETINGS_SINGLE: GREETINGS.single,
  GREETINGS_RETURN: GREETINGS.return,
  GREETINGS_PAIR: GREETINGS.pair,
  GREETINGS_GROUP: GREETINGS.group,
  GREETINGS_CROWD: GREETINGS.crowd,
  INTRO,
  PROMPTS_YOLOX: PROMPTS.yolox,
  PROMPTS_MOVENET: PROMPTS.movenet,
  INVITATIONS,
  GOODBYES,
  WAVE_FIRST,
  WAVE_AGAIN,
  SMILES,
  SMILES_AGAIN,
  MOUTH_OPEN,
  LOOKING,
  NODS,
  SHAKES,
  FOLLOW_UPS,
  CONTINUITY: Object.values(CONTINUITY).flat(),
} as const satisfies Record<string, readonly string[]>;

/** Last chosen variant of a line id, so a scene never repeats itself. */
function pickVariant(
  phrases: readonly string[],
  previous: string | null,
  rng: () => number,
): string {
  const fresh = phrases.filter((phrase) => phrase !== previous);
  const pool = fresh.length > 0 ? fresh : phrases;
  const index = Math.min(pool.length - 1, Math.floor(rng() * pool.length));
  return pool[index] ?? "";
}

export class SceneMachine {
  readonly options: Required<SceneOptions>;

  private name: SceneName = "attract";
  private started = false;
  private present = false;
  private people = 0;
  private crowd = false;
  private posePerson = false;
  private mode: WorldModelName = "yolox";
  private nextAttractAt = 0;
  private idleLines = 0;
  private engageStep = 0;
  private engageAt = 0;
  private farewellAt = 0;
  private goodbyeSent = false;
  private playSince = 0;
  private lastSpokenAt = Number.NEGATIVE_INFINITY;
  private lastObject: { label: string; count: number; at: number } | null = null;
  private followUp: { label: string; count: number; at: number } | null = null;
  private question: { id: string; askedAt: number } | null = null;
  private proposedQuestion: string | null = null;
  private lastReturnAfterS: number | null = null;
  private lastSmileAt: number | null = null;
  private quietUntil = 0;
  private readonly cooldowns = new Map<string, number>();
  private readonly stimuli: Stimulus[] = [];

  constructor(options: SceneOptions = {}) {
    this.options = { ...DEFAULTS, ...options };
  }

  get scene(): SceneName {
    return this.name;
  }

  /** The question waiting for a nod or a shake, for the "mente" strip. */
  get pendingQuestion(): string | null {
    if (this.question === null) {
      return null;
    }
    return QUESTIONS.find((question) => question.id === this.question?.id)?.text ?? null;
  }

  snapshot(now: number): SceneSnapshot {
    return {
      name: this.name,
      question: this.pendingQuestion,
      nextIdleInMs:
        this.name === "attract" ? Math.max(0, this.nextAttractAt - now) : null,
    };
  }

  /** Forget everything but the day's no-repeat bookkeeping (mode switch). */
  reset(now: number, present = false): void {
    this.name = present ? "play" : "attract";
    this.started = true;
    this.present = present;
    this.people = 0;
    this.crowd = false;
    this.posePerson = false;
    this.nextAttractAt = now + this.attractGap();
    this.idleLines = 0;
    this.engageStep = 0;
    this.goodbyeSent = false;
    this.playSince = now;
    this.lastSpokenAt = Number.NEGATIVE_INFINITY;
    this.lastObject = null;
    this.followUp = null;
    this.question = null;
    this.proposedQuestion = null;
    this.lastReturnAfterS = null;
    this.lastSmileAt = null;
    this.quietUntil = 0;
    this.cooldowns.clear();
    this.stimuli.length = 0;
  }

  /**
   * Seed the world silently for a moment (mode switch, reconnect): a person
   * already in front of the camera is not a new arrival (F-E4 #20/#22).
   */
  silence(now: number, ms = 3_000): void {
    this.quietUntil = now + ms;
  }

  /** Collect the stimuli of a batch of perception events. */
  observe(events: readonly CognitiveEvent[]): void {
    for (const event of events) {
      if (event.type === "pose" && this.name === "play") {
        this.push({ type: "pose", kind: event.kind, at: event.now });
      } else if (event.type === "face" && this.name === "play") {
        this.push({ type: "face", kind: event.kind, at: event.now });
      } else if (event.type === "appear" && !event.scenery && !event.doubtful) {
        this.push({
          type: "appear",
          label: event.label,
          count: event.count,
          at: event.now,
        });
      } else if (event.type === "count") {
        this.lastReturnAfterS = event.returnAfterS;
      }
    }
  }

  /**
   * One decision per heartbeat: the next line the scene wants to say, or
   * `null`. The engine decides whether and when it is actually spoken.
   */
  step(now: number, memory: CognitiveMemory, world: SceneWorld): SceneLine | null {
    if (!this.started) {
      // F-E4 #20: the first three seconds seed the world silently.
      this.reset(now);
      this.silence(now);
    }
    this.syncWorld(now, world);

    if (this.name === "attract") {
      return this.attractLine(now, memory);
    }
    if (this.name === "engage") {
      return this.engageLine(now, memory);
    }
    if (this.name === "farewell") {
      return this.farewellLine(memory);
    }
    return this.playLine(now, memory);
  }

  /** Remember what was actually spoken, for continuity and timing. */
  noteSpoken(utterance: Utterance, memory: CognitiveMemory, now: number): void {
    this.lastSpokenAt = now;
    // Object lines come from the rulebook too: continuity spans both sources.
    if (utterance.objectKey !== null && !utterance.objectKey.includes("+")) {
      this.lastObject = {
        label: utterance.objectKey,
        count: utterance.objectCount ?? 1,
        at: now,
      };
      this.scheduleFollowUp(utterance.objectKey, now);
    }
    if (utterance.scene === undefined) {
      return;
    }
    if (utterance.sceneKind === "fact") {
      memory.markIdleUsed("facts", utterance.ruleId.replace("scene.fact.", ""));
    } else if (utterance.sceneKind === "question" && this.proposedQuestion !== null) {
      memory.markIdleUsed("questions", this.proposedQuestion);
      this.question = { id: this.proposedQuestion, askedAt: now };
      this.proposedQuestion = null;
    } else if (
      utterance.sceneKind === "answer" ||
      utterance.sceneKind === "goodbye"
    ) {
      this.question = null;
      this.proposedQuestion = null;
    }
  }

  // --- scenes ---------------------------------------------------------------

  private syncWorld(now: number, world: SceneWorld): void {
    const wasPresent = this.present;
    this.mode = world.mode;
    this.people = world.people;
    this.crowd = world.crowd;
    this.posePerson = world.posePerson;
    this.present = world.people > 0 || (world.mode === "movenet" && world.posePerson);

    if (this.present) {
      if (this.name === "attract") {
        if (now < this.quietUntil) {
          this.enterPlay(now);
        } else {
          this.enterEngage(now);
        }
      } else if (this.name === "farewell") {
        // A short absence (< 10 s) is not a new arrival: no second greeting
        // (F-E4 #4), straight back to Play.
        if (now - this.farewellAt < this.options.farewellMs) {
          this.enterPlay(now);
        } else {
          this.enterEngage(now);
        }
      }
    } else if (wasPresent || this.name === "engage" || this.name === "play") {
      if (this.name === "engage" || this.name === "play") {
        this.enterFarewell(now);
      }
    } else if (this.name === "farewell" && now - this.farewellAt >= this.options.farewellMs) {
      this.enterAttract(now);
    }
  }

  private enterEngage(now: number): void {
    this.name = "engage";
    this.engageStep = 0;
    this.engageAt = now;
    this.stimuli.length = 0;
  }

  private enterPlay(now: number): void {
    this.name = "play";
    this.playSince = now;
    this.question = null;
    this.proposedQuestion = null;
    this.followUp = null;
  }

  private enterFarewell(now: number): void {
    this.name = "farewell";
    this.farewellAt = now;
    this.goodbyeSent = false;
    this.question = null;
    this.proposedQuestion = null;
    this.followUp = null;
    this.lastObject = null;
    this.stimuli.length = 0;
  }

  private enterAttract(now: number): void {
    this.name = "attract";
    this.nextAttractAt = now + this.attractGap();
    this.idleLines = 0;
    this.lastObject = null;
    this.followUp = null;
    this.stimuli.length = 0;
  }

  // --- Attract --------------------------------------------------------------

  private attractLine(now: number, memory: CognitiveMemory): SceneLine | null {
    if (now < this.nextAttractAt) {
      return null;
    }
    this.nextAttractAt = now + this.attractGap();
    this.idleLines += 1;

    // Every fourth idle line is a nudge, so the facts stay the star.
    if (this.idleLines % 4 === 0) {
      return this.line({
        ruleId: "scene.invitation",
        kind: "invitation",
        priority: 40,
        expression: "curious",
        gaze: "wander",
        phrases: INVITATIONS,
        memory,
        valid: () => this.name === "attract",
      });
    }

    const fact = this.pickFact(memory);
    if (fact === null) {
      return null;
    }
    return {
      ruleId: `scene.fact.${fact.id}`,
      scene: "attract",
      kind: "fact",
      text: fact.text,
      expression: "curious",
      gaze: "wander",
      priority: 38,
      valid: () => this.name === "attract",
    };
  }

  private pickFact(memory: CognitiveMemory): (typeof FACTS)[number] | null {
    const fresh = FACTS.filter((fact) => !memory.idleUsed("facts", fact.id));
    if (fresh.length === 0) {
      return null;
    }
    return fresh[Math.min(fresh.length - 1, Math.floor(this.options.rng() * fresh.length))] ?? null;
  }

  private attractGap(): number {
    const { attractMinMs, attractMaxMs } = this.options;
    return attractMinMs + this.options.rng() * Math.max(0, attractMaxMs - attractMinMs);
  }

  // --- Engage ---------------------------------------------------------------

  private engageLine(now: number, memory: CognitiveMemory): SceneLine | null {
    if (this.people === 0 && !(this.mode === "movenet" && this.posePerson)) {
      return null;
    }
    if (this.engageStep > 0 && now - this.engageAt < this.options.engageStepMs) {
      return null;
    }
    if (this.engageStep === 0) {
      this.engageStep = 1;
      this.engageAt = now;
      return this.greeting(memory);
    }
    if (this.engageStep === 1 && memory.fireCount("scene.intro") === 0) {
      this.engageStep = 2;
      this.engageAt = now;
      return this.line({
        ruleId: "scene.intro",
        kind: "intro",
        priority: 82,
        expression: "happy",
        gaze: "person",
        phrases: INTRO,
        memory,
        valid: () => this.present,
      });
    }
    // The invitation closes Engage and opens Play. It names the interaction the
    // current view can actually show: gestures in the skeleton view, objects in
    // the objects view (one model at a time on the board).
    this.engageStep = 3;
    const promptMode = this.mode;
    const prompts = promptMode === "movenet" ? PROMPTS.movenet : PROMPTS.yolox;
    const prompt = this.line({
      ruleId: `scene.prompt.${promptMode}`,
      kind: "prompt",
      priority: 80,
      expression: "curious",
      gaze: "person",
      phrases: prompts,
      memory,
      valid: () => this.present && this.mode === promptMode,
    });
    this.enterPlay(now);
    return prompt;
  }

  private greeting(memory: CognitiveMemory): SceneLine {
    const people = Math.max(1, this.people);
    const returning = this.lastReturnAfterS !== null && this.lastReturnAfterS >= 30;
    this.lastReturnAfterS = null;
    if (people >= 5) {
      return this.line({
        ruleId: "scene.greeting.crowd",
        kind: "greeting",
        priority: 104,
        expression: "surprised",
        gaze: "person",
        phrases: GREETINGS.crowd,
        memory,
        valid: () => this.present,
      });
    }
    if (people >= 3) {
      const template = pickVariant(
        GREETINGS.group,
        memory.lastPhrase("scene.greeting.group"),
        this.options.rng,
      );
      return {
        ruleId: "scene.greeting.group",
        scene: "engage",
        kind: "greeting",
        text: template.replaceAll("{n}", numberWord(people)),
        expression: "happy",
        gaze: "person",
        priority: 102,
        valid: () => this.present,
      };
    }
    if (people === 2) {
      return this.line({
        ruleId: "scene.greeting.pair",
        kind: "greeting",
        priority: 102,
        expression: "happy",
        gaze: "person",
        phrases: GREETINGS.pair,
        memory,
        valid: () => this.present,
      });
    }
    // Exactly one person: direct "tu" family (F-E4 #2). A return after more
    // than 30 s gets the welcome-back family, never an identity claim (#5).
    return this.line({
      ruleId: returning ? "scene.greeting.return" : "scene.greeting.single",
      kind: "greeting",
      priority: 102,
      expression: "happy",
      gaze: "person",
      phrases: returning ? GREETINGS.return : GREETINGS.single,
      memory,
      valid: () => this.present,
    });
  }

  // --- Play -----------------------------------------------------------------

  private playLine(now: number, memory: CognitiveMemory): SceneLine | null {
    if (this.question !== null && now - this.question.askedAt > this.options.answerWindowMs) {
      this.question = null;
    }

    let stimulus: Stimulus | undefined;
    while (this.stimuli.length > 0) {
      const next = this.stimuli.shift()!;
      // A queued cue older than the speech contract is no longer relevant
      // (F-E4 #13): a smile or a nod from four seconds ago is not an answer.
      if (now - next.at <= 3_000) {
        stimulus = next;
        break;
      }
    }
    if (stimulus !== undefined) {
      const line = this.react(stimulus, now, memory);
      if (line !== null) {
        return line;
      }
    }

    if (this.followUp !== null && now >= this.followUp.at) {
      const followUp = this.followUp;
      this.followUp = null;
      if (this.cooldown("follow_up", now, 45_000)) {
        return this.line({
          ruleId: "scene.follow_up",
          kind: "reaction",
          priority: 44,
          expression: "curious",
          gaze: "object",
          phrases: FOLLOW_UPS,
          slots: { oggetto: nounPhrase(followUp.label, followUp.count) },
          memory,
          objectKey: followUp.label,
          valid: () => this.name === "play" && this.present,
        });
      }
    }

    const soloOrSmallGroup =
      (this.people >= 1 && this.people <= 4) ||
      (this.mode === "movenet" && this.posePerson);
    if (
      this.question === null &&
      soloOrSmallGroup &&
      !this.crowd &&
      now - this.playSince >= this.options.questionAfterMs &&
      now - this.lastSpokenAt >= this.options.questionIdleMs
    ) {
      return this.questionLine(memory);
    }
    return null;
  }

  private react(
    stimulus: Stimulus,
    now: number,
    memory: CognitiveMemory,
  ): SceneLine | null {
    if (stimulus.type === "pose") {
      return this.reactPose(stimulus.kind, now, memory);
    }
    if (stimulus.type === "face") {
      return this.reactFace(stimulus.kind, now, memory);
    }
    return this.reactObject(stimulus.label, stimulus.count, now, memory);
  }

  private reactPose(
    kind: PoseKind,
    now: number,
    memory: CognitiveMemory,
  ): SceneLine | null {
    if (kind !== "wave") {
      // The other confirmed gestures already have their rulebook family.
      return null;
    }
    if (!this.cooldown("wave", now, 8_000)) {
      return null;
    }
    const firstTime = memory.fireCount("scene.wave.first") === 0;
    return this.line({
      ruleId: firstTime ? "scene.wave.first" : "scene.wave",
      kind: "reaction",
      priority: 76,
      expression: "happy",
      gaze: "person",
      phrases: firstTime ? WAVE_FIRST : WAVE_AGAIN,
      memory,
      valid: () => this.name === "play" && this.present,
    });
  }

  private reactFace(
    kind: FaceCueKind,
    now: number,
    memory: CognitiveMemory,
  ): SceneLine | null {
    if (kind === "nod" || kind === "head_shake") {
      const answer = this.answerLine(kind, now, memory);
      if (answer !== null) {
        return answer;
      }
      if (!this.cooldown(kind, now, 20_000)) {
        return null;
      }
      return this.line({
        ruleId: kind === "nod" ? "scene.nod" : "scene.head_shake",
        kind: "reaction",
        priority: 58,
        expression: kind === "nod" ? "happy" : "curious",
        gaze: "person",
        phrases: kind === "nod" ? NODS : SHAKES,
        memory,
        valid: () => this.name === "play" && this.present,
      });
    }
    if (kind === "smile") {
      if (!this.cooldown("smile", now, 10_000)) {
        return null;
      }
      const recently = this.lastSmileAt !== null && now - this.lastSmileAt < 60_000;
      this.lastSmileAt = now;
      return this.line({
        ruleId: recently ? "scene.smile.again" : "scene.smile",
        kind: "reaction",
        priority: 64,
        expression: "happy",
        gaze: "person",
        phrases: recently ? SMILES_AGAIN : SMILES,
        memory,
        valid: () => this.name === "play" && this.present,
      });
    }
    if (kind === "mouth_open") {
      if (!this.cooldown("mouth_open", now, 20_000)) {
        return null;
      }
      return this.line({
        ruleId: "scene.mouth_open",
        kind: "reaction",
        priority: 62,
        expression: "surprised",
        gaze: "person",
        phrases: MOUTH_OPEN,
        memory,
        valid: () => this.name === "play" && this.present,
      });
    }
    if (!this.cooldown("looking_at_me", now, 30_000)) {
      return null;
    }
    return this.line({
      ruleId: "scene.looking",
      kind: "reaction",
      priority: 48,
      expression: "curious",
      gaze: "person",
      phrases: LOOKING,
      memory,
      valid: () => this.name === "play" && this.present,
    });
  }

  private reactObject(
    label: string,
    count: number,
    now: number,
    memory: CognitiveMemory,
  ): SceneLine | null {
    if (this.lastObject === null || this.lastObject.label === label) {
      return null;
    }
    if (this.crowd || now - this.lastObject.at > this.options.continuityMs) {
      return null;
    }
    const previous = this.lastObject;
    const key = `${previous.label}+${label}`;
    const phrases = CONTINUITY[key];
    if (phrases === undefined) {
      // Only pairs that make a small story; the rulebook names the object.
      return null;
    }
    const text = pickVariant(
      phrases,
      memory.lastPhrase("scene.continuity"),
      this.options.rng,
    );
    return {
      ruleId: "scene.continuity",
      scene: "play",
      kind: "continuity",
      text,
      expression: "curious",
      gaze: "object",
      priority: 68,
      cooldown_s: 20,
      objectKey: label,
      objectCount: count,
      valid: () => this.name === "play" && this.present,
    };
  }

  private answerLine(
    kind: "nod" | "head_shake",
    now: number,
    memory: CognitiveMemory,
  ): SceneLine | null {
    if (this.question === null) {
      return null;
    }
    if (now - this.question.askedAt > this.options.answerWindowMs) {
      this.question = null;
      return null;
    }
    const question = QUESTIONS.find((entry) => entry.id === this.question?.id);
    if (question === undefined) {
      this.question = null;
      return null;
    }
    const phrases = kind === "nod" ? question.yes : question.no;
    const line = this.line({
      ruleId: `scene.answer.${question.id}.${kind === "nod" ? "yes" : "no"}`,
      kind: "answer",
      priority: 84,
      expression: "happy",
      gaze: "person",
      phrases,
      memory,
      valid: () => this.name === "play" && this.present,
    });
    this.question = null;
    return line;
  }

  private questionLine(memory: CognitiveMemory): SceneLine | null {
    const fresh = QUESTIONS.filter((question) => !memory.idleUsed("questions", question.id));
    if (fresh.length === 0) {
      return null;
    }
    const question =
      fresh[Math.min(fresh.length - 1, Math.floor(this.options.rng() * fresh.length))];
    if (question === undefined) {
      return null;
    }
    this.proposedQuestion = question.id;
    return {
      ruleId: `scene.ask.${question.id}`,
      scene: "play",
      kind: "question",
      text: question.text,
      expression: "curious",
      gaze: "person",
      priority: 72,
      valid: () => this.name === "play" && this.present,
    };
  }

  private scheduleFollowUp(label: string, now: number): void {
    if (this.mode !== "yolox" || this.crowd) {
      return;
    }
    if (this.options.rng() > 0.6) {
      return;
    }
    const { followUpMinMs, followUpMaxMs } = this.options;
    const delay = followUpMinMs + this.options.rng() * Math.max(0, followUpMaxMs - followUpMinMs);
    this.followUp = { label, count: 1, at: now + delay };
  }

  // --- Farewell -------------------------------------------------------------

  private farewellLine(memory: CognitiveMemory): SceneLine | null {
    if (this.goodbyeSent) {
      return null;
    }
    this.goodbyeSent = true;
    return this.line({
      ruleId: "scene.goodbye",
      kind: "goodbye",
      priority: 78,
      expression: "neutral",
      gaze: "wander",
      phrases: GOODBYES,
      memory,
      valid: () => this.name === "farewell",
    });
  }

  // --- helpers --------------------------------------------------------------

  private line(input: {
    ruleId: string;
    kind: SceneLineKind;
    priority: number;
    expression: Expression;
    gaze: GazeTarget;
    phrases: readonly string[];
    memory: CognitiveMemory;
    objectKey?: string | null;
    cooldown_s?: number;
    slots?: Record<string, string | number>;
    valid?: (memory: CognitiveMemory, now: number) => boolean;
  }): SceneLine {
    let text = pickVariant(input.phrases, input.memory.lastPhrase(input.ruleId), this.options.rng);
    if (input.slots !== undefined) {
      for (const [slot, value] of Object.entries(input.slots)) {
        text = text.replaceAll(`{${slot}}`, String(value));
      }
    }
    const line: SceneLine = {
      ruleId: input.ruleId,
      scene: this.name === "engage" ? "engage" : this.name,
      kind: input.kind,
      text,
      expression: input.expression,
      gaze: input.gaze,
      priority: input.priority,
      cooldown_s: input.cooldown_s ?? 0,
      objectKey: input.objectKey ?? null,
    };
    if (input.valid !== undefined) {
      line.valid = input.valid;
    }
    return line;
  }

  /** Per-cue cooldown; returns true when the cue may speak now. */
  private cooldown(key: string, now: number, ms: number): boolean {
    const last = this.cooldowns.get(key);
    if (last !== undefined && now - last < ms) {
      return false;
    }
    this.cooldowns.set(key, now);
    return true;
  }

  private push(stimulus: Stimulus): void {
    this.stimuli.push(stimulus);
    if (this.stimuli.length > 8) {
      this.stimuli.shift();
    }
  }
}
