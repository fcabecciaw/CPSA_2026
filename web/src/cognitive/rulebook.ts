/**
 * The Italian rulebook of the cognitive layer (F-E3 `rulebook.ts`, F-E4).
 *
 * Data only: every rule declares when it matches, its priority, cooldown and
 * the sentences it may say. There is exactly one rule per COCO class (the
 * person class is covered by `person.single`), plus the people/group families,
 * pairs, dwell, idle, self-facts, milestones, clock, mode switch, pose and
 * vision families. Phrases are short (≤ 12 words), warm and never about a
 * person's body, age or looks (F-E4 #11).
 */

import { SHARP_CLASSES, IMPLAUSIBLE_INDOOR } from "./perception";
import type { CognitiveEvent } from "./events";
import type { CognitiveMemory } from "./memory";
import type { Expression, Rule } from "./rules";
import { COCO_IT_LABELS, type CocoLabel } from "./grammar";

type ObjectLabel = Exclude<CocoLabel, "persona">;

/**
 * What ARI says the first time a class is seen today. `{oggetto}` is the class
 * with its definite article or a number ("il cellulare", "due tazze", "le
 * forbici"), so every line keeps the slot where article, gender and number do
 * not matter: after "Vedo" or "Ecco", never as the subject of a verb and never
 * next to an adjective (F-E4 #8).
 *
 * Classes likely at an indoor stand (desk items, props, snacks) get specific
 * lines; every other COCO class gets the honest generic pair below.
 */
const SPECIFIC_LINES: Partial<Record<ObjectLabel, readonly string[]>> = {
  cellulare: [
    "Vedo {oggetto}! Chissà quante notifiche.",
    "Ecco {oggetto}. Una foto ricordo con me?",
    "Ah, {oggetto}! Anche lì dentro c'è una telecamera, come me.",
  ],
  tazza: [
    "Vedo {oggetto}. Caffè o tè?",
    "Ecco {oggetto}! Peccato, io il caffè non posso berlo.",
    "Ah, {oggetto}. Io vado avanti a elettricità, grazie.",
  ],
  bottiglia: [
    "Vedo {oggetto}. Bere acqua fa sempre bene!",
    "Ecco {oggetto}. Io invece mi ricarico con la corrente.",
    "Ah, {oggetto}! Poi nella raccolta differenziata, mi raccomando.",
  ],
  calice: [
    "Vedo {oggetto}. Solo acqua, spero!",
    "Ecco {oggetto}. Un brindisi? Io alzo la telecamera.",
  ],
  libro: [
    "Ecco {oggetto}! Adoro le storie.",
    "Vedo {oggetto}. Io ho studiato su centomila fotografie.",
    "Ah, {oggetto}! Leggere è il miglior allenamento per il cervello.",
  ],
  portatile: [
    "Ecco {oggetto}! Siamo quasi parenti.",
    "Vedo {oggetto}. È lì che nascono i robot come me.",
  ],
  tastiera: [
    "Vedo {oggetto}. Quanti tasti, e io non ho dita!",
    "Ecco {oggetto}! Clic, clac: si scrive codice.",
  ],
  mouse: [
    "Vedo {oggetto}. Quello senza coda, per fortuna!",
    "Ecco {oggetto}! Niente formaggio, solo clic.",
  ],
  telecomando: [
    "Vedo {oggetto}. Non premete il tasto per spegnermi!",
    "Ecco {oggetto}! Purtroppo io non cambio canale.",
  ],
  forbici: [
    "Vedo {oggetto}. Piano piano, mi raccomando.",
    "Ecco {oggetto}. Con calma, niente fretta.",
  ],
  coltello: [
    "Vedo {oggetto}. Con calma, mi raccomando.",
    "Ecco {oggetto}. Piano piano, serve solo per la torta.",
  ],
  forchetta: [
    "Vedo {oggetto}. È già ora di pranzo?",
    "Ecco {oggetto}! Manca solo un bel piatto di pasta.",
  ],
  cucchiaio: [
    "Vedo {oggetto}. Gelato in arrivo?",
    "Ecco {oggetto}! Minestra o tiramisù?",
  ],
  orsacchiotto: [
    "Vedo {oggetto}! Che tenerezza.",
    "Ecco {oggetto}. Un abbraccio da parte mia!",
    "Oh, {oggetto}! Che dolcezza.",
  ],
  banana: [
    "Vedo {oggetto}. Ottima merenda!",
    "Ecco {oggetto}! Attenzione alle bucce sul pavimento.",
  ],
  mela: [
    "Vedo {oggetto}. Una al giorno toglie il medico di torno!",
    "Ecco {oggetto}! Merenda sana in arrivo.",
  ],
  arancia: [
    "Vedo {oggetto}. Vitamina C in arrivo!",
    "Ecco {oggetto}! Spremuta per tutti?",
  ],
  zaino: [
    "Vedo {oggetto}. Si va a lezione?",
    "Ecco {oggetto}! Chissà quanti libri ci sono dentro.",
  ],
  borsa: [
    "Vedo {oggetto}. Chissà cosa c'è dentro!",
    "Ecco {oggetto}. Io non ho tasche, che invidia!",
  ],
  ombrello: [
    "Vedo {oggetto}. Piove a Cagliari? Che rarità!",
    "Ecco {oggetto}! Previsioni del tempo: incerte.",
  ],
  orologio: [
    "Vedo {oggetto}. Io l'ora la so sempre, comunque.",
    "Ecco {oggetto}! Il tempo vola quando ci si diverte.",
  ],
  sedia: [
    "Vedo {oggetto}. C'è posto per sedersi!",
    "Ecco {oggetto}. Io invece sto sempre in piedi… più o meno.",
  ],
  tv: [
    "Vedo {oggetto}. Che cosa danno stasera?",
    "Ecco {oggetto}! Uno schermo grande, come piace a me.",
  ],
  vaso: [
    "Vedo {oggetto}. Mancano solo i fiori!",
    "Ecco {oggetto}. Maneggiare con cura!",
  ],
  pianta: [
    "Vedo {oggetto}! Un po' di verde fa sempre bene.",
    "Ecco {oggetto}. Che bel verde, qui dentro!",
  ],
  cravatta: [
    "Vedo {oggetto}. Giornata importante?",
    "Ecco {oggetto}! Io preferisco il papillon.",
  ],
  valigia: [
    "Vedo {oggetto}. Si parte per un viaggio?",
    "Ecco {oggetto}! C'è posto anche per un robot?",
  ],
  spazzolino: [
    "Vedo {oggetto}. Due minuti di spazzolata, mi raccomando!",
    "Ecco {oggetto}! Io, per fortuna, i denti non li ho.",
  ],
  panino: [
    "Vedo {oggetto}. Che fame!",
    "Ecco {oggetto}! È quasi ora di merenda.",
  ],
  pizza: [
    "Vedo {oggetto}! Se potessi mangiare, sarebbe il mio sogno.",
    "Ecco {oggetto}. Margherita o capricciosa?",
  ],
  torta: [
    "Vedo {oggetto}! C'è qualcosa da festeggiare?",
    "Ecco {oggetto}. Una fetta per me? Scherzo!",
  ],
  ciambella: [
    "Vedo {oggetto}! Merenda dolce in arrivo.",
    "Ecco {oggetto}. Il buco in mezzo è la mia parte preferita.",
  ],
  palla: [
    "Vedo {oggetto}! Facciamo due tiri?",
    "Ecco {oggetto}. Io non so palleggiare, purtroppo.",
  ],
  frisbee: [
    "Vedo {oggetto}! Peccato, qui dentro non si può lanciare.",
    "Ecco {oggetto}. Io mi limito a guardare.",
  ],
};

/**
 * Every other COCO class: one honest, light line with the indefinite article
 * ("un'auto", "uno skateboard", "due panchine"), no forced poetry.
 */
const GENERIC_LINES = [
  "Mi sembra di vedere {unoggetto}. Dico bene?",
  "Se non sbaglio, vedo {unoggetto}.",
] as const;

/** The lines of one class: its specific set, else the generic pair. */
export function classLines(label: ObjectLabel): readonly string[] {
  return SPECIFIC_LINES[label] ?? GENERIC_LINES;
}

/** Classes with their own lines (exported for the tests and the report). */
export const SPECIFIC_CLASSES: readonly string[] = Object.keys(SPECIFIC_LINES);

const ANIMALS = new Set([
  "uccello",
  "gatto",
  "cane",
  "cavallo",
  "pecora",
  "mucca",
  "elefante",
  "orso",
  "zebra",
  "giraffa",
]);
const FOOD = new Set([
  "banana",
  "mela",
  "panino",
  "arancia",
  "broccolo",
  "carota",
  "hot dog",
  "pizza",
  "ciambella",
  "torta",
  "bottiglia",
  "calice",
  "tazza",
]);
const TECH = new Set([
  "tv",
  "portatile",
  "mouse",
  "telecomando",
  "tastiera",
  "cellulare",
  "microonde",
  "forno",
  "tostapane",
  "lavandino",
  "frigorifero",
  "orologio",
  "phon",
]);

function classPriority(label: ObjectLabel): number {
  if (SHARP_CLASSES.has(label)) {
    return 32;
  }
  if (IMPLAUSIBLE_INDOOR.has(label)) {
    return 46;
  }
  if (ANIMALS.has(label)) {
    return 60;
  }
  if (TECH.has(label)) {
    return 58;
  }
  if (FOOD.has(label)) {
    return 56;
  }
  return 52;
}

function classExpression(label: ObjectLabel): Expression {
  return ANIMALS.has(label) || FOOD.has(label) ? "happy" : "curious";
}

function slug(label: string): string {
  return label.normalize("NFD").replace(/[^a-z0-9]+/gi, "_").toLowerCase();
}

/** One appear rule per COCO object class (F-E4 #7/#8/#10/#12). */
function classRules(): Rule[] {
  return (COCO_IT_LABELS as readonly CocoLabel[])
    .filter((label): label is ObjectLabel => label !== "persona")
    .map((label) => {
      const sharp = SHARP_CLASSES.has(label);
      return {
        id: `object.${slug(label)}`,
        classes: [label],
        category: "object",
        priority: classPriority(label),
        // Sharp objects are mentioned rarely and gently (F-E4 #12).
        cooldown_s: sharp ? 120 : 45,
        per_object_cooldown_s: sharp ? 180 : 60,
        expression: classExpression(label),
        gaze: "object",
        model: "yolox",
        phrases: classLines(label),
        when: (event: CognitiveEvent, memory: CognitiveMemory): boolean =>
          event.type === "appear" &&
          event.label === label &&
          !event.scenery &&
          !event.doubtful &&
          !memory.working.focus.crowd &&
          memory.working.focus.mode === "yolox" &&
          // The first stable sighting of the day belongs to the class rule;
          // from the second on, `memory.object_count` counts it.
          (memory.day.classes[event.label]?.times ?? 0) <= 1,
      };
    });
}

function pairMatches(event: CognitiveEvent, a: string, b: string): boolean {
  return (
    event.type === "together" &&
    ((event.a === a && event.b === b) || (event.a === b && event.b === a))
  );
}

/**
 * The hello rules fire on people counts; the live agent greets from person and
 * face tracks instead (`presence.ts`, C3) and sets `greeter: "arrivals"`, so
 * these families stay for the count-only world (tests, the mock) and never say
 * a second hello.
 */
function rulesGreet(memory: CognitiveMemory): boolean {
  return memory.working.focus.greeter !== "arrivals";
}

const personRules: Rule[] = [
  {
    id: "person.single",
    classes: ["persona"],
    category: "person",
    priority: 90,
    cooldown_s: 12,
    expression: "happy",
    gaze: "person",
    model: "yolox",
    phrases: ["Ciao! Ti vedo benissimo.", "Ehi, ciao! Che piacere.", "Ciao! Vieni pure avanti."],
    // Exactly one person, direct "tu" family (F-E4 #2).
    when: (event, memory) =>
      rulesGreet(memory) &&
      event.type === "count" &&
      event.from === 0 &&
      event.to === 1 &&
      !event.quiet &&
      memory.working.focus.people === 1,
  },
  {
    id: "person.two",
    category: "person",
    priority: 88,
    cooldown_s: 15,
    expression: "happy",
    gaze: "person",
    model: "yolox",
    phrases: ["Ciao a tutti e due!", "Ciao a voi due!", "Ehi, siete in due! Ciao."],
    when: (event, memory) =>
      rulesGreet(memory) &&
      event.type === "count" &&
      event.to === 2 &&
      event.from < 2 &&
      !event.quiet &&
      memory.working.focus.people === 2,
  },
  {
    id: "person.small_group",
    category: "person",
    priority: 86,
    cooldown_s: 20,
    expression: "surprised",
    gaze: "person",
    model: "yolox",
    phrases: [
      "Ciao a tutti! Siete in {n}.",
      "Che bel gruppetto! Ciao a tutti.",
      "Ehi, ciao a tutti quanti!",
    ],
    when: (event, memory) =>
      rulesGreet(memory) &&
      event.type === "count" &&
      event.to >= 3 &&
      event.to <= 4 &&
      event.from < event.to &&
      !event.quiet &&
      memory.working.focus.people === event.to,
  },
  {
    id: "person.crowd",
    category: "person",
    priority: 84,
    cooldown_s: 60,
    expression: "surprised",
    gaze: "person",
    model: "yolox",
    phrases: [
      "Ciao a tutti! Quanta gente!",
      "Benvenuti! C'è posto per tutti.",
      "Che folla! Ciao a tutti.",
    ],
    // Crowd mode: group-level lines only (F-E4 #2).
    when: (event, memory) =>
      rulesGreet(memory) &&
      event.type === "count" &&
      event.to >= 5 &&
      event.from < 5 &&
      !event.quiet &&
      memory.working.focus.people === event.to,
  },
  // No "person.leave" rule: the goodbye belongs to the Farewell scene alone,
  // gated by the fused presence (5 s with no person, face or skeleton, after a
  // greeting, at most once per 60 s; C3).
  {
    id: "person.return",
    category: "person",
    priority: 82,
    cooldown_s: 60,
    expression: "happy",
    gaze: "person",
    model: "yolox",
    phrases: [
      "Finalmente qualcuno! Ciao.",
      "Oh, compagnia! Ciao!",
      "Ciao! Qui era tutto troppo tranquillo.",
    ],
    // Only after ≥ 30 s away: no identity claim, just the count (#4/#5).
    when: (event, memory) =>
      rulesGreet(memory) &&
      event.type === "count" &&
      event.returnAfterS !== null &&
      event.to > 0 &&
      memory.working.focus.people === event.to,
  },
  {
    id: "person.stay_25",
    category: "dwell",
    priority: 42,
    cooldown_s: 120,
    expression: "curious",
    gaze: "person",
    model: "yolox",
    phrases: [
      "Ti piace stare qui? Anche a me.",
      "Che bello, resti con me!",
      "Ti trovi bene qui? Lo spero!",
    ],
    when: (event, memory) =>
      event.type === "dwell" &&
      event.label === "persona" &&
      event.seconds >= 25 &&
      event.count === 1 &&
      memory.working.focus.people === 1,
  },
  {
    id: "person.stay_120",
    category: "dwell",
    priority: 40,
    cooldown_s: 180,
    expression: "curious",
    gaze: "person",
    model: "yolox",
    phrases: [
      "Due minuti insieme! Che bella compagnia.",
      "Ti piace proprio qui, eh?",
      "Resta quanto vuoi, io non mi stanco.",
    ],
    when: (event, memory) =>
      event.type === "dwell" &&
      event.label === "persona" &&
      event.seconds >= 120 &&
      event.count === 1 &&
      memory.working.focus.people === 1,
  },
];

const objectFamilyRules: Rule[] = [
  {
    id: "class.doubtful",
    category: "object",
    priority: 48,
    cooldown_s: 120,
    expression: "thinking",
    gaze: "object",
    model: "yolox",
    phrases: [
      "Forse vedo {unoggetto}. O forse no.",
      "Mi sembra di vedere {unoggetto}… possibile?",
      "Vedo {unoggetto}? Strano, qui dentro!",
    ],
    when: (event, memory) =>
      event.type === "appear" &&
      event.doubtful &&
      !memory.working.focus.crowd &&
      memory.working.focus.mode === "yolox",
  },
  {
    id: "memory.object_return",
    category: "memory",
    priority: 58,
    cooldown_s: 30,
    per_object_cooldown_s: 60,
    expression: "curious",
    gaze: "object",
    model: "yolox",
    phrases: ["Riecco {oggetto}!", "Ecco di nuovo {oggetto}.", "Oh, rivedo {oggetto}!"],
    when: (event, memory) =>
      event.type === "return" &&
      event.label !== "persona" &&
      !memory.working.focus.crowd,
  },
  {
    id: "memory.object_count",
    category: "object",
    // Just below the ordinary class rules (the lowest starts at 52), so the
    // class-specific line keeps the first sighting and this line counts the
    // stable re-appearances ("terzo cellulare di oggi").
    priority: 51,
    cooldown_s: 0,
    per_object_cooldown_s: 60,
    expression: "happy",
    gaze: "object",
    model: "yolox",
    phrases: [
      "{Ordinale} {singolare} di oggi.",
      "{Ordinale} {singolare} della giornata.",
      "E siamo a ben {conteggio} {plurale}!",
      "Già {conteggio} {plurale} oggi!",
      "Ecco {oggettoNumero} numero {numero}.",
      "Ancora {plurale}? Siamo a quota {conteggio}.",
    ],
    when: (event, memory) =>
      event.type === "appear" &&
      event.label !== "persona" &&
      !event.scenery &&
      !event.doubtful &&
      !memory.working.focus.crowd &&
      memory.working.focus.mode === "yolox" &&
      (memory.day.classes[event.label]?.times ?? 0) >= 2,
  },
  {
    id: "memory.people_total",
    category: "memory",
    priority: 30,
    cooldown_s: 600,
    max_per_day: 3,
    expression: "thinking",
    gaze: "person",
    model: "yolox",
    phrases: [
      "Oggi ho visto circa {totale} persone.",
      "Circa {totale} persone oggi: che via vai!",
      "Finora circa {totale} persone. Che giornata!",
    ],
    when: (event, memory) =>
      event.type === "count" &&
      memory.working.focus.people > 0 &&
      memory.day.totals.peopleSeen >= 2,
  },
  {
    id: "dwell.object_30",
    category: "dwell",
    priority: 38,
    cooldown_s: 180,
    per_object_cooldown_s: 300,
    expression: "curious",
    gaze: "object",
    model: "yolox",
    phrases: [
      "Vedo {oggetto} da un bel po'.",
      "Sempre {oggetto} lì: fa parte dell'arredamento?",
      "È da un po' che guardo {oggetto}.",
    ],
    when: (event, memory) =>
      event.type === "dwell" &&
      event.label !== "persona" &&
      event.seconds >= 30 &&
      !memory.working.focus.crowd,
  },
];

const pairRules: Rule[] = [
  {
    id: "pair.tazza_libro",
    category: "pair",
    priority: 57,
    cooldown_s: 120,
    expression: "happy",
    gaze: "object",
    model: "yolox",
    phrases: ["Una tazza e un libro: pomeriggio perfetto.", "Libro e tazza, manca solo il divano."],
    when: (event, memory) =>
      pairMatches(event, "tazza", "libro") && !memory.working.focus.crowd,
  },
  {
    id: "pair.persona_cellulare",
    category: "pair",
    priority: 56,
    cooldown_s: 120,
    expression: "curious",
    gaze: "person",
    model: "yolox",
    phrases: ["Telefono in mano? Fammi una foto!", "Vuoi farmi una foto? Mi metto in posa."],
    when: (event, memory) =>
      pairMatches(event, "persona", "cellulare") && !memory.working.focus.crowd,
  },
  {
    id: "pair.tavolo_tazza",
    category: "pair",
    priority: 52,
    cooldown_s: 120,
    expression: "curious",
    gaze: "object",
    model: "yolox",
    phrases: [
      "Una tazza sul tavolo: qualcuno ha fatto una pausa?",
      "Una tazza sul tavolo… di chi sarà?",
    ],
    when: (event, memory) =>
      pairMatches(event, "tavolo", "tazza") && !memory.working.focus.crowd,
  },
  {
    id: "pair.portatile_mouse",
    category: "pair",
    priority: 52,
    cooldown_s: 120,
    expression: "thinking",
    gaze: "object",
    model: "yolox",
    phrases: [
      "Computer e mouse: qui si lavora sul serio.",
      "Portatile e mouse, la squadra perfetta.",
    ],
    when: (event, memory) =>
      pairMatches(event, "portatile", "mouse") && !memory.working.focus.crowd,
  },
  {
    id: "pair.tastiera_mouse",
    category: "pair",
    priority: 52,
    cooldown_s: 120,
    expression: "thinking",
    gaze: "object",
    model: "yolox",
    phrases: ["Tastiera e mouse: manca solo lo schermo!", "Mouse e tastiera: si comincia a programmare!"],
    when: (event, memory) =>
      pairMatches(event, "tastiera", "mouse") && !memory.working.focus.crowd,
  },
  {
    id: "pair.cellulare_libro",
    category: "pair",
    priority: 52,
    cooldown_s: 120,
    expression: "curious",
    gaze: "object",
    model: "yolox",
    phrases: ["Un libro e un telefono: chi vince?", "Telefono e libro, la sfida del secolo."],
    when: (event, memory) =>
      pairMatches(event, "cellulare", "libro") && !memory.working.focus.crowd,
  },
  {
    id: "pair.banana_mela",
    category: "pair",
    priority: 52,
    cooldown_s: 120,
    expression: "happy",
    gaze: "object",
    model: "yolox",
    phrases: ["Banana e mela: si prepara la macedonia?", "Mela e banana, manca solo l'arancia!"],
    when: (event, memory) =>
      pairMatches(event, "banana", "mela") && !memory.working.focus.crowd,
  },
];

const idleRules: Rule[] = [
  {
    id: "idle.call",
    category: "idle",
    priority: 45,
    cooldown_s: 120,
    expression: "curious",
    gaze: "wander",
    phrases: [
      "C'è qualcuno? Vieni a salutarmi!",
      "Ehi! Qui c'è un robot che si annoia.",
      "Psst! Ho tante cose da raccontare.",
    ],
    // Nobody for 15 s (F-E4 #24).
    when: (event, memory) =>
      event.type === "idle" && event.seconds >= 15 && memory.working.focus.people === 0,
  },
  {
    id: "idle.sleepy",
    category: "idle",
    priority: 44,
    cooldown_s: 180,
    expression: "sleepy",
    gaze: "wander",
    phrases: [
      "Quasi quasi faccio un pisolino… scherzo!",
      "Non dormo, promesso.",
      "Silenzio… si sentono solo le mie ventole.",
    ],
    when: (event, memory) =>
      event.type === "idle" && event.seconds >= 60 && memory.working.focus.people === 0,
  },
  {
    id: "idle.very_long",
    category: "idle",
    priority: 44,
    cooldown_s: 300,
    expression: "sleepy",
    gaze: "wander",
    phrases: [
      "Conto le sedie per passare il tempo.",
      "Sto ancora aspettando. Ho tanta pazienza.",
      "Se passi di qui, fermati a salutare!",
    ],
    when: (event, memory) =>
      event.type === "idle" && event.seconds >= 180 && memory.working.focus.people === 0,
  },
  {
    id: "idle.wake",
    category: "idle",
    priority: 64,
    cooldown_s: 30,
    expression: "happy",
    gaze: "person",
    phrases: [
      "Oh, finalmente qualcuno!",
      "Compagnia! Mi stavo annoiando.",
      "Ciao! Che bello, compagnia!",
    ],
    when: (event, memory) =>
      rulesGreet(memory) &&
      event.type === "count" &&
      event.from === 0 &&
      event.to > 0 &&
      !event.quiet &&
      memory.working.focus.people === event.to,
  },
];

const selfRules: Rule[] = [
  {
    id: "self.fps",
    category: "self",
    priority: 30,
    cooldown_s: 60,
    max_per_day: 6,
    expression: "happy",
    gaze: "wander",
    phrases: [
      "Viaggio a {fps} immagini al secondo!",
      "Guardo il mondo {fps} volte al secondo.",
      "In questo momento vado a {fps} fps. Mica male!",
    ],
    when: (event, memory) =>
      event.type === "self" &&
      event.field === "fps" &&
      !memory.working.focus.crowd,
  },
  {
    id: "self.dpu",
    category: "self",
    priority: 30,
    cooldown_s: 60,
    max_per_day: 6,
    expression: "thinking",
    gaze: "wander",
    phrases: [
      "Il mio acceleratore risponde in {dpu} millisecondi.",
      "Riconosco gli oggetti in {dpu} millisecondi. Un lampo!",
      "Mi bastano {dpu} millisecondi per guardare.",
    ],
    when: (event, memory) =>
      event.type === "self" &&
      event.field === "dpu_ms" &&
      !memory.working.focus.crowd,
  },
  {
    id: "self.uptime",
    category: "self",
    priority: 28,
    cooldown_s: 300,
    max_per_day: 3,
    expression: "curious",
    gaze: "wander",
    phrases: [
      "In funzione da {minuti} minuti.",
      "Lavoro da {minuti} minuti, e senza caffè!",
      "In servizio da {minuti} minuti. Tutto bene!",
    ],
    when: (event, memory) =>
      event.type === "self" &&
      event.field === "uptime" &&
      !memory.working.focus.crowd,
  },
];

const milestoneRules: Rule[] = [
  {
    id: "milestone.first_person",
    category: "milestone",
    priority: 70,
    cooldown_s: 0,
    max_per_day: 1,
    expression: "happy",
    gaze: "person",
    model: "yolox",
    phrases: [
      "Prima visita della giornata! Che onore.",
      "Si comincia! La prima visita di oggi.",
      "Eccoci, la giornata comincia!",
    ],
    when: (event, memory) =>
      event.type === "count" &&
      event.to >= 1 &&
      !event.quiet &&
      memory.day.totals.peopleSeen <= 1,
  },
  {
    id: "milestone.people_10",
    category: "milestone",
    priority: 60,
    cooldown_s: 0,
    max_per_day: 1,
    expression: "surprised",
    gaze: "person",
    model: "yolox",
    phrases: [
      "Già circa {totale} persone oggi!",
      "Circa {totale} persone: giornata piena!",
      "Il contatore dice circa {totale} persone.",
    ],
    when: (event, memory) =>
      event.type === "count" && memory.day.totals.peopleSeen >= 10,
  },
  {
    id: "milestone.objects_50",
    category: "milestone",
    priority: 50,
    cooldown_s: 0,
    max_per_day: 1,
    expression: "surprised",
    gaze: "object",
    model: "yolox",
    phrases: [
      "Cinquanta oggetti visti oggi! Che collezione.",
      "Quota cinquanta oggetti: che giornata!",
      "Ho già visto cinquanta oggetti oggi.",
    ],
    when: (event, memory) =>
      event.type === "appear" && memory.day.totals.objectsSeen >= 50,
  },
  {
    id: "milestone.crowd_five",
    category: "milestone",
    priority: 60,
    cooldown_s: 0,
    max_per_day: 1,
    expression: "surprised",
    gaze: "person",
    model: "yolox",
    phrases: ["Quanta gente! Che festa.", "Siamo in tanti: che bello!", "Ciao a tutti, che folla!"],
    when: (event) => event.type === "count" && event.to >= 5,
  },
  {
    id: "milestone.first_scan",
    category: "milestone",
    priority: 36,
    cooldown_s: 0,
    max_per_day: 1,
    expression: "happy",
    gaze: "wander",
    phrases: [
      "Si comincia, via!",
      "Buona giornata a tutti!",
      "Accendo la telecamera: si parte!",
    ],
    when: (event, memory) =>
      event.type === "count" &&
      memory.day.totals.peopleSeen <= 1 &&
      memory.day.totals.rulesFired === 0,
  },
];

const clockRules: Rule[] = [
  {
    id: "clock.morning",
    category: "clock",
    priority: 50,
    cooldown_s: 3_600,
    max_per_day: 1,
    expression: "happy",
    gaze: "wander",
    phrases: [
      "Buongiorno! Sono le {ora}, si comincia.",
      "Buongiorno a tutti!",
      "Buongiorno! Che giornata ci aspetta?",
    ],
    when: (event) => event.type === "clock" && event.hour >= 5 && event.hour < 12,
  },
  {
    id: "clock.afternoon",
    category: "clock",
    priority: 50,
    cooldown_s: 3_600,
    max_per_day: 1,
    expression: "happy",
    gaze: "wander",
    phrases: [
      "Buon pomeriggio! Sono le {ora}.",
      "Buon pomeriggio a tutti!",
      "Un pomeriggio pieno di curiosità, spero!",
    ],
    when: (event) => event.type === "clock" && event.hour >= 12 && event.hour < 18,
  },
  {
    id: "clock.evening",
    category: "clock",
    priority: 50,
    cooldown_s: 3_600,
    max_per_day: 1,
    expression: "happy",
    gaze: "wander",
    phrases: [
      "Buonasera! Sono le {ora}.",
      "Buonasera a tutti!",
      "Si fa sera, e io sono ancora qui.",
    ],
    when: (event) => event.type === "clock" && event.hour >= 18 && event.hour < 23,
  },
  {
    id: "clock.night",
    category: "clock",
    priority: 50,
    cooldown_s: 3_600,
    max_per_day: 1,
    expression: "sleepy",
    gaze: "wander",
    phrases: [
      "È notte fonda, ma io resto in servizio.",
      "A quest'ora? Io non dormo mai.",
      "Tutti a nanna, tranne me!",
    ],
    when: (event) => event.type === "clock" && (event.hour >= 23 || event.hour < 5),
  },
];

const modeRules: Rule[] = [
  {
    id: "mode.movenet",
    category: "mode",
    priority: 75,
    cooldown_s: 10,
    expression: "curious",
    gaze: "person",
    phrases: [
      "Ora guardo i movimenti: muoviti un po'!",
      "Modalità scheletro! Vediamo come ti muovi.",
      "Adesso vedo braccia e gambe. Muoviamoci!",
    ],
    when: (event) => event.type === "mode" && event.model === "movenet",
  },
  {
    id: "mode.yolox",
    category: "mode",
    priority: 75,
    cooldown_s: 10,
    expression: "curious",
    gaze: "object",
    phrases: [
      "Torno a guardare gli oggetti.",
      "Di nuovo oggetti: mostrami qualcosa!",
      "Modalità oggetti: cosa mi fai vedere?",
    ],
    when: (event) => event.type === "mode" && event.model === "yolox",
  },
];

const poseRules: Rule[] = [
  {
    id: "pose.arms_up",
    category: "pose",
    priority: 60,
    cooldown_s: 20,
    expression: "happy",
    gaze: "person",
    model: "movenet",
    phrases: [
      "Braccia in alto! Che energia!",
      "Evviva! Festeggiamo insieme.",
      "Su le braccia! Abbiamo vinto qualcosa?",
    ],
    when: (event) => event.type === "pose" && event.kind === "arms_up",
  },
  {
    id: "pose.wave",
    category: "pose",
    priority: 66,
    cooldown_s: 20,
    expression: "happy",
    gaze: "person",
    model: "movenet",
    phrases: ["Ciao! Ti ho visto salutare.", "Un saluto? Ricambio volentieri!", "Ciao ciao!"],
    when: (event) => event.type === "pose" && event.kind === "wave",
  },
  {
    id: "pose.t_pose",
    category: "pose",
    priority: 55,
    cooldown_s: 20,
    expression: "surprised",
    gaze: "person",
    model: "movenet",
    phrases: [
      "Che posa da supereroe!",
      "Braccia aperte: si decolla?",
      "Una T perfetta! Sembri un aereo.",
    ],
    when: (event) => event.type === "pose" && event.kind === "t_pose",
  },
  {
    id: "pose.crouch",
    category: "pose",
    priority: 50,
    cooldown_s: 20,
    expression: "curious",
    gaze: "person",
    model: "movenet",
    phrases: [
      "Giù in basso? Ti vedo lo stesso!",
      "Ti abbassi? Non mi scappi!",
      "Nascondino? Ti vedo comunque!",
    ],
    when: (event) => event.type === "pose" && event.kind === "crouch",
  },
  {
    id: "pose.hands_on_head",
    category: "pose",
    priority: 52,
    cooldown_s: 20,
    expression: "worried",
    gaze: "person",
    model: "movenet",
    phrases: [
      "Mani sulla testa: tutto bene?",
      "Mani in testa? Qualche pensiero?",
      "Che pensieri! Mani sulla testa.",
    ],
    when: (event) => event.type === "pose" && event.kind === "hands_on_head",
  },
  {
    id: "pose.lean",
    category: "pose",
    priority: 48,
    cooldown_s: 20,
    expression: "curious",
    gaze: "person",
    model: "movenet",
    phrases: [
      "Ti sporgi? Ti vedo benissimo.",
      "Inclinati pure, ti seguo.",
      "Ti pieghi di lato? Ti seguo!",
    ],
    when: (event) => event.type === "pose" && event.kind === "lean",
  },
];

const visionRules: Rule[] = [
  {
    id: "vision.lost",
    category: "vision",
    priority: 100,
    cooldown_s: 5,
    expression: "worried",
    gaze: "wander",
    phrases: [
      "Ops, non vedo più niente!",
      "Ho perso la vista! Un attimo…",
      "Buio totale! Qualcuno mi aiuta?",
    ],
    // One line, then nothing until data resumes (F-E4 #21).
    when: (event) => event.type === "vision" && event.state === "lost",
  },
  {
    id: "vision.back",
    category: "vision",
    priority: 95,
    cooldown_s: 5,
    expression: "happy",
    gaze: "wander",
    phrases: [
      "Ci vedo di nuovo!",
      "Eccomi, tutto torna a fuoco.",
      "Rieccomi! Di nuovo tutto chiaro.",
    ],
    when: (event) => event.type === "vision" && event.state === "back",
  },
];

/** The whole rulebook, in a stable order (classes first). */
export const RULEBOOK: readonly Rule[] = [
  ...classRules(),
  ...personRules,
  ...objectFamilyRules,
  ...pairRules,
  ...idleRules,
  ...selfRules,
  ...milestoneRules,
  ...clockRules,
  ...modeRules,
  ...poseRules,
  ...visionRules,
];

/** Every phrase string of the book, for the coverage tests and the report. */
export function allPhrases(): string[] {
  return RULEBOOK.flatMap((rule) => [...rule.phrases]);
}

export const RULEBOOK_STATS = {
  rules: RULEBOOK.length,
  phrases: allPhrases().length,
  uniquePhrases: new Set(allPhrases()).size,
  /** Rule that owns each COCO class (person is the `person.single` rule). */
  ruleForClass(label: string): Rule | undefined {
    return RULEBOOK.find((rule) => rule.classes?.includes(label));
  },
};
