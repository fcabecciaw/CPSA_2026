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
 * Two class-specific lines per object class. Both use `{oggetto}`, which the
 * grammar table renders with the right article or plural ("il telefono",
 * "due tazze", "le forbici"), so no sentence is ever ungrammatical (F-E4 #8).
 */
const CLASS_LINES: Record<ObjectLabel, readonly [string, string]> = {
  bicicletta: ["Ecco {oggetto}: si pedala?", "Vedo {oggetto}: che viaggio."],
  auto: ["Ecco {oggetto}: parcheggio?", "Vedo {oggetto}: si parte?"],
  motocicletta: ["Ecco {oggetto}: che rombo.", "Vedo {oggetto}: pronti al giro?"],
  aereo: ["Ecco {oggetto}: decollo?", "Vedo {oggetto}: che viaggio."],
  autobus: ["Ecco {oggetto}: si sale?", "Vedo {oggetto}: prossima fermata."],
  treno: ["Ecco {oggetto}: in stazione.", "Vedo {oggetto}: si parte."],
  camion: ["Ecco {oggetto}: che carico.", "Vedo {oggetto}: in arrivo."],
  barca: ["Ecco {oggetto}: aria di mare.", "Vedo {oggetto}: senza onde."],
  semaforo: ["Ecco {oggetto}: verde o rosso?", "Vedo {oggetto}: si aspetta."],
  idrante: ["Ecco {oggetto}: meglio non provarlo.", "Vedo {oggetto}: tutto tranquillo."],
  stop: ["Ecco {oggetto}: mi fermo.", "Vedo {oggetto}: ma resto qui."],
  parchimetro: ["Ecco {oggetto}: cerca monete.", "Vedo {oggetto}: che precisione."],
  panchina: ["Ecco {oggetto}: una pausa?", "Vedo {oggetto}: ci si siede."],
  uccello: ["Ecco {oggetto}: che visita.", "Vedo {oggetto}: vola piano."],
  gatto: ["Ecco {oggetto}: comanda lui.", "Vedo {oggetto}: che carattere."],
  cane: ["Ecco {oggetto}: che compagnia.", "Vedo {oggetto}: scodinzola."],
  cavallo: ["Ecco {oggetto}: che eleganza.", "Vedo {oggetto}: senza fretta."],
  pecora: ["Ecco {oggetto}: dolce e quieta.", "Vedo {oggetto}: che calma."],
  mucca: ["Ecco {oggetto}: che sorpresa.", "Vedo {oggetto}: pascola immaginaria."],
  elefante: ["Ecco {oggetto}: la stanza si stringe.", "Vedo {oggetto}: quasi incredibile."],
  orso: ["Ecco {oggetto}: calma, eh.", "Vedo {oggetto}: meglio sorridere."],
  zebra: ["Ecco {oggetto}: che fantasia.", "Vedo {oggetto}: fuori posto."],
  giraffa: ["Ecco {oggetto}: tocca il soffitto.", "Vedo {oggetto}: che altezza."],
  zaino: ["Ecco {oggetto}: si parte?", "Vedo {oggetto}: pronto per la gita."],
  ombrello: ["Ecco {oggetto}: pioggia in arrivo?", "Vedo {oggetto}: meglio portarlo."],
  borsa: ["Ecco {oggetto}: dentro c'è un mondo.", "Vedo {oggetto}: al suo posto."],
  cravatta: ["Ecco {oggetto}: che occasione.", "Vedo {oggetto}: pronto per la festa."],
  valigia: ["Ecco {oggetto}: si parte o si torna?", "Vedo {oggetto}: piena di storie."],
  frisbee: ["Ecco {oggetto}: chi lo lancia?", "Vedo {oggetto}: fermo per ora."],
  sci: ["Ecco {oggetto}: neve in arrivo?", "Vedo {oggetto}: fuori stagione."],
  snowboard: ["Ecco {oggetto}: che discesa.", "Vedo {oggetto}: aspetta la neve."],
  palla: ["Ecco {oggetto}: voglia di giocare.", "Vedo {oggetto}: ferma per ora."],
  aquilone: ["Ecco {oggetto}: serve vento.", "Vedo {oggetto}: sogna il cielo."],
  "mazza da baseball": ["Ecco {oggetto}: si gioca?", "Vedo {oggetto}: pronta al campo."],
  guantone: ["Ecco {oggetto}: manca la partita.", "Vedo {oggetto}: in attesa."],
  skateboard: ["Ecco {oggetto}: equilibrio.", "Vedo {oggetto}: pronto a rotolare."],
  "tavola da surf": ["Ecco {oggetto}: arriva un'onda?", "Vedo {oggetto}: mare in testa."],
  racchetta: ["Ecco {oggetto}: partita a due?", "Vedo {oggetto}: pronta al servizio."],
  bottiglia: ["Ecco {oggetto}: cin cin virtuale.", "Vedo {oggetto}: sul tavolo."],
  calice: ["Ecco {oggetto}: brindisi in vista.", "Vedo {oggetto}: vuoto per ora."],
  tazza: ["Ecco {oggetto}: caffè o tè?", "Vedo {oggetto}: che calore."],
  forchetta: ["Ecco {oggetto}: si mangia?", "Vedo {oggetto}: pronta all'uso."],
  coltello: ["Ecco {oggetto}: con calma, mi raccomando.", "Vedo {oggetto}: usiamolo piano."],
  cucchiaio: ["Ecco {oggetto}: minestra in arrivo.", "Vedo {oggetto}: pronto all'uso."],
  ciotola: ["Ecco {oggetto}: qualcosa bolle.", "Vedo {oggetto}: vuota per ora."],
  banana: ["Ecco {oggetto}: spuntino perfetto.", "Vedo {oggetto}: che fame."],
  mela: ["Ecco {oggetto}: salute prima di tutto.", "Vedo {oggetto}: rossa e tranquilla."],
  panino: ["Ecco {oggetto}: pausa meritata.", "Vedo {oggetto}: che profumo."],
  arancia: ["Ecco {oggetto}: vitamina in vista.", "Vedo {oggetto}: succosa."],
  broccolo: ["Ecco {oggetto}: coraggio, fa bene.", "Vedo {oggetto}: discreto."],
  carota: ["Ecco {oggetto}: croccante.", "Vedo {oggetto}: sana e semplice."],
  "hot dog": ["Ecco {oggetto}: festa.", "Vedo {oggetto}: che sorpresa."],
  pizza: ["Ecco {oggetto}: serata perfetta.", "Vedo {oggetto}: che meraviglia."],
  ciambella: ["Ecco {oggetto}: buco goloso.", "Vedo {oggetto}: attenzione alle dita."],
  torta: ["Ecco {oggetto}: compleanno?", "Vedo {oggetto}: pronta al taglio."],
  sedia: ["Ecco {oggetto}: si accomodi.", "Vedo {oggetto}: paziente."],
  divano: ["Ecco {oggetto}: invito al riposo.", "Vedo {oggetto}: chiama, resisto."],
  pianta: ["Ecco {oggetto}: aria pulita.", "Vedo {oggetto}: cresce discreta."],
  letto: ["Ecco {oggetto}: che tentazione.", "Vedo {oggetto}: aspetta la buonanotte."],
  tavolo: ["Ecco {oggetto}: ci si ritrova.", "Vedo {oggetto}: pronto alla compagnia."],
  water: ["Ecco {oggetto}: tutto sotto controllo.", "Vedo {oggetto}: che discrezione."],
  tv: ["Ecco {oggetto}: che si guarda?", "Vedo {oggetto}: spenta per ora."],
  portatile: ["Ecco {oggetto}: lavoro o svago?", "Vedo {oggetto}: aperto sul mondo."],
  mouse: ["Ecco {oggetto}: clic o doppio clic?", "Vedo {oggetto}: fermo."],
  telecomando: ["Ecco {oggetto}: il potere in mano.", "Vedo {oggetto}: che responsabilità."],
  tastiera: ["Ecco {oggetto}: chi scrive?", "Vedo {oggetto}: aspetta parole."],
  cellulare: ["Ecco {oggetto}: notifiche in arrivo.", "Vedo {oggetto}: sempre connesso."],
  microonde: ["Ecco {oggetto}: piatto caldo in tre minuti.", "Vedo {oggetto}: efficiente."],
  forno: ["Ecco {oggetto}: che profumo.", "Vedo {oggetto}: in attesa."],
  tostapane: ["Ecco {oggetto}: pane croccante.", "Vedo {oggetto}: ottimista."],
  lavandino: ["Ecco {oggetto}: acqua e sapone.", "Vedo {oggetto}: pulito."],
  frigorifero: ["Ecco {oggetto}: che fame dentro.", "Vedo {oggetto}: custode dei snack."],
  libro: ["Ecco {oggetto}: storie in arrivo.", "Vedo {oggetto}: che invito."],
  orologio: ["Ecco {oggetto}: il tempo passa.", "Vedo {oggetto}: sempre puntuale."],
  vaso: ["Ecco {oggetto}: fiori o niente?", "Vedo {oggetto}: elegante."],
  forbici: ["Ecco {oggetto}: con calma, mi raccomando.", "Vedo {oggetto}: meglio piano."],
  orsacchiotto: ["Ecco {oggetto}: abbraccio garantito.", "Vedo {oggetto}: dolce."],
  phon: ["Ecco {oggetto}: che ventilata.", "Vedo {oggetto}: rumoroso."],
  spazzolino: ["Ecco {oggetto}: igiene al top.", "Vedo {oggetto}: mattiniero."],
};

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
        phrases: CLASS_LINES[label],
        when: (event: CognitiveEvent, memory: CognitiveMemory): boolean =>
          event.type === "appear" &&
          event.label === label &&
          !event.scenery &&
          !event.doubtful &&
          !memory.working.focus.crowd &&
          memory.working.focus.mode === "yolox",
      };
    });
}

function pairMatches(event: CognitiveEvent, a: string, b: string): boolean {
  return (
    event.type === "together" &&
    ((event.a === a && event.b === b) || (event.a === b && event.b === a))
  );
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
    phrases: ["Ciao! Ti vedo.", "Ciao! Che bello vederti.", "Ehi, ciao!"],
    // Exactly one person, direct "tu" family (F-E4 #2).
    when: (event, memory) =>
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
    phrases: ["Ciao a tutti e due!", "Eccovi! Siete in due.", "Ciao voi due!"],
    when: (event, memory) =>
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
      "Che compagnia! Ciao a tutti.",
      "Ciao gruppo! Piacere di vedervi.",
    ],
    when: (event, memory) =>
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
      "Ciao a tutti! Che bella compagnia.",
      "Ciao folla! Mi fate compagnia.",
      "Benvenuti! Qui c'è posto per tutti.",
    ],
    // Crowd mode: group-level lines only (F-E4 #2).
    when: (event, memory) =>
      event.type === "count" &&
      event.to >= 5 &&
      event.from < 5 &&
      !event.quiet &&
      memory.working.focus.people === event.to,
  },
  {
    id: "person.leave",
    category: "person",
    priority: 55,
    cooldown_s: 20,
    expression: "neutral",
    gaze: "wander",
    model: "yolox",
    phrases: ["A presto!", "Ciao ciao, alla prossima.", "A dopo! Io resto qui."],
    when: (event) => event.type === "count" && event.to === 0 && event.from > 0,
  },
  {
    id: "person.return",
    category: "person",
    priority: 82,
    cooldown_s: 60,
    expression: "happy",
    gaze: "person",
    model: "yolox",
    phrases: [
      "Eccoti di nuovo! Che piacere.",
      "Di nuovo qui! Ti aspettavo.",
      "Eccoti! La stanza era vuota.",
    ],
    // Only after ≥ 30 s away: no identity claim, just the count (#4/#5).
    when: (event, memory) =>
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
    phrases: ["Sei ancora lì? Mi fa piacere.", "Ancora qui? Ottima compagnia.", "Non ti annoi, vero?"],
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
      "Sei ancora qui! Resisto anch'io.",
      "Che costanza! Io non mi muovo.",
      "Un'altra ora insieme, allora.",
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
      "Forse c'è {oggetto}: non ne sono certo.",
      "Vedo {oggetto}? Meglio non insistere.",
      "Sembra che ci sia {oggetto}… ma forse mi sbaglio.",
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
    phrases: ["Ecco di nuovo {oggetto}.", "Riecco {oggetto}!", "{oggetto} è tornato."],
    when: (event, memory) =>
      event.type === "return" &&
      event.label !== "persona" &&
      !memory.working.focus.crowd,
  },
  {
    id: "memory.first_object",
    category: "memory",
    priority: 62,
    cooldown_s: 30,
    per_object_cooldown_s: 120,
    expression: "happy",
    gaze: "object",
    model: "yolox",
    phrases: [
      "Prima volta oggi per {oggetto}: benvenuto.",
      "Novità di oggi: {oggetto}.",
      "Oggi è la prima volta che vedo {oggetto}.",
    ],
    // `first_seen` is synthesised only when the day memory has no class yet.
    when: (event, memory) =>
      event.type === "first_seen" &&
      event.label !== "persona" &&
      !memory.working.focus.crowd &&
      memory.working.focus.mode === "yolox",
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
      "Circa {totale} persone oggi: che via vai.",
      "Il conteggio dice circa {totale} persone.",
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
      "Vedo {oggetto} da un po': tutto bene?",
      "{oggetto} resta qui: che costanza.",
      "Ancora {oggetto}: non si muove.",
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
    phrases: ["Tazza e libro: pausa perfetta.", "Libro e tazza: che quiete."],
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
    phrases: ["Il telefono vicino a te: notifiche?", "Cellulare a portata: che comodità."],
    when: (event, memory) =>
      pairMatches(event, "persona", "cellulare") && !memory.working.focus.crowd,
  },
  {
    id: "pair.bottiglia_calice",
    category: "pair",
    priority: 54,
    cooldown_s: 120,
    expression: "happy",
    gaze: "object",
    model: "yolox",
    phrases: ["Bottiglia e calice: brindisi pronto.", "Calice e bottiglia: cin cin."],
    when: (event, memory) =>
      pairMatches(event, "bottiglia", "calice") && !memory.working.focus.crowd,
  },
  {
    id: "pair.tavolo_tazza",
    category: "pair",
    priority: 52,
    cooldown_s: 120,
    expression: "curious",
    gaze: "object",
    model: "yolox",
    phrases: ["Tazza sul tavolo: tutto al suo posto.", "Tavolo e tazza: ordine perfetto."],
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
    phrases: ["Computer e mouse: si lavora.", "Mouse e portatile: che squadra."],
    when: (event, memory) =>
      pairMatches(event, "portatile", "mouse") && !memory.working.focus.crowd,
  },
  {
    id: "pair.pizza_bottiglia",
    category: "pair",
    priority: 54,
    cooldown_s: 120,
    expression: "happy",
    gaze: "object",
    model: "yolox",
    phrases: ["Pizza e bibita: serata fatta.", "Bottiglia e pizza: che festa."],
    when: (event, memory) =>
      pairMatches(event, "pizza", "bottiglia") && !memory.working.focus.crowd,
  },
  {
    id: "pair.tv_divano",
    category: "pair",
    priority: 50,
    cooldown_s: 120,
    expression: "sleepy",
    gaze: "object",
    model: "yolox",
    phrases: ["Divano e tv: riposo programmato.", "Tv e divano: che tentazione."],
    when: (event, memory) =>
      pairMatches(event, "tv", "divano") && !memory.working.focus.crowd,
  },
  {
    id: "pair.gatto_persona",
    category: "pair",
    priority: 58,
    cooldown_s: 120,
    expression: "happy",
    gaze: "person",
    model: "yolox",
    phrases: ["Il gatto ti ha scelto: che onore.", "Gatto vicino: compagnia di lusso."],
    when: (event, memory) =>
      pairMatches(event, "persona", "gatto") && !memory.working.focus.crowd,
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
    phrases: ["Ci sei? Non vedo nessuno.", "Ehi, c'è qualcuno?", "Mi sento un po' solo qui."],
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
      "Mi addormento? No, sto aspettando.",
      "Il silenzio è lungo, ma resisto.",
      "Sveglio, sempre sveglio. Quasi.",
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
      "Il silenzio mi piace, ma non così tanto.",
      "Torno a contare le mattonelle.",
      "Se qualcuno passa, sono qui.",
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
    phrases: ["Eccoti! Mi stavo annoiando.", "Finalmente qualcuno!", "Ti aspettavo, davvero."],
    when: (event, memory) =>
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
    phrases: ["Giro a {fps} fps: mica male.", "Viaggio a {fps} fps, senza freni.", "Faccio {fps} fps: che ritmo."],
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
      "La DPU risponde in {dpu} millisecondi.",
      "Inferenza in {dpu} ms: veloce.",
      "{dpu} ms per sguardo: non male.",
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
      "Sono in servizio da {minuti} minuti.",
      "Sveglio da {minuti} minuti, senza caffè.",
      "In turno da {minuti} minuti: tutto bene.",
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
    phrases: ["Prima persona di oggi: che onore.", "Il primo incontro della giornata!", "Ecco la prima visita di oggi."],
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
      "Oggi circa {totale} persone: che via vai.",
      "Dieci visite circa: giornata piena.",
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
      "Cinquanta oggetti oggi: che collezione.",
      "Mezzo centinaio di oggetti: ottimo lavoro.",
      "Il contatore degli oggetti è a quota cinquanta.",
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
    phrases: ["Cinque o più persone: che festa.", "Modalità folla attiva: ciao a tutti.", "Siamo in tanti: che bello."],
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
    phrases: ["Primo sguardo della giornata: ci sono.", "Buona giornata a me e a voi.", "Si comincia: ci sono anch'io."],
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
      "Buongiorno! Sono le {ora}: cominciamo.",
      "Buongiorno, le {ora}: che giornata.",
      "Mattina, ore {ora}: pronti?",
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
    phrases: ["Buon pomeriggio! Ore {ora}.", "Le {ora}: pomeriggio tranquillo.", "Buon pomeriggio, che si dice?"],
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
    phrases: ["Buonasera! Ore {ora}.", "Sera, ore {ora}: si rallenta.", "Buonasera, che bella ora."],
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
      "Notte fonda, ore {ora}: sono sveglio.",
      "A quest'ora? Io non dormo mai.",
      "Le {ora}: la notte è nostra.",
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
    phrases: ["Passo alla modalità postura.", "Ora guardo le pose: che movimento.", "Modalità scheletro attiva."],
    when: (event) => event.type === "mode" && event.model === "movenet",
  },
  {
    id: "mode.yolox",
    category: "mode",
    priority: 75,
    cooldown_s: 10,
    expression: "curious",
    gaze: "object",
    phrases: ["Torno a guardare gli oggetti.", "Modalità oggetti di nuovo attiva.", "Ora guardo gli oggetti."],
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
    phrases: ["Braccia in alto! Ti seguo.", "Che energia! Alza pure.", "Braccia su: ottimo umore."],
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
    phrases: ["Ciao! Ti ho visto salutare.", "Un saluto per me? Ricambio.", "Ciao ciao! Bel gesto."],
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
    phrases: ["Che posa da supereroe!", "Braccia aperte: che presenza.", "T-pose: pronto al volo."],
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
    phrases: ["Ti sei abbassato: ti seguo.", "Giù in basso? Nessun problema.", "Accovacciato: che mossa."],
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
    phrases: ["Mani sulla testa: che succede?", "Tutto bene lassù?", "Mani in alto… sulla testa!"],
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
    phrases: ["Ti stai sporgendo: ti vedo.", "Inclina pure, ti seguo.", "Che curva! Resto con te."],
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
    phrases: ["Ho perso la vista!", "Non vedo più nulla!", "Ehi, mi è caduto lo sguardo."],
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
    phrases: ["Ci vedo di nuovo!", "Tutto torna a fuoco.", "Rieccomi a vedere."],
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
