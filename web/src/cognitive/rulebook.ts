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
 * Three conversational lines per COCO class. All use `{oggetto}`, which the
 * grammar table renders with the right article or plural ("il telefono",
 * "due tazze", "le forbici"), so no sentence is ever ungrammatical (F-E4 #8).
 * They are short, spoken Italian: an observation, a small question, a warm
 * aside — never a template.
 */
const CLASS_LINES: Record<ObjectLabel, readonly [string, string, string]> = {
  bicicletta: ["Ecco {oggetto}: si va a fare un giro?", "C'è {oggetto} qui: qualcuno ha pedalato fin qui.", "Mi piace {oggetto}: sa di gita."],
  auto: ["Ecco {oggetto}: si parte?", "C'è {oggetto} in vista: viaggio in arrivo.", "Mi ricorda {oggetto}: motore e libertà."],
  motocicletta: ["Ecco {oggetto}: che rombo.", "C'è {oggetto}: pronti al giro?", "Mi piace {oggetto}: sa di strada."],
  aereo: ["Ecco {oggetto}: decollo?", "C'è {oggetto} qui: che voglia di partire.", "Con {oggetto} il cielo sembra vicino."],
  autobus: ["Ecco {oggetto}: prossima fermata?", "C'è {oggetto}: qualcuno aspetta.", "Mi ricorda {oggetto}: si guarda fuori."],
  treno: ["Ecco {oggetto}: in stazione.", "C'è {oggetto}: binario e valigia.", "Con {oggetto} si parte davvero."],
  camion: ["Ecco {oggetto}: che carico.", "C'è {oggetto} in arrivo: largo.", "Mi piace {oggetto}: porta cose."],
  barca: ["Ecco {oggetto}: aria di mare.", "C'è {oggetto}: dov'è l'acqua?", "Con {oggetto} si sogna il largo."],
  semaforo: ["Ecco {oggetto}: verde o rosso?", "C'è {oggetto}: si aspetta un attimo.", "Con {oggetto} tutto si ferma."],
  idrante: ["Ecco {oggetto}: meglio non provarlo.", "C'è {oggetto}: tutto tranquillo.", "Mi ricorda {oggetto}: sicurezza in strada."],
  stop: ["Ecco {oggetto}: mi fermo.", "C'è {oggetto}: un attimo di pausa.", "Con {oggetto} si rallenta."],
  parchimetro: ["Ecco {oggetto}: cerca monete.", "C'è {oggetto}: che precisione.", "Mi ricorda {oggetto}: tempo misurato."],
  panchina: ["Ecco {oggetto}: una pausa?", "C'è {oggetto}: ci si siede volentieri.", "Mi piace {oggetto}: invita a restare."],
  uccello: ["Ecco {oggetto}: che visita.", "C'è {oggetto}: vola piano.", "Mi ricorda {oggetto}: un po' di cielo."],
  gatto: ["Ecco {oggetto}: comanda lui.", "C'è {oggetto}: che carattere.", "Mi piace {oggetto}: fa quello che vuole."],
  cane: ["Ecco {oggetto}: che compagnia.", "C'è {oggetto}: scodinzola.", "Mi ricorda {oggetto}: allegria semplice."],
  cavallo: ["Ecco {oggetto}: che eleganza.", "C'è {oggetto}: senza fretta.", "Mi piace {oggetto}: passo tranquillo."],
  pecora: ["Ecco {oggetto}: dolce e quieta.", "C'è {oggetto}: che calma.", "Mi ricorda {oggetto}: un prato immaginario."],
  mucca: ["Ecco {oggetto}: che sorpresa.", "C'è {oggetto}: pascola immaginaria.", "Mi piace {oggetto}: vita di campagna."],
  elefante: ["Ecco {oggetto}: la stanza si stringe.", "C'è {oggetto}: quasi incredibile.", "Mi ricorda {oggetto}: memoria lunga."],
  orso: ["Ecco {oggetto}: calma, eh.", "C'è {oggetto}: meglio fare amicizia.", "Mi piace {oggetto}: un po' selvatico."],
  zebra: ["Ecco {oggetto}: che fantasia.", "C'è {oggetto}: fuori posto.", "Mi ricorda {oggetto}: righe e sorprese."],
  giraffa: ["Ecco {oggetto}: tocca il soffitto.", "C'è {oggetto}: che altezza.", "Mi piace {oggetto}: guarda tutti dall'alto."],
  zaino: ["Ecco {oggetto}: si parte?", "C'è {oggetto}: pronto per la gita.", "Mi ricorda {oggetto}: gita e libertà."],
  ombrello: ["Ecco {oggetto}: pioggia in arrivo?", "C'è {oggetto}: meglio portarlo.", "Mi piace {oggetto}: sa di giornate grigie."],
  borsa: ["Ecco {oggetto}: dentro c'è un mondo.", "C'è {oggetto}: al suo posto.", "Mi ricorda {oggetto}: storie da portare."],
  cravatta: ["Ecco {oggetto}: che occasione.", "C'è {oggetto}: pronto per la festa.", "Mi piace {oggetto}: un tocco di eleganza."],
  valigia: ["Ecco {oggetto}: si parte o si torna?", "C'è {oggetto}: piena di storie.", "Mi ricorda {oggetto}: viaggi e ritorni."],
  frisbee: ["Ecco {oggetto}: chi lo lancia?", "C'è {oggetto}: fermo per ora.", "Mi piace {oggetto}: voglia di giocare."],
  sci: ["Ecco {oggetto}: neve in arrivo?", "C'è {oggetto}: fuori stagione.", "Mi ricorda {oggetto}: discese e risate."],
  snowboard: ["Ecco {oggetto}: che discesa.", "C'è {oggetto}: aspetta la neve.", "Mi piace {oggetto}: equilibrio e vento."],
  palla: ["Ecco {oggetto}: voglia di giocare.", "C'è {oggetto}: ferma per ora.", "Mi ricorda {oggetto}: partite improvvisate."],
  aquilone: ["Ecco {oggetto}: serve vento.", "C'è {oggetto}: sogna il cielo.", "Mi piace {oggetto}: filo e libertà."],
  "mazza da baseball": ["Ecco {oggetto}: si gioca?", "C'è {oggetto}: pronta al campo.", "Mi ricorda {oggetto}: una bella battuta."],
  guantone: ["Ecco {oggetto}: manca la partita.", "C'è {oggetto}: in attesa.", "Mi piace {oggetto}: prese sicure."],
  skateboard: ["Ecco {oggetto}: equilibrio.", "C'è {oggetto}: pronto a rotolare.", "Mi ricorda {oggetto}: trucchi e risate."],
  "tavola da surf": ["Ecco {oggetto}: arriva un'onda?", "C'è {oggetto}: mare in testa.", "Mi piace {oggetto}: sale e vento."],
  racchetta: ["Ecco {oggetto}: partita a due?", "C'è {oggetto}: pronta al servizio.", "Mi ricorda {oggetto}: scambi veloci."],
  bottiglia: ["Ecco {oggetto}: cin cin virtuale.", "C'è {oggetto}: sul tavolo.", "Mi piace {oggetto}: brindisi in arrivo."],
  calice: ["Ecco {oggetto}: brindisi in vista.", "C'è {oggetto}: vuoto per ora.", "Mi ricorda {oggetto}: una sera speciale."],
  tazza: ["Ecco {oggetto}: caffè o tè?", "C'è {oggetto}: che calore.", "Mi piace {oggetto}: sa di pausa."],
  forchetta: ["Ecco {oggetto}: si mangia?", "C'è {oggetto}: pronta all'uso.", "Mi ricorda {oggetto}: pranzi in compagnia."],
  coltello: ["Ecco {oggetto}: con calma, mi raccomando.", "C'è {oggetto}: usiamolo piano.", "Mi ricorda {oggetto}: meglio con calma."],
  cucchiaio: ["Ecco {oggetto}: minestra in arrivo.", "C'è {oggetto}: pronto all'uso.", "Mi piace {oggetto}: cucchiaiate lente."],
  ciotola: ["Ecco {oggetto}: qualcosa bolle.", "C'è {oggetto}: vuota per ora.", "Mi ricorda {oggetto}: colazioni tranquille."],
  banana: ["Ecco {oggetto}: spuntino perfetto.", "C'è {oggetto}: che fame.", "Mi piace {oggetto}: energia veloce."],
  mela: ["Ecco {oggetto}: salute prima di tutto.", "C'è {oggetto}: rossa e tranquilla.", "Mi ricorda {oggetto}: una merenda semplice."],
  panino: ["Ecco {oggetto}: pausa meritata.", "C'è {oggetto}: che profumo.", "Mi piace {oggetto}: pranzo veloce."],
  arancia: ["Ecco {oggetto}: vitamina in vista.", "C'è {oggetto}: succosa.", "Mi ricorda {oggetto}: profumo d'inverno."],
  broccolo: ["Ecco {oggetto}: coraggio, fa bene.", "C'è {oggetto}: discreto.", "Mi piace {oggetto}: verdure senza storie."],
  carota: ["Ecco {oggetto}: croccante.", "C'è {oggetto}: sana e semplice.", "Mi ricorda {oggetto}: merende di una volta."],
  "hot dog": ["Ecco {oggetto}: festa.", "C'è {oggetto}: che sorpresa.", "Mi piace {oggetto}: sapore di domenica."],
  pizza: ["Ecco {oggetto}: serata perfetta.", "C'è {oggetto}: che meraviglia.", "Mi ricorda {oggetto}: amici e risate."],
  ciambella: ["Ecco {oggetto}: buco goloso.", "C'è {oggetto}: attenzione alle dita.", "Mi piace {oggetto}: colazione decisa."],
  torta: ["Ecco {oggetto}: compleanno?", "C'è {oggetto}: pronta al taglio.", "Mi ricorda {oggetto}: candeline e auguri."],
  sedia: ["Ecco {oggetto}: si accomodi.", "C'è {oggetto}: paziente.", "Mi piace {oggetto}: aspetta qualcuno."],
  divano: ["Ecco {oggetto}: invito al riposo.", "C'è {oggetto}: chiama, resisto.", "Mi ricorda {oggetto}: domeniche lente."],
  pianta: ["Ecco {oggetto}: aria pulita.", "C'è {oggetto}: cresce discreta.", "Mi piace {oggetto}: un po' di verde."],
  letto: ["Ecco {oggetto}: che tentazione.", "C'è {oggetto}: aspetta la buonanotte.", "Mi ricorda {oggetto}: sonni tranquilli."],
  tavolo: ["Ecco {oggetto}: ci si ritrova.", "C'è {oggetto}: pronto alla compagnia.", "Mi piace {oggetto}: storie intorno."],
  water: ["Ecco {oggetto}: tutto sotto controllo.", "C'è {oggetto}: che discrezione.", "Mi ricorda {oggetto}: la vita quotidiana."],
  tv: ["Ecco {oggetto}: che si guarda?", "C'è {oggetto}: spenta per ora.", "Mi piace {oggetto}: storie in salotto."],
  portatile: ["Ecco {oggetto}: lavoro o svago?", "C'è {oggetto}: aperto sul mondo.", "Mi ricorda {oggetto}: giornate piene."],
  mouse: ["Ecco {oggetto}: clic o doppio clic?", "C'è {oggetto}: fermo.", "Mi piace {oggetto}: piccoli gesti."],
  telecomando: ["Ecco {oggetto}: il potere in mano.", "C'è {oggetto}: che responsabilità.", "Mi ricorda {oggetto}: zapping serale."],
  tastiera: ["Ecco {oggetto}: chi scrive?", "C'è {oggetto}: aspetta parole.", "Mi piace {oggetto}: storie che nascono."],
  cellulare: ["Ecco {oggetto}: notifiche in arrivo.", "C'è {oggetto}: sempre connesso.", "Mi ricorda {oggetto}: un mondo in tasca."],
  microonde: ["Ecco {oggetto}: piatto caldo in tre minuti.", "C'è {oggetto}: efficiente.", "Mi piace {oggetto}: pause veloci."],
  forno: ["Ecco {oggetto}: che profumo.", "C'è {oggetto}: in attesa.", "Mi ricorda {oggetto}: domeniche di casa."],
  tostapane: ["Ecco {oggetto}: pane croccante.", "C'è {oggetto}: ottimista.", "Mi piace {oggetto}: colazioni veloci."],
  lavandino: ["Ecco {oggetto}: acqua e sapone.", "C'è {oggetto}: pulito.", "Mi ricorda {oggetto}: piccoli riti."],
  frigorifero: ["Ecco {oggetto}: che fame dentro.", "C'è {oggetto}: custode dei snack.", "Mi piace {oggetto}: sorprese fredde."],
  libro: ["Ecco {oggetto}: storie in arrivo.", "C'è {oggetto}: che invito.", "Mi piace {oggetto}: mondi da aprire."],
  orologio: ["Ecco {oggetto}: il tempo passa.", "C'è {oggetto}: sempre puntuale.", "Mi ricorda {oggetto}: giornate che corrono."],
  vaso: ["Ecco {oggetto}: fiori o niente?", "C'è {oggetto}: elegante.", "Mi piace {oggetto}: un tocco di casa."],
  forbici: ["Ecco {oggetto}: con calma, mi raccomando.", "C'è {oggetto}: meglio piano.", "Mi ricorda {oggetto}: lavori delicati."],
  orsacchiotto: ["Ecco {oggetto}: abbraccio garantito.", "C'è {oggetto}: dolce.", "Mi piace {oggetto}: tenerezza semplice."],
  phon: ["Ecco {oggetto}: che ventilata.", "C'è {oggetto}: rumoroso.", "Mi ricorda {oggetto}: mattine di fretta."],
  spazzolino: ["Ecco {oggetto}: igiene al top.", "C'è {oggetto}: mattiniero.", "Mi piace {oggetto}: routine tranquilla."],
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
