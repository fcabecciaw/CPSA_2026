/**
 * Italian grammar for the 80 COCO labels the server publishes (`COCO80_IT` in
 * `src/cpsa/yolox.py`) and the slot renderer of the rulebook phrases.
 *
 * Every class carries its gender, definite article, singular and plural here, so
 * no sentence ever says "il tazza", "le telefono" or "un forbici". The table is
 * also the only place where the plural-only classes (forbici) are special-cased.
 */

/** Grammatical gender of a COCO class name. */
export type Gender = "m" | "f";

export interface NounGrammar {
  /** `null` for plural-only nouns such as "forbici". */
  singular: string | null;
  plural: string;
  /** Definite singular article: "il", "lo", "la", "l'". */
  article: string;
  /** Definite plural article: "i", "gli", "le". */
  pluralArticle: string;
  gender: Gender;
  /** Same word for singular and plural (auto, tv, camion…). */
  invariant?: boolean;
}

/**
 * The 80 Italian labels, in COCO-80 order: index `i` is `class_id` `i` of the
 * server. Kept in sync by a test that reads `COCO80_IT` from the Python source.
 */
export const COCO_IT_LABELS = [
  "persona",
  "bicicletta",
  "auto",
  "motocicletta",
  "aereo",
  "autobus",
  "treno",
  "camion",
  "barca",
  "semaforo",
  "idrante",
  "stop",
  "parchimetro",
  "panchina",
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
  "zaino",
  "ombrello",
  "borsa",
  "cravatta",
  "valigia",
  "frisbee",
  "sci",
  "snowboard",
  "palla",
  "aquilone",
  "mazza da baseball",
  "guantone",
  "skateboard",
  "tavola da surf",
  "racchetta",
  "bottiglia",
  "calice",
  "tazza",
  "forchetta",
  "coltello",
  "cucchiaio",
  "ciotola",
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
  "sedia",
  "divano",
  "pianta",
  "letto",
  "tavolo",
  "water",
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
  "libro",
  "orologio",
  "vaso",
  "forbici",
  "orsacchiotto",
  "phon",
  "spazzolino",
] as const;

export type CocoLabel = (typeof COCO_IT_LABELS)[number];

const NOUNS: Record<CocoLabel, NounGrammar> = {
  persona: { singular: "persona", plural: "persone", article: "la", pluralArticle: "le", gender: "f" },
  bicicletta: { singular: "bicicletta", plural: "biciclette", article: "la", pluralArticle: "le", gender: "f" },
  auto: { singular: "auto", plural: "auto", article: "l'", pluralArticle: "le", gender: "f", invariant: true },
  motocicletta: { singular: "motocicletta", plural: "motociclette", article: "la", pluralArticle: "le", gender: "f" },
  aereo: { singular: "aereo", plural: "aerei", article: "l'", pluralArticle: "gli", gender: "m" },
  autobus: { singular: "autobus", plural: "autobus", article: "l'", pluralArticle: "gli", gender: "m", invariant: true },
  treno: { singular: "treno", plural: "treni", article: "il", pluralArticle: "i", gender: "m" },
  camion: { singular: "camion", plural: "camion", article: "il", pluralArticle: "i", gender: "m", invariant: true },
  barca: { singular: "barca", plural: "barche", article: "la", pluralArticle: "le", gender: "f" },
  semaforo: { singular: "semaforo", plural: "semafori", article: "il", pluralArticle: "i", gender: "m" },
  idrante: { singular: "idrante", plural: "idranti", article: "l'", pluralArticle: "gli", gender: "m" },
  stop: { singular: "stop", plural: "stop", article: "lo", pluralArticle: "gli", gender: "m", invariant: true },
  parchimetro: { singular: "parchimetro", plural: "parchimetri", article: "il", pluralArticle: "i", gender: "m" },
  panchina: { singular: "panchina", plural: "panchine", article: "la", pluralArticle: "le", gender: "f" },
  uccello: { singular: "uccello", plural: "uccelli", article: "l'", pluralArticle: "gli", gender: "m" },
  gatto: { singular: "gatto", plural: "gatti", article: "il", pluralArticle: "i", gender: "m" },
  cane: { singular: "cane", plural: "cani", article: "il", pluralArticle: "i", gender: "m" },
  cavallo: { singular: "cavallo", plural: "cavalli", article: "il", pluralArticle: "i", gender: "m" },
  pecora: { singular: "pecora", plural: "pecore", article: "la", pluralArticle: "le", gender: "f" },
  mucca: { singular: "mucca", plural: "mucche", article: "la", pluralArticle: "le", gender: "f" },
  elefante: { singular: "elefante", plural: "elefanti", article: "l'", pluralArticle: "gli", gender: "m" },
  orso: { singular: "orso", plural: "orsi", article: "l'", pluralArticle: "gli", gender: "m" },
  zebra: { singular: "zebra", plural: "zebre", article: "la", pluralArticle: "le", gender: "f" },
  giraffa: { singular: "giraffa", plural: "giraffe", article: "la", pluralArticle: "le", gender: "f" },
  zaino: { singular: "zaino", plural: "zaini", article: "lo", pluralArticle: "gli", gender: "m" },
  ombrello: { singular: "ombrello", plural: "ombrelli", article: "l'", pluralArticle: "gli", gender: "m" },
  borsa: { singular: "borsa", plural: "borse", article: "la", pluralArticle: "le", gender: "f" },
  cravatta: { singular: "cravatta", plural: "cravatte", article: "la", pluralArticle: "le", gender: "f" },
  valigia: { singular: "valigia", plural: "valigie", article: "la", pluralArticle: "le", gender: "f" },
  frisbee: { singular: "frisbee", plural: "frisbee", article: "il", pluralArticle: "i", gender: "m", invariant: true },
  sci: { singular: "sci", plural: "sci", article: "lo", pluralArticle: "gli", gender: "m", invariant: true },
  snowboard: { singular: "snowboard", plural: "snowboard", article: "lo", pluralArticle: "gli", gender: "m", invariant: true },
  palla: { singular: "palla", plural: "palle", article: "la", pluralArticle: "le", gender: "f" },
  aquilone: { singular: "aquilone", plural: "aquiloni", article: "l'", pluralArticle: "gli", gender: "m" },
  "mazza da baseball": { singular: "mazza da baseball", plural: "mazze da baseball", article: "la", pluralArticle: "le", gender: "f" },
  guantone: { singular: "guantone", plural: "guantoni", article: "il", pluralArticle: "i", gender: "m" },
  skateboard: { singular: "skateboard", plural: "skateboard", article: "lo", pluralArticle: "gli", gender: "m", invariant: true },
  "tavola da surf": { singular: "tavola da surf", plural: "tavole da surf", article: "la", pluralArticle: "le", gender: "f" },
  racchetta: { singular: "racchetta", plural: "racchette", article: "la", pluralArticle: "le", gender: "f" },
  bottiglia: { singular: "bottiglia", plural: "bottiglie", article: "la", pluralArticle: "le", gender: "f" },
  calice: { singular: "calice", plural: "calici", article: "il", pluralArticle: "i", gender: "m" },
  tazza: { singular: "tazza", plural: "tazze", article: "la", pluralArticle: "le", gender: "f" },
  forchetta: { singular: "forchetta", plural: "forchette", article: "la", pluralArticle: "le", gender: "f" },
  coltello: { singular: "coltello", plural: "coltelli", article: "il", pluralArticle: "i", gender: "m" },
  cucchiaio: { singular: "cucchiaio", plural: "cucchiai", article: "il", pluralArticle: "i", gender: "m" },
  ciotola: { singular: "ciotola", plural: "ciotole", article: "la", pluralArticle: "le", gender: "f" },
  banana: { singular: "banana", plural: "banane", article: "la", pluralArticle: "le", gender: "f" },
  mela: { singular: "mela", plural: "mele", article: "la", pluralArticle: "le", gender: "f" },
  panino: { singular: "panino", plural: "panini", article: "il", pluralArticle: "i", gender: "m" },
  arancia: { singular: "arancia", plural: "arance", article: "l'", pluralArticle: "le", gender: "f" },
  broccolo: { singular: "broccolo", plural: "broccoli", article: "il", pluralArticle: "i", gender: "m" },
  carota: { singular: "carota", plural: "carote", article: "la", pluralArticle: "le", gender: "f" },
  "hot dog": { singular: "hot dog", plural: "hot dog", article: "l'", pluralArticle: "gli", gender: "m", invariant: true },
  pizza: { singular: "pizza", plural: "pizze", article: "la", pluralArticle: "le", gender: "f" },
  ciambella: { singular: "ciambella", plural: "ciambelle", article: "la", pluralArticle: "le", gender: "f" },
  torta: { singular: "torta", plural: "torte", article: "la", pluralArticle: "le", gender: "f" },
  sedia: { singular: "sedia", plural: "sedie", article: "la", pluralArticle: "le", gender: "f" },
  divano: { singular: "divano", plural: "divani", article: "il", pluralArticle: "i", gender: "m" },
  pianta: { singular: "pianta", plural: "piante", article: "la", pluralArticle: "le", gender: "f" },
  letto: { singular: "letto", plural: "letti", article: "il", pluralArticle: "i", gender: "m" },
  tavolo: { singular: "tavolo", plural: "tavoli", article: "il", pluralArticle: "i", gender: "m" },
  water: { singular: "water", plural: "water", article: "il", pluralArticle: "i", gender: "m", invariant: true },
  tv: { singular: "tv", plural: "tv", article: "la", pluralArticle: "le", gender: "f", invariant: true },
  portatile: { singular: "portatile", plural: "portatili", article: "il", pluralArticle: "i", gender: "m" },
  mouse: { singular: "mouse", plural: "mouse", article: "il", pluralArticle: "i", gender: "m", invariant: true },
  telecomando: { singular: "telecomando", plural: "telecomandi", article: "il", pluralArticle: "i", gender: "m" },
  tastiera: { singular: "tastiera", plural: "tastiere", article: "la", pluralArticle: "le", gender: "f" },
  cellulare: { singular: "cellulare", plural: "cellulari", article: "il", pluralArticle: "i", gender: "m" },
  microonde: { singular: "microonde", plural: "microonde", article: "il", pluralArticle: "i", gender: "m", invariant: true },
  forno: { singular: "forno", plural: "forni", article: "il", pluralArticle: "i", gender: "m" },
  tostapane: { singular: "tostapane", plural: "tostapane", article: "il", pluralArticle: "i", gender: "m", invariant: true },
  lavandino: { singular: "lavandino", plural: "lavandini", article: "il", pluralArticle: "i", gender: "m" },
  frigorifero: { singular: "frigorifero", plural: "frigoriferi", article: "il", pluralArticle: "i", gender: "m" },
  libro: { singular: "libro", plural: "libri", article: "il", pluralArticle: "i", gender: "m" },
  orologio: { singular: "orologio", plural: "orologi", article: "l'", pluralArticle: "gli", gender: "m" },
  vaso: { singular: "vaso", plural: "vasi", article: "il", pluralArticle: "i", gender: "m" },
  forbici: { singular: null, plural: "forbici", article: "le", pluralArticle: "le", gender: "f" },
  orsacchiotto: { singular: "orsacchiotto", plural: "orsacchiotti", article: "l'", pluralArticle: "gli", gender: "m" },
  phon: { singular: "phon", plural: "phon", article: "il", pluralArticle: "i", gender: "m", invariant: true },
  spazzolino: { singular: "spazzolino", plural: "spazzolini", article: "lo", pluralArticle: "gli", gender: "m" },
};

const NUMBER_WORDS = [
  "zero",
  "una",
  "due",
  "tre",
  "quattro",
  "cinque",
  "sei",
  "sette",
  "otto",
  "nove",
  "dieci",
] as const;

/** Number word for 1..10 ("una" only through agreement, see `nounPhrase`). */
export function numberWord(count: number): string {
  if (Number.isInteger(count) && count >= 0 && count <= 10) {
    return NUMBER_WORDS[count];
  }
  return String(count);
}

const ORDINALS: Record<Gender, readonly string[]> = {
  m: [
    "",
    "primo",
    "secondo",
    "terzo",
    "quarto",
    "quinto",
    "sesto",
    "settimo",
    "ottavo",
    "nono",
    "decimo",
  ],
  f: [
    "",
    "prima",
    "seconda",
    "terza",
    "quarta",
    "quinta",
    "sesta",
    "settima",
    "ottava",
    "nona",
    "decima",
  ],
};

/** Ordinal word agreed with the noun: "terzo cellulare", "terza tazza". */
export function ordinalWord(count: number, gender: Gender): string {
  if (Number.isInteger(count) && count >= 1 && count <= 10) {
    return ORDINALS[gender][count];
  }
  // From 11 on the spoken form is unusual: the digit plus the ordinal mark.
  return `${count}${gender === "f" ? "ª" : "°"}`;
}

/** Upper-case the first letter, for a sentence that starts with a slot. */
export function capitalizeFirst(text: string): string {
  return text.length === 0 ? text : text[0].toUpperCase() + text.slice(1);
}

/** Grammar of a label; unknown labels fall back to a neutral masculine noun. */
export function nounGrammar(label: string): NounGrammar {
  return (
    NOUNS[label as CocoLabel] ?? {
      singular: "oggetto",
      plural: "oggetti",
      article: "un",
      pluralArticle: "gli",
      gender: "m",
    }
  );
}

function joinArticle(article: string, noun: string): string {
  // "l'ombrello", not "l' ombrello".
  return article.endsWith("'") ? `${article}${noun}` : `${article} ${noun}`;
}

/** Definite singular/plural article of a class ("il", "la", "l'", "le"…). */
export function articleFor(label: string, count = 1): string {
  const noun = nounGrammar(label);
  return count > 1 || noun.singular === null ? noun.pluralArticle : noun.article;
}

/**
 * A class with its article, or a number and the plural: "il telefono",
 * "la tazza", "l'ombrello", "le forbici", "due tazze".
 */
export function nounPhrase(label: string, count = 1): string {
  const noun = nounGrammar(label);
  if (count > 1) {
    return `${numberWord(count)} ${noun.plural}`;
  }
  if (noun.singular === null) {
    return joinArticle(noun.pluralArticle, noun.plural);
  }
  return joinArticle(noun.article, noun.singular);
}

/** "persona"/"persone" agreement for group lines ("voi due", "in tre"). */
export function peopleWord(count: number): string {
  return count === 1 ? "persona" : "persone";
}

/**
 * A class with the indefinite article: "un telefono", "una tazza",
 * "uno zaino", "un'auto", "due tazze". Used by the scene continuity lines
 * ("Prima il telefono, ora una tazza").
 */
export function indefinitePhrase(label: string, count = 1): string {
  const noun = nounGrammar(label);
  if (count > 1) {
    return `${numberWord(count)} ${noun.plural}`;
  }
  if (noun.singular === null) {
    return joinArticle(noun.pluralArticle, noun.plural);
  }
  const article =
    noun.gender === "f"
      ? noun.article === "l'"
        ? "un'"
        : "una"
      : noun.article === "lo"
        ? "uno"
        : "un";
  return joinArticle(article, noun.singular);
}

/** Slots a rulebook phrase may use. */
export interface PhraseSlots {
  oggetto?: string;
  oggetto2?: string;
  n?: string | number;
  n2?: string | number;
  ora?: string;
  minuti?: string | number;
  secondi?: string | number;
  fps?: string | number;
  dpu?: string | number;
  /** Day total, used by the "circa" memory lines. */
  totale?: string | number;
  altro?: string;
  /** A whole sentence a scene already rendered (`scene.ts`). */
  testo?: string;
  /** Counting lines: capitalised ordinal, bare noun and agreed totals. */
  Ordinale?: string;
  singolare?: string;
  conteggio?: string;
  plurale?: string;
  oggettoNumero?: string;
  numero?: string;
  volte?: string;
}

/**
 * Slots of the `memory.object_count` family: the day's stable appearance count
 * rendered with the class gender and number ("Terzo cellulare", "Terza tazza",
 * "Terzo paio di forbici", "tre tazze"). Plural-only nouns take "paio di" in
 * the ordinal forms, so no sentence ever says "terze forbici".
 */
export function objectCountSlots(label: string, count: number): PhraseSlots {
  const noun = nounGrammar(label);
  const singular = noun.singular;
  const ordinal =
    singular === null
      ? `${ordinalWord(count, "m")} paio di`
      : ordinalWord(count, noun.gender);
  return {
    Ordinale: capitalizeFirst(ordinal),
    singolare: singular ?? noun.plural,
    conteggio: numberWord(count),
    plurale: noun.plural,
    oggettoNumero:
      singular === null
        ? joinArticle("il", `paio di ${noun.plural}`)
        : joinArticle(noun.article, singular),
    numero: numberWord(count),
    volte: numberWord(count),
  };
}

/** Fill `{slot}` placeholders; unknown slots stay visible for the test suite. */
export function renderPhrase(template: string, slots: PhraseSlots): string {
  return template.replace(/\{(\w+)\}/g, (match, key: string) => {
    const value = (slots as Record<string, unknown>)[key];
    return value === undefined ? match : String(value);
  });
}

/** Word count, used to keep every utterance short (F-E4 #17). */
export function wordCount(text: string): number {
  const trimmed = text.trim();
  return trimmed === "" ? 0 : trimmed.split(/\s+/).length;
}
