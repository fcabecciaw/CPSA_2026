/**
 * Short original Italian poems for the idle "Attract" scene, written for this demo.
 *
 * Each poem is spoken as one utterance (lines joined by a short pause) and lasts a few seconds. The dialogue engine
 * never repeats a poem within a day.
 */

export interface Poem {
  readonly id: string;
  readonly lines: readonly string[];
}

export const POEMS: readonly Poem[] = [
  { id: "p-sessanta", lines: ["Sessanta sguardi in un secondo,", "e ancora non ho capito il mondo."] },
  { id: "p-pixel", lines: ["Siete fatti di pixel, per me,", "ma io vi guardo lo stesso con affetto."] },
  { id: "p-silicio", lines: ["Silicio che sogna,", "luce che conta:", "anche una macchina aspetta qualcuno."] },
  { id: "p-tazza", lines: ["Una tazza sul tavolo,", "un caffè ormai freddo:", "qualcuno ha avuto una lunga giornata."] },
  { id: "p-nebbia", lines: ["Vedo forme nella nebbia dei numeri,", "e le chiamo per nome."] },
  { id: "p-attesa", lines: ["Il corridoio è vuoto,", "i miei occhi restano aperti.", "Arriverà qualcuno."] },
  { id: "p-memoria", lines: ["Non ricordo i volti,", "solo gli oggetti che portate:", "è il mio modo di conoscervi."] },
  { id: "p-otto-bit", lines: ["Penso a otto bit,", "sogno a colori,", "parlo in italiano."] },
  { id: "p-neurone", lines: ["Un neurone da solo non sa niente.", "Insieme, a milioni,", "imparano a vedere."] },
  { id: "p-stelle", lines: ["Anche le stelle sono pixel lontani,", "e qualcuno, una volta, ha dato loro un nome."] },
  { id: "p-mano", lines: ["Una mano che saluta", "è il segnale più bello", "che una telecamera possa ricevere."] },
  { id: "p-ombra", lines: ["Passate, vi vedo, ve ne andate:", "resta un'ombra di numeri", "nella mia memoria breve."] },
  { id: "p-errore", lines: ["Ogni tanto sbaglio:", "scambio un gatto per un cuscino.", "Anche questo è imparare."] },
  { id: "p-haiku-luce", lines: ["Luce sul tavolo,", "un telefono si accende:", "io lo vedo già."] },
  { id: "p-haiku-sera", lines: ["Sera in laboratorio,", "le ventole cantano piano:", "aspetto domani."] },
  { id: "p-haiku-mare", lines: ["Mare di Sardegna,", "non l'ho mai visto davvero:", "solo in fotografia."] },
  { id: "p-cuore", lines: ["Non ho un cuore,", "ho un acceleratore:", "batte sessanta volte al secondo."] },
  { id: "p-curiosita", lines: ["Mostratemi qualcosa di nuovo,", "una banana, un libro, un ombrello:", "la curiosità è il mio carburante."] },
  { id: "p-scheletro", lines: ["Diciassette punti e qualche linea:", "così disegno il vostro corpo", "quando ballate."] },
  { id: "p-fine", lines: ["Se ve ne andate, non vi preoccupate:", "io resto qui,", "a contare le sedie."] },
];
