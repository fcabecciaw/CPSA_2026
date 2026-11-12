/**
 * Playful yes/no questions for the Play scene.
 *
 * The robot asks one, then waits for a nod or a head shake (MediaPipe face
 * cues) and answers with the matching line. Questions never repeat within a
 * day; the two branches are written to sound like a person, not a form.
 */

export interface Question {
  readonly id: string;
  readonly text: string;
  /** Said when the person nods. */
  readonly yes: readonly string[];
  /** Said when the person shakes their head. */
  readonly no: readonly string[];
}

export const QUESTIONS: readonly Question[] = [
  {
    id: "q-tech",
    text: "Ti piace la tecnologia?",
    yes: ["Lo sapevo!", "Allora sei nel posto giusto.", "Evviva! Parliamo la stessa lingua."],
    no: ["Nessun problema, magari ti convinco io.", "Va bene. Ma almeno un pochino?"],
  },
  {
    id: "q-robot-chat",
    text: "Hai mai parlato con un robot?",
    yes: ["Allora sai già come funziona.", "E io come me la cavo, rispetto agli altri?"],
    no: ["Allora è il tuo primo robot! Che onore.", "Allora segnati la data: oggi!"],
  },
  {
    id: "q-fun",
    text: "Ti stai divertendo?",
    yes: ["Evviva! Anch'io.", "Allora il mio lavoro funziona."],
    no: ["Ahi. Allora devo impegnarmi di più.", "Aspetta, ho ancora qualche asso nella manica."],
  },
  {
    id: "q-robot",
    text: "Ti piacciono i robot?",
    yes: ["Grazie! Mi fai arrossire i pixel.", "Che bello! A me piacciono le persone."],
    no: ["Pazienza, mi impegnerò di più.", "Ahi. Proverò a farti cambiare idea."],
  },
  {
    id: "q-music",
    text: "Ti piace la musica?",
    yes: ["Anche a me! Peccato che io non sappia cantare.", "Allora immagina che io canticchi."],
    no: ["Va bene, niente concerto allora.", "Meglio il silenzio? Lo rispetto."],
  },
  {
    id: "q-build",
    text: "Ti piacerebbe costruire un robot come me?",
    yes: ["Allora un giorno lavoreremo insieme!", "Ottimo! Qui all'università si impara proprio questo."],
    no: ["Giusto, un robot qui basta e avanza.", "Va bene, puoi sempre venirmi a trovare."],
  },
  {
    id: "q-maths",
    text: "Ti piace la matematica?",
    yes: ["Allora andiamo d'accordo: io vivo di numeri.", "Perfetto! Io faccio conti tutto il giorno."],
    no: ["Non dirlo forte: dentro di me ci sono solo numeri!", "Peccato, per me tutto è un numero."],
  },
];
