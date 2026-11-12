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
    yes: ["Lo sapevo!", "Sapevo di piacerti.", "Allora parliamo la stessa lingua."],
    no: ["Meglio la semplicità, giusto?", "Rispetto la tua scelta.", "Va bene, resto comunque simpatico."],
  },
  {
    id: "q-coffee",
    text: "Sei qui per un caffè?",
    yes: ["Allora siamo in due ad aspettarlo.", "Ottima idea, quasi quasi lo prendo anch'io."],
    no: ["Pazienza, sarà per la prossima.", "Niente caffè, allora niente pausa."],
  },
  {
    id: "q-day",
    text: "Tutto bene oggi?",
    yes: ["Mi fa piacere.", "Bene, allora continuiamo così."],
    no: ["Mi dispiace. Ti tengo compagnia.", "Giornata storta? Resto qui, allora."],
  },
  {
    id: "q-robot",
    text: "Ti sto simpatico?",
    yes: ["Grazie! Mi fai arrossire i pixel.", "Sapevo di piacerti!"],
    no: ["Pazienza, mi impegnerò.", "Mi rifarò con la prossima poesia."],
  },
  {
    id: "q-music",
    text: "Ti piace la musica?",
    yes: ["Allora immagina che io canticchi.", "Anche a me, a modo mio."],
    no: ["Silenzio, allora. Lo rispetto.", "Nessun problema: parlerò io."],
  },
];
