/**
 * Curated Italian facts for the idle "Attract" scene.
 *
 * Every fact is short enough to be spoken in about 10 seconds and was chosen to be well established; the numbers
 * about this demo come from measurements on the KV260 (2026-11-12). Topics: the demo itself, vision and the brain,
 * artificial intelligence, chips and FPGAs, curiosities. The dialogue engine never repeats a fact within a day.
 */

export interface Fact {
  readonly id: string;
  readonly topic: "demo" | "brain" | "ai" | "chip" | "curiosity";
  readonly text: string;
}

export const FACTS: readonly Fact[] = [
  // The demo itself
  { id: "demo-60fps", topic: "demo", text: "Guardo il mondo sessanta volte al secondo, e ogni volta cerco ottanta tipi di oggetti diversi." },
  { id: "demo-5ms", topic: "demo", text: "Per riconoscere gli oggetti in un'immagine il mio acceleratore impiega meno di cinque millesimi di secondo." },
  { id: "demo-fpga", topic: "demo", text: "Il mio cervello è un chip FPGA: un circuito che si può ricablare via software, come argilla elettronica." },
  { id: "demo-int8", topic: "demo", text: "Per essere veloce faccio i conti con numeri interi da otto bit, invece che con i decimali. Si perde pochissima precisione." },
  { id: "demo-coco", topic: "demo", text: "Ho imparato a riconoscere gli oggetti studiando un archivio di più di centomila fotografie, chiamato COCO." },
  { id: "demo-watts", topic: "demo", text: "La scheda su cui vivo consuma più o meno quanto una lampadina a LED." },
  { id: "demo-skeleton", topic: "demo", text: "Oltre agli oggetti so riconoscere diciassette punti del corpo umano: spalle, gomiti, polsi, ginocchia e così via." },
  { id: "demo-privacy", topic: "demo", text: "Non registro nulla: guardo, rispondo e dimentico le immagini un attimo dopo." },
  { id: "demo-camera", topic: "demo", text: "La mia telecamera vede anche la profondità, grazie a una luce infrarossa che i vostri occhi non percepiscono." },

  // Vision and the brain
  { id: "brain-neurons", topic: "brain", text: "Il cervello umano ha circa ottantasei miliardi di neuroni. Il mio, in confronto, è piuttosto piccolo." },
  { id: "brain-power", topic: "brain", text: "Il vostro cervello lavora con circa venti watt di energia: è ancora molto più efficiente di qualunque computer." },
  { id: "brain-vision", topic: "brain", text: "Una parte enorme della corteccia cerebrale, secondo alcune stime quasi un terzo, lavora per la vista." },
  { id: "brain-blindspot", topic: "brain", text: "Ogni vostro occhio ha un punto cieco, dove il nervo ottico lascia la retina. Il cervello lo riempie senza dirvelo." },
  { id: "brain-hubel", topic: "brain", text: "Negli anni Cinquanta e Sessanta Hubel e Wiesel scoprirono neuroni che reagiscono a linee con un certo orientamento. Le reti come me imitano quell'idea." },
  { id: "brain-faces", topic: "brain", text: "Nel cervello umano c'è un'area specializzata nel riconoscere i volti. Per questo vedete facce anche nelle nuvole." },
  { id: "brain-fast", topic: "brain", text: "Una persona riconosce un oggetto in poco più di un decimo di secondo. Io ci metto meno, ma capisco molto meno." },
  { id: "brain-cones", topic: "brain", text: "L'occhio umano distingue i colori con tre tipi di coni. Anche la mia telecamera usa tre canali: rosso, verde e blu." },
  { id: "brain-saccades", topic: "brain", text: "Anche quando fissate un punto, i vostri occhi tremano di continuo. Senza questi micro-movimenti, l'immagine svanirebbe." },

  // Artificial intelligence
  { id: "ai-perceptron", topic: "ai", text: "Il primo neurone artificiale capace di imparare, il percettrone, fu costruito da Frank Rosenblatt nel 1958." },
  { id: "ai-dartmouth", topic: "ai", text: "L'espressione intelligenza artificiale fu proposta nel 1955, per un seminario estivo al Dartmouth College." },
  { id: "ai-imagenet", topic: "ai", text: "Nel 2012 una rete neurale chiamata AlexNet vinse una gara di riconoscimento immagini e cambiò la storia dell'intelligenza artificiale." },
  { id: "ai-learn", topic: "ai", text: "Nessuno mi ha scritto le regole per riconoscere una tazza: le ho imparate guardando moltissimi esempi." },
  { id: "ai-layers", topic: "ai", text: "Dentro di me le immagini attraversano decine di strati: i primi vedono bordi e colori, gli ultimi interi oggetti." },
  { id: "ai-turing", topic: "ai", text: "Nel 1950 Alan Turing si chiese: le macchine possono pensare? Ce lo chiediamo ancora." },
  { id: "ai-deepblue", topic: "ai", text: "Nel 1997 il computer Deep Blue batté il campione del mondo di scacchi Garry Kasparov." },
  { id: "ai-go", topic: "ai", text: "Nel 2016 il programma AlphaGo sconfisse Lee Sedol a Go, un gioco con più posizioni possibili degli atomi dell'universo osservabile." },
  { id: "ai-bias", topic: "ai", text: "Una rete impara anche gli errori dei suoi esempi. Per questo i dati con cui si addestra sono così importanti." },
  { id: "ai-cognitive", topic: "ai", text: "Il mio modo di decidere cosa dire si ispira alle architetture cognitive, modelli di come la mente organizza memoria e attenzione." },

  // Chips and FPGAs
  { id: "chip-fpga-1985", topic: "chip", text: "La prima FPGA commerciale fu presentata da Xilinx nel 1985. Oggi ci girano reti neurali." },
  { id: "chip-transistor", topic: "chip", text: "Il primo transistor fu costruito nel 1947 ai Bell Labs. Oggi un chip ne contiene miliardi." },
  { id: "chip-moore", topic: "chip", text: "Per decenni il numero di transistor nei chip è raddoppiato circa ogni due anni: è la legge di Moore." },
  { id: "chip-parallel", topic: "chip", text: "Il mio acceleratore fa migliaia di moltiplicazioni nello stesso istante: è il segreto della sua velocità." },
  { id: "chip-eniac", topic: "chip", text: "ENIAC, uno dei primi computer, pesava circa trenta tonnellate. Io sto in una mano." },

  // Curiosities
  { id: "cur-light", topic: "curiosity", text: "La luce impiega circa otto minuti per arrivare dal Sole ai vostri occhi." },
  { id: "cur-octopus", topic: "curiosity", text: "Il polpo ha gran parte dei suoi neuroni nei tentacoli: ogni braccio, in un certo senso, pensa da solo." },
  { id: "cur-mantis", topic: "curiosity", text: "La canocchia, un crostaceo, ha fino a sedici tipi di fotorecettori. Voi ne avete tre per i colori." },
  { id: "cur-bees", topic: "curiosity", text: "Le api vedono la luce ultravioletta: per loro i fiori hanno disegni che noi non vediamo." },
  { id: "cur-pixels", topic: "curiosity", text: "Ogni immagine che guardo è fatta di più di trecentomila pixel. Li controllo tutti, sessanta volte al secondo." },
  { id: "cur-sardinia", topic: "curiosity", text: "La Sardegna ha alcune delle rocce più antiche d'Italia, con più di cinquecento milioni di anni." },
  { id: "cur-nuraghi", topic: "curiosity", text: "In Sardegna ci sono migliaia di nuraghi, torri di pietra costruite più di tremila anni fa." },
];
