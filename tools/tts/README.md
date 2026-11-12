# cpsa-tts — voce italiana locale per la dashboard

Servizio HTTP di sintesi vocale **in italiano**, eseguito sul laptop accanto alla
dashboard. Nasce dal feedback R2-V: «la voce è chiaramente finta, non possiamo
usarne una più umana?». La dashboard parla ancora con la Web Speech API quando il
servizio non risponde, ma appena `http://127.0.0.1:8765/health` risponde usa i WAV
di questo servizio.

Due motori locali (nessuna richiesta di testo a servizi esterni) e un terzo
motore online, che è quello che suona più umano in italiano:

| Motore | Voci | Licenza | Dispositivo |
| --- | --- | --- | --- |
| **Kokoro 82M** | `if_sara` (Sara), `im_nicola` (Nicola) | Apache-2.0 | CPU |
| **XTTS-v2** (Coqui) | `ana_florence`, `sofia_hellen`, `gilberto_mathias`, `damien_black` | Coqui Public Model License, **solo non commerciale** | GPU se disponibile, altrimenti CPU |
| **Microsoft Neural** (`edge-tts`) | `it-IT-IsabellaNeural`, `it-IT-ElsaNeural`, `it-IT-DiegoNeural`, `it-IT-GiuseppeMultilingualNeural` | nessuna licenza: solo i termini d'uso del servizio "leggi ad alta voce" di Edge, client **non ufficiale** | online (internet) |

Kokoro è il motore predefinito: è Apache-2.0, occupa poco e su questo laptop una
frase calda costa poche decine di millisecondi. XTTS-v2 suona più naturale ma è
più lento e non commerciale: si sceglie con una variabile d'ambiente o per
singola richiesta.

Il motore `edge` è quello che ha risposto al «si sente ancora finta»: le voci
neurali Microsoft sono molto più naturali delle due locali, in cambio chiedono
internet una volta per frase (il testo passa dall'endpoint di Edge: non è più
un servizio solo locale) e **non** portano una licenza d'uso: `edge-tts` è un
client non ufficiale dell'endpoint "leggi ad alta voce" di Edge, quindi usalo
solo per questa demo e non in un prodotto. Le frasi già in cache restano
disponibili offline.

## Requisiti

- Ubuntu 24.04, `python3.12`, [`uv`](https://docs.astral.sh/uv/).
- Per il motore `edge` (voci Microsoft neurali): internet. `uv sync` installa
  `edge-tts`; se il laptop è offline il servizio resta in piedi, `/voices`
  elenca le voci note e `POST /tts` fallisce con un messaggio chiaro (le frasi
  già in cache continuano a suonare).
- **eSpeak NG** per la fonemizzazione italiana di Kokoro. Serve la libreria e i
  dati, non il programma: `sudo apt install espeak-ng` (su questo laptop
  `libespeak-ng.so.1` e `espeak-ng-data` erano già presenti).
- Per XTTS su GPU: driver NVIDIA recente (580 sul laptop) e un PyTorch con CUDA
  per Blackwell (`sm_120`). Il `uv sync` installa il wheel CUDA di default da
  PyPI, che contiene già il runtime; con driver 580 funziona.

## Installazione

```sh
cd tools/tts
uv sync                # progetto uv indipendente da quello della board
uv run cpsa-tts voices # elenca motori e voci (non carica i modelli)
```

Il primo avvio di un motore scarica i pesi da Hugging Face: ~330 MB per Kokoro,
~2 GB per XTTS-v2 (in `~/.cache/huggingface` e `~/.local/share/tts`). XTTS-v2 è
distribuito con la CPML: il servizio imposta `COQUI_TOS_AGREED=1` perché l'uso è
stato approvato nel task R2-V per questa demo **non commerciale**. Per un uso
commerciale serve una licenza Coqui.

## Avvio

```sh
uv run cpsa-tts serve --warm          # 127.0.0.1:8765, Kokoro già caricato
```

`--warm` precarica il motore predefinito, così la prima frase non paga il
caricamento (Kokoro ~5 s, XTTS ~20-60 s a seconda della cache dei pesi).

```sh
curl -s http://127.0.0.1:8765/health | python3 -m json.tool
curl -s http://127.0.0.1:8765/voices | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8765/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Ciao! Che bello vederti.","engine":"edge","voice":"it-IT-ElsaNeural"}' \
  -o ciao.wav

# La voce predefinita, scelta dalla dashboard (o da qui), resta dopo il riavvio:
curl -s -X PUT http://127.0.0.1:8765/default \
  -H 'Content-Type: application/json' \
  -d '{"engine":"edge","voice":"it-IT-IsabellaNeural"}'
curl -s -X POST http://127.0.0.1:8765/pregenerate \
  -H 'Content-Type: application/json' \
  -d '{"engine":"edge","voice":"it-IT-IsabellaNeural"}'
curl -s http://127.0.0.1:8765/pregenerate
```

### API

| Metodo | Percorso | Risposta |
| --- | --- | --- |
| `GET` | `/health` | stato, motori (disponibile/carico/device), cache, ultime misure, avanzamento della pre-generazione |
| `GET` | `/voices` | motori, licenze, device, voci selezionabili |
| `POST` | `/tts` | `{"text": str, "engine"?: str, "voice"?: str}` → `audio/wav` |
| `PUT` | `/default` | `{"engine": str, "voice": str}` → la nuova coppia predefinita, salvata in `<cache>/default.json` |
| `POST` | `/pregenerate` | `{"engine"?: str, "voice"?: str}` → `202` con lo stato del riempimento; `409` se un job è già in corso |
| `GET` | `/pregenerate` | `{running, engine, voice, total, done, synthesized, cached, failed, error, started_at, finished_at, elapsed_s}` |

La voce predefinita arriva dall'ambiente (`CPSA_TTS_DEFAULT_ENGINE`,
`CPSA_TTS_DEFAULT_VOICE`) e si cambia a caldo con `PUT /default`; la scelta viene
scritta in modo atomico in `~/.cache/cpsa-tts/default.json` e al riavvio ha la
precedenza sull'ambiente.

Ogni risposta contiene gli header `X-TTS-Engine`, `X-TTS-Voice`,
`X-TTS-Cache: hit\|miss` e `X-TTS-Synth-Ms`. Errori: `400` per motore/voce/text
non validi, `409` se un riempimento è già in corso, `422` per un corpo malformato,
`502` se il motore fallisce (per `edge` significa quasi sempre "senza internet").

La cache sta in `~/.cache/cpsa-tts/` con nome
`sha1(engine, voice, text).wav`: il testo non viene mai scritto su disco. Le
richieste identiche concorrenti sintetizzano una volta sola, e i file sono scritti
in modo atomico. Ogni motore produce lo stesso formato (mono, 24 kHz, PCM 16 bit)
anche quando l'endpoint restituisce altro: `edge` arriva in MP3 e viene
ricampionato dal servizio, così la cache e il percorso nel browser restano
identici per tutte le voci.

### CORS

Origini ammesse: `http://10.42.0.83:8080` (dashboard) e
`http://localhost:5173` (Vite). Si cambiano con
`CPSA_TTS_ALLOWED_ORIGINS="http://a,http://b"`.

### Configurazione

| Variabile | Default | Significato |
| --- | --- | --- |
| `CPSA_TTS_HOST` | `127.0.0.1` | indirizzo di ascolto |
| `CPSA_TTS_PORT` | `8765` | porta |
| `CPSA_TTS_DEFAULT_ENGINE` | `kokoro` | motore predefinito (accettato anche il vecchio `CPSA_TTS_ENGINE`) |
| `CPSA_TTS_DEFAULT_VOICE` | `if_sara` | voce predefinita (accettato anche il vecchio `CPSA_TTS_VOICE`) |
| `CPSA_TTS_CACHE_DIR` | `~/.cache/cpsa-tts` | cache WAV e `default.json` |
| `CPSA_TTS_KOKORO_DEVICE` | `cpu` | device di Kokoro (`cuda` per provarlo sulla GPU) |
| `CPSA_TTS_XTTS_DEVICE` | `auto` | `auto`, `cuda` o `cpu` |
| `CPSA_TTS_XTTS_SPEED` | `1.0` | velocità di XTTS |
| `CPSA_TTS_EDGE_RATE` | `+0%` | velocità delle voci Microsoft (`-5%`, `+10%`, …) |
| `CPSA_TTS_ALLOWED_ORIGINS` | le due origini sopra | CORS, separate da virgole |
| `CPSA_TTS_LOG_LEVEL` | `INFO` | log |
| `CPSA_TTS_SAMPLES_DIR` | `<repo>/build/tts-samples` | dove scrive `samples` |

## Pre-generazione delle frasi fisse

La dashboard parla soprattutto con frasi fisse: 100 frasi del rulebook senza slot
e 40 fatti. `pregenerate` le estrae dai sorgenti
TypeScript, esporta `build/phrases.json` e riempie la cache:

```sh
uv run cpsa-tts pregenerate                        # motore predefinito, ~1 min
uv run cpsa-tts pregenerate --engine xtts          # XTTS, ~3 min (0.9 s/frase calda)
uv run cpsa-tts pregenerate --engine edge --voice it-IT-IsabellaNeural
uv run cpsa-tts pregenerate --dry-run              # solo phrases.json
```

Dalla dashboard (e in generale via HTTP) lo stesso riempimento si lancia in
sottofondo, senza bloccare la pagina e con l'avanzamento in `GET /pregenerate`:

```sh
curl -s -X POST http://127.0.0.1:8765/pregenerate \
  -H 'Content-Type: application/json' \
  -d '{"engine":"edge","voice":"it-IT-IsabellaNeural"}'
# {"running": true, "engine": "edge", "voice": "...", "total": 157, "done": 0, ...}
```

Un solo job per volta (`409` se uno è già in corso): due riempimenti
scriverebbero la stessa cache e caricherebbero due volte lo stesso modello. Se il
motore `edge` è offline il job si ferma alla prima frase e riporta l'errore;
quello che era già in cache resta.

Il parsing è in `cpsa_tts/phrases.py`: nessuna dipendenza da Node. Se il rulebook
cambia, `tests/test_phrases.py` fallisce sui conteggi fissati e ricorda di
rieseguire la pre-generazione. Le frasi con slot (`{oggetto}`, `{n}`, `{ora}`,
`{dpu}`, …) sono escluse: si sintetizzano alla prima occorrenza e restano in cache.

## Ascolto comparato (scegliere la voce a orecchio)

```sh
uv run cpsa-tts samples
xdg-open build/tts-samples/index.html
```

Scrive cinque frasi rappresentative (saluto, due fatti, un oggetto, un congedo, incluse le
frasi reali della dashboard) per **ogni** voce dei motori e una pagina HTML
senza dipendenze per ascoltarle. `--voice`/`--engine` limitano la produzione,
`--force` rigenera. Le voci `edge` richiedono internet.

## Scegliere la voce dalla dashboard

Il modo più rapido per confrontare le voci è la dashboard stessa: in alto a
destra c'è un selettore **Voce** (`web/src/voicepicker.ts`).

- La tendina elenca tutte le voci di `/voices`, raggruppate per motore.
- **Prova** (o il tasto `v`) passa alla voce successiva e pronuncia una frase di
  prova; premendo `v` più volte si sentono tutte in fila. Il tasto non viene
  intercettato mentre si scrive in un campo di testo.
- La scelta vale per la scheda corrente (`localStorage`, chiave
  `cpsa.voice.pick.v1`) e diventa la predefinita del servizio (`PUT /default`,
  quindi `~/.cache/cpsa-tts/default.json`): sopravvive al ricaricamento e al
  riavvio del servizio.
- **Prepara frasi** lancia `POST /pregenerate` per la voce scelta e mostra
  l'avanzamento; alla fine le frasi fisse costano una lettura da disco.

Restano le regole di sempre: la prima frase parte solo dopo un clic, il mute
zittisce la scheda, il watchdog a 8 s interrompe una frase appesa e una sola
scheda parla.

## Prestazioni misurate su questo laptop

Misurate con `uv run cpsa-tts bench` (RTX PRO 500 Blackwell 6 GB, driver 580,
16 core; `build/tts-bench.json` è il report completo) mentre il laptop era sotto
carico (load average ~20): i valori sono quindi prudenti.

| Motore | Device | Caricamento | Prima frase | Frase calda (mediana) | RTF caldo | VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| Kokoro 82M | CPU | 10.5 s | 0.88 s | 0.10 s (min 82 ms) | 0.02 | — |
| XTTS-v2 | CUDA | 12.4 s | 1.96 s | 1.66 s (min 1.40 s) | 0.30 | ~2.6 GB |
| Microsoft Neural (`edge`) | online | — | ~0.3 s (rete) | ~0.2–0.4 s, poi cache | — | — |

(`RTF` = tempo di sintesi / durata dell'audio: sotto 1 è più veloce del parlato;
il testo di prova dura in media 4.7 s per Kokoro e 5.6 s per XTTS.) Sull'HTTP, la
prima richiesta XTTS comprensiva di caricamento ha impiegato 14.4 s, le successive
0.6–1.7 s. Con la cache piena una frase costa una lettura da disco (~1 ms). XTTS
ripiega su CPU se `torch.cuda.is_available()` è falso: aspettarsi circa 10 volte
tanto. Kokoro e XTTS restano caricati (CPU + ~2.6 GB di GPU), quindi `/health`
riporta `loaded: true` su entrambi dopo il primo uso; `edge` non carica nulla e
paga la rete per ogni frase nuova.

## Avvio automatico (systemd --user)

```sh
mkdir -p ~/.config/systemd/user
cp tools/tts/deploy/cpsa-tts.service ~/.config/systemd/user/
# Controlla WorkingDirectory/ExecStart se il repository non è ~/cpsa
systemctl --user daemon-reload
systemctl --user enable --now cpsa-tts
systemctl --user status cpsa-tts
journalctl --user -u cpsa-tts -f
```

Per farlo partire anche senza login: `sudo loginctl enable-linger "$USER"`.
La unit usa `uv run` e riavvia il servizio se cade; `--warm` tiene Kokoro pronto.
Per cambiare la voce predefinita basta il selettore della dashboard (o
`PUT /default`): scrive `~/.cache/cpsa-tts/default.json` e il servizio la
riprende al riavvio. Se invece la vuoi fissare nell'unit, aggiungi un
`Environment=CPSA_TTS_DEFAULT_ENGINE=edge` (e `..._VOICE=it-IT-IsabellaNeural`).

## Integrazione con la dashboard

`web/src/voice.ts` sonda `/health` all'avvio (e su `activate()`). Se risponde,
`speak()` scarica il WAV, lo decodifica e lo riproduce attraverso un
`AudioContext` con un `AnalyserNode` esposto da `voiceAudioNode()` (la bocca del
volto legge l'ampiezza da lì). Se il servizio non c'è, o la richiesta fallisce, si
torna alla Web Speech API. Restano il click gate, il watchdog a 8 s, il mute e la
regola «una sola scheda parla».

Nota: `GET /health` è pensato per essere interrogato dal browser della dashboard
sullo stesso laptop (`127.0.0.1`). Il servizio ascolta solo su loopback: dal
telefono o da un'altra macchina non è raggiungibile, e non deve esserlo.

## Test

```sh
cd tools/tts
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

I test usano un motore finto e un modulo `edge_tts` finto (in
`tests/test_edge.py`): non scaricano modelli, non toccano la rete e girano in
meno di un secondo.

## Licenze

- **Kokoro-82M**: Apache-2.0 (<https://huggingface.co/hexgrad/Kokoro-82M>).
- **XTTS-v2**: Coqui Public Model License, uso non commerciale
  (<https://coqui.ai/cpml>).
- **Voci Microsoft**: nessuna licenza. `edge-tts` è un client non ufficiale
  dell'endpoint "leggi ad alta voce" di Microsoft Edge: valgono solo i termini
  d'uso di quel servizio, il testo esce dal laptop e la cosa è accettabile per
  questa demo, non per un prodotto. Per un uso commerciale servono le API Azure
  Speech (con una chiave) o una voce locale con licenza.
- **eSpeak NG**: GPL-3.0 (usato solo come libreria di fonemizzazione).
- Il codice di questo servizio è Apache-2.0, come il resto del repository.

## Problemi noti

- `ImportError: torchcodec ... required` all'import di `TTS`: `uv sync` installa
  l'extra `codec`; se hai un ambiente vecchio, `uv add "coqui-tts[codec]"`.
- Kokoro pronuncia male una parola: aggiungi un `lexicon`/`misaki` custom oppure
  usa XTTS per quella frase.
- `CUDA out of memory` con XTTS: chiudi altri processi GPU, oppure
  `CPSA_TTS_XTTS_DEVICE=cpu` (più lento ma funziona).
- `edge` senza internet: `POST /tts` risponde `502` con un messaggio che parla di
  rete e cache; `GET /voices` continua a elencare le voci note (quelle salvate
  dall'ultima query riuscita o le quattro italiane predefinite). `POST
  /pregenerate` si ferma alla prima frase e riporta l'errore in `GET
  /pregenerate`.
- La cache cresce: `rm -rf ~/.cache/cpsa-tts` per svuotarla. Su questo laptop
  contiene 157 frasi × 2 voci Kokoro + 157 frasi XTTS (`ana_florence`):
  477 WAV, ~95 MB; le altre voci XTTS si riempiono alla prima frase, e ogni voce
  `edge` occupa ~157 WAV in più. Togliere la directory cancella anche
  `default.json` (la voce scelta torna quella d'ambiente).
- Due processi XTTS sulla stessa GPU da 6 GB non entrano (OOM al `.to("cuda")`):
  ferma il servizio (`systemctl --user stop cpsa-tts`) prima di
  `pregenerate --engine xtts`, oppure usa `CPSA_TTS_XTTS_DEVICE=cpu`.
