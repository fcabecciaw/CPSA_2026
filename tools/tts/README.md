# cpsa-tts — voce italiana locale per la dashboard

Servizio HTTP di sintesi vocale **in italiano**, eseguito sul laptop accanto alla
dashboard. Nasce dal feedback R2-V: «la voce è chiaramente finta, non possiamo
usarne una più umana?». La dashboard parla ancora con la Web Speech API quando il
servizio non risponde, ma appena `http://127.0.0.1:8765/health` risponde usa i WAV
di questo servizio.

Due motori, entrambi locali (nessuna richiesta di testo a servizi esterni):

| Motore | Voci | Licenza | Dispositivo |
| --- | --- | --- | --- |
| **Kokoro 82M** | `if_sara` (Sara), `im_nicola` (Nicola) | Apache-2.0 | CPU |
| **XTTS-v2** (Coqui) | `ana_florence`, `sofia_hellen`, `gilberto_mathias`, `damien_black` | Coqui Public Model License, **solo non commerciale** | GPU se disponibile, altrimenti CPU |

Kokoro è il motore predefinito: è Apache-2.0, occupa poco e su questo laptop una
frase calda costa poche decine di millisecondi. XTTS-v2 suona più naturale ma è
più lento e non commerciale: si sceglie con una variabile d'ambiente o per
singola richiesta.

## Requisiti

- Ubuntu 24.04, `python3.12`, [`uv`](https://docs.astral.sh/uv/).
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
  -d '{"text":"Ciao! Che bello vederti.","engine":"kokoro","voice":"if_sara"}' \
  -o ciao.wav
```

### API

| Metodo | Percorso | Risposta |
| --- | --- | --- |
| `GET` | `/health` | stato, motori (disponibile/carico/device), cache, ultime misure |
| `GET` | `/voices` | motori, licenze, device, voci selezionabili |
| `POST` | `/tts` | `{"text": str, "engine"?: str, "voice"?: str}` → `audio/wav` |

Ogni risposta contiene gli header `X-TTS-Engine`, `X-TTS-Voice`,
`X-TTS-Cache: hit\|miss` e `X-TTS-Synth-Ms`. Errori: `400` per motore/voce/text
non validi, `422` per un corpo malformato, `502` se il motore fallisce.

La cache sta in `~/.cache/cpsa-tts/` con nome
`sha1(engine, voice, text).wav`: il testo non viene mai scritto su disco. Le
richieste identiche concorrenti sintetizzano una volta sola, e i file sono scritti
in modo atomico.

### CORS

Origini ammesse: `http://10.42.0.83:8080` (dashboard) e
`http://localhost:5173` (Vite). Si cambiano con
`CPSA_TTS_ALLOWED_ORIGINS="http://a,http://b"`.

### Configurazione

| Variabile | Default | Significato |
| --- | --- | --- |
| `CPSA_TTS_HOST` | `127.0.0.1` | indirizzo di ascolto |
| `CPSA_TTS_PORT` | `8765` | porta |
| `CPSA_TTS_ENGINE` | `kokoro` | motore predefinito |
| `CPSA_TTS_VOICE` | `if_sara` | voce predefinita |
| `CPSA_TTS_CACHE_DIR` | `~/.cache/cpsa-tts` | cache WAV |
| `CPSA_TTS_KOKORO_DEVICE` | `cpu` | device di Kokoro (`cuda` per provarlo sulla GPU) |
| `CPSA_TTS_XTTS_DEVICE` | `auto` | `auto`, `cuda` o `cpu` |
| `CPSA_TTS_XTTS_SPEED` | `1.0` | velocità di XTTS |
| `CPSA_TTS_ALLOWED_ORIGINS` | le due origini sopra | CORS, separate da virgole |
| `CPSA_TTS_LOG_LEVEL` | `INFO` | log |
| `CPSA_TTS_SAMPLES_DIR` | `<repo>/build/tts-samples` | dove scrive `samples` |

## Pre-generazione delle frasi fisse

La dashboard parla soprattutto con frasi fisse: 97 frasi del rulebook senza slot,
40 fatti e 20 poesie (le righe di una poesia sono unite con uno spazio: la
punteggiatura di fine riga è la pausa breve). `pregenerate` le estrae dai sorgenti
TypeScript, esporta `build/phrases.json` e riempie la cache:

```sh
uv run cpsa-tts pregenerate                  # motore predefinito (Kokoro), ~1 min
uv run cpsa-tts pregenerate --engine xtts    # XTTS, ~3 min (0.9 s/frase calda)
uv run cpsa-tts pregenerate --dry-run        # solo phrases.json
```

Il parsing è in `cpsa_tts/phrases.py`: nessuna dipendenza da Node. Se il rulebook
cambia, `tests/test_phrases.py` fallisce sui conteggi fissati e ricorda di
rieseguire la pre-generazione. Le frasi con slot (`{oggetto}`, `{n}`, `{ora}`,
`{dpu}`, …) sono escluse: si sintetizzano alla prima occorrenza e restano in cache.

## Ascolto comparato (scegliere la voce a orecchio)

```sh
uv run cpsa-tts samples
xdg-open build/tts-samples/index.html
```

Scrive cinque frasi rappresentative (saluto, due fatti, due poesie, incluse le
frasi reali della dashboard) per **ogni** voce dei due motori e una pagina HTML
senza dipendenze per ascoltarle. `--voice`/`--engine` limitano la produzione,
`--force` rigenera.

## Prestazioni misurate su questo laptop

Misurate con `uv run cpsa-tts bench` (RTX PRO 500 Blackwell 6 GB, driver 580,
16 core; `build/tts-bench.json` è il report completo) mentre il laptop era sotto
carico (load average ~20): i valori sono quindi prudenti.

| Motore | Device | Caricamento | Prima frase | Frase calda (mediana) | RTF caldo | VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| Kokoro 82M | CPU | 10.5 s | 0.88 s | 0.10 s (min 82 ms) | 0.02 | — |
| XTTS-v2 | CUDA | 12.4 s | 1.96 s | 1.66 s (min 1.40 s) | 0.30 | ~2.6 GB |

(`RTF` = tempo di sintesi / durata dell'audio: sotto 1 è più veloce del parlato;
il testo di prova dura in media 4.7 s per Kokoro e 5.6 s per XTTS.) Sull'HTTP, la
prima richiesta XTTS comprensiva di caricamento ha impiegato 14.4 s, le successive
0.6–1.7 s. Con la cache piena una frase costa una lettura da disco (~1 ms). XTTS
ripiega su CPU se `torch.cuda.is_available()` è falso: aspettarsi circa 10 volte
tanto. Entrambi i motori restano caricati (CPU + ~2.6 GB di GPU), quindi
`/health` riporta `loaded: true` su entrambi dopo il primo uso.

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

I test usano un motore finto: non scaricano modelli e girano in meno di un
secondo.

## Licenze

- **Kokoro-82M**: Apache-2.0 (<https://huggingface.co/hexgrad/Kokoro-82M>).
- **XTTS-v2**: Coqui Public Model License, uso non commerciale
  (<https://coqui.ai/cpml>).
- **eSpeak NG**: GPL-3.0 (usato solo come libreria di fonemizzazione).
- Il codice di questo servizio è Apache-2.0, come il resto del repository.

## Problemi noti

- `ImportError: torchcodec ... required` all'import di `TTS`: `uv sync` installa
  l'extra `codec`; se hai un ambiente vecchio, `uv add "coqui-tts[codec]"`.
- Kokoro pronuncia male una parola: aggiungi un `lexicon`/`misaki` custom oppure
  usa XTTS per quella frase.
- `CUDA out of memory` con XTTS: chiudi altri processi GPU, oppure
  `CPSA_TTS_XTTS_DEVICE=cpu` (più lento ma funziona).
- La cache cresce: `rm -rf ~/.cache/cpsa-tts` per svuotarla. Su questo laptop
  contiene 157 frasi × 2 voci Kokoro + 157 frasi XTTS (`ana_florence`):
  477 WAV, ~95 MB; le altre voci XTTS si riempiono alla prima frase.
- Due processi XTTS sulla stessa GPU da 6 GB non entrano (OOM al `.to("cuda")`):
  ferma il servizio (`systemctl --user stop cpsa-tts`) prima di
  `pregenerate --engine xtts`, oppure usa `CPSA_TTS_XTTS_DEVICE=cpu`.
