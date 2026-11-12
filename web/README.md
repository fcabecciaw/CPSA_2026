# Showcase dashboard (`web/`)

The booth dashboard of the one CPSA demo, "Cosa vede l'IA": a Vite +
TypeScript static app that the Kria KV260 serves at `/`. It connects to the
same origin at `/ws`, renders the protocol v1 snapshot and the live stream, and
works with no network access and no CDN (NFR-07, FR-501). The UI copy is
Italian, as required by `PRODUCT.md`.

The server publishes what the board sees and the dashboard draws it; the
`Oggetti | Scheletro` switch picks the model (YOLOX objects or the MoveNet
skeleton room). The protocol is F-K1 of `docs/plans/showcase-only.md` —
`hello`, `status`, `showcase.state`, `pose.state`, `error` and `pong` from the
server, `ping` and `model.set` from the client.

## Commands

```sh
npm --prefix web install     # once, writes package-lock.json
npm --prefix web ci          # reproducible install from the lock file
npm --prefix web run dev     # dev server, open http://localhost:5173/?mock=1
npm --prefix web run build   # static build in web/dist
npm --prefix web run typecheck
```

Vite 8 needs Node `^20.19.0` or `>=22.12.0`. The runtime dependencies are
`three` (ARI's sphere and the skeleton room) and three `@fontsource-variable`
font packages, all bundled into `web/dist` by the build; the server's
`web.static_dir` points there (F-K3).

`vite.config.ts` sets `base: "./"` so `dist/index.html` references
`./assets/...`. That keeps the build portable when served at `/`, including
under the KV260's static handler. The fonts (Unbounded, Atkinson Hyperlegible
Next and Mono, SIL OFL) are self-hosted `woff2` files in `dist/assets`; nothing
is fetched from the network.

## Layout

A top bar over a diptych: the camera on the left, ARI on the right, both the
height of the 4:3 frame and centred. One visual language follows ARI's
particle sphere: cyan, violet and magenta over a violet-black background.

1. **Top bar** — the `CPSA` wordmark with `EOLAB, Università di Cagliari`, the
   `Oggetti | Scheletro` segmented control, the connection dot, the mute
   button and the debug toggle.
2. **Left pane** — `/video.mjpeg` (F-K2) in a rounded 4:3 frame with a thin
   gradient ring and the “Cosa vede l'IA” tab on its top edge. With no frames
   it shows “In attesa della telecamera…”; a broken stream shows “Video non
   disponibile” and retries after 1 s, doubling up to 10 s. While the socket is
   not open, the placeholder is back instead of a frozen last frame.
3. **Right pane** — ARI's sphere, its current sentence as subtitles and, in
   the objects view, one chip per kind of object (`persona`, with a small `2×`
   badge from two up) plus “Mostra un oggetto alla telecamera”. In the
   skeleton view the sphere shrinks and the MoveNet room fills the pane.

### Presentation and debug modes

The page opens in **presentation mode**: no scores, no percentages, no
numbers. The discreet button at the top right (or the `d` key, ignored while
typing in a field) switches to **debug mode**, which adds the connection label,
fps and frame time, “voce: in pausa” when the audio context is suspended, the
voice picker, the `RGB` badge, the counts and the most confident detections
with their score bars, the “mente” strip of the cognitive agent and the model
footer. The choice is stored in `localStorage` (`cpsa.dashboard.ui.v1`);
`?debug=1` or `?debug=0` force it for one load. The CSS does the hiding:
`<html data-ui>` hides `.debug-only` or `.presentation-only` elements.

Errors and transport warnings appear as a toast under the camera in both
modes; the mock badge only in debug mode.

The voice is fixed to XTTS-v2 Ana Florence (`FIXED_VOICE`) unless the debug
picker stored another one; “Attiva la voce” under the sphere is the click gate
the browser needs before any audio.

## Mock

`?mock=1` runs the dev-only stand-in (`src/mock.ts`, removed from the
production bundle by the `import.meta.env.DEV` guard). It speaks exactly the
F-K1 messages — `hello` on connect, `status` at 1 Hz, `showcase.state` at
10 Hz, `pong` for every `ping`, `error` `bad_command` for anything else — and
draws the matching fake picture: the RGB room with the same boxes and labels
the board would draw. The notice line says the data is fake.

```sh
# the room with two people and four objects on the table
http://localhost:5173/?mock=1
# nothing detected: the invitation only
http://localhost:5173/?mock=1&objects=0
```

`objects=0` empties the detector view; `video=0` leaves the camera pane on its
placeholder; `view=skeleton` opens the skeleton room; `debug=1` opens debug
mode.

`npm run shots` drives headless Chrome over the mock and writes screenshots to
`build/web-shots/` (`--width 1920 --height 1080`, `--only demo-present-objects`,
`--suffix` for the viewport in the name, `--port` when 5173 is taken, `--gpu`
with `bench=1` for a real frame-rate reading).

## Files

- `src/main.ts` — boot, stream routing, region updates;
- `src/views.ts` — top bar and notice toast;
- `src/uimode.ts` — presentation/debug mode, its button and the `d` key;
- `src/metrics.ts` — the fps and latency readout;
- `src/video.ts` — the MJPEG pane with its retry;
- `src/showcase.ts` — the object chips and the debug lists;
- `src/state.ts` — the last snapshot of each topic;
- `src/protocol.ts` — protocol v1 types and validation (F-K1);
- `src/labels.ts` — Italian copy and number formatting;
- `src/mock.ts` — the dev-only stand-in;
- `src/voice.ts` — Italian voice, local TTS client and per-tab selection;
- `src/voicepicker.ts` — the top-bar voice picker (`v` compares the voices).
