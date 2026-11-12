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

1. **Top bar** — the title “Cosa vede l'IA” with its line “Visione artificiale
   in tempo reale, su un chip”, the `Oggetti | Scheletro` segmented control,
   the connection dot, the mute button and the debug toggle. “EOLAB,
   Università di Cagliari” is built in but hidden until its wording is
   confirmed: `VITE_SHOW_BRAND_LINE=1 npm run build` shows it
   (`src/uiconfig.ts`).
2. **Left pane** — `/video.mjpeg` (F-K2) in a rounded 4:3 frame with a thin
   gradient ring and the “In diretta dalla telecamera” tab on its top edge. With no frames
   it shows “In attesa della telecamera…”; a broken stream shows “Video non
   disponibile” and retries after 1 s, doubling up to 10 s. While the socket is
   not open, the placeholder is back instead of a frozen last frame.
3. **Right pane** — ARI's sphere, its current sentence as subtitles and, in
   the objects view, one chip per kind of object (`persona`, with a small `2×`
   badge from two up) plus “Mostra un oggetto alla telecamera”. In the
   skeleton view the sphere shrinks and the MoveNet room fills the pane.

   The pane holds three independent subtrees with one writer each: ARI's
   stage (`#hologram`, `AvatarView` and the voice gate), the subtitles
   (`#subtitles`, `Subtitles`, written only when the spoken line changes) and
   the objects panel (`#showcase-panel`, `ShowcaseView`, refreshed every
   500 ms). The subtitles and the objects panel have fixed heights and all
   three use CSS containment, so a chip coming or going never rewrites the
   subtitle, moves it or resizes the sphere's canvas
   (`src/subtitles.test.ts` checks it over 100 panel updates).

### "Come funziona": the attract loop

The page opens on a full-screen loop for high-school students (`src/attract/`):
seven scenes of 10 s each, a headline, one or two lines and an animated SVG
diagram, with the live camera small in a corner and a progress bar at the top.

1. the camera: 640 × 480 at 60 frames per second, about 18 million pixels a
   second, three numbers per pixel;
2. preparing the image: 416 × 416 with grey letterbox bands, int8 values
   (`round(pixel ÷ 4)`, as `src/cpsa/yolox.py` does);
3. the chip: the KV260's FPGA and its DPU, up to about 1.2 TOPS int8, with the
   live DPU time and frame rate;
4. the neural network: edges, shapes, objects on the 52/26/13 grids; 80 COCO
   categories with box and confidence; MoveNet's 17 keypoints; what the board
   sees right now;
5. tracking: a jittery box against a stable one;
6. ARI's mind: perception, short memory, rules and voice, all on the laptop;
7. privacy: nothing is recorded or sent to the internet.

Live numbers come from the WebSocket; offline the scenes say “misurato” and
use the board measurements (60 fps, 4.7 ms). **One action starts the demo**:
the “Avvia la demo” button, Space or Enter. The same gesture unlocks the audio,
so it replaces the “Attiva la voce” click. `i` (or the loop button of debug
mode) brings the loop back and silences ARI; ←/→ step through the scenes.
While the loop is open the sphere and the room stop drawing. `?attract=0`
skips the loop and `?scene=N` pins one scene (screenshots).

### Presentation and debug modes

The page opens in **presentation mode**: no scores, no percentages, no
numbers. The discreet button at the top right (or the `d` key, ignored while
typing in a field) switches to **debug mode**, which adds the connection label,
fps and frame time, “voce: in pausa” when the audio context is suspended, the
voice picker, the `RGB` badge, the counts and the most confident detections
with their score bars, the “mente” strip of the cognitive agent, the presence
readout and the model footer. The presence readout shows why ARI greets, says
goodbye or reacts to a smile: one dot per presence source (YOLO person track,
laptop face, MoveNet skeleton with score ≥ 0.3), the fused state (absent only
after 3 s with no source), the live smile score against its per-face threshold,
the last face cue, the distance gate and every tracked person (`P` person box
height, `V` face width, as fractions of the frame; `●` at the stand, `✓`
greeted, `·` in the background). Only people at the stand count: a person box
at least `presence.min_box_height` (0.35) of the frame high or a face at least
`presence.min_face_width` (0.08) of the frame wide; `setPresenceThresholds` in
`src/cognitive/presence.ts` changes them and keeps them in `localStorage`
(`cpsa.presence.thresholds.v1`). ARI says hello to every newcomer (a track at
the stand for 0.6 s, people arriving together get one greeting with their
number, greetings at least 8 s apart) and goodbye only when everybody has left. The choice is stored in `localStorage` (`cpsa.dashboard.ui.v1`);
`?debug=1` or `?debug=0` force it for one load. The CSS does the hiding:
`<html data-ui>` hides `.debug-only` or `.presentation-only` elements.

Errors and transport warnings appear as a toast under the camera in both
modes; the mock badge only in debug mode.

### Calibrazione (debug mode)

The round button with a viewfinder icon in the top bar (or `c`)
opens **Calibrazione**, a compact card for the calibration day: the object
detector ("Modello oggetti": YOLOX-nano or YOLOv5-nano, the latter disabled
when `status.calibration.detectors` lacks it; with YOLOv5 the zoom controls are
greyed out and the outline hidden, since it analyses the whole frame at 640 px),
the YOLOX crop (zoom 1.0–2.0, centre X/Y), the camera gain (logarithmic slider, 64–2540),
the exposure (1/100 s or 1/50 s, both flicker-safe at 50 Hz), the saturation
and "Ripristina". Every change is shown at once and sent as one
`calibration.set` about 150 ms after the last movement; the board applies it on
the next frame and stores it in `~/cpsa/config/calibration.json`, so it
survives a restart, and "Ripristina" (`calibration.reset`) goes back to
`config/showcase.yaml`. While the card is open the video shows the region
YOLOX analyses as a thin dashed outline. "Distanza persone" sets the
laptop-side presence gate live (`setPresenceThresholds`: person box height
0.10–0.80 and face width 0.03–0.25 of the frame, kept in `localStorage`, never
sent to the board) and shows the sizes of the people tracked right now. Values come from `status.calibration`;
`?debug=1&calibration=1` opens the card at load. Presentation mode hides it.

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
`build/web-shots/` (`--width 1920 --height 1080`, `--only attract-3,demo-present-objects`,
`--suffix` for the viewport in the name, `--port` when 5173 is taken, `--gpu`
with `bench=1` for a real frame-rate reading).

## Files

- `src/main.ts` — boot, stream routing, region updates;
- `src/views.ts` — top bar and notice toast;
- `src/uimode.ts` — presentation/debug mode, its button and the `d` key;
- `src/attract/` — the “Come funziona” loop: state machine, scenes, view;
- `src/metrics.ts` — the fps and latency readout;
- `src/video.ts` — the MJPEG pane with its retry;
- `src/showcase.ts` — the object chips and the debug lists;
- `src/subtitles.ts` — ARI's current line under the sphere;
- `src/uiconfig.ts` — build-time switches (`VITE_SHOW_BRAND_LINE`);
- `src/state.ts` — the last snapshot of each topic;
- `src/protocol.ts` — protocol v1 types and validation (F-K1);
- `src/labels.ts` — Italian copy and number formatting;
- `src/mock.ts` — the dev-only stand-in;
- `src/voice.ts` — Italian voice, local TTS client and per-tab selection;
- `src/voicepicker.ts` — the top-bar voice picker (`v` compares the voices).
- `src/calibration.ts` — calibration ranges, debounce and crop geometry;
- `src/calibrationpanel.ts` — the debug "Calibrazione" card and the crop outline.
