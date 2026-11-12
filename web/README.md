# Showcase dashboard (`web/`)

The booth dashboard of the one CPSA demo, "Cosa vede l'IA": a Vite +
TypeScript static app that the Kria KV260 serves at `/`. It connects to the
same origin at `/ws`, renders the protocol v1 snapshot and the live stream, and
works with no network access and no CDN (NFR-07, FR-501). The UI copy is
Italian, as required by `PRODUCT.md`.

There is one view and no modes: the server publishes what the detector sees,
and the dashboard draws it. The protocol is F-K1 of
`docs/plans/showcase-only.md` — `hello`, `status`, `showcase.state`, `error`
and `pong` from the server, `ping` from the client.

## Commands

```sh
npm --prefix web install     # once, writes package-lock.json
npm --prefix web ci          # reproducible install from the lock file
npm --prefix web run dev     # dev server, open http://localhost:5173/?mock=1
npm --prefix web run build   # static build in web/dist
npm --prefix web run typecheck
```

Vite 8 needs Node `^20.19.0` or `>=22.12.0`. The only dev dependencies are
`vite` and `typescript`, pinned to exact versions; there are no runtime
dependencies. `npm run build` produces `web/dist`; the server's
`web.static_dir` points there (F-K3).

`vite.config.ts` sets `base: "./"` so `dist/index.html` references
`./assets/...`. That keeps the build portable when served at `/`, including
under the KV260's static handler. All fonts are the system font stack; there
are no font files, images or scripts fetched from the network.

## Layout

A slim top bar and one instrument surface split in two panes:

1. **Top bar** — the `CPSA` wordmark, the view title, the connection dot with
   its label, and the analysed frames per second and their processing time.
2. **Left pane** — `/video.mjpeg` (F-K2), letterboxed 4:3 with
   `object-fit: contain` and the `RGB` badge. With no frames it shows “In
   attesa della telecamera…”; a broken stream shows “Video non disponibile”
   and retries after 1 s, doubling up to 10 s. The stream follows the
   connection: while the socket is not open, the placeholder is back instead of
   a frozen last frame.
3. **Right pane** — what the detector sees: the counts as large rows
   (`2 × persona`), then the most confident detections with their score bars,
   and “Mostra un oggetto alla telecamera” while the frame is empty. The footer
   names the model and the frame size.

The bar numbers come from `showcase.state` (`fps` and `total_ms`, falling back
to `dpu_ms`); until the first frame arrives they come from `status`. The line
under the bar carries what the process reports in `status.errors`, a protocol
`error`, a transport problem, or the mock badge.

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
placeholder.

## Files

- `src/main.ts` — boot, stream routing, region updates;
- `src/views.ts` — top bar and notice line;
- `src/metrics.ts` — the fps and latency readout;
- `src/video.ts` — the MJPEG pane with its retry;
- `src/showcase.ts` — the "Cosa vede l'IA" pane;
- `src/state.ts` — the last snapshot of each topic;
- `src/protocol.ts` — protocol v1 types and validation (F-K1);
- `src/labels.ts` — Italian copy and number formatting;
- `src/mock.ts` — the dev-only stand-in.
