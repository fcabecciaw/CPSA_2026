# Hands dashboard (`web/`)

The booth dashboard of the hands demos: a Vite + TypeScript static app that the
Kria KV260 serves at `/`. It connects to the same origin at `/ws`, renders the
protocol v1 snapshot and the two hand streams, and works with no network access
and no CDN (NFR-07, FR-501). The UI copy is Italian, as required by
`PRODUCT.md`.

The medical views, their labels and their mock data are gone: this dashboard
shows only the two hand demos. The server may still publish `medical.*` and
`log` topics (medical mode still exists on the board); `protocol.ts` keeps
their types and the store ignores them, so an unexpected topic is never an
error.

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
`server.static_dir` points there (`docs/plans/medical-demo-slice.md`, F6).

`vite.config.ts` sets `base: "./"` so `dist/index.html` references
`./assets/...`. That keeps the build portable when served at `/`, including
under the KV260's static handler. All fonts are the system font stack; there
are no font files, images or scripts fetched from the network.

## Layout

A slim top bar and one instrument surface split in two panes:

1. **Top bar** — the `CPSA` wordmark, the segmented mode switch
   (`Ologramma` | `Sasso-carta-forbice`), the connection dot, and the analysed
   frames per second and their processing time. While Sasso-carta-forbice runs
   the score rests in the bar (`Tu 2 · Tavolo 1`, plus the draws once there is
   one).
2. **Left pane** — `/video.mjpeg` (F14 the IR camera), always in the page,
   letterboxed 4:3 with `object-fit: contain`. With no frames it shows
   “In attesa della telecamera…”; a broken stream shows “Video non
   disponibile” and retries after 1 s, doubling up to 10 s. The pane follows
   the mode: while no hand mode runs, no frames exist, so the placeholder is
   back instead of a frozen last frame.
3. **Right pane** — `hologram` mode draws `hologram.hands` as the existing
   glowing 3D hand on canvas 2D, with the hand count and the frame size in a
   small footer. `rps` mode draws `rps.state.hand.kpts2d` as a 2D skeleton,
   zoomed to fill the pane, with the countdown (“Sasso… carta… forbice!”) and
   the result banner over it and the player's gesture next to the board's move
   at the bottom. Any other mode shows a calm empty state.

The top-bar numbers come from the streams themselves: in `hologram` mode the
frame rate is measured over the last 2 s of `hologram.hands` arrivals and the
latency is its `timing.total_ms`; in `rps` mode they come from
`status.perception` (`timing.analyzed_fps`/`latency_ms` when present,
`fps`/`dpu_ms` otherwise), because `rps.state` carries no timing.

## Mock

`?mock=1` runs the dev-only stand-in (`src/mock.ts`, removed from the
production bundle by the `import.meta.env.DEV` guard). It replays the `hello`
and `status` fixtures, and synthesises both hand streams and a fake IR picture
for the camera pane; the notice line says the data is fake.

```sh
# hologram, Sasso-carta-forbice; `phase` freezes a round for a screenshot
http://localhost:5173/?mock=1&mode=hologram
http://localhost:5173/?mock=1&mode=rps&phase=countdown&count=2
http://localhost:5173/?mock=1&mode=rps&phase=result&outcome=win
http://localhost:5173/?mock=1&mode=idle
```

`count=3|2|1`, `outcome=win|lose|draw|unclear` refine a frozen round; `video=0`
leaves the camera pane on its placeholder.

## Files

- `src/main.ts` — boot, mode switch, stream routing;
- `src/views.ts` — top bar, notice line, score;
- `src/metrics.ts` — the fps and latency readout;
- `src/video.ts` — the MJPEG pane with its retry;
- `src/hologram.ts`, `src/rps.ts` — the two right-hand views;
- `src/state.ts` — the last snapshot of each stream;
- `src/protocol.ts` — protocol v1 types and validation (unchanged interfaces);
- `src/labels.ts` — Italian copy and number formatting;
- `src/mock.ts` — the dev-only stand-in.
