# Medical dashboard (`web/`)

The booth dashboard of the medical demo: a Vite + TypeScript static app that
the Kria KV260 serves at `/`. It connects to the same origin at `/ws`, renders
the protocol v1 snapshot and live updates, and works with no network access and
no CDN (NFR-07, FR-501). The UI copy is Italian, as required by `PRODUCT.md`.

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

## Views

1. **Header and control bar** — UNICA/EOLAB identity, connection state with
   round-trip time, `mode` and `mode_state`, camera/model/errors health, FPS,
   DPU time and uptime. The operator toggle sends `mode.set` for `idle` or
   `medical`; `mode.progress` steps and `error` messages appear in the bar.
2. **Stato live** — IMU state, the current tag (text plus colour), and whether
   an episode is active.
3. **Decisioni** — the latest 20 `medical.decision` messages, newest first and
   highlighted, each with time, tag, verdict, severity, reason and actuations.
4. **Ultimo episodio** — contact count, repetition rate, peak severity, labels
   and the contact table (side, region, duration, hand shape). While an episode
   is active the panel shows its current state; when it ends the summary
   replaces it.
5. **Scheletro** — a `<canvas>` drawing `medical.hands` in the HALFBUST_62
   layout, mirrored, with the bone list transcribed from
   `src/cpsa/vision/layouts.py` (body 0–10, nine face anchors 11–19, left hand
   20–40, right hand 41–61). Points below 0.5 confidence are dimmed and points
   with confidence 0 are hidden. With no frame it shows “In attesa di
   episodio…”.

Trigger source: when `medical.state.trigger` is `vision` (camera-only mode), the live
status shows an "Attivazione: telecamera" tile and the IMU tile reads "Non usato (solo
telecamera)". With `imu`, or when the field is absent (older server), the display is unchanged.

## Protocol handling

- `src/protocol.ts` is written by hand from `docs/protocol/v1/*.schema.json`.
  The parser checks the envelope and each known payload, ignores unknown
  message types and unknown fields (IF-103), and returns `invalid` for
  malformed known messages so the connection bar can report them.
- `src/client.ts` connects to `ws://<host>/ws` (or `wss:` on HTTPS) and stamps
  every command (`ping`, `mode.set`) with a per-connection `seq` starting at 1
  and reset on reconnect, so `error.ref_seq` can identify the command it
  answers. It sends `ping` every 10 s and measures `pong` RTT, and reconnects
  with exponential backoff from 0.5 s to 5 s. Reconnecting does not reset the
  rendering state: the server sends a fresh snapshot (FR-504), and the decision
  feed keeps its local history (seeded from `medical.state.last_decision` after
  a reload).
- `src/state.ts` bounds the decision feed to 20 entries and, when `status.mode`
  leaves `medical` (idle, lab, or a failed start), resets the IMU, tag, episode
  and skeleton tiles to their “not active” copy; the next `medical.*` states
  fill them again. A finished episode also clears the skeleton. It remembers
  the `seq` of the last `mode.set`; an `error` whose `ref_seq` matches it is
  shown next to the mode buttons (other errors go to the notice line).
- The protocol exposes a single `imu` state for the wrist IMUs. The dashboard
  labels the tile “IMU L/R” and notes that the state is the one reported by the
  protocol; it does not invent per-side values.

## Development mock

`?mock=1` (development only) imports `docs/protocol/v1/examples` with
`import.meta.glob` and replays them through a WebSocket-like object. It starts
with an active episode, cycles decisions through the four tags, animates a
plausible HALFBUST_62 pose at the fixture's 1.4 Hz, ends the episode after 22 s
with the example contact summary, then starts a new one. It also answers
`ping` with `pong` and implements `mode.set`/`mode.progress`, including the
`busy` error; leaving `medical` drops its `medical.*` states, so the tiles go
back to their “not active” copy. The pose is synthesized in `src/mock.ts`
because the example frame has all keypoints at one point and would not show the
skeleton.

The mock is behind `import.meta.env.DEV` and a dynamic `import()`. Vite removes
the branch and the module from the production bundle; `web/dist` contains no
fixture data and no mock badge.

`?mock=1&trigger=vision` replays the camera-only variant (`trigger: vision`, `imu: n/a`,
camera and vision lines in the connection log).

## Layout and accessibility

- Designed for 1280×720 and 1920×1080 booth screens; below 1180 px it falls
  back to two columns and below 780 px to one.
- Type sizes scale with the viewport, with a further increase at ≥1600 px for
  onlookers at 2–3 m. All text/background pairs in the palette are above the
  WCAG AA 4.5:1 contrast threshold (most above 8:1).
- Severity is always written out (e.g. “Pericoloso”), never colour alone; the
  newest decision also carries the word “ultima”. The skeleton canvas has an
  `aria-label` describing the number of visible points, and the mode buttons
  expose `aria-pressed` and are disabled during a switch.
- No animations or flashing patterns are used.

## Files

```text
web/
  index.html          static shell, Italian copy
  vite.config.ts      base "./", dev filesystem access for the mock
  tsconfig.json       strict TypeScript, no emit
  src/
    protocol.ts       protocol v1 types and validating parser
    client.ts         WebSocket connect, ping, backoff reconnect
    state.ts          snapshot state and decision history (max 20)
    views.ts          header, health, live tiles, feed, episode panel
    skeleton.ts       HALFBUST_62 canvas renderer
    labels.ts         Italian labels and formatters
    dom.ts            small text-safe DOM helpers
    mock.ts           dev-only fixture replay and synthetic pose
    main.ts           wiring
    styles.css        booth layout and palette
```
