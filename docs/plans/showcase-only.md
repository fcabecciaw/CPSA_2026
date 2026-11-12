# YOLOX-nano showcase only — Implementation Plan

**Status:** READY (2026-11-12)
**Goal:** the repository and the KV260 run exactly one demo, "Cosa vede l'IA": YOLOX-nano (COCO-80) on the F200
RGB camera at 640x480, **60 fps end to end**, boxes + Italian labels drawn on a live MJPEG stream and a
`showcase.state` summary on a slim dashboard. Everything else is deleted.
**Spec (user, 2026-11-12, verbatim):** "drop support for all demos other than the one using yolox-nano. The new,
only, real, true demo will be yolox-nano. Hard stop. No support for the old ones, no keeping the infrastructure
working for them. Just prepare the most optimized setup possible to run the model for the crowd pleaser demo. You're
allowed to remove anything you consider useless for this task. You can hard break with the old code."
Earlier: "aim at 60fps".
**Supersedes:** `docs/plans/hands-demos.md`, `docs/plans/hands-crop.md`, the medical plans, and the multi-mode parts
of `docs/plans/ai-showcase.md` (its F-S1 detector contract, AC-S2 reference check and facts remain valid).
**Non-goals:** the cognitive/agent layer (a later plan; it will attach to the WebSocket below), model selection,
any training.
**Constraints:** board `ssh kria@10.42.0.83`; DPU fingerprint `0x101000016010407`; the model comes from
`tools/vitis/zoo_yolox_nano.sh` (zoo QAT int xmodel compiled with our arch). Pushing/merging needs the user's OK.

## Target architecture

One process, no supervisor, no modes. It starts at boot and runs until stopped.

```
cpsa-showcase (systemd, after cpsa-dpu)
  camera thread : V4L2 YUYV 640x480@60 (manual exposure/gain) -> cv2 YUYV->RGB -> latest-frame slot (drop old)
  pipeline      : frame N preprocess -> DPU execute_async ; meanwhile decode+draw frame N-1 ; wait
                  -> overlay frame -> video slot ; showcase.state (<= 10 Hz) -> server
  server (aiohttp, own thread/loop): GET / (web/dist), GET /health, GET /video.mjpeg (shared encoder, up to 60 fps),
                  GET /ws (hello, status 1 Hz, showcase.state, error, pong)
```

Package layout after the change (hard break; old modules are deleted, code is moved where useful):

| Module | Content | From |
|---|---|---|
| `cpsa/__main__.py` | `main()`: config, signals, wiring, exit codes | new (small) |
| `cpsa/config.py` | `ShowcaseConfig` (camera, model, labels, web, log) from one YAML | rewrite of `application/config.py` |
| `cpsa/camera.py` | V4L2 capture, node auto-select by card name, controls, cv2 YUYV→RGB | `adapters/camera/uvc.py`, `devices.py` |
| `cpsa/dpu.py` | VART runner, fix-point LUT quantize/dequantize | `adapters/pose/dpu.py` helpers, `adapters/detect/yolox_dpu.py` |
| `cpsa/yolox.py` | preprocess, int8-prefiltered decode, NMS, unletterbox, `Detection`, COCO labels | `vision/yolox_decode.py`, `detection.py`, `coco.py`, `pose_decode` letterbox/NMS |
| `cpsa/overlay.py` | box/label drawing | `board/showcase_mode.draw_overlay` |
| `cpsa/pipeline.py` | the loop above, stats (fps, stage ms) | `board/showcase_mode.py` |
| `cpsa/server.py` | static, health, MJPEG, WebSocket | trimmed `adapters/web/server.py` |
| `cpsa/protocol.py` | envelope + builders for the messages below | `protocol/system.py`, `protocol/showcase.py` |

Kept tools: `tools/vitis/{Dockerfile,run_vitis.sh,compile.sh,inspect_xmodel.py,zoo_yolox_nano.sh,yolox_reference.py,README.md}`,
`scripts/profile_showcase.py`, `scripts/check_camera.py` if still useful.
Deleted: every other module under `src/cpsa/` (medical, hands, rps, hologram, IMU, actuators, BlueCoin, recorder,
replay, supervisor, vision analytics, native Kalman), `ml/`, `models/`, `native/`, other `scripts/` and
`tools/vitis/` files, `docs/specs/`, `docs/notes/`, the superseded plans, their tests and fixtures, protocol
schemas of removed messages. `PRODUCT.md` and `README.md` are rewritten short. `pyproject.toml`: only the
dependencies the app uses (numpy, pyyaml, opencv-python-headless, linuxpy, aiohttp; dev: pytest, ruff, mypy,
jsonschema); one console script `cpsa-showcase`.

## Frozen interfaces

### F-K1 — protocol (WebSocket `/ws`, envelope unchanged: `{"v": 1, "type", "seq", "data"}` as in `protocol/system.py`)

Server → client:
- `hello` `{ "server_version": str, "protocol_version": 1 }`
- `status` (1 Hz and on change) `{ "state": "starting"|"running"|"error", "camera": "ok"|"missing"|"error",
  "model": str, "fps": float|null, "dpu_ms": float|null, "errors": [str], "uptime_s": float }`
- `showcase.state` (<= 10 Hz) — unchanged from `docs/plans/ai-showcase.md` F-S2
  (`model, width, height, fps, dpu_ms, total_ms, objects[<=20], counts[]`)
- `error` `{ "code": str, "message": str, "ref_seq"?: int }`, `pong` `{ "id": any }`

Client → server: `ping` `{ "id": any }` only. Unknown types → `error` `bad_command`.
Schemas in `docs/protocol/v1/` only for these messages (+ envelope, client-envelope), with examples and a test that
validates examples and builder output.

### F-K2 — HTTP

`GET /` and static files from `web/dist`; `GET /health` → `{"state", "fps", "uptime_s"}`;
`GET /video.mjpeg` multipart JPEG, rate = `web.video_fps` (default 60), quality `web.jpeg_quality` (default 70).

### F-K3 — config (`config/showcase.yaml`, the only config file)

```yaml
camera: { device: auto, match: RealSense, width: 640, height: 480, fps: 60, exposure: 150, gain: <tuned>, power_line_hz: 50 }
model:  { path: models/yolox_nano.xmodel, conf_thr: 0.35, iou_thr: 0.45 }
labels: it            # it | en
web:    { host: 0.0.0.0, port: 8080, video_fps: 60, jpeg_quality: 70, static_dir: web/dist }
log:    { level: INFO }
```

## Verification contract

- **AC-K1** — only the showcase remains: no module, test, script, config, schema, doc or dependency of the removed
  demos (`rg -i "medical|hologram|rps|hand21|imu|bluecoin|metawear|kalman|recorder" src tests config deploy web/src`
  returns nothing relevant). Evidence: grep + tree listing.
- **AC-K2** — detector correctness preserved: unit tests for decode/prefilter/letterbox; `tools/vitis/yolox_reference.py check`
  still passes (min IoU >= 0.9, 0 failures).
- **AC-K3** — app contract: starts with a fake camera + fake DPU runner; publishes hello/status/showcase.state;
  `/health`, `/video.mjpeg`, `/ws` work; missing model → status `error` with `model_missing` and exit code != 0 at
  startup; camera unplugged → status camera `missing`/`error` and recovery on replug (as today's source does).
- **AC-K4** — performance on the board: `scripts/profile_showcase.py` (services stopped) shows a loop <= 16.7 ms per
  frame with DPU/CPU overlap; live `/health` fps >= 55 sustained for 60 s with the camera at 60 fps;
  `/video.mjpeg` delivers >= 55 fps to one client. Evidence: profile table, health samples.
- **AC-K5** — web: one view (top bar: title, connection, fps, latency; left video; right counts + objects), no mode
  buttons; `npm run typecheck && npm run build` pass; screenshots with the mock.
- **AC-K6** — deploy: `deploy/deploy.sh --model build/vitis/yolox_nano/compiled/yolox_nano.xmodel` installs the app,
  disables and removes `cpsa-supervisor.service`, enables `cpsa-showcase.service`, removes obsolete files on the
  board (`~/cpsa` synced with `--delete`, keeping `.venv`; old models removed), and the health check passes.
- **AC-K7** — gate: `uv run --no-sync pytest -q && uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy src`
  and `cd web && npm run typecheck && npm run build` all green (the old kalman hygiene failure disappears with `native/`).

## Tasks

### K0 — performance (running as S5)
Decode int8 prefilter, cv2 preprocess, cv2 YUYV conversion, DPU/CPU overlap, 60 fps camera settings, MJPEG rate,
`scripts/profile_showcase.py`. The medical-compatibility requirements in its brief are void now. Integrated before K1.

### K1 — backend hard cut (after K0 is integrated)
**Files:** everything outside `web/`. **Actions:** build the target layout by moving the kept code and deleting the
rest; rewrite config, `__main__`, server, protocol, deploy script, systemd unit (`deploy/systemd/cpsa-showcase.service`,
keep `cpsa-dpu.service`), `config/showcase.yaml`, `pyproject.toml` (+ `uv lock`), `README.md`, `PRODUCT.md`,
`deploy/README.md`, `tools/vitis/README.md`, `docs/protocol/v1/` (F-K1 only), tests for the kept code
(including a small architecture/hygiene test that still makes sense). Keep `docs/plans/ai-showcase.md` and this plan;
delete the other plans.
**Verification:** full Python gate (AC-K7); `bash -n deploy/deploy.sh`; `deploy/deploy.sh --dry-run ...`;
`tools/vitis/yolox_reference.py check` (AC-K2).

### K2 — web hard cut (parallel with K0/K1; only `web/`)
**Actions:** remove modes, mode buttons, hologram/rps/medical/hand code, types and mocks; one showcase view per
AC-K5 driven by F-K1 (`hello`, `status`, `showcase.state`); mock emits F-K1 messages; keep the D1/S3 look.
**Verification:** `cd web && npm run typecheck && npm run build`; screenshots 1280x720 and 1920x1080.

### K3 — board (coordinator)
Deploy (AC-K6), measure (AC-K4), user looks at boxes on real objects.

## Open decisions / risks

- **Exposure at 60 fps** (<= ~16 ms) darkens the image: gain must be tuned on site; the config makes it a one-line change.
- **MJPEG at 60 fps** costs ~13 ms of a core per frame; if CPU-bound, encode at 30 fps while detection runs at 60.
- **Hard break**: old demos are recoverable only from git history (commit before K1: current branch head).
