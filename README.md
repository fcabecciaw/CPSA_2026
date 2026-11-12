# CPSA_2026 — Cosa vede l'IA

A live object-detection showcase on the Kria KV260: a 60 fps camera feed, the
pre-trained YOLOX-nano detector running on the DPU, boxes and Italian labels
drawn on the frame, and a browser dashboard that shows the video next to the
list of what the model sees.

The whole system is one application: `cpsa-showcase`.

## What it does

- One Intel RealSense F200 RGB camera at 640x480, 60 fps, manual exposure.
- YOLOX-nano (COCO-80) on the KV260 DPU, letterboxed to 416x416, decoded on the
  ARM cores with an int8 prefilter that keeps the decode inside the frame
  budget.
- Boxes and `"<label> <score>%"` tags drawn on the very frame the model saw, so
  the video pane and the label pane can never disagree.
- A dashboard at `http://<board>:8080/` with the MJPEG stream and the protocol
  v1 WebSocket, plus `/health` for the deploy script and for monitoring.

## Architecture

Two processes, one configuration file, no other state:

```
        process A (cpsa-showcase)                    process B (cpsa-web)
  camera ──▶ detector ──▶ overlay ──┬──▶ shared memory frame ──▶ MJPEG (JPEG at 60 fps)
                                    └──▶ queue: status, showcase.state ──▶ WebSocket
```

- **A** owns the camera, the DPU, the overlay and the policy: it reads the
  newest frame from a two-slot slot, keeps two DPU jobs in flight, and never
  waits for the web server.
- **B** owns aiohttp: the static dashboard, `/health`, the MJPEG stream and the
  WebSocket. It copies the newest frame out of shared memory and encodes JPEG
  only while at least one client is watching, so an idle dashboard costs
  nothing.
- They share two things (`src/cpsa/channel.py`): a fixed 640x480x3 shared-memory
  frame with a sequence counter (double buffer, never torn, the reader drops
  stale frames) and a bounded queue of state snapshots (drop-oldest: every
  message replaces the previous one anyway).

Process A starts B before it opens the camera or the DPU, so the child is forked
before the Vitis AI bindings and the V4L2 device exist. On SIGTERM/SIGINT, A
stops B, releases the camera and the DPU and unlinks the shared memory.

### Modules

| Module | Responsibility |
|---|---|
| `cpsa/camera.py` | V4L2 capture through `linuxpy`, node selection, controls, YUYV→RGB (`cv2`). |
| `cpsa/dpu.py` | VART graph introspection: subgraphs, tensor shapes and fix points, LUT quantization. |
| `cpsa/yolox.py` | YOLOX-nano detector: preprocessing, `submit`/`wait` pipelining, int8 decode, COCO labels. |
| `cpsa/overlay.py` | Boxes and tags on the frame. |
| `cpsa/channel.py` | The frame channel and the message channel between the processes. |
| `cpsa/pipeline.py` | The loop: frame cadence, publishing, camera recovery, stage timings. |
| `cpsa/protocol.py` | Protocol v1 payload builders. |
| `cpsa/server.py` | HTTP, MJPEG and WebSocket. |
| `cpsa/config.py` | The one configuration file. |
| `cpsa/__main__.py` | Command line, signals, the two processes, exit codes. |

Dependencies point one way: `config`, `protocol`, `channel` and `dpu` are
leaves; `pipeline` and `server` may use them; `__main__` wires everything.
`tests/architecture/` enforces this.

## Installation

On the board (Ubuntu 22.04, Vitis AI runtime in the system dist-packages):

```sh
git clone <repository> ~/cpsa && cd ~/cpsa
uv venv --system-site-packages --python 3.10 .venv
uv sync --frozen
sudo install -m 0644 deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cpsa-dpu.service cpsa-showcase.service
```

The dashboard bundle must be built once (`npm --prefix web ci && npm --prefix
web run build`) and is deployed by `deploy/deploy.sh`, which does the whole
sequence, including the model, the venv and the health check. See
[`deploy/README.md`](deploy/README.md).

## Running

```sh
uv run cpsa-showcase --config config/showcase.yaml    # or: .venv/bin/cpsa-showcase
```

The dashboard is at `http://<board>:8080/`. Options: `--config PATH`,
`--version`. Exit codes: `0` orderly stop, `1` unexpected failure, `2` startup
failure (unreadable configuration, missing model or camera) after publishing
`status` with `state: "error"`; `journalctl -u cpsa-showcase` explains it.

## Configuration

One file, `config/showcase.yaml`, with six sections and no unknown keys:

```yaml
camera:    # device (auto or /dev/videoN), match, width, height, fps, exposure,
           # gain, white_balance, backlight_compensation, saturation, power_line_hz
tracker:   # enabled, iou_match, alpha, fast_alpha, deadband_iou, min_hits, max_misses,
           # still: enabled, motion_thr, force_every_s
model:     # path, conf_thr, iou_thr
labels:    # it | en
web:       # host, port, video_fps, jpeg_quality, static_dir
log:       # level
```

`camera.device: auto` resolves the node by card name at every start, so a
re-plug does not need a restart of the unit. The camera's automatic exposure,
white balance and backlight compensation are switched off/overridden at every
open, and `tracker` keeps the boxes still from frame to frame (`enabled: false`
publishes the raw detector boxes). `tracker.still` is the motion gate in front of
the DPU: while the 80x60 grey frame stops changing the loop draws the previous
boxes instead of running the detector, and a full detection still happens every
`force_every_s` seconds so a new still object cannot be missed.

## Protocol

`docs/protocol/v1/` defines protocol v1: the envelope, `hello`, `status`,
`showcase.state`, `error`, `pong`, the `ping` command and one example per
message, all validated against the JSON schemas by the test suite. Live video is
`GET /video.mjpeg` (`multipart/x-mixed-replace`), the health probe is
`GET /health` and the bundle is served from `web.static_dir`.

## Tests

```sh
uv run --no-sync pytest            # unit, contract and architecture tests
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src
cd web && npm ci && npm run typecheck && npm run build
```

The tests run on any machine: the DPU, the camera and the web server are driven
through the same seams the board uses (a fake `xir`/`vart`, a fake V4L2 stream,
a fake detector), and one test runs both processes end to end, over the real
shared memory and the real queue. Nothing in `tests/` needs the board.

On the board, with the services stopped:

```sh
sudo systemctl stop cpsa-showcase
uv run python scripts/profile_showcase.py --model models/yolox_nano.xmodel
```

That prints the per-stage cost (preprocess, DPU, decode, draw, JPEG) against the
16.7 ms frame budget.

## Building the model

`tools/vitis/` holds the quantize-and-compile pipeline (model zoo download,
calibration, `vai_q_pytorch`, `vai_c_xir`), the xmodel inspector and the decoder
reference check that proves the DPU output matches the float decode. See
[`tools/vitis/README.md`](tools/vitis/README.md).

## License

MIT.
