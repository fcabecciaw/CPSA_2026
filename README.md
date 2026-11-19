# CPSA_2026 — Cosa vede l'IA

A live object-detection showcase on the Kria KV260: a 60 fps camera feed, a
COCO-80 detector (YOLOv5-nano or YOLOX-nano) running on the DPU, boxes and
Italian labels drawn on the frame, a MoveNet skeleton view, and a browser
dashboard that shows the video next to the list of what the model sees.

The whole system is one application: `cpsa-showcase`.

- To run the demo on a board that is already set up, go to
  [Running the demo](#running-the-demo).
- To prepare a board from a blank microSD card, follow
  [Setting up the board](#setting-up-the-board), then
  [Preparing the PC](#preparing-the-pc) and [Deploying](#deploying).

## What it does

- One Intel RealSense F200 RGB camera at 640x480, 60 fps, manual exposure.
- YOLOv5-nano (640x640) or YOLOX-nano (416x416, letterboxed) on the KV260 DPU,
  decoded on the ARM cores with an int8 prefilter that keeps the decode inside
  the frame budget. The dashboard switches between them live.
- MoveNet single-pose on the DPU for the "Scheletro" view.
- Boxes and `"<label> <score>%"` tags drawn on the very frame the model saw, so
  the video pane and the label pane can never disagree.
- A dashboard at `http://<board>:8080/` with the MJPEG stream and the protocol
  v1 WebSocket, plus `/health` for the deploy script and for monitoring.

## Setting up the board

This section takes a KV260 from a blank microSD card to a board that
`deploy/deploy.sh` can install the showcase on. Every version below is the one
observed on the reference board on 2026-11-19.

| Component | Reference board |
|---|---|
| OS | Ubuntu 22.04.5 LTS for Kria (`oem-limerick-kria-meta`), kernel `5.15.0-1080-xilinx-zynqmp` |
| Python | 3.10.12 (system) |
| XRT | `xrt` 2.13.479 (Ubuntu Xilinx PPA) |
| DPU firmware | `xlnx-firmware-kv260-benchmark-b4096` 0.12, DPU `DPUCZDX8G_ISA1_B4096`, fingerprint `0x101000016010407`, 300 MHz |
| Vitis AI runtime | VART 3.5 (tag `v3.5`), built natively, installed under `/usr/local` |
| uv | 0.13.7 in `~/.local/bin` |
| User | `kria` (uid 1000), home `/home/kria` |

### What you need

- A Kria KV260 Vision AI Starter Kit, its power supply and a microSD card of
  32 GB or more.
- A Creative VF0800 camera. It enumerates as an Intel RealSense F200
  (`8086:0a66`) and works with the stock `uvcvideo` driver.
- A **powered** USB hub for the camera. With the camera plugged directly into
  the board, the SD card hit a read error and the root file system was
  remounted read-only.
- An Ethernet cable between the board and a Linux laptop. The laptop builds the
  dashboard, holds the compiled models, deploys and shows the dashboard.

### 1. Flash Ubuntu

1. Write Canonical's Ubuntu 22.04 image for Kria (the classic desktop image)
   to the microSD card, for example with `rpi-imager` or `dd`.
2. Boot the board with the card and complete the first login.
3. Make sure the account that will run the showcase is `kria` with home
   `/home/kria`. The systemd units and the deploy script hard-code both. If the
   first-boot user has another name, create `kria`:

   ```sh
   sudo adduser kria
   sudo usermod -aG sudo,video kria
   ```

   The `video` group gives access to `/dev/video*`.

### 2. Connect the board to the laptop

Share the laptop's Ethernet connection with NetworkManager. The laptop takes
`10.42.0.1/24` on that interface and hands the board an address by DHCP. The
reference board gets `10.42.0.83`, the default host of `deploy/deploy.sh`.

```sh
# On the laptop. <iface> is the Ethernet interface, for example enp0s31f6.
nmcli connection add type ethernet ifname <iface> con-name kria-share ipv4.method shared
nmcli connection up kria-share
```

If the board does not answer at `10.42.0.83`, find its address with one of:

```sh
ip neigh show dev <iface>
nmap -sn 10.42.0.0/24
journalctl -u NetworkManager | grep -i dnsmasq-dhcp
```

Pass a different address to every script with `--host <ip>`.

### 3. SSH key and password-less sudo

`deploy/deploy.sh` runs over `ssh -o BatchMode=yes`, so it cannot type a
password. Install a key from the laptop:

```sh
ssh-copy-id kria@10.42.0.83
ssh -o BatchMode=yes kria@10.42.0.83 true && echo key-ok
```

Then allow the few commands the deploy and the measurement scripts run with
`sudo`, without a password. On the board:

```sh
echo 'kria ALL=(root) NOPASSWD: /usr/bin/systemctl, /usr/bin/xmutil, /usr/bin/install' \
  > /tmp/cpsa-deploy.sudoers
sudo visudo -cf /tmp/cpsa-deploy.sudoers
sudo install -m 0440 -o root -g root /tmp/cpsa-deploy.sudoers /etc/sudoers.d/cpsa-deploy
sudo -k; sudo -n systemctl --version | head -1    # must not ask for a password
```

Everything else still asks for the password.

### 4. Xilinx packages: XRT and the DPU firmware

On the board:

```sh
sudo add-apt-repository ppa:xilinx-apps --yes
sudo add-apt-repository ppa:ubuntu-xilinx/default --yes
sudo apt update && sudo apt upgrade --yes
sudo apt install --yes xrt xlnx-firmware-kv260-benchmark-b4096 v4l-utils rsync curl git
```

The firmware package installs the DPU application under
`/lib/firmware/xilinx/kv260-benchmark-b4096/`. Check that the DFX manager sees
it and that it loads:

```sh
sudo xmutil listapps                         # kv260-benchmark-b4096 is listed
sudo xmutil unloadapp; sudo xmutil loadapp kv260-benchmark-b4096
ls /sys/kernel/config/device-tree/overlays    # kv260-benchmark-b4096 appears
```

`xmutil loadapp` exits 0 even when the load fails, so the overlay directory is
the only reliable sign of success. After the deploy, `cpsa-dpu.service` does
this load at every boot.

Point VART at the firmware's `xclbin`:

```sh
echo 'firmware: /lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin' \
  | sudo tee /etc/vart.conf
```

### 5. Build the Vitis AI 3.5 runtime (VART)

The showcase imports the `xir` and `vart` Python bindings from the system
`dist-packages`. No Ubuntu 22.04 package ships them for this board, so they are
built natively from the Vitis AI 3.5 sources: `unilog`, `xir`,
`target_factory` and `vart`, in this order. The build takes a while; `-j2`
keeps it inside the board's 4 GB of RAM.

Install the build dependencies:

```sh
sudo apt install --yes build-essential cmake pkg-config uuid-dev \
  libgoogle-glog-dev libgflags-dev libprotobuf-dev protobuf-compiler \
  libjson-c-dev libssl-dev libboost-all-dev libeigen3-dev \
  pybind11-dev python3-dev libpython3-dev
```

Get the sources and patch one call that XRT 2.13 no longer provides
(`xclIPSetReadRange`; without the patch, `xrt_device_handle_imp.cpp` does not
compile):

```sh
mkdir -p ~/vart_build && cd ~/vart_build
git clone --depth 1 --branch v3.5 https://github.com/Xilinx/Vitis-AI.git
sed -i 's|r = xclIPSetReadRange(x.handle, x.cu_index, 0x10, 0x1F0);|r = 0;  // XRT 2.13 has no xclIPSetReadRange()|' \
  Vitis-AI/src/vai_runtime/vart/xrt-device-handle/src/xrt_device_handle_imp.cpp
grep -n 'XRT 2.13 has no' Vitis-AI/src/vai_runtime/vart/xrt-device-handle/src/xrt_device_handle_imp.cpp
```

Build and install the four components:

```sh
cd ~/vart_build
SRC=Vitis-AI/src/vai_runtime
COMMON=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
        -DCMAKE_PREFIX_PATH=/usr/local -DBUILD_SHARED_LIBS=ON -DBUILD_TEST=OFF
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -s")
PY=(-DBUILD_PYTHON=ON -DPYTHON_SITE_PACKAGES=lib/python3.10/dist-packages)

build() {  # build <component> <extra cmake flags...>
  local name=$1; shift
  cmake -S "$SRC/$name" -B "build/$name" "${COMMON[@]}" "$@" &&
  cmake --build "build/$name" -j2 &&
  sudo cmake --install "build/$name"
}

build unilog         -DBUILD_PYTHON=OFF &&
build xir            "${PY[@]}" -DBUILD_DOC=OFF &&
build target_factory -DBUILD_PYTHON=OFF -DENABLE_IPU=OFF &&
build vart           "${PY[@]}" -DENABLE_CPU_RUNNER=OFF -DENABLE_SIM_RUNNER=OFF &&
sudo ldconfig && echo VART_BUILD_OK
```

Check the result:

```sh
python3 -c "import xir, vart; print(xir.__file__, vart.__file__)"
# /usr/local/lib/python3.10/dist-packages/xir.cpython-310-aarch64-linux-gnu.so ...
ls /usr/local/lib/libvart-dpu-runner.so.3.5.0
```

`xdputil query` is the upstream diagnostic that prints the loaded DPU's
architecture and fingerprint. It lives in `vai_library`, not in VART. The
reference board has it from a separate, minimal build of its `tools_extra_ops`
Python module. The showcase does not need it: `/health` reports a DPU that VART
cannot use as `dpu_error`.

### 6. Install uv

`deploy/deploy.sh` builds the application's venv on the board with `uv`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh    # installs ~/.local/bin/uv
ssh -o BatchMode=yes kria@10.42.0.83 'command -v uv'  # from the laptop: must print a path
```

The deploy creates the venv with `--system-site-packages`, so it sees the
system `xir`/`vart`, and pins Python 3.10. For a board without internet access,
see [Offline install](deploy/README.md#offline-install).

### 7. Connect the camera

Plug the F200 into the powered hub and the hub into the board. Check that the
RGB node appears:

```sh
v4l2-ctl --list-devices      # a card whose name contains "RealSense"
```

The showcase picks the node itself at every start (`camera.device: auto`): the
F200 exposes several nodes, and the application opens the one that offers
YUYV and not Z16, so the node numbers do not matter.

The board is ready. Continue on the laptop.

## Preparing the PC

The laptop needs `ssh`, `rsync`, `curl`, `python3`, `uv`, Node.js with `npm`
(Node 22 is the reference), and Docker for building the models.

### Build the models

The three compiled models are git-ignored build outputs of `tools/vitis/`. The
scripts run the Vitis AI 3.5 compiler inside the pinned Docker image and pin the
DPU fingerprint of the board's firmware:

```sh
tools/vitis/zoo_yolox_nano.sh      # build/vitis/yolox_nano/compiled/yolox_nano.xmodel
tools/vitis/zoo_yolov5_nano.sh     # build/vitis/yolov5_nano/compiled/yolov5_nano.xmodel
tools/vitis/zoo_movenet.sh         # build/vitis/movenet/compiled/movenet.xmodel
```

YOLOX-nano is required. YOLOv5-nano is optional: without it, the dashboard
offers YOLOX only. MoveNet is required unless `movenet.enabled: false` in
`config/showcase.yaml`. See [`tools/vitis/README.md`](tools/vitis/README.md)
for the images, the YOLOv5 prebuilt route and the reference checks.

### Build the dashboard

```sh
npm --prefix web ci
npm --prefix web run build          # writes web/dist/
```

## Deploying

```sh
deploy/deploy.sh --model build/vitis/yolox_nano/compiled/yolox_nano.xmodel
```

The script copies the repository to `/home/kria/cpsa`, copies `web/dist/` and
the models into `/home/kria/cpsa/models/`, builds `.venv` with `uv sync
--frozen`, installs and enables `cpsa-dpu.service` and `cpsa-showcase.service`,
and waits up to 120 s for `/health` to report `state: running` with a positive
fps. A successful run ends with that health check passing.

Useful options: `--host <ip>`, `--yolov5-model PATH`, `--movenet-model PATH`,
`--no-restart`, `--dry-run`. [`deploy/README.md`](deploy/README.md) describes
every step, the units, the offline install and the troubleshooting table.

The deploy keeps what the board owns: `.venv/`, `models/`, `runs/` and the live
calibration in `config/calibration.json`.

## Running the demo

### On the day

1. Connect the camera to the powered hub and the hub to the board, and the
   board's Ethernet to the laptop.
2. Power on the board. `cpsa-dpu.service` loads the DPU application and
   `cpsa-showcase.service` starts the showcase; nothing has to be typed.
3. On the laptop, open `http://10.42.0.83:8080/` and make the browser full
   screen.
4. Check that the board is healthy:

   ```sh
   curl -s http://10.42.0.83:8080/health   # "state": "running" and fps near 60
   ```

The dashboard opens in presentation mode. Its buttons switch between the
**Oggetti** view (the object detector) and the **Scheletro** view (MoveNet).
The button at the top right, or the `d` key, shows the technical details: the
fps gauge and the **Calibrazione** card, which switches the detector between
YOLOv5-nano and YOLOX-nano and tunes the YOLOX zoom and the camera image. The
card saves its values to `/home/kria/cpsa/config/calibration.json` on the board,
and they override `config/showcase.yaml` at the next start.

A camera that is unplugged and plugged back is picked up again without a
restart. If the showcase starts with no camera connected, it exits with
`camera_missing`, and systemd restarts it every 3 s until the camera appears.

The optional Italian voice of the dashboard comes from a separate service on
the laptop; see [`tools/tts/README.md`](tools/tts/README.md).

### Starting and stopping

On the board:

```sh
sudo systemctl stop cpsa-showcase       # the DPU stays loaded
sudo systemctl start cpsa-showcase
systemctl is-active cpsa-dpu cpsa-showcase
journalctl -u cpsa-showcase -b --no-pager | tail -40
```

To run it by hand instead of through systemd, stop the unit first, so the camera
and the DPU are free:

```sh
cd ~/cpsa
.venv/bin/cpsa-showcase --config config/showcase.yaml
```

Options: `--config PATH`, `--calibration PATH`, `--replay DIR`, `--version`.
Exit codes: `0` orderly stop, `1` unexpected failure, `2` startup failure
(unreadable configuration, missing model or camera, unreadable replay
directory) after publishing `status` with `state: "error"`;
`journalctl -u cpsa-showcase` explains it.

`--replay DIR` (or `camera.replay_dir` in the configuration) replaces the camera
with a loop of the JPG/PNG images of `DIR`, for an fps measurement with nobody in
front of the booth: see [Measuring the fps without the
camera](#measuring-the-fps-without-the-camera).

### When something is wrong

| Symptom | Check |
|---|---|
| The dashboard does not load | `ping 10.42.0.83`; `systemctl is-active cpsa-showcase` on the board. |
| `/health` shows `camera_missing` | The camera is not on the powered hub, or the hub has no power: `v4l2-ctl --list-devices`. |
| `/health` shows `model_missing` | A model named in `config/showcase.yaml` is not in `~/cpsa/models/`: deploy again. |
| `/health` shows `dpu_error`, or `cpsa-dpu` failed | `journalctl -u cpsa-dpu -b`; `sudo systemctl restart dfx-mgr cpsa-dpu cpsa-showcase`. |

The full table is in [`deploy/README.md`](deploy/README.md#troubleshooting).

## Architecture

Two processes, one configuration file, one optional calibration file, no other
state:

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
| `cpsa/camera.py` | V4L2 capture through `linuxpy`, node selection, controls, YUYV→RGB (`cv2`); the replay source for an fps measurement without the camera. |
| `cpsa/dpu.py` | VART graph introspection: subgraphs, tensor shapes and fix points, LUT quantization. |
| `cpsa/yolox.py` | YOLOX-nano detector: preprocessing, `submit`/`wait` pipelining, int8 decode, COCO labels. |
| `cpsa/yolov5.py` | YOLOv5-nano (640x640) detector: same seams, YOLOv5 anchors/exp decode, selectable live as `model.kind: yolov5`. |
| `cpsa/movenet.py` | MoveNet single-pose on the DPU: preprocessing, decoding, the 17 keypoints. |
| `cpsa/keypoints.py` | Keypoint stabilization between MoveNet and the overlay. |
| `cpsa/tracker.py` | Box stabilization between the detector and the overlay. |
| `cpsa/motion.py` | The motion check that keeps a still scene off the DPU. |
| `cpsa/calibration.py` | The live calibration of the debug panel and its file. |
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

## Configuration

One file, `config/showcase.yaml`, with seven sections and no unknown keys:

```yaml
camera:    # device (auto or /dev/videoN), match, width, height, fps, exposure,
           # gain, white_balance, backlight_compensation, saturation, power_line_hz,
           # mirror, replay_dir, replay_hold_frames, replay_jitter_px
tracker:   # enabled, iou_match, alpha, fast_alpha, deadband_iou, min_hits, max_misses,
           # still: enabled, motion_thr, force_every_s
model:     # path, kind (yolox | yolov5), yolov5_path, conf_thr, iou_thr, class_conf,
           # object_conf, zoom, zoom_center, alternate
movenet:   # path, enabled, publish_hz, smoothing
labels:    # it | en
web:       # host, port, video_fps, jpeg_quality, static_dir
log:       # level
```

`model.zoom` centre-crops the frame before the letterbox: `1.0` is the whole
frame, and `1.3`-`1.5` feed the central part of it into the same 416x416 input,
so a small object reaches the network larger without extra DPU work;
`zoom_center` moves that crop, and the boxes still come back in full-frame
pixels.

A zoom hides the frame edges, which is where a person walks in, so `model.alternate`
is on by default whenever `zoom` is above `1.0`: even frames run the whole frame
and report only persons, odd frames run the crop and report every other class.
It is one inference per frame, so the DPU cost does not change, and the tracker
ages each class group only on its own passes, so the person a zoom frame cannot
see is not dropped and neither is the bottle the whole-frame pass cannot see.
`alternate: false` goes back to running the configured zoom on every frame.

`camera.device: auto` resolves the node by card name at every start, so a
re-plug does not need a restart of the unit. The camera's automatic exposure,
white balance and backlight compensation are switched off/overridden at every
open, and `tracker` keeps the boxes still from frame to frame (`enabled: false`
publishes the raw detector boxes). `tracker.still` is the motion gate in front of
the DPU: while the 80x60 grey frame stops changing the loop draws the previous
boxes instead of running the detector, and a full detection still happens every
`force_every_s` seconds so a new still object cannot be missed.

The dashboard's Calibrazione card writes `config/calibration.json` on the board
(git-ignored, kept by the deploy). Its values override the configuration at
start; delete the file to go back to `config/showcase.yaml`.

## Protocol

`docs/protocol/v1/` defines protocol v1: the envelope, `hello`, `status`,
`showcase.state`, `error`, `pong`, the `ping` command and one example per
message, all validated against the JSON schemas by the test suite. Live video is
`GET /video.mjpeg` (`multipart/x-mixed-replace`), the health probe is
`GET /health` and the bundle is served from `web.static_dir`.

## Tests

On the PC:

```sh
uv sync --extra dev
uv run --no-sync pytest            # unit, contract and architecture tests
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src
cd web && npm ci && npm run typecheck && npm test && npm run build
```

The tests run on any machine: the DPU, the camera and the web server are driven
through the same seams the board uses (a fake `xir`/`vart`, a fake V4L2 stream,
a fake detector), and one test runs both processes end to end, over the real
shared memory and the real queue. Nothing in `tests/` needs the board.

On the board, with the services stopped:

```sh
sudo systemctl stop cpsa-showcase
uv run python scripts/profile_showcase.py --model models/yolox_nano.xmodel
uv run python scripts/profile_showcase.py --yolov5-model models/yolov5_nano.xmodel
```

That prints the per-stage cost (preprocess, DPU, decode, draw, JPEG) against the
16.7 ms frame budget. `--busy` measures the two synthetic decode lines on a
crowded frame (150 planted objects in clusters) instead of a near-empty one, so
the class-aware NMS is measured with the work a crowd in front of the booth
gives it.

### Measuring the fps without the camera

The fps of the whole app can be measured with nobody in front of the camera, on
a loop of photos (real COCO validation images with people and objects are
enough). Keep them out of git under `build/replay`, then on the board:

```sh
scripts/measure_replay.sh ~/cpsa/build/replay 15
```

The script stops `cpsa-showcase.service` (the DPU stays loaded by
`cpsa-dpu.service`), runs `cpsa-showcase --config config/showcase.yaml --replay
DIR` for 15 s per case, switches the view and the object detector over the
WebSocket (`model.set`, `calibration.set`) and prints the status fps and the
stage log of each case, then restores the service. Every image is held for
`camera.replay_hold_frames` frames (6, so ten scene changes per second at 60
fps) and shifted by up to `camera.replay_jitter_px` pixels (2), so the still gate
keeps inferring instead of freezing the measurement.

## YOLOv5-nano

`src/cpsa/yolov5.py` runs the model-zoo YOLOv5-nano at its native 640x640
(`4.6 G` float ops against YOLOX-nano's `1.0 G` at 416): same
preprocessing/`submit`/`wait`/int8-prefilter seams, YOLOv5's anchors and
`sigmoid(t) * 2` decode, and the zoo's per-cell, multi-label post-processing.
`model.kind: yolov5` starts on it, and the dashboard's Calibrazione card
switches the loaded detector live; the YOLOX zoom and alternation do not apply,
because YOLOv5 always sees the whole frame. A 640x480 frame needs no resize for
the square input (the letterbox only pads), so its rows are quantized straight
into the DPU input, and the decode thresholds the int8 class logits before it
takes their sigmoid.

The DPU artifact comes from `tools/vitis/zoo_yolov5_nano.sh`; the reference
check (`tools/vitis/yolov5_reference.py`) is described in
[`tools/vitis/README.md`](tools/vitis/README.md), together with the zoo
packaging constraint that forces the prebuilt route.

## Building the model

`tools/vitis/` holds the quantize-and-compile pipeline (model zoo download,
calibration, `vai_q_pytorch`, `vai_c_xir`), the xmodel inspector and the decoder
reference check that proves the DPU output matches the float decode. See
[`tools/vitis/README.md`](tools/vitis/README.md).

## License

MIT.
