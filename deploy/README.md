# Deploying the showcase

`deploy/deploy.sh` installs the showcase on the KV260 and enables it at boot.
It copies the repository, the built dashboard and the two compiled models (YOLOX
and MoveNet), syncs the Python environment, installs two systemd units and
verifies the result through `/health`.

```sh
npm --prefix web ci && npm --prefix web run build
deploy/deploy.sh --model build/vitis/yolox_nano/compiled/yolox_nano.xmodel
```

`--movenet-model PATH` defaults to
`build/vitis/movenet/compiled/movenet.xmodel` (from `tools/vitis/zoo_movenet.sh`).
Options (`deploy/deploy.sh --help`): `--host` (default `10.42.0.83`), `--user`
(default `kria`), `--wheels DIR` for an offline install, `--no-restart`,
`--dry-run` (prints every command and touches nothing).

## What it does

| Step | Effect |
|---|---|
| 1 | `rsync --delete` of the repository to `/home/kria/cpsa`, keeping `.venv/`, `models/`, `wheels/`, `runs/` and `web/dist/`. |
| 2 | `web/dist/` (with `--delete`) and both models as `/home/kria/cpsa/models/<basename>`. |
| 3 | Optional offline wheels. |
| 4 | `.venv` with `--system-site-packages` (the VART `xir`/`vart` bindings are system packages) and `uv sync --frozen`; fails if `import xir, vart` does not work. |
| 5 | Installs and enables `cpsa-dpu.service` and `cpsa-showcase.service`, removes the superseded `cpsa-supervisor.service`, starts the DPU loader and (re)starts the showcase. |
| 6 | Requires `systemctl is-active` on both units and `/health` with `state=running` and `fps > 0` (up to 120 s). |

The model basenames must be the ones `config/showcase.yaml` names
(`models/yolox_nano.xmodel`, `models/movenet.xmodel`), because the configuration
is the only place the paths appear. `movenet.enabled: false` in that file runs
the YOLOX-only showcase without a compiled MoveNet model.

## Prerequisites

On the PC: `ssh`, `rsync`, `curl`, `python3`, key-based SSH access to the board,
and a built dashboard in `web/dist`.

On the board:

- Ubuntu 22.04 with the Kria firmware and the Vitis AI 3.5 runtime installed in
  the system dist-packages (`xir`, `vart`), so the DPU app
  `kv260-benchmark-b4096` can be loaded.
- `uv` on `PATH` (`~/.local/bin` is enough) and a package index, unless
  `--wheels` is used.
- Password-less `sudo` for the deploy user (the script installs units and starts
  services over a `BatchMode` SSH session, which has no tty to answer a password
  prompt).
- Python 3.10 (`uv venv --python 3.10`); `uv` downloads it if the board's system
  Python is a different version.

## The units

`deploy/systemd/cpsa-dpu.service` loads the DPU application once at boot
(`xmutil loadapp kv260-benchmark-b4096`) and waits for the device-tree overlay,
because `xmutil` returns 0 even when the load failed. `cpsa-showcase.service`
then runs `/home/kria/cpsa/.venv/bin/cpsa-showcase --config
config/showcase.yaml` as `kria` in `/home/kria/cpsa`, restarting every 3 s:

- `KillMode=mixed` sends `SIGTERM` to the pipeline (process A) only; A stops the
  web server (process B), releases the camera and the DPU and unlinks the shared
  memory. `TimeoutStopSec=15` kills whatever is left.
- `StartLimitIntervalSec=0` keeps restarting: the showcase runs unattended for a
  whole day, and a camera that was unplugged is picked up again by the
  pipeline's own backoff.

## Offline install

Prepare a directory of `linux-aarch64` / `cp310` wheels on the PC (including the
build backend: `hatchling`, `editable` support packages, `setuptools`, `wheel`)
and pass it:

```sh
deploy/deploy.sh --model ... --wheels ~/cpsa-wheels
```

The board then installs with `uv sync --offline --find-links
/home/kria/cpsa/wheels`, so no network is needed.

## Checking a deployment by hand

```sh
systemctl is-active cpsa-dpu.service cpsa-showcase.service
curl -s http://127.0.0.1:8080/health        # {"state": "running", "fps": 59.4, ...}
journalctl -u cpsa-showcase -b --no-pager | tail -40
```

The journal carries the startup lines
(`cpsa-showcase 0.1.0: config config/showcase.yaml`, `serving on
http://0.0.0.0:8080`), the per-stage timing line every five seconds
(`showcase stages over <n> frames: camera=… preprocess=… dpu=… decode=…
movenet_dpu=… movenet_decode=… draw=… total=… total_fps=…`) and one line per camera stop and reopen.

The dashboard is at `http://<board>:8080/`. The video pane is
`/video.mjpeg`; nothing is encoded until a browser connects.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/health` reports `state: error`, `errors: ["camera_missing"]` | No V4L2 node matched `camera.match`. Check `v4l2-ctl --list-devices`, the USB cable, and that `kria` is in the `video` group (`sudo usermod -aG video kria`, log in again). |
| `/health` reports `errors: ["model_missing"]` | `models/yolox_nano.xmodel` or `models/movenet.xmodel` is missing; re-run the deploy with `--model` / `--movenet-model` (or set `movenet.enabled: false`). |
| `/health` reports `errors: ["dpu_error"]` | VART could not load the graph: `xdputil query` must report `DPUCZDX8G_ISA1_B4096` with fingerprint `0x101000016010407`. A stale `/tmp/DPU_0` lock file, or a unit started before `cpsa-dpu.service`, are the usual causes. |
| `cpsa-showcase` exits with code 2 | Startup failure: the journal names the reason (`cannot start the showcase: <code>: <message>`). The process published `status` with `state: "error"` before exiting, so `/health` reports the same. |
| The deploy fails at step 4 | `xir`/`vart` do not import from `.venv`: the VART bindings must be in the system dist-packages, and the venv must be created with `--system-site-packages` (the script rebuilds one that was not). |
| fps collapses or the video stutters | Something else is using the DPU or the camera: `systemctl status cpsa-dpu`, `ps aux \| grep cpsa`. Run `scripts/profile_showcase.py` with the unit stopped to see which stage is slow. |
| The board is unreachable after a power cut | `cpsa-showcase` waits for `cpsa-dpu.service`; if the DFX manager did not come up, `systemctl restart dfx-mgr` and then `systemctl restart cpsa-showcase`. |

## Undeploying

```sh
sudo systemctl disable --now cpsa-showcase.service
sudo systemctl disable cpsa-dpu.service
sudo rm /etc/systemd/system/cpsa-showcase.service
sudo systemctl daemon-reload
```

Model, venv, dashboard and shared memory go away with `/home/kria/cpsa`;
`/dev/shm/cpsa-*` is unlinked by the process itself on a clean stop.
