# Deploying the medical demo on the KV260

`deploy/deploy.sh` installs the demo on the board with one command and enables
it at boot. Two systemd units do the work:

| Unit | Role |
|---|---|
| `cpsa-dpu.service` | Oneshot that loads the DPU application `kv260-benchmark-b4096` after every boot (`xmutil loadapp`). Stays "active (exited)" so the supervisor can require it. |
| `cpsa-supervisor.service` | Runs `cpsa-supervisor --config config/board.demo.yaml` as `kria`, restarts it after a crash (`Restart=always`, `RestartSec=3`, no start rate limit). |

The supervisor unit is ordered after `cpsa-dpu.service` and
`bluetooth.service` and requires the DPU unit (system FR-201, NFR-05). It does
not wait for `network-online.target`: it binds `0.0.0.0` and the demo works
without any network link, so a booth laptop unplugged at boot cannot delay the
start.

## Prerequisites

The board prerequisites are the ones in the root README; this file only links
to them:

- [Installation](../README.md#installation): `board` extra, BlueZ, the
  `bluetooth` group, the `blue-st-sdk` and MetaWear notes. The two fixes those
  notes describe (`blue_st_sdk` on Python 3.10 and the debug `warble`) are
  applied by `deploy/deploy.sh` itself now, so they need no manual step; the
  notes remain the manual fallback.
- [Installation](../README.md#installation) for the bluepy helper
  capabilities. `deploy/deploy.sh` sets them on the venv's `bluepy-helper` after
  every `uv sync` (the step is idempotent and skipped when the capability is
  already there), so this is only the manual fallback:

  ```bash
  sudo setcap "cap_net_raw+eip cap_net_admin+eip" \
    /home/kria/cpsa/.venv/lib/python3.10/site-packages/bluepy/bluepy-helper
  ```

  The unit runs as `kria`, not root; the capability stays on the helper. The
  script uses `sudo -n` for `setcap` (and plain `sudo` for the unit commands),
  so `kria` needs password-less sudo: a password prompt cannot be answered over
  the `BatchMode=yes` SSH connection the deploy uses.

On the PC:

- `ssh`, `rsync`, `curl` and `python3` (the post-check parses the `/health`
  JSON).
- Key-based SSH to `kria@10.42.0.83` (the script uses `BatchMode=yes`).
- A built dashboard: `npm --prefix web run build` must produce
  `web/dist/index.html`.
- A compiled `.xmodel` for `--model`.

On the board:

- Ubuntu 22.04, Python 3.10, VART 3.5 and the `kv260-benchmark-b4096` app
  (NFR-04).
- `uv` on `PATH` for the first install. Without `uv`, the script reuses an
  existing `/home/kria/cpsa/.venv` if `cpsa-supervisor` is in it.
- `sudo` for installing the units, starting the services and setting the
  bluepy capability (password-less, see above).
- The VART `xir`/`vart` Python bindings importable from
  `/home/kria/cpsa/.venv`. The build in `docs/notes/vart-kv260.md` installs them
  into `/usr/local/lib/python3.10/dist-packages`, so the venv must see the
  system site packages. The script creates it with
  `uv venv --system-site-packages --python 3.10`, replaces a `.venv` that was
  made without that flag, and then checks `python -c 'import xir, vart'`. It
  never deletes a venv: the old one is moved to `.venv.bak-<timestamp>` and
  moved back if the rebuild or the import check fails, so a failed deploy leaves
  the board as it was.

### One-time board setup

These are outside `deploy.sh` on purpose: they change the system rather than the
venv, they survive every `uv sync`, and they were applied during the
2026-11-11 bring-up. They are listed here so a re-imaged board can be brought
back without the bring-up report.

**Password-less sudo.** The deploy runs `sudo install`, `sudo systemctl`,
`sudo -n setcap` and `sudo xmutil` over `BatchMode=yes` SSH, which has no tty
for a password. The drop-in used on the board (`mode 0440`, validated with
`visudo -cf`):

```sudoers
kria ALL=(root) NOPASSWD: /usr/bin/systemctl, /usr/sbin/setcap, /usr/bin/xmutil, /usr/bin/install
```

`/usr/bin/install` is in the list because the deploy installs the units with
`sudo install`; `install` as root is effectively root on this board, so keep the
file at `/etc/sudoers.d/cpsa-deploy` with mode 0440 and do not widen it.

**Bluetooth firmware for the RTL8761BU dongle.** Without these two files `hci0`
never comes up, no BlueCoin is discovered and the medical mode reports a missing
BlueCoin instead of starting:

```text
/lib/firmware/rtl_bt/rtl8761bu_fw.bin      1d7a9597349ad89344fa16c1913d3e39e9a12e966e417ca16871bc79bbe59edb
/lib/firmware/rtl_bt/rtl8761bu_config.bin  6c28a3f07c6a30ed208c4b64862a23f02b7d93543ea980edd24df16bab45095f
```

Installed from the `linux-firmware` package *without* installing the package,
so the board keeps its kernel and its firmware set:

```bash
mkdir -p /tmp/fw && cd /tmp/fw
apt download linux-firmware                  # 316 MB
dpkg-deb -x linux-firmware_*.deb .
sudo install -m 0644 ./lib/firmware/rtl_bt/rtl8761bu_fw.bin \
  ./lib/firmware/rtl_bt/rtl8761bu_config.bin /lib/firmware/rtl_bt/
sudo modprobe -r btusb && sudo modprobe btusb
hciconfig hci0                               # expect UP RUNNING
bluetoothctl show                            # expect Powered: yes
```

`/tmp/fw` keeps the deb and the extracted tree (316 MB); remove it by hand once
`hciconfig` shows the adapter.

**Build tools for the debug `warble`.** Step 6 of the deploy compiles `warble`
on the board, which needs `make`, `g++`, `libbluetooth-dev` (for libblepp) and
`libglib2.0-dev`:

```bash
sudo apt install -y build-essential libbluetooth-dev libglib2.0-dev
```

The unit files and the deploy destination are fixed to the user `kria` and the
directory `/home/kria/cpsa`. `--user` changes the SSH account only; that account
must be able to write `/home/kria/cpsa`. Edit the units and this script for a
different service user.

## One-command deploy

```bash
npm --prefix web run build
deploy/deploy.sh --model ~/build/vitis/trained/pose_halfbust62.xmodel
```

The script:

1. Checks SSH, `web/dist/index.html` and the `--model` file.
2. Rsyncs the repository to `kria@HOST:~/cpsa` (excluding `.git`, `ml/`,
   `build/`, `.venv`, `node_modules`, `runs/` and PC caches), then `web/dist`
   and the model to `~/cpsa/models/pose_halfbust62.xmodel`.
3. Runs `uv sync --frozen --python 3.10 --extra board --extra server --extra vision`
   on the board, in a venv created with `uv venv --system-site-packages`. An
   existing venv without that flag is replaced, because `xir`/`vart` are system
   packages: it is moved to `.venv.bak-<timestamp>` and only then is a new venv
   built, and it is moved back (and the deploy stops) if `uv venv`, `uv sync` or
   the `python -c 'import xir, vart'` check fails. A venv the deploy replaced is
   kept on the board and the script prints its path; delete it by hand once the
   demo runs. Without `--wheels` a rebuild needs a package index or a warm uv
   cache, so an offline board has to use `--wheels DIR`. `vision` provides OpenCV
   for the camera and evidence clips; `onnxruntime` is pulled in by that extra
   but is not used with `model.kind: xmodel` (the DPU path uses the board's
   `xir`/`vart` bindings).
4. Re-applies the two board patches that the sync reverts, both idempotent (see
   [Board patches](#board-patches-the-deploy-re-applies)): the `blue_st_sdk`
   Python 3.10 fix, as a guarded `sed` on the installed
   `dict_put_single_element.py`, and the debug build of `warble`, built once
   into `~/cpsa-build`, installed with
   `uv pip install --no-build-isolation`, verified with
   `python -c 'import mbientlab.warble'` and reported with the size of
   `libwarble.so`. `--skip-warble` skips the second one.
5. `sudo -n setcap "cap_net_raw+eip cap_net_admin+eip"` on the venv's
   `bluepy-helper`, skipped when `getcap` already reports the capability, and
   prints the result of `getcap`. Without it the IMU reports `imu_not_found`.
6. Installs both units into `/etc/systemd/system`, runs `daemon-reload` and
   enables them, starts `cpsa-dpu` once, then starts or restarts the supervisor
   exactly once (a `restart` unless `--no-restart`).
7. Checks `systemctl is-active` for both units, then polls
   `http://HOST:8080/health` for up to 120 s and reads the JSON: it requires
   `"mode_state": "running"`, so a 200 that reports `mode_state: failed` (no
   model, no BlueCoin, no camera) fails the deploy with a warning and pointers
   to the journal instead of claiming success.

| Option | Default | Meaning |
|---|---|---|
| `--host` | `10.42.0.83` | Board host or SSH alias. |
| `--user` | `kria` | SSH user; files still go to `/home/kria/cpsa` and the service runs as `kria`. |
| `--model` | required | `.xmodel` to install as `~/cpsa/models/pose_halfbust62.xmodel`. |
| `--wheels` | none | Offline wheel directory (see below). |
| `--no-restart` | off | Leave an already-running supervisor alone; start it only if inactive. |
| `--skip-warble` | off | Do not rebuild and reinstall the debug `warble`. Faster; the MetaWear then aborts with `*** buffer overflow detected ***` unless the debug build is already in the venv. |
| `--dry-run` | off | Print every planned command and change nothing on the board. |

`deploy/deploy.sh --dry-run --model path/to/model.xmodel` still reads local
files, but it never opens an SSH connection. A missing local prerequisite is
reported as a warning in dry-run so that the plan is still printed; a real
deploy stops there.

## Board patches the deploy re-applies

Two packages on the board need a fix that `uv sync --frozen` undoes on every
deploy. The bring-up applied them by hand; step 4 of the deploy now applies and
verifies both, so a fresh deploy leaves the board in the state the demo needs.
Both steps live on the board, need no PC-side preparation and are idempotent: a
second run of the deploy changes nothing.

**`blue_st_sdk` on Python 3.10.** The 1.4.0 sdist uses
`collections.MutableMapping`, which Python 3.10 moved to `collections.abc`;
without the fix the medical mode cannot start (`AttributeError: module
'collections' has no attribute 'MutableMapping'`). The step rewrites the
installed `~/cpsa/.venv/lib/python3.10/site-packages/blue_st_sdk/utils/dict_put_single_element.py`
with a guarded `sed` (`collections.MutableMapping` ->
`collections.abc.MutableMapping`, and `import collections` ->
`import collections.abc`), keeps the original once as
`dict_put_single_element.py.cpsa-orig`, and prints "already patched" when the
file is already fixed.

**Debug `warble`.** The release build of `warble` (a `metawear` dependency)
aborts the supervisor with `*** buffer overflow detected ***` as soon as the
MetaWear connects. `deploy.sh` builds the documented debug variant once and
reuses it:

- the sources are the PyPI sdist that `uv.lock` pins (PyWarble's GitHub
  repository is 404, and the sdist carries the `libblepp` sources). The version,
  URL and sha256 are read from `uv.lock`; the tarball is taken from
  `~/cpsa/wheels` when that exists, otherwise it is downloaded once and kept in
  `~/cpsa-build/`, and its sha256 is checked before it is unpacked;
- `~/cpsa-build/warble-<version>/` keeps the patched source tree and its build
  objects (`setup.py.orig` next to `setup.py`). `setup.py` is patched to build
  with `CONFIG=debug` and to take the library from `dist/debug` instead of
  `dist/release`, so a later install only relinks what changed;
- the library is installed into the venv with
  `uv pip install --no-build-isolation` (`setuptools` and `wheel` are added to
  the venv first if they are missing) and the result is cached as
  `~/cpsa-build/libwarble-<version>-debug.so`;
- the step then imports `mbientlab.warble` from the venv and prints the size of
  `libwarble.so` (about 4.3 MB for the debug build). A failure to download,
  unpack, patch, build, install or import stops the deploy with the reason, so a
  board never goes on stage with the release `warble`.

When the venv already holds that same library (the usual case for a deploy that
did not touch `warble`), the install is skipped and the step costs a few
seconds. The first build takes several minutes and needs `make`, `g++` and
`libbluetooth-dev` on the board (see One-time board setup). `--skip-warble`
skips the whole rebuild for a quick redeploy, at the price of the MetaWear
abort. To force a clean rebuild, remove the cache:

```bash
ssh kria@10.42.0.83 'rm -rf ~/cpsa-build'
```

## Service hardening and boot ordering

For this slice the supervisor unit runs **without** systemd sandboxing
directives. The three cheap ones (`ProtectSystem=full`, `PrivateTmp=yes`,
`ProtectKernelModules=yes`) were reviewed against the code but never exercised
on the KV260, and the demo comes before there is time for a board test: a
sandbox that breaks the DPU, the camera or the IMU would show up only on the
board. They are kept in the unit as a commented block and marked
`enable after a board test`.

To enable them after the demo (or once a board test is possible):

1. Uncomment the three lines at the end of that block in
   `deploy/systemd/cpsa-supervisor.service`.
2. Reinstall the unit (`deploy/deploy.sh`, or `sudo install -m 0644` plus
   `daemon-reload`) and `sudo systemctl restart cpsa-supervisor`.
3. Run a full IMU + camera + DPU session and read
   `journalctl -u cpsa-supervisor -b`. If the IMU reports `imu_not_found`, the
   camera does not open or the DPU fails to start, comment the lines out again
   and restart; nothing else in the deploy depends on them.

Why each one is expected to be safe once tested: `ProtectSystem=full` makes
`/usr`, `/boot` and `/etc` read-only, and the service only writes under
`/home/kria/cpsa` (`runs/logs`, `runs/evidence`); `PrivateTmp=yes` gives the
unit a private `/tmp` and `/var/tmp`, where the supervisor and the bluepy helper
it spawns share one namespace (bluepy talks to the helper over stdin/stdout
pipes), while BlueZ is reached over D-Bus (`/run/dbus`) and VART through `/dev`
and `/dev/shm`; `ProtectKernelModules=yes` blocks module loading only, and the
Bluetooth and DPU modules are loaded at boot and by `bluetooth.service`.

`NoNewPrivileges=yes` is deliberately **not** set: it would discard the file
capabilities on `bluepy-helper`, and the IMU would report `imu_not_found`.
`RestrictAddressFamilies` is not set either: the bluepy helper opens
`AF_BLUETOOTH` raw sockets, which the allow-list would have to include.
`ProtectHome` and a read-only `/home` are ruled out because the service lives in
`/home/kria/cpsa`.

The unit does not use `network-online.target`: it binds `0.0.0.0` and the demo
works with no network link, so waiting for the link would only add delay (up to
the wait-online timeout) when the board boots without the booth laptop. It is
ordered after `cpsa-dpu.service` and `bluetooth.service` and requires the DPU
unit; Bluetooth is wanted, not required, so a missing Bluetooth stack does not
block the web dashboard.

## Offline installation (NFR-07)

The board may have no internet. Prepare cp310 / linux-aarch64 wheels on the PC
and pass their directory to the script:

```bash
# Requirements of the board extras at the locked versions, evaluated for the
# board's Python 3.10 and aarch64 (some markers, such as onnxruntime's two
# versions, depend on the Python version).
uv export --frozen --no-dev --no-hashes --no-emit-project \
  --extra board --extra server --extra vision \
  | uv pip compile - --python-version 3.10 --python-platform aarch64-manylinux_2_28 \
      --no-annotate --no-header -o /tmp/cpsa-board-reqs.txt

# aarch64 cp310 wheels for every dependency that publishes them. Ubuntu 22.04
# has glibc 2.35, so manylinux_2_28 wheels (onnxruntime, opencv) are valid too.
mkdir -p wheels
grep -viE '^(bluepy|blue-st-sdk|metawear|warble)==' /tmp/cpsa-board-reqs.txt \
  > /tmp/cpsa-board-wheels.txt
pip download -r /tmp/cpsa-board-wheels.txt -d wheels \
  --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 \
  --python-version 310 --only-binary=:all:

# bluepy, blue-st-sdk, metawear and warble are source-only. Fetch their sdists
# at the versions pinned in uv.lock; uv builds them on the board, which needs
# the build tools from the Installation section.
grep -iE '^(bluepy|blue-st-sdk|metawear|warble)==' /tmp/cpsa-board-reqs.txt \
  > /tmp/cpsa-board-sources.txt
pip download -r /tmp/cpsa-board-sources.txt -d wheels --no-deps --no-binary=:all:

# uv builds the cpsa project and the source-only board packages without build
# isolation; keep the backends available offline as well.
pip download -d wheels --python-version 310 --only-binary=:all: \
  hatchling editables setuptools wheel

deploy/deploy.sh --model ~/build/vitis/trained/pose_halfbust62.xmodel --wheels wheels
```

The script copies the wheels to `~/cpsa/wheels`, creates the venv if needed
(with `--system-site-packages`, so the system VART bindings stay importable),
installs `hatchling`, `editables`, `setuptools` and `wheel` into it, and runs
`uv sync --frozen --no-build-isolation --offline --find-links ~/cpsa/wheels ...`
on the board. The build backends are preinstalled because uv's build isolation
does not consult `--find-links`. If a package has no aarch64 wheel, build it on
the board or place the built wheel in `wheels/` before deploying. `pip` is the
PC's `python3 -m pip`. When the sync fails, the script stops and puts the
previous venv back, so an offline board keeps the demo it had; without
`--wheels` the sync needs a package index or a warm uv cache.

Offline also covers the debug `warble` (step 4): the sdist in `wheels/` is used
instead of a download, the sha256 from `uv.lock` is still checked, and
`setuptools`/`wheel` come from the same directory. The build itself runs on the
board, so the build tools from One-time board setup must be installed; once the
source tree is cached in `~/cpsa-build`, later deploys need neither the sdist
nor the build tools again.

## Checking and restarting the services

On the board (or through `ssh kria@10.42.0.83`):

```bash
systemctl is-active cpsa-dpu cpsa-supervisor     # expect: active, active
systemctl status cpsa-dpu cpsa-supervisor
curl -fsS http://10.42.0.83:8080/health          # expect: "mode_state": "running"
getcap /home/kria/cpsa/.venv/lib/python3.10/site-packages/bluepy/bluepy-helper

# the two board patches the deploy re-applies
grep -c 'collections.abc' \
  /home/kria/cpsa/.venv/lib/python3.10/site-packages/blue_st_sdk/utils/dict_put_single_element.py
ls -l /home/kria/cpsa/.venv/lib/python3.10/site-packages/mbientlab/warble/libwarble.so
# expect about 4.3 MB for the debug build (the release one is much smaller)

sudo systemctl restart cpsa-supervisor           # restart after a code change
sudo systemctl start cpsa-dpu                    # only if the DPU unit is inactive
```

`/health` answers `200` as soon as the web server is up, even when the medical
mode failed, so read the JSON: `"mode_state": "failed"` means the mode did not
start (the reason is in the journal, and the `medical.state` event carries an
IF-304 code such as `model_missing`, `camera_missing` or `imu_not_found`). `cpsa-supervisor` starts the configured `default_mode` by itself at
boot and after every restart, so no `mode.set` command is needed.

`deploy.sh` already enables both units. `--no-restart` is for installing new
files without interrupting a running demo; the new code is picked up at the
next restart.

## Logs

```bash
journalctl -u cpsa-supervisor -f                 # follow the service
journalctl -u cpsa-supervisor -b --no-pager      # this boot
journalctl -u cpsa-dpu -b --no-pager             # xmutil output
```

Application logs and evidence are written under the board paths in
`config/board.demo.yaml`:

- `/home/kria/cpsa/runs/logs/<YYYY-MM-DD>/system.log` and `events.jsonl`;
- `/home/kria/cpsa/runs/evidence/<YYYY-MM-DD>/<episode>/`.

`journalctl` is still the first place to look: Python startup errors, missing
model files and BLE failures appear there with the rest of the service output.

## Reboot test (AC-10)

```bash
sudo reboot
# Wait for the board to come back (about 40 s), then from the PC:
ssh kria@10.42.0.83 'systemctl is-active cpsa-dpu cpsa-supervisor'
curl -fsS http://10.42.0.83:8080/health
```

Both units must be `active` without a manual start, and the dashboard must
answer on port 8080 with the medical mode (`"mode_state": "running"`; a mode
that is still discovering BlueCoins reports `"starting"` for a few tens of
seconds). `cpsa-dpu` loads `kv260-benchmark-b4096` before the supervisor starts.

## Restart-on-crash test (AC-10)

`kill -9` the supervisor's main process and time the recovery:

```bash
ssh kria@10.42.0.83 '
  pid=$(systemctl show -p MainPID --value cpsa-supervisor)
  start=$(date +%s)
  kill -9 "$pid"
  until curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; do sleep 0.5; done
  echo "recovered after $(( $(date +%s) - start )) s"
  systemctl show -p NRestarts --value cpsa-supervisor
'
```

`RestartSec=3` and `StartLimitIntervalSec=0` in
`deploy/systemd/cpsa-supervisor.service` mean systemd restarts the process
within a few seconds and never disables the unit after repeated crashes. The
loop above measures the web server (AC-10); the medical mode itself is only
`running` again after its BlueCoin discovery, so the dashboard may show
`starting` for a while after the restart.

## Swapping the model

1. Compile the new model for `kv260-benchmark-b4096` and check the DPU
   fingerprint (`0x101000016010407`, AC-04), for example with
   `tools/vitis/inspect_xmodel.py`.
2. Set `vision.model.input_width` / `input_height` in
   `config/board.demo.yaml` to the resolution the model was compiled for.
   The trained model uses 512x384.
3. Deploy and restart:

   ```bash
   deploy/deploy.sh --model ~/build/vitis/trained/pose_halfbust62.xmodel
   # or, if the files are already on the board:
   sudo systemctl restart cpsa-supervisor
   ```

4. Confirm `curl -fsS http://10.42.0.83:8080/health` and the pose output in
   `journalctl -u cpsa-supervisor -f`.

The random-weights bring-up model
(`video-pipeline/build/vitis/compiled/pose_halfbust62.xmodel`) was compiled at
480x640 (width x height). For that model only, temporarily set
`input_width: 640` and `input_height: 480`; restore 512x384 for the trained
model.
