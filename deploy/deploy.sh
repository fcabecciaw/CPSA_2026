#!/usr/bin/env bash
#
# Install the CPSA medical demo on a KV260 and enable it at boot.
#
# The script copies the repository, the built dashboard (web/dist) and the
# compiled pose model to the board, syncs the Python environment with uv
# (a venv that sees the system VART bindings), grants the bluepy helper its
# raw-socket capabilities, installs the two systemd units and verifies the
# deployment, including the mode reported by /health.
#
# Run `deploy/deploy.sh --help` for the options; see deploy/README.md for the
# prerequisites and the board-side tests.

set -euo pipefail

readonly DEFAULT_HOST="10.42.0.83"
readonly DEFAULT_USER="kria"
readonly REMOTE_HOME="/home/kria"
readonly REMOTE_DIR="cpsa"
readonly REMOTE_MODEL="models/pose_halfbust62.xmodel"
readonly SUPERVISOR_PORT="8080"

host="$DEFAULT_HOST"
user="$DEFAULT_USER"
model=""
wheels=""
dry_run=0
restart=1

usage() {
  cat <<'EOF'
Usage: deploy/deploy.sh --model PATH [options]

Install the CPSA medical demo on the KV260 and enable cpsa-supervisor at boot.

Required:
  --model PATH       Compiled pose model (.xmodel) to install as
                     ~/cpsa/models/pose_halfbust62.xmodel on the board.

Options:
  --host HOST        Board host or SSH alias (default: 10.42.0.83).
  --user USER        SSH user on the board (default: kria).
  --wheels DIR       Directory of linux-aarch64 / cp310 wheels prepared on this
                     PC. The board then installs fully offline.
  --no-restart       Do not restart an already-running cpsa-supervisor. The
                     new code takes effect on the next restart.
  --dry-run          Print the planned commands and do not connect to the
                     board or change anything.
  -h, --help         Show this help.

The script expects key-based SSH, password-less sudo on the board, `uv` there,
and a built dashboard in web/dist (run `npm --prefix web run build` first).
It creates the venv with --system-site-packages (for the VART bindings),
rebuilds one that lacks the flag while keeping the old venv as
.venv.bak-<timestamp> until the new one imports xir/vart, grants the bluepy
helper its capabilities, and requires /health to report mode_state=running.
See deploy/README.md.
EOF
}

die() {
  printf 'deploy: error: %s\n' "$*" >&2
  exit 1
}

note() {
  printf 'deploy: %s\n' "$*"
}

warn() {
  printf 'deploy: warning: %s\n' "$*" >&2
}

print_cmd() {
  local arg
  printf '  '
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
  printf '\n'
}

run() {
  if ((dry_run)); then
    print_cmd "$@"
  else
    "$@"
  fi
}

remote() {
  local command="$1"
  if ((dry_run)); then
    print_cmd ssh "${ssh_opts[@]}" "$ssh_target" "$command"
  else
    ssh "${ssh_opts[@]}" "$ssh_target" "$command"
  fi
}

while (($# > 0)); do
  case "$1" in
    --host)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--host needs a value"
      host="$2"
      shift 2
      ;;
    --user)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--user needs a value"
      user="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--model needs a value"
      model="$2"
      shift 2
      ;;
    --wheels)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--wheels needs a value"
      wheels="$2"
      shift 2
      ;;
    --no-restart)
      restart=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (try --help)"
      ;;
  esac
done

[[ -n "$model" ]] || die "--model is required (try --help)"
[[ -n "$host" ]] || die "--host must not be empty"
[[ -n "$user" ]] || die "--user must not be empty"
if [[ -n "$wheels" && ! -d "$wheels" ]]; then
  die "wheels directory not found: $wheels"
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" ||
  die "cannot determine the repository root from ${BASH_SOURCE[0]}"
ssh_target="${user}@${host}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
rsync_opts=(-e "ssh -o BatchMode=yes -o ConnectTimeout=10")
healthy_url="http://${host}:${SUPERVISOR_PORT}/health"

preflight_failed=0
if ! command -v ssh >/dev/null 2>&1; then
  warn "ssh is not installed"
  preflight_failed=1
fi
if ! command -v rsync >/dev/null 2>&1; then
  warn "rsync is not installed"
  preflight_failed=1
fi
if ! command -v curl >/dev/null 2>&1; then
  warn "curl is not installed"
  preflight_failed=1
fi
if ! command -v python3 >/dev/null 2>&1; then
  warn "python3 is not installed (it parses the /health JSON in the post-check)"
  preflight_failed=1
fi
if [[ ! -f "$repo_root/web/dist/index.html" ]]; then
  warn "web/dist/index.html is missing; build the dashboard first:"
  warn "  npm --prefix web run build"
  preflight_failed=1
fi
if [[ ! -f "$model" ]]; then
  warn "model file not found: $model"
  preflight_failed=1
fi
if ((preflight_failed)); then
  if ((dry_run)); then
    note "dry run: showing the plan despite the missing prerequisites above"
  else
    die "preflight failed"
  fi
fi

if ((dry_run)); then
  note "dry run: no connection to ${ssh_target} and no changes"
fi

note "preflight: checking SSH to ${ssh_target}"
if ((dry_run)); then
  remote true
elif ! remote true; then
  die "cannot reach ${ssh_target} over SSH (key-based access is required)"
fi

note "step 1: syncing the repository to ${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}"
rsync_excludes=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude 'ml/'
  --exclude 'build/'
  --exclude 'node_modules/'
  --exclude 'runs/'
  --exclude 'web/dist/'
  --exclude 'wheels/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
)
run rsync -az "${rsync_opts[@]}" "${rsync_excludes[@]}" "$repo_root/" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/"

note "step 2: syncing web/dist and the model"
remote "mkdir -p \"${REMOTE_HOME}/${REMOTE_DIR}/models\" \"${REMOTE_HOME}/${REMOTE_DIR}/runs/logs\" \"${REMOTE_HOME}/${REMOTE_DIR}/runs/evidence\""
run rsync -az "${rsync_opts[@]}" --delete "$repo_root/web/dist/" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/web/dist/"
run rsync -az "${rsync_opts[@]}" "$model" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/${REMOTE_MODEL}"

if [[ -n "$wheels" ]]; then
  note "step 3: syncing offline wheels"
  remote "mkdir -p \"${REMOTE_HOME}/${REMOTE_DIR}/wheels\""
  run rsync -az "${rsync_opts[@]}" "$wheels/" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/wheels/"
fi

note "step 4: syncing the Python environment on the board"
# The VART `xir`/`vart` bindings live in the system dist-packages
# (docs/notes/vart-kv260.md), so the venv must be created with
# --system-site-packages. A venv made without it cannot import them and the DPU
# path fails, so such a venv is replaced here instead of asking for a manual
# step (R-11). The old venv is never deleted: it is moved to .venv.bak-<stamp>
# and put back when the new one cannot be created or synced, so a failed deploy
# cannot leave the board without a working demo (R-21). Step 5 puts it back too
# when the VART bindings still do not import from the new venv.
method="online"
if [[ -n "$wheels" ]]; then
  method="wheels"
fi
uv_sync=$(sed -e "s|@REMOTE_DIR@|${REMOTE_HOME}/${REMOTE_DIR}|g" -e "s|@METHOD@|${method}|g" <<'REMOTE'
set -e
cd "@REMOTE_DIR@" || { echo "deploy: error: cannot change directory to @REMOTE_DIR@" >&2; exit 1; }
PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PATH

backup=""
restore_previous_venv() {
  if [ -n "$backup" ] && [ -d "$backup" ]; then
    if [ -d .venv ]; then
      failed=".venv.failed-$(date +%Y%m%d-%H%M%S)-$$"
      mv .venv "$failed"
      echo "deploy: the venv that failed this deploy is kept at $PWD/$failed" >&2
    fi
    mv "$backup" .venv
    echo "deploy: restored the previous .venv from $PWD/$backup" >&2
  fi
}

fail_venv() {
  restore_previous_venv
  echo "deploy: error: $1" >&2
  if [ -z "$backup" ]; then
    echo "deploy: the incomplete .venv is left in place; fix the cause and re-run the deploy" >&2
  fi
  exit 1
}

# A marker left by an earlier deploy must not restore the wrong venv in step 5.
rm -f .venv.previous

if ! command -v uv >/dev/null 2>&1; then
  if [ -x .venv/bin/cpsa-supervisor ]; then
    echo "deploy: uv is not installed; reusing the existing .venv" >&2
    if [ -f .venv/pyvenv.cfg ] && ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg; then
      echo "deploy: warning: the existing .venv was created without --system-site-packages and uv is not installed to replace it; xir/vart will probably not import (deploy/README.md)" >&2
    fi
    exit 0
  fi
  echo "deploy: error: uv is not installed and .venv/bin/cpsa-supervisor is missing" >&2
  exit 1
fi

if [ -f .venv/pyvenv.cfg ] && ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg; then
  backup=".venv.bak-$(date +%Y%m%d-%H%M%S)-$$"
  echo "deploy: the existing .venv was created without --system-site-packages (xir/vart are system packages); moving it to $backup and building a new one" >&2
  mv .venv "$backup"
fi

if [ ! -x .venv/bin/python ]; then
  if ! uv venv --system-site-packages --python 3.10 .venv; then
    fail_venv "uv venv --system-site-packages failed"
  fi
fi

if [ "@METHOD@" = wheels ]; then
  # uv's build isolation does not read --find-links, so install the build
  # backend into the venv first and build the project without isolation.
  if ! uv pip install --offline --python .venv/bin/python --find-links "@REMOTE_DIR@/wheels" hatchling editables setuptools wheel; then
    fail_venv "installing the build backends from @REMOTE_DIR@/wheels failed"
  fi
  if ! uv sync --frozen --no-build-isolation --python 3.10 --offline --find-links "@REMOTE_DIR@/wheels" --extra board --extra server --extra vision; then
    fail_venv "uv sync --offline from @REMOTE_DIR@/wheels failed"
  fi
else
  if ! uv sync --frozen --python 3.10 --extra board --extra server --extra vision; then
    fail_venv "uv sync failed (without --wheels the board needs a package index or a warm uv cache; pass --wheels DIR for an offline install)"
  fi
fi

if [ -n "$backup" ]; then
  printf '%s\n' "$backup" > .venv.previous
  echo "deploy: the previous .venv is kept at $PWD/$backup; step 5 puts it back if xir/vart do not import" >&2
fi
echo "deploy: .venv is ready"
REMOTE
)
remote "$uv_sync"

note "step 5: checking that the VART bindings import from the venv"
# Without xir/vart the DPU pose estimator cannot start; report it here instead of
# letting the demo fail later with an unrelated error (R-11). When step 4 rebuilt
# the venv and the bindings still do not import, the previous venv goes back so
# the board keeps a working demo (R-21).
check_env=$(sed -e "s|@REMOTE_DIR@|${REMOTE_HOME}/${REMOTE_DIR}|g" <<'REMOTE'
set -e
cd "@REMOTE_DIR@" || { echo "deploy: error: cannot change directory to @REMOTE_DIR@" >&2; exit 1; }

if .venv/bin/python -c 'import xir, vart' 2>/dev/null; then
  echo "deploy: xir and vart import from .venv"
  if [ -f .venv.previous ] && [ -d "$(cat .venv.previous)" ]; then
    echo "deploy: the venv this deploy replaced is kept at $PWD/$(cat .venv.previous); remove it by hand once the demo runs (deploy/README.md)"
  fi
  exit 0
fi

previous=""
if [ -f .venv.previous ]; then
  previous=$(cat .venv.previous)
fi
if [ -n "$previous" ] && [ -d "$previous" ]; then
  failed=".venv.failed-$(date +%Y%m%d-%H%M%S)-$$"
  mv .venv "$failed"
  mv "$previous" .venv
  echo "deploy: error: xir and vart do not import from the new .venv, so the previous one ($PWD/$previous) was put back" >&2
  echo "deploy: the board keeps the demo it had; the failed venv is at $PWD/$failed. Check docs/notes/vart-kv260.md before the next deploy." >&2
  exit 1
fi

echo "deploy: error: xir and vart are not importable from .venv (the VART bindings come from the system dist-packages; see docs/notes/vart-kv260.md and deploy/README.md)" >&2
exit 1
REMOTE
)
remote "$check_env"

note "step 6: granting raw-socket capabilities to the bluepy helper"
# Idempotent: the capability is granted only when it is missing. The unit runs as
# kria, so the capability sits on the helper, not on the service. `sudo -n` keeps
# the remote shell from hanging on a password prompt (BatchMode SSH has no tty).
bluepy_caps="set -e; cd \"${REMOTE_HOME}/${REMOTE_DIR}\" || { echo 'deploy: error: cannot change directory to ${REMOTE_HOME}/${REMOTE_DIR}' >&2; exit 1; }"
bluepy_caps+='; if ! command -v setcap >/dev/null 2>&1 || ! command -v getcap >/dev/null 2>&1; then'
bluepy_caps+=' echo "deploy: warning: setcap/getcap are missing (sudo apt install libcap2-bin); the IMU will report imu_not_found" >&2; exit 0; fi'
bluepy_caps+='; helper=$(ls .venv/lib/python*/site-packages/bluepy/bluepy-helper 2>/dev/null | head -n 1 || true)'
bluepy_caps+='; if [ -z "$helper" ]; then echo "deploy: warning: bluepy-helper is not in .venv; the IMU will report imu_not_found" >&2; exit 0; fi'
bluepy_caps+='; if getcap "$helper" 2>/dev/null | grep -q cap_net_raw; then echo "deploy: capabilities already set: $(getcap "$helper")"; exit 0; fi'
bluepy_caps+='; if sudo -n setcap "cap_net_raw+eip cap_net_admin+eip" "$helper"; then'
bluepy_caps+=' echo "deploy: capabilities set: $(getcap "$helper")";'
bluepy_caps+=' else echo "deploy: warning: could not run \"sudo -n setcap\" (password-less sudo is required over BatchMode SSH); run it by hand: sudo setcap \"cap_net_raw+eip cap_net_admin+eip\" $helper (deploy/README.md); the IMU will report imu_not_found" >&2; fi'
remote "$bluepy_caps"

note "step 7: installing and enabling the systemd units"
units='deploy/systemd/cpsa-dpu.service deploy/systemd/cpsa-supervisor.service'
remote "set -e; cd \"${REMOTE_HOME}/${REMOTE_DIR}\" || { echo 'deploy: error: cannot change directory to ${REMOTE_HOME}/${REMOTE_DIR}' >&2; exit 1; }; sudo install -m 0644 ${units} /etc/systemd/system/; sudo systemctl daemon-reload; sudo systemctl enable cpsa-dpu.service cpsa-supervisor.service"
# Enable only, then start each unit once: `enable --now` followed by `restart`
# starts the supervisor twice and interrupts the BLE discovery of the first
# start (R-12). A `start` of an already active unit is a no-op, so --no-restart
# still leaves a running demo alone.
remote "sudo systemctl start cpsa-dpu.service"
if ((restart)); then
  remote "sudo systemctl restart cpsa-supervisor.service"
else
  remote "sudo systemctl start cpsa-supervisor.service"
fi

note "post-check: unit state and health endpoint"
if ! remote "systemctl is-active cpsa-dpu.service cpsa-supervisor.service"; then
  die "one or more units are not active; inspect: journalctl -u cpsa-supervisor -e"
fi

if ((dry_run)); then
  print_cmd curl -fsS --max-time 5 "$healthy_url"
  note "dry run: the post-check would require mode_state=running in the /health JSON"
else
  # A 200 from /health only proves that the HTTP server answers: it is served
  # with mode_state "failed" too. Require mode_state=running, so a demo that did
  # not start (no model, no BlueCoin, no camera) is not reported as a success
  # (R-12).
  note "waiting for ${healthy_url} to report mode_state=running (up to 120 s)"
  body=""
  state=""
  mode=""
  mode_state=""
  for ((attempt = 1; attempt <= 60; attempt++)); do
    if body=$(curl -fsS --max-time 5 "$healthy_url" 2>/dev/null); then
      state=$(printf '%s' "$body" | python3 -c 'import json, sys; health = json.load(sys.stdin); print(health.get("mode", "?"), health.get("mode_state", "?"))' 2>/dev/null) || state=""
      mode="${state%% *}"
      mode_state="${state#* }"
    fi
    if [[ "$mode_state" == "running" || "$mode_state" == "failed" ]]; then
      break
    fi
    sleep 2
  done
  if [[ "$mode_state" != "running" ]]; then
    warn "health reports mode=${mode:-unknown} mode_state=${mode_state:-unknown}, not mode_state=running"
    warn "the files, the units and the venv are installed, but the demo is not running."
    warn "common causes: the DPU app is not loaded, models/pose_halfbust62.xmodel or the"
    warn "camera is missing, bluepy-helper has no capabilities (see step 6) or no"
    warn "BlueCoin is powered on."
    warn "inspect: ssh ${ssh_target} 'journalctl -u cpsa-supervisor -e'"
    exit 1
  fi
  note "health: mode=${mode} mode_state=${mode_state}"
fi

if ((dry_run)); then
  note "dry run complete: nothing was changed"
else
  note "deployment complete: ${healthy_url}"
fi
