#!/usr/bin/env bash
#
# Install the CPSA showcase on a KV260 and enable it at boot.
#
# The script copies the repository, the built dashboard (web/dist) and the
# compiled YOLOX-nano model to the board, syncs the Python environment with uv
# (a venv that sees the system VART bindings), installs the two systemd units
# (the DPU application loader and the showcase itself) and verifies the
# deployment through /health.
#
# Run `deploy/deploy.sh --help` for the options; see deploy/README.md for the
# prerequisites and the board-side checks.

set -euo pipefail

readonly DEFAULT_HOST="10.42.0.83"
readonly DEFAULT_USER="kria"
readonly REMOTE_HOME="/home/kria"
readonly REMOTE_DIR="cpsa"
readonly MODELS_DIR="models"
readonly SHOWCASE_PORT="8080"
readonly UNITS="cpsa-dpu.service cpsa-showcase.service"
readonly OBSOLETE_UNITS="cpsa-supervisor.service"

host="$DEFAULT_HOST"
user="$DEFAULT_USER"
model=""
wheels=""
dry_run=0
restart=1

usage() {
  cat <<'EOF'
Usage: deploy/deploy.sh --model PATH [options]

Install the showcase (YOLOX-nano, "Cosa vede l'IA") on the KV260 and enable
cpsa-showcase at boot.

Required:
  --model PATH       Compiled YOLOX-nano model (.xmodel). It is installed as
                     ~/cpsa/models/<basename>, so the basename must be the one
                     config/showcase.yaml names (models/yolox_nano.xmodel).

Options:
  --host HOST        Board host or SSH alias (default: 10.42.0.83).
  --user USER        SSH user on the board (default: kria).
  --wheels DIR       Directory of linux-aarch64 / cp310 wheels prepared on this
                     PC. The board then builds its venv fully offline.
  --no-restart       Do not restart an already-running cpsa-showcase. The new
                     code takes effect on the next restart.
  --dry-run          Print the planned commands and do not connect to the
                     board or change anything.
  -h, --help         Show this help.

The script expects key-based SSH, password-less sudo on the board, `uv` there,
and a built dashboard in web/dist (run `npm --prefix web run build` first).
It creates the venv with --system-site-packages (the VART xir/vart bindings are
system packages), removes the superseded cpsa-supervisor unit, and requires
/health to report state=running with a positive fps.
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

[[ -n "$host" ]] || die "--host must not be empty"
[[ -n "$user" ]] || die "--user must not be empty"
[[ -n "$model" ]] || die "--model is required (try --help)"
if [[ -n "$wheels" && ! -d "$wheels" ]]; then
  die "wheels directory not found: $wheels"
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" ||
  die "cannot determine the repository root from ${BASH_SOURCE[0]}"
ssh_target="${user}@${host}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
rsync_opts=(-e "ssh -o BatchMode=yes -o ConnectTimeout=10")
healthy_url="http://${host}:${SHOWCASE_PORT}/health"
model_name="$(basename -- "$model")"

preflight_failed=0
for tool in ssh rsync curl python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    warn "$tool is not installed"
    preflight_failed=1
  fi
done
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
# `--delete` keeps the board free of the files this hard break removed, while
# the excludes protect what the board owns: its venv, its runs, its wheels, its
# models and the dashboard bundle copied in step 2.
rsync_excludes=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude '.venv.*'
  --exclude 'build/'
  --exclude 'node_modules/'
  --exclude 'runs/'
  --exclude 'wheels/'
  --exclude 'models/'
  --exclude 'web/dist/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
)
run rsync -az --delete "${rsync_opts[@]}" "${rsync_excludes[@]}" \
  "$repo_root/" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/"

note "step 2: syncing web/dist and the model"
remote "mkdir -p \"${REMOTE_HOME}/${REMOTE_DIR}/${MODELS_DIR}\" \"${REMOTE_HOME}/${REMOTE_DIR}/runs/logs\""
run rsync -az --delete "${rsync_opts[@]}" \
  "$repo_root/web/dist/" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/web/dist/"
run rsync -az "${rsync_opts[@]}" \
  "$model" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/${MODELS_DIR}/${model_name}"

if [[ -n "$wheels" ]]; then
  note "step 3: syncing offline wheels"
  remote "mkdir -p \"${REMOTE_HOME}/${REMOTE_DIR}/wheels\""
  run rsync -az "${rsync_opts[@]}" \
    "$wheels/" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/wheels/"
fi

note "step 4: syncing the Python environment on the board"
# The VART bindings live in the system dist-packages, so the venv must be
# created with --system-site-packages: without them the DPU cannot start. A venv
# made without the flag is moved aside instead of being silently reused.
method="online"
if [[ -n "$wheels" ]]; then
  method="wheels"
fi
uv_sync=$(sed -e "s|@REMOTE_DIR@|${REMOTE_HOME}/${REMOTE_DIR}|g" -e "s|@METHOD@|${method}|g" <<'REMOTE'
set -e
cd "@REMOTE_DIR@" || { echo "deploy: error: cannot change directory to @REMOTE_DIR@" >&2; exit 1; }
PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PATH

if ! command -v uv >/dev/null 2>&1; then
  if [ -x .venv/bin/cpsa-showcase ]; then
    echo "deploy: uv is not installed; reusing the existing .venv" >&2
    exit 0
  fi
  echo "deploy: error: uv is not installed and .venv/bin/cpsa-showcase is missing" >&2
  exit 1
fi

if [ -f .venv/pyvenv.cfg ] && ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg; then
  stale=".venv.bak-$(date +%Y%m%d-%H%M%S)-$$"
  echo "deploy: the existing .venv lacks --system-site-packages (xir/vart are system packages); it is kept at $stale and rebuilt" >&2
  mv .venv "$stale"
fi

if [ ! -x .venv/bin/python ]; then
  uv venv --system-site-packages --python 3.10 .venv
fi

if [ "@METHOD@" = wheels ]; then
  # uv's build isolation does not read --find-links, so install the build
  # backend into the venv first and build the project without isolation.
  uv pip install --offline --python .venv/bin/python \
    --find-links "@REMOTE_DIR@/wheels" hatchling editables setuptools wheel
  uv sync --frozen --no-build-isolation --python 3.10 --offline \
    --find-links "@REMOTE_DIR@/wheels"
else
  uv sync --frozen --python 3.10
fi

if ! .venv/bin/python -c 'import xir, vart' 2>/dev/null; then
  echo "deploy: error: xir and vart do not import from .venv; the DPU cannot start (deploy/README.md)" >&2
  exit 1
fi
echo "deploy: .venv is ready and sees the VART bindings"
REMOTE
)
remote "$uv_sync"

note "step 5: installing the systemd units"
# The supervisor that used to boot the demo is gone with the modes it managed:
# stop and disable it before the new unit takes over the same port.
remote "set -e; \
if systemctl list-unit-files ${OBSOLETE_UNITS} >/dev/null 2>&1; then \
  sudo systemctl stop ${OBSOLETE_UNITS} 2>/dev/null || true; \
  sudo systemctl disable ${OBSOLETE_UNITS} 2>/dev/null || true; \
  sudo rm -f /etc/systemd/system/${OBSOLETE_UNITS}; \
fi; \
cd \"${REMOTE_HOME}/${REMOTE_DIR}\" || { echo 'deploy: error: cannot change directory to ${REMOTE_HOME}/${REMOTE_DIR}' >&2; exit 1; }; \
sudo install -m 0644 deploy/systemd/cpsa-dpu.service deploy/systemd/cpsa-showcase.service /etc/systemd/system/; \
sudo systemctl daemon-reload; \
sudo systemctl enable ${UNITS}"
remote "sudo systemctl start cpsa-dpu.service"
if ((restart)); then
  remote "sudo systemctl restart cpsa-showcase.service"
else
  remote "sudo systemctl start cpsa-showcase.service"
fi

note "post-check: unit state and health endpoint"
if ! remote "systemctl is-active ${UNITS}"; then
  die "one or more units are not active; inspect: journalctl -u cpsa-showcase -e"
fi

if ((dry_run)); then
  print_cmd curl -fsS --max-time 5 "$healthy_url"
  note "dry run: the post-check would require state=running and fps > 0 in /health"
  note "dry run complete: nothing was changed"
  exit 0
fi

# A 200 from /health only proves that the HTTP server answers: it does so while
# the pipeline is still starting or has failed. Require state=running and a
# positive fps, so a showcase without a camera, a model or a loaded DPU app is
# never reported as deployed.
note "waiting for ${healthy_url} to report state=running (up to 120 s)"
body=""
health=""
for ((attempt = 1; attempt <= 60; attempt++)); do
  if body=$(curl -fsS --max-time 5 "$healthy_url" 2>/dev/null); then
    health=$(printf '%s' "$body" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
fps = payload.get("fps")
print(payload.get("state", "?"), "ok" if isinstance(fps, (int, float)) and fps > 0 else "no-fps")
' 2>/dev/null) || health=""
  fi
  if [[ "$health" == "running ok" || "$health" == "error "* ]]; then
    break
  fi
  sleep 2
done
if [[ "$health" != "running ok" ]]; then
  warn "health reports ${health:-no answer}, not state=running with fps > 0"
  warn "the files, the units and the venv are installed, but the showcase is not running."
  warn "common causes: the DPU app is not loaded (systemctl status cpsa-dpu),"
  warn "models/${model_name} is missing, the camera is not connected or VART"
  warn "cannot be reached."
  warn "inspect: ssh ${ssh_target} 'journalctl -u cpsa-showcase -e'"
  exit 1
fi
note "health: state=running, fps > 0"
note "deployment complete: http://${host}:${SHOWCASE_PORT}/"
