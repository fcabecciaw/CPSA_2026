#!/usr/bin/env bash
#
# Install the CPSA medical demo on a KV260 and enable it at boot.
#
# The script copies the repository, the built dashboard (web/dist) and the
# compiled pose model to the board, syncs the Python environment with uv
# (a venv that sees the system VART bindings), re-applies the two board patches
# that the sync reverts (the blue_st_sdk Python 3.10 fix and the debug build of
# warble), grants the bluepy helper its raw-socket capabilities, installs the
# two systemd units and verifies the deployment, including the mode reported by
# /health.
#
# Run `deploy/deploy.sh --help` for the options; see deploy/README.md for the
# prerequisites and the board-side tests.

set -euo pipefail

readonly DEFAULT_HOST="10.42.0.83"
readonly DEFAULT_USER="kria"
readonly REMOTE_HOME="/home/kria"
readonly REMOTE_DIR="cpsa"
readonly REMOTE_MODEL="models/pose_halfbust62.xmodel"
readonly REMOTE_HAND_MODEL="models/hand21_topdown.xmodel"
readonly REMOTE_RPS_MODEL="models/rps_mlp.npz"
readonly REMOTE_MODELS_DIR="models"
readonly SUPERVISOR_PORT="8080"

host="$DEFAULT_HOST"
user="$DEFAULT_USER"
model=""
hand_model=""
showcase_models=()
wheels=""
dry_run=0
restart=1
skip_warble=0

usage() {
  cat <<'EOF'
Usage: deploy/deploy.sh --model PATH [options]

Install the CPSA medical demo on the KV260 and enable cpsa-supervisor at boot.

Required:
  --model PATH       Compiled pose model (.xmodel) to install as
                     ~/cpsa/models/pose_halfbust62.xmodel on the board.

Options:
  --hand-model PATH  Compiled HAND21 hand model (.xmodel) to install as
                     ~/cpsa/models/hand21_topdown.xmodel, for the `hologram`
                     and `rps` modes. Without it those modes report
                     model_missing; the medical demo is unaffected.
                     models/rps_mlp.npz is copied when it exists on this PC.
  --showcase-model PATH
                     Compiled detector (.xmodel) for the `showcase` mode
                     ("Cosa vede l'IA"), installed as ~/cpsa/models/<basename>.
                     Repeatable, one flag per model; the name must match a
                     `showcase.models[].path` of config/board.demo.yaml.
                     Without it the mode reports model_missing.
  --host HOST        Board host or SSH alias (default: 10.42.0.83).
  --user USER        SSH user on the board (default: kria).
  --wheels DIR       Directory of linux-aarch64 / cp310 wheels prepared on this
                     PC. The board then installs fully offline.
  --no-restart       Do not restart an already-running cpsa-supervisor. The
                     new code takes effect on the next restart.
  --skip-warble      Do not rebuild and reinstall the debug build of warble on
                     the board. Faster, but the MetaWear actuator then aborts
                     with "*** buffer overflow detected ***" (deploy/README.md).
  --dry-run          Print the planned commands and do not connect to the
                     board or change anything.
  -h, --help         Show this help.

The script expects key-based SSH, password-less sudo on the board, `uv` there,
and a built dashboard in web/dist (run `npm --prefix web run build` first).
It creates the venv with --system-site-packages (for the VART bindings),
rebuilds one that lacks the flag while keeping the old venv as
.venv.bak-<timestamp> until the new one imports xir/vart, grants the bluepy
helper its capabilities, restores the two board patches that `uv sync` reverts
(the blue_st_sdk Python 3.10 fix and the debug build of warble, built once into
~/cpsa-build), and requires /health to report mode_state=running.
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
    --hand-model)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--hand-model needs a value"
      hand_model="$2"
      shift 2
      ;;
    --showcase-model)
      [[ $# -ge 2 && -n "${2:-}" ]] || die "--showcase-model needs a value"
      showcase_models+=("$2")
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
    --skip-warble)
      skip_warble=1
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
if [[ -n "$hand_model" && ! -f "$hand_model" ]]; then
  warn "hand model file not found: $hand_model"
  preflight_failed=1
fi
for showcase_model in "${showcase_models[@]}"; do
  if [[ ! -f "$showcase_model" ]]; then
    warn "showcase model file not found: $showcase_model"
    preflight_failed=1
  fi
done
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
  # A host-built Kalman kernel must never reach the board; step 3 builds the
  # aarch64 one there, and the skip rule below would otherwise trust it.
  --exclude 'src/cpsa/adapters/native/lib/libcpsa_kalman2d.so'
)
run rsync -az "${rsync_opts[@]}" "${rsync_excludes[@]}" "$repo_root/" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/"

note "step 2: syncing web/dist and the model"
remote "mkdir -p \"${REMOTE_HOME}/${REMOTE_DIR}/models\" \"${REMOTE_HOME}/${REMOTE_DIR}/runs/logs\" \"${REMOTE_HOME}/${REMOTE_DIR}/runs/evidence\""
run rsync -az "${rsync_opts[@]}" --delete "$repo_root/web/dist/" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/web/dist/"
run rsync -az "${rsync_opts[@]}" "$model" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/${REMOTE_MODEL}"
# The hands demos: the classifier is small and committed, the hand xmodel is
# only installed when --hand-model is given (the medical demo needs neither).
rps_model="$repo_root/models/rps_mlp.npz"
if [[ -f "$rps_model" ]]; then
  run rsync -az "${rsync_opts[@]}" "$rps_model" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/${REMOTE_RPS_MODEL}"
else
  warn "models/rps_mlp.npz is missing; the rps mode will report model_missing"
fi
if [[ -n "$hand_model" ]]; then
  run rsync -az "${rsync_opts[@]}" "$hand_model" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/${REMOTE_HAND_MODEL}"
fi
# The showcase detectors keep their file name: the mode's config names them by
# path, and each --showcase-model lands in ~/cpsa/models/<basename>.
for showcase_model in "${showcase_models[@]}"; do
  run rsync -az "${rsync_opts[@]}" "$showcase_model" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/${REMOTE_MODELS_DIR}/$(basename -- "$showcase_model")"
done

note "step 3: building the native Kalman kernel on the board"
# The tracker's Kalman kernel is a C artifact, not a Python package, so `uv sync`
# does not build it. The build is skipped when the board's own library is newer
# than its sources; a missing compiler or a failed build only warns, because the
# tracker then runs the NumPy kernel and the demo is unaffected (NFR-001).
kalman_build=$(sed -e "s|@REMOTE_DIR@|${REMOTE_HOME}/${REMOTE_DIR}|g" <<'REMOTE'
set -e
cd "@REMOTE_DIR@" || { echo "deploy: error: cannot change directory to @REMOTE_DIR@" >&2; exit 1; }
source_file="native/kalman2d/kalman2d.c"
header_file="native/kalman2d/kalman2d.h"
library="src/cpsa/adapters/native/lib/libcpsa_kalman2d.so"

if [ ! -f "$source_file" ]; then
  echo "deploy: warning: $source_file is not in the synced tree, so the native Kalman kernel was not built; the tracker uses its NumPy kernel" >&2
  exit 0
fi
if [ -f "$library" ] && [ "$library" -nt "$source_file" ] &&
  { [ ! -f "$header_file" ] || [ "$library" -nt "$header_file" ]; }; then
  echo "deploy: the native Kalman kernel is up to date: $PWD/$library ($(wc -c < "$library") bytes)"
  exit 0
fi
if ! command -v cc >/dev/null 2>&1; then
  echo "deploy: warning: cc is not installed, so $library was not built; the tracker uses its NumPy Kalman kernel (deploy/README.md)" >&2
  exit 0
fi
mkdir -p "$(dirname "$library")"
if ! cc -O2 -std=c99 -fPIC -shared -o "$library" "$source_file" -lm; then
  rm -f "$library"
  echo "deploy: warning: building $library failed, so the tracker uses its NumPy Kalman kernel (deploy/README.md)" >&2
  exit 0
fi
echo "deploy: built the native Kalman kernel: $PWD/$library ($(wc -c < "$library") bytes)"
REMOTE
)
remote "$kalman_build"

if [[ -n "$wheels" ]]; then
  note "step 4: syncing offline wheels"
  remote "mkdir -p \"${REMOTE_HOME}/${REMOTE_DIR}/wheels\""
  run rsync -az "${rsync_opts[@]}" "$wheels/" "${ssh_target}:${REMOTE_HOME}/${REMOTE_DIR}/wheels/"
fi

note "step 5: syncing the Python environment on the board"
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

# A marker left by an earlier deploy must not restore the wrong venv in step 6.
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
  echo "deploy: the previous .venv is kept at $PWD/$backup; step 6 puts it back if xir/vart do not import" >&2
fi
echo "deploy: .venv is ready"
REMOTE
)
remote "$uv_sync"

note "step 6: checking that the VART bindings import from the venv"
# Without xir/vart the DPU pose estimator cannot start; report it here instead of
# letting the demo fail later with an unrelated error (R-11). When step 5 rebuilt
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

note "step 7: restoring the board patches that uv sync reverts"
# `uv sync --frozen` puts the released packages back, and two of them do not
# work on this board (bring-up 2026-11-11):
#   - blue_st_sdk's utils/dict_put_single_element.py uses
#     collections.MutableMapping, which Python 3.10 moved to collections.abc
#     and the sdist was never fixed for; without the patch the medical mode
#     cannot start;
#   - the release build of warble aborts the supervisor with
#     "*** buffer overflow detected ***" as soon as the MetaWear connects, so
#     the debug build is built once from the locked sdist, kept in
#     ~/cpsa-build and installed again on every deploy.
# Both steps are idempotent and run after step 6, so the venv they patch is the
# one the deploy keeps (step 6 moves a venv without xir/vart back out).
warble_version=""
warble_sdist_url=""
warble_sdist_sha=""
if [[ -f "$repo_root/uv.lock" ]] && command -v python3 >/dev/null 2>&1; then
  # PyWarble's GitHub repository is 404 and its sdist carries the libblepp
  # sources, so the board builds from the sdist the lock file pins. Take the
  # version, URL and hash from uv.lock instead of hard-coding them.
  warble_lock_info=$(python3 - "$repo_root/uv.lock" <<'PY' || true
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(
    r'\[\[package\]\]\nname = "warble"\nversion = "([^"]+)"\n'
    r'source = \{[^\n]*\}\n'
    r'sdist = \{ url = "([^"]+)"[^}]*sha256:([0-9a-f]{64})',
    text,
)
print(" ".join(match.groups()) if match else "")
PY
)
  read -r warble_version warble_sdist_url warble_sdist_sha <<<"$warble_lock_info" || true
fi
if [[ -z "$warble_version" ]]; then
  warn "uv.lock does not pin warble or python3 could not read it; the board then needs"
  warn "warble-<version>.tar.gz in ~/cpsa-build or in the --wheels directory"
fi

post_sync=$(sed \
  -e "s|@REMOTE_DIR@|${REMOTE_HOME}/${REMOTE_DIR}|g" \
  -e "s|@METHOD@|${method}|g" \
  -e "s|@SKIP_WARBLE@|${skip_warble}|g" \
  -e "s|@WARBLE_VERSION@|${warble_version}|g" \
  -e "s|@WARBLE_SDIST_URL@|${warble_sdist_url}|g" \
  -e "s|@WARBLE_SDIST_SHA@|${warble_sdist_sha}|g" <<'REMOTE'
set -e
cd "@REMOTE_DIR@" || { echo "deploy: error: cannot change directory to @REMOTE_DIR@" >&2; exit 1; }
PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PATH

build_root="$HOME/cpsa-build"

# (a) blue_st_sdk: collections.MutableMapping -> collections.abc.MutableMapping.
# Guarded sed: an already patched file is left untouched, and `import
# collections` is widened to `import collections.abc` (which is what makes the
# attribute exist at all in Python 3.10). The original is kept once, next to it.
blue_st_seen=0
for py_dir in .venv/lib/python*; do
  target="$py_dir/site-packages/blue_st_sdk/utils/dict_put_single_element.py"
  [ -f "$target" ] || continue
  blue_st_seen=1
  if ! grep -q 'collections\.MutableMapping' "$target" &&
    grep -qE '(import collections\.abc|from collections\.abc import)' "$target"; then
    echo "deploy: blue_st_sdk is already patched for Python 3.10: $target"
    continue
  fi
  if [ ! -e "$target.cpsa-orig" ]; then
    cp -p "$target" "$target.cpsa-orig"
  fi
  sed -i \
    -e 's/collections\.MutableMapping/collections.abc.MutableMapping/g' \
    -e 's/^import collections[[:space:]]*$/import collections.abc/' \
    "$target"
  if grep -q 'collections\.MutableMapping' "$target" ||
    ! grep -qE '(import collections\.abc|from collections\.abc import)' "$target"; then
    echo "deploy: error: could not apply the Python 3.10 fix to $target (the original is at $target.cpsa-orig; deploy/README.md)" >&2
    exit 1
  fi
  echo "deploy: patched blue_st_sdk for Python 3.10: $target"
done
if [ "$blue_st_seen" = 0 ]; then
  echo "deploy: warning: blue_st_sdk is not installed in .venv, so the Python 3.10 fix was not applied (deploy/README.md)" >&2
fi

if [ "@SKIP_WARBLE@" = 1 ]; then
  echo "deploy: --skip-warble: the venv's warble was left as it is; unless the debug build is already installed, the MetaWear aborts with '*** buffer overflow detected ***' (deploy/README.md)" >&2
else
  # (b) debug warble. Built once into $build_root (the source tree, its build
  # objects and the tarball stay there) and installed into the venv again only
  # when the venv does not already carry that same library, which is what
  # `uv sync` undoes. `make` is incremental, so a cached tree costs seconds.
  warble_version=$(.venv/bin/python -c 'import importlib.metadata as md; print(md.version("warble"))' 2>/dev/null || true)
  if [ -z "$warble_version" ]; then
    echo "deploy: error: warble is not installed in .venv; pass --skip-warble or sync the board extra first (deploy/README.md)" >&2
    exit 1
  fi
  if [ -n "@WARBLE_VERSION@" ] && [ "$warble_version" != "@WARBLE_VERSION@" ]; then
    echo "deploy: warning: .venv has warble $warble_version but uv.lock pins @WARBLE_VERSION@; building the installed version (deploy/README.md)" >&2
  fi
  src="$build_root/warble-$warble_version"
  tarball="$build_root/warble-$warble_version.tar.gz"
  cached_lib="$build_root/libwarble-$warble_version-debug.so"
  mkdir -p "$build_root"

  if [ ! -f "$src/setup.py" ]; then
    if [ ! -f "$tarball" ]; then
      sdist_name="warble-$warble_version.tar.gz"
      if [ -f "wheels/$sdist_name" ]; then
        cp "wheels/$sdist_name" "$tarball"
        echo "deploy: using the warble source from wheels/$sdist_name"
      elif [ -n "@WARBLE_SDIST_URL@" ]; then
        echo "deploy: downloading the warble source: @WARBLE_SDIST_URL@"
        if ! curl -fsSL --max-time 180 -o "$tarball.part" "@WARBLE_SDIST_URL@"; then
          rm -f "$tarball.part"
          echo "deploy: error: could not download the warble source; copy $sdist_name to $build_root on the board or pass --wheels (deploy/README.md)" >&2
          exit 1
        fi
        mv "$tarball.part" "$tarball"
      else
        echo "deploy: error: no $sdist_name in wheels/ or $build_root and uv.lock gave no URL; copy the sdist to $build_root (deploy/README.md)" >&2
        exit 1
      fi
    fi
    if [ -n "@WARBLE_SDIST_SHA@" ]; then
      actual_sha=$(sha256sum "$tarball" | cut -d ' ' -f 1)
      if [ "$actual_sha" != "@WARBLE_SDIST_SHA@" ]; then
        echo "deploy: error: $tarball has sha256 $actual_sha, expected @WARBLE_SDIST_SHA@; remove it and re-run (deploy/README.md)" >&2
        exit 1
      fi
    fi
    tar -xzf "$tarball" -C "$build_root"
    if [ ! -f "$src/setup.py" ]; then
      echo "deploy: error: $tarball did not extract $src/setup.py; the version in uv.lock and the sdist name disagree (deploy/README.md)" >&2
      exit 1
    fi
    echo "deploy: extracted the warble source into $src"
  fi

  # Guarded patch: CONFIG=debug for the C++ build, and the library taken from
  # dist/debug instead of dist/release (both are what setup.py hard-codes).
  if grep -q '"CONFIG=debug"' "$src/setup.py" && grep -q "'dist', 'debug', 'lib'" "$src/setup.py"; then
    echo "deploy: warble setup.py already builds CONFIG=debug"
  else
    [ -e "$src/setup.py.orig" ] || cp -p "$src/setup.py" "$src/setup.py.orig"
    sed -i \
      -e 's/"-j%d" % (cpu_count())\]/"-j%d" % (cpu_count()), "CONFIG=debug"]/' \
      -e "s/'dist', 'release', 'lib'/'dist', 'debug', 'lib'/" \
      "$src/setup.py"
    if ! grep -q '"CONFIG=debug"' "$src/setup.py" || ! grep -q "'dist', 'debug', 'lib'" "$src/setup.py"; then
      cp -p "$src/setup.py.orig" "$src/setup.py"
      echo "deploy: error: $src/setup.py is not the warble 1.2.9 one (the CONFIG=debug patch did not apply); it was restored from setup.py.orig (deploy/README.md)" >&2
      exit 1
    fi
    echo "deploy: patched $src/setup.py for a debug build"
  fi

  # The patch rewrites Python source; a quoting mistake only surfaces later as a
  # `uv pip install` failure with the whole deploy rolled back, so compile the
  # file here. A tree patched by an older deploy.sh (broken quote) also lands
  # here: the greps above accept it, this does not.
  if command -v python3 >/dev/null 2>&1; then
    if ! python3 -m py_compile "$src/setup.py"; then
      if [ -e "$src/setup.py.orig" ]; then
        cp -p "$src/setup.py.orig" "$src/setup.py"
        echo "deploy: error: the patched $src/setup.py does not compile (python3 -m py_compile failed); it was restored from setup.py.orig, re-run to re-patch it (deploy/README.md)" >&2
      else
        echo "deploy: error: the patched $src/setup.py does not compile (python3 -m py_compile failed) and there is no setup.py.orig to restore (deploy/README.md)" >&2
      fi
      exit 1
    fi
  else
    warn "python3 is not installed; the patched warble setup.py was not compile-checked"
  fi

  built_lib=$(find "$src" -name 'libwarble.so' -type f 2>/dev/null | head -n 1) || true
  if [ -z "$built_lib" ]; then
    echo "deploy: building the debug warble in $src (first run, several minutes)"
    if ! make -C "$src/clibs/warble" -j"$(nproc)" CONFIG=debug SKIP_VERSION=1; then
      echo "deploy: error: the debug warble did not compile; the board needs make, g++ and libbluetooth-dev (deploy/README.md)" >&2
      exit 1
    fi
    built_lib=$(find "$src" -name 'libwarble.so' -type f 2>/dev/null | head -n 1) || true
    if [ -z "$built_lib" ]; then
      echo "deploy: error: make left no libwarble.so under $src (deploy/README.md)" >&2
      exit 1
    fi
  fi

  installed_lib=$(ls .venv/lib/python*/site-packages/mbientlab/warble/libwarble.so* 2>/dev/null | head -n 1) || true
  if [ -n "$installed_lib" ] && [ -f "$cached_lib" ] && cmp -s "$installed_lib" "$cached_lib"; then
    echo "deploy: the venv already has the debug warble: $PWD/$installed_lib"
  else
    if ! .venv/bin/python -c 'import setuptools' >/dev/null 2>&1; then
      echo "deploy: installing setuptools and wheel for the without-isolation build"
      if [ "@METHOD@" = wheels ]; then
        uv pip install --offline --find-links "@REMOTE_DIR@/wheels" --python .venv/bin/python setuptools wheel || {
          echo "deploy: error: could not install setuptools/wheel from @REMOTE_DIR@/wheels (deploy/README.md)" >&2
          exit 1
        }
      else
        uv pip install --python .venv/bin/python setuptools wheel || {
          echo "deploy: error: could not install setuptools/wheel (the board needs a package index or a warm uv cache; deploy/README.md)" >&2
          exit 1
        }
      fi
    fi
    # setup.py moves the built libwarble.so* into mbientlab/warble/ of the
    # source tree and refuses to overwrite them ("Destination path ... already
    # exists"), so a copy left by an earlier deploy breaks this install.
    rm -f "$src"/mbientlab/warble/libwarble.so*
    echo "deploy: installing the debug warble from $src"
    if ! uv pip install --python .venv/bin/python --no-build-isolation "$src"; then
      echo "deploy: error: uv pip install of the debug warble failed (deploy/README.md)" >&2
      exit 1
    fi
    installed_lib=$(ls .venv/lib/python*/site-packages/mbientlab/warble/libwarble.so* 2>/dev/null | head -n 1) || true
    if [ -z "$installed_lib" ]; then
      echo "deploy: error: the debug warble installed no libwarble.so in .venv (deploy/README.md)" >&2
      exit 1
    fi
    cp "$installed_lib" "$cached_lib.tmp" && mv "$cached_lib.tmp" "$cached_lib"
  fi

  if ! .venv/bin/python -c 'import mbientlab.warble'; then
    echo "deploy: error: mbientlab.warble does not import after installing the debug build (deploy/README.md)" >&2
    exit 1
  fi
  lib_size=$(wc -c < "$installed_lib")
  echo "deploy: mbientlab.warble imports; libwarble.so is $lib_size bytes ($(du -h "$installed_lib" | cut -f 1)): $PWD/$installed_lib"
fi

if [ "@SKIP_WARBLE@" = 1 ]; then
  echo "deploy: blue_st_sdk is patched for Python 3.10; the debug warble was skipped (--skip-warble)"
else
  echo "deploy: the board patches are in place (blue_st_sdk fixed, debug warble installed)"
fi
REMOTE
)
remote "$post_sync"

note "step 8: granting raw-socket capabilities to the bluepy helper"
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

note "step 9: installing and enabling the systemd units"
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
    warn "camera is missing, bluepy-helper has no capabilities (see step 8) or no"
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
