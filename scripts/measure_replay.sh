#!/usr/bin/env bash
# Measure the live fps of the showcase on the board from a directory of photos.
#
# The camera is replaced with a loop of JPG/PNG images (`cpsa-showcase --replay
# DIR`), so the measurement needs nobody in front of the booth: the app runs the
# whole chain (replay source, detector, tracker, still gate, overlay, shared
# memory, web server, status fps, stage log) exactly as it does with the camera,
# and this script reports the fps and the stage timings of every view.
#
# Run it ON the board, from the repository root, with the service installed:
#
#     scripts/measure_replay.sh ~/cpsa/build/replay 15
#
# It stops cpsa-showcase.service (the DPU stays loaded by cpsa-dpu.service),
# starts the app with --replay for CASE_SECONDS per case, samples status fps and
# the stage log over the WebSocket, then stops the app and restarts the service
# if it was running. The images are not part of the repository: keep them under
# build/replay, which git ignores.
#
# Environment: CPSA_ROOT, CPSA_BIN, CPSA_PYTHON, CPSA_PORT, CPSA_LOG,
# CPSA_SERVICE, SUDO (empty when already root).
set -euo pipefail

CASE_SECONDS=${2:-15}
DIR=${1:-}

if [[ -z $DIR ]]; then
  echo "usage: $0 DIR [CASE_SECONDS]" >&2
  echo "  DIR: directory of JPG/PNG images to loop instead of the camera" >&2
  exit 2
fi
if [[ ! -d $DIR ]]; then
  echo "no replay directory: $DIR" >&2
  exit 2
fi

ROOT=${CPSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BIN=${CPSA_BIN:-$ROOT/.venv/bin/cpsa-showcase}
PYTHON=${CPSA_PYTHON:-$ROOT/.venv/bin/python}
PORT=${CPSA_PORT:-8080}
LOG=${CPSA_LOG:-$ROOT/build/replay-measure.log}
SERVICE=${CPSA_SERVICE:-cpsa-showcase.service}
SUDO=${SUDO-sudo}

if [[ ! -x $BIN ]]; then
  echo "no cpsa-showcase at $BIN (set CPSA_BIN)" >&2
  exit 2
fi
if [[ ! -x $PYTHON ]]; then
  echo "no python at $PYTHON (set CPSA_PYTHON)" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")"
: >"$LOG"

WAS_ACTIVE=0
if systemctl is-active --quiet "$SERVICE"; then
  WAS_ACTIVE=1
fi

APP_PID=""
cleanup() {
  local status=$?
  if [[ -n $APP_PID ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill -TERM "$APP_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$APP_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "$APP_PID" 2>/dev/null || true
  fi
  if [[ $WAS_ACTIVE -eq 1 ]]; then
    $SUDO systemctl start "$SERVICE" || true
  fi
  echo "app log: $LOG"
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "stopping $SERVICE (the DPU stays loaded by cpsa-dpu.service)"
$SUDO systemctl stop "$SERVICE"

echo "command: $BIN --config config/showcase.yaml --replay $DIR"
(
  cd "$ROOT"
  exec "$BIN" --config config/showcase.yaml --replay "$DIR"
) >>"$LOG" 2>&1 &
APP_PID=$!

HEALTH="http://127.0.0.1:$PORT/health"
READY=0
for _ in $(seq 1 240); do
  if curl -fsS --max-time 2 "$HEALTH" 2>/dev/null | grep -q '"state": *"running"'; then
    READY=1
    break
  fi
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "cpsa-showcase --replay exited during startup; see $LOG" >&2
    exit 1
  fi
  sleep 0.5
done
if [[ $READY -ne 1 ]]; then
  echo "$HEALTH did not report a running app; see $LOG" >&2
  exit 1
fi

echo "measuring ${CASE_SECONDS}s per case from $DIR"
"$PYTHON" "$ROOT/scripts/measure_replay.py" \
  --url "ws://127.0.0.1:$PORT/ws" \
  --log "$LOG" \
  --seconds "$CASE_SECONDS"
