#!/usr/bin/env bash
# Run a command inside the pinned Vitis AI 3.5 CPU PyTorch container.
#
#   tools/vitis/run_vitis.sh bash tools/vitis/compile.sh build/vitis/quant/pose_halfbust62.xmodel
#   tools/vitis/run_vitis.sh                      # interactive shell in /repo
#
# The base image is pinned by tag and digest (see tools/vitis/README.md).  The
# derived image only adds Ultralytics, which the container needs to rebuild the
# deploy graph from the model YAML; the Vitis AI stack itself is untouched.
#
# Docker socket access: the script uses docker directly when the current user
# can reach the socket, otherwise it re-executes itself through `sg docker`.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
BASE_IMAGE="${VITIS_AI_BASE_IMAGE:-xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306}"
IMAGE="${VITIS_AI_IMAGE:-cpsa-vitis-pytorch:3.5.0.306}"

if ! docker info >/dev/null 2>&1; then
  if command -v sg >/dev/null 2>&1 && sg docker -c "docker info" >/dev/null 2>&1; then
    # shellcheck disable=SC2016
    exec sg docker -c "$(printf '%q ' "$SCRIPT_PATH" "$@")"
  fi
  echo "run_vitis.sh: no access to the docker socket (add the user to the docker group)" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "run_vitis.sh: building $IMAGE from $BASE_IMAGE" >&2
  docker build -t "$IMAGE" -f "$REPO_ROOT/tools/vitis/Dockerfile" "$REPO_ROOT/tools/vitis"
fi

docker_args=(
  --rm
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp/vitis-home
  -v "$REPO_ROOT:/repo"
  -w /repo
)
if [ -n "${VITIS_ARCH_JSON:-}" ]; then
  docker_args+=(-e "VITIS_ARCH_JSON=$VITIS_ARCH_JSON")
fi

if [ "$#" -eq 0 ]; then
  exec docker run -it "${docker_args[@]}" "$IMAGE" bash -l
fi

exec docker run "${docker_args[@]}" "$IMAGE" bash -lc '
  mkdir -p "$HOME"
  source /opt/vitis_ai/conda/etc/profile.d/conda.sh
  conda activate vitis-ai-pytorch
  exec "$@"
' vitis "$@"
