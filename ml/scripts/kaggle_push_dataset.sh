#!/usr/bin/env bash
# Publish a new version of a Kaggle dataset. Run from the repository root or ml/.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <dataset-folder> [version-message]" >&2
  exit 2
fi

folder=$1
message=${2:-"CPSA dataset update $(date -u +%Y-%m-%dT%H:%M:%SZ)"}

if [[ ! -d "$folder" ]]; then
  echo "dataset folder not found: $folder" >&2
  exit 1
fi

kaggle datasets version -p "$folder" -m "$message"
