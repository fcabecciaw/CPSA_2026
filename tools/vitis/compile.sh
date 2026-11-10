#!/usr/bin/env bash
# Compile a quantized `.xmodel` for the KV260 DPU (runs inside the container).
#
#   tools/vitis/compile.sh build/vitis/quant/pose_halfbust62.xmodel \
#       build/vitis/compiled pose_halfbust62
#
# The KV260 benchmark B4096 firmware reports fingerprint 0x101000016010407.
# Vitis AI 3.5's target alias `DPUCZDX8G_ISA1_B4096` (the arch.json shipped in
# /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260) maps to 0x101000056010407 (the
# 3.x feature code), and the DPU runner rejects a model built that way with
# "model_fingerprint ... is un-matched with actual dpu_fingerprint".  The script
# therefore compiles against an explicit fingerprint file, as AMD documents for
# cross-version targets, and verifies the fingerprint written into the artifact.
#
# Override the fingerprint with VITIS_DPU_FINGERPRINT, or pass a complete arch
# file with VITIS_ARCH_JSON.
set -euo pipefail

xmodel="${1:?usage: compile.sh <quantized.xmodel> [out_dir] [net_name]}"
out="${2:-$(dirname "$xmodel")/../compiled}"
name="${3:-$(basename "$xmodel" .xmodel)}"
fingerprint="${VITIS_DPU_FINGERPRINT:-0x101000016010407}"
target="DPUCZDX8G_ISA1_B4096"

mkdir -p "$out"
if [ -n "${VITIS_ARCH_JSON:-}" ]; then
  arch="$VITIS_ARCH_JSON"
  if [ ! -f "$arch" ]; then
    echo "compile.sh: arch file not found: $arch" >&2
    exit 1
  fi
  echo "compile.sh: using VITIS_ARCH_JSON=$arch"
else
  arch="$out/arch_kv260.json"
  printf '{"fingerprint": "%s"}\n' "$fingerprint" > "$arch"
  echo "compile.sh: compiling for $target fingerprint $fingerprint ($arch)"
  echo "compile.sh: Vitis AI 3.5's target alias for $target is 0x101000056010407"
fi

vai_c_xir -x "$xmodel" -a "$arch" -o "$out" -n "$name"

python - "$out/$name.xmodel" "$fingerprint" <<'PY'
import sys

import xir

path, expected = sys.argv[1], sys.argv[2].lower()
graph = xir.Graph.deserialize(path)
found = None
target = None
for subgraph in graph.get_root_subgraph().toposort_child_subgraph():
    if subgraph.has_attr("dpu_fingerprint"):
        found = hex(int(subgraph.get_attr("dpu_fingerprint")))
        target = subgraph.get_attr("dpu_name")
        break
print(f"compile.sh: compiled fingerprint {found} for {target} (expected {expected})")
if found is None or found.lower() != expected:
    raise SystemExit("compile.sh: compiled xmodel fingerprint mismatch")
PY

ls -l "$out"
