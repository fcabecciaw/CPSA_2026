#!/usr/bin/env bash
# MoveNet (COCO-17 single pose, 192x192) from the Vitis AI 3.0 model zoo,
# compiled for the KV260 (M1).
#
#   tools/vitis/zoo_movenet.sh                 # download, extract, compile, inspect
#   tools/vitis/zoo_movenet.sh --skip-download # reuse build/models/zoo/movenet/pt.zip
#
# The package ships `quantized/MoveNet_int.xmodel` precompiled for the zoo's
# target, so, as with YOLOX (AC-S1), this script recompiles that int8 graph with
# the repo's pinned-fingerprint `compile.sh` to get a 0x101000016010407 xmodel
# that loads on our board.  No training, no requantization.
#
# The compiled graph keeps only the four raw heads on the DPU (192x192x3 int8
# in, four 48x48 int8 heads out); the sigmoid, the center prior and the
# keypoint search stay on the CPU in `src/cpsa/movenet.py`.  The input fix
# point is 6, i.e. the int8 grid covers [-2, 2), so the runtime feeds
# `pixel / 128` (see MOVENET_INPUT_SCALE).
#
# Inputs (git-ignored):
#
#   build/models/zoo/movenet/pt.zip                        downloaded package
#   build/models/zoo/movenet/pt_movenet-.../               extracted package
#
# Outputs:
#
#   build/vitis/movenet/compiled/movenet.xmodel            compiled graph
#   build/vitis/report/inspect_movenet.json                tensor/fix-point report
#   build/vitis/logs/zoo_movenet_*.log                     compile/inspect logs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ZOO_DIR="${MOVENET_ZOO_DIR:-$REPO_ROOT/build/models/zoo/movenet}"
PACKAGE="pt_movenet_coco_192_192_0.5G_3.0"
PACKAGE_DIR="$ZOO_DIR/$PACKAGE"
URL="${MOVENET_ZOO_URL:-https://www.xilinx.com/bin/public/openDownload?filename=pt_movenet_coco_192_192_0.5G_3.0.zip}"
COMPILED_DIR="${MOVENET_COMPILED_DIR:-$REPO_ROOT/build/vitis/movenet/compiled}"
REPORT_DIR="${MOVENET_REPORT_DIR:-$REPO_ROOT/build/vitis/report}"
LOG_DIR="$REPO_ROOT/build/vitis/logs"
NAME="movenet"
FINGERPRINT="0x101000016010407"
INT_XMODEL="$PACKAGE_DIR/quantized/MoveNet_int.xmodel"

download=1
for arg in "$@"; do
  case "$arg" in
    --skip-download) download=0 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "zoo_movenet.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$ZOO_DIR" "$COMPILED_DIR" "$REPORT_DIR" "$LOG_DIR"
zip_path="$ZOO_DIR/pt.zip"

if [ ! -f "$INT_XMODEL" ]; then
  if [ ! -f "$zip_path" ]; then
    if [ "$download" -eq 0 ]; then
      echo "zoo_movenet.sh: $zip_path is missing and --skip-download was given" >&2
      exit 1
    fi
    echo "zoo_movenet.sh: downloading $URL"
    curl -fL --retry 3 --max-time 1800 -o "$zip_path.part" "$URL"
    unzip -tq "$zip_path.part" >/dev/null
    mv "$zip_path.part" "$zip_path"
  else
    echo "zoo_movenet.sh: using existing $zip_path"
  fi
  echo "zoo_movenet.sh: extracting $zip_path"
  unzip -oq "$zip_path" -d "$ZOO_DIR"
fi

if [ ! -f "$INT_XMODEL" ]; then
  echo "zoo_movenet.sh: $INT_XMODEL is missing after extraction" >&2
  exit 1
fi

# The container mounts the repository at /repo, so every path it sees must be
# repository relative (see variants.sh).
rel() {
  case "$1" in
    "$REPO_ROOT"/*) printf '%s\n' "${1#"$REPO_ROOT"/}" ;;
    *) echo "zoo_movenet.sh: $1 is outside the repository" >&2; exit 1 ;;
  esac
}

rel_int_xmodel="$(rel "$INT_XMODEL")"
rel_compiled="$(rel "$COMPILED_DIR")"
rel_report="$(rel "$REPORT_DIR/inspect_movenet.json")"

echo "zoo_movenet.sh: compiling $rel_int_xmodel for $FINGERPRINT"
( cd "$REPO_ROOT" && "$SCRIPT_DIR/run_vitis.sh" bash tools/vitis/compile.sh \
    "$rel_int_xmodel" "$rel_compiled" "$NAME" ) \
  2>&1 | tee "$LOG_DIR/zoo_movenet_compile.log"

echo "zoo_movenet.sh: inspecting $rel_compiled/$NAME.xmodel"
( cd "$REPO_ROOT" && "$SCRIPT_DIR/run_vitis.sh" python tools/vitis/inspect_xmodel.py \
    "$rel_compiled/$NAME.xmodel" --out "$rel_report" ) \
  2>&1 | tee "$LOG_DIR/zoo_movenet_inspect.log"

python3 - "$REPORT_DIR/inspect_movenet.json" "$FINGERPRINT" <<'PY'
import json
import sys

path, expected = sys.argv[1], sys.argv[2].lower()
report = json.load(open(path))
problems = []
fingerprint = (report.get("fingerprint") or "").lower()
if fingerprint != expected:
    problems.append(f"fingerprint {fingerprint or 'missing'} != {expected}")
if not report.get("fingerprint_matches"):
    problems.append("fingerprint_matches is false")
if report.get("dpu_subgraph_count") != 1:
    problems.append(f"expected one DPU subgraph, got {report.get('dpu_subgraph_count')}")
if report.get("input_size_hw") != [192, 192]:
    problems.append(f"expected a 192x192 input, got {report.get('input_size_hw')}")
dpu_inputs = report.get("dpu_inputs") or []
if len(dpu_inputs) != 1 or dpu_inputs[0]["dims"] != [1, 192, 192, 3]:
    problems.append(f"expected one [1, 192, 192, 3] DPU input, got {dpu_inputs}")
else:
    if dpu_inputs[0].get("dtype") != "xint8":
        problems.append(f"expected an xint8 input, got {dpu_inputs[0].get('dtype')}")
    if dpu_inputs[0].get("fix_point") != 6:
        problems.append(f"expected input fix_point 6, got {dpu_inputs[0].get('fix_point')}")

# head -> (channels, fix_point), as `src/cpsa/movenet.py` expects them.
heads = {"offsets": (34, 7), "regs": (34, 2), "centers": (1, 3), "heatmaps": (17, 3)}
outputs = report.get("output_order") or []
if len(outputs) != len(heads):
    problems.append(f"expected 4 DPU heads, got {len(outputs)}")
seen = set()
for tensor in outputs:
    head = next((name for name in heads if name in tensor["name"].lower()), None)
    if head is None or head in seen:
        problems.append(f"unexpected or repeated head {tensor['name']}")
        continue
    seen.add(head)
    channels, fix_point = heads[head]
    if tensor["dims"][3] != channels:
        problems.append(f"{head} must have {channels} channels, got {tensor['dims']}")
    if tensor.get("stride") != 4:
        problems.append(f"{head} must be at stride 4, got {tensor.get('stride')}")
    if tensor.get("fix_point") != fix_point:
        problems.append(f"{head} must have fix_point {fix_point}, got {tensor.get('fix_point')}")
    if tensor.get("dtype") != "xint8":
        problems.append(f"expected xint8 output, got {tensor.get('dtype')}")
if seen != set(heads):
    problems.append(f"missing heads {sorted(set(heads) - seen)}")
if problems:
    raise SystemExit("zoo_movenet.sh: " + "; ".join(problems))
print(
    "zoo_movenet.sh: ok "
    f"{report['fingerprint']} {report['target']} "
    f"input {dpu_inputs[0]['dims']} fix_point {dpu_inputs[0].get('fix_point')} "
    f"outputs {[(t['dims'], t.get('fix_point')) for t in outputs]}"
)
PY

echo "zoo_movenet.sh: xmodel $COMPILED_DIR/$NAME.xmodel"
echo "zoo_movenet.sh: report $REPORT_DIR/inspect_movenet.json"
