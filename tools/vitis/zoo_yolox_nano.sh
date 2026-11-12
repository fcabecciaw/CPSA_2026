#!/usr/bin/env bash
# YOLOX-nano (COCO-80, 416x416) from the Vitis AI 3.0 model zoo, compiled for
# the KV260 (AC-S1).
#
#   tools/vitis/zoo_yolox_nano.sh              # download, extract, compile, inspect
#   tools/vitis/zoo_yolox_nano.sh --skip-download  # reuse build/models/zoo/yolox_nano/pt.zip
#
# The zoo's *prebuilt* xmodel (`yolox_nano_pt-zcu102_zcu104_kv260`) targets
# DPU fingerprint 0x101000056010407 and does not load on our board
# (0x101000016010407), so this script uses the float+QAT package instead: the
# QAT-converted `qat/convert_qat_results/YOLOX_0_int.xmodel` compiled with the
# repo's pinned-fingerprint `compile.sh` matches the board.  No training, no
# requantization.
#
# Inputs (git-ignored):
#
#   build/models/zoo/yolox_nano/pt.zip                     downloaded package
#   build/models/zoo/yolox_nano/pt_yolox-.../              extracted package
#
# Outputs:
#
#   build/vitis/yolox_nano/compiled/yolox_nano.xmodel      compiled graph
#   build/vitis/report/inspect_yolox_nano.json             tensor/fix-point report
#   build/vitis/logs/zoo_yolox_nano_*.log                  compile/inspect logs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ZOO_DIR="${YOLOX_ZOO_DIR:-$REPO_ROOT/build/models/zoo/yolox_nano}"
PACKAGE="pt_yolox-nano_coco_416_416_1G_3.0"
PACKAGE_DIR="$ZOO_DIR/$PACKAGE"
URL="${YOLOX_ZOO_URL:-https://www.xilinx.com/bin/public/openDownload?filename=pt_yolox-nano_coco_416_416_1G_3.0.zip}"
COMPILED_DIR="${YOLOX_COMPILED_DIR:-$REPO_ROOT/build/vitis/yolox_nano/compiled}"
REPORT_DIR="${YOLOX_REPORT_DIR:-$REPO_ROOT/build/vitis/report}"
LOG_DIR="$REPO_ROOT/build/vitis/logs"
NAME="yolox_nano"
FINGERPRINT="0x101000016010407"
INT_XMODEL="$PACKAGE_DIR/qat/convert_qat_results/YOLOX_0_int.xmodel"

download=1
for arg in "$@"; do
  case "$arg" in
    --skip-download) download=0 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "zoo_yolox_nano.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$ZOO_DIR" "$COMPILED_DIR" "$REPORT_DIR" "$LOG_DIR"
zip_path="$ZOO_DIR/pt.zip"

if [ ! -f "$INT_XMODEL" ]; then
  if [ ! -f "$zip_path" ]; then
    if [ "$download" -eq 0 ]; then
      echo "zoo_yolox_nano.sh: $zip_path is missing and --skip-download was given" >&2
      exit 1
    fi
    echo "zoo_yolox_nano.sh: downloading $URL"
    curl -fL --retry 3 --max-time 1800 -o "$zip_path.part" "$URL"
    unzip -tq "$zip_path.part" >/dev/null
    mv "$zip_path.part" "$zip_path"
  else
    echo "zoo_yolox_nano.sh: using existing $zip_path"
  fi
  echo "zoo_yolox_nano.sh: extracting $zip_path"
  unzip -oq "$zip_path" -d "$ZOO_DIR"
fi

if [ ! -f "$INT_XMODEL" ]; then
  echo "zoo_yolox_nano.sh: $INT_XMODEL is missing after extraction" >&2
  exit 1
fi
if [ ! -f "$PACKAGE_DIR/qat/convert_qat_results/deployable.pth" ]; then
  echo "zoo_yolox_nano.sh: the reference weights deployable.pth are missing" >&2
  exit 1
fi

# The container mounts the repository at /repo, so every path it sees must be
# repository relative (see variants.sh).
rel() {
  case "$1" in
    "$REPO_ROOT"/*) printf '%s\n' "${1#"$REPO_ROOT"/}" ;;
    *) echo "zoo_yolox_nano.sh: $1 is outside the repository" >&2; exit 1 ;;
  esac
}

rel_int_xmodel="$(rel "$INT_XMODEL")"
rel_compiled="$(rel "$COMPILED_DIR")"
rel_report="$(rel "$REPORT_DIR/inspect_yolox_nano.json")"

echo "zoo_yolox_nano.sh: compiling $rel_int_xmodel for $FINGERPRINT"
( cd "$REPO_ROOT" && "$SCRIPT_DIR/run_vitis.sh" bash tools/vitis/compile.sh \
    "$rel_int_xmodel" "$rel_compiled" "$NAME" ) \
  2>&1 | tee "$LOG_DIR/zoo_yolox_nano_compile.log"

echo "zoo_yolox_nano.sh: inspecting $rel_compiled/$NAME.xmodel"
( cd "$REPO_ROOT" && "$SCRIPT_DIR/run_vitis.sh" python tools/vitis/inspect_xmodel.py \
    "$rel_compiled/$NAME.xmodel" --out "$rel_report" ) \
  2>&1 | tee "$LOG_DIR/zoo_yolox_nano_inspect.log"

python3 - "$REPORT_DIR/inspect_yolox_nano.json" "$FINGERPRINT" <<'PY'
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
if report.get("input_size_hw") != [416, 416]:
    problems.append(f"expected a 416x416 input, got {report.get('input_size_hw')}")
dpu_inputs = report.get("dpu_inputs") or []
if len(dpu_inputs) != 1 or dpu_inputs[0]["dims"] != [1, 416, 416, 3]:
    problems.append(f"expected one [1, 416, 416, 3] DPU input, got {dpu_inputs}")
outputs = report.get("output_order") or []
strides = [tensor.get("stride") for tensor in outputs]
if strides != [8, 16, 32]:
    problems.append(f"expected output strides [8, 16, 32], got {strides}")
for tensor in outputs:
    if tensor["dims"][3] != 85:
        problems.append(f"expected 85 output channels, got {tensor['dims']} for {tensor['name']}")
    if tensor.get("dtype") != "xint8":
        problems.append(f"expected xint8 output, got {tensor.get('dtype')}")
if problems:
    raise SystemExit("zoo_yolox_nano.sh: " + "; ".join(problems))
print(
    "zoo_yolox_nano.sh: ok "
    f"{report['fingerprint']} {report['target']} "
    f"input {dpu_inputs[0]['dims']} fix_point {dpu_inputs[0].get('fix_point')} "
    f"outputs {[tensor['dims'] for tensor in outputs]}"
)
PY

echo "zoo_yolox_nano.sh: xmodel $COMPILED_DIR/$NAME.xmodel"
echo "zoo_yolox_nano.sh: report $REPORT_DIR/inspect_yolox_nano.json"
