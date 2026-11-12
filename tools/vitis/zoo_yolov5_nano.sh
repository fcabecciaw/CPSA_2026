#!/usr/bin/env bash
# YOLOv5-nano (COCO-80, 640x640) from the Vitis AI 3.0 model zoo, prepared for
# the KV260 (AC-S1).
#
#   tools/vitis/zoo_yolov5_nano.sh              # download, extract, prepare, inspect
#   tools/vitis/zoo_yolov5_nano.sh --skip-download  # reuse the files in build/models/zoo
#
# Two routes, in order of preference:
#
# 1. Package route.  If the float+QAT package
#    `pt_yolov5-nano_coco_640_640_4.6G_3.0.zip` is present and holds a converted
#    `qat/convert_qat_results/*_int.xmodel`, that int8 graph is compiled with the
#    repo's pinned-fingerprint `compile.sh`, exactly like `zoo_yolox_nano.sh`.
#    As of 2026 the zoo returns a 1005-byte README-only archive for this
#    package (the YOLO packages are no longer downloadable; the README tells
#    customers to contact AMD), so this route normally does not trigger.
#
# 2. Prebuilt route.  The prebuilt `yolov5_nano_pt-zcu102_zcu104_kv260-r3.0.0`
#    tar *is* downloadable.  It is already compiled (Vitis AI 3.0, target
#    DPUCZDX8G_ISA1_B4096) and only records fingerprint 0x101000056010407, which
#    the KV260 firmware rejects.  `restamp_xmodel.py` rewrites that one protobuf
#    varint to 0x101000016010407 without touching the graph or the instructions.
#    This is a fingerprint rewrite of a prebuilt artifact, **not** a compile, and
#    the re-stamped file has not been loaded on a board: report it as such.
#
# Inputs (git-ignored):
#
#   build/models/zoo/yolov5_nano/model.yaml                       zoo metadata
#   build/models/zoo/yolov5_nano/prebuilt_kv260.tar.gz            prebuilt KV260 tar
#   build/models/zoo/yolov5_nano/yolov5_nano_pt/                  extracted artifacts
#   build/models/zoo/yolov5_nano/pt.zip                           float+QAT package, if any
#
# Outputs:
#
#   build/vitis/yolov5_nano/compiled/yolov5_nano.xmodel           graph for the KV260
#   build/vitis/report/inspect_yolov5_nano.json                   tensor/fix-point report
#   build/vitis/logs/zoo_yolov5_nano_*.log                        compile/inspect logs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ZOO_DIR="${YOLOV5_ZOO_DIR:-$REPO_ROOT/build/models/zoo/yolov5_nano}"
PACKAGE="pt_yolov5-nano_coco_640_640_4.6G_3.0"
PACKAGE_DIR="$ZOO_DIR/$PACKAGE"
PKG_URL="${YOLOV5_ZOO_URL:-https://www.xilinx.com/bin/public/openDownload?filename=pt_yolov5-nano_coco_640_640_4.6G_3.0.zip}"
PREBUILT_URL="${YOLOV5_PREBUILT_URL:-https://www.xilinx.com/bin/public/openDownload?filename=yolov5_nano_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz}"
PREBUILT_MD5="61210fe6d964e34301af0640e0b17d93"
PREBUILT_DIR="$ZOO_DIR/yolov5_nano_pt"
PREBUILT_XMODEL="$PREBUILT_DIR/yolov5_nano_pt.xmodel"
COMPILED_DIR="${YOLOV5_COMPILED_DIR:-$REPO_ROOT/build/vitis/yolov5_nano/compiled}"
REPORT_DIR="${YOLOV5_REPORT_DIR:-$REPO_ROOT/build/vitis/report}"
LOG_DIR="$REPO_ROOT/build/vitis/logs"
NAME="yolov5_nano"
FINGERPRINT="0x101000016010407"

download=1
for arg in "$@"; do
  case "$arg" in
    --skip-download) download=0 ;;
    -h|--help) sed -n "2,37p" "$0"; exit 0 ;;
    *) echo "zoo_yolov5_nano.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$ZOO_DIR" "$COMPILED_DIR" "$REPORT_DIR" "$LOG_DIR"

# The container mounts the repository at /repo, so every path it sees must be
# repository relative (see compile.sh).
rel() {
  case "$1" in
    "$REPO_ROOT"/*) printf '%s\n' "${1#"$REPO_ROOT"/}" ;;
    *) echo "zoo_yolov5_nano.sh: $1 is outside the repository" >&2; exit 1 ;;
  esac
}

# --- prebuilt artifact (route 2 and the fallback) -------------------------- #
tar_path="$ZOO_DIR/prebuilt_kv260.tar.gz"
if [ ! -f "$PREBUILT_XMODEL" ]; then
  if [ ! -f "$tar_path" ]; then
    if [ "$download" -eq 0 ]; then
      echo "zoo_yolov5_nano.sh: $tar_path is missing and --skip-download was given" >&2
      exit 1
    fi
    echo "zoo_yolov5_nano.sh: downloading $PREBUILT_URL"
    curl -fL --retry 3 --max-time 1800 -o "$tar_path.part" "$PREBUILT_URL"
    echo "$PREBUILT_MD5  $tar_path.part" | md5sum -c - >/dev/null \
      || { echo "zoo_yolov5_nano.sh: $tar_path.part does not match $PREBUILT_MD5" >&2; exit 1; }
    mv "$tar_path.part" "$tar_path"
  else
    echo "zoo_yolov5_nano.sh: using existing $tar_path"
  fi
  echo "zoo_yolov5_nano.sh: extracting $tar_path"
  tar xzf "$tar_path" -C "$ZOO_DIR"
fi
if [ ! -f "$PREBUILT_XMODEL" ]; then
  echo "zoo_yolov5_nano.sh: $PREBUILT_XMODEL is missing after extraction" >&2
  exit 1
fi

# --- float+QAT package (route 1, normally unavailable) --------------------- #
if [ "$download" -eq 1 ] && [ ! -d "$PACKAGE_DIR" ]; then
  zip_path="$ZOO_DIR/pt.zip"
  echo "zoo_yolov5_nano.sh: fetching $PKG_URL"
  curl -fL --retry 3 --max-time 1800 -o "$zip_path.part" "$PKG_URL" || true
  if [ -s "$zip_path.part" ] && unzip -tq "$zip_path.part" >/dev/null 2>&1; then
    mv "$zip_path.part" "$zip_path"
    unzip -oq "$zip_path" -d "$ZOO_DIR"
  else
    rm -f "$zip_path.part"
    echo "zoo_yolov5_nano.sh: the float+QAT package is not downloadable; using the prebuilt tar"
  fi
fi

int_xmodel=""
if [ -d "$PACKAGE_DIR" ]; then
  int_xmodel="$(find "$PACKAGE_DIR" -path '*convert_qat_results*' -name '*_int.xmodel' -print -quit)"
fi

if [ -n "$int_xmodel" ]; then
  echo "zoo_yolov5_nano.sh: compiling $(rel "$int_xmodel") for $FINGERPRINT"
  ( cd "$REPO_ROOT" && "$SCRIPT_DIR/run_vitis.sh" bash tools/vitis/compile.sh \
      "$(rel "$int_xmodel")" "$(rel "$COMPILED_DIR")" "$NAME" ) \
    2>&1 | tee "$LOG_DIR/zoo_yolov5_nano_compile.log"
else
  echo "zoo_yolov5_nano.sh: no package int8 xmodel; re-stamping the prebuilt artifact"
  echo "zoo_yolov5_nano.sh: this is a fingerprint rewrite of a Vitis AI 3.0 compile, not a compile"
  python3 "$SCRIPT_DIR/restamp_xmodel.py" "$PREBUILT_XMODEL" "$COMPILED_DIR/$NAME.xmodel" \
    2>&1 | tee "$LOG_DIR/zoo_yolov5_nano_restamp.log"
fi

echo "zoo_yolov5_nano.sh: inspecting $(rel "$COMPILED_DIR/$NAME.xmodel")"
( cd "$REPO_ROOT" && "$SCRIPT_DIR/run_vitis.sh" python tools/vitis/inspect_xmodel.py \
    "$(rel "$COMPILED_DIR/$NAME.xmodel")" --out "$(rel "$REPORT_DIR/inspect_yolov5_nano.json")" ) \
  2>&1 | tee "$LOG_DIR/zoo_yolov5_nano_inspect.log"

python3 - "$REPORT_DIR/inspect_yolov5_nano.json" "$FINGERPRINT" <<'PY'
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
if report.get("target") != "DPUCZDX8G_ISA1_B4096":
    problems.append(f"unexpected target {report.get('target')}")
if report.get("dpu_subgraph_count") != 1:
    problems.append(f"expected one DPU subgraph, got {report.get('dpu_subgraph_count')}")
if report.get("input_size_hw") != [640, 640]:
    problems.append(f"expected a 640x640 input, got {report.get('input_size_hw')}")
dpu_inputs = report.get("dpu_inputs") or []
if len(dpu_inputs) != 1 or dpu_inputs[0]["dims"] != [1, 640, 640, 3]:
    problems.append(f"expected one [1, 640, 640, 3] DPU input, got {dpu_inputs}")
elif dpu_inputs[0].get("dtype") != "xint8":
    problems.append(f"expected an xint8 input, got {dpu_inputs[0].get('dtype')}")
outputs = report.get("output_order") or []
strides = [tensor.get("stride") for tensor in outputs]
if strides != [8, 16, 32]:
    problems.append(f"expected output strides [8, 16, 32], got {strides}")
for tensor in outputs:
    if tensor["dims"][3] != 255:
        problems.append(f"expected 255 output channels, got {tensor['dims']} for {tensor['name']}")
    if tensor.get("dtype") != "xint8":
        problems.append(f"expected an xint8 output, got {tensor.get('dtype')}")
if problems:
    raise SystemExit("zoo_yolov5_nano.sh: " + "; ".join(problems))
print(
    "zoo_yolov5_nano.sh: ok "
    f"{report['fingerprint']} {report['target']} "
    f"input {dpu_inputs[0]['dims']} fix_point {dpu_inputs[0].get('fix_point')} "
    f"outputs {[tensor['dims'] for tensor in outputs]} "
    f"fix_points {[tensor.get('fix_point') for tensor in outputs]}"
)
PY

echo "zoo_yolov5_nano.sh: xmodel $COMPILED_DIR/$NAME.xmodel"
echo "zoo_yolov5_nano.sh: report $REPORT_DIR/inspect_yolov5_nano.json"
