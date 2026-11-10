#!/usr/bin/env bash
# Build the KV260 DPU speed matrix: activation x input size.
#
# For every cell this exports the raw-head deploy bundle, runs PTQ
# (`quantize.py`), compiles for the KV260 (`compile.sh`) and inspects the
# artifact (`inspect_xmodel.py`).  It then enforces the two structural
# invariants of the matrix -- exactly one DPU subgraph, no compute on the CPU
# (only the `data-fix`/`fix2float` dtype boundaries), and the pinned
# fingerprint 0x101000016010407 -- and writes a one-line summary per variant.
#
#   tools/vitis/variants.sh                                  # full 3x3 matrix
#   tools/vitis/variants.sh --activations relu --sizes 288x384
#   tools/vitis/variants.sh --force                          # rebuild everything
#
# Outputs (git-ignored):
#
#   build/vitis/variants/<act>_<H>x<W>/artifacts   host deploy bundle
#   build/vitis/variants/<act>_<H>x<W>/quant       quantized .xmodel + report
#   build/vitis/variants/<act>_<H>x<W>/compiled    compiled KV260 .xmodel
#   build/vitis/variants/<act>_<H>x<W>/report      inspect_compiled.json, summary.json
#   build/vitis/variants/<act>_<H>x<W>/logs        export/quantize/compile/inspect logs
#   build/vitis/variants/calib_<H>x<W>             calibration frames (shared per size)
#   build/vitis/variants/matrix.tsv                one row per built variant
#
# Calibration frames depend only on the input size, so they are generated once
# per size and shared by the three activations.  Board latency for a variant:
#
#   PYTHONPATH=src python3 scripts/bench_pose.py --xmodel pose_<act>.xmodel \
#     --layout halfbust62 --frames 200 --warmup 10 --conf-thr 0.25 --json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_ROOT="$REPO_ROOT/build/vitis/variants"

ACTIVATIONS="hswish,relu,lrelu"
SIZES="480x640,384x512,288x384"
CALIB_COUNT=40
TEST_COUNT=8
THREADS=1
FORCE=0

usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Options:
  --activations LIST   comma list of activations (default hswish,relu,lrelu)
  --sizes LIST         comma list of HxW inputs (default 480x640,384x512,288x384)
  --count N            calibration frames per size (default 40)
  --test-count N       test frames per variant (default 8)
  --threads N          torch/OpenMP thread cap for PTQ (default 1)
  --force              rebuild calibration and every variant, even if present
  -h, --help           show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --activations) ACTIVATIONS="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --count) CALIB_COUNT="$2"; shift 2 ;;
    --test-count) TEST_COUNT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "variants.sh: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a activations <<< "$ACTIVATIONS"
IFS=',' read -r -a sizes <<< "$SIZES"

size_height() { echo "${1%x*}"; }
size_width() { echo "${1#*x}"; }

step() { printf '\n=== %s ===\n' "$*"; }

# -- calibration ----------------------------------------------------------

calib_ready() {
  local dir="$1" size="$2"
  [ -f "$dir/manifest.json" ] || return 1
  python3 - "$dir/manifest.json" "$CALIB_COUNT" "$(size_height "$size")" "$(size_width "$size")" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
frames, height, width = (int(value) for value in sys.argv[2:5])
ok = manifest.get("frames") == frames and manifest.get("size_hw") == [height, width]
raise SystemExit(0 if ok else 1)
PY
}

make_calibration() {
  local size="$1"
  local dir="$OUT_ROOT/calib_$size"
  if [ "$FORCE" -eq 0 ] && calib_ready "$dir" "$size"; then
    echo "calib $size: reuse $dir"
    return
  fi
  echo "calib $size: generating $CALIB_COUNT frames"
  mkdir -p "$dir"
  ( cd "$REPO_ROOT" && uv run python tools/vitis/make_calib.py \
      --count "$CALIB_COUNT" --height "$(size_height "$size")" \
      --width "$(size_width "$size")" --out "$dir" ) > "$dir/make_calib.log" 2>&1
  calib_ready "$dir" "$size" || {
    echo "variants.sh: calibration manifest mismatch in $dir" >&2
    exit 1
  }
}

# -- structural guards ----------------------------------------------------

# Fail unless the artifact has one DPU subgraph, the pinned fingerprint and no
# compute outside the DPU (only the int8<->float boundaries may run on the CPU).
guard_inspect() {
  local report="$1" act="$2" size="$3"
  python3 - "$report" "$act" "$size" <<'PY'
import json
import sys

path, activation, size = sys.argv[1], sys.argv[2], sys.argv[3]
height, width = (int(value) for value in size.split("x"))
report = json.load(open(path, encoding="utf-8"))
failures = []
if not report.get("fingerprint_matches"):
    failures.append(f"fingerprint {report.get('fingerprint')} != 0x101000016010407")
if report.get("dpu_subgraph_count") != 1:
    failures.append(f"expected 1 DPU subgraph, got {report.get('dpu_subgraph_count')}")
dpu_inputs = report.get("dpu_inputs") or []
if not dpu_inputs:
    failures.append("the DPU subgraph has no input tensor")
else:
    dims = dpu_inputs[0]["dims"]
    if dims not in ([1, height, width, 3], [1, 3, height, width]):
        failures.append(f"DPU input {dims} != 1x3x{height}x{width}")
boundary = {"data-fix", "fix2float", "float2fix"}
for subgraph in report.get("subgraphs", []):
    if subgraph["device"] == "DPU":
        continue
    extra = sorted(set(subgraph["op_types"]) - boundary)
    if extra:
        failures.append(f"CPU subgraph {subgraph['index']} computes {extra}")
if failures:
    raise SystemExit(
        f"variants.sh: {activation} {size} failed the graph guard:\n  "
        + "\n  ".join(failures)
    )
PY
}

# One summary JSON (and TSV row) per variant, from the inspect report.
write_summary() {
  local report="$1" xmodel="$2" act="$3" size="$4" out="$5"
  python3 - "$report" "$xmodel" "$act" "$size" "$out" <<'PY'
import json
import sys
from pathlib import Path

report_path, xmodel, activation, size, out = sys.argv[1:6]
report = json.load(open(report_path, encoding="utf-8"))
dpu = next(sub for sub in report["subgraphs"] if sub["device"] == "DPU")
ops = dpu["op_types"]
interesting = {
    "conv2d": sum(count for name, count in ops.items() if name.startswith("conv2d")),
    "eltwise": sum(count for name, count in ops.items() if name.startswith("eltwise")),
    "hard_sigmoid": sum(count for name, count in ops.items() if name.startswith("hard-sigmoid")),
    "leakyrelu": sum(count for name, count in ops.items() if "leaky" in name),
    "relu": sum(count for name, count in ops.items() if name.startswith("relu")),
}
summary = {
    "activation": activation,
    "size_hw": [int(value) for value in size.split("x")],
    "xmodel": xmodel,
    "xmodel_size_bytes": Path(xmodel).stat().st_size,
    "fingerprint": report["fingerprint"],
    "fingerprint_matches": report["fingerprint_matches"],
    "subgraph_count": report["subgraph_count"],
    "dpu_subgraph_count": report["dpu_subgraph_count"],
    "dpu_op_count": dpu["op_count"],
    "dpu_op_types": ops,
    "dpu_op_families": interesting,
    "dpu_input": report["dpu_inputs"][0],
    "output_order": [
        {key: item[key] for key in ("stride", "dims", "fix_point")}
        for item in report["output_order"]
    ],
}
Path(out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(
    f"{activation} {size}: dpu_ops conv={interesting['conv2d']} "
    f"eltwise={interesting['eltwise']} hard-sigmoid={interesting['hard_sigmoid']} "
    f"leakyrelu={interesting['leakyrelu']} total={dpu['op_count']} "
    f"xmodel={summary['xmodel_size_bytes']} bytes"
)
PY
}

# -- one variant ----------------------------------------------------------

build_variant() {
  local act="$1" size="$2" calib="$3"
  local height width dir rel summary
  height="$(size_height "$size")"
  width="$(size_width "$size")"
  dir="$OUT_ROOT/${act}_${size}"
  rel="${dir#"$REPO_ROOT"/}"
  summary="$dir/report/summary.json"

  if [ "$FORCE" -eq 0 ] && [ -f "$summary" ]; then
    echo "variant $act $size: reuse $dir (--force to rebuild)"
    return
  fi

  step "variant $act ${height}x${width}"
  mkdir -p "$dir/logs" "$dir/report"

  echo "export   -> $rel/artifacts"
  ( cd "$REPO_ROOT/ml" && uv run python -m cpsa_ml.deploy \
      --schema halfbust62 --seed 0 --activation "$act" \
      --cfg "configs/yolov8n-pose-$act.yaml" \
      --height "$height" --width "$width" \
      --out "$dir/artifacts" ) > "$dir/logs/export.log" 2>&1

  echo "quantize -> $rel/quant (threads=$THREADS)"
  ( cd "$REPO_ROOT" && "$SCRIPT_DIR/run_vitis.sh" python tools/vitis/quantize.py \
      --artifacts "$rel/artifacts" --calib "${calib#"$REPO_ROOT"/}" \
      --out "$rel/quant" --name "pose_$act" \
      --calib-count "$CALIB_COUNT" --test-count "$TEST_COUNT" \
      --threads "$THREADS" ) > "$dir/logs/quantize.log" 2>&1

  echo "compile  -> $rel/compiled"
  ( cd "$REPO_ROOT" && "$SCRIPT_DIR/run_vitis.sh" bash tools/vitis/compile.sh \
      "$rel/quant/pose_$act.xmodel" "$rel/compiled" "pose_$act" ) \
      > "$dir/logs/compile.log" 2>&1

  echo "inspect  -> $rel/report/inspect_compiled.json"
  ( cd "$REPO_ROOT" && "$SCRIPT_DIR/run_vitis.sh" python tools/vitis/inspect_xmodel.py \
      "$rel/compiled/pose_$act.xmodel" \
      --out "$rel/report/inspect_compiled.json" --input-hw "$height,$width" ) \
      > "$dir/logs/inspect.log" 2>&1

  guard_inspect "$dir/report/inspect_compiled.json" "$act" "$size"
  write_summary "$dir/report/inspect_compiled.json" \
    "$dir/compiled/pose_$act.xmodel" "$act" "$size" "$summary"
}

# -- main -----------------------------------------------------------------

mkdir -p "$OUT_ROOT"

for size in "${sizes[@]}"; do
  make_calibration "$size"
done

for act in "${activations[@]}"; do
  cfg="$REPO_ROOT/ml/configs/yolov8n-pose-$act.yaml"
  [ -f "$cfg" ] || { echo "variants.sh: no model config $cfg" >&2; exit 1; }
  for size in "${sizes[@]}"; do
    build_variant "$act" "$size" "$OUT_ROOT/calib_$size"
  done
done

python3 - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/report/summary.json")):
    summary = json.load(open(path, encoding="utf-8"))
    families = summary["dpu_op_families"]
    rows.append(
        [
            summary["activation"],
            f"{summary['size_hw'][0]}x{summary['size_hw'][1]}",
            str(families["conv2d"]),
            str(families["eltwise"]),
            str(families["hard_sigmoid"]),
            str(families["leakyrelu"]),
            str(summary["dpu_op_count"]),
            str(summary["xmodel_size_bytes"]),
            summary["fingerprint"],
        ]
    )
header = [
    "activation",
    "input",
    "conv",
    "eltwise",
    "hard_sigmoid",
    "leakyrelu",
    "dpu_ops",
    "xmodel_bytes",
    "fingerprint",
]
lines = ["\t".join(header)] + ["\t".join(row) for row in rows]
(root / "matrix.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n" + (root / "matrix.tsv").read_text(encoding="utf-8"), end="")
PY
