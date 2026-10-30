#!/bin/bash
# Unattended GreenFuse queue (Bhargav, Oct 30 2026). Runs on an allocated GPU node, survives the editor
# session ending (launch with setsid nohup). Steps run strictly one at a time so energy is not disturbed.
#   1. wait for the running calibration batch (exp7-exp11) to finish
#   2. redo any of exp8-exp11 without a final decision (interrupted or invalid), and exp7 (CPU contention)
#   3. runtime screens: fuse_epilogue (W2 decode shapes, W6) and W6 default
#   4. calibrations exp12 (W2 fuse_epilogue, 3 decode shapes), exp13 (W6 64x4096), exp14 (W6 512x4096)
#   5. export readable folders, threshold sensitivity (1.05/1.10/1.20), summary table, tests, local commit
set -u
cd /home/bsrinid/ondemand/efficientfuse
source .venv/bin/activate
export C_INCLUDE_PATH=/sw/pkgs/arc/python/3.12.1/include/python3.12
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH
mkdir -p results/bhargav/logs
LOG=results/bhargav/logs/overnight.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

declare -A LABEL=(
  [exp7_w1_launch_bound]="W1 relu(x+bias)+residual"
  [exp8_w3_layernorm_small]="W3 add+LayerNorm+GELU"
  [exp9_w4_fan_out_launch_bound]="W4 fan-out (h - mean(h))"
  [exp10_w5_softmax_small]="W5 softmax(gelu(x+bias))"
  [exp11_w2_gemm_bound]="W2 linear+GELU (cuBLAS)"
  [exp12_w2_fuse_epilogue_decode]="W2 linear+GELU (fuse_epilogue)"
  [exp13_w6_transformer_mlp_decode]="W6 add+LN+linear+GELU (fuse_epilogue)"
  [exp14_w6_transformer_mlp_prefill]="W6 add+LN+linear+GELU (fuse_epilogue)"
)

final_statuses() {  # space-separated statuses of every decision.json for one experiment ("" if none)
  python - "$1" <<'EOF'
import json, sys, glob
paths = glob.glob(f"results/bhargav/runs/{sys.argv[1]}/*/*/*/decision.json")
print(" ".join(json.load(open(p)).get("status", "pending") for p in paths))
EOF
}

calibrate() {  # $1 = experiment name; keeps any previous attempt beside the new one
  local exp=$1
  mkdir -p results/bhargav/$exp/logs
  if [ -d results/bhargav/runs/$exp ]; then
    local n=1; while [ -d results/bhargav/runs/${exp}_attempt$n ]; do n=$((n+1)); done
    mv results/bhargav/runs/$exp results/bhargav/runs/${exp}_attempt$n
    [ -f results/bhargav/$exp/logs/calibrate.log ] && mv results/bhargav/$exp/logs/calibrate.log results/bhargav/$exp/logs/calibrate_attempt$n.log
  fi
  say "START $exp"
  greenfuse calibrate --config configs/$exp.yaml --exclusive-gpu \
    --output-dir results/bhargav/runs/$exp --cache-dir results/decisions \
    > results/bhargav/$exp/logs/calibrate.log 2>&1
  say "END $exp exit=$? statuses=[$(final_statuses $exp)] $(grep -E '"reason"' results/bhargav/$exp/logs/calibrate.log | tail -1 | tr -s ' ')"
}

needs_run() {  # true unless every shape has a final validated/inconclusive decision
  local s; s=$(final_statuses "$1")
  [ -z "$s" ] && return 0
  for status in $s; do case $status in validated|inconclusive) ;; *) return 0 ;; esac; done
  return 1
}

say "Overnight queue started on $(hostname)"
while pgrep -u "$USER" -f "greenfuse calibrate" >/dev/null || pgrep -u "$USER" -f run_batch2.sh >/dev/null; do sleep 30; done
say "Earlier batch finished"

for exp in exp8_w3_layernorm_small exp9_w4_fan_out_launch_bound exp10_w5_softmax_small exp11_w2_gemm_bound; do
  if needs_run $exp; then calibrate $exp; else say "SKIP $exp statuses=[$(final_statuses $exp)]"; fi
done

say "SCREEN fuse_epilogue"
python scripts/screen_candidates.py --gemm-backend fuse_epilogue --tol 1e-3 \
  --output results/bhargav/screening/screen_fuse_epilogue.json \
  --cases w2:16x4096 w2:64x4096 w2:128x4096 w2:512x4096 w6:64x4096 w6:512x4096 2>&1 \
  | grep -v -i warn | grep -E "PASS|----|ERROR|Saved" | tee -a "$LOG"
say "SCREEN w6 default"
python scripts/screen_candidates.py --gemm-backend default --tol 1e-3 \
  --output results/bhargav/screening/screen_w6_default.json \
  --cases w6:64x4096 w6:512x4096 2>&1 | grep -v -i warn | grep -E "PASS|----|ERROR|Saved" | tee -a "$LOG"

calibrate exp7_w1_launch_bound            # first attempt's validation pass was disturbed by CPU contention
calibrate exp12_w2_fuse_epilogue_decode
calibrate exp13_w6_transformer_mlp_decode
calibrate exp14_w6_transformer_mlp_prefill

say "EXPORT"
for exp in exp7_w1_launch_bound exp8_w3_layernorm_small exp9_w4_fan_out_launch_bound exp10_w5_softmax_small \
           exp11_w2_gemm_bound exp12_w2_fuse_epilogue_decode exp13_w6_transformer_mlp_decode exp14_w6_transformer_mlp_prefill; do
  for decision in results/bhargav/runs/$exp/*/*/*/decision.json; do
    [ -f "$decision" ] || continue
    run=$(dirname "$decision"); shape=$(basename "$(dirname "$run")"); shape=${shape#*_}
    [ -f "$run/initial.json" ] || { say "no initial.json for $exp $shape"; continue; }
    python scripts/export_experiment.py "$run" results/bhargav/$exp/$shape --shape "$shape" --label "${LABEL[$exp]}" 2>&1 \
      | grep -v -i warn | tee -a "$LOG"
  done
done
python scripts/compare_workloads.py 2>&1 | tail -1 | tee -a "$LOG"
python scripts/threshold_sensitivity.py 2>&1 | tail -1 | tee -a "$LOG"

python - <<'EOF' 2>&1 | tee -a "$LOG"
# One readable table per run: fresh validation pass, every candidate, runtime and energy.
import csv, glob
from pathlib import Path
lines = ["# Overnight results (fresh validation pass; runtime us / energy mJ per call)", ""]
for path in sorted(glob.glob("results/bhargav/exp*/*/summary.csv"), key=lambda p: (int(Path(p).parts[2].split("_")[0][3:]), p)):
    rows = [r for r in csv.DictReader(open(path)) if r["pass"] == "validation"] or \
           [r for r in csv.DictReader(open(path)) if r["pass"] == "initial"]
    exp, shape = Path(path).parts[2], Path(path).parts[3]
    lines += [f"## {exp} {shape} ({rows[0]['pass'] if rows else 'no data'} pass)", "",
              "| candidate | runtime us | energy mJ | selected |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: float(r["energy_mJ_median"] or 1e18)):
        lines.append(f"| {r['candidate']} | {r['runtime_us_median']} | {r['energy_mJ_median']} | {'yes' if r['selected'] == 'True' else ''} |")
    lines.append("")
Path("results/bhargav/overnight_summary.md").write_text("\n".join(lines) + "\n")
print("Wrote results/bhargav/overnight_summary.md")
EOF

say "TESTS"
python -m unittest discover -s tests 2>&1 | tail -1 | tee -a "$LOG"

say "COMMIT (local only; push needs GitHub login)"
git add -A src scripts tests configs
git add -f results/bhargav ':!results/bhargav/runs'
git commit -q -m "Overnight runs: small-shape controls, fuse_epilogue W2 decode shapes, W6 five-op workload

- replay_input_refresh=on_change (inputs resident on GPU), W5 softmax(gelu) and W6
  gelu(layer_norm(x + residual) @ W + b) workloads, fuse_epilogue GEMM policy, runtime screen,
  threshold sensitivity (1.05/1.10/1.20) over every run
- Results under results/bhargav: exp7-exp14, screening/, overnight_summary.md, logs/overnight.log

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && say "Committed $(git rev-parse --short HEAD)"
say "OVERNIGHT_DONE"
