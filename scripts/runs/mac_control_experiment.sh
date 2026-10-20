#!/usr/bin/env bash
# Control experiment for the paper (E2): the same three attribution cases run with the calibrated judge and with
# the single-measurement judge (one run per verdict, no interleaved baseline, no identical-program guard), in ONE
# fresh session so both modes see the same first measurements and differ only in policy. Mac, plugged in, idle.
set -uo pipefail
cd "$(dirname "$0")/../.."
export REWRITE_BLAME_THREADS=8 REWRITE_BLAME_STAT=min TOKENIZERS_PARALLELISM=false
PY=".venv/bin/python"; F='^DEBUG\|^INFO\|Graph Metrics'; S="$(date +%F)-ctrl"
run_case() {  # name fast slow
  for MODE in calibrated single; do
    echo "=== $(date) attribute $1 [$MODE]"
    $PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 --judge-mode $MODE attribute --model "$1" --fast "$2" --slow "$3" \
      --out "results/control_${1}_${MODE}.json" 2>&1 | grep -v "$F" | grep -E "^## Verdict|^- |culprit|^\| [0-9]+ \|" | head -14
  done
}
run_case attention_block "default" "default-*_sfdp_pattern_*_inference"
run_case decode_mlp "default" "default~optimus"
run_case resnet18 "default" "default-lowering/layout_optimization-sched/inplace_buffers-master/post_grad_passes-joint/joint_graph.early_patterns/pointless_view-sched/reorder_for_peak_memory-cpp/tiling_heuristics"
echo "=== $(date) control experiment done"
