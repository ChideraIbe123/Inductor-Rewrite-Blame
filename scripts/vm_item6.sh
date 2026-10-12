#!/usr/bin/env bash
# After the corpus3 sweeps finish: stop the queued pair scan, run the Qwen case studies alone, then restart the pair scan.
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q "chain done" logs/corpus3.log 2>/dev/null; do sleep 120; done
tmux kill-session -t corpus3 2>/dev/null; pkill -f "[p]ython -m rewrite_blame"; sleep 5
export REWRITE_BLAME_THREADS=4
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
S=$(date +%F)-cases
echo "=== $(date) case: qwen2_0_5b_prefill default vs six opt-in passes on (iterative)"
$PY -m rewrite_blame --quiet --session "$S" attribute-all --model qwen2_0_5b_prefill --fast default \
  --slow "default+optimus/normalization_pass+optimus/split_cat_pass+optimus/merge_splits_pass+optimus/batch_linear+optimus/decompose_mm_pass+cpp/decompose_tanh" \
  --out results/attribution_all_qwen2_0_5b_prefill.json 2>&1 | grep -v "$F" | tail -10
echo "=== $(date) case: qwen2_0_5b_prefill default-on rewrites that hurt (fast = default without reorder_for_locality and constant_folding + two no-ops; slow = default)"
$PY -m rewrite_blame --quiet --session "$S" attribute-all --model qwen2_0_5b_prefill \
  --fast "default-post_grad/reorder_for_locality-joint/constant_folding-sched/reorder_for_peak_memory-cpp/tiling_heuristics" --slow default \
  --out results/attribution_all_qwen2_0_5b_prefill_defaults.json 2>&1 | grep -v "$F" | tail -10
echo "=== $(date) case: resnet50 VM, freezing + conv_1x1_as_mm + layout (iterative)"
$PY -m rewrite_blame --quiet --session "$S" attribute-all --model resnet50 --fast "default-lowering/layout_optimization" --slow "default+lowering/conv_1x1_as_mm+optimus/normalization_pass" \
  --out results/attribution_all_resnet50.json 2>&1 | grep -v "$F" | tail -10
echo "=== $(date) restarting the cat_slice_cat pair scan"
tmux new-session -d -s pairs3 "REWRITE_BLAME_THREADS=4 .venv/bin/python -m rewrite_blame --quiet --session pairs-v3 interactions --model cat_slice_cat --out results/interactions_cat_slice_cat_v3.json > logs/pairs3.log 2>&1; echo \"=== all done\" >> logs/pairs3.log"
echo "=== $(date) vm item6 done"
