#!/usr/bin/env bash
# Second Mac batch: waits for mac_chain.sh to finish, then runs the attribution case studies.
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
while ! grep -q "mac chain done" logs/mac_chain.log 2>/dev/null; do sleep 30; done
echo "=== $(date) attribute decode_mlp: default vs all optimus passes on"
$PY -m rewrite_blame --quiet attribute --model decode_mlp --fast default --slow "default~optimus" \
   --noise-file results/noise_decode_mlp.json --out results/attribution_decode_mlp.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -30
echo "=== $(date) attribute norm_mlp: default vs scheduler+joint knobs off"
$PY -m rewrite_blame --quiet attribute --model norm_mlp --fast default --slow "default-sched/inplace_buffers-joint/constant_folding-sched/epilogue_fusion-sched/prologue_fusion-sched/split_reductions-sched/loop_ordering_after_fusion-cpp/tiling_heuristics-cpp/loop_tail_vec" \
   --noise-file results/noise_norm_mlp.json --out results/attribution_norm_mlp.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -30
echo "=== $(date) mac chain2 done"
