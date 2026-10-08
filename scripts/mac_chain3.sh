#!/usr/bin/env bash
# Third Mac batch: waits for mac_chain2.sh, then runs the resnet18 case study ("several flags changed after an upgrade").
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
while ! grep -q "mac chain2 done" logs/mac_chain2.log 2>/dev/null; do sleep 30; done
echo "=== $(date) attribute resnet18: default vs four changed flags"
$PY -m rewrite_blame --quiet attribute --model resnet18 --fast default \
   --slow "default-lowering/layout_optimization-sched/inplace_buffers-master/post_grad_passes-joint/joint_graph.early_patterns/pointless_view-sched/reorder_for_peak_memory-cpp/tiling_heuristics" \
   --noise-file results/noise_resnet18.json --out results/attribution_resnet18.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -30
echo "=== $(date) mac chain3 done"
