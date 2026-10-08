#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
echo "=== $(date) attribute-all norm_mlp"
$PY -m rewrite_blame --quiet attribute-all --model norm_mlp --fast default \
  --slow "default-sched/inplace_buffers-joint/constant_folding-sched/epilogue_fusion-sched/prologue_fusion-sched/split_reductions-sched/loop_ordering_after_fusion-cpp/tiling_heuristics-cpp/loop_tail_vec" \
  --out results/attribution_all_norm_mlp.json 2>&1 | grep -v "$F" | tail -20
echo "=== $(date) tau-scan attention_block"
$PY -m rewrite_blame --quiet tau-scan --model attention_block --fast default --slow "default-*_sfdp_pattern_*_inference" \
  --out results/tau_scan_attention_block.json 2>&1 | grep -v "$F" | tail -8
echo "=== $(date) tau-scan norm_mlp"
$PY -m rewrite_blame --quiet tau-scan --model norm_mlp --fast default \
  --slow "default-sched/inplace_buffers-joint/constant_folding-sched/epilogue_fusion-sched/prologue_fusion-sched/split_reductions-sched/loop_ordering_after_fusion-cpp/tiling_heuristics-cpp/loop_tail_vec" \
  --out results/tau_scan_norm_mlp.json 2>&1 | grep -v "$F" | tail -8
echo "=== $(date) interactions decode_mlp"
$PY -m rewrite_blame --quiet interactions --model decode_mlp --out results/interactions_decode_mlp.json 2>&1 | grep -v "$F" | tail -25
echo "=== $(date) interactions norm_mlp"
$PY -m rewrite_blame --quiet interactions --model norm_mlp --out results/interactions_norm_mlp.json 2>&1 | grep -v "$F" | tail -25
echo "=== $(date) interactions attention_block (all opt-in families on top of default)"
$PY -m rewrite_blame --quiet interactions --model attention_block --out results/interactions_attention_block.json 2>&1 | grep -v "$F" | tail -25
echo "=== $(date) item5 done"
