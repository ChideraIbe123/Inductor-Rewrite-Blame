#!/usr/bin/env bash
# VM attribution case studies for proposal item 5 (run alone in tmux; everything is cached in the store).
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_THREADS=4
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
echo "=== $(date) attribute-all cat_slice_cat: default vs all 16 opt-in passes on"
$PY -m rewrite_blame --quiet attribute-all --model cat_slice_cat --fast default --slow "default~optimus" \
  --out results/attribution_all_cat_slice_cat.json 2>&1 | grep -v "$F" | tail -15
echo "=== $(date) attribute-all resnet18: default vs six flags off"
$PY -m rewrite_blame --quiet attribute-all --model resnet18 --fast default \
  --slow "default-lowering/layout_optimization-sched/inplace_buffers-master/post_grad_passes-joint/joint_graph.early_patterns/pointless_view-sched/reorder_for_peak_memory-cpp/tiling_heuristics" \
  --out results/attribution_all_resnet18.json 2>&1 | grep -v "$F" | tail -15
echo "=== $(date) attribute decode_mlp: freezing + post-grad flags"
$PY -m rewrite_blame --quiet attribute --model decode_mlp --fast default --slow "default+lowering/freezing+optimus/decompose_mm_pass+optimus/batch_linear+optimus/normalization_pass" \
  --out results/attribution_decode_mlp.json 2>&1 | grep -v "$F" | tail -15
echo "=== $(date) attribute bert_base_eagerattn: no attention fusion (fast) vs default (slow)"
$PY -m rewrite_blame --quiet attribute --model bert_base_eagerattn --fast "default-*_sfdp_pattern_*_inference" --slow default \
  --out results/attribution_bert_base_eagerattn.json 2>&1 | grep -v "$F" | tail -15
echo "=== $(date) interactions cat_slice_cat (graph-changing switches, pairs)"
$PY -m rewrite_blame --quiet interactions --model cat_slice_cat --out results/interactions_cat_slice_cat.json 2>&1 | grep -v "$F" | tail -20
echo "=== $(date) tau-scan cat_slice_cat"
$PY -m rewrite_blame --quiet tau-scan --model cat_slice_cat --fast default --slow "default~optimus" --out results/tau_scan_cat_slice_cat.json 2>&1 | grep -v "$F" | tail -8
echo "=== $(date) vm item5 done"
