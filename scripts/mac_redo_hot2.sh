#!/usr/bin/env bash
# Re-measure the models whose Mac session overlapped the throttling episode, then run the new case studies.
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q "redo hot done" logs/mac_redo_hot.log 2>/dev/null; do sleep 120; done
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
S=$(date +%F)-cool2
for M in gpt2_prefill_eagerattn mobilenet_v3_small efficientnet_b0; do
  echo "=== $(date) cool session $S: $M"
  $PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 noise --model $M --runs 7 --out results/noise_$M.json 2>&1 | grep -v "$F" | tail -2
  $PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 sweep --model $M --noise-file results/noise_$M.json --out results/sweep_$M.json 2>&1 | grep -v "$F" | tail -2
done
echo "=== $(date) case: smollm_135m_decode default vs all opt-in passes (iterative)"
$PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 attribute-all --model smollm_135m_decode --fast default --slow "default~optimus" --out results/attribution_all_smollm_135m_decode.json 2>&1 | grep -v "$F" | tail -12
echo "=== $(date) case: smollm_135m_decode tau-scan"
$PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 tau-scan --model smollm_135m_decode --fast default --slow "default~optimus" --out results/tau_scan_smollm_135m_decode.json 2>&1 | grep -v "$F" | tail -6
echo "=== $(date) case: efficientnet_b0 layout optimisation hurts (fast = default without it and three no-op flags)"
$PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 attribute --model efficientnet_b0 --fast "default-lowering/layout_optimization-sched/reorder_for_peak_memory-cpp/tiling_heuristics-joint/constant_folding" --slow default --out results/attribution_efficientnet_b0.json 2>&1 | grep -v "$F" | tail -12
echo "=== $(date) case: mobilenet_v3_small same question"
$PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 attribute --model mobilenet_v3_small --fast "default-lowering/layout_optimization-sched/reorder_for_peak_memory-cpp/tiling_heuristics-joint/constant_folding" --slow default --out results/attribution_mobilenet_v3_small.json 2>&1 | grep -v "$F" | tail -12
echo "=== $(date) case: resnet50 default vs freezing + conv_1x1_as_mm + two no-ops (iterative; one hurts, one helps)"
$PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 attribute-all --model resnet50 --fast default --slow "default+lowering/freezing+lowering/conv_1x1_as_mm+optimus/normalization_pass+cpp/decompose_tanh" --out results/attribution_all_resnet50.json 2>&1 | grep -v "$F" | tail -12
echo "=== $(date) case: distilbert default vs max_fusion_size/inplace_buffers/normalization off+on (iterative, several causes)"
$PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 attribute-all --model distilbert --fast default --slow "default-sched/max_fusion_size-sched/inplace_buffers+optimus/normalization_pass-sched/reorder_for_peak_memory-cpp/tiling_heuristics" --out results/attribution_all_distilbert.json 2>&1 | grep -v "$F" | tail -12
echo "=== $(date) redo hot2 done"
