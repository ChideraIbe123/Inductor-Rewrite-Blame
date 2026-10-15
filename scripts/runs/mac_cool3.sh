#!/usr/bin/env bash
# Cool-session re-measurement of the models whose first Mac session ran on a warm machine.
set -uo pipefail
cd "$(dirname "$0")/../.."
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
S=$(date +%F)-cool3
for M in resnet18 distilbert gpt2_decode smollm_135m_decode t5_small_encoder bert_base bert_base_eagerattn gpt2_prefill; do
  echo "=== $(date) cool session $S: $M"
  $PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 noise --model $M --runs 7 --out results/noise_$M.json 2>&1 | grep -v "$F" | tail -2
  $PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 sweep --model $M --noise-file results/noise_$M.json --out results/sweep_$M.json 2>&1 | grep -v "$F" | tail -2
done
echo "=== $(date) case: distilbert iterative (cool)"
$PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 attribute-all --model distilbert --fast default --slow "default-sched/max_fusion_size-sched/inplace_buffers+optimus/normalization_pass-sched/reorder_for_peak_memory-cpp/tiling_heuristics" --out results/attribution_all_distilbert.json 2>&1 | grep -v "$F" | tail -10
echo "=== $(date) case: bert_base_eagerattn attention rule (cool)"
$PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 attribute --model bert_base_eagerattn --fast default --slow "default-*_sfdp_pattern_*_inference" --out results/attribution_bert_base_eagerattn.json 2>&1 | grep -v "$F" | tail -10
echo "=== $(date) cool3 done"
