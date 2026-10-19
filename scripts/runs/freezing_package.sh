#!/usr/bin/env bash
# The freezing package on one machine (Mac or VM): (1) freezing on/off on the pretrained corpus models AND their
# random-init twins (same shapes) -> does the finding depend on the weights?; (2) layout on/off likewise;
# (3) two real end-to-end workloads. Rows carry compile time and peak memory. Usage: scripts/runs/freezing_package.sh [threads]
set -uo pipefail
cd "$(dirname "$0")/../.."
export REWRITE_BLAME_THREADS=${1:-8} REWRITE_BLAME_STAT=min TOKENIZERS_PARALLELISM=false
PY=".venv/bin/python"; F='^DEBUG\|^INFO\|Graph Metrics'; D=$(date +%F)
PRE="resnet18_pretrained resnet50_pretrained mobilenet_v3_small_pretrained efficientnet_b0_pretrained convnext_tiny_pretrained bert_base_eagerattn_pretrained distilbert_pretrained gpt2_prefill_pretrained gpt2_decode_pretrained smollm_135m_decode_pretrained smollm_135m_prefill_pretrained qwen2_0_5b_decode_pretrained"
RND="resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny bert_base_eagerattn distilbert gpt2_prefill gpt2_decode smollm_135m_decode smollm_135m_prefill qwen2_0_5b_decode"
MODELS=$(echo $PRE $RND | tr ' ' ',')
PIN=""; command -v taskset >/dev/null && PIN="taskset -c 1-3"
for SW in lowering/freezing lowering/layout_optimization; do
  slug=$(echo "$SW" | tr '/' '_')
  echo "=== $(date) dataset --switch $SW on pretrained + random-init corpus"
  $PIN $PY -m rewrite_blame --quiet --session "$D-freeze" --cooldown 0.5 dataset --switch "$SW" --models "$MODELS" --noise-config resnet18 \
    --out "results/dataset_switch_${slug}_pretrained.jsonl" 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -40
done
echo "=== $(date) real workloads (generate, classify)"
$PIN $PY scripts/real_workload_freezing.py --runs 5 --out results/real_workload_freezing.json 2>&1 | grep -v "$F" | tail -12
echo "=== $(date) freezing package done"
