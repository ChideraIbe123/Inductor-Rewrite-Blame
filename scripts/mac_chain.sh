#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
F='grep -v ^DEBUG\|^INFO\|Graph\ Metrics'
scripts/sweep_chain.sh --threads 8 attention_block norm_mlp decode_mlp
echo "=== $(date) attribute attention_block (timing)"
$PY -m rewrite_blame --quiet attribute --model attention_block --fast default --slow "default-*_sfdp_pattern_*_inference" \
   --noise-file results/noise_attention_block.json --out results/attribution_attention_block.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -40
scripts/sweep_chain.sh --threads 8 resnet18 bert_base_eagerattn gpt2_prefill
echo "=== $(date) mac chain done"
