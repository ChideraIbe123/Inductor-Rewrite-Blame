#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
true
echo "=== $(date) tau-scan attention_block (with slow-verdict confirmation)"
$PY -m rewrite_blame --quiet tau-scan --model attention_block --fast default --slow "default-*_sfdp_pattern_*_inference" \
  --out results/tau_scan_attention_block.json 2>&1 | grep -v "$F" | tail -8
echo "=== $(date) attribute attention_block (today's session)"
$PY -m rewrite_blame --quiet attribute --model attention_block --fast default --slow "default-*_sfdp_pattern_*_inference" \
  --out results/attribution_attention_block.json 2>&1 | grep -v "$F" | sed -n 1,16p
echo "=== $(date) repeat attention_block / norm_mlp / decode_mlp / resnet18"
for M in attention_block norm_mlp decode_mlp resnet18 bert_base_eagerattn gpt2_prefill; do
  $PY -m rewrite_blame --quiet repeat --model $M --out results/repeat_$M.json 2>&1 | grep -v "$F" | tail -3
done
echo "=== $(date) item5b done"
