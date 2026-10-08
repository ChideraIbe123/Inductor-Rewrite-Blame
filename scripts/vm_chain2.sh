#!/usr/bin/env bash
# VM: once the norm_mlp sweep is written, pause chain1, run the attention attribution alone
# (timing needs the cores to itself), then restart chain1 for the remaining models (cached steps are skipped).
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_THREADS=4
PY=.venv/bin/python
while [ ! -f results/sweep_norm_mlp.json ]; do sleep 60; done
tmux kill-session -t chain1 2>/dev/null; pkill -f "[p]ython -m rewrite_blame"; sleep 5
echo "=== $(date) attribute attention_block on VM: no-fusion (fast) vs default (slow)"
$PY -m rewrite_blame --quiet attribute --model attention_block --fast "default-*_sfdp_pattern_*_inference" --slow default \
   --out results/attribution_attention_block.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -30
echo "=== $(date) restarting chain1"
tmux new-session -d -s chain1 "scripts/sweep_chain.sh --threads 4 decode_mlp split_cat_mlp cat_slice_cat conv_bn_relu_stack resnet18 bert_base_eagerattn gpt2_prefill smollm_135m_prefill_eagerattn resnet50 >> logs/chain1.log 2>&1"
echo "=== $(date) vm chain2 done"
