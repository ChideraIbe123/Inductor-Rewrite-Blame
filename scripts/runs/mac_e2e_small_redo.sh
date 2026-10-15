#!/usr/bin/env bash
# Daytime, plugged-in redo: (1) small/fast models with the default base; (2) the matmul guard in the world where the
# decomposition pass is enabled (base = default+decompose_mm_wide), on the decode models.
set -uo pipefail
cd "$(dirname "$0")/../.."
# (timer removed: launched directly)
export REWRITE_BLAME_THREADS=8
S=$(date +%F)-guards2
echo "=== $(date) e2e redo small models (session $S); power: $(pmset -g batt | tail -1 | cut -c1-60)"
.venv/bin/python -m rewrite_blame --quiet --session "$S" --cooldown 1.0 guards-eval --bundle results/guards_combined.pkl --families conv attn \
  --models decode_mlp gpt2_decode smollm_135m_decode qwen2_0_5b_decode attention_block resnet18 --out results/guards_eval_e2e_small.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -14
echo "=== $(date) e2e matmul guard with the decomposition pass enabled"
.venv/bin/python -m rewrite_blame --quiet --session "$S" --cooldown 1.0 guards-eval --bundle results/guards_combined.pkl --families mlp \
  --base-state "default+optimus/decompose_mm_wide" --models decode_mlp smollm_135m_decode qwen2_0_5b_decode gpt2_decode smollm_135m_prefill --out results/guards_eval_e2e_mlp.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -14
echo "=== $(date) e2e redo done"
