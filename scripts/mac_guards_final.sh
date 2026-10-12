#!/usr/bin/env bash
# After the dataset collection: encoder fine-tune evaluation per family, final bundle, end-to-end guards on real models.
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q "dataset done" logs/mac_dataset.log 2>/dev/null; do sleep 120; done
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
echo "=== $(date) guards-train: trees + api on all families (final bundle)"
$PY -m rewrite_blame guards-train --kinds tree gbt logreg api-context --out results/guards_eval.json --bundle results/guards.pkl 2>&1 | grep -v "$F" | tail -30
echo "=== $(date) guards-train: encoder (Laya) per family"
$PY -m rewrite_blame guards-train --kinds encoder --epochs 4 --out results/guards_eval_encoder.json --bundle results/guards_encoder_unused.pkl 2>&1 | grep -v "$F" | tail -20
echo "=== $(date) guards-eval end-to-end on real models (cool session)"
$PY -m rewrite_blame --quiet --session "$(date +%F)-guards" --cooldown 1.0 guards-eval --bundle results/guards.pkl --families mlp conv attn \
  --models smollm_135m_decode qwen2_0_5b_decode gpt2_decode decode_mlp mobilenet_v3_small efficientnet_b0 resnet18 resnet50 attention_block bert_base_eagerattn gpt2_prefill_eagerattn smollm_135m_prefill_eagerattn \
  --out results/guards_eval_e2e.json 2>&1 | grep -v "$F" | tail -40
echo "=== $(date) guards final done"
