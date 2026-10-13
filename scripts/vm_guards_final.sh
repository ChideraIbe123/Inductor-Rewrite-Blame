#!/usr/bin/env bash
# VM: after its dataset finishes, train guards on BOTH machines' data (best kind per family) and run the
# end-to-end evaluation on the VM's real models. Mac datasets are expected at results/mac_dataset_<family>.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q "dataset done" logs/vm_dataset_resume.log 2>/dev/null; do sleep 300; done
export REWRITE_BLAME_THREADS=4
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
echo "=== $(date) guards-train on combined data"
$PY -m rewrite_blame guards-train --kinds tree gbt logreg --data results/dataset_mlp.jsonl --data results/dataset_conv.jsonl --data results/dataset_attn.jsonl \
  --data results/mac_dataset_mlp.jsonl --data results/mac_dataset_conv.jsonl --data results/mac_dataset_attn.jsonl \
  --out results/guards_eval_combined.json --bundle results/guards_combined.pkl 2>&1 | tail -25
echo "=== $(date) guards-eval end-to-end on VM real models"
$PY -m rewrite_blame --quiet --session "$(date +%F)-guards" guards-eval --bundle results/guards_combined.pkl --families mlp conv attn \
  --models attention_block bert_base_eagerattn smollm_135m_prefill_eagerattn gpt2_prefill_eagerattn resnet18 resnet50 mobilenet_v3_small efficientnet_b0 decode_mlp smollm_135m_decode gpt2_decode \
  --out results/guards_eval_e2e.json 2>&1 | grep -v "$F" | tail -30
echo "=== $(date) guards-eval matmul guard with the pass enabled"
$PY -m rewrite_blame --quiet --session "$(date +%F)-guards" guards-eval --bundle results/guards_combined.pkl --families mlp --base-state "default+optimus/decompose_mm_wide" \
  --models decode_mlp smollm_135m_decode gpt2_decode qwen2_0_5b_decode --out results/guards_eval_e2e_mlp.json 2>&1 | grep -v "$F" | tail -14
echo "=== $(date) vm guards final done"
