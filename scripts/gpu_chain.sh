#!/usr/bin/env bash
# GPU campaign (runs ON the pod, in tmux, resumable through the store):
#   phase 1  noise -> verify -> sweep on the corpus (session <date>-gpu)
#   phase 2  attribute-all on every model whose sweep flagged a confirmed slowdown
#   phase 3  guard dataset for the conv and attention families
#   phase 4  second session of noise+sweep for the repeatability check
# Usage: scripts/gpu_chain.sh [phase...]   (default: all)
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
S=$(date +%F)-gpu
MODELS="attention_block norm_mlp decode_mlp split_cat_mlp conv_bn_relu_stack cat_slice_cat resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny vit_b_16 bert_base_eagerattn distilbert gpt2_prefill gpt2_prefill_eagerattn gpt2_decode t5_small_encoder smollm_135m_prefill smollm_135m_prefill_eagerattn smollm_135m_decode qwen2_0_5b_prefill qwen2_0_5b_decode"
mkdir -p logs results
phases=${*:-1 2 3 4}
for ph in $phases; do
case $ph in
1)
  for M in $MODELS; do
    echo "=== $(date) noise $M"
    $PY -m rewrite_blame --quiet --session "$S" noise  --model "$M" --runs 7 --out "results/noise_$M.json" 2>&1 | grep -v "$F" | tail -2
    echo "=== $(date) verify $M"
    $PY -m rewrite_blame --quiet --session "$S" verify --model "$M" --out "results/verify_$M.json" 2>&1 | grep -v "$F" | tail -2
    echo "=== $(date) sweep $M"
    $PY -m rewrite_blame --quiet --session "$S" sweep --model "$M" --noise-file "results/noise_$M.json" --out "results/sweep_$M.json" 2>&1 | grep -v "$F" | tail -3
  done ;;
2)
  for M in $MODELS; do
    [ -f "results/sweep_$M.json" ] || continue
    if $PY - "results/sweep_$M.json" <<'PY'
import json, sys; d = json.load(open(sys.argv[1])); rows = d.get("rows", d if isinstance(d, list) else [])
sys.exit(0 if any(r.get("verdict") == "slower" for r in rows) else 1)
PY
    then
      echo "=== $(date) attribute-all $M"
      $PY -m rewrite_blame --quiet --session "$S" attribute-all --model "$M" --out "results/attribute_all_$M.json" 2>&1 | grep -v "$F" | tail -6
    fi
  done ;;
3)
  for FAM in conv attn; do
    echo "=== $(date) dataset $FAM"
    $PY -m rewrite_blame --quiet --session "$S-ds-$FAM" dataset --family $FAM --level full --shuffle 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -500
  done ;;
4)
  S2=$(date +%F)-gpu2
  for M in $MODELS; do
    echo "=== $(date) repeat noise+sweep $M"
    $PY -m rewrite_blame --quiet --session "$S2" noise --model "$M" --runs 7 --out "results/noise2_$M.json" 2>&1 | grep -v "$F" | tail -1
    $PY -m rewrite_blame --quiet --session "$S2" sweep --model "$M" --noise-file "results/noise2_$M.json" --out "results/sweep2_$M.json" 2>&1 | grep -v "$F" | tail -2
  done ;;
esac
done
echo "=== $(date) gpu chain done"
