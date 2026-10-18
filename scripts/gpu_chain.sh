#!/usr/bin/env bash
# GPU campaign (runs ON the pod, in tmux, resumable through the store):
#   phase 1  noise -> verify -> sweep on the corpus (session <date>-gpu)
#   phase 2  attribute-all on every model whose sweep flagged a confirmed slowdown
#   phase 3  guard dataset for the conv and attention families
#   phase 4  second session of noise+sweep for the repeatability check
# Usage: scripts/gpu_chain.sh [phase...]   (default: all)
set -uo pipefail
cd "$(dirname "$0")/.."

self_terminate() {   # called at the end: stop billing even if nobody is watching (key passed via env, never written)
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== $(date) self-terminating pod $RUNPOD_POD_ID"
    sleep ${SELF_TERMINATE_DELAY:-600}      # leave 10 minutes for a final pull
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
  fi
}
trap self_terminate EXIT
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
S=${GPU_SESSION:-$(date +%F)-gpu}
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
  # the same fast/slow pairs as the CPU case studies, so the three machines can be compared
  echo "=== $(date) attribute-all cat_slice_cat: default vs all opt-in passes on"
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model cat_slice_cat --fast default --slow "default~optimus" --out results/attribution_all_cat_slice_cat.json 2>&1 | grep -v "$F" | tail -12
  echo "=== $(date) attribute-all resnet18: default vs six flags off"
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model resnet18 --fast default \
    --slow "default-lowering/layout_optimization-sched/inplace_buffers-master/post_grad_passes-joint/joint_graph.early_patterns/pointless_view-sched/reorder_for_peak_memory-sched/max_fusion_size" \
    --out results/attribution_all_resnet18.json 2>&1 | grep -v "$F" | tail -12
  echo "=== $(date) attribute-all decode_mlp: default vs opt-in passes incl. decompose_mm_wide"
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model decode_mlp --fast default --slow "default~optimus" --out results/attribution_all_decode_mlp.json 2>&1 | grep -v "$F" | tail -12
  echo "=== $(date) attribute-all smollm_135m_decode: default vs opt-in passes"
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model smollm_135m_decode --fast default --slow "default~optimus" --out results/attribution_all_smollm_135m_decode.json 2>&1 | grep -v "$F" | tail -12
  echo "=== $(date) attribute bert_base_eagerattn: fused attention (default) vs all GPU-side attention rules off"
  $PY -m rewrite_blame --quiet --session "$S" attribute --model bert_base_eagerattn --fast default --slow "default-*_sfdp_pattern_*" --out results/attribution_bert_base_eagerattn.json 2>&1 | grep -v "$F" | tail -12
  echo "=== $(date) attribute mobilenet_v3_small: default vs layout optimisation off"
  $PY -m rewrite_blame --quiet --session "$S" attribute --model mobilenet_v3_small --fast default --slow "default-lowering/layout_optimization-sched/inplace_buffers-sched/max_fusion_size" --out results/attribution_mobilenet_v3_small.json 2>&1 | grep -v "$F" | tail -12
  echo "=== $(date) interactions attention_block (pairs over graph-changing switches)"
  $PY -m rewrite_blame --quiet --session "$S" interactions --model attention_block --out results/interactions_attention_block.json 2>&1 | grep -v "$F" | tail -12
  ;;
3)
  for FAM in conv attn; do
    echo "=== $(date) dataset $FAM"
    $PY -m rewrite_blame --quiet --session "$S-ds-$FAM" dataset --family $FAM --level full --shuffle 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -500
  done ;;
4)
  S2=${GPU_SESSION2:-$(date +%F)-gpu2}
  for M in $MODELS; do
    echo "=== $(date) repeat noise+sweep $M"
    $PY -m rewrite_blame --quiet --session "$S2" noise --model "$M" --runs 7 --out "results/noise2_$M.json" 2>&1 | grep -v "$F" | tail -1
    $PY -m rewrite_blame --quiet --session "$S2" sweep --model "$M" --noise-file "results/noise2_$M.json" --out "results/sweep2_$M.json" 2>&1 | grep -v "$F" | tail -2
  done ;;
esac
done
echo "=== $(date) gpu chain done"
