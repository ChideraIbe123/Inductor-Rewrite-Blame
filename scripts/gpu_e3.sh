#!/usr/bin/env bash
# E3 on a pod (bf16, self-terminating): (1) end-to-end evaluation of this GPU's switch guards on the held-out corpus
# with the per-model oracle and max-autotune; (2) freezing_discard_parameters on/off (base = freezing ON) over the 12
# pretrained twins and their random-init versions, rows carry peak CUDA memory and compile time.
# Usage: RUNPOD_API_KEY=.. RUNPOD_POD_ID=.. scripts/gpu_e3.sh <bundle.pkl>
cd /workspace/rewrite-blame
self_terminate() {
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== $(date) self-terminating pod $RUNPOD_POD_ID in ${SELF_TERMINATE_DELAY:-900}s"; sleep ${SELF_TERMINATE_DELAY:-900}
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
  fi
}
trap self_terminate EXIT
BUNDLE=${1:-results/guards_switches.pkl}
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_DTYPE=bf16 REWRITE_BLAME_THREADS=4 REWRITE_BLAME_STAT=min TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python; F='^DEBUG\|^INFO\|Graph Metrics'; D=$(date +%F)
mkdir -p logs results
echo "=== $(date) discover (bf16)"; $PY -m rewrite_blame discover 2>&1 | grep -c "" >/dev/null
FAMS=$($PY - "$BUNDLE" <<'PY'
import pickle, sys; b = pickle.load(open(sys.argv[1], "rb")); print(" ".join(sorted(b.guards)))
PY
)
HELD_OUT="resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny bert_base_eagerattn distilbert gpt2_prefill gpt2_decode smollm_135m_decode qwen2_0_5b_decode"
echo "=== $(date) guards-eval families: $FAMS"
$PY -m rewrite_blame --quiet --session "$D-gpu-e3" guards-eval --bundle "$BUNDLE" --families $FAMS --models $HELD_OUT --max-autotune --out results/guards_eval_e2e_switches.json 2>&1 | grep -v "$F" | tail -30
PRE="resnet18_pretrained resnet50_pretrained mobilenet_v3_small_pretrained efficientnet_b0_pretrained convnext_tiny_pretrained bert_base_eagerattn_pretrained distilbert_pretrained gpt2_prefill_pretrained gpt2_decode_pretrained smollm_135m_decode_pretrained smollm_135m_prefill_pretrained qwen2_0_5b_decode_pretrained"
RND="resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny bert_base_eagerattn distilbert gpt2_prefill gpt2_decode smollm_135m_decode smollm_135m_prefill qwen2_0_5b_decode"
echo "=== $(date) dataset --switch lowering/freezing_discard_parameters (base: freezing ON)"
$PY -m rewrite_blame --quiet --session "$D-gpu-e3" dataset --switch lowering/freezing_discard_parameters --base-state "default+lowering/freezing" \
  --models "$(echo $PRE $RND | tr ' ' ',')" --noise-config resnet18 --out results/dataset_switch_freezing_discard_parameters.jsonl 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -30
echo "=== $(date) gpu e3 done"
