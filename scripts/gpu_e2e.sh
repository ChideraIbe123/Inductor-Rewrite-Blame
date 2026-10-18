#!/usr/bin/env bash
# On the pod: end-to-end effect of the conv (channels-last) guard on the GPU image models, stock base.
# Usage: scripts/gpu_e2e.sh [bundle.pkl] [out.json]
cd /workspace/rewrite-blame

self_terminate() {   # called at the end: stop billing even if nobody is watching (key passed via env, never written)
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== $(date) self-terminating pod $RUNPOD_POD_ID"
    sleep ${SELF_TERMINATE_DELAY:-600}      # leave 10 minutes for a final pull
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
  fi
}
trap self_terminate EXIT
BUNDLE=${1:-results/guards_3machines.pkl}
OUT=${2:-results/guards_eval_e2e.json}
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
echo "=== $(date) guards-eval conv on GPU image models with $BUNDLE"
.venv/bin/python -m rewrite_blame --quiet --session "$(date +%F)-gpu-e2e" guards-eval --bundle "$BUNDLE" --families conv \
  --models resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny conv_bn_relu_stack vit_b_16 --out "$OUT" 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -16
echo "=== $(date) gpu e2e done"
