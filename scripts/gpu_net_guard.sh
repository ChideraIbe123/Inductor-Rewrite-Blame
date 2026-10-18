#!/usr/bin/env bash
# On the pod: the real-network layout dataset (torchvision zoo, batches 1/8/32), a guard trained on it, and the
# end-to-end check on the seven held-out corpus image models. Resumable (rows append; store caches measurements).
cd /workspace/rewrite-blame

self_terminate() {   # called at the end: stop billing even if nobody is watching (key passed via env, never written)
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== $(date) self-terminating pod $RUNPOD_POD_ID"
    sleep ${SELF_TERMINATE_DELAY:-600}      # leave 10 minutes for a final pull
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
  fi
}
trap self_terminate EXIT
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
F='^DEBUG\|^INFO\|Graph Metrics'; D=$(date +%F)
# session fixed so the on/off measurements already in the store are reused; tau from one stable batch-32 network
# (the median-of-3 default picked two batch-1 networks whose GPU noise bands were 36-43% and labelled everything neutral)
S=${NET_SESSION:-2026-10-15-gpu-ds-net}
[ -f results/dataset_net.jsonl ] && mv results/dataset_net.jsonl "results/dataset_net_tau36_$(date +%H%M).jsonl"
echo "=== $(date) dataset net (real networks)"
.venv/bin/python -m rewrite_blame --quiet --session "$S" dataset --family net --level full --shuffle --noise-config zoo_resnet34_b32 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -120
echo "=== $(date) guards-train net (GPU rows only)"
.venv/bin/python -m rewrite_blame guards-train --families net --kinds tree gbt logreg --data results/dataset_net.jsonl --out results/guards_eval_net.json --bundle results/guards_net.pkl 2>&1 | grep "^| net\|bundled"
echo "=== $(date) guards-eval net on the held-out corpus image models"
.venv/bin/python -m rewrite_blame --quiet --session "$D-gpu-e2e-net" guards-eval --bundle results/guards_net.pkl --families net \
  --models resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny conv_bn_relu_stack vit_b_16 --out results/guards_eval_e2e_net.json 2>&1 | grep -v "$F" | tail -16
echo "=== $(date) gpu net guard done"
