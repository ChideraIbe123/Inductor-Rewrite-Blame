#!/usr/bin/env bash
# On the pod: end-to-end effect of the conv (channels-last) guard on the GPU image models, stock base.
cd /workspace/rewrite-blame
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
echo "=== $(date) guards-eval conv on GPU image models"
.venv/bin/python -m rewrite_blame --quiet --session "$(date +%F)-gpu-e2e" guards-eval --bundle results/guards_3machines.pkl --families conv \
  --models resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny conv_bn_relu_stack vit_b_16 --out results/guards_eval_e2e.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -16
echo "=== $(date) gpu e2e done"
