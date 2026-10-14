#!/usr/bin/env bash
# On the pod: recollect the conv guard dataset with GPU-sized batches (>=64M activation elements), train a
# GPU-only conv guard from it, and measure the guard end-to-end on the GPU image models. Then stop (idle) so
# the pod can be terminated from the Mac.
cd /workspace/rewrite-blame
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
F='^DEBUG\|^INFO\|Graph Metrics'
[ -f results/dataset_conv.jsonl ] && mv results/dataset_conv.jsonl results/dataset_conv_smallbatch.jsonl
echo "=== $(date) dataset conv v2 (GPU batches)"
.venv/bin/python -m rewrite_blame --quiet --session "$(date +%F)-gpu-ds2-conv" dataset --family conv --level full --shuffle 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -300
echo "=== $(date) guards-train GPU-only conv"
.venv/bin/python -m rewrite_blame guards-train --families conv --kinds tree gbt logreg --data results/dataset_conv.jsonl --out results/guards_eval_gpu2.json --bundle results/guards_gpu2.pkl 2>&1 | grep "^| conv\|bundled"
scripts/gpu_e2e.sh results/guards_gpu2.pkl results/guards_eval_e2e_gpu2.json
echo "=== $(date) gpu dataset v2 done"
