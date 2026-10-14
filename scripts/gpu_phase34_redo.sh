#!/usr/bin/env bash
# On the pod: redo phase 3 (conv family only; attention rules never fire on CUDA, all 414 rows were no_effect)
# and phase 4 (repeat session) ALONE on the GPU. The first run of both was contaminated: two copies of the
# phase 2-4 chain ran concurrently (one from a hung ssh launch), so their timings competed for the GPU.
cd /workspace/rewrite-blame
while ! grep -q "gpu phase 2b done" logs/gpu_chain_p2b.log 2>/dev/null; do sleep 120; done
mkdir -p results/contaminated
mv results/dataset_conv.jsonl results/contaminated/ 2>/dev/null
for f in results/noise2_*.json results/sweep2_*.json; do [ -f "$f" ] && mv "$f" results/contaminated/; done
export GPU_SESSION=2026-10-14-gpu-redo GPU_SESSION2=2026-10-14-gpu-redo2
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
echo "=== $(date) dataset conv (redo, alone)"
.venv/bin/python -m rewrite_blame --quiet --session "$GPU_SESSION-ds-conv" dataset --family conv --level full --shuffle 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | grep -E "^\[dataset\]|^\{" | tail -300
scripts/gpu_chain.sh 4
echo "=== $(date) gpu redo done"
