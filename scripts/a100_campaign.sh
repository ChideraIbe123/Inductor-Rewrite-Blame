#!/usr/bin/env bash
# Full campaign on a persistent GPU host (e.g. an A100 server): bf16 sweep (phase 1), top-switch real-network datasets
# (phase 2), guards + end-to-end with oracle and max-autotune + attributions (phase 3), then the freezing package.
# Run inside tmux on the host:  ROOT=$PWD scripts/a100_campaign.sh
# Everything is resumable through the store; results stay on the host under $ROOT/results.
set -uo pipefail
export ROOT="${ROOT:-$PWD}"
cd "$ROOT"
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_DTYPE=${REWRITE_BLAME_DTYPE:-bf16} REWRITE_BLAME_THREADS=${REWRITE_BLAME_THREADS:-4} REWRITE_BLAME_STAT=min TOKENIZERS_PARALLELISM=false
mkdir -p logs results
echo "=== $(date) host: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
TOPK=${TOPK:-5} scripts/gpu_campaign2.sh 1 2 3
scripts/gpu_freezing_package.sh
echo "=== $(date) a100 campaign done"
