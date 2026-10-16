#!/usr/bin/env bash
# Xeon VM: real-model rows for the two CPU-relevant switches over the small zoos (batch 8), pinned to cores 1-3,
# min statistic. Resumable across the daily reboot (rows append; measurements cached).
set -uo pipefail
cd "$(dirname "$0")/../.."
export REWRITE_BLAME_THREADS=3 REWRITE_BLAME_STAT=min TOKENIZERS_PARALLELISM=false
PY=".venv/bin/python"; F='^DEBUG\|^INFO\|Graph Metrics'; D=$(date +%F)
for SW in lowering/layout_optimization optimus/decompose_mm_wide; do
  slug=$(echo "$SW" | tr '/' '_')
  echo "=== $(date) dataset --switch $SW (zoo small, pinned 1-3)"
  taskset -c 1-3 $PY -m rewrite_blame --quiet --session "$D-vm-sw" dataset --switch "$SW" --models zoo:all --level small --shuffle \
    --noise-config zoo_resnet34_b8 --out "results/dataset_switch_$slug.jsonl" 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -200
done
echo "=== $(date) vm switch datasets done"
