#!/usr/bin/env bash
# Mac: real-model rows for the layout and matmul-decomposition switches over the small zoos (batch 8).
# Run plugged in, lid open, nothing else running; cooldown between measurements; min statistic.
set -uo pipefail
cd "$(dirname "$0")/../.."
export REWRITE_BLAME_THREADS=8 REWRITE_BLAME_STAT=min TOKENIZERS_PARALLELISM=false
PY=".venv/bin/python"; F='^DEBUG\|^INFO\|Graph Metrics'; D=$(date +%F)
for SW in lowering/layout_optimization optimus/decompose_mm_wide; do
  slug=$(echo "$SW" | tr '/' '_')
  echo "=== $(date) dataset --switch $SW (zoo small); power: $(pmset -g batt | tail -1 | cut -c1-50)"
  $PY -m rewrite_blame --quiet --session "$D-mac-sw" --cooldown 1.0 dataset --switch "$SW" --models zoo:all --level small --shuffle \
    --noise-config zoo_resnet34_b8 --out "results/dataset_switch_$slug.jsonl" 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -200
done
echo "=== $(date) mac switch datasets done"
