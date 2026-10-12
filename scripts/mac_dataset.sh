#!/usr/bin/env bash
# Guard dataset collection on the Mac: three families, shuffled order (resumable; rows append to results/dataset_<family>.jsonl).
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
for FAM in mlp conv attn; do
  echo "=== $(date) dataset $FAM"
  $PY -m rewrite_blame --quiet --session "$(date +%F)-ds-$FAM" --cooldown 0.5 dataset --family $FAM --level full --shuffle 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -400
done
echo "=== $(date) dataset done"
