#!/usr/bin/env bash
# VM: resume the guard dataset collection for all three families (rows append to results/dataset_<family>.jsonl;
# configurations already present are skipped, so this is safe to relaunch after the daily reboot).
set -uo pipefail
cd "$(dirname "$0")/.."
export REWRITE_BLAME_THREADS=4
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
for FAM in attn mlp conv; do
  echo "=== $(date) dataset $FAM (resume)"
  $PY -m rewrite_blame --quiet --session "$(date +%F)-ds-$FAM" dataset --family $FAM --level full --shuffle 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -500
done
echo "=== $(date) dataset done"
