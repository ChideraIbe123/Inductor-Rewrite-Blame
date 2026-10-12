#!/usr/bin/env bash
# VM: wait for the pair scan to finish, then collect the guard dataset (three families, shuffled, resumable).
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q "=== all done" logs/pairs3.log 2>/dev/null; do sleep 300; done
export REWRITE_BLAME_THREADS=4
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
for FAM in mlp conv attn; do
  echo "=== $(date) dataset $FAM"
  $PY -m rewrite_blame --quiet --session "$(date +%F)-ds-$FAM" --cooldown 0 dataset --family $FAM --level full --shuffle 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -400
done
echo "=== $(date) dataset done"
