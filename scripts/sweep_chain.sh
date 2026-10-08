#!/usr/bin/env bash
# Sequentially: noise -> verify -> sweep for each model given on the command line.
# Usage: scripts/sweep_chain.sh [--threads N] model1 model2 ...
set -uo pipefail
cd "$(dirname "$0")/.."
THREADS=${REWRITE_BLAME_THREADS:-0}
if [ "${1:-}" = "--threads" ]; then THREADS=$2; shift 2; fi
export REWRITE_BLAME_THREADS=$THREADS
PY=.venv/bin/python
mkdir -p logs results
for M in "$@"; do
  echo "=== $(date) noise $M"
  $PY -m rewrite_blame --quiet noise  --model "$M" --runs 7 --out "results/noise_$M.json"  2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -3
  echo "=== $(date) verify $M"
  $PY -m rewrite_blame --quiet verify --model "$M" --out "results/verify_$M.json" 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -3
  echo "=== $(date) sweep $M"
  $PY -m rewrite_blame --quiet sweep  --model "$M" --noise-file "results/noise_$M.json" --out "results/sweep_$M.json" 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -3
done
echo "=== $(date) chain done"
