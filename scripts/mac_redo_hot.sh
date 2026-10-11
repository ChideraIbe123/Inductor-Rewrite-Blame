#!/usr/bin/env bash
# After the clean toy re-run, re-measure the models whose first session on the Mac was thermally throttled.
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q "toys clean done" logs/mac_toys_clean.log 2>/dev/null; do sleep 120; done
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
S=$(date +%F)-cool
for M in smollm_135m_prefill smollm_135m_prefill_eagerattn; do
  echo "=== $(date) cool session $S: $M"
  $PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 noise --model $M --runs 7 --out results/noise_$M.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -2
  $PY -m rewrite_blame --quiet --session "$S" --cooldown 1.0 sweep --model $M --noise-file results/noise_$M.json --out results/sweep_$M.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -2
done
echo "=== $(date) redo hot done"
