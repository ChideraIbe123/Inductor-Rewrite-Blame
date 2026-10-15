#!/usr/bin/env bash
# VM pinning study, corrected: OMP_PROC_BIND=true made PyTorch 3x slower on this VM (all threads serialised),
# so pin with taskset only.  B2 = taskset -c 1-3 (off core 0), C2 = taskset -c 0-2 (on core 0); A from study 1.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; F='^DEBUG\|^INFO\|Graph Metrics'; D=$(date +%F)
export REWRITE_BLAME_THREADS=3 TOKENIZERS_PARALLELISM=false
run() { local label=$1; shift; echo "=== $(date) $label"; "$@" 2>&1 | grep -v "$F" | tail -2; }
for M in resnet18 distilbert smollm_135m_decode; do
  run "B2 noise $M" taskset -c 1-3 $PY -m rewrite_blame --quiet --session "$D-pinB2" noise --model $M --runs 7 --out results/noise_pinB2_$M.json
  run "C2 noise $M" taskset -c 0-2 $PY -m rewrite_blame --quiet --session "$D-pinC2" noise --model $M --runs 7 --out results/noise_pinC2_$M.json
done
run "B2 sweep resnet18" taskset -c 1-3 $PY -m rewrite_blame --quiet --session "$D-pinB2" sweep --model resnet18 --noise-file results/noise_pinB2_resnet18.json --out results/sweep_pinB2_resnet18.json
echo "=== $(date) pinning study 2 done"
