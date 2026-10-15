#!/usr/bin/env bash
# VM: does pinning the measurement to fixed cores (and away from core 0) reduce between-run noise, and do
# the sweep verdicts still line up?  Three arrangements, same models, same day:
#   A  3 threads, no pinning                       (session <date>-pinA)
#   B  3 threads, taskset -c 1-3 + OMP_PROC_BIND   (session <date>-pinB)  <- the professor's advice
#   C  3 threads, taskset -c 0-2 + OMP_PROC_BIND   (session <date>-pinC)  <- same, but including core 0
# Then a resnet18 sweep under A and under B.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; F='^DEBUG\|^INFO\|Graph Metrics'; D=$(date +%F)
export REWRITE_BLAME_THREADS=3 TOKENIZERS_PARALLELISM=false
MODELS="resnet18 distilbert smollm_135m_decode"
run() { # label, session, prefix-command..., then the rewrite_blame args
  local label=$1 sess=$2; shift 2
  echo "=== $(date) $label"
  "$@" 2>&1 | grep -v "$F" | tail -2
}
for M in $MODELS; do
  run "A noise $M"  "$D-pinA" $PY -m rewrite_blame --quiet --session "$D-pinA" noise --model $M --runs 7 --out results/noise_pinA_$M.json
  run "B noise $M"  "$D-pinB" env OMP_PROC_BIND=true OMP_PLACES=cores taskset -c 1-3 $PY -m rewrite_blame --quiet --session "$D-pinB" noise --model $M --runs 7 --out results/noise_pinB_$M.json
  run "C noise $M"  "$D-pinC" env OMP_PROC_BIND=true OMP_PLACES=cores taskset -c 0-2 $PY -m rewrite_blame --quiet --session "$D-pinC" noise --model $M --runs 7 --out results/noise_pinC_$M.json
done
run "A sweep resnet18" "$D-pinA" $PY -m rewrite_blame --quiet --session "$D-pinA" sweep --model resnet18 --noise-file results/noise_pinA_resnet18.json --out results/sweep_pinA_resnet18.json
run "B sweep resnet18" "$D-pinB" env OMP_PROC_BIND=true OMP_PLACES=cores taskset -c 1-3 $PY -m rewrite_blame --quiet --session "$D-pinB" sweep --model resnet18 --noise-file results/noise_pinB_resnet18.json --out results/sweep_pinB_resnet18.json
echo "=== $(date) pinning study done"
