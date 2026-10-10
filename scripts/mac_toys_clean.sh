#!/usr/bin/env bash
# After the corpus chain finishes, re-measure the toy models in a clean session (the first ones in
# the chain overlapped with development activity on this machine).
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q "chain done" logs/mac_corpus.log 2>/dev/null; do sleep 60; done
export REWRITE_BLAME_THREADS=8
PY=.venv/bin/python
S=$(date +%F)-clean
for M in attention_block norm_mlp decode_mlp split_cat_mlp cat_slice_cat conv_bn_relu_stack; do
  echo "=== $(date) clean session $S: $M"
  $PY -m rewrite_blame --quiet --session "$S" noise  --model $M --runs 7 --out results/noise_$M.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -2
  $PY -m rewrite_blame --quiet --session "$S" sweep  --model $M --noise-file results/noise_$M.json --out results/sweep_$M.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -2
done
echo "=== $(date) toys clean done"
