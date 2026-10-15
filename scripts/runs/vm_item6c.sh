#!/usr/bin/env bash
# Once item6b hands over to the pair scan: pause it, run the confirmed Qwen normalization check, resume it.
set -uo pipefail
cd "$(dirname "$0")/../.."
while ! grep -q "restarting the cat_slice_cat pair scan" logs/item6b.log 2>/dev/null; do sleep 120; done
sleep 30; tmux kill-session -t pairs3 2>/dev/null; pkill -f "[p]ython -m rewrite_blame.worker"; pkill -f "[p]ython -m rewrite_blame --quiet --session pairs-v3"; sleep 5
export REWRITE_BLAME_THREADS=4
PY=.venv/bin/python
F='^DEBUG\|^INFO\|Graph Metrics'
echo "=== $(date) confirmed check: qwen2_0_5b_prefill default vs normalization_pass alone"
$PY -m rewrite_blame --quiet --session "$(date +%F)-cases" attribute --model qwen2_0_5b_prefill --fast default --slow "default+optimus/normalization_pass" --out results/attribution_qwen2_0_5b_prefill_norm.json 2>&1 | grep -v "$F" | sed -n 1,14p
echo "=== $(date) resuming the pair scan"
tmux new-session -d -s pairs3 "REWRITE_BLAME_THREADS=4 .venv/bin/python -m rewrite_blame --quiet --session pairs-v3 interactions --model cat_slice_cat --out results/interactions_cat_slice_cat_v3.json > logs/pairs3.log 2>&1; echo \"=== all done\" >> logs/pairs3.log"
echo "=== $(date) item6c done"
