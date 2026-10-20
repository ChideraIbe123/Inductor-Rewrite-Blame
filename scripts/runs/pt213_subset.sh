#!/usr/bin/env bash
# E1: the same noise/verify/sweep on a 10-model subset under PyTorch 2.13.0 (separate venv, registry, store, results
# dir), so the sign matrix gets a second-version column per machine. Usage: scripts/runs/pt213_subset.sh [threads]
set -uo pipefail
cd "$(dirname "$0")/../.."
export REWRITE_BLAME_THREADS=${1:-8} REWRITE_BLAME_STAT=min TOKENIZERS_PARALLELISM=false
# torch 2.13's C++ builder does not quote paths, so the venv must live at a path without spaces (Mac: ~/venvs/rb-pt213)
PY="${PT213_PY:-$HOME/venvs/rb-pt213/bin/python}"; [ -x "$PY" ] || PY=".venv-pt213/bin/python"
F='^DEBUG\|^INFO\|Graph Metrics'; S="$(date +%F)-pt213"
R="results/pt213"; mkdir -p "$R" logs
PIN=""; command -v taskset >/dev/null && PIN="taskset -c 1-3"
COMMON="--registry $R/switches.json --store $R/measurements.sqlite --session $S"
[ -f "$R/switches.json" ] || $PY -m rewrite_blame --registry "$R/switches.json" --store "$R/measurements.sqlite" discover >/dev/null 2>&1
for M in resnet18 resnet50 mobilenet_v3_small efficientnet_b0 bert_base_eagerattn distilbert gpt2_prefill gpt2_decode smollm_135m_decode qwen2_0_5b_decode; do
  echo "=== $(date) [pt2.13] noise $M";  $PIN $PY -m rewrite_blame --quiet $COMMON --cooldown 0.5 noise  --model "$M" --runs 7 --out "$R/noise_$M.json" 2>&1 | grep -v "$F" | tail -1
  echo "=== $(date) [pt2.13] verify $M"; $PIN $PY -m rewrite_blame --quiet $COMMON verify --model "$M" --out "$R/verify_$M.json" 2>&1 | grep -v "$F" | tail -1
  echo "=== $(date) [pt2.13] sweep $M";  $PIN $PY -m rewrite_blame --quiet $COMMON --cooldown 0.5 sweep --model "$M" --noise-file "$R/noise_$M.json" --out "$R/sweep_$M.json" 2>&1 | grep -v "$F" | tail -2
done
echo "=== $(date) pt213 subset done"
