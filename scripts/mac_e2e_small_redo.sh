#!/usr/bin/env bash
# After the end-to-end run: repeat the small/fast models in a daytime, plugged-in session (the night run was on a low battery budget).
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q "guards final done" logs/mac_guards_final.log 2>/dev/null; do sleep 300; done
now=$(date +%s); target=$(date -j -f "%H:%M" "10:30" +%s 2>/dev/null || echo $now); [ "$target" -le "$now" ] && target=$((target + 86400)); sleep $((target - now))
export REWRITE_BLAME_THREADS=8
echo "=== $(date) e2e redo small models (session $(date +%F)-guards2); power: $(pmset -g batt | tail -1 | cut -c1-60)"
.venv/bin/python -m rewrite_blame --quiet --session "$(date +%F)-guards2" --cooldown 1.0 guards-eval --bundle results/guards.pkl --families mlp conv attn \
  --models decode_mlp gpt2_decode smollm_135m_decode qwen2_0_5b_decode attention_block resnet18 --out results/guards_eval_e2e_small.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | tail -30
echo "=== $(date) e2e redo done"
