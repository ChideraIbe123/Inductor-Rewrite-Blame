#!/usr/bin/env bash
# Repeat the ResNet-50 pair scan in a quiet daytime session (the night run had 18/33 disturbed baselines).
set -uo pipefail
cd "$(dirname "$0")/../.."
now=$(date +%s); target=$(date -j -f "%H:%M" "10:00" +%s 2>/dev/null || echo $now)
[ "$target" -le "$now" ] && target=$((target + 86400))
sleep $((target - now))
export REWRITE_BLAME_THREADS=8
echo "=== $(date) quiet pair scan resnet50"
.venv/bin/python -m rewrite_blame --quiet --session "$(date +%F)-pairs-quiet" --cooldown 1.0 interactions --model resnet50 \
  --ids "lowering/freezing,lowering/conv_1x1_as_mm,lowering/layout_optimization,sched/inplace_buffers,cpp/loop_tail_vec,sched/max_fusion_size,post_grad/reorder_for_locality,master/post_grad_passes" \
  --out results/interactions_resnet50.json 2>&1 | grep -v "^DEBUG\|^INFO\|Graph Metrics" | sed -n 1,12p
echo "=== $(date) quiet pair scan done"
