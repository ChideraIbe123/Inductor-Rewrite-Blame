#!/usr/bin/env bash
# One-line status per running pod: GPU, cost, last chain log line. Usage: scripts/pods_status.sh
cd "$(dirname "$0")/.."
scripts/runpod.sh status | while read -r id name status gpu cost ip port; do
  [ "$ip" = "None" ] && { echo "$name: no ip yet"; continue; }
  last=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$port" root@"$ip" \
    'cd /workspace/rewrite-blame 2>/dev/null && for f in logs/c2_*.log logs/*.log; do [ -f "$f" ] && { echo -n "$(basename $f): "; grep "^=== " "$f" | tail -1 | cut -c1-110; break; }; done; echo -n "  rows: "; wc -l results/dataset_switch_*.jsonl 2>/dev/null | tail -1' </dev/null 2>/dev/null)
  echo "$name ($cost): $last"
done
