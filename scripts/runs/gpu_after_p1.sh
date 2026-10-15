#!/usr/bin/env bash
# On the pod: wait for GPU phase 1 to finish, then run phases 2-4 back to back (no idle billing in between).
cd /workspace/rewrite-blame
while ! grep -q "gpu chain done" logs/gpu_chain_p1.log 2>/dev/null; do sleep 120; done
exec scripts/gpu_chain.sh 2 3 4 > logs/gpu_chain_p234.log 2>&1
