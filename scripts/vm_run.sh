#!/usr/bin/env bash
# Run a rewrite_blame command on the VM inside tmux (detached). Usage: scripts/vm_run.sh NAME "args..."
set -euo pipefail
VM=${VM:-fa26-cs598a-010.cs.illinois.edu}
NAME=$1; shift
ssh "$VM" "cd ~/rewrite-blame && mkdir -p logs && tmux new-session -d -s $NAME \
  \"REWRITE_BLAME_THREADS=4 .venv/bin/python -m rewrite_blame $* 2>&1 | tee logs/$NAME.log\""
echo "started tmux session $NAME on $VM (logs/$NAME.log)"
