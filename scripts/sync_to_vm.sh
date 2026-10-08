#!/usr/bin/env bash
# Push code (not results/caches) to the VM.
set -euo pipefail
VM=${VM:-fa26-cs598a-010.cs.illinois.edu}
cd "$(dirname "$0")/.."
rsync -az --delete --exclude .venv --exclude .git --exclude .inductor_cache --exclude logs --exclude __pycache__ \
  --exclude results/ --exclude report/ \
  ./ "$VM:~/rewrite-blame/"
echo synced
