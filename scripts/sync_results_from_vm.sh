#!/usr/bin/env bash
# Pull the VM's results into results/vm/ (never overwrites the Mac store).
set -euo pipefail
VM=${VM:-fa26-cs598a-010.cs.illinois.edu}
cd "$(dirname "$0")/.."
mkdir -p results/vm
rsync -az "$VM:~/rewrite-blame/results/" results/vm/
echo pulled into results/vm/
