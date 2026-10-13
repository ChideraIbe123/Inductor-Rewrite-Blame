#!/usr/bin/env bash
# Runs ON the pod once: Python 3.12 venv via uv, torch 2.14.1 (CUDA wheel), project deps. Idempotent.
set -euo pipefail
cd /workspace/rewrite-blame
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d .venv ] || uv venv -p 3.12 .venv
# the hosts run driver 570 (CUDA 12.8): torch 2.14.1 ships cu126 and cu130 wheels only, so use cu126
uv pip install -p .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1+cu126" torchvision
uv pip install -p .venv/bin/python transformers pytest numpy scikit-learn tabulate
apt-get install -y -qq rsync tmux >/dev/null 2>&1 || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('triton', triton.__version__)"
