#!/usr/bin/env bash
# On a pod: end-to-end evaluation of switch-level guards on held-out models, with the oracle; self-terminates.
# Usage (on the pod): RUNPOD_API_KEY=... RUNPOD_POD_ID=... scripts/gpu_e2e_switches.sh <bundle.pkl> <out.json> [--max-autotune] [models...]
cd /workspace/rewrite-blame
self_terminate() {
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== $(date) self-terminating pod $RUNPOD_POD_ID in ${SELF_TERMINATE_DELAY:-600}s"
    sleep ${SELF_TERMINATE_DELAY:-600}
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
  fi
}
trap self_terminate EXIT
BUNDLE=$1; OUT=$2; shift 2
MA=""; if [ "${1:-}" = "--max-autotune" ]; then MA="--max-autotune"; shift; fi
MODELS=${*:-resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny bert_base distilbert gpt2_prefill gpt2_decode smollm_135m_prefill smollm_135m_decode qwen2_0_5b_decode}
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_DTYPE=${REWRITE_BLAME_DTYPE:-bf16} REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
F='^DEBUG\|^INFO\|Graph Metrics'
FAMS=$(.venv/bin/python - "$BUNDLE" <<'PY'
import pickle, sys; b = pickle.load(open(sys.argv[1], "rb")); print(" ".join(sorted(b.guards)))
PY
)
echo "=== $(date) discover ($REWRITE_BLAME_DTYPE)"; .venv/bin/python -m rewrite_blame discover 2>&1 | grep -c "" >/dev/null
echo "=== $(date) guards-eval families: $FAMS"
.venv/bin/python -m rewrite_blame --quiet --session "$(date +%F)-gpu-$REWRITE_BLAME_DTYPE-e2e" guards-eval --bundle "$BUNDLE" --families $FAMS --models $MODELS $MA --out "$OUT" 2>&1 | grep -v "$F" | tail -40
echo "=== $(date) gpu e2e switches done"
