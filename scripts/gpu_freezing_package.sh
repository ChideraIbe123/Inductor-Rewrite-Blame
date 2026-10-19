#!/usr/bin/env bash
# On a pod (bf16): the freezing package = freezing and layout on/off over the pretrained corpus models and their
# random-init twins (rows carry compile time and peak CUDA memory), the two real end-to-end workloads, and the
# 20-line reproducer numbers. Self-terminates when done (RUNPOD_API_KEY + RUNPOD_POD_ID in env).
cd /workspace/rewrite-blame
self_terminate() {
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== $(date) self-terminating pod $RUNPOD_POD_ID in ${SELF_TERMINATE_DELAY:-900}s"; sleep ${SELF_TERMINATE_DELAY:-900}
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
  fi
}
trap self_terminate EXIT
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_DTYPE=${REWRITE_BLAME_DTYPE:-bf16} REWRITE_BLAME_THREADS=4 REWRITE_BLAME_STAT=min TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python; F='^DEBUG\|^INFO\|Graph Metrics'; D=$(date +%F)
PRE="resnet18_pretrained resnet50_pretrained mobilenet_v3_small_pretrained efficientnet_b0_pretrained convnext_tiny_pretrained bert_base_eagerattn_pretrained distilbert_pretrained gpt2_prefill_pretrained gpt2_decode_pretrained smollm_135m_decode_pretrained smollm_135m_prefill_pretrained qwen2_0_5b_decode_pretrained"
RND="resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny bert_base_eagerattn distilbert gpt2_prefill gpt2_decode smollm_135m_decode smollm_135m_prefill qwen2_0_5b_decode"
MODELS=$(echo $PRE $RND | tr ' ' ',')
mkdir -p logs results
echo "=== $(date) discover ($REWRITE_BLAME_DTYPE)"; $PY -m rewrite_blame discover 2>&1 | grep -c "" >/dev/null
for SW in lowering/freezing lowering/layout_optimization; do
  slug=$(echo "$SW" | tr '/' '_')
  echo "=== $(date) dataset --switch $SW on pretrained + random-init corpus"
  $PY -m rewrite_blame --quiet --session "$D-gpu-freeze" dataset --switch "$SW" --models "$MODELS" --noise-config resnet18 \
    --out "results/dataset_switch_${slug}_pretrained.jsonl" 2>&1 | grep -v "$F" | grep -E "^\[dataset\]|^\{" | tail -40
done
echo "=== $(date) real workloads (generate, classify)"
$PY scripts/real_workload_freezing.py --device cuda --dtype $REWRITE_BLAME_DTYPE --runs 5 --out results/real_workload_freezing.json 2>&1 | grep -v "$F" | tail -12
echo "=== $(date) reproducer (regnet_y_400mf, fresh process per setting)"
for FZ in False True; do $PY - <<PY
import time, torch, torchvision, torch._inductor.config as cfg
cfg.freezing = $FZ; cfg.fx_graph_cache = False
m = torchvision.models.regnet_y_400mf(weights=None).cuda().to(torch.bfloat16).eval()
x = torch.randn(8, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    c = torch.compile(m); c(x); torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        t = time.perf_counter(); c(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
print("freezing=$FZ: median %.3f ms  min %.3f ms" % (sorted(ts)[25] * 1e3, min(ts) * 1e3))
PY
done 2>&1 | grep "freezing="
echo "=== $(date) gpu freezing package done"
