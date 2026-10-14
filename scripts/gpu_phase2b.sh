#!/usr/bin/env bash
# On the pod, after phases 2-4: attribution case studies built from the GPU's OWN sweep findings
# (the CPU pairs turned out not to be slow on the A40, which the tool correctly reported).
cd /workspace/rewrite-blame
while ! grep -q "gpu chain done" logs/gpu_chain_p234.log 2>/dev/null; do sleep 120; done
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python; F='^DEBUG\|^INFO\|Graph Metrics'; S=$(date +%F)-gpu-attr
run() { echo "=== $(date) $1"; shift; "$@" 2>&1 | grep -v "$F" | tail -14; }
run "attribute-all resnet50: default vs conv_1x1_as_mm + all opt-in passes" \
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model resnet50 --fast default --slow "default~optimus+lowering/conv_1x1_as_mm" --out results/attribution_all_resnet50_gpu.json
run "attribute-all gpt2_decode: default vs all opt-in passes" \
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model gpt2_decode --fast default --slow "default~optimus" --out results/attribution_all_gpt2_decode_gpu.json
run "attribute-all resnet18: best (layout off, freezing on) vs default" \
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model resnet18 --fast "default-lowering/layout_optimization+lowering/freezing" --slow default --out results/attribution_all_resnet18_gpu.json
run "attribute-all convnext_tiny: default vs four scheduler/pass flags off" \
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model convnext_tiny --fast default --slow "default-sched/max_fusion_size-post_grad/reorder_for_locality-sched/inplace_buffers-joint/joint_graph.early_patterns/pointless_view-sched/reorder_for_peak_memory" --out results/attribution_all_convnext_tiny_gpu.json
run "attribute-all bert_base_eagerattn: default vs five flags off" \
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model bert_base_eagerattn --fast default --slow "default-sched/max_fusion_size-joint/constant_folding-post_grad/reorder_for_locality-sched/inplace_buffers-sched/reorder_for_peak_memory" --out results/attribution_all_bert_base_eagerattn_gpu.json
run "attribute-all smollm_135m_prefill: default+freezing vs default minus max_fusion_size" \
  $PY -m rewrite_blame --quiet --session "$S" attribute-all --model smollm_135m_prefill --fast "default+lowering/freezing" --slow "default-sched/max_fusion_size" --out results/attribution_all_smollm_135m_prefill_gpu.json
run "tau-scan resnet50 (stability of the conv_1x1_as_mm verdict)" \
  $PY -m rewrite_blame --quiet --session "$S" tau-scan --model resnet50 --fast default --slow "default+lowering/conv_1x1_as_mm" --out results/tau_scan_resnet50_gpu.json
echo "=== $(date) gpu phase 2b done"
