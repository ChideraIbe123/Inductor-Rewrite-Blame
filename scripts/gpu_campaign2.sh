#!/usr/bin/env bash
# GPU campaign 2 (runs ON a pod, in one tmux session, bf16 throughout):
#   phase 1  discover + noise/verify/sweep over the corpus in bf16            (session <date>-gpu-bf16)
#   phase 2  rank switches by confirmed effect; per-switch real-model datasets over the zoos
#   phase 3  train one guard per switch (block or enable), end-to-end on the held-out corpus with oracle
#            and max-autotune columns; attribution case studies seeded from the bf16 sweep
# Usage: scripts/gpu_campaign2.sh [phase...]   (default: 1 2 3). Resumable: the store caches every measurement.
set -uo pipefail
cd /workspace/rewrite-blame

self_terminate() {   # called at the end: stop billing even if nobody is watching (key passed via env, never written)
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "=== $(date) self-terminating pod $RUNPOD_POD_ID"
    sleep ${SELF_TERMINATE_DELAY:-600}      # leave 10 minutes for a final pull
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
  fi
}
trap self_terminate EXIT
export REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_DTYPE=${REWRITE_BLAME_DTYPE:-bf16} REWRITE_BLAME_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python; F='^DEBUG\|^INFO\|Graph Metrics'
D=${CAMPAIGN_DATE:-$(date +%F)}; S="$D-gpu-$REWRITE_BLAME_DTYPE"
MODELS="attention_block norm_mlp decode_mlp conv_bn_relu_stack resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny vit_b_16 bert_base bert_base_eagerattn distilbert gpt2_prefill gpt2_prefill_eagerattn gpt2_decode t5_small_encoder smollm_135m_prefill smollm_135m_prefill_eagerattn smollm_135m_decode qwen2_0_5b_prefill qwen2_0_5b_decode"
HELD_OUT="resnet18 resnet50 mobilenet_v3_small efficientnet_b0 convnext_tiny conv_bn_relu_stack vit_b_16 bert_base distilbert gpt2_prefill gpt2_decode smollm_135m_prefill smollm_135m_decode qwen2_0_5b_decode"
TOPK=${TOPK:-5}
mkdir -p logs results
run() { local label=$1; shift; echo "=== $(date) $label"; "$@" 2>&1 | grep -v "$F" | tail -${TAIL:-3}; }
for ph in ${*:-1 2 3}; do
case $ph in
1)
  run "discover ($REWRITE_BLAME_DTYPE)" $PY -m rewrite_blame discover
  for M in $MODELS; do
    run "noise $M"  $PY -m rewrite_blame --quiet --session "$S" noise  --model "$M" --runs 7 --out "results/noise_$M.json"
    run "verify $M" $PY -m rewrite_blame --quiet --session "$S" verify --model "$M" --out "results/verify_$M.json"
    run "sweep $M"  $PY -m rewrite_blame --quiet --session "$S" sweep --model "$M" --noise-file "results/noise_$M.json" --out "results/sweep_$M.json"
  done ;;
2)
  echo "=== $(date) ranking switches"
  $PY scripts/rank_switches.py results --top $TOPK --min-pct 3 --exclude-master --verbose | tee results/top_switches.tsv || true
  # freezing is default-off and a confirmed win on most GPU models: always include it (an "enable" guard)
  # SWITCHES_OVERRIDE lets a second pod collect datasets in parallel with the sweep (fixed list from the fp32 ranking)
  SWITCHES="${SWITCHES_OVERRIDE:-$($PY scripts/rank_switches.py results --top $TOPK --min-pct 3 --exclude-master | tr '\n' ' ') lowering/freezing}"
  for SW in $SWITCHES; do
    case "$SW" in
      *layout*|*conv_1x1*|*freezing*|*max_fusion*|*inplace*|*reorder*|*post_grad_passes*|*pattern_matcher*) SET="zoo:all" ;;
      *sfdp*|*softmax*|*unfuse*|*addmm*|*decompose*|*normalization*|*batch_*) SET="zoo:hf" ;;
      *) SET="zoo:all" ;;
    esac
    slug=$(echo "$SW" | tr '/' '_')
    TAIL=400 run "dataset --switch $SW over $SET" $PY -m rewrite_blame --quiet --session "$S-sw" dataset --switch "$SW" --models "$SET" --shuffle --noise-config zoo_resnet34_b32 --out "results/dataset_switch_$slug.jsonl"
  done ;;
3)
  TAIL=60 run "guards-train (all switch families in results/dataset_switch_*.jsonl)" \
    $PY -m rewrite_blame guards-train --kinds tree gbt logreg $(for f in results/dataset_switch_*.jsonl; do echo --data "$f"; done) --out results/guards_eval_switches.json --bundle results/guards_switches.pkl
  FAMS=$($PY - <<'PY'
import pickle; b = pickle.load(open("results/guards_switches.pkl", "rb")); print(" ".join(sorted(b.guards)))
PY
)
  TAIL=40 run "guards-eval end-to-end (held-out corpus, oracle + max-autotune)" \
    $PY -m rewrite_blame --quiet --session "$S-e2e" guards-eval --bundle results/guards_switches.pkl --families $FAMS --models $HELD_OUT --max-autotune --out results/guards_eval_e2e_switches.json
  echo "=== $(date) attribution case studies from the bf16 sweep"
  $PY - <<'PY' > results/attr_cases.txt
import glob, json
cases = []
for f in sorted(glob.glob("results/sweep_*.json")):
    d = json.load(open(f)); rows = d.get("rows", [])
    model = f.split("sweep_")[1][:-5]
    slow = [r for r in rows if r.get("slower") and r["delta_pct"] >= 5]
    if slow:
        changes = "".join(("-" if r["direction"] == "off" else "+") + r["switch"] for r in slow[:1])
        # hide the culprit among the other switches that change this model's program (max 6)
        others = [r for r in rows if r["switch"] != slow[0]["switch"]][:5]
        dec = "".join(("-" if r["direction"] == "off" else "+") + r["switch"] for r in others)
        cases.append((model, "default" + changes + dec))
for m, spec in cases[:6]:
    print(m, spec)
PY
  while read -r M SPEC; do
    [ -z "$M" ] && continue
    TAIL=14 run "attribute-all $M: default vs $SPEC" $PY -m rewrite_blame --quiet --session "$S-attr" attribute-all --model "$M" --fast default --slow "$SPEC" --out "results/attribution_all_${M}_bf16.json"
  done < results/attr_cases.txt ;;
esac
done
echo "=== $(date) gpu campaign 2 done"
