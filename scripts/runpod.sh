#!/usr/bin/env bash
# Minimal RunPod control for the GPU campaign. Key is read from ~/.runpod_key (never printed).
#   scripts/runpod.sh create [name]      create a 24-48 GB pod (RTX A5000 -> A40 -> L4), Secure Cloud, no volume
#   scripts/runpod.sh status             list pods with status, GPU, $/h and the SSH endpoint
#   scripts/runpod.sh wait <podId>       block until the pod is RUNNING and SSH answers; prints "host port"
#   scripts/runpod.sh ssh <podId> [cmd]  ssh into the pod (root)
#   scripts/runpod.sh sync <podId>       rsync the code to /workspace/rewrite-blame
#   scripts/runpod.sh pull <podId>       rsync results back to results/gpu/
#   scripts/runpod.sh stop <podId>       stop (keeps the container disk, bills only storage)
#   scripts/runpod.sh rm <podId>         terminate (bills nothing afterwards)
#   scripts/runpod.sh balance            account balance and this month's pod spend
set -euo pipefail
cd "$(dirname "$0")/.."
K=$(cat ~/.runpod_key)
API=https://rest.runpod.io/v1
hdr=(-H "Authorization: Bearer $K" -H "Content-Type: application/json")
PUBKEY=$(cat ~/.ssh/id_ed25519.pub)
IMAGE=${RUNPOD_IMAGE:-runpod/pytorch:1.4.1-rc.177-cu1281-torch291-ubuntu2404}

case "${1:-}" in
  create)
    name=${2:-rewrite-blame-gpu}
    body=$(python3 - "$name" "$IMAGE" "$PUBKEY" <<'PY'
import json, sys
name, image, pub = sys.argv[1:4]
print(json.dumps({
  "name": name, "imageName": image, "cloudType": __import__("os").environ.get("RUNPOD_CLOUD", "SECURE"), "computeType": "GPU", "gpuCount": 1,
  "gpuTypeIds": [g for g in __import__("os").environ.get("RUNPOD_GPUS", "NVIDIA RTX A5000,NVIDIA A40,NVIDIA L4,NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090").split(",") if g],
  "gpuTypePriority": "custom", "allowedCudaVersions": ["12.8", "12.9", "13.0"],
  "containerDiskInGb": 40, "volumeInGb": 0, "ports": ["22/tcp"], "minVCPUPerGPU": 4, "minRAMPerGPU": 16,
  "env": {"PUBLIC_KEY": pub}, "interruptible": False, "supportPublicIp": True}))
PY
)
    curl -s "${hdr[@]}" -X POST "$API/pods" -d "$body" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('id') or d)"
    ;;
  status)
    curl -s "${hdr[@]}" "$API/pods" | python3 -c "
import json,sys
for p in json.load(sys.stdin):
    pm = p.get('portMappings') or {}
    print(p['id'], p['name'], p.get('desiredStatus'), (p.get('machine') or {}).get('gpuTypeId'), f\"\${p.get('costPerHr')}/h\", p.get('publicIp'), pm.get('22'))"
    ;;
  wait)
    for i in $(seq 1 120); do
      read -r ip port < <(curl -s "${hdr[@]}" "$API/pods/$2" | python3 -c "
import json,sys; p=json.load(sys.stdin); pm=p.get('portMappings') or {}; print(p.get('publicIp') or '-', pm.get('22') or '-')")
      if [ "$ip" != "-" ] && [ "$port" != "-" ] && ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -p "$port" root@"$ip" true 2>/dev/null; then
        echo "$ip $port"; exit 0; fi
      sleep 10
    done
    echo "timed out waiting for $2" >&2; exit 1
    ;;
  ssh)
    read -r ip port < <("$0" wait "$2"); shift 2
    ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -p "$port" root@"$ip" "$@"
    ;;
  sync)
    read -r ip port < <("$0" wait "$2")
    rsync -az --delete -e "ssh -o StrictHostKeyChecking=no -p $port" --exclude .venv --exclude .git --exclude results/ --exclude logs/ \
      --exclude .inductor_cache --exclude report/ --exclude '__pycache__' --exclude '*.pyc' ./ root@"$ip":/workspace/rewrite-blame/
    ;;
  pull)
    read -r ip port < <("$0" wait "$2"); mkdir -p results/gpu logs/gpu
    # the store is kept per pod so a later pod can never overwrite an earlier pod's raw samples
    rsync -az --exclude measurements.sqlite -e "ssh -o StrictHostKeyChecking=no -p $port" root@"$ip":/workspace/rewrite-blame/results/ results/gpu/
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $port" root@"$ip":/workspace/rewrite-blame/results/measurements.sqlite "results/gpu/measurements_$2.sqlite" 2>/dev/null || true
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $port" root@"$ip":/workspace/rewrite-blame/logs/ logs/gpu/ || true
    ;;
  stop) curl -s "${hdr[@]}" -X POST "$API/pods/$2/stop" | head -c 200; echo ;;
  rm)   curl -s "${hdr[@]}" -X DELETE "$API/pods/$2" -w "terminate: %{http_code}\n" ;;
  balance)
    curl -s "https://api.runpod.io/graphql" "${hdr[@]}" -d '{"query":"{ myself { clientBalance spendLimit } }"}'; echo ;;
  *) sed -n 2,12p "$0" ;;
esac
