#!/usr/bin/env bash
# Create an A40 pod (retrying secure, then community, for up to ~30 min), set it up and launch the E3 guard
# end-to-end rerun (SKIP_DISCARD=1) with self-termination. Prints POD/IP/PORT at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
POD=""
for i in $(seq 1 15); do
  for CLOUD in SECURE COMMUNITY; do
    OUT=$(RUNPOD_GPUS="NVIDIA A40" RUNPOD_CLOUD=$CLOUD scripts/runpod.sh create rewrite-blame-e3b 2>&1 | tail -1)
    if ! echo "$OUT" | grep -q "error"; then POD=$OUT; echo "created $POD on $CLOUD"; break 2; fi
  done
  echo "$(date +%H:%M) no A40 available; retrying in 2 min"; sleep 120
done
[ -n "$POD" ] || { echo "gave up: no A40"; exit 1; }
sleep 20
read -r ip port < <(scripts/runpod.sh wait "$POD") || exit 1
scripts/runpod.sh sync "$POD" >/dev/null
scripts/runpod.sh ssh "$POD" 'bash /workspace/rewrite-blame/scripts/pod_setup.sh' 2>&1 | grep "NVIDIA\|torch 2" | tail -2
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" root@"$ip" 'mkdir -p /workspace/rewrite-blame/results /workspace/rewrite-blame/logs' </dev/null
scp -q -o StrictHostKeyChecking=no -P "$port" results/gpu/a40/guards_switches.pkl root@"$ip":/workspace/rewrite-blame/results/guards_switches.pkl
K=$(cat ~/.runpod_key)
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" root@"$ip" "cat > /workspace/run_e3.sh <<EOS
#!/usr/bin/env bash
export RUNPOD_API_KEY=$K RUNPOD_POD_ID=$POD SKIP_DISCARD=1
exec /workspace/rewrite-blame/scripts/gpu_e3.sh results/guards_switches.pkl
EOS
chmod 700 /workspace/run_e3.sh && cd /workspace/rewrite-blame && (nohup setsid /workspace/run_e3.sh > /workspace/rewrite-blame/logs/gpu_e3.log 2>&1 < /dev/null &) ; sleep 15; tail -1 logs/gpu_e3.log" </dev/null
echo "POD=$POD IP=$ip PORT=$port"
