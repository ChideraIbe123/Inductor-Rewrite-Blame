"""Freezing on vs off on two real end-to-end workloads, measured the same way as the harness (median per run,
several runs, min across runs reported too). Usage:
  python scripts/real_workload_freezing.py [--device cuda] [--dtype bf16] [--runs 5] [--out results/real_workload_freezing.json]
Workloads: (1) HF text generation with SmolLM-135M (pretrained), 32 new tokens greedy, KV cache, torch.compile on the
model; (2) torchvision ResNet-50 (pretrained) classifying a batch of 32 images. Each is compiled twice, with
torch._inductor.config.freezing False and True, in fresh subprocesses so caches never leak."""
import argparse, json, os, statistics, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--device", default=os.environ.get("REWRITE_BLAME_DEVICE", "cpu")); ap.add_argument("--dtype", default=os.environ.get("REWRITE_BLAME_DTYPE", "fp32"))
ap.add_argument("--runs", type=int, default=5); ap.add_argument("--out", default="results/real_workload_freezing.json")
ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
a = ap.parse_args()

WORKER = r'''
import json, sys, time, torch, torch._inductor.config as cfg
workload, freezing, device, dtype_name, runs = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4], int(sys.argv[5])
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
cfg.freezing = freezing
cfg.fx_graph_cache = False
def sync():
    if device.startswith("cuda"): torch.cuda.synchronize()
if workload == "generate":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", torch_dtype=dtype).to(device).eval()
    ids = tok("The compiler decided that", return_tensors="pt").input_ids.to(device)
    m.forward = torch.compile(m.forward, dynamic=True)
    def run():
        with torch.no_grad():
            return m.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
else:
    import torchvision
    m = torchvision.models.resnet50(weights="DEFAULT").to(device=device, dtype=dtype).eval()
    x = torch.randn(32, 3, 224, 224, device=device, dtype=dtype)
    m = torch.compile(m)
    def run():
        with torch.no_grad():
            return m(x).argmax(-1)
t0 = time.perf_counter()
with torch.no_grad():
    out0 = run(); sync()
compile_s = time.perf_counter() - t0
for _ in range(3):
    run(); sync()
samples = []
for _ in range(runs):
    per = []
    for _ in range(5):
        t = time.perf_counter(); run(); sync(); per.append((time.perf_counter() - t) * 1000)
    samples.append(sorted(per)[len(per) // 2])
peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
import resource; rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10
print("@@RESULT@@" + json.dumps({"workload": workload, "freezing": freezing, "compile_s": compile_s, "medians_ms": samples,
      "median_ms": statistics.median(samples) if (statistics := __import__("statistics")) else None, "min_ms": min(samples),
      "peak_cuda_mb": peak, "peak_rss_mb": rss_mb, "tokens_or_images": 32,
      "output": (out0.tolist()[0][-8:] if workload == "generate" else out0.tolist()[:8])}))
'''
results = []
for workload in ("generate", "classify"):
    for freezing in (0, 1):
        for rep in range(2):   # two independent processes per setting (the harness's AND rule)
            cmd = [sys.executable, "-c", WORKER, workload, str(freezing), a.device, a.dtype, str(a.runs)]
            p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
            line = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
            if not line:
                print(f"{workload} freezing={freezing} rep{rep}: FAILED\n{p.stderr[-1500:]}", file=sys.stderr); continue
            r = json.loads(line[0][len("@@RESULT@@"):]); r["rep"] = rep; results.append(r)
            print(f"{workload:<9} freezing={freezing} rep{rep}: median {r['median_ms']:.1f} ms, min {r['min_ms']:.1f} ms, compile {r['compile_s']:.1f}s, "
                  f"peak {'%.0f MB cuda' % r['peak_cuda_mb'] if r['peak_cuda_mb'] else '%.0f MB rss' % r['peak_rss_mb']}", flush=True)
summary = {}
for workload in ("generate", "classify"):
    off = [r for r in results if r["workload"] == workload and not r["freezing"]]
    on = [r for r in results if r["workload"] == workload and r["freezing"]]
    if off and on:
        ratios = [o["median_ms"] / f["median_ms"] for o, f in zip(on, off)]
        summary[workload] = {"speedup_pct": [100 * (1 - x) for x in ratios], "compile_s": (statistics.mean(r["compile_s"] for r in off), statistics.mean(r["compile_s"] for r in on)),
                             "peak_mb": ((off[0]["peak_cuda_mb"] or off[0]["peak_rss_mb"]), (on[0]["peak_cuda_mb"] or on[0]["peak_rss_mb"])),
                             "same_output": all(o["output"] == f["output"] for o, f in zip(on, off))}
        print(f"{workload}: freezing saves {summary[workload]['speedup_pct'][0]:+.1f}% / {summary[workload]['speedup_pct'][-1]:+.1f}% (two pairs); compile {summary[workload]['compile_s'][0]:.1f}s -> {summary[workload]['compile_s'][1]:.1f}s; peak {summary[workload]['peak_mb'][0]:.0f} -> {summary[workload]['peak_mb'][1]:.0f} MB; same output {summary[workload]['same_output']}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"device": a.device, "dtype": a.dtype, "results": results, "summary": summary}, open(a.out, "w"), indent=1)
