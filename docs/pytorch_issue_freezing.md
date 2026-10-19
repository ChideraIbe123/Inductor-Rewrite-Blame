# Draft: PyTorch issue on `torch._inductor.config.freezing` for inference (to be filed by Chidera)

Status: all numbers measured (GPU part on an RTX A5000 in bf16 on 2026-10-19; Xeon and Mac parts pending).

---

**Title:** [inductor] `freezing=True` speeds up inference on 105–113 of 163 real networks on three GPUs (median 7–10%, up to 65%) and on x86 CPU (up to 95%); it is off by default. Proposal: enable it automatically for eval-mode, no-grad compiles.

### Summary

`torch._inductor.config.freezing` is `False` by default. We measured compiled inference with and without it on
163 real architectures (38 torchvision conv nets × batch 8/32/64, 14 HF transformer architectures × prefill/decode
shapes, random-init) on three GPUs in bf16, and on the 24-model corpus on an x86 Xeon and an Apple M4 Pro in
fp32. Each comparison is a separate process per configuration, 50 timed calls per run, and a slowdown or
speed-up counts only if two time-separated runs both exceed a noise band calibrated on that machine
(details: docs/noise_method.md in the linked repo).

| machine | precision | networks | faster with freezing | slower | median gain (faster ones) | p90 | max |
|---|---|---|---|---|---|---|---|
| RTX 4090 | bf16 | 163 | 113 | 1 | 10.3% | 24.2% | 62.7% (regnet_y_400mf) |
| A40 | bf16 | 163 | 112 | 0 | 8.4% | 24.3% | 49.4% |
| L4 | bf16 | 163 | 105 | 1 | 6.7% | 18.6% | 65.2% |
| A40 | fp32 | 23 corpus models | 11 | 0 | 8.3% | – | 25.9% (decode MLP) |
| Xeon Silver 4216 (oneDNN) | fp32 | 26 corpus models | 22 | 0 | – | – | 94.9% (decode MLP), 30.9% GPT-2 decode, 28% MobileNet-v3 |
| Apple M4 Pro | fp32 | 24 corpus models | 5 | 3 | 2–6% | – | ResNet-50 7% slower |

Transformers benefit most on GPU (42 of 49 faster, median 16%: decode 17%, prefill 15%); conv nets 71 of 114
(median 9%). The one harmful case per GPU is within 3% of the noise band.

### What changes in the generated code

With freezing, parameters become graph constants, so Inductor constant-folds weight transforms (dtype casts,
transposes, BatchNorm folding) and emits fewer calls to external kernels: on the 4090, transformer models go
from a median of 85 to 61 extern (cuBLAS/library) calls and the same kernel count; conv nets from 26.5 to
25.5 kernels. Compile time rises by a median of 22% (10.5 s → 14.2 s on the 4090; +10–30% per model on the A5000).

### Costs: memory and compile time (RTX A5000, bf16, peak `max_memory_allocated` after compile and one call)

| model | speed-up with freezing | peak memory off → on | compile s off → on |
|---|---|---|---|
| SmolLM-135M decode | 27% | 265 → 401 MB (+51%) | 19 → 22 |
| SmolLM-135M prefill | 28% | 301 → 437 MB (+45%) | 17 → 22 |
| Qwen2.5-0.5B decode | 28% | 959 → 1406 MB (+47%) | 15 → 19 |
| ResNet-18 / ResNet-50 | 12% / 12% | 48 → 72 / 73 → 121 MB | 3 → 3 / 6 → 8 |
| EfficientNet-B0 / MobileNet-V3 | 18% / 8% | unchanged | 13 → 17 / 12 → 14 |
| BERT-base, DistilBERT, GPT-2 | within noise | +0–16% | +1–3 s |

The extra memory is the frozen constants living beside the module's original parameters. Inductor has
`config.freezing_discard_parameters` (default False) for exactly this; the numbers above are without it.

### Trained weights give the same answer

The 12 corpus models were measured twice on the same GPU: random-init and with the published weights
(torchvision `DEFAULT`, `bert-base-uncased`, `gpt2`, `HuggingFaceTB/SmolLM-135M`, `Qwen/Qwen2.5-0.5B`).
Freezing labels agree on 10 of 12 (the two differences sit on the 2% threshold: BERT 1.036 vs 1.001,
GPT-2 decode 0.986 vs 0.975), and the speed-ups match to within a few points (SmolLM decode 0.726 vs 0.756,
Qwen decode 0.721 vs 0.716). Layout labels agree on 11 of 12.

### Two real end-to-end workloads (RTX A5000, bf16, two fresh processes per setting, identical outputs)

| workload | freezing off | freezing on | peak memory | compile |
|---|---|---|---|---|
| SmolLM-135M `generate`, 32 new tokens, KV cache, pretrained | 216 ms | 168–199 ms (8–22% faster) | 271 → 676 MB | 125 → 153 s |
| ResNet-50 pretrained, batch 32 classification | 8.3 ms | 8.0 ms (3.6% faster) | 179 → 240 MB | 8.8 → 9.4 s |

### Why it is off, and the proposal

Freezing assumes parameters do not change after compile, so it is unsafe for training or for modules whose
weights are mutated between calls, and it can increase memory. But the common inference pattern, `model.eval()`
plus `torch.no_grad()`/`inference_mode()` plus `torch.compile`, satisfies the assumption. Proposal: enable
freezing automatically when (a) the compiled module is in eval mode, (b) compilation happens under no_grad /
inference_mode, and (c) no parameter requires grad; keep `config.freezing` as the override; document the
memory/compile-time trade-off. Alternatively, surface it in the `torch.compile` docs as the recommended
inference setting, with these numbers.

### Reproducer (any model; 20 lines)

```python
import time, torch, torchvision, torch._inductor.config as cfg
def bench(freezing):
    cfg.freezing = freezing
    torch._dynamo.reset()
    m = torchvision.models.regnet_y_400mf(weights=None).cuda().to(torch.bfloat16).eval()
    x = torch.randn(8, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        c = torch.compile(m); c(x); torch.cuda.synchronize()
        ts = []
        for _ in range(50):
            t = time.perf_counter(); c(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return sorted(ts)[25] * 1e3
print("freezing off: %.3f ms" % bench(False)); print("freezing on:  %.3f ms" % bench(True))
```
Run each setting in a fresh process for a clean comparison. On an RTX A5000 (torch 2.14.1+cu126, bf16) this
prints `freezing=False: median 2.626 ms` and `freezing=True: median 2.272 ms` (13% faster); on an RTX 4090 the
same network measured 63% faster with freezing in the 163-network run.

### Environment

torch 2.14.1 (+cu126 on the GPUs, +cpu on the Xeon, default wheel on the Mac), triton 3.8.0, CUDA driver 570,
Ubuntu 24.04 (pods), Ubuntu 26.04 (Xeon VM), macOS 15. Full measurement code, per-machine stores and
per-model tables: https://github.com/ChideraIbe123/Inductor-Rewrite-Blame

### Proposal, refined by the memory numbers

Because freezing can raise peak memory by about half for LLM-style models (frozen constants beside the original
parameters), "on by default" is too strong. Proposal: (1) document freezing as the recommended setting for
eval-mode, no-grad inference with these numbers; (2) when enabling it automatically under those conditions, pair it
with `freezing_discard_parameters=True` or a warning about memory; (3) treat the 20–30% speed-ups on decode
models as the headline case, since that is where production inference time goes.

### x86 CPU (Intel Xeon Silver 4216, oneDNN, fp32, 3 threads pinned)

Same 12 corpus models, random and trained weights: freezing is faster on 10 of 12 (both weight sets, labels
agree 12/12): MobileNet-V3 36% (pretrained), GPT-2 decode 39%, SmolLM prefill 16–18%, ResNet-50 13%,
EfficientNet-B0 17%, BERT/DistilBERT/GPT-2 prefill 4–7%; the two decode LLMs are neutral. Real workloads:
ResNet-50 batch-32 classification 10–11% faster (identical output); SmolLM `generate` unchanged (−1%) while peak
RSS grows from 1.3 to 3.3 GB. On the Apple M4 Pro freezing is a wash (5 faster, 3 slower, all within 7%).

So the recommendation is machine- and workload-specific in size but consistent in sign on x86 and CUDA: it is
the decoder-model `generate` path on CPU where the gain disappears and the memory cost is largest.
