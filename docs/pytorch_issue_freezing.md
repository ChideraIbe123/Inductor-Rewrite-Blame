# Draft: PyTorch issue on `torch._inductor.config.freezing` for inference (to be filed by Chidera)

Status: numbers below are measured; items marked TODO are being collected (pretrained weights, peak memory,
real end-to-end workloads). File once those are in.

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
25.5 kernels. Compile time rises by a median of 22% (10.5 s → 14.2 s on the 4090). Peak memory: TODO (being
measured; freezing can duplicate folded weights).

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
Run each setting in a fresh process for a clean comparison. On an RTX 4090 (torch 2.14.1+cu126) this prints
roughly 1.9 ms vs 0.7 ms (TODO: paste exact numbers from the reproducer run).

### Environment

torch 2.14.1 (+cu126 on the GPUs, +cpu on the Xeon, default wheel on the Mac), triton 3.8.0, CUDA driver 570,
Ubuntu 24.04 (pods), Ubuntu 26.04 (Xeon VM), macOS 15. Full measurement code, per-machine stores and
per-model tables: https://github.com/ChideraIbe123/Inductor-Rewrite-Blame

### TODO before filing
- [ ] pretrained-weight twins of 12 corpus models on Mac/VM/GPU (same shapes): do gains persist?
- [ ] peak memory with/without freezing per model (CPU RSS, CUDA max_memory_allocated)
- [ ] two end-to-end workloads: SmolLM-135M generate (32 tokens, KV cache) and ResNet-50 batch-32 classification
- [ ] exact numbers from the 20-line reproducer on one GPU
