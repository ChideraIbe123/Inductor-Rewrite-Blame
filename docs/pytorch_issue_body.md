For a course project I built a tool that turns Inductor's rewrite rules on and off and measures the effect. The freezing flag stood out on every GPU I tested, so I measured it carefully and the write up is below.

The summary below was drafted with AI assistance from my own measurements. I ran the experiments and checked the numbers.

### What I measured" down)

**Title:** `config.freezing` makes compiled inference 7–10% faster (median) on most real networks across three GPUs and x86, but it's off by default — should it be on for eval-mode/no_grad compiles?

---

### What I measured

I measured compiled inference with freezing on and off for 163 real architectures (38 torchvision conv nets at
batch 8/32/64, 14 HF transformer architectures at prefill and decode shapes, random init) on three GPUs in bf16,
and for a 24-model corpus on an x86 Xeon and an Apple M4 Pro in fp32. Timing is the median of 50 calls per run
in a fresh process per configuration, and I only count a change if two separate runs, taken at different times,
both exceed a noise band calibrated on that machine. So "faster" below means confirmed, not a lucky run.

| machine | precision | networks | faster with freezing | slower | median gain (faster ones) | p90 | max |
|---|---|---|---|---|---|---|---|
| RTX 4090 | bf16 | 163 | 113 | 1 | 10.3% | 24.2% | 62.7% (regnet_y_400mf) |
| A40 | bf16 | 163 | 112 | 0 | 8.4% | 24.3% | 49.4% |
| L4 | bf16 | 163 | 105 | 1 | 6.7% | 18.6% | 65.2% |
| A40 | fp32 | 23 corpus models | 11 | 0 | 8.3% | – | 25.9% |
| Xeon Silver 4216 (oneDNN) | fp32 | 26 corpus models | 22 | 0 | – | – | 94.9% (decode MLP), 39% GPT-2 decode, 36% MobileNet-v3 |
| Apple M4 Pro | fp32 | 24 corpus models | 5 | 3 | 2–6% | – | ResNet-50 7% slower |

Transformers gain the most on GPU (42 of 49 faster, median 16%; decode 17%, prefill 15%); conv nets 71 of 114
(median 9%). The single "slower" case per GPU is within 3% of the noise band. On the Mac, which has no oneDNN
path, freezing is basically a no-op: 11 of 12 corpus models neutral.

### It's not a random-weights artifact

I was worried the random-init models were misleading, so I reran the 12 corpus models with their published
weights (torchvision `DEFAULT`, `bert-base-uncased`, `gpt2`, `HuggingFaceTB/SmolLM-135M`,
`Qwen/Qwen2.5-0.5B`) on the same machines. The verdicts match random init on 10 of 12 on the A5000 (the two
differences sit right on the 2% threshold), 12 of 12 on the Xeon, 12 of 12 on the Mac, and the magnitudes are
the same to within a few points (SmolLM decode 0.726 vs 0.756 of the unfrozen time, Qwen decode 0.721 vs 0.716).

### Two end-to-end workloads, not just forward passes

| workload | machine | freezing off | freezing on | peak memory | compile |
|---|---|---|---|---|---|
| SmolLM-135M `generate`, 32 new tokens, KV cache, pretrained | RTX A5000 bf16 | 216 ms | 168–199 ms (8–22% faster) | 271 → 676 MB | 125 → 153 s |
| ResNet-50 pretrained, batch 32 | RTX A5000 bf16 | 8.3 ms | 8.0 ms (3.6% faster) | 179 → 240 MB | 8.8 → 9.4 s |
| ResNet-50 pretrained, batch 32 | Xeon fp32 | 1.80 s | 1.61 s (10–11% faster) | – | – |
| SmolLM-135M `generate` | Xeon fp32 | 1.0× | 1.0× (no change) | 1.3 → 3.3 GB RSS | – |

Outputs were identical with and without freezing in every case (greedy tokens and argmax labels compared).

### What freezing actually changes in the generated code

With parameters as constants, Inductor folds weight transforms (dtype casts, transposes, BatchNorm folding) and
emits fewer external library calls. On the 4090 the transformer models drop from a median of 85 to 61 extern
calls with the same kernel count; conv nets go from 26.5 to 25.5 kernels. That lines up with where it helps:
backends where a folded constant replaces a cuBLAS/oneDNN call win, the Mac doesn't.

### The costs, which are real

- **Peak memory.** Frozen constants live next to the module's original parameters. On the A5000 that is
  +45–51% peak for SmolLM/Qwen decode and prefill (265 → 401 MB, 959 → 1406 MB), +50% for ResNet-18,
  and nothing for the attention encoders. On the Xeon the `generate` run went from 1.3 to 3.3 GB RSS.
  `config.freezing_discard_parameters` exists for exactly this; all numbers above are without it.
- **Compile time.** +22% median on the 4090 (10.5 s → 14.2 s), +10–30% per model on the A5000.

### What I'd propose

I understand why it's off: freezing assumes the weights never change after compile, which is wrong for training
and for modules that get mutated between calls. But the common inference pattern — `model.eval()`,
`torch.no_grad()`/`inference_mode()`, `torch.compile` — satisfies that assumption, and it's leaving 7–10%
on the table for most models and 20–60% for some. So, in order of ambition:

1. Document `freezing` as the recommended setting for eval-mode inference, with numbers like the ones above and
   the memory caveat.
2. Enable it automatically when the compiled module is in eval mode, compilation happens under
   no_grad/inference_mode, and no parameter requires grad — paired with `freezing_discard_parameters=True`
   or at least a warning about the memory cost.

Happy to run more configurations if that helps, or to share the full per-network tables (they're in the repo
linked below).

### Minimal reproducer (fresh process per setting)

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

On an RTX A5000 (torch 2.14.1+cu126, bf16) this gives `2.626 ms` off vs `2.272 ms` on; on the RTX 4090 the same
network measured 63% faster with freezing in the 163-network run.

### Environment

torch 2.14.1 (+cu126 on the GPUs, +cpu on the Xeon, default wheel on the Mac), triton 3.8.0, CUDA driver 570,
Ubuntu 24.04 (GPU pods), Ubuntu 26.04 (Xeon), macOS 15.

Measurement code and the per-network evidence files behind every number above (`evidence/freezing/`):
https://github.com/ChideraIbe123/Inductor-Rewrite-Blame
