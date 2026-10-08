# Attribution report: decode_mlp

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- judge: timing (median_ms)
- reference (fast) median: 0.194 ms from 7 repeated runs; MAD 0.004 ms; threshold tau = 0.011 ms (k=3.0, floor 1%); max |diff| between identical runs 0.046 ms
- fast state: 0.195 ms, slow state: 0.222 ms (+0.027 ms)
- candidate changes: 16; judge evaluations: 7; wall 17 s; new measurements 6, cache hits 4

## Verdict: single

Culprit change(s):
- `+optimus/decompose_mm_pass`  (alone: slow)

## Kernel-level difference (fast -> fast + culprits)
- median ms: 0.195 -> 0.223
- fused kernels: 1 -> 1
- extern/fallback calls: 3 -> 0
- intermediate allocation bytes: 20,480 -> 20,480
- static loads/stores: 2/1 -> 8/4
- kernels only in fast: cpp_fused_mul_silu
- kernels only with culprits: cpp_fused_mm_mul_silu_t
- externs only in fast: extern_kernels.mm, extern_kernels.mm, extern_kernels.mm
- rule firings changed: post_grad/post_grad.POST_GRAD_PATTERNS[decompose_mm_pass]/decompose_mm: 0->3

## Search trace
| step | changes tried | verdict | value | reference | note |
|---|---|---|---|---|---|
| 0 |  | fast | 0.195 | 0.194 |  |
| 16 | +optimus/batch_aten_add, +optimus/batch_aten_mul, +optimus/batch_layernorm, +optimus/batch_linear, +optimus/batch_linear | slow | 0.222 | 0.194 |  |
| 8 | +optimus/batch_aten_add, +optimus/batch_aten_mul, +optimus/batch_layernorm, +optimus/batch_linear, +optimus/batch_linear | fast | 0.199 | 0.194 |  |
| 8 | +optimus/decompose_mm_pass, +optimus/merge_getitem_cat_pass, +optimus/merge_splits_pass, +optimus/mutate_cat_pass, +opti | slow | 0.221 | 0.194 |  |
| 4 | +optimus/decompose_mm_pass, +optimus/merge_getitem_cat_pass, +optimus/merge_splits_pass, +optimus/mutate_cat_pass | slow | 0.221 | 0.194 |  |
| 2 | +optimus/decompose_mm_pass, +optimus/merge_getitem_cat_pass | slow | 0.236 | 0.194 |  |
| 1 | +optimus/decompose_mm_pass | slow | 0.223 | 0.194 |  |
