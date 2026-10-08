# Attribution report: norm_mlp

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- judge: timing (median_ms)
- reference (fast) median: 1.058 ms from 7 repeated runs; MAD 0.006 ms; threshold tau = 0.018 ms (k=3.0, floor 1%); max |diff| between identical runs 0.047 ms
- fast state: 1.061 ms, slow state: 1.099 ms (+0.038 ms)
- candidate changes: 8; judge evaluations: 8; wall 22 s; new measurements 7, cache hits 5

## Verdict: single

Culprit change(s):
- `-joint/constant_folding`  (alone: slow)

## Kernel-level difference (fast -> fast + culprits)
- median ms: 1.061 -> 1.085
- fused kernels: 3 -> 3
- extern/fallback calls: 2 -> 2
- intermediate allocation bytes: 1,573,376 -> 1,573,376
- static loads/stores: 13/7 -> 13/7
- kernels only in fast: cpp_fused_add_gelu_mean_mul_pow
- kernels only with culprits: cpp_fused_add_gelu_mean_mul_pow_reciprocal

## Search trace
| step | changes tried | verdict | value | reference | note |
|---|---|---|---|---|---|
| 0 |  | fast | 1.061 | 1.058 |  |
| 8 | -cpp/loop_tail_vec, -cpp/tiling_heuristics, -joint/constant_folding, -sched/epilogue_fusion, -sched/inplace_buffers, -sc | slow | 1.099 | 1.058 |  |
| 4 | -cpp/loop_tail_vec, -cpp/tiling_heuristics, -joint/constant_folding, -sched/epilogue_fusion | slow | 1.094 | 1.058 |  |
| 2 | -cpp/loop_tail_vec, -cpp/tiling_heuristics | fast | 1.051 | 1.058 |  |
| 2 | -joint/constant_folding, -sched/epilogue_fusion | fast | 1.074 | 1.058 | re-measured near threshold (second median 1.074) |
| 1 | -cpp/loop_tail_vec | fast | 1.058 | 1.058 |  |
| 1 | -cpp/tiling_heuristics | fast | 1.058 | 1.058 |  |
| 1 | -joint/constant_folding | slow | 1.085 | 1.058 |  |
