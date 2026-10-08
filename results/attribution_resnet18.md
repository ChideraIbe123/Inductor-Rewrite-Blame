# Attribution report: resnet18

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- judge: timing (median_ms)
- reference (fast) median: 26.118 ms from 7 repeated runs; MAD 0.192 ms; threshold tau = 0.576 ms (k=3.0, floor 1%); max |diff| between identical runs 0.451 ms
- fast state: 26.205 ms, slow state: 29.119 ms (+2.914 ms)
- candidate changes: 6; judge evaluations: 5; wall 26 s; new measurements 4, cache hits 4

## Verdict: single

Culprit change(s):
- `-lowering/layout_optimization`  (alone: slow)

## Kernel-level difference (fast -> fast + culprits)
- median ms: 26.205 -> 29.007
- fused kernels: 14 -> 12
- extern/fallback calls: 21 -> 21
- intermediate allocation bytes: 30,052,352 -> 6,470,912
- static loads/stores: 113/42 -> 119/16
- kernels only in fast: cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu, cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu, cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu, cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu, cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu, cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu, cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu, cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu, cpp_fused_convolution
- kernels only with culprits: cpp_fused__native_batch_norm_legit_no_training_add_relu, cpp_fused__native_batch_norm_legit_no_training_add_relu, cpp_fused__native_batch_norm_legit_no_training_add_relu, cpp_fused__native_batch_norm_legit_no_training_add_relu, cpp_fused__native_batch_norm_legit_no_training_add_relu, cpp_fused__native_batch_norm_legit_no_training_add_relu, cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu

## Search trace
| step | changes tried | verdict | value | reference | note |
|---|---|---|---|---|---|
| 0 |  | fast | 26.205 | 26.118 |  |
| 6 | -cpp/tiling_heuristics, -joint/joint_graph.early_patterns/pointless_view, -lowering/layout_optimization, -master/post_gr | slow | 29.119 | 26.118 |  |
| 3 | -cpp/tiling_heuristics, -joint/joint_graph.early_patterns/pointless_view, -lowering/layout_optimization | slow | 27.499 | 26.118 |  |
| 2 | -cpp/tiling_heuristics, -joint/joint_graph.early_patterns/pointless_view | fast | 26.11 | 26.118 |  |
| 1 | -lowering/layout_optimization | slow | 29.007 | 26.118 |  |
