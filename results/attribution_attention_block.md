# Attribution report: attention_block

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- judge: timing (median_ms)
- reference (fast) median: 0.926 ms from 7 repeated runs; MAD 0.006 ms; threshold tau = 0.017 ms (k=3.0, floor 1%); max |diff| between identical runs 0.029 ms
- fast state: 0.919 ms, slow state: 1.250 ms (+0.331 ms)
- candidate changes: 63; judge evaluations: 16; wall 49 s; new measurements 16, cache hits 4

## Verdict: interaction
- note: 2 rewrites only hurt together (no single one reproduces)

Culprit change(s):
- `-joint/joint_graph.patterns/_sfdp_pattern_11_inference`  (alone: fast)
- `-joint/joint_graph.patterns/_sfdp_pattern_12_inference`  (alone: fast)

## Kernel-level difference (fast -> fast + culprits)
- median ms: 0.919 -> 1.244
- fused kernels: 1 -> 4
- extern/fallback calls: 3 -> 4
- intermediate allocation bytes: 4,198,400 -> 11,591,680
- static loads/stores: 12/4 -> 19/14
- kernels only with culprits: cpp_fused__softmax_abs_all_amax_div_eq_mul_ne_sub_view, cpp_fused_clone_split_transpose_view, cpp_fused_clone_transpose_view
- externs only in fast: torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
- externs only with culprits: extern_kernels.bmm, extern_kernels.bmm
- rule firings changed: joint/joint_graph.pass_patterns[1]/div_softmax_pattern: 0->1, joint/joint_graph.patterns/_sfdp_pattern_11_inference: 1->0, joint/joint_graph.patterns/bmm_to_mm: 0->2
- rules suppressed by the culprit state: joint/joint_graph.patterns/_sfdp_pattern_11_inference x1, joint/joint_graph.patterns/_sfdp_pattern_12_inference x1

## Search trace
| step | changes tried | verdict | value | reference | note |
|---|---|---|---|---|---|
| 0 |  | fast | 0.919 | 0.926 |  |
| 63 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference, -joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, -jo | slow | 1.25 | 0.926 |  |
| 32 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference, -joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, -jo | slow | 1.25 | 0.926 |  |
| 16 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference, -joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, -jo | slow | 1.238 | 0.926 |  |
| 8 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference, -joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, -jo | slow | 1.249 | 0.926 |  |
| 4 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference, -joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, -jo | slow | 1.243 | 0.926 |  |
| 2 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference, -joint/joint_graph.patterns/_sfdp_pattern_12_half_inference | fast | 0.926 | 0.926 |  |
| 2 | -joint/joint_graph.patterns/_sfdp_pattern_12_inference, -joint/joint_graph.patterns/_sfdp_pattern_13_half_inference | fast | 0.933 | 0.926 |  |
| 1 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference | fast | 0.931 | 0.926 |  |
| 1 | -joint/joint_graph.patterns/_sfdp_pattern_12_half_inference | fast | 0.93 | 0.926 |  |
| 1 | -joint/joint_graph.patterns/_sfdp_pattern_12_inference | fast | 0.922 | 0.926 |  |
| 1 | -joint/joint_graph.patterns/_sfdp_pattern_13_half_inference | fast | 0.936 | 0.926 | re-measured near threshold (second median 0.936) |
| 3 | -joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, -joint/joint_graph.patterns/_sfdp_pattern_12_inference, -jo | fast | 0.927 | 0.926 |  |
| 3 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference, -joint/joint_graph.patterns/_sfdp_pattern_12_inference, -joint/j | slow | 1.237 | 0.926 |  |
| 2 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference, -joint/joint_graph.patterns/_sfdp_pattern_13_half_inference | fast | 0.922 | 0.926 |  |
| 2 | -joint/joint_graph.patterns/_sfdp_pattern_11_inference, -joint/joint_graph.patterns/_sfdp_pattern_12_inference | slow | 1.244 | 0.926 |  |
