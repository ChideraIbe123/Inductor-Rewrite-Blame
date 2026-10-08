# Attribution report: attention_block

- environment: `fa26-cs598a-010|x86_64|torch2.14.1+cpu|T4`
- judge: timing (median_ms)
- reference (fast) median: 7.650 ms from 7 repeated runs; MAD 0.054 ms; threshold tau = 0.162 ms (k=3.0, floor 1%); max |diff| between identical runs 1.692 ms
- fast state: 7.621 ms, slow state: 8.265 ms (+0.643 ms)
- candidate changes: 63; judge evaluations: 8; wall 145 s; new measurements 14, cache hits 4

## Verdict: single

Culprit change(s):
- `+joint/joint_graph.patterns/_sfdp_pattern_11_inference`  (alone: slow)

## Kernel-level difference (fast -> fast + culprits)
- median ms: 7.621 -> 8.326
- fused kernels: 4 -> 1
- extern/fallback calls: 4 -> 3
- intermediate allocation bytes: 11,591,680 -> 4,198,400
- static loads/stores: 19/14 -> 12/4
- kernels only in fast: cpp_fused__softmax_abs_all_amax_div_eq_mul_ne_sub_view, cpp_fused_clone_split_transpose_view, cpp_fused_clone_transpose_view
- externs only in fast: extern_kernels.bmm, extern_kernels.bmm
- externs only with culprits: torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
- rule firings changed: joint/joint_graph.pass_patterns[1]/div_softmax_pattern: 1->0, joint/joint_graph.patterns/_sfdp_pattern_11_inference: 0->1, joint/joint_graph.patterns/bmm_to_mm: 2->0

## Search trace
| step | changes tried | verdict | value | reference | note |
|---|---|---|---|---|---|
| 0 |  | fast | 7.621 | 7.65 |  |
| 63 | +joint/joint_graph.patterns/_sfdp_pattern_11_inference, +joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, +jo | slow | 8.265 | 7.65 |  |
| 32 | +joint/joint_graph.patterns/_sfdp_pattern_11_inference, +joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, +jo | slow | 8.353 | 7.65 |  |
| 16 | +joint/joint_graph.patterns/_sfdp_pattern_11_inference, +joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, +jo | slow | 8.318 | 7.65 |  |
| 8 | +joint/joint_graph.patterns/_sfdp_pattern_11_inference, +joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, +jo | slow | 10.789 | 7.65 |  |
| 4 | +joint/joint_graph.patterns/_sfdp_pattern_11_inference, +joint/joint_graph.patterns/_sfdp_pattern_12_half_inference, +jo | slow | 9.658 | 7.65 |  |
| 2 | +joint/joint_graph.patterns/_sfdp_pattern_11_inference, +joint/joint_graph.patterns/_sfdp_pattern_12_half_inference | slow | 8.246 | 7.65 |  |
| 1 | +joint/joint_graph.patterns/_sfdp_pattern_11_inference | slow | 8.326 | 7.65 |  |
