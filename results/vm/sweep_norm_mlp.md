# Leave-one-out sweep: norm_mlp

- environment: `fa26-cs598a-010|x86_64|torch2.14.1+cpu|T4`
- baseline median 12.478 ms; tau 0.155 ms (1.2%); identical-run max |diff| 1.660 ms

| switch | toggled | median ms | delta ms | delta % | beyond noise | kernels | correct |
|---|---|---|---|---|---|---|---|
| `lowering/freezing` | on | 6.664 | -5.814 | -46.6 | faster | 3 | True |
| `joint/joint_graph.early_patterns/pointless_view_pair` | off | 12.369 | -0.109 | -0.9 |  | 3 | True |
| `master/post_grad_passes` | off | 12.391 | -0.086 | -0.7 |  | 3 | True |
| `sched/inplace_buffers` | off | 12.552 | +0.074 | +0.6 |  | 3 | True |
| `joint/constant_folding` | off | 12.418 | -0.060 | -0.5 |  | 3 | True |
| `master/pattern_matcher` | off | 12.507 | +0.029 | +0.2 |  | 3 | True |
| `joint/joint_graph.patterns/pointless_convert` | off | 12.462 | -0.016 | -0.1 |  | 3 | True |
| `post_grad/post_grad.pass_patterns[1]/reciprocal_sqrt_to_rsqrt` | off | 12.487 | +0.009 | +0.1 |  | 3 | True |
