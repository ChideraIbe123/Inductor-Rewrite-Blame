# Leave-one-out sweep: norm_mlp

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- baseline median 1.058 ms; tau 0.018 ms (1.7%); identical-run max |diff| 0.047 ms

| switch | toggled | median ms | delta ms | delta % | beyond noise | kernels | correct |
|---|---|---|---|---|---|---|---|
| `sched/inplace_buffers` | off | 1.103 | +0.045 | +4.3 | SLOWER | 3 | True |
| `joint/constant_folding` | off | 1.085 | +0.028 | +2.6 | SLOWER | 3 | True |
| `master/post_grad_passes` | off | 1.067 | +0.010 | +0.9 |  | 3 | True |
| `joint/joint_graph.early_patterns/pointless_view_pair` | off | 1.065 | +0.007 | +0.7 |  | 3 | True |
| `lowering/freezing` | on | 1.052 | -0.006 | -0.5 |  | 3 | True |
| `master/pattern_matcher` | off | 1.062 | +0.004 | +0.4 |  | 3 | True |
| `joint/joint_graph.patterns/pointless_convert` | off | 1.059 | +0.002 | +0.2 |  | 3 | True |
| `post_grad/post_grad.pass_patterns[1]/reciprocal_sqrt_to_rsqrt` | off | 1.058 | +0.000 | +0.0 |  | 3 | True |
