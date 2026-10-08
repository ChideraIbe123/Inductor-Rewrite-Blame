# Leave-one-out sweep: gpt2_prefill

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- baseline median 62.447 ms; tau 1.800 ms (2.9%); identical-run max |diff| 1.125 ms

| switch | toggled | median ms | delta ms | delta % | beyond noise | kernels | correct |
|---|---|---|---|---|---|---|---|
| `sched/inplace_buffers` | off | 63.725 | +1.278 | +2.0 |  | 9 | True |
| `master/post_grad_passes` | off | 61.797 | -0.650 | -1.0 |  | 10 | True |
| `lowering/freezing` | on | 62.840 | +0.393 | +0.6 |  | 10 | True |
| `optimus/normalization_pass` | on | 62.741 | +0.294 | +0.5 |  | 10 | True |
| `master/pattern_matcher` | off | 62.696 | +0.249 | +0.4 |  | 10 | True |
| `joint/constant_folding` | off | 62.683 | +0.235 | +0.4 |  | 10 | True |
| `joint/joint_graph.early_patterns/pointless_view` | off | 62.559 | +0.112 | +0.2 |  | 10 | True |
