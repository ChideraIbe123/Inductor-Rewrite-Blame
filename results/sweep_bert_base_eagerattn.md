# Leave-one-out sweep: bert_base_eagerattn

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- baseline median 84.363 ms; tau 2.905 ms (3.4%); identical-run max |diff| 1.708 ms

| switch | toggled | median ms | delta ms | delta % | beyond noise | kernels | correct |
|---|---|---|---|---|---|---|---|
| `joint/joint_graph.patterns/_sfdp_pattern_28_inference` | off | 88.183 | +3.820 | +4.5 | SLOWER | 7 | True |
| `master/pattern_matcher` | off | 87.762 | +3.400 | +4.0 | SLOWER | 7 | True |
| `master/post_grad_passes` | off | 84.110 | -0.253 | -0.3 |  | 5 | True |
| `lowering/freezing` | on | 84.159 | -0.204 | -0.2 |  | 5 | True |
| `sched/inplace_buffers` | off | 84.197 | -0.166 | -0.2 |  | 5 | True |
