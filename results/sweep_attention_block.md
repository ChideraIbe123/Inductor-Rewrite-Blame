# Leave-one-out sweep: attention_block

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- baseline median 0.926 ms; tau 0.017 ms (1.8%); identical-run max |diff| 0.029 ms

| switch | toggled | median ms | delta ms | delta % | beyond noise | kernels | correct |
|---|---|---|---|---|---|---|---|
| `master/pattern_matcher` | off | 1.232 | +0.306 | +33.1 | SLOWER | 4 | True |
| `master/post_grad_passes` | off | 0.955 | +0.030 | +3.2 | SLOWER | 2 | True |
| `optimus/normalization_pass` | on | 0.916 | -0.010 | -1.1 |  | 1 | True |
| `sched/inplace_buffers` | off | 0.928 | +0.003 | +0.3 |  | 1 | True |
| `lowering/freezing` | on | 0.925 | -0.000 | -0.0 |  | 1 | True |
