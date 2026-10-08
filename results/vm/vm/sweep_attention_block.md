# Leave-one-out sweep: attention_block

- environment: `fa26-cs598a-010|x86_64|torch2.14.1+cpu|T4`
- baseline median 8.526 ms; tau 0.198 ms (2.3%); identical-run max |diff| 0.320 ms

| switch | toggled | median ms | delta ms | delta % | beyond noise | kernels | correct |
|---|---|---|---|---|---|---|---|
| `master/pattern_matcher` | off | 7.619 | -0.907 | -10.6 | faster | 4 | True |
| `lowering/freezing` | on | 8.078 | -0.448 | -5.3 | faster | 2 | True |
| `sched/inplace_buffers` | off | 8.244 | -0.282 | -3.3 | faster | 1 | True |
| `optimus/normalization_pass` | on | 8.292 | -0.234 | -2.7 | faster | 1 | True |
| `master/post_grad_passes` | off | 8.418 | -0.108 | -1.3 |  | 2 | True |
