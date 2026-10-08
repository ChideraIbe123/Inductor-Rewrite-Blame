# Leave-one-out sweep: decode_mlp

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- baseline median 0.194 ms; tau 0.011 ms (5.5%); identical-run max |diff| 0.046 ms

| switch | toggled | median ms | delta ms | delta % | beyond noise | kernels | correct |
|---|---|---|---|---|---|---|---|
| `optimus/decompose_mm_pass` | on | 0.223 | +0.029 | +15.0 | SLOWER | 1 | True |
| `master/post_grad_passes` | off | 0.191 | -0.003 | -1.6 |  | 1 | True |
| `lowering/freezing` | on | 0.191 | -0.003 | -1.4 |  | 1 | True |
| `sched/inplace_buffers` | off | 0.191 | -0.002 | -1.1 |  | 1 | True |
