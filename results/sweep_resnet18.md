# Leave-one-out sweep: resnet18

- environment: `Chideras-MacBook-Pro-2|arm64|torch2.14.1|T8`
- baseline median 26.118 ms; tau 0.576 ms (2.2%); identical-run max |diff| 0.451 ms

| switch | toggled | median ms | delta ms | delta % | beyond noise | kernels | correct |
|---|---|---|---|---|---|---|---|
| `lowering/layout_optimization` | off | 29.007 | +2.889 | +11.1 | SLOWER | 12 | True |
| `lowering/freezing` | on | 25.827 | -0.291 | -1.1 |  | 11 | True |
| `sched/inplace_buffers` | off | 26.239 | +0.121 | +0.5 |  | 14 | True |
| `master/post_grad_passes` | off | 26.099 | -0.019 | -0.1 |  | 14 | True |
