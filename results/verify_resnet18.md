# Switch verification: resnet18

- baseline fused kernels: 14
- rules fired in baseline: 1
- candidates: 1 fired patterns, 26 config flags, 16 opt-in options

| switch | kind | toggled | status | kernels | externs | allocation bytes | ops changed |
|---|---|---|---|---|---|---|---|
| `joint/joint_graph.early_patterns/pointless_view` | pattern | off | fires_only | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `master/pattern_matcher` | config | off | fires_only | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `master/pre_grad_passes` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `master/post_grad_passes` | config | off | changes_graph | 14->14 | 21->21 | 30,052,352->30,052,352 | <built-in function getitem> 1->0, add.Tensor 48->0, addmm.default 1->0, convolution.default 20->0, mean.dim 1->0, mul.Tensor 40->0 |
| `joint/constant_folding` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `joint/scatter_upon_const_tensor` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `post_grad/reorder_for_locality` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `post_grad/linear_binary_folding` | config | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `lowering/conv_1x1_as_mm` | config | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `lowering/layout_optimization` | config | off | changes_graph | 14->12 | 21->21 | 30,052,352->6,470,912 |  |
| `lowering/freezing` | config | on | changes_graph | 14->11 | 21->21 | 30,052,352->11,287,808 | add.Tensor 48->8, mul.Tensor 40->0, permute.default 1->0, rsqrt.default 20->0, sub.Tensor 20->0, unsqueeze.default 160->0 |
| `freezing/weight_prepack` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `sched/epilogue_fusion` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `sched/prologue_fusion` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `sched/aggressive_fusion` | config | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `sched/loop_ordering_after_fusion` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `sched/inplace_buffers` | config | off | changes_graph | 14->14 | 21->21 | 30,052,352->67,801,088 |  |
| `sched/split_reductions` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `sched/reorder_for_peak_memory` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `sched/loop_index_inversion` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `sched/max_fusion_size` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `sched/unroll_reductions` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `cpp/tiling_heuristics` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `cpp/loop_tail_vec` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `cpp/horizontal_fusion` | config | off | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `cpp/decompose_tanh` | config | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `cpp/concat_linear` | config | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/decompose_mm_pass` | option | on | fires_only | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/normalization_pass` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/remove_split_with_size_one_pass` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/merge_splits_pass` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/merge_getitem_cat_pass` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/split_cat_pass` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/unbind_stack_pass` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/mutate_cat_pass` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/batch_linear` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/batch_layernorm` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/batch_tanh` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/batch_relu` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/batch_sigmoid` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/batch_aten_mul` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/batch_aten_add` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
| `optimus/batch_linear_post_grad` | option | on | no_effect | 14->14 | 21->21 | 30,052,352->30,052,352 |  |
