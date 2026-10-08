# Switch verification: decode_mlp

- baseline fused kernels: 1
- rules fired in baseline: 0
- candidates: 0 fired patterns, 26 config flags, 16 opt-in options

| switch | kind | toggled | status | kernels | externs | allocation bytes | ops changed |
|---|---|---|---|---|---|---|---|
| `master/pattern_matcher` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `master/pre_grad_passes` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `master/post_grad_passes` | config | off | changes_graph | 1->1 | 3->3 | 20,480->20,480 | add.Tensor 1->0, div.Tensor 1->0, exp.default 1->0, mm.default 3->0, mul.Tensor 1->0, neg.default 1->0 |
| `joint/constant_folding` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `joint/scatter_upon_const_tensor` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `post_grad/reorder_for_locality` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `post_grad/linear_binary_folding` | config | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `lowering/conv_1x1_as_mm` | config | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `lowering/layout_optimization` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `lowering/freezing` | config | on | changes_graph | 1->1 | 3->3 | 20,480->20,480 | permute.default 3->0 |
| `freezing/weight_prepack` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `sched/epilogue_fusion` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `sched/prologue_fusion` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `sched/aggressive_fusion` | config | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `sched/loop_ordering_after_fusion` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `sched/inplace_buffers` | config | off | changes_graph | 1->1 | 3->3 | 20,480->28,672 |  |
| `sched/split_reductions` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `sched/reorder_for_peak_memory` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `sched/loop_index_inversion` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `sched/max_fusion_size` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `sched/unroll_reductions` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `cpp/tiling_heuristics` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `cpp/loop_tail_vec` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `cpp/horizontal_fusion` | config | off | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `cpp/decompose_tanh` | config | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `cpp/concat_linear` | config | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/decompose_mm_pass` | option | on | changes_graph | 1->1 | 3->0 | 20,480->20,480 | mm.default 3->0, mul.Tensor 1->4, sum.dim_IntList 0->3, unsqueeze.default 0->6 |
| `optimus/normalization_pass` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/remove_split_with_size_one_pass` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/merge_splits_pass` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/merge_getitem_cat_pass` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/split_cat_pass` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/unbind_stack_pass` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/mutate_cat_pass` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/batch_linear` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/batch_layernorm` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/batch_tanh` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/batch_relu` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/batch_sigmoid` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/batch_aten_mul` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/batch_aten_add` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
| `optimus/batch_linear_post_grad` | option | on | no_effect | 1->1 | 3->3 | 20,480->20,480 |  |
