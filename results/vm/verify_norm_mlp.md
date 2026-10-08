# Switch verification: norm_mlp

- baseline fused kernels: 3
- rules fired in baseline: 4
- candidates: 4 fired patterns, 26 config flags, 16 opt-in options

| switch | kind | toggled | status | kernels | externs | allocation bytes | ops changed |
|---|---|---|---|---|---|---|---|
| `joint/joint_graph.early_patterns/pointless_view` | pattern | off | fires_only | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `joint/joint_graph.early_patterns/pointless_view_pair` | pattern | off | changes_graph | 3->3 | 2->2 | 1,573,376->1,573,376 | reshape.default 0->2 |
| `joint/joint_graph.patterns/pointless_convert` | pattern | off | changes_graph | 3->3 | 2->2 | 1,573,376->1,573,376 | prims.convert_element_type.default 0->2 |
| `post_grad/post_grad.pass_patterns[1]/reciprocal_sqrt_to_rsqrt` | pattern | off | changes_graph | 3->3 | 2->2 | 1,573,376->1,573,376 | reciprocal.default 0->1, rsqrt.default 2->1, sqrt.default 0->1 |
| `master/pattern_matcher` | config | off | changes_graph | 3->3 | 2->2 | 1,573,376->1,573,376 | reciprocal.default 0->1, reshape.default 0->2, rsqrt.default 2->1, sqrt.default 0->1, prims.convert_element_type.default 0->2 |
| `master/pre_grad_passes` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `master/post_grad_passes` | config | off | changes_graph | 3->3 | 2->2 | 1,573,376->1,573,376 | <built-in function getitem> 2->0, add.Tensor 5->0, addmm.default 2->0, erf.default 1->0, mean.dim 1->0, mul.Tensor 6->0 |
| `joint/constant_folding` | config | off | changes_graph | 3->3 | 2->2 | 1,573,376->1,573,376 | mul.Tensor 6->7 |
| `joint/scatter_upon_const_tensor` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `post_grad/reorder_for_locality` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `post_grad/linear_binary_folding` | config | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `lowering/conv_1x1_as_mm` | config | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `lowering/layout_optimization` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `lowering/freezing` | config | on | changes_graph | 3->3 | 2->2 | 1,573,376->262,656 | addmm.default 2->0, permute.default 2->0, mkl._mkl_linear 0->2 |
| `freezing/weight_prepack` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `sched/epilogue_fusion` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `sched/prologue_fusion` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `sched/aggressive_fusion` | config | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `sched/loop_ordering_after_fusion` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `sched/inplace_buffers` | config | off | changes_graph | 3->3 | 2->2 | 1,573,376->2,884,352 |  |
| `sched/split_reductions` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `sched/reorder_for_peak_memory` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `sched/loop_index_inversion` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `sched/max_fusion_size` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `sched/unroll_reductions` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `cpp/tiling_heuristics` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `cpp/loop_tail_vec` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `cpp/horizontal_fusion` | config | off | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `cpp/decompose_tanh` | config | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `cpp/concat_linear` | config | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/decompose_mm_pass` | option | on | fires_only | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/normalization_pass` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/remove_split_with_size_one_pass` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/merge_splits_pass` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/merge_getitem_cat_pass` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/split_cat_pass` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/unbind_stack_pass` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/mutate_cat_pass` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/batch_linear` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/batch_layernorm` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/batch_tanh` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/batch_relu` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/batch_sigmoid` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/batch_aten_mul` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/batch_aten_add` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
| `optimus/batch_linear_post_grad` | option | on | no_effect | 3->3 | 2->2 | 1,573,376->1,573,376 |  |
