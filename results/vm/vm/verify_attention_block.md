# Switch verification: attention_block

- baseline fused kernels: 1
- rules fired in baseline: 4
- candidates: 4 fired patterns, 26 config flags, 16 opt-in options

| switch | kind | toggled | status | kernels | externs | allocation bytes | ops changed |
|---|---|---|---|---|---|---|---|
| `joint/joint_graph.early_patterns/pointless_view` | pattern | off | fires_only | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `joint/joint_graph.early_patterns/pointless_view_pair` | pattern | off | fires_only | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `joint/joint_graph.early_patterns/pointless_permute_pair` | pattern | off | fires_only | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `joint/joint_graph.patterns/_sfdp_pattern_11_inference` | pattern | off | fires_only | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `master/pattern_matcher` | config | off | changes_graph | 1->4 | 3->4 | 4,198,400->11,571,200 | <built-in function getitem> 6->5, _scaled_dot_product_flash_attention_for_cpu.default 1->0, amax.default 0->1, bmm.default 0->2, clone.default 0->4, div.Tensor 0->2 |
| `master/pre_grad_passes` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `master/post_grad_passes` | config | off | changes_graph | 1->2 | 3->3 | 4,198,400->4,198,400 | <built-in function getitem> 6->0, _scaled_dot_product_flash_attention_for_cpu.default 1->0, add.Tensor 3->0, addmm.default 2->0, mul.Tensor 2->0, permute.default 6->0 |
| `joint/constant_folding` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `joint/scatter_upon_const_tensor` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `post_grad/reorder_for_locality` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `post_grad/linear_binary_folding` | config | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `lowering/conv_1x1_as_mm` | config | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `lowering/layout_optimization` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `lowering/freezing` | config | on | changes_graph | 1->2 | 3->3 | 4,198,400->3,149,824 | addmm.default 2->0, permute.default 6->4, mkl._mkl_linear 0->2 |
| `freezing/weight_prepack` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `sched/epilogue_fusion` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `sched/prologue_fusion` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `sched/aggressive_fusion` | config | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `sched/loop_ordering_after_fusion` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `sched/inplace_buffers` | config | off | changes_graph | 1->1 | 3->3 | 4,198,400->4,200,448 |  |
| `sched/split_reductions` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `sched/reorder_for_peak_memory` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `sched/loop_index_inversion` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `sched/max_fusion_size` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `sched/unroll_reductions` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `cpp/tiling_heuristics` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `cpp/loop_tail_vec` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `cpp/horizontal_fusion` | config | off | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `cpp/decompose_tanh` | config | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `cpp/concat_linear` | config | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/decompose_mm_pass` | option | on | fires_only | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/normalization_pass` | option | on | changes_graph | 1->1 | 3->3 | 4,198,400->4,198,400 | split.Tensor 1->0, split_with_sizes.default 0->1 |
| `optimus/remove_split_with_size_one_pass` | option | on | fires_only | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/merge_splits_pass` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/merge_getitem_cat_pass` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/split_cat_pass` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/unbind_stack_pass` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/mutate_cat_pass` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/batch_linear` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/batch_layernorm` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/batch_tanh` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/batch_relu` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/batch_sigmoid` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/batch_aten_mul` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/batch_aten_add` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
| `optimus/batch_linear_post_grad` | option | on | no_effect | 1->1 | 3->3 | 4,198,400->4,198,400 |  |
