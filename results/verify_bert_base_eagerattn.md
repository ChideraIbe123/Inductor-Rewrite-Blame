# Switch verification: bert_base_eagerattn

- baseline fused kernels: 5
- rules fired in baseline: 4
- candidates: 4 fired patterns, 26 config flags, 16 opt-in options

| switch | kind | toggled | status | kernels | externs | allocation bytes | ops changed |
|---|---|---|---|---|---|---|---|
| `joint/joint_graph.early_patterns/pointless_view` | pattern | off | fires_only | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `joint/joint_graph.early_patterns/pointless_view_pair` | pattern | off | fires_only | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `joint/joint_graph.early_patterns/pointless_permute_pair` | pattern | off | fires_only | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `joint/joint_graph.patterns/_sfdp_pattern_28_inference` | pattern | off | changes_graph | 5->7 | 84->96 | 383,885,312->499,048,448 | <built-in function getitem> 62->50, _scaled_dot_product_flash_attention_for_cpu.default 12->0, abs.default 0->12, amax.default 0->24, any.dims 0->12, bmm.default 0->24 |
| `master/pattern_matcher` | config | off | changes_graph | 5->7 | 84->96 | 383,885,312->498,311,168 | <built-in function getitem> 62->50, _scaled_dot_product_flash_attention_for_cpu.default 12->0, amax.default 0->12, bmm.default 0->24, div.Tensor 0->12, exp.default 0->12 |
| `master/pre_grad_passes` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `master/post_grad_passes` | config | off | changes_graph | 5->5 | 84->84 | 383,885,312->383,885,312 | <built-in function getitem> 62->0, _scaled_dot_product_flash_attention_for_cpu.default 12->0, add.Tensor 88->0, addmm.default 72->0, clone.default 48->0, embedding.default 3->0 |
| `joint/constant_folding` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `joint/scatter_upon_const_tensor` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `post_grad/reorder_for_locality` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `post_grad/linear_binary_folding` | config | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `lowering/conv_1x1_as_mm` | config | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `lowering/layout_optimization` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `lowering/freezing` | config | on | changes_graph | 5->5 | 84->84 | 383,885,312->383,885,312 | embedding.default 3->1, expand.default 2->0, gather.default 1->0, permute.default 120->48, reshape.default 192->168, slice.Tensor 1->0 |
| `freezing/weight_prepack` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `sched/epilogue_fusion` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `sched/prologue_fusion` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `sched/aggressive_fusion` | config | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `sched/loop_ordering_after_fusion` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `sched/inplace_buffers` | config | off | changes_graph | 5->5 | 84->84 | 383,885,312->538,079,232 |  |
| `sched/split_reductions` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `sched/reorder_for_peak_memory` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `sched/loop_index_inversion` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `sched/max_fusion_size` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `sched/unroll_reductions` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `cpp/tiling_heuristics` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `cpp/loop_tail_vec` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `cpp/horizontal_fusion` | config | off | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `cpp/decompose_tanh` | config | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `cpp/concat_linear` | config | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/decompose_mm_pass` | option | on | fires_only | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/normalization_pass` | option | on | fires_only | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/remove_split_with_size_one_pass` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/merge_splits_pass` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/merge_getitem_cat_pass` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/split_cat_pass` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/unbind_stack_pass` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/mutate_cat_pass` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/batch_linear` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/batch_layernorm` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/batch_tanh` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/batch_relu` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/batch_sigmoid` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/batch_aten_mul` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/batch_aten_add` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
| `optimus/batch_linear_post_grad` | option | on | no_effect | 5->5 | 84->84 | 383,885,312->383,885,312 |  |
