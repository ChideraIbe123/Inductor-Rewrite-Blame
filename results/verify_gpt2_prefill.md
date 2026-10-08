# Switch verification: gpt2_prefill

- baseline fused kernels: 10
- rules fired in baseline: 3
- candidates: 3 fired patterns, 26 config flags, 16 opt-in options

| switch | kind | toggled | status | kernels | externs | allocation bytes | ops changed |
|---|---|---|---|---|---|---|---|
| `joint/joint_graph.early_patterns/pointless_view` | pattern | off | changes_graph | 10->10 | 61->61 | 118,659,072->118,659,072 | reshape.default 146->148 |
| `joint/joint_graph.early_patterns/pointless_view_pair` | pattern | off | fires_only | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `joint/joint_graph.patterns/fix_iota_device` | pattern | off | fires_only | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `master/pattern_matcher` | config | off | changes_graph | 10->10 | 61->61 | 118,659,072->118,659,072 | reshape.default 146->148 |
| `master/pre_grad_passes` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `master/post_grad_passes` | config | off | changes_graph | 10->10 | 61->61 | 118,659,072->118,659,072 | <built-in function getitem> 98->0, _scaled_dot_product_flash_attention_for_cpu.default 12->0, add.Tensor 99->0, addmm.default 48->0, clone.default 24->0, embedding.default 2->0 |
| `joint/constant_folding` | config | off | changes_graph | 10->10 | 61->61 | 118,659,072->118,659,072 | add.Tensor 99->100 |
| `joint/scatter_upon_const_tensor` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `post_grad/reorder_for_locality` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `post_grad/linear_binary_folding` | config | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `lowering/conv_1x1_as_mm` | config | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `lowering/layout_optimization` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `lowering/freezing` | config | on | changes_graph | 10->10 | 61->61 | 118,659,072->118,659,072 | permute.default 49->48 |
| `freezing/weight_prepack` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `sched/epilogue_fusion` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `sched/prologue_fusion` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `sched/aggressive_fusion` | config | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `sched/loop_ordering_after_fusion` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `sched/inplace_buffers` | config | off | changes_graph | 10->9 | 61->61 | 118,659,072->126,525,440 |  |
| `sched/split_reductions` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `sched/reorder_for_peak_memory` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `sched/loop_index_inversion` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `sched/max_fusion_size` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `sched/unroll_reductions` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `cpp/tiling_heuristics` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `cpp/loop_tail_vec` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `cpp/horizontal_fusion` | config | off | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `cpp/decompose_tanh` | config | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `cpp/concat_linear` | config | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/decompose_mm_pass` | option | on | fires_only | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/normalization_pass` | option | on | changes_graph | 10->10 | 61->61 | 118,659,072->118,659,072 | split.Tensor 12->0, split_with_sizes.default 0->12 |
| `optimus/remove_split_with_size_one_pass` | option | on | fires_only | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/merge_splits_pass` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/merge_getitem_cat_pass` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/split_cat_pass` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/unbind_stack_pass` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/mutate_cat_pass` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/batch_linear` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/batch_layernorm` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/batch_tanh` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/batch_relu` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/batch_sigmoid` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/batch_aten_mul` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/batch_aten_add` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
| `optimus/batch_linear_post_grad` | option | on | no_effect | 10->10 | 61->61 | 118,659,072->118,659,072 |  |
