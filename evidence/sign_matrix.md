| switch (ON vs OFF) | Mac M4 (24 models) | Xeon (26 models) | A40 fp32 (23 models) | A40 bf16 (21 models) | L4 bf16 (21 models) | 4090 bf16 (21 models) |
|---|---|---|---|---|---|---|
| `freezing` | hurts 1, helps 1 (7%) | helps 22 (95%) | helps 11 (26%) | helps 14 (39%) | helps 12 (37%) | helps 19 (49%) |
| `max_fusion_size` | helps 2 (7%) | hurts 1 (5%) | hurts 1, helps 7 (9%) | helps 13 (15%) | hurts 1, helps 12 (15%) | hurts 1, helps 11 (21%) |
| `layout_optimization` | hurts 4, helps 1 (40%) | helps 7 (90%) | hurts 2, helps 1 (24%) | helps 2 (22%) | hurts 1, helps 3 (26%) | hurts 1, helps 3 (9%) |
| `inplace_buffers` | · | hurts 2, helps 11 (283%) | hurts 1, helps 1 (17%) | · | helps 3 (19%) | helps 2 (22%) |
| `pattern_matcher` | helps 3 (34%) | hurts 1 (12%) | helps 3 (28%) | · | helps 2 (7%) | · |
| `post_grad_passes` | hurts 1, helps 2 (17%) | hurts 3, helps 2 (283%) | helps 1 (11%) | · | · | · |
| `conv_1x1_as_mm` | helps 2 (6%) | helps 2 (20%) | hurts 1, helps 1 (24%) | · | · | helps 2 (9%) |
| `constant_folding` | · | hurts 1, helps 1 (10%) | · | helps 3 (25%) | helps 1 (44%) | helps 1 (48%) |
| `normalization_pass` | hurts 1 (16%) | hurts 3 (217%) | helps 1 (11%) | · | · | · |
| `decompose_mm_pass` | hurts 3 (135%) | helps 2 (95%) | · | · | · | · |
| `reorder_for_locality` | hurts 1 (4%) | hurts 1, helps 1 (5%) | · | helps 1 (3%) | hurts 1 (16%) | · |
| `decompose_tanh` | helps 2 (6%) | · | · | · | · | · |
| `unfuse_bias_add_to_pointwise` | · | · | helps 2 (12%) | · | · | · |
| `_sfdp_pattern_28_half_inference` | · | · | · | helps 1 (5%) | helps 1 (8%) | · |
| `split_reductions` | · | · | · | helps 1 (4%) | hurts 1 (6%) | · |
| `_sfdp_pattern_28_inference` | helps 1 (5%) | · | · | · | · | · |
