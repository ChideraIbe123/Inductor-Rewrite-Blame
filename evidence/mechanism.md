| switch | machine | ON helps: n, median %, Δkernels, Δextern calls, Δalloc MB | ON hurts: n, median %, Δkernels, Δextern calls, Δalloc MB |
|---|---|---|---|
| `layout_optimization` | Mac M4 | 1, 10%, +2, +0, +22 | 4, 28%, +2, +0, +4 |
| `layout_optimization` | Xeon | 7, 14%, +2, +0, +10 | · |
| `layout_optimization` | 4090 bf16 | 3, 4%, +6, +0, +7 | 1, 6%, +12, +0, +2 |
| `layout_optimization` | L4 bf16 | 3, 19%, +6, +0, +7 | 1, 4%, +18, +0, +1 |
| `layout_optimization` | A40 bf16 | 2, 21%, +8, +0, +9 | · |
| `freezing` | Mac M4 | 1, 6%, +0, +0, -0 | 1, 7%, -10, +0, -12 |
| `freezing` | Xeon | 22, 8%, +0, +0, -18 | · |
| `freezing` | 4090 bf16 | 19, 11%, +0, -1, +0 | · |
| `freezing` | L4 bf16 | 12, 11%, -0, -0, -0 | · |
| `freezing` | A40 bf16 | 14, 13%, +0, -18, +0 | · |
| `max_fusion_size` | Mac M4 | 2, 5%, +0, +0, +0 | · |
| `max_fusion_size` | Xeon | · | 1, 5%, +0, +0, +0 |
| `max_fusion_size` | 4090 bf16 | 11, 12%, -8, +0, -0 | 1, 21%, -11, +0, +0 |
| `max_fusion_size` | L4 bf16 | 12, 7%, -12, +0, -0 | 1, 15%, -11, +0, +0 |
| `max_fusion_size` | A40 bf16 | 13, 8%, -8, +0, -0 | · |
| `decompose_mm_pass` | Mac M4 | · | 3, 39%, -3, -96, +0 |
| `decompose_mm_pass` | Xeon | 2, 53%, -4, -106, +0 | · |
| `pattern_matcher` | Mac M4 | 3, 18%, -2, -1, -7 | · |
| `pattern_matcher` | Xeon | · | 1, 12%, -3, -1, -7 |
| `pattern_matcher` | L4 bf16 | 2, 6%, -4, -6, -43 | · |
| `post_grad_passes` | Mac M4 | 2, 10%, -0, +0, -0 | 1, 3%, +2, +0, +0 |
| `post_grad_passes` | Xeon | 2, 144%, +0, +0, -0 | 3, 27%, +0, +0, +0 |
| `inplace_buffers` | Xeon | 11, 9%, +0, +0, -64 | 2, 50%, +0, +0, -1 |
| `inplace_buffers` | 4090 bf16 | 2, 16%, +0, +0, -4 | · |
| `inplace_buffers` | L4 bf16 | 3, 8%, +0, +0, -1 | · |
| `conv_1x1_as_mm` | Mac M4 | 2, 5%, +0, +0, +36 | · |
| `conv_1x1_as_mm` | Xeon | 2, 16%, +0, +0, +30 | · |
| `conv_1x1_as_mm` | 4090 bf16 | 2, 9%, +0, +0, +15 | · |
