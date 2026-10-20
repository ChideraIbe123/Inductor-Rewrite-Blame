# Evidence files referenced by the write-ups and the PyTorch issue

Small, self-contained extracts of the measurement results (the full per-machine stores are hundreds of MB and
are not in the repository). One JSON line per measured network; `ratio` = time with the switch ON / time with
it OFF (median of 50 calls per run, `ratios` lists the independent run pairs), `label` is confirmed only when
two time-separated pairs agree beyond the machine's noise band (`tau_pct`). `cfg_*` are the whole-model profile
features; `compile_s_*` and `peak_mem_mb_*` the compile time and peak memory (CUDA `max_memory_allocated`,
or process RSS on CPU, which is not reliable for the random-init rows).

- `freezing/*_zoo.jsonl`: `torch._inductor.config.freezing` on vs off over 163 real networks per GPU (bf16).
- `freezing/*_corpus_pretrained_vs_random.jsonl`: the 12 corpus models with random and with trained weights.
- `freezing/real_workload_*.json`: SmolLM-135M `generate` (32 tokens, KV cache) and ResNet-50 batch-32
  classification, freezing on vs off, two fresh processes per setting.
- `sign_matrix.md`: rule x machine confirmed effects over six machine/precision targets.
- `mechanism.md`: confirmed verdicts joined with the code-level change (kernels, external calls, allocation).
