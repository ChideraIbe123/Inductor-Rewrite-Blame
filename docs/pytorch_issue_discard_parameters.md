# Draft: second PyTorch issue (bug) — `freezing_discard_parameters=True` crashes Dynamo on tied-embedding models

To be filed by Chidera after #201502, with an own-words intro and the AI-assistance disclosure line, same as before.

**Title:** `torch._inductor.config.freezing_discard_parameters=True` makes Dynamo assert "Guard failed on the same frame it was created" on models with tied embeddings (GPT-2, Llama/SmolLM, Qwen2)

### Summary

While measuring the memory cost of `freezing` (see #201502), I tried the documented way to avoid it,
`config.freezing_discard_parameters=True`. With it on, every model whose output head shares weights with the input
embedding (`tie_word_embeddings`) fails on the second call of the compiled module with a Dynamo guard assertion.
Conv nets and BERT/DistilBERT (no tied head) are fine and do get the memory back: on an RTX A40 in bf16 the peak
memory with freezing goes from 72 to 51 MB on ResNet-18 (unfrozen: 48), 121 to 75 on ResNet-50 (73), 280 to 235 on
BERT-base (241), with no speed change.

### Minimal reproducer (CPU is enough; torch 2.14.1, transformers 5.19)

```python
import torch, torch._inductor.config as cfg
from transformers import GPT2Config, GPT2LMHeadModel
cfg.freezing = True
cfg.freezing_discard_parameters = True
m = GPT2LMHeadModel(GPT2Config(n_embd=256, n_layer=2, n_head=4)).eval()   # tie_word_embeddings=True by default
ids = torch.randint(0, 50257, (1, 8))
with torch.no_grad():
    c = torch.compile(m)
    c(input_ids=ids)          # first call: compiles fine
    c(input_ids=ids)          # second call:
```
```
AssertionError: Guard failed on the same frame it was created. This is a bug - please create an issue.
Guard fail reason: 0/0: self._modules['transformer']._modules['wte']._parameters[...]
```
Same failure, on CPU and on an A40 in bf16, for GPT-2 small (pretrained `gpt2` and random init), SmolLM-135M
(`HuggingFaceTB/SmolLM-135M`, Llama architecture) and Qwen2.5-0.5B, in both prefill (seq 128) and single-token
decode. With `freezing_discard_parameters=False` the same models compile and run.

### What seems to happen

Discarding the parameters after folding replaces `wte.weight`, which is also `lm_head.weight`; the guard installed on
the first frame still refers to the original parameter identity, so it fails on the very frame that created it.

### Why it matters

`freezing` is worth 7–10% median on compiled inference across three GPUs (#201502) and its only real cost is the
duplicate parameter memory (+45–51% peak on decoder models). `freezing_discard_parameters` is the knob that removes
that cost, and decoder LLMs are exactly the models where it is needed most, and exactly where it crashes.

### Environment

torch 2.14.1 (+cu126 on the GPU, default wheel on macOS 15 / Apple M4 Pro for the CPU repro), transformers 5.19/5.20,
triton 3.8.0.
