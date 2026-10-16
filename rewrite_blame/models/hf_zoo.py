"""Transformer zoo: HuggingFace architectures built from explicit configs (random weights, no downloads).

Names are ``hfzoo_<arch>_<prefill|decode>_s<seq>_b<batch>``. The corpus transformers (bert_base*, distilbert,
gpt2_*, t5_small_encoder, smollm_*, qwen2_0_5b_*) stay out of the zoo so they can be held out for the
end-to-end evaluation. Attention uses the library default implementation (sdpa where available)."""

from __future__ import annotations

import re

import torch

from . import ModelSpec


def _bert(h, L, a, inter):
    from transformers import BertConfig
    return BertConfig(hidden_size=h, num_hidden_layers=L, num_attention_heads=a, intermediate_size=inter), "BertModel", "last_hidden_state"


def _roberta():
    from transformers import RobertaConfig
    return RobertaConfig(), "RobertaModel", "last_hidden_state"


def _gpt2(n_embd, n_layer, n_head):
    from transformers import GPT2Config
    return GPT2Config(n_embd=n_embd, n_layer=n_layer, n_head=n_head), "GPT2LMHeadModel", "logits"


def _opt125m():
    from transformers import OPTConfig
    return OPTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, ffn_dim=3072, word_embed_proj_dim=768), "OPTForCausalLM", "logits"


def _llama(h, L, a, kv, inter, vocab=32000):
    from transformers import LlamaConfig
    return LlamaConfig(hidden_size=h, num_hidden_layers=L, num_attention_heads=a, num_key_value_heads=kv,
                       intermediate_size=inter, vocab_size=vocab), "LlamaForCausalLM", "logits"


def _qwen15b():
    from transformers import Qwen2Config
    return Qwen2Config(hidden_size=1536, num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2,
                       intermediate_size=8960, vocab_size=151936, tie_word_embeddings=True), "Qwen2ForCausalLM", "logits"


def _t5base_enc():
    from transformers import T5Config
    return T5Config(d_model=768, d_ff=3072, num_layers=12, num_heads=12), "T5EncoderModel", "last_hidden_state"


def _vit(h, L, a, inter, patch=16, image=224):
    from transformers import ViTConfig
    return ViTConfig(hidden_size=h, num_hidden_layers=L, num_attention_heads=a, intermediate_size=inter, patch_size=patch, image_size=image), "ViTModel", "last_hidden_state"


def _swin_t():
    from transformers import SwinConfig
    return SwinConfig(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]), "SwinModel", "last_hidden_state"


# arch -> (config factory, input kind, size class)
ARCHS = {
    "bert_tiny": (lambda: _bert(128, 2, 2, 512), "ids", "tiny"),
    "bert_small": (lambda: _bert(512, 4, 8, 2048), "ids", "small"),
    "bert_large": (lambda: _bert(1024, 24, 16, 4096), "ids", "large"),
    "roberta_base": (_roberta, "ids", "medium"),
    "gpt2_medium": (lambda: _gpt2(1024, 24, 16), "ids", "large"),
    "gpt2_tiny": (lambda: _gpt2(256, 4, 4), "ids", "tiny"),
    "opt_125m": (_opt125m, "ids", "medium"),
    "llama_tiny": (lambda: _llama(512, 4, 8, 2, 1376), "ids", "small"),
    "llama_1b": (lambda: _llama(2048, 16, 32, 8, 8192, vocab=128256), "ids", "large"),
    "qwen2_1_5b": (_qwen15b, "ids", "large"),
    "t5_base_encoder": (_t5base_enc, "ids", "medium"),
    "vit_small": (lambda: _vit(384, 12, 6, 1536), "pixels", "medium"),
    "vit_tiny": (lambda: _vit(192, 12, 3, 768), "pixels", "small"),
    "swin_tiny": (_swin_t, "pixels", "medium"),
}
# (mode, seq, batch) grid; decode = one token per sequence, prefill = seq tokens; pixel models use batch only
GRID = {"full": [("prefill", 128, 8), ("prefill", 512, 1), ("decode", 1, 1), ("decode", 1, 8)],
        "small": [("prefill", 128, 8)]}
PIXEL_GRID = {"full": [8, 32], "small": [8]}
_NAME = re.compile(r"^hfzoo_(?P<arch>[a-z0-9_]+?)_(?P<mode>prefill|decode|pixels)_s(?P<seq>\d+)_b(?P<B>\d+)$")


class _Logits(torch.nn.Module):
    def __init__(self, m, attr, kind):
        super().__init__()
        self.m, self.attr, self.kind = m, attr, kind

    def forward(self, x):
        out = self.m(input_ids=x) if self.kind == "ids" else self.m(pixel_values=x)
        return getattr(out, self.attr)


def names(level: str = "full") -> list[str]:
    out = []
    for arch, (_, kind, _) in ARCHS.items():
        if kind == "ids":
            out += [f"hfzoo_{arch}_{mode}_s{seq}_b{B}" for mode, seq, B in GRID[level]]
        else:
            out += [f"hfzoo_{arch}_pixels_s224_b{B}" for B in PIXEL_GRID[level]]
    return out


def parse(name: str):
    m = _NAME.match(name)
    if not m or m.group("arch") not in ARCHS:
        return None
    return m.group("arch"), m.group("mode"), int(m.group("seq")), int(m.group("B"))


def arch_of(name: str) -> str | None:
    p = parse(name)
    return p[0] if p else None


def resolve(name: str) -> ModelSpec | None:
    p = parse(name)
    if p is None:
        return None
    arch, mode, seq, B = p
    factory, kind, size = ARCHS[arch]

    def build():
        import transformers
        cfg, cls_name, attr = factory()
        m = getattr(transformers, cls_name)(cfg)
        if kind == "ids":
            x = torch.randint(0, cfg.vocab_size, (B, seq))
        else:
            x = torch.randn(B, 3, 224, 224)
        return _Logits(m, attr, kind), (x,)
    return ModelSpec(name=name, build=build, family="hfzoo", size=size,
                     description=f"{arch} {mode} seq {seq} batch {B}", tags=("transformer",),
                     exercises=("joint:sfdp", "scheduler", "post_grad"))
