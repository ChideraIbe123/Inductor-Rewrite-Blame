"""HuggingFace transformer models built from fixed configs (random init, no downloads)."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import ModelSpec, register


class _Logits(nn.Module):
    def __init__(self, m, attr):
        super().__init__()
        self.m, self.attr = m, attr

    def forward(self, input_ids):
        return getattr(self.m(input_ids=input_ids), self.attr)


def _set_attn(cfg, impl):
    if impl is None:
        return cfg
    try:
        cfg._attn_implementation = impl
    except Exception:
        pass
    return cfg


def _hf(name, make_cfg, model_cls_name, attr, batch, seq, size, desc, attn=None, vocab=None, **kw):
    def build():
        import transformers
        cfg = _set_attn(make_cfg(), attn)
        cls = getattr(transformers, model_cls_name)
        m = cls(cfg)
        V = vocab or cfg.vocab_size
        ids = torch.randint(0, V, (batch, seq))
        return _Logits(m, attr), (ids,)
    ex = ("joint:sfdp",) if attn == "eager" else ()
    return register(ModelSpec(name=name, build=build, family="hf", size=size,
                              description=f"{desc}, batch {batch}, seq {seq}, attn={attn or 'default'}",
                              exercises=ex + ("scheduler", "post_grad"), **kw))


def _bert():
    from transformers import BertConfig
    return BertConfig()  # bert-base-uncased dims


def _distilbert():
    from transformers import DistilBertConfig
    return DistilBertConfig()


def _gpt2():
    from transformers import GPT2Config
    return GPT2Config()  # gpt2 small


def _t5():
    from transformers import T5Config
    return T5Config()  # t5-small dims


def _smollm():
    from transformers import LlamaConfig
    return LlamaConfig(hidden_size=576, intermediate_size=1536, num_hidden_layers=30, num_attention_heads=9,
                       num_key_value_heads=3, vocab_size=49152, max_position_embeddings=2048,
                       rms_norm_eps=1e-5, tie_word_embeddings=True)


def _qwen05b():
    from transformers import Qwen2Config
    return Qwen2Config(hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
                       num_key_value_heads=2, vocab_size=151936, max_position_embeddings=4096, tie_word_embeddings=True)


_hf("bert_base", _bert, "BertModel", "last_hidden_state", 8, 128, "medium", "BERT-base encoder")
_hf("bert_base_eagerattn", _bert, "BertModel", "last_hidden_state", 8, 128, "medium", "BERT-base encoder", attn="eager")
_hf("distilbert", _distilbert, "DistilBertModel", "last_hidden_state", 8, 128, "small", "DistilBERT encoder")
_hf("gpt2_prefill", _gpt2, "GPT2LMHeadModel", "logits", 4, 128, "medium", "GPT-2 small prefill")
_hf("gpt2_prefill_eagerattn", _gpt2, "GPT2LMHeadModel", "logits", 4, 128, "medium", "GPT-2 small prefill", attn="eager")
_hf("gpt2_decode", _gpt2, "GPT2LMHeadModel", "logits", 4, 1, "small", "GPT-2 small single-token step (no KV cache)")
_hf("t5_small_encoder", _t5, "T5EncoderModel", "last_hidden_state", 8, 128, "small", "T5-small encoder")
_hf("smollm_135m_prefill", _smollm, "LlamaForCausalLM", "logits", 1, 128, "medium", "SmolLM-135M (LLaMA arch) prefill")
_hf("smollm_135m_prefill_eagerattn", _smollm, "LlamaForCausalLM", "logits", 1, 128, "medium",
    "SmolLM-135M (LLaMA arch) prefill", attn="eager")
_hf("smollm_135m_decode", _smollm, "LlamaForCausalLM", "logits", 1, 1, "small", "SmolLM-135M single-token step (no KV cache)")
_hf("qwen2_0_5b_prefill", _qwen05b, "Qwen2ForCausalLM", "logits", 1, 128, "large", "Qwen2.5-0.5B prefill")
_hf("qwen2_0_5b_decode", _qwen05b, "Qwen2ForCausalLM", "logits", 1, 1, "medium", "Qwen2.5-0.5B single-token step (no KV cache)")
