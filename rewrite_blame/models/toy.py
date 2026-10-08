"""Hand-written graphs, each exercising a known family of Inductor rewrites."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelSpec, register


class AttentionBlock(nn.Module):
    """Manual softmax attention: the _sfdp_pattern_* rules rewrite it into SDPA."""

    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.norm(x + self.out(y))


class ConvBnReluStack(nn.Module):
    """conv-bn-relu x3: under freezing, bn folds into conv and relu fuses via mkldnn."""

    def __init__(self, c=32, layers=3):
        super().__init__()
        mods = []
        for _ in range(layers):
            mods += [nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU()]
        self.body = nn.Sequential(*mods)

    def forward(self, x):
        return self.body(x)


class SplitCatMLP(nn.Module):
    """split -> per-chunk linear -> cat: the optimus split/cat and batch_linear passes target this."""

    def __init__(self, dim=1024, chunks=4):
        super().__init__()
        self.chunks = chunks
        self.lins = nn.ModuleList(nn.Linear(dim // chunks, dim // chunks) for _ in range(chunks))
        self.final = nn.Linear(dim, dim)

    def forward(self, x):
        parts = torch.split(x, x.shape[-1] // self.chunks, dim=-1)
        ys = [lin(p) for lin, p in zip(self.lins, parts)]
        y = torch.cat(ys, dim=-1)
        return self.final(F.relu(y))


class NormMLP(nn.Module):
    """layernorm/gelu MLP with a pointless dtype round trip and a 1/sqrt: joint-graph patterns
    (pointless_convert, pointless_view) and post-grad reciprocal_sqrt_to_rsqrt fire here."""

    def __init__(self, dim=1024):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        h = self.ln(x)
        h = h.double().float()                      # pointless convert pair
        h = F.gelu(self.fc1(h))
        h = h.view(h.shape[0], -1, h.shape[-1]).view(h.shape)  # pointless view pair
        s = 1.0 / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)  # rsqrt candidate
        return self.fc2(h * s) + x


class DecodeMLP(nn.Module):
    """Single-token (M=1) MLP as in LLM decode: the skinny matmuls are what decompose_mm_pass rewrites."""

    def __init__(self, dim=1024, hidden=2048):
        super().__init__()
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CatSliceCat(nn.Module):
    """cat(slice(cat(..)), ..) chains: post-grad cat_slice_cat / splitwithsizes_cat targets."""

    def __init__(self, dim=512):
        super().__init__()
        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)

    def forward(self, x):
        y = torch.cat([self.a(x), self.b(x)], dim=-1)
        z = torch.cat([y[..., : x.shape[-1]], torch.tanh(x)], dim=-1)
        parts = torch.split_with_sizes(z, [x.shape[-1], x.shape[-1]], dim=-1)
        return torch.cat(parts, dim=-1) * 0.5


def _spec(name, cls_kwargs, input_shape, cls, size, desc, exercises, **kw):
    def build():
        m = cls(**cls_kwargs)
        x = torch.randn(*input_shape)
        return m, (x,)
    return register(ModelSpec(name=name, build=build, family="toy", size=size, description=desc,
                              exercises=tuple(exercises), **kw))


_spec("attention_block", {}, (4, 128, 512), AttentionBlock, "tiny",
      "manual attention (B=4,T=128,C=512,H=8)", ["joint:sfdp", "scheduler"])
_spec("conv_bn_relu_stack", {}, (8, 32, 56, 56), ConvBnReluStack, "tiny",
      "3x conv3x3-bn-relu on 8x32x56x56", ["freezing:conv_bn", "freezing:mkldnn", "lowering"])
_spec("split_cat_mlp", {}, (64, 1024), SplitCatMLP, "tiny",
      "split->4 linears->cat->linear", ["optimus:split_cat", "optimus:batch_linear", "post_grad:cat"])
_spec("norm_mlp", {}, (64, 1024), NormMLP, "tiny",
      "layernorm/gelu MLP with pointless convert/view and 1/sqrt", ["joint:pointless", "post_grad:rsqrt"])
_spec("decode_mlp", {}, (1, 1024), DecodeMLP, "tiny",
      "M=1 gated MLP (LLM decode shape)", ["optimus:decompose_mm"])
_spec("cat_slice_cat", {}, (64, 512), CatSliceCat, "tiny",
      "cat/slice/split_with_sizes/cat chains", ["post_grad:cat_slice_cat", "post_grad:splitwithsizes_cat"])
