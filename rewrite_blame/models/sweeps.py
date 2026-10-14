"""Parameterised synthetic models for building the guard dataset.

Three families, one per rewrite family we want a guard for. Names encode the configuration so
they can be resolved without pre-registration:

* ``sweep_attn_B{B}_T{T}_H{H}_D{D}``        manual attention (attention-fusion rules)
* ``sweep_conv_C{C}_R{R}_K{K}_L{L}_{dw|dense}`` conv-bn-act stack (channels-last layout optimisation)
* ``sweep_mlp_M{M}_I{I}_Hd{H}``             gated MLP with batch M (decompose_mm_pass)
"""

from __future__ import annotations

import itertools
import os
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelSpec


class SweepAttention(nn.Module):
    def __init__(self, heads, dim_head):
        super().__init__()
        dim = heads * dim_head
        self.h, self.d = heads, dim_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(self.d)).softmax(dim=-1)
        return self.out((att @ v).transpose(1, 2).reshape(B, T, C)) + x


class SweepConv(nn.Module):
    def __init__(self, channels, kernel, layers, depthwise):
        super().__init__()
        mods = []
        for _ in range(layers):
            mods += [nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels if depthwise else 1),
                     nn.BatchNorm2d(channels), nn.SiLU()]
            if depthwise:  # a 1x1 mixing conv, as in MobileNet/EfficientNet blocks
                mods += [nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.SiLU()]
        self.body = nn.Sequential(*mods)

    def forward(self, x):
        return self.body(x)


class SweepMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


_ATTN = re.compile(r"^sweep_attn_B(\d+)_T(\d+)_H(\d+)_D(\d+)$")
_CONV = re.compile(r"^sweep_conv_C(\d+)_R(\d+)_K(\d+)_L(\d+)_(dw|dense)$")
_MLP = re.compile(r"^sweep_mlp_M(\d+)_I(\d+)_Hd(\d+)$")

# which rewrite family each sweep family targets, and how to express "rule off" relative to default
FAMILY_RULE = {
    "attn": {"family": "_sfdp_pattern", "off_spec": "default-*_sfdp_pattern_*_inference", "on_spec": "default"},
    "conv": {"family": "layout_optimization", "off_spec": "default-lowering/layout_optimization", "on_spec": "default"},
    "mlp": {"family": "decompose_mm", "off_spec": "default", "on_spec": "default+optimus/decompose_mm_wide"},
}


def conv_batch(C: int, R: int) -> int:
    """Batch size of a conv sweep configuration. CPU: 4 (the original grid). GPU: the activation must be
    large enough that the kernels, not their launches, dominate the time (sub-millisecond runs gave 14%
    noise bands on the A40), so the batch is raised until one activation holds at least 32M elements
    (128 MB fp32), between 64 and 4096. REWRITE_BLAME_SWEEP_BATCH overrides both."""
    env = os.environ.get("REWRITE_BLAME_SWEEP_BATCH")
    if env:
        return int(env)
    from .. import env as _env
    if not _env.is_cuda():
        return 4
    per_sample = C * R * R
    b = 64
    while b < 4096 and b * per_sample < 32 * 1024 * 1024:
        b *= 2
    return b


def resolve(name: str) -> ModelSpec | None:
    """Build a ModelSpec from a sweep name, or None if the name is not a sweep name."""
    m = _ATTN.match(name)
    if m:
        B, T, H, D = map(int, m.groups())
        def build():
            return SweepAttention(H, D), (torch.randn(B, T, H * D),)
        return ModelSpec(name=name, build=build, family="sweep", size="tiny", description=f"attention B={B} T={T} H={H} D={D}",
                         tags=("attn",), exercises=("joint:sfdp",))
    m = _CONV.match(name)
    if m:
        C, R, K, L, kind = m.groups(); C, R, K, L = int(C), int(R), int(K), int(L)
        def build():
            return SweepConv(C, K, L, kind == "dw"), (torch.randn(conv_batch(C, R), C, R, R),)
        return ModelSpec(name=name, build=build, family="sweep", size="tiny", description=f"conv C={C} R={R} K={K} L={L} {kind}",
                         tags=("conv",), exercises=("lowering:layout",))
    m = _MLP.match(name)
    if m:
        M, I, Hd = map(int, m.groups())
        def build():
            return SweepMLP(I, Hd), (torch.randn(M, I),)
        return ModelSpec(name=name, build=build, family="sweep", size="tiny", description=f"gated MLP M={M} in={I} hidden={Hd}",
                         tags=("mlp",), exercises=("optimus:decompose_mm",))
    return None


def sweep_family_of(name: str) -> str | None:
    for fam, rx in (("attn", _ATTN), ("conv", _CONV), ("mlp", _MLP)):
        if rx.match(name):
            return fam
    return None


def config_of(name: str) -> dict:
    """Numeric configuration parameters parsed from the name (features for the guard)."""
    m = _ATTN.match(name)
    if m:
        B, T, H, D = map(int, m.groups())
        return {"B": B, "T": T, "H": H, "D": D, "tokens": B * T, "qk_numel": B * H * T * T}
    m = _CONV.match(name)
    if m:
        C, R, K, L, kind = m.groups()
        C, R = int(C), int(R)
        return {"C": C, "R": R, "K": int(K), "L": int(L), "depthwise": int(kind == "dw"),
                "act_numel": conv_batch(C, R) * C * R * R}
    m = _MLP.match(name)
    if m:
        M, I, Hd = map(int, m.groups())
        return {"M": M, "I": I, "Hd": Hd, "weight_numel": 2 * I * Hd + Hd * I, "flops_per_byte": M / 2.0}
    return {}


def grid(family: str, level: str = "full") -> list[str]:
    """Configuration grid per family. ``level='small'`` is a quick subset."""
    if family == "attn":
        Bs, Ts, Hs, Ds = ([1, 4], [32, 128, 512], [4, 12], [64]) if level == "small" else \
                         ([1, 2, 4, 8], [16, 32, 64, 128, 256, 512, 1024], [1, 4, 8, 12, 16], [32, 64, 128])
        # cap the attention-score tensor (B*H*T*T elements, fp32) at 64M elements so the largest
        # configurations do not dominate the collection time
        names = [f"sweep_attn_B{B}_T{T}_H{H}_D{D}" for B, T, H, D in itertools.product(Bs, Ts, Hs, Ds) if B * H * T * T <= 64 * 1024 * 1024]
    elif family == "conv":
        Cs, Rs, Ks, Ls, kinds = ([32, 128], [28, 56], [3], [2], ["dw", "dense"]) if level == "small" else \
                                ([16, 32, 64, 128, 256], [14, 28, 56, 112], [1, 3, 5], [1, 3], ["dw", "dense"])
        # cap the activation (4*C*R*R) * layers * kernel area so the largest configurations stay minutes, not tens of minutes
        names = [f"sweep_conv_C{C}_R{R}_K{K}_L{L}_{k}" for C, R, K, L, k in itertools.product(Cs, Rs, Ks, Ls, kinds)
                 if not (k == "dense" and K == 1) and 4 * C * R * R * L * K * K <= 4 * 256 * 56 * 56 * 3 * 9]
    elif family == "mlp":
        Ms, Is, Hds = ([1, 8], [1024], [2048, 8192]) if level == "small" else \
                      ([1, 2, 4, 8, 16, 32, 64, 128], [512, 1024, 2048, 4096], [1024, 2048, 4096, 8192, 11008])
        names = [f"sweep_mlp_M{M}_I{I}_Hd{Hd}" for M, I, Hd in itertools.product(Ms, Is, Hds)]
    else:
        raise KeyError(family)
    return names
