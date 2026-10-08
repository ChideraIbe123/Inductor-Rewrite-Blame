"""torchvision models (random init)."""

from __future__ import annotations

import torch

from . import ModelSpec, register


def _tv(name, ctor, batch, size, res=224, **kw):
    def build():
        import torchvision
        m = getattr(torchvision.models, ctor)(weights=None)
        x = torch.randn(batch, 3, res, res)
        return m, (x,)
    return register(ModelSpec(name=name, build=build, family="vision", size=size,
                              description=f"torchvision {ctor}, batch {batch}, {res}x{res}",
                              exercises=("freezing:conv_bn", "freezing:mkldnn", "lowering:layout", "scheduler"), **kw))


_tv("resnet18", "resnet18", 8, "small")
_tv("resnet50", "resnet50", 4, "medium")
_tv("mobilenet_v3_small", "mobilenet_v3_small", 8, "small")
_tv("efficientnet_b0", "efficientnet_b0", 4, "medium")
_tv("convnext_tiny", "convnext_tiny", 4, "medium")
_tv("vit_b_16", "vit_b_16", 4, "large")
