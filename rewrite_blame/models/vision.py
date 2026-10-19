"""torchvision models (random init)."""

from __future__ import annotations

import torch

from . import ModelSpec, register


def _tv(name, ctor, batch, size, res=224, pretrained=False, **kw):
    def build():
        import torchvision
        m = getattr(torchvision.models, ctor)(weights="DEFAULT" if pretrained else None)   # DEFAULT downloads once
        x = torch.randn(batch, 3, res, res)
        return m, (x,)
    return register(ModelSpec(name=name, build=build, family="vision", size=size,
                              description=f"torchvision {ctor}, batch {batch}, {res}x{res}" + (", pretrained" if pretrained else ""),
                              tags=("pretrained",) if pretrained else (),
                              exercises=("freezing:conv_bn", "freezing:mkldnn", "lowering:layout", "scheduler"), **kw))


_tv("resnet18", "resnet18", 8, "small")
_tv("resnet50", "resnet50", 4, "medium")
_tv("mobilenet_v3_small", "mobilenet_v3_small", 8, "small")
_tv("efficientnet_b0", "efficientnet_b0", 4, "medium")
_tv("convnext_tiny", "convnext_tiny", 4, "medium")
_tv("vit_b_16", "vit_b_16", 4, "large")

# trained weights (same shapes as the random-init models above): do the findings depend on the weights?
_tv("resnet18_pretrained", "resnet18", 8, "small", pretrained=True)
_tv("resnet50_pretrained", "resnet50", 4, "medium", pretrained=True)
_tv("mobilenet_v3_small_pretrained", "mobilenet_v3_small", 8, "small", pretrained=True)
_tv("efficientnet_b0_pretrained", "efficientnet_b0", 4, "medium", pretrained=True)
_tv("convnext_tiny_pretrained", "convnext_tiny", 4, "medium", pretrained=True)
