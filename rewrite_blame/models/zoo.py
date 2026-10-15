"""Real convolutional networks from torchvision (random weights) for the real-network layout dataset.

Names are ``zoo_<torchvision ctor>_b<batch>``. The seven corpus image models (resnet18, resnet50,
mobilenet_v3_small, efficientnet_b0, convnext_tiny, conv_bn_relu_stack, vit_b_16) are deliberately NOT in
the zoo: they are the held-out end-to-end test for a guard trained on it."""

from __future__ import annotations

import re

import torch

from . import ModelSpec

# (torchvision constructor, input resolution)
ARCHS = [
    ("resnet34", 224), ("resnet101", 224), ("resnext50_32x4d", 224), ("wide_resnet50_2", 224),
    ("densenet121", 224), ("mobilenet_v2", 224), ("mobilenet_v3_large", 224),
    ("efficientnet_b1", 240), ("efficientnet_v2_s", 384), ("regnet_y_400mf", 224), ("regnet_x_800mf", 224),
    ("mnasnet1_0", 224), ("shufflenet_v2_x1_0", 224), ("squeezenet1_1", 224),
    ("vgg11_bn", 224), ("vgg16_bn", 224), ("googlenet", 224), ("inception_v3", 299),
    ("convnext_small", 224), ("alexnet", 224),
]
# batch 1 is launch-bound on a GPU (36-43% noise bands on the A40), so the grid starts at 8
BATCHES = {"full": (8, 32, 64), "small": (8,)}
_NAME = re.compile(r"^zoo_(?P<arch>[a-z0-9_]+?)_b(?P<B>\d+)$")
_FEATURES: dict[str, dict] = {}


def names(level: str = "full") -> list[str]:
    return [f"zoo_{a}_b{b}" for a, _ in ARCHS for b in BATCHES[level]]


def parse(name: str):
    m = _NAME.match(name)
    if not m:
        return None
    arch, B = m.group("arch"), int(m.group("B"))
    res = dict(ARCHS).get(arch)
    return (arch, B, res) if res else None


def resolve(name: str) -> ModelSpec | None:
    p = parse(name)
    if p is None:
        return None
    arch, B, res = p

    def build():
        import torchvision
        kw = {}
        if arch in ("googlenet", "inception_v3"):
            kw = {"aux_logits": False, "init_weights": False}
        m = getattr(torchvision.models, arch)(weights=None, **kw)
        return m, (torch.randn(B, 3, res, res),)
    return ModelSpec(name=name, build=build, family="zoo", size="medium", description=f"torchvision {arch}, batch {B}, {res}x{res}",
                     tags=("conv", "net"), exercises=("lowering:layout",))


def features_of(name: str) -> dict:
    """Whole-network features (built on the CPU, forward pass with hooks); cached per name."""
    if name in _FEATURES:
        return _FEATURES[name]
    spec = resolve(name)
    if spec is None:
        return {}
    from ..live_guards import network_features
    torch.manual_seed(0)
    model, inputs = spec.build()
    model.eval()
    f = network_features(model, inputs) or {}
    f["arch"] = parse(name)[0]
    _FEATURES[name] = f
    return f
