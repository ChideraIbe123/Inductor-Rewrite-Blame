"""Model corpus: name -> ModelSpec. Models are built with random weights from fixed configs
(no downloads), in eval mode, fp32, with batch sizes sized for CPU runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import torch


@dataclass
class ModelSpec:
    name: str
    build: Callable[[], tuple[torch.nn.Module, tuple[Any, ...]]]
    family: str            # toy | vision | hf | llm
    size: str              # tiny | small | medium | large
    description: str = ""
    tags: tuple[str, ...] = ()
    # which rewrite families this model is expected to exercise (used by the verify report)
    exercises: tuple[str, ...] = ()


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in REGISTRY:
        raise KeyError(f"duplicate model {spec.name}")
    REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> ModelSpec:
    _ensure_loaded()
    if name not in REGISTRY:
        from . import sweeps, hf_zoo
        spec = sweeps.resolve(name) or hf_zoo.resolve(name)   # synthetic sweeps, torchvision zoo, transformer zoo
        if spec is None:
            raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)} or a sweep name like sweep_attn_B4_T128_H8_D64")
        REGISTRY[name] = spec
    return REGISTRY[name]


def names(family: str | None = None, size: str | None = None) -> list[str]:
    _ensure_loaded()
    out = []
    for s in REGISTRY.values():
        if family and s.family != family:
            continue
        if size and s.size != size:
            continue
        out.append(s.name)
    return out


def to_device(obj, device, dtype=None):
    """Move every tensor inside a (possibly nested) tuple/list/dict of inputs to *device*; floating tensors
    are also cast to *dtype* when one is given (token ids and masks keep their integer types)."""
    if isinstance(obj, torch.Tensor):
        t = obj.to(device)
        return t.to(dtype) if dtype is not None and t.is_floating_point() else t
    if isinstance(obj, tuple):
        return tuple(to_device(o, device, dtype) for o in obj)
    if isinstance(obj, list):
        return [to_device(o, device, dtype) for o in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device, dtype) for k, v in obj.items()}
    return obj


def build(name: str, seed: int = 0) -> tuple[torch.nn.Module, tuple[Any, ...]]:
    """Build a model and its example inputs on the measurement device (``REWRITE_BLAME_DEVICE``, default cpu)."""
    from .. import env
    spec = get(name)
    torch.manual_seed(seed)
    model, inputs = spec.build()
    model.eval()
    dev = env.device() if env.is_cuda() else None
    dt = env.torch_dtype()
    if dt == torch.float32:
        dt = None
    if dev is not None or dt is not None:
        model = model.to(device=dev, dtype=dt) if dt is not None else model.to(dev)
        inputs = to_device(inputs, dev if dev is not None else torch.device("cpu"), dt)
    return model, inputs


_PROFILES: dict[str, dict] = {}


def profile_of(name: str, seed: int = 0) -> dict:
    """Whole-model profile features (``cfg_*``) for a model name: built on the CPU in fp32 once per process
    and cached. Used as guard features for real-model datasets and by the live switch-level guards."""
    if name in _PROFILES:
        return _PROFILES[name]
    from ..live_guards import model_profile
    spec = get(name)
    torch.manual_seed(seed)
    model, inputs = spec.build()
    model.eval()
    prof = model_profile(model, inputs) or {}
    prof["cfg_arch"] = arch_of(name)
    _PROFILES[name] = prof
    return prof


def arch_of(name: str) -> str:
    """Architecture key used to group cross-validation folds (batch/sequence variants stay together)."""
    from . import zoo
    p = zoo.parse(name)
    if p:
        return p[0]
    from . import hf_zoo
    a = hf_zoo.arch_of(name)
    if a:
        return a
    import re
    return re.sub(r"_(b|s)\d+", "", name)


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import toy  # noqa: F401
    from . import vision  # noqa: F401
    from . import hf  # noqa: F401


def expand_spec(spec: str, level: str = "full") -> list[str]:
    """Model-set specs for datasets and evaluations: ``zoo:conv`` (torchvision zoo), ``zoo:hf`` (transformer
    zoo), ``zoo:all``, ``corpus:images`` (the seven held-out image models), ``corpus:hf``, ``corpus:all``, a
    comma-separated list of names, or a glob over known names."""
    import fnmatch
    from . import zoo, hf_zoo
    _ensure_loaded()
    images = ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0", "convnext_tiny", "conv_bn_relu_stack", "vit_b_16"]
    table = {"zoo:conv": zoo.names(level), "zoo:hf": hf_zoo.names(level), "zoo:all": zoo.names(level) + hf_zoo.names(level),
             "corpus:images": images, "corpus:hf": names("hf"), "corpus:all": names()}
    if spec in table:
        return list(table[spec])
    out = []
    for term in spec.split(","):
        term = term.strip()
        if not term:
            continue
        if "*" in term or "?" in term:
            out += [n for n in names() + zoo.names(level) + hf_zoo.names(level) if fnmatch.fnmatch(n, term)]
        else:
            out.append(term)
    return out
