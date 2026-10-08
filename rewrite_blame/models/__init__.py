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
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
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


def build(name: str, seed: int = 0) -> tuple[torch.nn.Module, tuple[Any, ...]]:
    spec = get(name)
    torch.manual_seed(seed)
    model, inputs = spec.build()
    model.eval()
    return model, inputs


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import toy  # noqa: F401
    from . import vision  # noqa: F401
    from . import hf  # noqa: F401
