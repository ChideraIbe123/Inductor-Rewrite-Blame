"""Apply a switch state to a live torch process (config patches + pattern-entry gating + cache key)."""

from __future__ import annotations

import contextlib
import hashlib
import warnings
from collections import Counter
from typing import Any, Iterable

import torch
import torch._inductor.config as inductor_config
from torch._inductor.custom_graph_pass import CustomGraphPass

from .switches import Registry, Switch, State, state_hash

# ids of pattern switches that are currently OFF, consulted by the extra_check wrappers
_DISABLED: frozenset[str] = frozenset()
# number of times each pattern switch's original extra_check approved a match (i.e. the rule fired)
FIRED: Counter = Counter()
# number of times a disabled switch would have fired had it been on
SUPPRESSED: Counter = Counter()
# histogram of post-grad graph ops seen by SwitchSetPass (one compile may have several graphs)
GRAPH_OPS: Counter = Counter()
# features of each rule firing (switch id + shapes/dtypes of the matched tensors); the training
# data for the stretch goal (guards that predict whether a firing will hurt)
FIRINGS: list = []
MAX_FIRINGS = 500


def match_features(match) -> dict:
    """Cheap, backend-independent description of a pattern match: per matched tensor its shape,
    dtype and element count; plus the number of nodes the rewrite will replace."""
    feats: dict = {"n_nodes": len(getattr(match, "nodes", []) or [])}
    tensors = []

    def visit(x):
        node = x
        val = getattr(getattr(node, "meta", None), "get", lambda k, d=None: d)("val") if hasattr(node, "meta") else None
        if val is not None and hasattr(val, "shape"):
            try:
                tensors.append({"shape": [int(d) if not hasattr(d, "node") else str(d) for d in val.shape],
                                "dtype": str(val.dtype).replace("torch.", ""),
                                "numel": int(val.numel()) if all(isinstance(d, int) for d in val.shape) else None,
                                "device": str(val.device.type)})
            except Exception:
                pass
    for a in list(getattr(match, "args", []) or []):
        visit(a)
    for v in (getattr(match, "kwargs", {}) or {}).values():
        visit(v)
    feats["tensors"] = tensors[:8]
    if tensors:
        numels = [t["numel"] for t in tensors if t["numel"]]
        feats["max_numel"] = max(numels) if numels else None
        feats["dtypes"] = sorted({t["dtype"] for t in tensors})
    return feats


class GatedExtraCheck:
    """Replacement for a PatternEntry.extra_check that consults the active state."""

    __slots__ = ("switch_id", "original")

    def __init__(self, switch_id: str, original) -> None:
        self.switch_id = switch_id
        self.original = original

    def __call__(self, match):
        ok = self.original(match)
        if self.switch_id in _DISABLED:
            if ok:
                SUPPRESSED[self.switch_id] += 1
            return False
        if ok:
            FIRED[self.switch_id] += 1
            if len(FIRINGS) < MAX_FIRINGS:
                try:
                    FIRINGS.append({"switch": self.switch_id, **match_features(match)})
                except Exception:  # feature extraction must never break a compile
                    FIRINGS.append({"switch": self.switch_id})
        return ok

    def __repr__(self) -> str:
        return f"GatedExtraCheck({self.switch_id})"


class SwitchSetPass(CustomGraphPass):
    """No-op post-grad pass whose uuid() carries the switch state into Inductor's FX-graph cache key.

    Config switches are already part of the key through ``config.save_config_portable``; pattern
    switches are not, so without this a cached artifact compiled under one pattern state would be
    served for another.
    """

    def __init__(self, state: Iterable[str]) -> None:
        self.state = frozenset(state)

    def __call__(self, graph: torch.fx.Graph) -> None:
        # does not modify the graph; records an op histogram so callers can see rewrite effects
        for node in graph.nodes:
            if node.op == "call_function":
                GRAPH_OPS[str(node.target)] += 1
            elif node.op in ("call_method", "call_module"):
                GRAPH_OPS[f"{node.op}:{node.target}"] += 1
        return None

    def uuid(self) -> Any:
        payload = "rewrite_blame|" + torch.__version__ + "|" + "\n".join(sorted(self.state))
        return hashlib.sha256(payload.encode()).hexdigest()


def resolve_config_path(path: str):
    obj = inductor_config
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def config_switch_exists(sw: Switch) -> bool:
    try:
        obj, leaf = resolve_config_path(sw.config_path)
        return hasattr(obj, leaf)
    except AttributeError:
        return False


def prune_missing_config_switches(reg: Registry) -> tuple[Registry, list[str]]:
    """Drop config/option switches whose config path is absent in this torch; return (registry, dropped)."""
    kept, dropped = [], []
    for s in reg:
        if s.kind in ("config", "option") and not config_switch_exists(s):
            dropped.append(s.id)
        else:
            kept.append(s)
    if dropped:
        warnings.warn(f"dropping switches missing from torch {torch.__version__}: {dropped}")
    return Registry(kept), dropped


def config_patch_for_state(reg: Registry, state: State) -> dict[str, Any]:
    """Build the dict for ``torch._inductor.config.patch`` that realises ``state``."""
    patch: dict[str, Any] = {}
    option_dicts: dict[str, dict[str, Any]] = {}
    for s in reg:
        if s.kind == "config":
            patch[s.config_path] = s.on_value if s.id in state else s.off_value
        elif s.kind == "option":
            d = option_dicts.setdefault(s.config_path, {})
            if s.id in state:
                d[s.option_key] = dict(s.option_value)
    for path, d in option_dicts.items():
        patch[path] = d
    return patch


@contextlib.contextmanager
def active(reg: Registry, state: Iterable[str], *, tag_cache_key: bool = True):
    """Make ``state`` the live Inductor configuration for the duration of the block.

    Pattern switches must already be installed (see ``discover.install``); unknown ids raise.
    """
    global _DISABLED
    st = reg.validate_state(state)
    pattern_ids = {s.id for s in reg.of_kind("pattern")}
    prev_disabled = _DISABLED
    _DISABLED = frozenset(pattern_ids - st)
    patch = config_patch_for_state(reg, st)
    if tag_cache_key:
        patch["post_grad_custom_post_pass"] = SwitchSetPass(st)
    # A hit in Inductor's FX-graph cache skips every graph pass, which would hide which rules fired
    # and the post-grad op histogram. Measurements therefore always run the passes; the C++ kernel
    # cache (keyed by generated source) still makes recompiles cheap, and our own Store caches results.
    patch["fx_graph_cache"] = False
    import torch._functorch.config as functorch_config
    prev_autograd_cache = functorch_config.enable_autograd_cache
    functorch_config.enable_autograd_cache = False
    try:
        with inductor_config.patch(patch):
            yield st
    finally:
        _DISABLED = prev_disabled
        functorch_config.enable_autograd_cache = prev_autograd_cache


def snapshot_fired() -> dict[str, int]:
    return dict(FIRED)


def reset_fired() -> None:
    FIRED.clear()
    SUPPRESSED.clear()
    FIRINGS.clear()


def fired_since(before: dict[str, int]) -> dict[str, int]:
    out = {}
    for k, v in FIRED.items():
        d = v - before.get(k, 0)
        if d:
            out[k] = d
    return out
