"""Enumerate TorchInductor's pattern-matcher rewrite rules and install per-rule switches.

Inductor registers most rules lazily (``lazy_init`` functions decorated with
``init_once_fakemode``), so we first trigger those initialisers, then walk the ``fx_passes``
modules for every ``PatternMatcherPass`` instance and wrap each entry's ``extra_check``.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Iterable

import torch
from torch._inductor import pattern_matcher as pm

from .apply import GatedExtraCheck, prune_missing_config_switches
from .switches import Registry, Switch, builtin_registry

# modules that hold PatternMatcherPass instances (directly, in lists, or in dicts)
PASS_MODULES = [
    "torch._inductor.fx_passes.joint_graph",
    "torch._inductor.fx_passes.post_grad",
    "torch._inductor.fx_passes.pre_grad",
    "torch._inductor.fx_passes.split_cat",
    "torch._inductor.fx_passes.freezing_patterns",
    "torch._inductor.fx_passes.mkldnn_fusion",
    "torch._inductor.fx_passes.binary_folding",
    "torch._inductor.fx_passes.decompose_mem_bound_mm",
    "torch._inductor.fx_passes.b2b_gemm",
    "torch._inductor.fx_passes.efficient_conv_bn_eval",
    "torch._inductor.fx_passes.misc_patterns",
    "torch._inductor.fx_passes.quantization",
    "torch._inductor.fx_passes.apply_gumbel_max_trick",
]

_FAMILY_BY_MODULE = {
    "joint_graph": "joint", "post_grad": "post_grad", "pre_grad": "pre_grad", "split_cat": "optimus",
    "freezing_patterns": "freezing", "mkldnn_fusion": "freezing", "binary_folding": "freezing",
    "decompose_mem_bound_mm": "optimus", "b2b_gemm": "post_grad", "efficient_conv_bn_eval": "pre_grad",
    "misc_patterns": "joint", "quantization": "quant", "apply_gumbel_max_trick": "pre_grad",
}


class _WarmupNet(torch.nn.Module):
    """Tiny graph touching attention, conv+bn, split/cat, mm and a dtype round trip, so that every
    lazy pattern initialiser Inductor has (sfdp, pad_mm, misc, mkldnn, freezing) runs exactly the way
    a real compile runs it (same functools.cache keys)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.lin = torch.nn.Linear(16, 16)

    def forward(self, x, img):
        q = k = v = x.view(1, 2, 8, 8)
        att = ((q @ k.transpose(-2, -1)) / 8 ** 0.5).softmax(-1) @ v
        a, b = torch.split(att.reshape(1, 8, 16), 8, dim=-1)
        y = torch.cat([a, b], dim=-1).double().float()
        y = self.lin(y) + 1.0 / torch.sqrt(y.abs() + 1)
        z = torch.relu(self.bn(self.conv(img))).mean()
        return y + z


def warmup_compile(freezing_too: bool = True) -> list[str]:
    """Trigger Inductor's lazy pattern registration via real (tiny) compiles. Returns what ran."""
    import torch._inductor.config as cfg
    ran = []
    net = _WarmupNet().eval()
    x = torch.randn(1, 8, 16)
    img = torch.randn(1, 4, 8, 8)
    prev = torch._dynamo.config.cache_size_limit
    try:
        with torch.no_grad():
            torch._dynamo.reset()
            torch.compile(net)(x, img)
            ran.append("default")
            if freezing_too:
                torch._dynamo.reset()
                with cfg.patch({"freezing": True}):
                    torch.compile(net)(x, img)
                ran.append("freezing")
    finally:
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = prev
    return ran


# kept for backwards compatibility of the CLI; direct lazy-init calls are NOT safe because they
# populate functools.cache with different arguments than compile_fx uses (-> duplicate patterns).
def trigger_lazy_inits(device: torch.device | None = None) -> list[str]:
    return warmup_compile()


def _iter_passes(module) -> Iterable[tuple[str, pm.PatternMatcherPass]]:
    modshort = module.__name__.rsplit(".", 1)[-1]
    for name, val in vars(module).items():
        if name.startswith("__"):
            continue
        if isinstance(val, pm.PatternMatcherPass):
            yield f"{modshort}.{name}", val
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                if isinstance(v, pm.PatternMatcherPass):
                    yield f"{modshort}.{name}[{i}]", v
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, pm.PatternMatcherPass):
                    yield f"{modshort}.{name}[{k}]", v


def _entry_name(entry: pm.PatternEntry, target) -> str:
    name = getattr(entry, "pattern_name", None)
    if name:
        return name
    handler = getattr(entry, "handler", None)
    if handler is not None:
        n = getattr(handler, "__name__", None)
        if n and n != "<lambda>":
            return n
        fn = getattr(handler, "func", None)  # functools.partial
        if fn is not None and getattr(fn, "__name__", None):
            return fn.__name__
    chk = getattr(entry, "extra_check", None)
    if isinstance(chk, GatedExtraCheck):
        chk = chk.original
    n = getattr(chk, "__name__", None)
    if n and n not in ("_return_true", "<lambda>"):
        return f"replacement_{n}"
    return f"replacement_on_{_target_name(target)}"


def _target_name(target) -> str:
    n = getattr(target, "__name__", None) or str(target)
    return n.replace("aten.", "").replace("torch.ops.", "")


def discover(*, trigger: bool = True, base: Registry | None = None) -> Registry:
    """Build the full registry (config + option + pattern switches) and install the pattern gates.

    Idempotent within a process: entries that are already gated keep their switch id.
    """
    if trigger:
        warmup_compile()
    reg = base if base is not None else builtin_registry()
    reg, _dropped = prune_missing_config_switches(reg)

    seen_entries: dict[int, str] = {}
    seen_passes: set[int] = set()
    for modname in PASS_MODULES:
        try:
            module = importlib.import_module(modname)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"cannot import {modname}: {e}")
            continue
        family = _FAMILY_BY_MODULE.get(modname.rsplit(".", 1)[-1], "other")
        for label, pass_obj in _iter_passes(module):
            if id(pass_obj) in seen_passes:
                continue
            seen_passes.add(id(pass_obj))
            # patterns: dict[(op, target)] -> list[PatternEntry]; one entry can sit under several targets
            targets_by_entry: dict[int, list[str]] = {}
            order: list[pm.PatternEntry] = []
            for (op, target), entries in pass_obj.patterns.items():
                for e in entries:
                    if id(e) not in targets_by_entry:
                        targets_by_entry[id(e)] = []
                        order.append(e)
                    targets_by_entry[id(e)].append(_target_name(target))
            for e in order:
                if id(e) in seen_entries:
                    continue
                if isinstance(e.extra_check, GatedExtraCheck):
                    sid = e.extra_check.switch_id
                    seen_entries[id(e)] = sid
                    if sid not in reg:
                        reg.add(Switch(id=sid, kind="pattern", family=family, pass_label=label,
                                       entry_kind=type(e).__name__, targets=tuple(sorted(set(targets_by_entry[id(e)])))))
                    continue
                name = _entry_name(e, None if not targets_by_entry[id(e)] else targets_by_entry[id(e)][0])
                sid = reg.unique_id(f"{family}/{label}/{name}")
                e.extra_check = GatedExtraCheck(sid, e.extra_check)
                seen_entries[id(e)] = sid
                reg.add(Switch(id=sid, kind="pattern", family=family, pass_label=label,
                               description=f"{type(e).__name__} on {', '.join(sorted(set(targets_by_entry[id(e)])))}",
                               entry_kind=type(e).__name__, targets=tuple(sorted(set(targets_by_entry[id(e)])))))
    return reg


def summarize(reg: Registry) -> dict:
    out: dict = {"total": len(reg), "by_kind": {}, "by_family": {}, "by_pass": {}}
    for s in reg:
        out["by_kind"][s.kind] = out["by_kind"].get(s.kind, 0) + 1
        out["by_family"][s.family] = out["by_family"].get(s.family, 0) + 1
        if s.kind == "pattern":
            out["by_pass"][s.pass_label] = out["by_pass"].get(s.pass_label, 0) + 1
    return out
