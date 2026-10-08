"""Switch registry: a uniform on/off handle over TorchInductor graph rewrites.

Three kinds of switch exist:

* ``config``  -- a ``torch._inductor.config`` flag (``epilogue_fusion``, ``cpp.weight_prepack`` ...).
* ``option``  -- a key inside one of Inductor's fusion-option dicts
  (``pre_grad_fusion_options`` / ``post_grad_fusion_options``), which is how the opt-in
  "optimus" rewrites such as split/cat normalisation or ``decompose_mm_pass`` are enabled.
* ``pattern`` -- one entry of a ``PatternMatcherPass`` (a single rewrite rule), switched by
  wrapping the entry's ``extra_check`` so it refuses to fire while disabled.

A *state* is simply the ``frozenset`` of switch ids that are ON. ``Registry.default_state()``
reproduces Inductor's out-of-the-box behaviour.

This module is importable without torch; the torch-touching parts live in ``apply.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

KINDS = ("config", "option", "pattern")


@dataclass(frozen=True)
class Switch:
    id: str
    kind: str
    family: str
    description: str = ""
    default_on: bool = True
    # kind == "config"
    config_path: str | None = None
    on_value: Any = None
    off_value: Any = None
    # kind == "option": config_path names the dict, option_key the key to insert when ON
    option_key: str | None = None
    option_value: Any = field(default_factory=dict)
    # kind == "pattern"
    pass_label: str | None = None
    entry_kind: str | None = None  # GraphPatternEntry / ReplacementPatternEntry / LoweringPatternEntry
    targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown switch kind {self.kind!r}")
        if self.kind in ("config", "option") and not self.config_path:
            raise ValueError(f"{self.kind} switch {self.id} needs config_path")
        if self.kind == "option" and not self.option_key:
            raise ValueError(f"option switch {self.id} needs option_key")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["targets"] = list(self.targets)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Switch":
        d = dict(d)
        d["targets"] = tuple(d.get("targets", ()))
        return cls(**d)


State = frozenset  # frozenset[str] of switch ids that are ON


def state_hash(state: Iterable[str]) -> str:
    """Stable short hash of a state (order independent)."""
    payload = "\n".join(sorted(state)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class Registry:
    """An ordered collection of switches with helpers for states."""

    def __init__(self, switches: Iterable[Switch] = ()) -> None:
        self._switches: dict[str, Switch] = {}
        for s in switches:
            self.add(s)

    # -- construction -------------------------------------------------------------------
    def add(self, s: Switch) -> None:
        if s.id in self._switches:
            raise KeyError(f"duplicate switch id {s.id}")
        self._switches[s.id] = s

    def unique_id(self, base: str) -> str:
        """Return ``base`` or ``base#N`` so that the id is unused in this registry."""
        if base not in self._switches:
            return base
        n = 2
        while f"{base}#{n}" in self._switches:
            n += 1
        return f"{base}#{n}"

    # -- lookup ---------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._switches)

    def __iter__(self):
        return iter(self._switches.values())

    def __contains__(self, sid: str) -> bool:
        return sid in self._switches

    def __getitem__(self, sid: str) -> Switch:
        return self._switches[sid]

    def ids(self) -> list[str]:
        return list(self._switches)

    def of_kind(self, kind: str) -> list[Switch]:
        return [s for s in self if s.kind == kind]

    def of_family(self, family: str) -> list[Switch]:
        return [s for s in self if s.family == family]

    def families(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self:
            seen.setdefault(s.family, None)
        return list(seen)

    # -- states ---------------------------------------------------------------------------
    def default_state(self) -> State:
        return frozenset(s.id for s in self if s.default_on)

    def all_on(self) -> State:
        return frozenset(self.ids())

    def validate_state(self, state: Iterable[str]) -> State:
        unknown = sorted(set(state) - set(self._switches))
        if unknown:
            raise KeyError(f"unknown switch ids: {unknown[:5]}{'...' if len(unknown) > 5 else ''}")
        return frozenset(state)

    def toggled(self, state: Iterable[str], sid: str) -> State:
        st = set(state)
        if sid in st:
            st.remove(sid)
        else:
            st.add(sid)
        return frozenset(st)

    def describe_diff(self, a: Iterable[str], b: Iterable[str]) -> dict[str, list[str]]:
        a, b = set(a), set(b)
        return {"only_in_a": sorted(a - b), "only_in_b": sorted(b - a)}

    # -- (de)serialisation ------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps([s.to_dict() for s in self], indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Registry":
        return cls(Switch.from_dict(d) for d in json.loads(text))

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path) -> "Registry":
        with open(path) as f:
            return cls.from_json(f.read())


# ----------------------------------------------------------------------------------------
# Config / option switches. Values verified against torch 2.14.1 defaults; ``apply.py`` drops
# any whose config path does not exist in the running torch (and says so).
# ----------------------------------------------------------------------------------------

def _cfg(id_, path, family, desc, on=True, off=False, default_on=True):
    return Switch(id=id_, kind="config", family=family, description=desc, default_on=default_on,
                  config_path=path, on_value=on, off_value=off)


def _opt(id_, dict_path, key, family, desc, value=None):
    return Switch(id=id_, kind="option", family=family, description=desc, default_on=False,
                  config_path=dict_path, option_key=key, option_value=value or {})


CONFIG_SWITCHES: list[Switch] = [
    # master switches over whole pass groups (useful for coarse bisection, like CompilerBisector)
    _cfg("master/pattern_matcher", "pattern_matcher", "master",
         "All PatternMatcherPass-based rewrites (joint, post-grad, pre-grad)"),
    _cfg("master/pre_grad_passes", "use_pre_grad_passes", "master", "Run pre-grad FX passes at all"),
    _cfg("master/post_grad_passes", "use_post_grad_passes", "master", "Run post-grad FX passes at all"),
    # joint-graph level
    _cfg("joint/constant_folding", "joint_graph_constant_folding", "joint",
         "Fold tensors with a uniform value into aten.full"),
    _cfg("joint/scatter_upon_const_tensor", "optimize_scatter_upon_const_tensor", "joint",
         "Rewrite scatter onto a constant tensor as a where()"),
    # post-grad level
    _cfg("post_grad/reorder_for_locality", "reorder_for_locality", "post_grad",
         "Topologically reorder nodes so producers sit next to consumers"),
    _cfg("post_grad/linear_binary_folding", "enable_linear_binary_folding", "post_grad",
         "Fold linear followed by binary op into the linear (freezing)", default_on=False),
    # lowering-level rewrites
    _cfg("lowering/conv_1x1_as_mm", "conv_1x1_as_mm", "lowering",
         "Lower 1x1 convolutions as matrix multiplies", default_on=False),
    _cfg("lowering/layout_optimization", "layout_optimization", "lowering",
         "Channels-last layout optimisation for convolutions"),
    _cfg("lowering/freezing", "freezing", "freezing",
         "Constant-fold parameters and enable mkldnn conv/linear fusion + weight prepacking",
         default_on=False),
    _cfg("freezing/weight_prepack", "cpp.weight_prepack", "freezing",
         "Pre-pack conv/linear weights into mkldnn layout (only under freezing)"),
    # scheduler-level fusion knobs (step two of the pipeline)
    _cfg("sched/epilogue_fusion", "epilogue_fusion", "scheduler", "Fuse pointwise epilogues into templates/reductions"),
    _cfg("sched/prologue_fusion", "prologue_fusion", "scheduler", "Fuse pointwise prologues into templates"),
    _cfg("sched/aggressive_fusion", "aggressive_fusion", "scheduler",
         "Fuse even when no memory traffic is saved", default_on=False),
    _cfg("sched/loop_ordering_after_fusion", "loop_ordering_after_fusion", "scheduler",
         "Reorder loops after fusion to enable more fusion"),
    _cfg("sched/inplace_buffers", "inplace_buffers", "scheduler", "Reuse input buffers for outputs in place"),
    _cfg("sched/split_reductions", "split_reductions", "scheduler", "Split large reductions into two stages"),
    _cfg("sched/reorder_for_peak_memory", "reorder_for_peak_memory", "scheduler", "Reorder nodes to reduce peak memory"),
    _cfg("sched/loop_index_inversion", "loop_index_inversion_in_fusion", "scheduler",
         "Invert loop indices to make more fusions legal"),
    _cfg("sched/max_fusion_size", "max_fusion_size", "scheduler",
         "Maximum nodes per fused kernel (off = 1, i.e. no horizontal/vertical growth)", on=64, off=1),
    _cfg("sched/unroll_reductions", "unroll_reductions_threshold", "scheduler",
         "Unroll reductions with few elements (off = 0)", on=8, off=0),
    # CPU codegen knobs that change how fused kernels are formed
    _cfg("cpp/tiling_heuristics", "cpp.enable_tiling_heuristics", "cpp", "Tiling heuristics in the C++ backend"),
    _cfg("cpp/loop_tail_vec", "cpp.enable_loop_tail_vec", "cpp", "Vectorise loop tails"),
    _cfg("cpp/horizontal_fusion", "cpp.max_horizontal_fusion_size", "cpp",
         "Max kernels merged horizontally in C++ (off = 1)", on=16, off=1),
    _cfg("cpp/decompose_tanh", "cpp.use_decompose_tanh", "cpp",
         "Decompose tanh into exp-based formula", default_on=False),
    _cfg("cpp/concat_linear", "cpp.enable_concat_linear", "cpp",
         "Concatenate linears sharing an input into one GEMM", default_on=False),
]

# Opt-in rewrite families enabled through Inductor's option dicts ("optimus" passes).
OPTION_SWITCHES: list[Switch] = [
    _opt("optimus/decompose_mm_pass", "post_grad_fusion_options", "decompose_mm_pass", "optimus",
         "Decompose skinny mm/bmm/addmm into pointwise mul + sum"),
    _opt("optimus/normalization_pass", "pre_grad_fusion_options", "normalization_pass", "optimus",
         "Normalise split/cat/stack call forms (prerequisite of other split/cat passes)"),
    _opt("optimus/remove_split_with_size_one_pass", "pre_grad_fusion_options", "remove_split_with_size_one_pass",
         "optimus", "Drop splits that produce one chunk"),
    _opt("optimus/merge_splits_pass", "pre_grad_fusion_options", "merge_splits_pass", "optimus",
         "Merge consecutive splits"),
    _opt("optimus/merge_getitem_cat_pass", "pre_grad_fusion_options", "merge_getitem_cat_pass", "optimus",
         "Merge getitem+cat chains"),
    _opt("optimus/split_cat_pass", "pre_grad_fusion_options", "split_cat_pass", "optimus",
         "Cancel split followed by cat"),
    _opt("optimus/unbind_stack_pass", "pre_grad_fusion_options", "unbind_stack_pass", "optimus",
         "Cancel unbind followed by stack"),
    _opt("optimus/mutate_cat_pass", "pre_grad_fusion_options", "mutate_cat_pass", "optimus",
         "Rewrite cat of split outputs as a slice"),
    _opt("optimus/batch_linear", "pre_grad_fusion_options", "batch_linear", "optimus",
         "Batch independent linears into one bmm"),
    _opt("optimus/batch_layernorm", "pre_grad_fusion_options", "batch_layernorm", "optimus",
         "Batch independent layer norms"),
    _opt("optimus/batch_tanh", "pre_grad_fusion_options", "batch_tanh", "optimus", "Batch independent tanh ops"),
    _opt("optimus/batch_relu", "pre_grad_fusion_options", "batch_relu", "optimus", "Batch independent relu ops"),
    _opt("optimus/batch_sigmoid", "pre_grad_fusion_options", "batch_sigmoid", "optimus", "Batch independent sigmoids"),
    _opt("optimus/batch_aten_mul", "post_grad_fusion_options", "batch_aten_mul", "optimus",
         "Batch independent aten.mul ops (post-grad)"),
    _opt("optimus/batch_aten_add", "post_grad_fusion_options", "batch_aten_add", "optimus",
         "Batch independent aten.add ops (post-grad)"),
    _opt("optimus/batch_linear_post_grad", "post_grad_fusion_options", "batch_linear_post_grad", "optimus",
         "Batch independent addmm/mm ops (post-grad)"),
]


def builtin_registry() -> Registry:
    """Registry with the config/option switches only (pattern switches need torch; see apply.py)."""
    return Registry(CONFIG_SWITCHES + OPTION_SWITCHES)
