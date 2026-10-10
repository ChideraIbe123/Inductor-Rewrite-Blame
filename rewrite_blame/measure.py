"""Compile a model under a switch state, capture generated code and metrics, time it."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import torch

from . import apply, codegen_stats, env
from .switches import Registry, state_diff, state_hash
from .stats import summarize


@dataclass
class Measurement:
    model: str
    state: list[str]
    state_hash: str
    env: str
    threads: int
    torch_version: str
    timestamp: float
    compile_s: float = 0.0
    code: dict = field(default_factory=dict)          # CodeStats.to_dict()
    metrics: dict = field(default_factory=dict)       # torch._inductor.metrics snapshot
    counters: dict = field(default_factory=dict)      # counters["inductor"] delta
    fired: dict = field(default_factory=dict)         # pattern switch id -> fire count
    suppressed: dict = field(default_factory=dict)    # disabled pattern switch id -> would-have-fired count
    graph_ops: dict = field(default_factory=dict)     # post-grad op histogram
    diff: dict = field(default_factory=dict)          # state relative to default: on_extra / off_defaults
    program_hash: str = ""                            # normalized hash of the generated program
    timing: dict = field(default_factory=dict)        # summarize(samples) + samples
    eager_timing: dict = field(default_factory=dict)
    correct: bool | None = None
    max_abs_err: float | None = None
    error: str | None = None
    protocol: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Measurement":
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def median_ms(self) -> float | None:
        return self.timing.get("median") if self.timing else None

    @property
    def kernel_count(self) -> int | None:
        return self.code.get("kernel_count") if self.code else None


def _flatten(out) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for o in out for t in _flatten(o)]
    if hasattr(out, "to_tuple"):
        return _flatten(out.to_tuple())
    if isinstance(out, dict):
        return [t for o in out.values() for t in _flatten(o)]
    return []


def _time_fn(fn, inputs, warmup: int, rounds: int, iters: int) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn(*inputs)
        samples = []
        for _ in range(rounds):
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                fn(*inputs)
                samples.append((time.perf_counter_ns() - t0) / 1e6)
    return samples


def _counters_snapshot() -> dict[str, int]:
    from torch._dynamo.utils import counters
    return {k: int(v) for k, v in counters["inductor"].items() if isinstance(v, int)}


def measure(reg: Registry, model_name: str, state: Iterable[str], *, threads: int | None = None,
            warmup: int = 10, rounds: int = 5, iters: int = 10, time_it: bool = True,
            check_correct: bool = True, time_eager: bool = False, seed: int = 0,
            dynamic: bool | None = False) -> Measurement:
    """Run one configuration in *this* process. Prefer ``worker.run`` for isolation."""
    from torch._inductor import metrics
    from torch._inductor.utils import run_and_get_code
    from . import models

    threads = threads or env.default_threads()
    env.setup_threads(threads)
    st = reg.validate_state(state)
    m = Measurement(model=model_name, state=sorted(st), state_hash=state_hash(st), env=env.fingerprint(threads),
                    threads=threads, torch_version=torch.__version__, timestamp=time.time(),
                    protocol={"warmup": warmup, "rounds": rounds, "iters": iters, "seed": seed, "dynamic": dynamic},
                    diff=state_diff(reg, st))
    try:
        model, inputs = models.build(model_name, seed=seed)
        with torch.no_grad():
            eager_out = _flatten(model(*inputs)) if check_correct else None
            if time_eager:
                m.eager_timing = summarize(_time_fn(model, inputs, warmup, rounds, iters))

        torch._dynamo.reset()
        metrics.reset()
        apply.reset_fired()
        apply.GRAPH_OPS.clear()
        # make Inductor compute per-graph byte estimates (only done when this artifact logger is on)
        torch._logging.set_logs(inductor_metrics=True)
        _lg = logging.getLogger("torch._inductor.compile_fx.__inductor_metrics")
        _lg.setLevel(logging.INFO)
        _lg.propagate = False
        if not any(isinstance(h, logging.NullHandler) for h in _lg.handlers):
            _lg.handlers[:] = [logging.NullHandler()]
        before = _counters_snapshot()

        with apply.active(reg, st), torch.no_grad():
            compiled = torch.compile(model, dynamic=dynamic)
            t0 = time.perf_counter()
            out, codes = run_and_get_code(compiled, *inputs)
            m.compile_s = time.perf_counter() - t0
            m.code = codegen_stats.analyze_sources(list(codes)).to_dict()
            m.program_hash = codegen_stats.program_hash(list(codes))
            m.metrics = {
                "generated_kernel_count": metrics.generated_kernel_count,
                "generated_cpp_vec_kernel_count": metrics.generated_cpp_vec_kernel_count,
                "num_bytes_accessed": metrics.num_bytes_accessed,
                "ir_nodes_pre_fusion": metrics.ir_nodes_pre_fusion,
                "num_loop_reordering": metrics.num_loop_reordering,
                "cpp_to_dtype_count": metrics.cpp_to_dtype_count,
            }
            after = _counters_snapshot()
            m.counters = {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0)}
            m.fired = dict(apply.FIRED)
            m.suppressed = dict(apply.SUPPRESSED)
            m.graph_ops = dict(apply.GRAPH_OPS)
            # entries registered only during this compile would have been un-switchable: record them
            from .discover import discover as _rediscover
            n_before = len(reg)
            _rediscover(trigger=False, base=reg)
            if len(reg) != n_before:
                m.protocol["late_registered_switches"] = reg.ids()[n_before:]

            if check_correct and eager_out is not None:
                comp = _flatten(out)
                errs = []
                ok = len(comp) == len(eager_out)
                for a, b in zip(comp, eager_out):
                    if a.shape != b.shape:
                        ok = False
                        continue
                    if a.is_floating_point():
                        errs.append((a.float() - b.float()).abs().max().item())
                        ok = ok and torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-3)
                    else:
                        ok = ok and torch.equal(a, b)
                m.correct = ok
                m.max_abs_err = max(errs) if errs else 0.0

            if time_it:
                samples = _time_fn(compiled, inputs, warmup, rounds, iters)
                m.timing = summarize(samples) | {"samples": samples}
    except Exception as e:  # keep the record, report the failure
        import traceback
        m.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    return m
