"""Orchestration: cached measurements through subprocess workers, noise estimation, switch
verification, sweeps and attribution runs."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import env
from .attribute import AttributionResult, apply_changes, attribute, changes_between
from .judge import Judge, MetricJudge, TimingJudge
from .measure import Measurement
from .stats import NoiseModel, noise_model
from .store import Store
from .switches import Registry, State, state_hash
from . import worker


def parse_state_spec(reg: Registry, spec: str) -> State:
    """``default``, ``all``, ``none``, optionally followed by ``+id`` / ``-id`` terms or ``~family``
    (toggle every switch of a family) and ``-glob`` with ``*`` (fnmatch) e.g.
    ``default-*sfdp*`` or ``default+optimus/decompose_mm_pass``. Terms are separated by the sign."""
    import fnmatch
    import re
    spec = spec.strip()
    m = re.match(r"^(default|all|none)(.*)$", spec)
    if not m:
        raise ValueError(f"state spec must start with default|all|none: {spec!r}")
    base, rest = m.groups()
    st = set({"default": reg.default_state(), "all": reg.all_on(), "none": frozenset()}[base])
    for sign, term in re.findall(r"([+\-~])([^+\-~]+)", rest):
        term = term.strip()
        if sign == "~":
            ids = [s.id for s in reg.of_family(term)]
        elif "*" in term or "?" in term:
            ids = [i for i in reg.ids() if fnmatch.fnmatch(i, term)]
        else:
            if term not in reg:
                raise KeyError(f"unknown switch {term!r}")
            ids = [term]
        if not ids:
            raise KeyError(f"no switch matches {term!r}")
        for i in ids:
            if sign == "+":
                st.add(i)
            elif sign == "-":
                st.discard(i)
            else:
                st.symmetric_difference_update({i})
    return frozenset(st)


@dataclass
class RunnerConfig:
    registry_path: Path = Path("results/switches.json")
    store_path: Path = Path("results/measurements.sqlite")
    cache_dir: str | None = None
    threads: int = 0
    warmup: int = 10
    rounds: int = 5
    iters: int = 10
    timeout_s: int = 3600
    verbose: bool = True

    @property
    def protocol(self) -> str:
        return f"w{self.warmup}r{self.rounds}i{self.iters}"


class Runner:
    def __init__(self, cfg: RunnerConfig, reg: Registry | None = None) -> None:
        self.cfg = cfg
        self.reg = reg or Registry.load(cfg.registry_path)
        self.store = Store(cfg.store_path)
        self.threads = cfg.threads or env.default_threads()
        self.env = env.fingerprint(self.threads)
        self.new_measurements = 0
        self.cache_hits = 0

    def log(self, *a) -> None:
        if self.cfg.verbose:
            print(*a, file=sys.stderr, flush=True)

    # ---------------------------------------------------------------- measurements
    def measure(self, model: str, state: Iterable[str], *, time_it: bool = True, rep: int = 0,
                force: bool = False, check_correct: bool = True) -> Measurement:
        st = self.reg.validate_state(state)
        proto = (self.cfg.protocol if time_it else "compile_only") + (f"/rep{rep}" if rep else "")
        if not force:
            cached = self.store.get(model, self.env, state_hash(st), proto)
            if cached is not None and not cached.get("error"):
                self.cache_hits += 1
                return Measurement.from_dict(cached)
        t0 = time.time()
        m = worker.run(model, st, self.cfg.registry_path, threads=self.threads, warmup=self.cfg.warmup,
                       rounds=self.cfg.rounds, iters=self.cfg.iters, time_it=time_it, check_correct=check_correct,
                       timeout_s=self.cfg.timeout_s, cache_dir=self.cfg.cache_dir)
        self.new_measurements += 1
        d = m.to_dict()
        d["protocol_key"] = proto
        d["rep"] = rep
        self.store.put(model, self.env, state_hash(st), d, proto)
        if m.error:
            self.log(f"  ! {model} {state_hash(st)} failed: {m.error.splitlines()[0][:200]}")
        else:
            self.log(f"  measured {model} {state_hash(st)} proto={proto}: kernels={m.kernel_count} "
                     f"median={m.median_ms and round(m.median_ms, 3)} ms compile={m.compile_s:.1f}s wall={time.time()-t0:.0f}s")
        return m

    # ---------------------------------------------------------------- noise
    def noise(self, model: str, state: Iterable[str], runs: int = 7, k: float = 3.0, floor_frac: float = 0.01,
              spacing_s: float = 0.0) -> tuple[NoiseModel, list[Measurement]]:
        ms = []
        for r in range(1, runs + 1):
            m = self.measure(model, state, rep=r)
            if m.error:
                raise RuntimeError(f"noise run {r} failed: {m.error[:300]}")
            ms.append(m)
            if spacing_s:
                time.sleep(spacing_s)
        nm = noise_model([m.median_ms for m in ms], k=k, floor_frac=floor_frac)
        return nm, ms

    # ---------------------------------------------------------------- verification of switches
    def candidate_universe(self, model: str, base: State | None = None, include_options: bool = True) -> dict[str, list[str]]:
        """Switches whose toggling *can* change this model's compilation, split by kind.

        A pattern switch that never fired in the base compile cannot change anything when turned
        off, so the pattern candidates are exactly the fired ones; config switches are always
        candidates; option (opt-in) switches are candidates when ``include_options``."""
        base = self.reg.default_state() if base is None else frozenset(base)
        m0 = self.measure(model, base, time_it=False)
        if m0.error:
            raise RuntimeError(m0.error)
        fired = [i for i in m0.fired if i in self.reg]
        cfg = [s.id for s in self.reg.of_kind("config")]
        opts = [s.id for s in self.reg.of_kind("option")] if include_options else []
        return {"pattern": fired, "config": cfg, "option": opts}

    def verify_switches(self, model: str, base: State | None = None, include_options: bool = True,
                        only: Iterable[str] | None = None) -> dict:
        base = self.reg.default_state() if base is None else frozenset(base)
        m0 = self.measure(model, base, time_it=False)
        uni = self.candidate_universe(model, base, include_options)
        ids = [i for k in uni.values() for i in k]
        if only is not None:
            ids = [i for i in ids if i in set(only)]
        rows = []
        for sid in ids:
            st = self.reg.toggled(base, sid)
            direction = "off" if sid in base else "on"
            m = self.measure(model, st, time_it=False)
            rows.append(switch_effect(self.reg, sid, direction, m0, m))
        return {"model": model, "base": sorted(base), "universe": uni, "effects": rows,
                "baseline": {"kernel_count": m0.kernel_count, "fired": m0.fired, "externs": m0.code.get("extern_calls")}}

    # ---------------------------------------------------------------- sweeps
    def sweep(self, model: str, base: State | None = None, ids: Iterable[str] | None = None,
              include_options: bool = True, only_graph_changing: bool = True, noise: NoiseModel | None = None) -> dict:
        base = self.reg.default_state() if base is None else frozenset(base)
        if noise is None:
            noise, _ = self.noise(model, base)
        m0s = [Measurement.from_dict(d) for d in self.store.all(model=model, env=self.env)
               if d.get("state_hash") == state_hash(base) and d.get("timing") and not d.get("error")]
        ref = noise.center
        if ids is None:
            ver = self.verify_switches(model, base, include_options)
            ids = [r["switch"] for r in ver["effects"] if (not only_graph_changing) or r["changes_graph"]]
        rows = []
        for sid in ids:
            st = self.reg.toggled(base, sid)
            m = self.measure(model, st)
            if m.error:
                rows.append({"switch": sid, "error": m.error.splitlines()[0][:200]})
                continue
            d = m.median_ms - ref
            rows.append({"switch": sid, "direction": "off" if sid in base else "on", "median_ms": m.median_ms,
                         "delta_ms": d, "delta_pct": 100.0 * d / ref, "beyond_noise": abs(d) > noise.tau,
                         "slower": d > noise.tau, "faster": d < -noise.tau, "kernel_count": m.kernel_count,
                         "alloc_bytes": m.code.get("alloc_bytes"), "correct": m.correct})
        rows.sort(key=lambda r: -abs(r.get("delta_ms", 0)))
        return {"model": model, "env": self.env, "base": sorted(base), "reference_ms": ref, "noise": noise.to_dict(),
                "rows": rows}

    # ---------------------------------------------------------------- attribution
    def timing_judge(self, model: str, fast_state: State, noise: NoiseModel) -> TimingJudge:
        def measure_ms(changes: frozenset) -> float:
            st = apply_changes(fast_state, changes)
            m = self.measure(model, st)
            if m.error:
                raise RuntimeError(m.error)
            return m.median_ms
        return TimingJudge(measure_ms, reference_ms=noise.center, tau=noise.tau)

    def metric_judge(self, model: str, fast_state: State, metric: str = "kernel_count", delta: float = 0.0,
                     higher_is_slow: bool = True) -> MetricJudge:
        def metric_of(m: Measurement) -> float:
            if metric in m.code:
                return float(m.code[metric])
            if metric in m.metrics:
                return float(m.metrics[metric])
            raise KeyError(metric)
        m0 = self.measure(model, fast_state, time_it=False)
        if m0.error:
            raise RuntimeError(m0.error)

        def measure(changes: frozenset) -> float:
            m = self.measure(model, apply_changes(fast_state, changes), time_it=False)
            if m.error:
                raise RuntimeError(m.error)
            return metric_of(m)
        return MetricJudge(measure, reference_value=metric_of(m0), delta=delta, higher_is_slow=higher_is_slow, name=metric)

    def attribute(self, model: str, fast_state: State, slow_state: State, *, judge: str = "timing",
                  noise: NoiseModel | None = None, metric: str = "kernel_count", noise_runs: int = 7) -> dict:
        fast_state, slow_state = frozenset(fast_state), frozenset(slow_state)
        t0 = time.time()
        if judge == "timing":
            if noise is None:
                noise, _ = self.noise(model, fast_state, runs=noise_runs)
            J: Judge = self.timing_judge(model, fast_state, noise)
            m_fast = self.measure(model, fast_state)
            m_slow = self.measure(model, slow_state)
        else:
            J = self.metric_judge(model, fast_state, metric=metric)
            m_fast = self.measure(model, fast_state, time_it=False)
            m_slow = self.measure(model, slow_state, time_it=False)
        res = attribute(fast_state, slow_state, J, verbose=self.cfg.verbose)
        culprit_state = apply_changes(fast_state, res.culprits)
        m_culprit = self.measure(model, culprit_state, time_it=(judge == "timing")) if res.culprits else None
        return {
            "model": model, "env": self.env, "judge": judge, "metric": metric if judge != "timing" else "median_ms",
            "noise": noise.to_dict() if noise else None,
            "result": res.to_dict(), "kind": res.kind(),
            "fast": m_fast.to_dict(), "slow": m_slow.to_dict(), "culprit_state": m_culprit.to_dict() if m_culprit else None,
            "wall_s": time.time() - t0, "new_measurements": self.new_measurements, "cache_hits": self.cache_hits,
        }


def switch_effect(reg: Registry, sid: str, direction: str, m0: Measurement, m: Measurement) -> dict:
    """Compare a toggled compilation with the base compilation."""
    from .codegen_stats import CodeStats, diff_stats
    if m.error:
        return {"switch": sid, "direction": direction, "error": m.error.splitlines()[0][:200], "changes_graph": False}
    g0, g1 = m0.graph_ops, m.graph_ops
    ops_changed = {k: (g0.get(k, 0), g1.get(k, 0)) for k in set(g0) | set(g1) if g0.get(k, 0) != g1.get(k, 0)}
    f0, f1 = m0.fired, m.fired
    fired_changed = {k: (f0.get(k, 0), f1.get(k, 0)) for k in set(f0) | set(f1) if f0.get(k, 0) != f1.get(k, 0)}
    d = diff_stats(CodeStats.from_dict(m0.code), CodeStats.from_dict(m.code))
    kernels_changed = d["kernel_count"][0] != d["kernel_count"][1] or bool(d["kernels_only_in_a"] or d["kernels_only_in_b"])
    externs_changed = bool(d["externs_only_in_a"] or d["externs_only_in_b"])
    changes_graph = bool(ops_changed) or kernels_changed or externs_changed or d["alloc_bytes"][0] != d["alloc_bytes"][1]
    status = "changes_graph" if changes_graph else ("fires_only" if fired_changed else "no_effect")
    return {"switch": sid, "kind": reg[sid].kind, "family": reg[sid].family, "direction": direction, "status": status,
            "changes_graph": changes_graph, "ops_changed": ops_changed, "fired_changed": fired_changed,
            "kernel_count": d["kernel_count"], "alloc_bytes": d["alloc_bytes"], "extern_call_count": d["extern_call_count"],
            "kernels_only_in_base": d["kernels_only_in_a"], "kernels_only_in_toggled": d["kernels_only_in_b"],
            "externs_only_in_base": d["externs_only_in_a"], "externs_only_in_toggled": d["externs_only_in_b"],
            "suppressed": m.suppressed, "correct": m.correct}
