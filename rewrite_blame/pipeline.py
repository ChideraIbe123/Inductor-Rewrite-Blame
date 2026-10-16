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
from .attribute import AttributionResult, apply_changes, attribute, attribute_iteratively, changes_between
from .interactions import pair_scan, tau_scan
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
    # Timed measurements are only comparable within one session (machine speed drifts between
    # days, thermal states, VPN/background load). Default: today's date. Compile-only results
    # are deterministic and shared across sessions.
    session: str | None = None
    # laptops throttle under sustained load: pause between timed measurements for this fraction of
    # the previous measurement's wall time (0 on a server), and re-measure a run whose calls got
    # slower as it went (trend_ratio > 1.10) once after a longer pause
    cooldown_factor: float = 0.5
    retry_unsteady: bool = True

    @property
    def protocol(self) -> str:
        return f"w{self.warmup}r{self.rounds}i{self.iters}"

    @property
    def session_tag(self) -> str:
        if self.session:
            return self.session
        import datetime
        return datetime.date.today().isoformat()


class Runner:
    def __init__(self, cfg: RunnerConfig, reg: Registry | None = None) -> None:
        self.cfg = cfg
        self.reg = reg if reg is not None else refresh_registry(cfg.registry_path, log=self.log)
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
                force: bool = False, check_correct: bool = True, guards: str | None = None,
                mode: str | None = None) -> Measurement:
        st = self.reg.validate_state(state)
        proto = (f"{self.cfg.protocol}@{self.cfg.session_tag}" if time_it else "compile_only") + (f"/rep{rep}" if rep else "")
        if guards:
            import hashlib
            proto += "+guards:" + hashlib.sha256(Path(guards).read_bytes()).hexdigest()[:10]
        if mode:
            proto += f"+mode:{mode}"
        if not force:
            cached = self.store.get(model, self.env, state_hash(st), proto)
            if cached is not None and not cached.get("error"):
                self.cache_hits += 1
                return Measurement.from_dict(cached)
        t0 = time.time()
        m = worker.run(model, st, self.cfg.registry_path, threads=self.threads, warmup=self.cfg.warmup,
                       rounds=self.cfg.rounds, iters=self.cfg.iters, time_it=time_it, check_correct=check_correct,
                       timeout_s=self.cfg.timeout_s, cache_dir=self.cfg.cache_dir, guards=guards, mode=mode)
        self.new_measurements += 1
        wall = time.time() - t0
        if time_it and not m.error:
            if self.cfg.retry_unsteady and m.timing.get("unsteady"):
                # the machine got slower during the run (throttling): cool down, measure again
                pause = min(120.0, 20.0 + wall)
                self.log(f"  ~ unsteady run (trend x{m.timing.get('trend_ratio', 0):.2f}); cooling {pause:.0f}s and re-measuring")
                time.sleep(pause)
                m2 = worker.run(model, st, self.cfg.registry_path, threads=self.threads, warmup=self.cfg.warmup,
                                rounds=self.cfg.rounds, iters=self.cfg.iters, time_it=time_it, check_correct=check_correct,
                                timeout_s=self.cfg.timeout_s, cache_dir=self.cfg.cache_dir, guards=guards, mode=mode)
                self.new_measurements += 1
                if not m2.error:
                    m2.protocol["retried_unsteady"] = {"first_median": m.median_ms, "first_trend": m.timing.get("trend_ratio")}
                    m = m2
            if self.cfg.cooldown_factor > 0:
                time.sleep(min(60.0, self.cfg.cooldown_factor * wall))
        d = m.to_dict()
        d["protocol_key"] = proto
        d["rep"] = rep
        d["session"] = self.cfg.session_tag if time_it else None
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
        steady = [m for m in ms if not m.timing.get("unsteady")]
        used = steady if len(steady) >= max(5, runs - 2) else ms
        nm = noise_model([m.median_ms for m in used], k=k, floor_frac=floor_frac)
        nm.n_unsteady = len(ms) - len(steady)
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
              include_options: bool = True, only_graph_changing: bool = True, noise: NoiseModel | None = None,
              rebaseline_every: int = 8, confirm: bool = True) -> dict:
        base = self.reg.default_state() if base is None else frozenset(base)
        if noise is None:
            noise, _ = self.noise(model, base)
        m0s = [self.measure(model, base, time_it=False)]
        ref = noise.center
        if ids is None:
            ver = self.verify_switches(model, base, include_options)
            ids = [r["switch"] for r in ver["effects"] if (not only_graph_changing) or r["changes_graph"]]
        rows = []
        # the machine drifts (thermal state, background load) over a long sweep: re-measure the base
        # state every few toggles and compare each toggle with the running baseline (median of the
        # last three baseline runs), not with the noise centre measured minutes or hours earlier
        baselines = [ref]
        drift_events = 0
        brep = 200
        for i, sid in enumerate(ids):
            if i and i % rebaseline_every == 0:
                brep += 1
                mb = self.measure(model, base, rep=brep)
                if not mb.error:
                    if abs(mb.median_ms - ref) > noise.tau:
                        drift_events += 1
                        self.log(f"  ~ baseline drifted: {mb.median_ms:.3f} vs centre {ref:.3f} (tau {noise.tau:.3f})")
                    baselines.append(mb.median_ms)
                    recent = sorted(baselines[-3:])
                    ref = recent[len(recent) // 2]
            st = self.reg.toggled(base, sid)
            m = self.measure(model, st)
            if m.error:
                rows.append({"switch": sid, "error": m.error.splitlines()[0][:200]})
                continue
            d = m.median_ms - ref
            same = bool(m0s) and bool(m0s[0].program_hash) and m.program_hash == m0s[0].program_hash
            slower, faster = d > noise.tau and not same, d < -noise.tau and not same
            confirm_note = ""
            if (slower or faster) and confirm:
                # the same AND rule as the judge: a verdict needs an independent second process that agrees
                m2 = self.measure(model, st, rep=2)
                if not m2.error:
                    d2 = m2.median_ms - ref
                    agree = (slower and d2 > noise.tau) or (faster and d2 < -noise.tau)
                    confirm_note = f"second run {m2.median_ms:.3f} ms ({'agrees' if agree else 'disagrees'})"
                    if agree:
                        d = 0.5 * (d + d2)
                    else:
                        slower = faster = False
                        d = d2 if abs(d2) < abs(d) else d
            rows.append({"switch": sid, "direction": "off" if sid in base else "on", "median_ms": m.median_ms,
                         "delta_ms": d, "delta_pct": 100.0 * d / ref, "beyond_noise": slower or faster,
                         "slower": slower, "faster": faster, "confirmation": confirm_note,
                         "same_program": same, "kernel_count": m.kernel_count, "baseline_ms": ref,
                         "alloc_bytes": m.code.get("alloc_bytes"), "correct": m.correct})
        rows.sort(key=lambda r: -abs(r.get("delta_ms", 0)))
        return {"model": model, "env": self.env, "base": sorted(base), "reference_ms": noise.center, "noise": noise.to_dict(),
                "baselines": baselines, "drift_events": drift_events, "rows": rows}

    # ---------------------------------------------------------------- attribution
    def timing_judge(self, model: str, fast_state: State, noise: NoiseModel) -> TimingJudge:
        m_ref = self.measure(model, fast_state, time_it=False)
        ref_hash = m_ref.program_hash if not m_ref.error else ""

        def measure_ms(changes: frozenset, rep: int = 0):
            if not changes:
                # the empty change set *is* the reference: its value is the noise-model centre
                # (7 independent runs), not one more noisy measurement
                return noise.center
            st = apply_changes(fast_state, changes)
            m = self.measure(model, st, rep=rep)
            if m.error:
                raise RuntimeError(m.error)
            same = bool(ref_hash) and m.program_hash == ref_hash
            return (m.median_ms, same)

        def baseline_ms(rep: int) -> float:
            m = self.measure(model, fast_state, rep=rep)
            if m.error:
                raise RuntimeError(m.error)
            return m.median_ms
        return TimingJudge(measure_ms, reference_ms=noise.center, tau=noise.tau, baseline_ms=baseline_ms)

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


    # ---------------------------------------------------------------- iterative attribution
    def attribute_all(self, model: str, fast_state: State, slow_state: State, *, noise: NoiseModel | None = None,
                      noise_runs: int = 7, max_rounds: int = 6, k: float = 3.0, floor_frac: float = 0.01) -> dict:
        """Repeat attribution until the residual difference is within noise; each round re-measures
        the noise of its own fast state (cheap: cached after the first round for identical states)."""
        fast_state, slow_state = frozenset(fast_state), frozenset(slow_state)
        t0 = time.time()
        noises: dict[frozenset, NoiseModel] = {}
        if noise is not None:
            noises[fast_state] = noise

        def make_judge(cur: frozenset):
            if cur not in noises:
                noises[cur], _ = self.noise(model, cur, runs=noise_runs, k=k, floor_frac=floor_frac)
            return self.timing_judge(model, cur, noises[cur])

        res = attribute_iteratively(fast_state, slow_state, make_judge, max_rounds=max_rounds, verbose=self.cfg.verbose)
        m_fast = self.measure(model, fast_state)
        m_slow = self.measure(model, slow_state)
        groups = []
        cur = fast_state
        for r in res.rounds:
            if not r["culprits"]:
                continue
            nxt = apply_changes(cur, r["culprits"])
            m_prev, m_next = self.measure(model, cur), self.measure(model, nxt)
            groups.append({"round": r["round"], "culprits": r["culprits"], "kind": r["kind"],
                           "delta_ms": m_next.median_ms - m_prev.median_ms,
                           "delta_pct": 100.0 * (m_next.median_ms - m_prev.median_ms) / m_fast.median_ms,
                           "kernels": (m_prev.kernel_count, m_next.kernel_count),
                           "alloc_bytes": (m_prev.code.get("alloc_bytes"), m_next.code.get("alloc_bytes"))})
            cur = nxt
        return {"model": model, "env": self.env, "judge": "timing", "iterative": res.to_dict(), "groups": groups,
                "fast_ms": m_fast.median_ms, "slow_ms": m_slow.median_ms,
                "explained_ms": sum(g["delta_ms"] for g in groups), "total_ms": m_slow.median_ms - m_fast.median_ms,
                "noises": {state_hash(k_): v.to_dict() for k_, v in noises.items()},
                "wall_s": time.time() - t0, "new_measurements": self.new_measurements, "cache_hits": self.cache_hits}

    # ---------------------------------------------------------------- pairwise interactions
    def interactions(self, model: str, base: State | None = None, ids: Iterable[str] | None = None,
                     noise: NoiseModel | None = None, max_pairs: int | None = None, only_graph_changing: bool = True) -> dict:
        """Toggle each candidate alone and in pairs on top of ``base``; flag super-additive pairs."""
        base = self.reg.default_state() if base is None else frozenset(base)
        if noise is None:
            noise, _ = self.noise(model, base)
        if ids is None:
            ver = self.verify_switches(model, base, include_options=True)
            ids = [r["switch"] for r in ver["effects"] if (not only_graph_changing) or r["changes_graph"]]
        ids = list(ids)

        m_base = self.measure(model, base, time_it=False)
        base_hash = m_base.program_hash if not m_base.error else ""

        failures: list[dict] = []

        def measure_ms(state: frozenset, rep: int = 0):
            if state == base and rep == 0:
                return noise.center
            m = self.measure(model, state, rep=rep)
            if m.error:
                # a failed compile must not abort a scan of hundreds of pairs: report NaN, record it
                failures.append({"state_diff": self.reg.describe_diff(base, state), "error": m.error.splitlines()[0][:200]})
                return (float("nan"), False)
            return (m.median_ms, bool(base_hash) and m.program_hash == base_hash)
        res = pair_scan(base, ids, measure_ms, noise.tau, max_pairs=max_pairs, toggled=self.reg.toggled)
        return {"model": model, "env": self.env, "base": sorted(base), "ids": ids, "noise": noise.to_dict(),
                "failures": failures, **res.to_dict()}

    # ---------------------------------------------------------------- threshold sensitivity
    def tau_sensitivity(self, model: str, fast_state: State, slow_state: State, ks: Iterable[float] = (2.0, 3.0, 4.0, 6.0),
                        noise_runs: int = 7) -> dict:
        """Re-run the (single-round) attribution for several k; measurements are cached, so only
        ddmin paths that visit new states cost compiles."""
        fast_state, slow_state = frozenset(fast_state), frozenset(slow_state)
        _, ms = self.noise(model, fast_state, runs=noise_runs)
        medians = [m.median_ms for m in ms]

        def run(k: float) -> dict:
            nm = noise_model(medians, k=k)
            J = self.timing_judge(model, fast_state, nm)
            r = attribute(fast_state, slow_state, J, verbose=False)
            return {"tau": nm.tau, "culprits": r.culprits, "kind": r.kind(), "judge_calls": J.calls, "notes": r.notes}
        res = tau_scan(run, list(ks))
        return {"model": model, "env": self.env, **res.to_dict()}


    # ---------------------------------------------------------------- end-to-end guard evaluation
    ORACLE_SWITCHES = {"conv": ["lowering/layout_optimization"], "net": ["lowering/layout_optimization"],
                       "mlp": ["optimus/decompose_mm_wide"], "attn": ["*_sfdp_pattern_*_inference"]}

    def _oracle_switches(self, families: list[str], bundle) -> list[str]:
        """The switches a guard set can touch: switch:<id> families name theirs; sweep families map to a rule."""
        import fnmatch
        out = []
        for fam in families:
            if fam.startswith("switch:"):
                out.append(fam[len("switch:"):])
            for pat in self.ORACLE_SWITCHES.get(fam, []):
                out += [i for i in self.reg.ids() if fnmatch.fnmatch(i, pat)] if ("*" in pat) else [pat]
        return [i for i in dict.fromkeys(out) if i in self.reg]

    def _confirmed_delta(self, model, state, nm, m0_hash, **kw):
        """Median change vs the noise centre, confirmed by a second time-separated run (AND rule); returns
        (delta_ms, verdict, measurement)."""
        m1 = self.measure(model, state, **kw)
        if m1.error:
            return None, "error", m1
        d = m1.median_ms - nm.center
        verdict = "same"
        same_prog = bool(m0_hash) and m1.program_hash == m0_hash
        if abs(d) > nm.tau and not same_prog:
            m2 = self.measure(model, state, rep=2, **kw)
            if not m2.error:
                d2 = m2.median_ms - nm.center
                if d > nm.tau and d2 > nm.tau:
                    verdict = "slower"
                elif d < -nm.tau and d2 < -nm.tau:
                    verdict = "faster"
                d = 0.5 * (d + d2)
        return d, verdict, m1

    def guards_eval(self, models: Iterable[str], bundle_path: str, families: list[str], noise_runs: int = 7,
                    base: State | None = None, oracle: bool = True, max_autotune: bool = False) -> dict:
        """Stock Inductor vs Inductor with guards added one family at a time, on real models.
        Each comparison is confirmed by the two-run rule against the model's noise threshold. With ``oracle``,
        every switch the guards can touch is also toggled by hand (best of on/off = what a perfect guard would
        get) and the guard's gain is reported as a fraction of that; with ``max_autotune`` the stock compile in
        mode="max-autotune" is measured as the baseline Inductor already offers."""
        import tempfile
        from .guard_models import GuardBundle
        full = GuardBundle.load(bundle_path)
        steps = []
        for i in range(1, len(families) + 1):
            sub = GuardBundle()
            for fam in families[:i]:
                if fam in full.guards:
                    g = full.guards[fam]
                    sub.add(fam, g["kind"], g["clf"], g["names"], g["threshold"])
            f = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, prefix=f"guards_{'_'.join(families[:i])}_")
            sub.save(f.name)
            steps.append(("+".join(families[:i]), f.name))
        base = self.reg.default_state() if base is None else frozenset(base)
        oracle_ids = self._oracle_switches(families, full) if oracle else []
        rows = []
        for model in models:
            nm, _ = self.noise(model, base, runs=noise_runs)
            m0 = self.measure(model, base)
            row = {"model": model, "default_ms": nm.center, "tau_ms": nm.tau, "steps": []}
            if oracle_ids:
                # a perfect per-model guard: the best of toggling each touchable switch (confirmed), or stock
                toggles = {}
                for sid in oracle_ids:
                    st = frozenset(base - {sid}) if sid in base else frozenset(base | {sid})
                    d, verdict, _m = self._confirmed_delta(model, st, nm, m0.program_hash)
                    if d is not None:
                        toggles[sid] = {"delta_pct": 100 * d / nm.center, "verdict": verdict}
                best = min((v["delta_pct"] for v in toggles.values() if v["verdict"] == "faster"), default=0.0)
                row["oracle"] = {"toggles": toggles, "best_delta_pct": best}
            if max_autotune:
                d, verdict, _m = self._confirmed_delta(model, base, nm, m0.program_hash, mode="max-autotune")
                row["max_autotune"] = {"delta_pct": 100 * d / nm.center, "verdict": verdict} if d is not None else {"error": True}
            for label, gpath in steps:
                d, verdict, m1 = self._confirmed_delta(model, base, nm, m0.program_hash, guards=gpath)
                if d is None:
                    row["steps"].append({"guards": label, "error": m1.error.splitlines()[0][:200]})
                    continue
                decisions = [x for x in m1.guard_decisions if x.get("blocked")]
                step = {"guards": label, "median_ms": m1.median_ms, "delta_pct": 100 * d / nm.center, "verdict": verdict,
                        "blocked": sum(m1.guarded.values()) + int(bool(m1.protocol.get("guard_disabled_layout"))) + len(m1.protocol.get("guard_changed_switches", [])),
                        "decisions": decisions[:10], "same_program": bool(m0.program_hash) and m1.program_hash == m0.program_hash}
                if oracle_ids and row["oracle"]["best_delta_pct"] < 0:
                    step["oracle_fraction"] = min(step["delta_pct"], 0.0) / row["oracle"]["best_delta_pct"]
                row["steps"].append(step)
                self.log(f"  {model} guards={label}: {m1.median_ms:.3f} vs {nm.center:.3f} ms ({100*d/nm.center:+.1f}%) {verdict}, blocked {step['blocked']}")
            rows.append(row)
        return {"env": self.env, "families": families, "bundle": bundle_path, "base": sorted(base), "rows": rows,
                "oracle_switches": oracle_ids, "max_autotune": max_autotune}


def refresh_registry(path, log=print) -> Registry:
    """Discover the switch universe of *this* machine in a subprocess and merge it into ``path``.

    Pattern universes differ by platform (e.g. oneDNN rules only exist on x86 builds), so a
    registry copied from another machine would silently shrink the universe. Existing ids are kept
    so stored measurements stay valid; new switches are appended."""
    import subprocess, tempfile
    path = Path(path)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    r = subprocess.run([sys.executable, "-m", "rewrite_blame", "--registry", tmp, "discover"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"discovery failed:\n{r.stderr[-2000:]}")
    fresh = Registry.load(tmp)
    if path.exists():
        saved = Registry.load(path)
        new = [sw for sw in fresh if sw.id not in saved]
        missing = [i for i in saved.ids() if i not in fresh]
        if new or missing:
            log(f"registry refresh: +{len(new)} new switches, {len(missing)} saved switches not present on this machine")
            for sw in new:
                saved.add(sw)
            saved.save(path)
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.save(path)
    log(f"registry written to {path} ({len(fresh)} switches)")
    return fresh


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
    program_changed = bool(m0.program_hash) and m.program_hash != m0.program_hash
    changes_graph = program_changed or bool(ops_changed) or kernels_changed or externs_changed or d["alloc_bytes"][0] != d["alloc_bytes"][1]
    status = "changes_graph" if changes_graph else ("fires_only" if fired_changed else "no_effect")
    return {"switch": sid, "kind": reg[sid].kind, "family": reg[sid].family, "direction": direction, "status": status,
            "changes_graph": changes_graph, "program_changed": program_changed, "ops_changed": ops_changed, "fired_changed": fired_changed,
            "kernel_count": d["kernel_count"], "alloc_bytes": d["alloc_bytes"], "extern_call_count": d["extern_call_count"],
            "kernels_only_in_base": d["kernels_only_in_a"], "kernels_only_in_toggled": d["kernels_only_in_b"],
            "externs_only_in_base": d["externs_only_in_a"], "externs_only_in_toggled": d["externs_only_in_b"],
            "suppressed": m.suppressed, "correct": m.correct}
