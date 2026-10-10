"""Pairwise interaction scan: which pairs of switches hurt more together than their parts?

Pure functions over a ``measure_ms(state) -> float`` callable so they can be tested with a
synthetic cost model and driven by the Runner (which memoises real measurements).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable


@dataclass
class PairScanResult:
    base_ms: float
    tau: float
    singles: dict[str, float]            # switch -> delta ms when toggled alone
    pairs: list[dict]                    # one row per pair
    superadditive: list[dict]            # pairs where delta(pair) - delta(a) - delta(b) > tau
    masking: list[dict]                  # pairs where delta(pair) < max(delta(a), delta(b)) - tau
    measurements: int
    baselines: list[float] = field(default_factory=list)   # interleaved baseline runs over the scan
    drift_events: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def pair_scan(base_state: Iterable[str], ids: Iterable[str], measure_ms: Callable[..., float], tau: float,
              *, max_pairs: int | None = None, toggled=None, confirm: bool = True, rebaseline_every: int = 20) -> PairScanResult:
    """Toggle every switch in ``ids`` alone and every pair together on top of ``base_state``.

    ``toggled(state, sid)`` flips one switch (default: set symmetric difference). A pair is
    *superadditive* when its extra cost beyond the sum of its singles exceeds ``tau``; it is
    *masking* when the pair is faster than the slower single by more than ``tau``.

    Long scans drift with the machine, so the base state is re-measured every
    ``rebaseline_every`` measurements and every delta is taken against the running baseline (the
    median of the last three baseline runs). With ``confirm``, a pair that would be flagged is
    re-measured after a fresh baseline run: the flag stands only if that baseline is within tau of
    the running baseline (machine not disturbed) and the second run agrees (AND rule).
    ``measure_ms(state, rep=0)``; functions without a ``rep`` parameter are accepted.
    """
    base = frozenset(base_state)
    ids = list(ids)
    if toggled is None:
        def toggled(st, sid):
            return frozenset(set(st) ^ {sid})

    same_program: dict[frozenset, bool] = {}

    def meas(st, rep=0):
        try:
            r = measure_ms(st, rep)
        except TypeError:
            r = measure_ms(st)
        if isinstance(r, tuple):
            same_program[frozenset(st)] = bool(r[1])
            return r[0]
        return r
    n = 0
    baselines: list[float] = [meas(base)]; n += 1
    brep = [100]
    drift_events = 0

    def running_base() -> float:
        recent = baselines[-3:]
        return sorted(recent)[len(recent) // 2]

    def fresh_baseline() -> tuple[bool, float]:
        nonlocal n, drift_events
        brep[0] += 1
        b = meas(base, brep[0]); n += 1
        ok = abs(b - running_base()) <= tau
        baselines.append(b)
        if not ok:
            drift_events += 1
        return ok, b

    singles: dict[str, float] = {}
    for a in ids:
        singles[a] = meas(toggled(base, a)) - running_base(); n += 1
        if n % rebaseline_every == 0:
            fresh_baseline()
    rows, superadd, masking = [], [], []
    combos = list(itertools.combinations(ids, 2))
    if max_pairs is not None:
        combos = combos[:max_pairs]
    for a, b in combos:
        st = toggled(toggled(base, a), b)
        d = meas(st) - running_base(); n += 1
        if n % rebaseline_every == 0:
            fresh_baseline()
        extra = d - singles[a] - singles[b]
        flag_super = extra > tau
        flag_mask = d < max(singles[a], singles[b]) - tau
        note = ""
        if same_program.get(frozenset(st)):
            # identical generated program to the base: the two switches did nothing together
            flag_super = flag_mask = False
            note = "identical generated program to the base"
        if confirm and (flag_super or flag_mask):
            ok, bval = fresh_baseline()                 # interleaved baseline separates the two runs in time
            if not ok:
                note = f"baseline run {bval:.3f} deviated from running baseline: disturbed, flag dropped"
                flag_super = flag_mask = False
            else:
                d2 = meas(st, 2) - running_base(); n += 1
                extra2 = d2 - singles[a] - singles[b]
                agree = (flag_super and extra2 > tau) or (flag_mask and d2 < max(singles[a], singles[b]) - tau)
                if agree:
                    d = 0.5 * (d + d2); extra = d - singles[a] - singles[b]
                    note = f"confirmed by a time-separated second run ({d2:+.3f})"
                else:
                    note = f"second run disagreed ({d2:+.3f}): flag dropped"
                    flag_super = flag_mask = False
                    d = d2; extra = d - singles[a] - singles[b]
        row = {"a": a, "b": b, "delta_ms": d, "delta_a": singles[a], "delta_b": singles[b], "interaction_ms": extra,
               "superadditive": flag_super, "masking": flag_mask, "pair_slow": d > tau and not same_program.get(frozenset(st)),
               "same_program": bool(same_program.get(frozenset(st))), "note": note}
        rows.append(row)
        if flag_super:
            superadd.append(row)
        if flag_mask:
            masking.append(row)
    rows.sort(key=lambda r: -abs(r["interaction_ms"]))
    superadd.sort(key=lambda r: -r["interaction_ms"])
    masking.sort(key=lambda r: r["interaction_ms"])
    res = PairScanResult(baselines[0], tau, singles, rows, superadd, masking, n)
    res.baselines = baselines
    res.drift_events = drift_events
    return res


@dataclass
class TauScanResult:
    ks: list[float]
    rows: list[dict]                     # per k: tau, culprits, kind, judge_calls
    stable: bool                         # same culprit set for every k that reproduced the slowdown

    def to_dict(self) -> dict:
        return asdict(self)


def tau_scan(run_attribution: Callable[[float], dict], ks: Iterable[float]) -> TauScanResult:
    """Re-run an attribution for several threshold multipliers ``k`` (tau = k * MAD, with floor).

    ``run_attribution(k)`` must return a dict with keys tau, culprits, kind, judge_calls. With
    memoised measurements this costs no new compiles unless ddmin takes a different path.
    """
    rows = []
    for k in ks:
        r = run_attribution(k)
        rows.append({"k": k, **r})
    sets = {tuple(sorted(r["culprits"])) for r in rows if r["culprits"]}
    return TauScanResult(list(ks), rows, stable=len(sets) <= 1)
