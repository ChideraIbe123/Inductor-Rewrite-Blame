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

    def to_dict(self) -> dict:
        return asdict(self)


def pair_scan(base_state: Iterable[str], ids: Iterable[str], measure_ms: Callable[..., float], tau: float,
              *, max_pairs: int | None = None, toggled=None, confirm: bool = True) -> PairScanResult:
    """Toggle every switch in ``ids`` alone and every pair together on top of ``base_state``.

    ``toggled(state, sid)`` flips one switch (default: set symmetric difference). A pair is
    *superadditive* when its extra cost beyond the sum of its singles exceeds ``tau``; it is
    *masking* when the pair is faster than the slower single by more than ``tau``. With
    ``confirm``, a pair that would be flagged is measured a second time (``rep=2``) and keeps the
    flag only if both runs agree (AND rule), so one noisy process cannot create an interaction.
    ``measure_ms(state, rep=0)``; functions without a ``rep`` parameter are accepted.
    """
    base = frozenset(base_state)
    ids = list(ids)
    if toggled is None:
        def toggled(st, sid):
            return frozenset(set(st) ^ {sid})

    def meas(st, rep=0):
        try:
            return measure_ms(st, rep)
        except TypeError:
            return measure_ms(st)
    n = 0
    base_ms = meas(base); n += 1
    singles: dict[str, float] = {}
    for a in ids:
        singles[a] = meas(toggled(base, a)) - base_ms; n += 1
    rows, superadd, masking = [], [], []
    combos = list(itertools.combinations(ids, 2))
    if max_pairs is not None:
        combos = combos[:max_pairs]
    for a, b in combos:
        st = toggled(toggled(base, a), b)
        d = meas(st) - base_ms; n += 1
        extra = d - singles[a] - singles[b]
        if confirm and (extra > tau or d < max(singles[a], singles[b]) - tau):
            # AND rule: a flagged pair keeps its flag only if an independent second run agrees
            d2 = meas(st, 2) - base_ms; n += 1
            extra2 = d2 - singles[a] - singles[b]
            agree = (extra > tau and extra2 > tau) or (d < max(singles[a], singles[b]) - tau and d2 < max(singles[a], singles[b]) - tau)
            if agree:
                d = 0.5 * (d + d2)                 # report the mean of the two agreeing runs
                extra = d - singles[a] - singles[b]
            else:
                d = max(d, d2) if d2 < d else min(d, d2)   # keep the run closer to "no interaction"
                extra = d - singles[a] - singles[b]
                if extra > tau:                    # still flagged by arithmetic: force unflag (runs disagreed)
                    extra = tau
                if d < max(singles[a], singles[b]) - tau:
                    d = max(singles[a], singles[b]) - tau
        row = {"a": a, "b": b, "delta_ms": d, "delta_a": singles[a], "delta_b": singles[b], "interaction_ms": extra,
               "superadditive": extra > tau, "masking": d < max(singles[a], singles[b]) - tau,
               "pair_slow": d > tau}
        rows.append(row)
        if row["superadditive"]:
            superadd.append(row)
        if row["masking"]:
            masking.append(row)
    rows.sort(key=lambda r: -abs(r["interaction_ms"]))
    superadd.sort(key=lambda r: -r["interaction_ms"])
    masking.sort(key=lambda r: r["interaction_ms"])
    return PairScanResult(base_ms, tau, singles, rows, superadd, masking, n)


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
