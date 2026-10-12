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
    unmeasured: list[str] = field(default_factory=list)    # switches whose configuration failed to compile/measure
    level_shifts: int = 0                                  # times the running baseline moved to a new, confirmed level

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
    # Sanity check before spending hours: a fresh baseline run must agree with the session's
    # reference. If two fresh runs agree with each other but not with the reference, the reference
    # (measured earlier) is stale and the fresh level is used; if nothing agrees, abort.
    brep[0] += 1
    b1 = meas(base, brep[0]); n += 1
    if abs(b1 - baselines[0]) > tau:
        brep[0] += 1
        b2 = meas(base, brep[0]); n += 1
        if abs(b2 - b1) <= tau:
            baselines = [b1, b2]
        elif abs(b2 - baselines[0]) > tau:
            raise RuntimeError(f"pair scan aborted: baseline runs disagree ({baselines[0]:.3f}, {b1:.3f}, {b2:.3f} ms, tau {tau:.3f}); "
                               "the machine is unstable or the noise reference is stale")

    good_baselines: list[float] = [baselines[0]]
    pending: list[float] = []          # deviating baseline runs not yet accepted as a new level
    level_shifts = 0

    def running_base() -> float:
        recent = good_baselines[-5:]
        return sorted(recent)[len(recent) // 2]

    def fresh_baseline() -> tuple[bool, float]:
        """One more baseline run. A single run that deviates from the running baseline by more
        than tau is a disturbed stretch of machine time: counted and logged but not used as a
        reference. Two consecutive deviating runs that agree with each other are a genuine level
        shift (slow drift): the running baseline moves to the new level."""
        nonlocal n, drift_events, level_shifts
        brep[0] += 1
        b = meas(base, brep[0]); n += 1
        ok = abs(b - running_base()) <= tau
        baselines.append(b)
        if ok:
            good_baselines.append(b)
            pending.clear()
        else:
            drift_events += 1
            if pending and abs(pending[-1] - b) <= tau:
                good_baselines[:] = [pending[-1], b]     # accept the new level
                level_shifts += 1
                pending.clear()
                ok = True
            else:
                pending.append(b)
        return ok, b

    import math
    singles: dict[str, float] = {}
    singles_abs: dict[str, float] = {}
    unmeasured: list[str] = []
    for a in ids:
        v = meas(toggled(base, a)); n += 1
        if v != v:                      # NaN: the configuration could not be measured
            unmeasured.append(a)
            continue
        singles_abs[a] = v
        singles[a] = v - running_base()
        if n % rebaseline_every == 0:
            fresh_baseline()
    ids = [a for a in ids if a not in unmeasured]
    rows, superadd, masking = [], [], []
    combos = list(itertools.combinations(ids, 2))
    if max_pairs is not None:
        combos = combos[:max_pairs]
    for a, b in combos:
        st = toggled(toggled(base, a), b)
        v = meas(st); n += 1
        if v != v:
            rows.append({"a": a, "b": b, "delta_ms": None, "delta_a": singles[a], "delta_b": singles[b], "interaction_ms": 0.0,
                         "superadditive": False, "masking": False, "pair_slow": False, "same_program": False, "note": "unmeasured"})
            continue
        d = v - running_base()
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
            settled = False
            for attempt in range(2):
                ok, bval = fresh_baseline()             # interleaved baseline separates the two runs in time
                if ok:
                    settled = True
                    break
                note = f"baseline run {bval:.3f} deviated from running baseline: disturbed, retrying"
            if not settled:
                # the machine never settled: this pair's measurements are not trustworthy
                flag_super = flag_mask = False
                d = None; extra = 0.0
                note = "disturbed machine during confirmation: unmeasured"
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
        kind = ""
        if flag_super:
            # both switches help alone but the pair does not add up: the gains overlap (same cause)
            kind = "redundant gains" if (singles[a] < -tau or singles[b] < -tau) and d is not None and d >= min(singles[a], singles[b]) - tau else "compounding costs"
        elif flag_mask:
            kind = "masking"
        row = {"a": a, "b": b, "delta_ms": d, "delta_a": singles[a], "delta_b": singles[b], "interaction_ms": extra,
               "abs_ms": (d + running_base()) if d is not None else None, "abs_a": singles_abs[a], "abs_b": singles_abs[b],
               "superadditive": flag_super, "masking": flag_mask, "kind": kind,
               "pair_slow": d is not None and d > tau and not same_program.get(frozenset(st)),
               "same_program": bool(same_program.get(frozenset(st))), "note": note}
        rows.append(row)
        if flag_super:
            superadd.append(row)
        if flag_mask:
            masking.append(row)
    # Final pass: the running baseline may have moved during the scan (level shifts). Re-express
    # every delta against the consolidated baseline (median of all accepted baseline runs) so
    # singles measured early and pairs measured late are on the same footing, and re-derive flags.
    final_base = sorted(good_baselines)[len(good_baselines) // 2]
    for a in singles:
        singles[a] = singles_abs[a] - final_base
    superadd, masking = [], []
    for r in rows:
        if r.get("abs_ms") is None:
            continue
        r["delta_a"], r["delta_b"] = singles[r["a"]], singles[r["b"]]
        r["delta_ms"] = r["abs_ms"] - final_base
        r["interaction_ms"] = r["delta_ms"] - r["delta_a"] - r["delta_b"]
        confirmed = (not confirm) or ("confirmed" in r.get("note", ""))
        if r.get("same_program"):
            r["superadditive"] = r["masking"] = False
        else:
            r["superadditive"] = confirmed and r["interaction_ms"] > tau
            r["masking"] = confirmed and r["delta_ms"] < max(r["delta_a"], r["delta_b"]) - tau
        r["pair_slow"] = r["delta_ms"] > tau and not r.get("same_program")
        if r["superadditive"]:
            r["kind"] = "redundant gains" if (r["delta_a"] < -tau or r["delta_b"] < -tau) and r["delta_ms"] >= min(r["delta_a"], r["delta_b"]) - tau else "compounding costs"
            superadd.append(r)
        elif r["masking"]:
            r["kind"] = "masking"
            masking.append(r)
        else:
            r["kind"] = ""
    rows.sort(key=lambda r: -abs(r["interaction_ms"]) if r.get("delta_ms") is not None else 0.0)
    superadd.sort(key=lambda r: -r["interaction_ms"])
    masking.sort(key=lambda r: r["interaction_ms"])
    res = PairScanResult(final_base, tau, singles, rows, superadd, masking, n)
    res.baselines = baselines
    res.drift_events = drift_events
    res.level_shifts = level_shifts
    res.unmeasured = unmeasured
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
