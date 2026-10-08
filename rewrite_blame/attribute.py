"""Attribution: find the minimal set of rewrites that reproduces a slowdown.

Setting. A *fast* reference state F and a *slow* state S (both frozensets of ON switch ids).
Their symmetric difference, expressed as signed *change tokens* ("+id" = turn id on, "-id" =
turn id off), is the set of candidate causes D. A judge answers, for any c ⊆ D, whether applying
the changes c to F reproduces the slowdown.

We run Zeller & Hildebrandt's ddmin over D. Its result C is 1-minimal: removing any single
element makes the slowdown disappear. A pair-interaction check then tells whether C's elements
also hurt individually or only together.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Sequence

from .judge import Judge, Verdict


def changes_between(fast_state: Iterable[str], slow_state: Iterable[str]) -> list[str]:
    """Signed tokens that transform ``fast_state`` into ``slow_state``."""
    F, S = frozenset(fast_state), frozenset(slow_state)
    return sorted([f"+{x}" for x in S - F] + [f"-{x}" for x in F - S])


def apply_changes(state: Iterable[str], changes: Iterable[str]) -> frozenset:
    st = set(state)
    for t in changes:
        if t.startswith("+"):
            st.add(t[1:])
        elif t.startswith("-"):
            st.discard(t[1:])
        else:
            raise ValueError(f"change token must start with + or -: {t!r}")
    return frozenset(st)


def _split(seq: Sequence, n: int) -> list[list]:
    """Split ``seq`` into ``n`` nearly equal, non-empty chunks (n <= len(seq))."""
    k, m = divmod(len(seq), n)
    out, start = [], 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        out.append(list(seq[start:start + size]))
        start += size
    return [c for c in out if c]


def ddmin(changes: Iterable[str], test: Callable[[Iterable[str]], Verdict], *, on_step=None) -> list[str]:
    """Return a 1-minimal subset of ``changes`` for which ``test`` says SLOW.

    Precondition: test(changes) == SLOW. ``test`` is expected to memoise; ``on_step`` (optional)
    receives (phase, candidate, verdict) for tracing.
    """
    cs = list(changes)
    if not cs:
        return cs
    if test(cs) != Verdict.SLOW:
        raise ValueError("ddmin precondition violated: the full change set does not reproduce the slowdown")
    n = 2
    while len(cs) >= 2:
        subsets = _split(cs, n)
        reduced = False
        # 1. reduce to a subset
        for s in subsets:
            v = test(s)
            if on_step:
                on_step("subset", s, v)
            if v == Verdict.SLOW:
                cs, n, reduced = s, 2, True
                break
        if reduced:
            continue
        # 2. reduce to a complement
        for s in subsets:
            comp = [c for c in cs if c not in s]
            if not comp:
                continue
            v = test(comp)
            if on_step:
                on_step("complement", comp, v)
            if v == Verdict.SLOW:
                cs, n, reduced = comp, max(n - 1, 2), True
                break
        if reduced:
            continue
        # 3. increase granularity
        if n >= len(cs):
            break
        n = min(len(cs), 2 * n)
    return cs


@dataclass
class AttributionResult:
    fast_state: list[str]
    slow_state: list[str]
    candidates: list[str]           # signed change tokens F -> S
    culprits: list[str]             # 1-minimal subset of the tokens
    singles: dict[str, str]         # verdict of each culprit alone (slow/fast)
    interaction: bool               # True iff |C| >= 2 and no single culprit reproduces
    judge_calls: int
    trace: list[dict]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def kind(self) -> str:
        if not self.culprits:
            return "none"
        if len(self.culprits) == 1:
            return "single"
        return "interaction" if self.interaction else "multiple"


def attribute(fast_state: Iterable[str], slow_state: Iterable[str], judge: Judge, *,
              check_singles: bool = True, verbose: bool = False) -> AttributionResult:
    """Blame the changes between ``fast_state`` and ``slow_state`` using ``judge``.

    ``judge(c)`` receives a set of signed change tokens and must evaluate the configuration
    ``apply_changes(fast_state, c)``.
    """
    F, S = frozenset(fast_state), frozenset(slow_state)
    if F == S:
        raise ValueError("fast and slow states are identical: nothing to attribute")
    D = changes_between(F, S)
    notes: list[str] = []

    def log(*a):
        if verbose:
            print(*a)

    steps: list[dict] = []

    def on_step(phase, cand, v):
        steps.append({"phase": phase, "size": len(cand), "verdict": v.value})
        log(f"  [{phase}] |c|={len(cand)} -> {v.value}")

    if judge(frozenset()) == Verdict.SLOW:
        notes.append("fast reference itself judged SLOW: threshold too tight or reference mis-specified")
        return AttributionResult(sorted(F), sorted(S), D, [], {}, False, judge.calls,
                                 [asdict(r) | {"verdict": r.verdict.value} for r in judge.trace], notes)
    if judge(D) != Verdict.SLOW:
        notes.append("slow state not reproduced by the judge (difference below threshold)")
        return AttributionResult(sorted(F), sorted(S), D, [], {}, False, judge.calls,
                                 [asdict(r) | {"verdict": r.verdict.value} for r in judge.trace], notes)

    culprits = ddmin(D, judge, on_step=on_step)
    singles: dict[str, str] = {}
    interaction = False
    if check_singles and len(culprits) >= 2:
        for c in culprits:
            singles[c] = judge([c]).value
        interaction = all(v == Verdict.FAST.value for v in singles.values())
        if interaction:
            notes.append(f"{len(culprits)} rewrites only hurt together (no single one reproduces)")
    elif len(culprits) == 1:
        singles[culprits[0]] = Verdict.SLOW.value
    trace = [asdict(r) | {"verdict": r.verdict.value} for r in judge.trace]
    return AttributionResult(sorted(F), sorted(S), D, culprits, singles, interaction, judge.calls, trace, notes)


# ----------------------------------------------------------------------------------------
# Iterative attribution: find every independent cause, not just one 1-minimal set.
# ----------------------------------------------------------------------------------------

@dataclass
class IterativeResult:
    fast_state: list[str]
    slow_state: list[str]
    rounds: list[dict]              # one entry per ddmin round: culprits, kind, judge_calls, residual info
    all_culprits: list[str]         # union of culprits over rounds, in discovery order
    residual_explained: bool        # True iff fast + all culprits is judged as slow as the slow state
    judge_calls: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def attribute_iteratively(fast_state: Iterable[str], slow_state: Iterable[str], make_judge: Callable[[frozenset], Judge], *,
                          max_rounds: int = 6, verbose: bool = False) -> IterativeResult:
    """Repeat ddmin, moving each round's culprits into the fast state, until the remaining
    difference no longer reproduces a slowdown.

    ``make_judge(fast_state)`` must return a Judge whose ``judge(c)`` evaluates
    ``apply_changes(fast_state, c)`` *relative to that fast_state* (i.e. the reference is
    re-measured for the new fast state each round). Rounds stop when the full remaining change
    set is judged FAST, when no candidates remain, or after ``max_rounds``.
    """
    F, S = frozenset(fast_state), frozenset(slow_state)
    rounds: list[dict] = []
    found: list[str] = []
    calls = 0
    notes: list[str] = []
    cur = F
    explained = False
    for r in range(1, max_rounds + 1):
        if cur == S:
            explained = True
            break
        judge = make_judge(cur)
        res = attribute(cur, S, judge, verbose=verbose)
        calls += judge.calls
        rounds.append({"round": r, "fast_state_size": len(cur), "candidates": len(res.candidates), "culprits": res.culprits,
                       "kind": res.kind(), "interaction": res.interaction, "judge_calls": judge.calls, "notes": res.notes})
        if not res.culprits:
            # either the residual is within noise (explained) or the reference itself is slow (mis-specified)
            explained = any("not reproduced" in n for n in res.notes)
            break
        found.extend(res.culprits)
        cur = apply_changes(cur, res.culprits)
    else:
        notes.append(f"stopped after {max_rounds} rounds with residual still slow")
    return IterativeResult(sorted(F), sorted(S), rounds, found, explained, calls, notes)
