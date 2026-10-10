"""Judges decide whether a configuration reproduces the slowdown under investigation.

A judge maps a *candidate set* ``c`` (switch ids to turn ON on top of the fast reference) to a
verdict. ``TimingJudge`` uses real measurements and the noise threshold; ``MetricJudge`` uses a
deterministic compile-time metric (kernel count, bytes), which is what the unit tests on planted
slowdowns rely on; ``SyntheticJudge`` is a pure cost model for testing the search algorithm.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping


class Verdict(enum.Enum):
    SLOW = "slow"      # the slowdown is reproduced  (delta debugging's FAIL)
    FAST = "fast"      # not reproduced               (PASS)


@dataclass
class JudgeRecord:
    candidate: tuple[str, ...]
    verdict: Verdict
    value: float | None = None       # measured quantity for the candidate (ms, kernels, ...)
    reference: float | None = None   # the reference quantity it was compared against
    threshold: float | None = None
    note: str = ""


class Judge:
    """Base class: memoises verdicts per candidate set and records a trace."""

    def __init__(self) -> None:
        self.calls = 0            # number of *new* evaluations
        self.trace: list[JudgeRecord] = []
        self._memo: dict[frozenset, Verdict] = {}

    def __call__(self, candidate: Iterable[str]) -> Verdict:
        c = frozenset(candidate)
        if c in self._memo:
            return self._memo[c]
        self.calls += 1
        rec = self.evaluate(c)
        self.trace.append(rec)
        self._memo[c] = rec.verdict
        return rec.verdict

    def evaluate(self, candidate: frozenset) -> JudgeRecord:  # pragma: no cover - abstract
        raise NotImplementedError


class SyntheticJudge(Judge):
    """Cost model: cost(c) = base + sum(single[s] for s in c) + sum(pair[(a,b)] for both in c)
    (+ optional higher-order terms), with optional Gaussian noise; SLOW iff cost - base > tau."""

    def __init__(self, singles: Mapping[str, float] | None = None, pairs: Mapping[frozenset, float] | None = None,
                 groups: Mapping[frozenset, float] | None = None, tau: float = 1.0, noise_sd: float = 0.0,
                 seed: int = 0, base: float = 100.0) -> None:
        super().__init__()
        self.singles = dict(singles or {})
        self.pairs = {frozenset(k): v for k, v in (pairs or {}).items()}
        self.groups = {frozenset(k): v for k, v in (groups or {}).items()}
        self.tau = tau
        self.noise_sd = noise_sd
        self.base = base
        self._rng = random.Random(seed)

    def cost(self, c: frozenset) -> float:
        total = self.base + sum(self.singles.get(s, 0.0) for s in c)
        total += sum(v for k, v in self.pairs.items() if k <= c)
        total += sum(v for k, v in self.groups.items() if k <= c)
        if self.noise_sd:
            total += self._rng.gauss(0.0, self.noise_sd)
        return total

    def evaluate(self, candidate: frozenset) -> JudgeRecord:
        value = self.cost(candidate)
        verdict = Verdict.SLOW if value - self.base > self.tau else Verdict.FAST
        return JudgeRecord(tuple(sorted(candidate)), verdict, value, self.base, self.tau)


class MetricJudge(Judge):
    """SLOW iff metric(candidate) - metric(reference) > delta (or < -delta when ``higher_is_slow`` is False).

    ``measure(candidate) -> float`` must return the metric for "reference ∪ candidate".
    """

    def __init__(self, measure: Callable[[frozenset], float], reference_value: float, delta: float = 0.0,
                 higher_is_slow: bool = True, name: str = "metric") -> None:
        super().__init__()
        self.measure = measure
        self.reference_value = reference_value
        self.delta = delta
        self.higher_is_slow = higher_is_slow
        self.name = name

    def evaluate(self, candidate: frozenset) -> JudgeRecord:
        v = self.measure(candidate)
        d = v - self.reference_value
        slow = d > self.delta if self.higher_is_slow else -d > self.delta
        return JudgeRecord(tuple(sorted(candidate)), Verdict.SLOW if slow else Verdict.FAST, v,
                           self.reference_value, self.delta, note=self.name)


class TimingJudge(Judge):
    """SLOW iff ``confirmations`` independent runs of the candidate each exceed the reference by
    more than tau (AND rule), with an interleaved baseline run between them.

    Evidence (scripts/noise_study_replication.py, null = identical configurations): one run
    exceeding tau is wrong 8.9% of the time; two independent runs both exceeding tau, 1.5%; three,
    0.9%. Averaging the runs instead does *not* help (one 20% outlier process dominates the mean).

    The two candidate runs must not be back-to-back: a disturbed stretch of machine time (shared
    host, background load) would slow both. So after a first run exceeds tau, a fresh *baseline*
    run is taken (``baseline_ms(rep)``) before the confirmation run. If that baseline itself
    deviates from the reference centre by more than tau, the machine is disturbed: the attempt is
    discarded and retried later (up to ``max_retries``); if it never settles the verdict is FAST
    with a note, never a blame. This is the classic interleaved (ABAB) design.

    ``measure_ms(changes, rep)`` returns the steady-state median of one process; ``rep``
    distinguishes independent repeats of the same configuration.
    """

    def __init__(self, measure_ms: Callable[..., float], reference_ms: float, tau: float,
                 confirmations: int = 2, baseline_ms: Callable[[int], float] | None = None,
                 max_retries: int = 2, **_ignored) -> None:
        super().__init__()
        self.measure_ms = measure_ms
        self.reference_ms = reference_ms
        self.tau = tau
        self.confirmations = max(1, int(confirmations))
        self.baseline_ms = baseline_ms
        self.max_retries = max_retries
        self.remeasured = 0
        self.baseline_runs = 0
        self.disturbed_events = 0
        self._baseline_rep = 100          # rep indices for interleaved baseline runs

    def _measure(self, candidate: frozenset, rep: int) -> float:
        try:
            r = self.measure_ms(candidate, rep)
        except TypeError:  # measure functions that take no rep argument
            r = self.measure_ms(candidate)
        if isinstance(r, tuple):             # (median_ms, same_program_as_reference)
            self._last_same_program = bool(r[1])
            return r[0]
        self._last_same_program = False
        return r

    def _baseline_ok(self) -> tuple[bool, float | None]:
        if self.baseline_ms is None:
            return True, None
        self._baseline_rep += 1
        b = self.baseline_ms(self._baseline_rep)
        self.baseline_runs += 1
        ok = abs(b - self.reference_ms) <= self.tau
        if not ok:
            self.disturbed_events += 1
        return ok, b

    def evaluate(self, candidate: frozenset) -> JudgeRecord:
        notes = []
        rep = 0
        for attempt in range(self.max_retries + 1):
            values = []
            v = self._measure(candidate, rep)
            values.append(v)
            if getattr(self, "_last_same_program", False):
                # byte-identical generated program: the rewrites in question did nothing here, so a
                # timing difference cannot be theirs
                return JudgeRecord(tuple(sorted(candidate)), Verdict.FAST, v, self.reference_ms, self.tau,
                                   "identical generated program to the reference: not a rewrite effect")
            if v - self.reference_ms <= self.tau:
                return JudgeRecord(tuple(sorted(candidate)), Verdict.FAST, v, self.reference_ms, self.tau, "; ".join(notes))
            disturbed = False
            for i in range(1, self.confirmations):
                ok, b = self._baseline_ok()                 # interleaved baseline between candidate runs
                if not ok:
                    notes.append(f"interleaved baseline {b:.3f} deviates from reference {self.reference_ms:.3f} by > tau: machine disturbed, retrying")
                    disturbed = True
                    break
                rep = max(rep, 1) + 1                      # rep 2, 3, ... (rep 1 is reserved for noise runs)
                v2 = self._measure(candidate, rep)
                self.remeasured += 1
                values.append(v2)
                if v2 - self.reference_ms <= self.tau:
                    break
            if disturbed:
                rep = max(rep, 1) + 10 * (attempt + 1)     # retry with fresh rep indices
                continue
            slow = len(values) == self.confirmations and all(x - self.reference_ms > self.tau for x in values)
            if len(values) > 1:
                notes.append(f"{len(values)} time-separated runs: " + ", ".join(f"{x:.3f}" for x in values)
                             + (" (all exceed tau)" if slow else " (not all exceed tau)"))
            return JudgeRecord(tuple(sorted(candidate)), Verdict.SLOW if slow else Verdict.FAST,
                               min(values) if slow else values[-1], self.reference_ms, self.tau, "; ".join(notes))
        notes.append("unresolved: machine stayed disturbed; not blaming")
        return JudgeRecord(tuple(sorted(candidate)), Verdict.FAST, values[0], self.reference_ms, self.tau, "; ".join(notes))
