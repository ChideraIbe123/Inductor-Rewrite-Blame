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
    more than tau (AND rule); default two runs.

    Evidence (scripts/noise_study_replication.py, null = identical configurations): one run
    exceeding tau is wrong 8.9% of the time; two independent runs both exceeding tau, 1.5%; three,
    0.9%. Averaging the runs instead does *not* help (one 20% outlier process dominates the mean).
    A FAST verdict needs only the first run (a false FAST is far cheaper for ddmin than a false
    SLOW, which sends the search down the wrong branch), so the extra cost is one run per SLOW
    verdict, of which ddmin issues O(log n).

    ``measure_ms(changes, rep)`` returns the steady-state median of one process; ``rep``
    distinguishes independent repeats of the same configuration.
    """

    def __init__(self, measure_ms: Callable[..., float], reference_ms: float, tau: float,
                 confirmations: int = 2, **_ignored) -> None:
        super().__init__()
        self.measure_ms = measure_ms
        self.reference_ms = reference_ms
        self.tau = tau
        self.confirmations = max(1, int(confirmations))
        self.remeasured = 0

    def _measure(self, candidate: frozenset, rep: int) -> float:
        try:
            return self.measure_ms(candidate, rep)
        except TypeError:  # measure functions that take no rep argument
            return self.measure_ms(candidate)

    def evaluate(self, candidate: frozenset) -> JudgeRecord:
        values = []
        for i in range(self.confirmations):
            rep = 0 if i == 0 else i + 1          # rep 0, then 2, 3, ... (rep 1 is reserved for noise runs)
            v = self._measure(candidate, rep)
            values.append(v)
            if i > 0:
                self.remeasured += 1
            if v - self.reference_ms <= self.tau:
                break                              # one run within tau -> FAST, no more runs needed
        slow = len(values) == self.confirmations and all(v - self.reference_ms > self.tau for v in values)
        note = ""
        if len(values) > 1:
            note = (f"{len(values)} independent runs: " + ", ".join(f"{v:.3f}" for v in values)
                    + (" (all exceed tau)" if slow else " (not all exceed tau)"))
        return JudgeRecord(tuple(sorted(candidate)), Verdict.SLOW if slow else Verdict.FAST,
                           min(values) if slow else values[-1], self.reference_ms, self.tau, note)
