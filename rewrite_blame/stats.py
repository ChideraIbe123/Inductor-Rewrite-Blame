"""Small, dependency-free statistics used for timing and the noise threshold."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence


def median(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("median of empty sequence")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def mad(xs: Sequence[float], scale: float = 1.4826) -> float:
    """Median absolute deviation, scaled to be sigma-consistent for normal data."""
    m = median(xs)
    return scale * median([abs(x - m) for x in xs])


def quantile(xs: Sequence[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def steady_state_start(samples: Sequence[float], window: int = 5) -> int:
    """Index where the timed calls reach steady state.

    Conservative rule validated on ~480 stored runs (scripts/noise_study_*.py): the first index k
    such that the median of ``samples[k:k+window]`` is within one IQR of the median of the second
    half of the run. For most runs this is 0 (the warm-up calls were enough); it trims only runs
    whose warm-up leaked into the timed window. More aggressive rules (kneedle elbow, discarding
    half) threw away good samples and *increased* between-run spread.
    """
    n = len(samples)
    if n < 2 * window:
        return 0
    tail = list(samples[n // 2:])
    m = median(tail)
    q1, q3 = quantile(tail, 0.25), quantile(tail, 0.75)
    band = max(q3 - q1, 1e-12)
    for k in range(0, n - window + 1):
        if median(list(samples[k:k + window])) <= m + band:
            return k
    return n - window


def run_statistic() -> str:
    import os
    st = os.environ.get("REWRITE_BLAME_STAT", "median").lower()
    if st not in ("median", "min"):
        raise ValueError(f"REWRITE_BLAME_STAT must be median|min, got {st!r}")
    return st


def summarize(samples_ms: Sequence[float]) -> dict:
    xs = list(samples_ms)
    k = steady_state_start(xs)
    steady = xs[k:]
    q1, q3 = quantile(steady, 0.25), quantile(steady, 0.75)
    # thermal throttling shows up as a run whose calls get slower as it goes: compare the last
    # fifth of the steady window with the first fifth
    w = max(3, len(steady) // 5)
    trend_ratio = median(steady[-w:]) / median(steady[:w]) if len(steady) >= 2 * w and median(steady[:w]) else 1.0
    return {
        "n": len(xs),
        "steady_start": k,
        "trend_ratio": trend_ratio,
        "unsteady": trend_ratio > 1.10 or trend_ratio < 0.90,
        # the run's statistic. Default: median over the steady-state calls. REWRITE_BLAME_STAT=min uses the
        # fastest call instead (interference only ever adds time; fewer false alarms in the null calibration,
        # see docs/noise_method.md); "median_raw" always keeps the plain median so either can be recomputed.
        "median": min(steady) if run_statistic() == "min" else median(steady),
        "statistic": run_statistic(),
        "median_raw": median(xs),
        "mad": mad(steady),
        "iqr_rel": (q3 - q1) / median(steady) if median(steady) else 0.0,
        "min": min(xs),
        "p10": quantile(steady, 0.10),
        "p90": quantile(steady, 0.90),
        "mean": sum(steady) / len(steady),
    }


@dataclass
class NoiseModel:
    """Threshold derived from repeated measurements of one unchanged configuration.

    ``medians`` are the per-run medians (each run = a separate process, warmup + trials).
    The threshold tau is ``max(k * MAD(medians), floor_frac * median(medians))``: a robust
    spread estimate with a floor so that a very quiet machine does not produce a zero threshold.
    ``pairwise_abs_diff`` quantiles tell how big a difference two *identical* configurations can
    show, which is the empirical false-positive scale the threshold must exceed.

    Why this and not something else (null calibration on 180 leave-one-out trials over identical
    runs on two machines, scripts/noise_study_methods.py): MAD k=3 gives 8.9% false positives for
    a *single* candidate run; IQR-, SD-, Mann-Whitney-, bootstrap- and Wasserstein-based rules all
    land between 6% and 16%, because the between-process noise has a fat right tail (whole
    processes that run 2-22% slow). No threshold fixes that; replication does: requiring two
    independent candidate runs to each exceed tau brings false positives to 1.5% (0.9% with
    three) while keeping every effect above 6% detectable. That AND rule lives in TimingJudge.
    """

    medians: list[float]
    k: float
    floor_frac: float
    tau: float
    center: float
    spread: float
    pairwise_p50: float
    pairwise_p90: float
    pairwise_max: float
    n_unsteady: int = 0       # baseline runs excluded because their calls got slower as they went

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseModel":
        return cls(**d)

    def is_slower(self, candidate_median: float, reference_median: float | None = None) -> bool:
        ref = self.center if reference_median is None else reference_median
        return candidate_median - ref > self.tau


def noise_model(medians: Sequence[float], k: float = 3.0, floor_frac: float = 0.01) -> NoiseModel:
    if len(medians) < 2:
        raise ValueError("need at least two baseline runs to estimate noise")
    center = median(medians)
    spread = mad(medians)
    tau = max(k * spread, floor_frac * center)
    diffs = [abs(a - b) for i, a in enumerate(medians) for b in medians[i + 1:]]
    return NoiseModel(
        medians=list(medians), k=k, floor_frac=floor_frac, tau=tau, center=center, spread=spread,
        pairwise_p50=quantile(diffs, 0.5), pairwise_p90=quantile(diffs, 0.9), pairwise_max=max(diffs),
    )
