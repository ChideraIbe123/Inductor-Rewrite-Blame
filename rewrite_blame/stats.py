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


def summarize(samples_ms: Sequence[float]) -> dict:
    return {
        "n": len(samples_ms),
        "median": median(samples_ms),
        "mad": mad(samples_ms),
        "min": min(samples_ms),
        "p10": quantile(samples_ms, 0.10),
        "p90": quantile(samples_ms, 0.90),
        "mean": sum(samples_ms) / len(samples_ms),
    }


@dataclass
class NoiseModel:
    """Threshold derived from repeated measurements of one unchanged configuration.

    ``medians`` are the per-run medians (each run = a separate process, warmup + trials).
    The threshold tau is ``max(k * MAD(medians), floor_frac * median(medians))``: a robust
    spread estimate with a floor so that a very quiet machine does not produce a zero threshold.
    ``pairwise_abs_diff`` quantiles tell how big a difference two *identical* configurations can
    show, which is the empirical false-positive scale the threshold must exceed.
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
