import math
import pytest

from rewrite_blame.judge import MetricJudge, TimingJudge, Verdict
from rewrite_blame.stats import mad, median, noise_model, quantile, summarize


def test_median_and_mad():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5
    assert mad([1, 1, 1, 1]) == 0
    assert math.isclose(mad([1, 2, 3, 4, 100], scale=1.0), 1.0)
    with pytest.raises(ValueError):
        median([])


def test_quantile_edges():
    xs = [10, 20, 30, 40]
    assert quantile(xs, 0) == 10 and quantile(xs, 1) == 40
    assert quantile(xs, 0.5) == 25
    assert quantile([7], 0.9) == 7
    with pytest.raises(ValueError):
        quantile(xs, 1.5)


def test_summarize_keys():
    s = summarize([1.0, 2.0, 3.0])
    assert s["n"] == 3 and s["median"] == 2.0 and s["min"] == 1.0 and set(s) >= {"p10", "p90", "mad"}


def test_noise_model_threshold_uses_mad_with_floor():
    nm = noise_model([10.0, 10.1, 9.9, 10.05, 9.95], k=3.0, floor_frac=0.01)
    assert nm.center == 10.0
    assert nm.tau >= 0.01 * 10.0
    assert nm.tau == max(3.0 * nm.spread, 0.1)
    assert nm.pairwise_max == pytest.approx(0.2)
    # a 1 ms slowdown is far outside the noise; a 0.05 ms one is inside
    assert nm.is_slower(11.0) and not nm.is_slower(10.05)


def test_noise_model_quiet_machine_floor_prevents_zero_threshold():
    nm = noise_model([5.0, 5.0, 5.0], k=3.0, floor_frac=0.02)
    assert nm.tau == pytest.approx(0.1)


def test_noise_model_roundtrip_and_min_runs():
    nm = noise_model([1.0, 1.2, 0.9])
    assert noise_model.__name__ and nm.to_dict()["tau"] == nm.tau
    from rewrite_blame.stats import NoiseModel
    assert NoiseModel.from_dict(nm.to_dict()) == nm
    with pytest.raises(ValueError):
        noise_model([1.0])


def test_metric_judge_higher_is_slow_and_delta():
    metric = {frozenset(): 10, frozenset({"a"}): 12, frozenset({"b"}): 10}
    j = MetricJudge(lambda c: metric[c], reference_value=10, delta=1)
    assert j({"a"}) == Verdict.SLOW and j({"b"}) == Verdict.FAST and j(set()) == Verdict.FAST
    j2 = MetricJudge(lambda c: 5 if c else 10, reference_value=10, delta=0, higher_is_slow=False)
    assert j2({"x"}) == Verdict.SLOW


def test_timing_judge_remeasures_only_near_threshold():
    calls = []
    seq = {frozenset({"far"}): [20.0], frozenset({"near"}): [11.4, 10.2], frozenset({"ok"}): [10.0]}
    def measure(c):
        calls.append(c)
        return seq[c].pop(0)
    j = TimingJudge(measure, reference_ms=10.0, tau=1.0, band=0.5)
    assert j({"far"}) == Verdict.SLOW
    assert j({"ok"}) == Verdict.FAST
    # 11.4-10 = 1.4 is within (0.5, 1.5) -> re-measured -> mean 10.8 -> FAST
    assert j({"near"}) == Verdict.FAST
    assert j.remeasured == 1
    assert calls.count(frozenset({"near"})) == 2 and calls.count(frozenset({"far"})) == 1
    assert "re-measured" in j.trace[-1].note
