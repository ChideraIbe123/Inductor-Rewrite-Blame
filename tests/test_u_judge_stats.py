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


def test_timing_judge_remeasures_near_threshold_and_confirms_slow():
    calls = []
    seq = {frozenset({"far"}): [20.0, 20.5], frozenset({"near"}): [11.4, 10.2], frozenset({"ok"}): [10.0],
           frozenset({"fluke"}): [12.5, 9.0]}
    def measure(c, rep=0):
        calls.append((c, rep))
        return seq[c].pop(0)
    j = TimingJudge(measure, reference_ms=10.0, tau=1.0, band=0.5)
    assert j({"far"}) == Verdict.SLOW            # confirmed by a second measurement (rep 2)
    assert j({"ok"}) == Verdict.FAST             # no re-measure
    assert j({"near"}) == Verdict.FAST           # 1.4 within the band -> mean 10.8 -> FAST
    assert j({"fluke"}) == Verdict.FAST          # 12.5 then 9.0 -> mean 10.75 -> FAST (a fluke caught)
    assert j.remeasured == 3
    assert (frozenset({"far"}), 2) in calls and (frozenset({"ok"}), 0) in calls
    assert sum(1 for c, _ in calls if c == frozenset({"ok"})) == 1
    assert "confirmed" in j.trace[0].note and "near threshold" in j.trace[2].note


def test_timing_judge_without_slow_confirmation():
    j = TimingJudge(lambda c, rep=0: 20.0, reference_ms=10.0, tau=1.0, confirm_slow=False)
    assert j({"x"}) == Verdict.SLOW and j.remeasured == 0


def test_timing_judge_accepts_measure_without_rep_argument():
    j = TimingJudge(lambda c: 20.0, reference_ms=10.0, tau=1.0)
    assert j({"x"}) == Verdict.SLOW and j.remeasured == 1
