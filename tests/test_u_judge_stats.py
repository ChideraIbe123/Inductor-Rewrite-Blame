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
    assert s["n"] == 3 and s["median"] == 2.0 and s["min"] == 1.0 and set(s) >= {"p10", "p90", "mad", "steady_start"}


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


def test_timing_judge_and_rule_two_runs():
    calls = []
    seq = {frozenset({"far"}): [20.0, 20.5], frozenset({"ok"}): [10.0], frozenset({"fluke"}): [15.0, 10.1],
           frozenset({"near"}): [11.05, 11.2]}
    def measure(c, rep=0):
        calls.append((c, rep))
        return seq[c].pop(0)
    j = TimingJudge(measure, reference_ms=10.0, tau=1.0)
    assert j({"far"}) == Verdict.SLOW            # both runs exceed tau
    assert j({"ok"}) == Verdict.FAST             # first run within tau: no second run
    assert j({"fluke"}) == Verdict.FAST          # 15.0 then 10.1: runs disagree -> FAST
    assert j({"near"}) == Verdict.SLOW           # 11.05 and 11.2 both exceed 11.0
    assert j.remeasured == 3
    assert (frozenset({"far"}), 2) in calls and sum(1 for c, _ in calls if c == frozenset({"ok"})) == 1
    assert "all exceed" in j.trace[0].note and "not all exceed" in j.trace[2].note


def test_timing_judge_three_confirmations_and_single():
    vals = iter([20.0, 20.0, 10.0])
    j = TimingJudge(lambda c, rep=0: next(vals), reference_ms=10.0, tau=1.0, confirmations=3)
    assert j({"x"}) == Verdict.FAST and j.remeasured == 2       # third run disagreed
    j1 = TimingJudge(lambda c, rep=0: 20.0, reference_ms=10.0, tau=1.0, confirmations=1)
    assert j1({"x"}) == Verdict.SLOW and j1.remeasured == 0


def test_timing_judge_accepts_measure_without_rep_argument():
    j = TimingJudge(lambda c: 20.0, reference_ms=10.0, tau=1.0)
    assert j({"x"}) == Verdict.SLOW and j.remeasured == 1


def test_steady_state_start_trims_only_a_leaked_warmup():
    from rewrite_blame.stats import steady_state_start
    flat = [10.0, 10.1, 9.9, 10.0, 10.1, 10.0, 9.9, 10.0, 10.1, 10.0, 10.0, 10.1]
    assert steady_state_start(flat) == 0
    leaked = [14.0, 13.0, 12.0, 11.5] + flat
    assert 2 <= steady_state_start(leaked) <= 4   # window median tolerates a couple of high calls
    assert steady_state_start([1.0, 2.0, 3.0]) == 0      # too short: untouched
    s = summarize(leaked)
    assert s["steady_start"] >= 2 and s["median"] == pytest.approx(10.0, abs=0.15) and s["median_raw"] >= s["median"]
    assert "iqr_rel" in s and s["n"] == len(leaked)


def test_timing_judge_interleaved_baseline_detects_disturbance_and_retries():
    seq = {"cand": [20.0, 20.0, 20.0], "base": [25.0, 10.0]}   # first baseline disturbed, second fine
    calls = []
    def measure(c, rep=0):
        calls.append(("cand", rep)); return seq["cand"].pop(0)
    def baseline(rep):
        calls.append(("base", rep)); return seq["base"].pop(0)
    j = TimingJudge(measure, reference_ms=10.0, tau=1.0, baseline_ms=baseline)
    assert j({"x"}) == Verdict.SLOW
    assert j.disturbed_events == 1 and j.baseline_runs == 2
    assert "disturbed" in j.trace[0].note and "time-separated" in j.trace[0].note
    reps = [r for k, r in calls if k == "cand"]
    assert len(reps) == 3 and len(set(reps)) == 3     # three distinct candidate runs


def test_timing_judge_never_blames_when_machine_stays_disturbed():
    j = TimingJudge(lambda c, rep=0: 20.0, reference_ms=10.0, tau=1.0, baseline_ms=lambda rep: 30.0, max_retries=1)
    assert j({"x"}) == Verdict.FAST
    assert "unresolved" in j.trace[0].note and j.disturbed_events == 2
