"""Unit tests for ddmin-based attribution on synthetic cost models (no torch)."""
import itertools
import pytest

from rewrite_blame.attribute import apply_changes, attribute, changes_between, ddmin, _split
from rewrite_blame.judge import SyntheticJudge, Verdict

U = [f"r{i}" for i in range(12)]
T = [f"+{u}" for u in U]  # the tokens attribute() will see when fast=[] and slow=U


def test_split_balanced_nonempty():
    chunks = _split(list(range(7)), 3)
    assert [len(c) for c in chunks] == [3, 2, 2]
    assert sum(chunks, []) == list(range(7))
    assert _split([1], 1) == [[1]]


def test_single_culprit_found():
    j = SyntheticJudge(singles={"+r5": 10.0}, tau=1.0)
    res = attribute([], U, j)
    assert res.culprits == ["+r5"]
    assert res.kind() == "single"
    assert res.judge_calls < 2 * len(U)  # far fewer than exhaustive


def test_pair_only_culprit_found_and_flagged_as_interaction():
    j = SyntheticJudge(pairs={("+r2", "+r9"): 10.0}, tau=1.0)
    res = attribute([], U, j)
    assert sorted(res.culprits) == ["+r2", "+r9"]
    assert res.interaction is True
    assert res.kind() == "interaction"
    assert res.singles == {"+r2": "fast", "+r9": "fast"}


def test_two_independent_culprits_gives_one_minimal_set():
    # either one alone reproduces; ddmin returns one of them (1-minimal), never both
    j = SyntheticJudge(singles={"+r1": 5.0, "+r8": 5.0}, tau=1.0)
    res = attribute([], U, j)
    assert len(res.culprits) == 1 and res.culprits[0] in ("+r1", "+r8")


def test_culprit_plus_independent_culprit_each_reproduce():
    j = SyntheticJudge(singles={"+r1": 5.0}, pairs={("+r3", "+r4"): 5.0}, tau=1.0)
    res = attribute([], U, j)
    # 1-minimal: either {r1} or {r3,r4}
    assert res.culprits == ["+r1"] or sorted(res.culprits) == ["+r3", "+r4"]


def test_masking_non_monotone_case_still_returns_valid_minimal_set():
    # r0 hurts, but r6 cancels it (speeds up when both on); the judge's full set is still slow
    j = SyntheticJudge(singles={"+r0": 10.0, "+r6": -3.0}, tau=1.0)
    res = attribute([], U, j)
    assert res.culprits == ["+r0"]
    assert j(frozenset(res.culprits)) == Verdict.SLOW
    for c in res.culprits:  # 1-minimality
        assert j(frozenset(res.culprits) - {c}) == Verdict.FAST


def test_fast_reference_itself_slow_is_reported_not_crashed():
    j = SyntheticJudge(singles={}, tau=-1.0)  # everything (even empty) is "slow"
    res = attribute([], U, j)
    assert res.culprits == [] and any("fast reference" in n for n in res.notes)


def test_slow_state_not_reproduced_is_reported():
    j = SyntheticJudge(singles={"+r5": 0.1}, tau=1.0)
    res = attribute([], U, j)
    assert res.culprits == [] and any("not reproduced" in n for n in res.notes)


def test_identical_states_rejected_and_tokens_are_signed():
    with pytest.raises(ValueError):
        attribute(["x"], ["x"], SyntheticJudge())
    assert changes_between(["a", "b"], ["b", "c"]) == ["+c", "-a"]
    assert apply_changes(["a", "b"], ["+c", "-a"]) == frozenset({"b", "c"})
    with pytest.raises(ValueError):
        apply_changes([], ["c"])


def test_turning_a_rewrite_off_can_be_the_culprit():
    # fast has r3 ON; slow has it OFF -> the token "-r3" must be blamed
    j = SyntheticJudge(singles={"-r3": 10.0}, tau=1.0)
    res = attribute(U, [u for u in U if u != "r3"] + ["r99"], j)
    assert res.culprits == ["-r3"]


def test_ddmin_precondition_violation_raises():
    j = SyntheticJudge(tau=1.0)
    with pytest.raises(ValueError):
        ddmin(T, j)


def test_ddmin_three_way_interaction():
    j = SyntheticJudge(groups={("+r1", "+r4", "+r7"): 10.0}, tau=1.0)
    res = attribute([], U, j)
    assert sorted(res.culprits) == ["+r1", "+r4", "+r7"]
    assert res.interaction is True


@pytest.mark.parametrize("seed", range(25))
def test_random_models_result_is_slow_and_one_minimal(seed):
    import random
    rng = random.Random(seed)
    singles = {f"+r{i}": rng.choice([0.0, 0.0, 0.0, 4.0]) for i in range(10)}
    pairs = {}
    for _ in range(2):
        a, b = rng.sample(range(10), 2)
        pairs[(f"+r{a}", f"+r{b}")] = rng.choice([0.0, 4.0])
    j = SyntheticJudge(singles=singles, pairs=pairs, tau=1.0)
    res = attribute([], [f"r{i}" for i in range(10)], j)
    full_cost = j.cost(frozenset(f"+r{i}" for i in range(10))) - j.base
    if full_cost <= 1.0:
        assert res.culprits == []
        return
    C = frozenset(res.culprits)
    assert j(C) == Verdict.SLOW
    for c in C:
        assert j(C - {c}) == Verdict.FAST


def test_memoisation_counts_only_new_evaluations():
    j = SyntheticJudge(singles={"+r3": 10.0}, tau=1.0)
    j(["+r3"]); j(["+r3"]); j(frozenset(["+r3"]))
    assert j.calls == 1


def test_noisy_judge_mostly_right_with_margin():
    # effect 10 with noise sd 1 and tau 3: ddmin should land on r7 in the large majority of seeds
    hits = 0
    for seed in range(20):
        j = SyntheticJudge(singles={"+r7": 10.0}, tau=3.0, noise_sd=1.0, seed=seed)
        res = attribute([], U, j)
        hits += res.culprits == ["+r7"]
    assert hits >= 18


def test_judge_calls_scale_logarithmically_for_single_culprit():
    big = [f"r{i}" for i in range(256)]
    j = SyntheticJudge(singles={"+r200": 10.0}, tau=1.0)
    res = attribute([], big, j)
    assert res.culprits == ["+r200"]
    assert res.judge_calls <= 40  # ~2*log2(256)+constant, versus 256 for leave-one-out
