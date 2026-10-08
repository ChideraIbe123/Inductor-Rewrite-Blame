import pytest

from rewrite_blame.attribute import apply_changes, attribute_iteratively
from rewrite_blame.interactions import pair_scan, tau_scan
from rewrite_blame.judge import SyntheticJudge, Verdict

U = [f"r{i}" for i in range(10)]


class Model:
    """Shared synthetic cost model whose judges are relative to a moving fast state."""

    def __init__(self, singles=None, pairs=None, tau=1.0, noise_sd=0.0):
        self.singles, self.pairs, self.tau, self.noise_sd = singles or {}, pairs or {}, tau, noise_sd
        self.calls = 0

    def cost(self, state: frozenset) -> float:
        c = 100.0 + sum(v for k, v in self.singles.items() if k in state)
        c += sum(v for (a, b), v in self.pairs.items() if a in state and b in state)
        return c

    def make_judge(self, fast_state: frozenset):
        model = self

        class J(SyntheticJudge):
            def evaluate(self_, cand):
                model.calls += 1
                v = model.cost(apply_changes(fast_state, cand)) - model.cost(fast_state)
                from rewrite_blame.judge import JudgeRecord
                return JudgeRecord(tuple(sorted(cand)), Verdict.SLOW if v > model.tau else Verdict.FAST, v, 0.0, model.tau)
        return J(tau=self.tau)


def test_iterative_finds_two_independent_causes():
    m = Model(singles={"r2": 5.0, "r7": 4.0})
    res = attribute_iteratively([], U, m.make_judge)
    assert sorted(res.all_culprits) == ["+r2", "+r7"]
    assert len(res.rounds) >= 2 and res.residual_explained


def test_iterative_single_cause_stops_after_one_round():
    m = Model(singles={"r3": 5.0})
    res = attribute_iteratively([], U, m.make_judge)
    assert res.all_culprits == ["+r3"] and res.residual_explained
    assert [r["culprits"] for r in res.rounds][0] == ["+r3"]
    assert len(res.rounds) == 2  # second round confirms the residual is within noise


def test_iterative_mixed_single_and_pair():
    m = Model(singles={"r1": 5.0}, pairs={("r4", "r8"): 5.0})
    res = attribute_iteratively([], U, m.make_judge)
    assert set(res.all_culprits) == {"+r1", "+r4", "+r8"}
    kinds = [r["kind"] for r in res.rounds if r["culprits"]]
    assert "interaction" in kinds and "single" in kinds


def test_iterative_nothing_to_blame():
    m = Model(singles={"r1": 0.1})
    res = attribute_iteratively([], U, m.make_judge)
    assert res.all_culprits == [] and res.residual_explained


def test_iterative_respects_max_rounds():
    m = Model(singles={f"r{i}": 5.0 for i in range(10)})
    res = attribute_iteratively([], U, m.make_judge, max_rounds=3)
    assert len(res.all_culprits) == 3 and not res.residual_explained and res.notes


def test_pair_scan_flags_superadditive_and_masking():
    singles = {"a": 2.0, "b": 0.0, "c": 3.0, "d": 0.0}
    pairs = {frozenset({"a", "b"}): 5.0, frozenset({"c", "d"}): -3.5}

    def measure(state):
        v = 10.0 + sum(singles.get(s, 0.0) for s in state)
        v += sum(val for k, val in pairs.items() if k <= state)
        return v
    res = pair_scan(frozenset(), ["a", "b", "c", "d"], measure, tau=1.0)
    assert res.measurements == 1 + 4 + 6
    assert res.singles == {"a": 2.0, "b": 0.0, "c": 3.0, "d": 0.0}
    assert [(r["a"], r["b"]) for r in res.superadditive] == [("a", "b")]
    assert res.superadditive[0]["interaction_ms"] == pytest.approx(5.0)
    assert [(r["a"], r["b"]) for r in res.masking] == [("c", "d")]
    assert all(not r["superadditive"] for r in res.pairs if (r["a"], r["b"]) != ("a", "b"))


def test_pair_scan_max_pairs_and_custom_toggle():
    calls = []

    def measure(state):
        calls.append(state)
        return 1.0
    res = pair_scan({"x"}, ["x", "y", "z"], measure, tau=0.1, max_pairs=1)
    assert res.measurements == 1 + 3 + 1
    # toggling "x" on a base that has it turns it off
    assert frozenset() in calls


def test_tau_scan_reports_stability():
    def run(k):
        return {"tau": k * 0.1, "culprits": ["+r2"] if k < 10 else [], "kind": "single" if k < 10 else "none", "judge_calls": 3}
    res = tau_scan(run, [2, 3, 4, 20])
    assert res.stable and len(res.rows) == 4 and res.rows[-1]["culprits"] == []

    def run2(k):
        return {"tau": k, "culprits": ["+r2"] if k < 3 else ["+r2", "+r5"], "kind": "x", "judge_calls": 1}
    assert not tau_scan(run2, [2, 4]).stable
