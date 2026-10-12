import math
import pytest

from rewrite_blame.guard_models import GuardBundle, evaluate, featurize, fit_all, load_rows, render_eval, tree_rules

pytest.importorskip("sklearn")


def _rows(n_per=6):
    rows = []
    # mlp family: harmful iff M <= 2 (decode shapes), independent of sizes
    for M in (1, 2, 4, 8, 16, 32):
        for I in (512, 1024, 2048):
            for Hd in (1024, 4096):
                harmful = M <= 2
                rows.append({"family": "mlp", "model": f"sweep_mlp_M{M}_I{I}_Hd{Hd}", "env": "Apple arm64|t|T8", "x86": 0, "threads": 8,
                             "cfg_M": M, "cfg_I": I, "cfg_Hd": Hd, "cfg_weight_numel": 3 * I * Hd, "cfg_flops_per_byte": M / 2,
                             "ratio": 1.3 if harmful else 0.98, "label": "harmful" if harmful else "neutral"})
    return rows


def test_featurize_log_scales_sizes():
    rows = _rows()
    X, y, names = featurize(rows, "mlp")
    j = names.index("cfg_I")
    assert X[0][j] == pytest.approx(math.log2(513)) and sum(y) == 12


def test_tree_learns_the_threshold_and_beats_always_fire():
    rows = _rows()
    res = evaluate(rows, "mlp", "tree")
    assert res.n == 36 and res.harmful == 12
    assert res.accuracy >= 0.9 and res.recall >= 0.9 and res.false_alarm_rate <= 0.1
    assert res.accuracy > res.default_accuracy and res.cost_weighted < res.default_cost_weighted
    clf, names = fit_all(rows, "mlp", "tree")
    assert ("cfg_M" in tree_rules(clf, names)) or ("cfg_flops_per_byte" in tree_rules(clf, names))   # equivalent splits


def test_bundle_predicts_and_roundtrips(tmp_path):
    rows = _rows()
    clf, names = fit_all(rows, "mlp", "tree")
    b = GuardBundle(); b.add("mlp", "tree", clf, names)
    assert b.predict_harmful("mlp", {"cfg_M": 1, "cfg_I": 1024, "cfg_Hd": 4096, "cfg_weight_numel": 3 * 1024 * 4096, "cfg_flops_per_byte": 0.5, "x86": 0, "threads": 8}) is True
    assert b.predict_harmful("mlp", {"cfg_M": 32, "cfg_I": 1024, "cfg_Hd": 4096, "cfg_weight_numel": 3 * 1024 * 4096, "cfg_flops_per_byte": 16, "x86": 0, "threads": 8}) is False
    assert b.predict_harmful("attn", {}) is None
    p = tmp_path / "g.pkl"; b.save(p)
    assert GuardBundle.load(p).predict_harmful("mlp", {"cfg_M": 1, "cfg_I": 512, "cfg_Hd": 1024, "cfg_weight_numel": 1, "cfg_flops_per_byte": 0.5, "x86": 0, "threads": 8}) is True


def test_load_rows_filters_and_render(tmp_path):
    import json
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows()[:5] + [{"family": "mlp", "label": "no_effect"}]) + "\nnot json\n")
    rows = load_rows([p], "mlp")
    assert len(rows) == 5
    text = render_eval([evaluate(_rows(), "mlp", "logreg")], counts={"mlp": {"harmful": 12, "neutral": 24}})
    assert "always-fire" in text and "| mlp |" in text


def test_inductor_policy_baseline():
    from rewrite_blame.guard_models import evaluate_policy, inductor_policy_blocks
    assert inductor_policy_blocks({"cfg_M": 1, "cfg_I": 1024, "cfg_Hd": 2048}, "mlp") == 0
    assert inductor_policy_blocks({"cfg_M": 1, "cfg_I": 1024, "cfg_Hd": 8192}, "mlp") == 1
    assert inductor_policy_blocks({"cfg_M": 8, "cfg_I": 512, "cfg_Hd": 512}, "mlp") == 1
    assert inductor_policy_blocks({"cfg_depthwise": 1}, "conv") == 0
    res = evaluate_policy(_rows(), "mlp")
    assert res.model == "inductor-heuristic" and 0 <= res.accuracy <= 1 and res.folds == 0
