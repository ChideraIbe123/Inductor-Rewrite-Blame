from rewrite_blame.summary import render_summary, summarize


def test_summary_from_minimal_results():
    res = {
        "sweep": [{"model": "m", "env": "Apple M4|t|T8", "reference_ms": 1.0, "noise": {"tau": 0.02},
                   "rows": [{"switch": "x/a", "direction": "off", "delta_pct": 5.0, "slower": True, "faster": False},
                            {"switch": "x/b", "direction": "on", "delta_pct": -3.0, "slower": False, "faster": True},
                            {"switch": "x/c", "error": "boom"}]}],
        "attribution": [{"model": "m", "env": "Intel|t|T4", "kind": "single", "judge": "timing",
                         "result": {"candidates": ["+x/a", "+x/b"], "judge_calls": 3, "culprits": ["+x/a"]},
                         "fast": {"timing": {"median": 1.0}}, "slow": {"timing": {"median": 1.1}}}],
        "attribution_all": [{"model": "m", "env": "Intel|t|T4", "iterative": {"rounds": [{"candidates": 4}], "judge_calls": 6, "residual_explained": True},
                             "groups": [{"culprits": ["-x/a"], "delta_pct": 4.0}], "total_ms": 0.05, "fast_ms": 1.0, "explained_ms": 0.04}],
        "interactions": [{"model": "m", "env": "Intel|t|T4", "ids": ["a", "b"], "pairs": [{}], "measurements": 5, "base_ms": 1.0,
                          "superadditive": [{"a": "x/a", "b": "x/b", "interaction_ms": 0.3, "delta_ms": 0.4}], "masking": [], "drift_events": 1}],
        "tau_scan": [{"model": "m", "env": "Apple|t|T8", "stable": True, "rows": [{"k": 2, "kind": "single", "culprits": ["+a"]}]}],
        "repeat": [{"model": "m", "agreement": {"same_machine": {"agree": 3, "disagree": 1}, "cross_machine": {"agree": 1, "disagree": 0}}, "baselines": [1, 2]}],
    }
    sm = summarize(res)
    assert sm["sweeps"][0]["slower"] == [("x/a", "off", 5.0)] and sm["sweeps"][0]["n_toggled"] == 2
    assert any(a["delta_pct"] is not None and abs(a["delta_pct"] - 10.0) < 1e-6 for a in sm["attributions"])
    text = render_summary(sm)
    assert "Leave-one-out sweeps" in text and "| m | Mac |" in text and "iterative" in text and "drift" in text
