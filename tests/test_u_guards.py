from rewrite_blame.guards import build_dataset, family_of, fit_stump, firing_features, leave_one_model_out, render_guards


def test_family_of_strips_variants():
    assert family_of("joint/joint_graph.patterns/_sfdp_pattern_11_inference") == "_sfdp_pattern"
    assert family_of("joint/joint_graph.patterns/_sfdp_pattern_12_half_inference") == "_sfdp_pattern"
    assert family_of("post_grad/post_grad.pass_patterns[1]/fn#12") == "fn"
    assert family_of("joint/joint_graph.patterns/pointless_convert") == "pointless_convert"


def test_firing_features_aggregate():
    fs = [{"switch": "s", "max_numel": 1000, "n_nodes": 3, "dtypes": ["float32"]},
          {"switch": "s", "max_numel": 10, "n_nodes": 5, "dtypes": ["float16"]}, {"switch": "t", "max_numel": 1}]
    f = firing_features(fs, "s")
    assert f["n_fires"] == 2 and f["log_max_numel"] == 3.0 and f["log_min_numel"] == 1.0 and f["n_nodes"] == 5 and f["low_precision"] == 1
    assert firing_features(fs, "nope") is None


def _sweep(model, env, rows, tau=0.02, ref=1.0):
    return {"model": model, "env": env, "reference_ms": ref, "noise": {"tau": tau},
            "rows": [{"switch": s, "direction": "off", "delta_pct": d} for s, d in rows]}


def test_build_dataset_labels_and_lomo_eval():
    # the attention rule: harmful on x86 (model gets faster without it), helpful on arm
    rule = "joint/joint_graph.patterns/_sfdp_pattern_11_inference"
    default_rows, sweeps = [], []
    for model, env, delta in [("a", "Apple arm64|torch|T8", +30.0), ("b", "Apple arm64|torch|T8", +4.0),
                              ("c", "Intel x86_64|torch|T4", -8.0), ("d", "Intel x86_64|torch|T4", -3.0)]:
        default_rows.append({"model": model, "env": env, "firings": [{"switch": rule, "max_numel": 1e6, "n_nodes": 10, "dtypes": ["float32"]}]})
        sweeps.append(_sweep(model, env, [(rule, delta), ("other/never_fired", 0.1)]))
    data = build_dataset(default_rows, sweeps)
    assert len(data) == 4 and {r["label"] for r in data} == {"harmful", "helpful"}
    assert all(r["family"] == "_sfdp_pattern" for r in data)
    ev = leave_one_model_out(data)["_sfdp_pattern"]
    assert ev["n"] == 4 and ev["harmful"] == 2
    assert ev["guard_accuracy"] == 1.0 and ev["default_accuracy"] == 0.5   # x86 flag separates them perfectly
    assert ev["stump_on_all"]["feature"] == "x86"
    text = render_guards(data, leave_one_model_out(data))
    assert "_sfdp_pattern" in text and "harmful" in text


def test_fit_stump_prefers_fewer_positives_on_ties_and_handles_no_harmful():
    rows = [{"label": "helpful", "x86": 0, "log_max_numel": 1.0}, {"label": "neutral", "x86": 1, "log_max_numel": 2.0}]
    assert fit_stump(rows) is None
    rows.append({"label": "harmful", "x86": 1, "log_max_numel": 3.0})
    st = fit_stump(rows)
    assert st.train_accuracy == 1.0
