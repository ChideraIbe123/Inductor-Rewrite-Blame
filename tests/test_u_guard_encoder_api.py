from rewrite_blame.guard_api import build_request, evaluate_api, parse_probability
from rewrite_blame.guard_encoder import serialize


def _row(M, label, ratio, x86=0):
    return {"family": "mlp", "env": "e", "x86": x86, "threads": 8, "cfg_M": M, "cfg_I": 1024, "cfg_Hd": 4096,
            "cfg_weight_numel": 3 * 1024 * 4096, "cfg_flops_per_byte": M / 2, "label": label, "ratio": ratio}


def test_serialize_is_readable_and_complete():
    t = serialize(_row(1, "harmful", 1.3), "mlp")
    assert "decomposing this skinny matmul" in t and "M: 1" in t and "Apple M4 Pro" in t and "threads: 8" in t
    t2 = serialize({"cfg_C": 64, "cfg_R": 56, "cfg_K": 3, "cfg_L": 2, "cfg_depthwise": 1, "cfg_act_numel": 1, "x86": 1, "threads": 4}, "conv")
    assert "depthwise convolutions: yes" in t2 and "x86 Xeon" in t2


def test_build_request_and_context():
    body = build_request(_row(1, "harmful", 1.3), "mlp")
    assert body["model"] == "gpt-6-luna" and body["questions"][0]["type"] == "predicate" and "M: 1" in body["input"]
    assert body["questions"][0]["name"] == "harmful" and body["questions"][0]["instructions"].startswith("Will ")
    body2 = build_request(_row(1, "harmful", 1.3), "mlp", context_rows=[_row(32, "neutral", 0.99)])
    assert "Measured examples" in body2["input"] and "not slower" in body2["input"]


def test_parse_probability_variants():
    assert parse_probability({"decisions": [{"id": "harmful", "probability": 0.8}]}) == 0.8
    assert parse_probability({"answers": [{"value": 0.2}]}) == 0.2
    assert parse_probability({"probability": 0.5}) == 0.5
    import pytest
    with pytest.raises(ValueError):
        parse_probability({"nothing": 1})


def test_evaluate_api_with_fake_oracle():
    rows = [_row(1, "harmful", 1.3), _row(2, "harmful", 1.2), _row(16, "neutral", 1.0), _row(64, "helpful", 0.9)]
    def oracle(body):
        return 0.9 if "M: 1;" in body["input"] or "M: 2;" in body["input"] else 0.1
    res = evaluate_api(rows, "mlp", ask_fn=oracle, log=lambda *a: None)
    assert res["accuracy"] == 1.0 and res["recall"] == 1.0 and res["false_alarm_rate"] == 0.0
    res2 = evaluate_api(rows, "mlp", with_context=True, ask_fn=lambda b: 0.0, log=lambda *a: None)
    assert res2["recall"] == 0.0 and res2["cost_weighted"] > 0
