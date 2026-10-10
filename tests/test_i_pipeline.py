"""Runner-level tests: subprocess worker isolation, store caching, metric-judge attribution on a
planted interaction, verification of switch effects. Uses a temporary store + the discovered registry."""
import json
from pathlib import Path

import pytest

from rewrite_blame.attribute import apply_changes
from rewrite_blame.discover import discover
from rewrite_blame.pipeline import Runner, RunnerConfig, parse_state_spec, switch_effect

SFDP11 = "joint/joint_graph.patterns/_sfdp_pattern_11_inference"
SFDP12 = "joint/joint_graph.patterns/_sfdp_pattern_12_inference"


@pytest.fixture(scope="session")
def reg_path(tmp_path_factory):
    reg = discover()
    p = tmp_path_factory.mktemp("reg") / "switches.json"
    reg.save(p)
    return p


@pytest.fixture(scope="session")
def runner(reg_path, tmp_path_factory):
    cfg = RunnerConfig(registry_path=reg_path, store_path=tmp_path_factory.mktemp("store") / "m.sqlite",
                       threads=4, warmup=2, rounds=2, iters=3, verbose=False)
    return Runner(cfg)


def test_parse_state_spec(runner):
    reg = runner.reg
    d = reg.default_state()
    assert parse_state_spec(reg, "default") == d
    assert parse_state_spec(reg, "none") == frozenset() and parse_state_spec(reg, "all") == reg.all_on()
    s = parse_state_spec(reg, "default-sched/epilogue_fusion+optimus/decompose_mm_pass")
    assert "sched/epilogue_fusion" not in s and "optimus/decompose_mm_pass" in s
    s2 = parse_state_spec(reg, "default-*_sfdp_pattern_*")
    assert not any("_sfdp_pattern_" in i for i in s2) and len(s2) < len(d)
    s3 = parse_state_spec(reg, "default~scheduler")
    assert "sched/epilogue_fusion" not in s3 and "sched/aggressive_fusion" in s3
    with pytest.raises(KeyError):
        parse_state_spec(reg, "default-nope/x")
    with pytest.raises(ValueError):
        parse_state_spec(reg, "everything")


def test_worker_subprocess_and_store_caching(runner):
    d = runner.reg.default_state()
    m = runner.measure("norm_mlp", d, time_it=False)
    assert m.error is None, m.error
    assert m.env == runner.env and m.code["kernel_count"] >= 1 and m.correct
    n_new = runner.new_measurements
    m2 = runner.measure("norm_mlp", d, time_it=False)
    assert runner.new_measurements == n_new and runner.cache_hits >= 1
    assert m2.state_hash == m.state_hash and m2.code == m.code
    # a timed measurement is a different protocol -> new worker run with timing data
    m3 = runner.measure("norm_mlp", d, time_it=True)
    assert m3.timing["n"] == runner.cfg.rounds * runner.cfg.iters and m3.median_ms > 0


def test_worker_reports_failures_instead_of_raising(runner):
    m = runner.measure("no_such_model", runner.reg.default_state(), time_it=False)
    assert m.error and "unknown model" in m.error


def test_metric_attribution_finds_planted_sfdp_pair(runner):
    """fast = default (SDPA fused, 1 kernel); slow = default with all inference sfdp rules off.
    Kernel count is the judge. Only patterns 11 and 12 matter and both must be off: ddmin must
    return exactly {-11, -12} and flag it as an interaction."""
    reg = runner.reg
    fast = reg.default_state()
    sfdp_inf = {s.id for s in reg.of_kind("pattern") if "_sfdp_pattern_" in s.id and s.id.endswith("_inference")}
    slow = fast - sfdp_inf
    res = runner.attribute("attention_block", fast, slow, judge="metric", metric="kernel_count")
    r = res["result"]
    assert sorted(r["culprits"]) == [f"-{SFDP11}", f"-{SFDP12}"], r
    assert r["interaction"] is True and res["kind"] == "interaction"
    assert r["judge_calls"] < len(sfdp_inf)  # far fewer compiles than leave-one-out
    assert res["culprit_state"]["code"]["kernel_count"] > res["fast"]["code"]["kernel_count"]


def test_metric_attribution_single_culprit_among_noops(runner):
    """slow = default minus three rules that never fire on norm_mlp minus inplace_buffers; judge =
    intermediate allocation bytes. Only the inplace_buffers change can move the metric."""
    reg = runner.reg
    fast = reg.default_state()
    noise_ids = {"joint/joint_graph.patterns/bmm_to_mm", "post_grad/post_grad.pass_patterns[1]/cat_slice_cat",
                 "joint/joint_graph.patterns/fix_iota_device"}
    slow = (fast - noise_ids) - {"sched/inplace_buffers"}
    res = runner.attribute("norm_mlp", fast, slow, judge="metric", metric="alloc_bytes")
    assert res["result"]["culprits"] == ["-sched/inplace_buffers"], res["result"]
    assert res["kind"] == "single"
    assert res["result"]["judge_calls"] <= 6


def test_switch_effect_classification(runner):
    reg = runner.reg
    d = reg.default_state()
    m0 = runner.measure("norm_mlp", d, time_it=False)
    pc = "joint/joint_graph.patterns/pointless_convert"
    eff = switch_effect(reg, pc, "off", m0, runner.measure("norm_mlp", d - {pc}, time_it=False))
    assert eff["status"] == "changes_graph" and eff["fired_changed"][pc][1] == 0
    never = "joint/joint_graph.patterns/fix_iota_device"
    eff2 = switch_effect(reg, never, "off", m0, runner.measure("norm_mlp", d - {never}, time_it=False))
    assert eff2["status"] == "no_effect"


def test_candidate_universe_is_fired_patterns_plus_flags(runner):
    uni = runner.candidate_universe("norm_mlp")
    assert "joint/joint_graph.patterns/pointless_convert" in uni["pattern"]
    assert "joint/joint_graph.patterns/fix_iota_device" not in uni["pattern"]
    assert "sched/epilogue_fusion" in uni["config"] and "optimus/decompose_mm_pass" in uni["option"]


def test_noise_model_from_repeated_runs(runner):
    nm, ms = runner.noise("norm_mlp", runner.reg.default_state(), runs=3)
    assert len(nm.medians) == 3 and nm.tau > 0 and nm.center > 0
    assert all(m.timing["n"] == runner.cfg.rounds * runner.cfg.iters for m in ms)
    # a second call is fully served from the store
    n_new = runner.new_measurements
    runner.noise("norm_mlp", runner.reg.default_state(), runs=3)
    assert runner.new_measurements == n_new


def test_attribution_report_renders(runner):
    from rewrite_blame.report import render_attribution
    reg = runner.reg
    fast = reg.default_state()
    res = runner.attribute("norm_mlp", fast, fast - {"sched/inplace_buffers"}, judge="metric", metric="alloc_bytes")
    text = render_attribution(res)
    assert "Verdict: single" in text and "-sched/inplace_buffers" in text and "allocation bytes" in text


def test_program_hash_is_stable_across_processes_and_sensitive_to_rewrites(runner):
    d = runner.reg.default_state()
    m1 = runner.measure("norm_mlp", d, time_it=False)
    m2 = runner.measure("norm_mlp", d, time_it=False, force=True)
    assert m1.program_hash and m1.program_hash == m2.program_hash
    m3 = runner.measure("norm_mlp", d - {"joint/joint_graph.patterns/pointless_convert"}, time_it=False)
    assert m3.program_hash != m1.program_hash
    m4 = runner.measure("norm_mlp", d - {"joint/joint_graph.patterns/fix_iota_device"}, time_it=False)
    assert m4.program_hash == m1.program_hash     # a rule that never fires leaves the program identical
