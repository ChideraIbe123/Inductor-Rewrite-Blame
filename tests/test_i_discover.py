"""Discovery of Inductor's pattern entries and installation of per-rule gates (compiles tiny graphs)."""
import json
import pytest
import torch

from rewrite_blame import apply
from rewrite_blame.apply import GatedExtraCheck, SwitchSetPass, config_patch_for_state
from rewrite_blame.discover import discover, summarize
from rewrite_blame.switches import Registry


@pytest.fixture(scope="session")
def reg():
    return discover()


def test_registry_has_substantial_pattern_universe(reg):
    s = summarize(reg)
    assert s["by_kind"]["pattern"] >= 100
    assert s["by_kind"]["config"] >= 20 and s["by_kind"]["option"] >= 10


@pytest.mark.parametrize("sid", [
    "joint/joint_graph.patterns/pointless_convert",
    "joint/joint_graph.early_patterns/pointless_view",
    "post_grad/post_grad.pass_patterns[1]/cat_slice_cat",
    "post_grad/post_grad.pass_patterns[1]/reciprocal_sqrt_to_rsqrt",
    "post_grad/post_grad.pass_patterns[1]/pointless_cumsum_replacement",
    "joint/joint_graph.patterns/_sfdp_pattern_1_inference",
    "joint/joint_graph.patterns/bmm_to_mm",
])
def test_known_rules_are_discovered(reg, sid):
    assert sid in reg
    assert reg[sid].kind == "pattern" and reg[sid].targets


def test_every_config_switch_exists_in_this_torch(reg):
    from rewrite_blame.apply import config_switch_exists
    assert all(config_switch_exists(s) for s in reg if s.kind in ("config", "option"))


def test_gates_installed_on_every_discovered_entry(reg):
    from torch._inductor.fx_passes import joint_graph, post_grad
    for p in [joint_graph.patterns, joint_graph.early_patterns, *post_grad.pass_patterns]:
        for entries in p.patterns.values():
            for e in entries:
                assert isinstance(e.extra_check, GatedExtraCheck)
                assert e.extra_check.switch_id in reg


def test_discover_is_idempotent(reg):
    ids = reg.ids()
    reg2 = discover(trigger=False)
    assert reg2.ids() == ids


def test_registry_roundtrips_through_json_file(reg, tmp_path):
    p = tmp_path / "s.json"
    reg.save(p)
    reg2 = Registry.load(p)
    assert reg2.ids() == reg.ids()
    assert all(reg2[i] == reg[i] for i in reg.ids())


def test_registry_is_deterministic_across_processes(reg, tmp_path):
    import subprocess, sys
    code = ("from rewrite_blame.discover import discover; import json; "
            "print(json.dumps(discover().ids()))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == reg.ids()


def test_config_patch_reflects_state(reg):
    st = reg.default_state()
    patch = config_patch_for_state(reg, st)
    assert patch["epilogue_fusion"] is True and patch["freezing"] is False
    assert patch["max_fusion_size"] == 64 and patch["pre_grad_fusion_options"] == {}
    st2 = (st - {"sched/epilogue_fusion"}) | {"optimus/split_cat_pass", "optimus/decompose_mm_pass"}
    patch2 = config_patch_for_state(reg, st2)
    assert patch2["epilogue_fusion"] is False
    assert patch2["pre_grad_fusion_options"] == {"split_cat_pass": {}}
    assert patch2["post_grad_fusion_options"] == {"decompose_mm_pass": {}}


def test_active_sets_and_restores_config_and_gates(reg):
    import torch._inductor.config as cfg
    before = (cfg.epilogue_fusion, cfg.post_grad_custom_post_pass, apply._DISABLED)
    st = reg.default_state() - {"sched/epilogue_fusion", "joint/joint_graph.patterns/pointless_convert"}
    with apply.active(reg, st):
        assert cfg.epilogue_fusion is False
        assert isinstance(cfg.post_grad_custom_post_pass, SwitchSetPass)
        assert "joint/joint_graph.patterns/pointless_convert" in apply._DISABLED
        assert "joint/joint_graph.early_patterns/pointless_view" not in apply._DISABLED
    assert (cfg.epilogue_fusion, cfg.post_grad_custom_post_pass, apply._DISABLED) == before


def test_active_rejects_unknown_ids(reg):
    with pytest.raises(KeyError):
        with apply.active(reg, reg.default_state() | {"bogus/x"}):
            pass


def test_switch_set_pass_uuid_depends_only_on_state():
    a = SwitchSetPass({"x", "y"}).uuid()
    assert a == SwitchSetPass(["y", "x"]).uuid()
    assert a != SwitchSetPass({"x"}).uuid()
    assert torch.__version__ in "" or len(a) == 64
