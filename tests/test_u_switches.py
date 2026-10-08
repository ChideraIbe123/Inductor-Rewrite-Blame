import json
import pytest

from rewrite_blame.switches import (CONFIG_SWITCHES, OPTION_SWITCHES, Registry, Switch, builtin_registry,
                                    state_hash)


def test_builtin_registry_ids_unique_and_kinds_valid():
    reg = builtin_registry()
    assert len(reg) == len(CONFIG_SWITCHES) + len(OPTION_SWITCHES)
    assert all(s.kind in ("config", "option") for s in reg)
    assert "master/pattern_matcher" in reg


def test_default_state_matches_default_on_flags():
    reg = builtin_registry()
    st = reg.default_state()
    assert "sched/epilogue_fusion" in st
    assert "optimus/decompose_mm_pass" not in st
    assert "lowering/freezing" not in st


def test_state_hash_is_order_independent_and_sensitive():
    assert state_hash(["a", "b"]) == state_hash(["b", "a"])
    assert state_hash(["a"]) != state_hash(["a", "b"])
    assert len(state_hash([])) == 16


def test_toggle_and_validate():
    reg = builtin_registry()
    st = reg.default_state()
    st2 = reg.toggled(st, "sched/epilogue_fusion")
    assert "sched/epilogue_fusion" not in st2 and reg.toggled(st2, "sched/epilogue_fusion") == st
    with pytest.raises(KeyError):
        reg.validate_state({"nope/none"})


def test_json_roundtrip_preserves_everything():
    reg = builtin_registry()
    reg.add(Switch(id="joint/pointless_convert", kind="pattern", family="joint",
                   pass_label="joint_graph.patterns", entry_kind="GraphPatternEntry", targets=("prims.convert_element_type",)))
    reg2 = Registry.from_json(reg.to_json())
    assert reg2.ids() == reg.ids()
    assert reg2["joint/pointless_convert"] == reg["joint/pointless_convert"]
    assert reg2["sched/max_fusion_size"].on_value == 64 and reg2["sched/max_fusion_size"].off_value == 1


def test_unique_id_suffixing_and_duplicate_rejection():
    reg = Registry()
    s = Switch(id="x/a", kind="pattern", family="x")
    reg.add(s)
    assert reg.unique_id("x/a") == "x/a#2"
    reg.add(Switch(id="x/a#2", kind="pattern", family="x"))
    assert reg.unique_id("x/a") == "x/a#3"
    with pytest.raises(KeyError):
        reg.add(s)


def test_switch_validation():
    with pytest.raises(ValueError):
        Switch(id="bad", kind="weird", family="f")
    with pytest.raises(ValueError):
        Switch(id="bad", kind="config", family="f")
    with pytest.raises(ValueError):
        Switch(id="bad", kind="option", family="f", config_path="pre_grad_fusion_options")


def test_families_and_describe_diff():
    reg = builtin_registry()
    assert {"master", "scheduler", "optimus", "cpp"} <= set(reg.families())
    d = reg.describe_diff({"a", "b"}, {"b", "c"})
    assert d == {"only_in_a": ["a"], "only_in_b": ["c"]}
