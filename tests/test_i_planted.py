"""Hand-written graphs with a known rewrite planted in each. For every family we check that the
rule fires by default, that its switch suppresses it, that the graph/kernels change in the expected
way, and that outputs stay correct under both states."""
import pytest

from rewrite_blame.discover import discover
from rewrite_blame.measure import measure

SFDP = "joint/joint_graph.patterns/_sfdp_pattern_11_inference"
SDPA_EXTERN = "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu"


@pytest.fixture(scope="session")
def reg():
    return discover()


@pytest.fixture(scope="session")
def default(reg):
    return reg.default_state()


def _m(reg, model, state, **kw):
    m = measure(reg, model, state, time_it=False, threads=4, **kw)
    assert m.error is None, m.error
    assert m.correct, f"{model} produced wrong outputs under {sorted(reg.default_state() ^ set(state))}: max err {m.max_abs_err}"
    return m


def test_attention_block_sdpa_fusion_and_its_switch(reg, default):
    m_on = _m(reg, "attention_block", default)
    assert m_on.fired.get(SFDP) == 1
    assert SDPA_EXTERN in m_on.code["extern_calls"]
    sfdp_ids = {s.id for s in reg.of_kind("pattern") if "_sfdp_pattern" in s.id}
    m_off = _m(reg, "attention_block", default - sfdp_ids)
    assert SDPA_EXTERN not in m_off.code["extern_calls"]
    assert m_off.suppressed.get(SFDP) == 1
    assert m_off.code["kernel_count"] > m_on.code["kernel_count"]
    assert m_off.code["extern_calls"].get("extern_kernels.bmm") == 2
    # the manual softmax now shows up as a fused reduction kernel
    assert any("softmax" in k["name"] for k in m_off.code["kernels"])
    # disabling only pattern 11 is not enough: pattern 12 catches the same graph (planted interaction)
    m_only11 = _m(reg, "attention_block", default - {SFDP})
    assert SDPA_EXTERN in m_only11.code["extern_calls"]
    assert "joint/joint_graph.patterns/_sfdp_pattern_12_inference" in m_only11.fired


def test_norm_mlp_pointless_convert_and_rsqrt(reg, default):
    pc = "joint/joint_graph.patterns/pointless_convert"
    rs = "post_grad/post_grad.pass_patterns[1]/reciprocal_sqrt_to_rsqrt"
    m_on = _m(reg, "norm_mlp", default)
    assert m_on.fired.get(pc, 0) >= 1, m_on.fired
    assert m_on.fired.get(rs, 0) >= 1, m_on.fired
    assert m_on.graph_ops.get("aten.rsqrt.default", 0) >= 1
    m_off = _m(reg, "norm_mlp", default - {pc, rs})
    assert m_off.suppressed.get(pc, 0) >= 1 and m_off.suppressed.get(rs, 0) >= 1
    # without the rsqrt rule the post-grad graph keeps sqrt + reciprocal/div
    assert m_off.graph_ops.get("aten.rsqrt.default", 0) < m_on.graph_ops.get("aten.rsqrt.default", 0)
    assert m_off.graph_ops.get("aten.sqrt.default", 0) > m_on.graph_ops.get("aten.sqrt.default", 0)
    # without pointless_convert the dtype round trip survives into the graph
    assert m_off.graph_ops.get("prims.convert_element_type.default", 0) > m_on.graph_ops.get("prims.convert_element_type.default", 0)


def test_decode_mlp_decompose_mm_opt_in(reg, default):
    m_base = _m(reg, "decode_mlp", default)
    assert m_base.code["extern_calls"].get("extern_kernels.mm", 0) == 3
    m_dec = _m(reg, "decode_mlp", default | {"optimus/decompose_mm_pass"})
    assert m_dec.counters.get("decompose_mm", 0) >= 1, m_dec.counters
    assert m_dec.code["extern_calls"].get("extern_kernels.mm", 0) < 3
    fired_dec = [k for k in m_dec.fired if "decompose_mm" in k]
    assert fired_dec, m_dec.fired
    # the pattern switch alone (opt-in option off) must not fire
    assert not any("decompose_mm" in k for k in m_base.fired)


def test_split_cat_mlp_optimus_passes(reg, default):
    m_base = _m(reg, "split_cat_mlp", default)
    opt = {"optimus/normalization_pass", "optimus/split_cat_pass", "optimus/batch_linear"}
    m_opt = _m(reg, "split_cat_mlp", default | opt)
    fired_pre = {k: v for k, v in m_opt.fired.items() if k.startswith("optimus/") or k.startswith("pre_grad/")}
    assert fired_pre, m_opt.fired
    assert m_opt.graph_ops != m_base.graph_ops
    assert any(k in m_opt.counters for k in ("batch_linear", "normalization_pass", "split_cat_pass", "pattern_matcher_count"))


def test_conv_bn_relu_freezing_folds_bn(reg, default):
    import torch
    m_base = _m(reg, "conv_bn_relu_stack", default)
    # in inference the bn is decomposed, but its origin still names the fused kernels
    assert any("batch_norm" in k["name"] for k in m_base.code["kernels"]), m_base.code["kernels"]
    assert m_base.code["extern_calls"].get("extern_kernels.convolution") == 3
    m_fz = _m(reg, "conv_bn_relu_stack", default | {"lowering/freezing"})
    # freezing folds bn into the conv weights: no bn anywhere, only conv + relu remain
    assert not any("batch_norm" in k["name"] for k in m_fz.code["kernels"]), m_fz.code["kernels"]
    assert not any("batch_norm" in op for op in m_fz.graph_ops)
    assert m_fz.graph_ops.get("aten.convolution.default", 0) == 3 or any("mkldnn" in op for op in m_fz.graph_ops)
    assert m_fz.metrics["ir_nodes_pre_fusion"] < m_base.metrics["ir_nodes_pre_fusion"]
    if torch._C._has_mkldnn and not torch.backends.mkldnn.is_available():
        pytest.skip("mkldnn present but unavailable")
    if torch._C._has_mkldnn:
        # with oneDNN (x86 VM) the conv+relu pairs become mkldnn fused convolutions
        assert any("mkldnn" in op for op in m_fz.graph_ops) or any("mkldnn" in e for e in m_fz.code["extern_calls"]), (
            m_fz.graph_ops, m_fz.code["extern_calls"])
        assert any(k.startswith("freezing/") for k in m_fz.fired), m_fz.fired


def test_cat_slice_cat_rules_fire(reg, default):
    m = _m(reg, "cat_slice_cat", default)
    post = {k: v for k, v in m.fired.items() if "cat" in k}
    assert post, m.fired


def test_master_switch_disables_every_pattern(reg, default):
    m = _m(reg, "norm_mlp", default - {"master/pattern_matcher"})
    assert not m.fired, m.fired
    assert m.counters.get("pattern_matcher_count", 0) == 0


def test_inplace_buffers_off_allocates_more(reg, default):
    m_on = _m(reg, "norm_mlp", default)
    m_off = _m(reg, "norm_mlp", default - {"sched/inplace_buffers"})
    assert m_off.code["alloc_bytes"] > m_on.code["alloc_bytes"]
    assert m_off.code["kernel_count"] == m_on.code["kernel_count"]
    # in-place kernels take in_out_ptr arguments; without the optimisation they disappear
    assert sum(k["n_inout_ptr"] for k in m_on.code["kernels"]) > sum(k["n_inout_ptr"] for k in m_off.code["kernels"])


def test_sfdp_pattern_11_alone_is_caught_by_pattern_12(reg, default):
    """The planted interaction: disabling rule 11 changes which rule fires but not the graph."""
    m_on = _m(reg, "attention_block", default)
    m_11off = _m(reg, "attention_block", default - {SFDP})
    assert m_on.code["kernel_count"] == m_11off.code["kernel_count"]
    assert m_on.code["extern_calls"] == m_11off.code["extern_calls"]
    assert "joint/joint_graph.patterns/_sfdp_pattern_12_inference" in m_11off.fired
    assert m_11off.suppressed.get(SFDP) == 1
