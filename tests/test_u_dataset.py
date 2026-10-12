from rewrite_blame.dataset import label_pair
from rewrite_blame.models import sweeps


def test_label_pair_and_rule():
    assert label_pair([1.20, 1.15], 0.02) == ("harmful", True)
    assert label_pair([1.20, 1.01], 0.02) == ("neutral", True)      # second pair disagrees
    assert label_pair([0.90, 0.92], 0.02) == ("helpful", True)
    assert label_pair([1.01], 0.02) == ("neutral", False)
    assert label_pair([1.20], 0.02) == ("harmful", False)           # unconfirmed single pair
    assert label_pair([], 0.02) == ("neutral", False)


def test_sweep_names_roundtrip_and_grids():
    for n in ["sweep_attn_B2_T64_H4_D32", "sweep_conv_C32_R28_K5_L1_dense", "sweep_mlp_M16_I512_Hd1024"]:
        assert sweeps.resolve(n) is not None and sweeps.config_of(n)
        assert sweeps.sweep_family_of(n) in ("attn", "conv", "mlp")
    assert sweeps.resolve("resnet18") is None and sweeps.sweep_family_of("resnet18") is None
    assert len(sweeps.grid("attn", "small")) < len(sweeps.grid("attn"))
    assert all(sweeps.resolve(n) for n in sweeps.grid("conv", "small"))
    assert sweeps.config_of("sweep_conv_C32_R28_K3_L2_dw")["depthwise"] == 1
    assert sweeps.config_of("sweep_mlp_M1_I1024_Hd2048")["flops_per_byte"] == 0.5
