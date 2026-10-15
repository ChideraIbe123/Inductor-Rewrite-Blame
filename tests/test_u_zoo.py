"""Real-network (zoo) family: names, features from hooks, and the sweep plumbing."""
import torch
import torch.nn as nn

from rewrite_blame.live_guards import network_features
from rewrite_blame.models import sweeps, zoo


def test_network_features_counts_work_by_kind():
    net = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, groups=8),   # depthwise
                        nn.Conv2d(8, 16, 1),                        # 1x1
                        nn.Conv2d(16, 16, 3, padding=1)).eval()     # dense
    x = torch.randn(2, 8, 16, 16)
    f = network_features(net, (x,))
    assert f["cfg_B"] == 2 and f["cfg_R"] == 16 and f["cfg_n_conv"] == 3 and f["cfg_max_channels"] == 16
    dw = 16 * 16 * 8 * 1 * 9; one = 16 * 16 * 16 * 8 * 1; dense = 16 * 16 * 16 * 16 * 9
    assert abs(f["cfg_macs"] - (dw + one + dense)) < 1e-6
    assert abs(f["cfg_dw_frac"] - dw / (dw + one + dense)) < 1e-9
    assert abs(f["cfg_one_frac"] - one / (dw + one + dense)) < 1e-9
    assert f["cfg_act_numel"] == 2 * (8 * 256 + 16 * 256 + 16 * 256)
    assert network_features(nn.Linear(4, 4), (torch.zeros(1, 4),)) is None


def test_zoo_names_resolve_and_are_a_net_family():
    names = zoo.names("full")
    assert len(names) == 20 * 3 and "zoo_resnet34_b8" in names and sweeps.grid("net") == names
    assert zoo.parse("zoo_resnet34_b8") == ("resnet34", 8, 224) and zoo.parse("sweep_conv_C16_R14_K1_L1_dw") is None
    assert zoo.parse("zoo_nope_b8") is None
    assert sweeps.sweep_family_of("zoo_resnet34_b8") == "net"
    assert sweeps.FAMILY_RULE["net"]["off_spec"].endswith("layout_optimization")
    spec = sweeps.resolve("zoo_squeezenet1_1_b1")
    assert spec is not None and spec.family == "zoo"


def test_zoo_features_come_from_a_real_build():
    f = sweeps.config_of("zoo_squeezenet1_1_b1")
    assert f["arch"] == "squeezenet1_1" and f["B"] == 1 and f["R"] == 224 and f["n_conv"] == 26
    assert 0 < f["one_frac"] < 1 and f["dw_frac"] == 0
    from rewrite_blame.guard_models import FEATURES, featurize
    X, y, names = featurize([{**{f"cfg_{k}": v for k, v in f.items()}, "label": "harmful", "cuda": 1, "x86": 0, "threads": 4}], "net")
    assert len(X[0]) == len(FEATURES["net"]) and y == [1]
