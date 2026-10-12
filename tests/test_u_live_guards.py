from rewrite_blame.live_guards import family_key, live_features


def test_family_key():
    assert family_key("joint/joint_graph.patterns/_sfdp_pattern_11_inference") == "attn"
    assert family_key("optimus/post_grad.POST_GRAD_PATTERNS[decompose_mm_pass]/decompose_mm") == "mlp"
    assert family_key("joint/joint_graph.patterns/pointless_convert") is None


def test_live_features_attn_and_mlp():
    f = live_features("attn", {"tensors": [{"shape": [4, 8, 128, 64]}, {"shape": [4, 8, 128, 64]}]}, 0, 8)
    assert f["cfg_B"] == 4 and f["cfg_T"] == 128 and f["cfg_H"] == 8 and f["cfg_D"] == 64 and f["cfg_qk_numel"] == 4 * 8 * 128 * 128
    g = live_features("mlp", {"tensors": [{"shape": [1, 1024]}, {"shape": [1024, 4096]}]}, 1, 4)
    assert g["cfg_M"] == 1 and g["cfg_I"] == 1024 and g["cfg_Hd"] == 4096 and g["x86"] == 1
    assert live_features("attn", {"tensors": [{"shape": [2, 3]}]}, 0, 8) is None
    assert live_features("mlp", {"tensors": []}, 0, 8) is None
    assert live_features("other", {}, 0, 8) is None


def test_model_features_conv():
    import torch, torch.nn as nn
    from rewrite_blame.live_guards import model_features_conv
    m = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.Conv2d(32, 64, 1))
    f = model_features_conv(m, (torch.randn(4, 32, 56, 56),), 0, 8)
    assert f["cfg_depthwise"] == 1 and f["cfg_C"] == 64 and f["cfg_K"] == 3 and f["cfg_L"] == 2 and f["cfg_R"] == 56
    assert model_features_conv(nn.Linear(3, 3), (torch.randn(2, 3),), 0, 8) is None
