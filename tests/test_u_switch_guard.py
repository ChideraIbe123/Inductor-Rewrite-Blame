"""Switch-level real-model guards: profile features, dataset state construction, model-set specs, dtype build."""
import torch
import torch.nn as nn

from rewrite_blame.live_guards import model_profile
from rewrite_blame.switches import Registry, Switch


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(32, 96); self.out = nn.Linear(32, 32); self.ln = nn.LayerNorm(32)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, 4, 8).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-1, -2)) / 8 ** 0.5).softmax(-1) @ v        # manual attention (Tensor.softmax)
        return self.ln(self.out(a.transpose(1, 2).reshape(B, T, 32)))


def test_model_profile_sees_linear_attention_and_norm():
    m = _Attn().eval()
    p = model_profile(m, (torch.randn(2, 16, 32),))
    assert p["cfg_B"] == 2 and p["cfg_tokens"] == 32 and p["cfg_n_linear"] == 2 and p["cfg_n_norm"] == 1
    assert p["cfg_n_attn"] == 1 and p["cfg_attn_sdpa"] == 0 and (p["cfg_attn_B"], p["cfg_attn_H"], p["cfg_attn_T"]) == (2, 4, 16)
    assert p["cfg_linear_macs"] == 32 * 32 * 96 + 32 * 32 * 32 and p["cfg_lin_decode_frac"] == 0.0
    assert p["cfg_n_conv"] == 0 and p["cfg_params"] == sum(q.numel() for q in m.parameters())
    # decode-shaped: one token per sequence -> every linear has M = B <= 2
    p1 = model_profile(m, (torch.randn(1, 1, 32),))
    assert p1["cfg_lin_decode_frac"] == 1.0 and p1["cfg_tokens"] == 1


def test_switch_dataset_states_follow_the_default_direction(tmp_path, monkeypatch):
    reg = Registry([Switch(id="a/on_by_default", kind="config", family="a", default_on=True, config_path="x.a", on_value=True, off_value=False),
                    Switch(id="b/off_by_default", kind="config", family="b", default_on=False, config_path="x.b", on_value=True, off_value=False)])
    import rewrite_blame.dataset as ds
    seen = {}

    registry = reg

    class FakeRunner:
        env = "Apple M4 Pro arm64|torch|T8"; threads = 8
        def __init__(self):
            self.reg = registry
        class cfg: session_tag = "t"
        def noise(self, name, state, runs=7):
            class NM: center = 10.0; tau = 0.3
            return NM(), None
        def measure(self, name, state, rep=0):
            seen.setdefault(name, []).append(frozenset(state))
            class M: error = None; program_hash = "h" + str(len(state)); fired = {}; median_ms = 10.0 + len(state); firings = []; kernel_count = 1; code = {}; compile_s = 1.0; peak_mem_mb = 100.0
            return M()
    monkeypatch.setattr(ds, "profile_of", lambda name: {"cfg_B": 1, "cfg_arch": "x"}, raising=False)
    import rewrite_blame.models as models
    monkeypatch.setattr(models, "profile_of", lambda name, seed=0: {"cfg_B": 1, "cfg_arch": "x"})
    rows = ds.collect_switch(FakeRunner(), "b/off_by_default", ["m1"], tmp_path / "d.jsonl", tau_pct=3.0, log=lambda *a: None)
    assert rows[0]["default_on"] is False and rows[0]["family"] == "switch:b/off_by_default"
    on, off = seen["m1"][0], seen["m1"][1]
    assert "b/off_by_default" in on and "b/off_by_default" not in off and "a/on_by_default" in on and "a/on_by_default" in off
    rows = ds.collect_switch(FakeRunner(), "a/on_by_default", ["m2"], tmp_path / "e.jsonl", tau_pct=3.0, log=lambda *a: None)
    assert rows[0]["default_on"] is True and rows[0]["label"] in ("harmful", "helpful", "neutral", "no_effect")


def test_target_and_mode_depend_on_default_direction():
    from rewrite_blame.guard_models import target_of, features_for
    rows_on = [{"family": "switch:a", "default_on": True, "label": "harmful", "cfg_B": 1, "cfg_x": 2.0, "cfg_arch": "r"}]
    rows_off = [{"family": "switch:b", "default_on": False, "label": "helpful", "cfg_B": 1, "cfg_arch": "r"}]
    assert target_of("switch:a", rows_on) == "harmful" and target_of("switch:b", rows_off) == "helpful"
    assert target_of("conv", rows_off) == "harmful"
    assert features_for("switch:a", rows_on) == ["cfg_B", "cfg_x", "x86", "cuda", "threads"]   # numeric cfg_* only, arch excluded


def test_expand_spec_and_zoos():
    from rewrite_blame import models
    from rewrite_blame.models import hf_zoo
    assert len(models.expand_spec("corpus:images")) == 7
    assert len(models.expand_spec("zoo:conv")) == 38 * 3 and len(models.expand_spec("zoo:all")) > len(models.expand_spec("zoo:conv"))
    assert models.expand_spec("resnet18, zoo_resnet34_b*") == ["resnet18", "zoo_resnet34_b8", "zoo_resnet34_b32", "zoo_resnet34_b64"]
    assert hf_zoo.parse("hfzoo_bert_tiny_prefill_s128_b8") == ("bert_tiny", "prefill", 128, 8) and hf_zoo.parse("hfzoo_nope_prefill_s1_b1") is None
    assert models.arch_of("hfzoo_llama_tiny_decode_s1_b8") == "llama_tiny" and models.arch_of("zoo_vgg11_b32") == "vgg11"
    spec = models.get("hfzoo_bert_tiny_prefill_s128_b8")
    m, inputs = spec.build()
    assert inputs[0].shape == (8, 128) and inputs[0].dtype == torch.long


def test_bf16_build_casts_floats_only(monkeypatch):
    monkeypatch.setenv("REWRITE_BLAME_DTYPE", "bf16")
    from rewrite_blame import env, models
    assert env.fingerprint(4).endswith("|bf16")
    m, inputs = models.build("hfzoo_bert_tiny_prefill_s128_b8")
    assert next(m.parameters()).dtype == torch.bfloat16 and inputs[0].dtype == torch.long
    m2, inputs2 = models.build("attention_block")
    assert all(t.dtype == torch.bfloat16 for t in inputs2 if t.is_floating_point())


def test_guard_step_tempfile_prefix_is_filesystem_safe(tmp_path):
    import tempfile
    families = ["switch:joint/constant_folding", "switch:lowering/freezing"]
    safe = "_".join(families).replace("/", "_").replace(":", "_")[:80]
    f = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, prefix=f"guards_{safe}_", dir=tmp_path)
    assert "/" not in safe and f.name.endswith(".pkl")


def test_guards_eval_subbundles_keep_the_mode(tmp_path):
    """The end-to-end evaluation rebuilds one bundle per step; an 'enable' guard must stay an enable guard."""
    from rewrite_blame.guard_models import GuardBundle, fit_all
    rows = [{"cfg_B": 8, "cfg_n_conv": 10 + i, "cfg_arch": f"a{i}", "cuda": 1, "x86": 0, "threads": 4,
             "label": "helpful" if i % 2 else "neutral", "env": "x", "default_on": False, "family": "switch:f"} for i in range(8)]
    clf, names = fit_all(rows, "switch:f", "tree")
    b = GuardBundle(); b.add("switch:f", "tree", clf, names, mode="enable"); b.save(tmp_path / "b.pkl")
    full = GuardBundle.load(tmp_path / "b.pkl")
    sub = GuardBundle()
    g = full.guards["switch:f"]
    sub.add("switch:f", g["kind"], g["clf"], g["names"], g["threshold"], mode=g.get("mode", "block"))
    assert sub.guards["switch:f"]["mode"] == "enable"
