"""Turning trained guards into compile-time decisions.

Two kinds of decision:
* pattern-level: when a gated rule matches, derive the guard's features from the matched tensors
  and ask the bundle; a "harmful" answer makes the rule refuse to fire (counted in GUARDED);
* model-level (channels-last layout optimisation is a per-compile flag): derive features from the
  model's modules and inputs before compiling and drop the switch from the state if the guard says
  harmful.
"""

from __future__ import annotations

from collections import Counter

GUARDED: Counter = Counter()      # switch id -> firings blocked by a guard
DECISIONS: list = []              # log of guard decisions for the current measurement


def family_key(switch_id: str) -> str | None:
    if "_sfdp_pattern" in switch_id:
        return "attn"
    if "decompose_mm" in switch_id or "decompose_addmm" in switch_id or "decompose_bmm" in switch_id:
        return "mlp"
    return None


def live_features(family: str, match_feats: dict, x86: int, threads: int) -> dict | None:
    """Map the features recorded at a match to the dataset's cfg_* columns."""
    tensors = match_feats.get("tensors") or []
    shapes = [t["shape"] for t in tensors if t.get("shape") and all(isinstance(d, int) for d in t["shape"])]
    if family == "attn":
        four = [s for s in shapes if len(s) == 4]
        if not four:
            return None
        B, H, T, D = four[0]
        return {"cfg_B": B, "cfg_T": T, "cfg_H": H, "cfg_D": D, "cfg_tokens": B * T, "cfg_qk_numel": B * H * T * T,
                "x86": x86, "threads": threads}
    if family == "mlp":
        two = [s for s in shapes if len(s) == 2]
        if len(two) < 2:
            return None
        (M, K), (K2, N) = two[0], two[1]
        I, Hd = min(K, N), max(K, N)
        return {"cfg_M": M, "cfg_I": I, "cfg_Hd": Hd, "cfg_weight_numel": 3 * I * Hd, "cfg_flops_per_byte": M / 2.0,
                "x86": x86, "threads": threads}
    return None


def model_features_conv(model, inputs, x86: int, threads: int) -> dict | None:
    """Features for the layout (channels-last) guard from a module tree: any depthwise conv, max
    channels, kernel size, number of convs, input resolution."""
    try:
        import torch.nn as nn
    except Exception:
        return None
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return None
    dw = any(m.groups == m.in_channels and m.groups > 1 for m in convs)
    C = max(m.out_channels for m in convs)
    K = max(max(m.kernel_size) for m in convs)
    x = inputs[0] if inputs else None
    R = int(x.shape[-1]) if x is not None and x.dim() == 4 else 0
    B = int(x.shape[0]) if x is not None and x.dim() >= 1 else 1
    return {"cfg_C": C, "cfg_R": R, "cfg_K": K, "cfg_L": len(convs), "cfg_depthwise": int(dw),
            "cfg_act_numel": B * C * R * R, "x86": x86, "threads": threads}


def network_features(model, inputs) -> dict | None:
    """Features describing a whole convolutional network, from one forward pass with hooks on every Conv2d:
    batch, input resolution, number of convs, total multiply-accumulates, the fraction of that work done by
    depthwise and by 1x1 convolutions, work-weighted mean and max output channels, and the total activation
    volume. The same numbers exist for a 3-layer synthetic stack and an 80-layer real network."""
    import torch
    import torch.nn as nn
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return None
    recs = []

    def hook(mod, inp, out):
        x = inp[0]
        k = mod.kernel_size[0] * mod.kernel_size[1]
        macs = out.numel() / out.shape[0] * (mod.in_channels // mod.groups) * k
        recs.append({"macs": macs, "dw": mod.groups == mod.in_channels and mod.groups > 1, "one": k == 1,
                     "out_c": mod.out_channels, "act": out.numel() / out.shape[0], "k": mod.kernel_size[0]})
    hs = [m.register_forward_hook(hook) for m in convs]
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for h in hs:
            h.remove()
    if not recs:
        return None
    x = inputs[0]
    B = int(x.shape[0]); R = int(x.shape[-1]) if x.dim() == 4 else 0
    total = sum(r["macs"] for r in recs) or 1.0
    return {"cfg_B": B, "cfg_R": R, "cfg_n_conv": len(recs), "cfg_macs": total,
            "cfg_dw_frac": sum(r["macs"] for r in recs if r["dw"]) / total,
            "cfg_one_frac": sum(r["macs"] for r in recs if r["one"]) / total,
            "cfg_mean_channels": sum(r["macs"] * r["out_c"] for r in recs) / total,
            "cfg_max_channels": max(r["out_c"] for r in recs),
            "cfg_mean_k": sum(r["macs"] * r["k"] for r in recs) / total,
            "cfg_act_numel": B * sum(r["act"] for r in recs)}


def model_profile(model, inputs) -> dict | None:
    """General whole-model profile from one forward pass: the conv statistics of ``network_features`` plus
    linear-layer work (count, MACs, fraction of MACs with M <= 2 i.e. decode-shaped, mean K/N), attention
    shapes seen by ``scaled_dot_product_attention`` or by a 4-D softmax (manual attention), normalisation
    layer count, parameter count, batch/tokens and precision. Every key is ``cfg_*`` so it can be a guard
    feature directly."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    x = inputs[0] if inputs else None
    B = int(x.shape[0]) if isinstance(x, torch.Tensor) and x.dim() >= 1 else 1
    tokens = B * int(x.shape[1]) if isinstance(x, torch.Tensor) and x.dim() in (2, 3) else B
    prof = {"cfg_B": B, "cfg_tokens": tokens,
            "cfg_params": sum(p.numel() for p in model.parameters()),
            "cfg_n_norm": sum(1 for m in model.modules() if "Norm" in type(m).__name__),
            "cfg_bf16": int(any(p.dtype in (torch.bfloat16, torch.float16) for p in model.parameters()))}
    lin = []
    def lin_hook(mod, inp, out):
        a = inp[0]; M = a.numel() // a.shape[-1] if a.dim() >= 1 else 1
        lin.append((M, mod.in_features, mod.out_features))
    attn = []
    orig_sdpa, orig_softmax, orig_tsoftmax, orig_msoftmax = F.scaled_dot_product_attention, F.softmax, torch.softmax, torch.Tensor.softmax
    def sdpa(q, k, v, *a, **kw):
        if q.dim() == 4:
            attn.append(("sdpa", int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3])))
        return orig_sdpa(q, k, v, *a, **kw)
    def softmax(t, *a, **kw):
        if t.dim() == 4 and t.shape[-1] == t.shape[-2]:
            attn.append(("manual", int(t.shape[0]), int(t.shape[1]), int(t.shape[2]), 0))
        return orig_softmax(t, *a, **kw)
    def conv1d_hook(mod, inp, out):   # HF GPT-2 style Conv1D: a linear layer with weight (in, nf)
        a = inp[0]; M = a.numel() // a.shape[-1] if a.dim() >= 1 else 1
        lin.append((M, int(mod.weight.shape[0]), int(mod.nf)))
    hs = [m.register_forward_hook(lin_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    hs += [m.register_forward_hook(conv1d_hook) for m in model.modules() if type(m).__name__ == "Conv1D" and hasattr(m, "nf")]
    def tsoftmax(t, *a, **kw):
        if t.dim() == 4 and t.shape[-1] == t.shape[-2]:
            attn.append(("manual", int(t.shape[0]), int(t.shape[1]), int(t.shape[2]), 0))
        return orig_tsoftmax(t, *a, **kw)
    def msoftmax(self, *a, **kw):
        if self.dim() == 4 and self.shape[-1] == self.shape[-2]:
            attn.append(("manual", int(self.shape[0]), int(self.shape[1]), int(self.shape[2]), 0))
        return orig_msoftmax(self, *a, **kw)
    F.scaled_dot_product_attention, F.softmax, torch.softmax, torch.Tensor.softmax = sdpa, softmax, tsoftmax, msoftmax
    try:
        conv = network_features(model, inputs)          # runs the forward pass itself when the model has convs
        if conv is None:                                  # conv-free model: run the forward so the hooks fire
            conv = {}
            with torch.no_grad():
                model(*inputs)
    finally:
        F.scaled_dot_product_attention, F.softmax, torch.softmax, torch.Tensor.softmax = orig_sdpa, orig_softmax, orig_tsoftmax, orig_msoftmax
        for h in hs:
            h.remove()
    prof.update({k: v for k, v in conv.items() if k not in ("cfg_B",)})
    prof.setdefault("cfg_R", int(x.shape[-1]) if isinstance(x, torch.Tensor) and x.dim() == 4 else 0)
    if lin:
        macs = [M * K * N for M, K, N in lin]; tot = sum(macs) or 1.0
        prof.update({"cfg_n_linear": len(lin), "cfg_linear_macs": tot,
                     "cfg_lin_decode_frac": sum(m for (M, K, N), m in zip(lin, macs) if M <= 2) / tot,
                     "cfg_lin_mean_M": sum(M * m for (M, K, N), m in zip(lin, macs)) / tot,
                     "cfg_lin_mean_K": sum(K * m for (M, K, N), m in zip(lin, macs)) / tot,
                     "cfg_lin_mean_N": sum(N * m for (M, K, N), m in zip(lin, macs)) / tot,
                     "cfg_lin_max_N": max(N for _, _, N in lin)})
    else:
        prof.update({"cfg_n_linear": 0, "cfg_linear_macs": 0.0, "cfg_lin_decode_frac": 0.0, "cfg_lin_mean_M": 0.0,
                     "cfg_lin_mean_K": 0.0, "cfg_lin_mean_N": 0.0, "cfg_lin_max_N": 0})
    if attn:
        # manual attention is recorded at the softmax (no head dim); prefer an sdpa record for the shape
        kind, b, h, t, d = max(attn, key=lambda r: (r[1] * r[2] * r[3] * r[3], r[0] == "sdpa"))
        if d == 0:
            d_cands = [r[4] for r in attn if r[4]]
            d = d_cands[0] if d_cands else 0
        prof.update({"cfg_n_attn": len(attn), "cfg_attn_B": b, "cfg_attn_H": h, "cfg_attn_T": t, "cfg_attn_D": d,
                     "cfg_attn_qk_numel": sum(r[1] * r[2] * r[3] * r[3] for r in attn),
                     "cfg_attn_sdpa": int(kind == "sdpa")})
    else:
        prof.update({"cfg_n_attn": 0, "cfg_attn_B": 0, "cfg_attn_H": 0, "cfg_attn_T": 0, "cfg_attn_D": 0,
                     "cfg_attn_qk_numel": 0, "cfg_attn_sdpa": 0})
    if "cfg_n_conv" not in prof:
        prof.update({"cfg_n_conv": 0, "cfg_macs": 0.0, "cfg_dw_frac": 0.0, "cfg_one_frac": 0.0, "cfg_mean_channels": 0.0,
                     "cfg_max_channels": 0, "cfg_mean_k": 0.0, "cfg_act_numel": 0.0})
    return prof


def reset() -> None:
    GUARDED.clear()
    DECISIONS.clear()
