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


def reset() -> None:
    GUARDED.clear()
    DECISIONS.clear()
