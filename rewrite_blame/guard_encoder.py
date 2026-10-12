"""Encoder-based guard: a fine-tuned text encoder (Laya, i.e. ModernBERT-large with a decision
head, or any sequence-classification checkpoint) that reads a text serialisation of a firing and
scores "this firing will hurt".

Heavy dependencies (transformers, torch training) are imported lazily; the serialisation and the
evaluation protocol are shared with guard_models so all guard families are compared on identical
splits. Default checkpoint: the open-weight Laya English model (Apache 2.0); any HF id works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .guard_models import FEATURES

DEFAULT_CHECKPOINT = "convai-innovations/laya"   # override with --checkpoint; falls back to answerdotai/ModernBERT-base if unavailable

QUESTION = {
    "attn": "Will fusing this attention pattern into one SDPA kernel make the model slower on this machine?",
    "conv": "Will forcing channels-last layout on this convolution network make it slower on this machine?",
    "mlp": "Will decomposing this skinny matmul into multiply and sum make the model slower on this machine?",
}


def serialize(row: dict, family: str) -> str:
    """Text the encoder sees: the question plus the same features the trees get, spelled out."""
    parts = []
    for n in FEATURES[family]:
        v = row.get(n, 0)
        key = n.replace("cfg_", "")
        if key == "x86":
            parts.append("machine: x86 Xeon with oneDNN" if v else "machine: Apple M4 Pro arm64")
        elif key == "threads":
            parts.append(f"threads: {int(v)}")
        elif key == "depthwise":
            parts.append("depthwise convolutions: yes" if v else "depthwise convolutions: no")
        else:
            parts.append(f"{key}: {int(v) if float(v).is_integer() else v}")
    return QUESTION[family] + " State: " + "; ".join(parts) + "."


@dataclass
class EncoderGuard:
    checkpoint: str
    family: str
    model: object = None
    tokenizer: object = None

    def fit(self, rows: list[dict], epochs: int = 6, lr: float = 2e-5, batch_size: int = 16, device: str = "cpu", log=print):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        texts = [serialize(r, self.family) for r in rows]
        y = torch.tensor([1 if r["label"] == "harmful" else 0 for r in rows])
        tok = AutoTokenizer.from_pretrained(self.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=2, ignore_mismatched_sizes=True)
        model.to(device).train()
        # class weights: harmful firings are the minority
        pos = max(1, int(y.sum())); neg = max(1, len(y) - pos)
        w = torch.tensor([len(y) / (2 * neg), len(y) / (2 * pos)], dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        n = len(texts)
        for ep in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                out = model(input_ids=enc["input_ids"][idx].to(device), attention_mask=enc["attention_mask"][idx].to(device))
                loss = torch.nn.functional.cross_entropy(out.logits, y[idx].to(device), weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss) * len(idx)
            log(f"[encoder] epoch {ep+1}/{epochs} loss {total/n:.4f}")
        model.eval()
        self.model, self.tokenizer = model, tok
        return self

    def predict_proba(self, rows: list[dict], device: str = "cpu") -> list[float]:
        import torch
        texts = [serialize(r, self.family) for r in rows]
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
        return torch.softmax(logits, dim=-1)[:, 1].tolist()

    def predict(self, rows: list[dict], threshold: float = 0.5) -> list[int]:
        return [int(p >= threshold) for p in self.predict_proba(rows)]


def evaluate_encoder(rows: list[dict], family: str, checkpoint: str = DEFAULT_CHECKPOINT, n_folds: int = 5,
                     epochs: int = 6, log=print) -> dict:
    """Same grouped K-fold as the trees; returns the same metrics dict shape as EvalResult.to_dict()."""
    from sklearn.model_selection import GroupKFold
    from .guard_models import _groups
    groups = _groups(rows, family)
    folds = max(2, min(n_folds, len(set(groups))))
    y = [1 if r["label"] == "harmful" else 0 for r in rows]
    pred = [0] * len(rows)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(rows, y, groups)):
        train = [rows[i] for i in tr]
        if len({r["label"] == "harmful" for r in train}) < 2:
            continue
        g = EncoderGuard(checkpoint, family).fit(train, epochs=epochs, log=log)
        for i, p in zip(te, g.predict([rows[i] for i in te])):
            pred[i] = p
        log(f"[encoder] fold {k+1}/{folds} done")
    n = len(rows)
    harmful = [i for i, t in enumerate(y) if t == 1]
    nonharm = [i for i, t in enumerate(y) if t == 0]
    cost = sum(abs(r["ratio"] - 1) * 100 for r, p, t in zip(rows, pred, y) if (t == 1 and p == 0) or (r["label"] == "helpful" and p == 1)) / n
    default_cost = sum(abs(r["ratio"] - 1) * 100 for r, t in zip(rows, y) if t == 1) / n
    return {"family": family, "model": f"encoder:{checkpoint.split('/')[-1]}", "n": n, "harmful": len(harmful), "folds": folds,
            "accuracy": sum(int(p == t) for p, t in zip(pred, y)) / n, "default_accuracy": len(nonharm) / n,
            "recall": (sum(pred[i] for i in harmful) / len(harmful)) if harmful else 1.0,
            "false_alarm_rate": (sum(pred[i] for i in nonharm) / len(nonharm)) if nonharm else 0.0,
            "cost_weighted": cost, "default_cost_weighted": default_cost, "per_fold": []}
