"""Guard models trained on the sweep dataset (results/dataset_<family>.jsonl).

A guard answers, for one firing of a rewrite rule, "will this firing hurt?". Three model families
are compared on identical splits:

* trees   -- decision tree / gradient-boosted trees on numeric features (scikit-learn);
* encoder -- a fine-tuned text encoder (Laya / ModernBERT cross-encoder) on a text serialisation
             of the same features (optional, heavy; see guard_encoder.py);
* api     -- OpenAI Decisions API zero-shot / with context (optional; see guard_api.py).

This module holds the shared dataset -> features step, the tree models, the evaluation protocol
(grouped cross-validation over configurations, plus held-out real models) and serialisation of a
trained tree guard so the compile-time gate can load it.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

LABELS = ("harmful", "neutral", "helpful")

# per family: the configuration features used (cfg_* columns) plus shared machine features
FEATURES = {
    "attn": ["cfg_B", "cfg_T", "cfg_H", "cfg_D", "cfg_tokens", "cfg_qk_numel", "x86", "threads"],
    "conv": ["cfg_C", "cfg_R", "cfg_K", "cfg_L", "cfg_depthwise", "cfg_act_numel", "x86", "threads"],
    "mlp": ["cfg_M", "cfg_I", "cfg_Hd", "cfg_weight_numel", "cfg_flops_per_byte", "x86", "threads"],
}


def load_rows(paths: Iterable[Path], family: str | None = None, drop_no_effect: bool = True) -> list[dict]:
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if family and r.get("family") != family:
                continue
            if drop_no_effect and r.get("label") == "no_effect":
                continue   # the rule did not change the program: nothing for a guard to decide
            rows.append(r)
    return rows


def featurize(rows: list[dict], family: str):
    names = FEATURES[family]
    X = [[float(r.get(n, 0) or 0) for n in names] for r in rows]
    # log-scale the size-like features so thresholds are sane across orders of magnitude
    for j, n in enumerate(names):
        if n.endswith(("numel", "tokens", "_T", "_I", "_Hd", "_R", "_M", "_C")):
            for i in range(len(X)):
                X[i][j] = math.log2(X[i][j] + 1.0)
    y = [1 if r["label"] == "harmful" else 0 for r in rows]
    return X, y, names


@dataclass
class EvalResult:
    family: str
    model: str
    n: int
    harmful: int
    folds: int
    accuracy: float
    default_accuracy: float          # "always fire" = never predict harmful
    recall: float                    # harmful firings caught
    false_alarm_rate: float          # helpful/neutral firings blocked
    cost_weighted: float             # mean % lost: harmful allowed -> |ratio-1|, helpful blocked -> |ratio-1|
    default_cost_weighted: float
    per_fold: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _groups(rows: list[dict], family: str) -> list[str]:
    """Group key for cross-validation so that all sizes of one shape 'slice' leave together:
    attn by (H, D), conv by (C, depthwise), mlp by (I, Hd). Prevents near-duplicate leakage."""
    key = {"attn": ("cfg_H", "cfg_D"), "conv": ("cfg_C", "cfg_depthwise"), "mlp": ("cfg_I", "cfg_Hd")}[family]
    return [f"{r['env'].split('|')[0]}|" + "|".join(str(r.get(k)) for k in key) for r in rows]


def make_model(kind: str):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=0)
    if kind == "gbt":
        return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    raise KeyError(kind)


def evaluate(rows: list[dict], family: str, kind: str = "tree", n_folds: int = 5) -> EvalResult:
    """Grouped K-fold over configuration slices; returns accuracy, recall on harmful, false alarms
    and a cost-weighted score (mean % of runtime lost by following the guard's decisions)."""
    from sklearn.model_selection import GroupKFold
    X, y, _ = featurize(rows, family)
    groups = _groups(rows, family)
    n_groups = len(set(groups))
    folds = max(2, min(n_folds, n_groups))
    pred = [None] * len(rows)
    per_fold = []
    if sum(y) == 0 or sum(y) == len(y):
        pred = [0] * len(rows)
    else:
        for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
            clf = make_model(kind)
            ytr = [y[i] for i in tr]
            if len(set(ytr)) < 2:
                for i in te:
                    pred[i] = ytr[0]
                continue
            clf.fit([X[i] for i in tr], ytr)
            p = clf.predict([X[i] for i in te])
            for i, pi in zip(te, p):
                pred[i] = int(pi)
            per_fold.append({"fold": k, "n_test": len(te), "acc": sum(int(pred[i] == y[i]) for i in te) / len(te)})
    n = len(rows)
    acc = sum(int(p == t) for p, t in zip(pred, y)) / n
    default_acc = sum(1 for t in y if t == 0) / n
    harmful_idx = [i for i, t in enumerate(y) if t == 1]
    recall = sum(pred[i] for i in harmful_idx) / len(harmful_idx) if harmful_idx else 1.0
    nonharm = [i for i, t in enumerate(y) if t == 0]
    far = sum(pred[i] for i in nonharm) / len(nonharm) if nonharm else 0.0

    def cost(decisions):
        c = 0.0
        for r, d in zip(rows, decisions):
            eff = abs(r["ratio"] - 1.0) * 100
            if r["label"] == "harmful" and d == 0:   # allowed a harmful firing
                c += eff
            elif r["label"] == "helpful" and d == 1:  # blocked a helpful firing
                c += eff
        return c / len(rows)
    return EvalResult(family, kind, n, len(harmful_idx), folds, acc, default_acc, recall, far, cost(pred), cost([0] * n), per_fold)


def fit_all(rows: list[dict], family: str, kind: str = "tree"):
    X, y, names = featurize(rows, family)
    clf = make_model(kind)
    clf.fit(X, y)
    return clf, names


def tree_rules(clf, names: list[str]) -> str:
    """Readable rules of a fitted decision tree (what would become the extra_check)."""
    from sklearn.tree import export_text
    try:
        return export_text(clf, feature_names=list(names), decimals=2)
    except Exception:
        return "(not a single tree)"


@dataclass
class GuardBundle:
    """Serialisable set of guards: family -> (kind, fitted model, feature names, threshold of harm prob)."""
    guards: dict = field(default_factory=dict)

    def add(self, family: str, kind: str, clf, names: list[str], threshold: float = 0.5) -> None:
        self.guards[family] = {"kind": kind, "clf": clf, "names": names, "threshold": threshold}

    def predict_harmful(self, family: str, features: dict) -> bool | None:
        g = self.guards.get(family)
        if g is None:
            return None
        x = [[float(features.get(n, 0) or 0) for n in g["names"]]]
        for j, n in enumerate(g["names"]):
            if n.endswith(("numel", "tokens", "_T", "_I", "_Hd", "_R", "_M", "_C")):
                x[0][j] = math.log2(x[0][j] + 1.0)
        clf = g["clf"]
        if hasattr(clf, "predict_proba"):
            return bool(clf.predict_proba(x)[0][1] >= g["threshold"])
        return bool(clf.predict(x)[0])

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "GuardBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


def render_eval(results: list[EvalResult], rules: dict[str, str] | None = None, counts: dict | None = None) -> str:
    L = ["# Guard evaluation (grouped cross-validation over configuration slices)", ""]
    if counts:
        L += ["| family | rows | harmful | helpful | neutral |", "|---|---|---|---|---|"]
        for fam, c in counts.items():
            L.append(f"| {fam} | {sum(c.values())} | {c.get('harmful',0)} | {c.get('helpful',0)} | {c.get('neutral',0)} |")
        L.append("")
    L += ["| family | model | rows | harmful | folds | accuracy | always-fire acc | harmful caught | false alarms | cost %/firing | always-fire cost |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r.family} | {r.model} | {r.n} | {r.harmful} | {r.folds} | {r.accuracy:.2f} | {r.default_accuracy:.2f} | "
                 f"{r.recall:.2f} | {r.false_alarm_rate:.2f} | {r.cost_weighted:.2f} | {r.default_cost_weighted:.2f} |")
    if rules:
        for fam, txt in rules.items():
            L += ["", f"## Decision tree for {fam} (fit on all rows)", "```", txt.rstrip(), "```"]
    return "\n".join(L) + "\n"
