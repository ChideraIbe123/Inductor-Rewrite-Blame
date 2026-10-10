"""Stretch goal groundwork: learn, per rule family, when a firing is harmful.

A *labelled firing* joins (a) the features recorded when a rule fired in the default compile of a
model on a machine with (b) the measured effect of turning that rule off in the leave-one-out
sweep of the same model and machine: ``harmful`` if the model got faster beyond tau without the
rule, ``helpful`` if it got slower, ``neutral`` otherwise. A guard is a classifier that predicts
``harmful`` from the features; it is evaluated leave-one-model-out against Inductor's default
policy (always fire, i.e. never predict harmful). Pure Python, no torch.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable


def family_of(switch_id: str) -> str:
    """Rule family: strip pattern variant suffixes (_sfdp_pattern_11_inference -> _sfdp_pattern)."""
    name = switch_id.rsplit("/", 1)[-1]
    import re
    name = re.sub(r"_pattern_\d+(_half)?(_inference|_training)?$", "_pattern", name)
    name = re.sub(r"#\d+$", "", name)
    return name


def firing_features(firings: list[dict], switch_id: str) -> dict | None:
    fs = [f for f in firings if f.get("switch") == switch_id]
    if not fs:
        return None
    numels = [f.get("max_numel") for f in fs if f.get("max_numel")]
    n_nodes = [f.get("n_nodes", 0) for f in fs]
    dtypes = sorted({d for f in fs for d in f.get("dtypes", [])})
    return {
        "n_fires": len(fs),
        "log_max_numel": math.log10(max(numels)) if numels else 0.0,
        "log_min_numel": math.log10(min(numels)) if numels else 0.0,
        "n_nodes": max(n_nodes) if n_nodes else 0,
        "low_precision": int(any(d in ("float16", "bfloat16") for d in dtypes)),
    }


def build_dataset(default_rows: Iterable[dict], sweeps: Iterable[dict]) -> list[dict]:
    """``default_rows``: measurement rows of the default state (with ``firings``) keyed by
    (model, env); ``sweeps``: sweep result dicts (rows with switch, direction, delta_pct, noise)."""
    by_key = {}
    for r in default_rows:
        if r.get("firings"):
            by_key[(r["model"], r["env"])] = r
    data = []
    for s in sweeps:
        key = (s["model"], s["env"])
        base = by_key.get(key)
        if base is None:
            continue
        tau_pct = 100.0 * s["noise"]["tau"] / s["reference_ms"]
        for row in s["rows"]:
            if "error" in row or row.get("direction") != "off":
                continue
            feats = firing_features(base["firings"], row["switch"])
            if feats is None:
                continue
            d = row["delta_pct"]
            label = "harmful" if d < -tau_pct else ("helpful" if d > tau_pct else "neutral")
            data.append({"model": s["model"], "env": s["env"], "switch": row["switch"], "family": family_of(row["switch"]),
                         "x86": int("x86" in s["env"]), "threads": int(s["env"].rsplit("T", 1)[-1]) if "T" in s["env"] else 0,
                         "delta_pct": d, "tau_pct": tau_pct, "label": label, **feats})
    return data


NUMERIC = ("log_max_numel", "log_min_numel", "n_fires", "n_nodes", "x86", "threads", "low_precision")


@dataclass
class Stump:
    feature: str
    threshold: float
    harmful_if_greater: bool
    train_accuracy: float

    def predict(self, row: dict) -> bool:
        v = row.get(self.feature, 0.0)
        return (v > self.threshold) if self.harmful_if_greater else (v <= self.threshold)


def fit_stump(rows: list[dict]) -> Stump | None:
    """Best single-feature threshold for predicting label == harmful (ties -> fewer predicted harmful)."""
    y = [r["label"] == "harmful" for r in rows]
    if not rows or not any(y):
        return None
    best = None
    for f in NUMERIC:
        vals = sorted({r.get(f, 0.0) for r in rows})
        cands = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)] or [vals[0]]
        for t in cands:
            for greater in (True, False):
                pred = [((r.get(f, 0.0) > t) if greater else (r.get(f, 0.0) <= t)) for r in rows]
                acc = sum(p == yy for p, yy in zip(pred, y)) / len(rows)
                n_pos = sum(pred)
                key = (acc, -n_pos)
                if best is None or key > best[0]:
                    best = (key, Stump(f, t, greater, acc))
    return best[1]


def leave_one_model_out(rows: list[dict]) -> dict:
    """Per family: accuracy of the stump guard vs the default policy, leaving one model out at a time."""
    out = {}
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    for fam, frows in by_fam.items():
        models = sorted({r["model"] for r in frows})
        correct_guard = correct_default = n = 0
        harm_caught = harm_total = false_alarms = 0
        for m in models:
            train = [r for r in frows if r["model"] != m]
            test = [r for r in frows if r["model"] == m]
            stump = fit_stump(train)
            for r in test:
                truth = r["label"] == "harmful"
                pred = stump.predict(r) if stump else False
                n += 1
                correct_guard += pred == truth
                correct_default += (not truth)
                harm_total += truth
                harm_caught += truth and pred
                false_alarms += pred and not truth
        out[fam] = {"n": n, "models": len(models), "harmful": harm_total,
                    "guard_accuracy": correct_guard / n if n else None, "default_accuracy": correct_default / n if n else None,
                    "harmful_caught": harm_caught, "false_alarms": false_alarms,
                    "stump_on_all": asdict(fit_stump(frows)) if fit_stump(frows) else None}
    return out


def render_guards(data: list[dict], evaluation: dict) -> str:
    lines = [f"# Guard dataset: {len(data)} labelled firings", "",
             "| family | firings | models | harmful | guard acc (LOMO) | always-fire acc | caught | false alarms | stump |", "|---|---|---|---|---|---|---|---|---|"]
    for fam, e in sorted(evaluation.items(), key=lambda kv: -kv[1]["n"]):
        st = e["stump_on_all"]
        stxt = f"{st['feature']} {'>' if st['harmful_if_greater'] else '<='} {st['threshold']:.2f}" if st else "-"
        ga = f"{e['guard_accuracy']:.2f}" if e["guard_accuracy"] is not None else "-"
        da = f"{e['default_accuracy']:.2f}" if e["default_accuracy"] is not None else "-"
        lines.append(f"| {fam} | {e['n']} | {e['models']} | {e['harmful']} | {ga} | {da} | {e['harmful_caught']} | {e['false_alarms']} | {stxt} |")
    lines += ["", "## Labelled firings", "| model | machine | switch | delta % (rule off) | tau % | label | log10 max numel | fires |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(data, key=lambda r: (r["family"], r["model"])):
        lines.append(f"| {r['model']} | {r['env'].split('|')[0][:14]} | {r['switch'].rsplit('/',1)[-1]} | {r['delta_pct']:+.1f} | {r['tau_pct']:.1f} | {r['label']} | {r['log_max_numel']:.1f} | {r['n_fires']} |")
    return "\n".join(lines) + "\n"
