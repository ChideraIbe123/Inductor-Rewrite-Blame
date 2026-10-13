"""LaTeX tables for the guards section of the final report.  Usage: python scripts/guard_tables.py [out_dir]

Writes dataset_dist.tex, guard_cv_<tag>.tex (one per evaluation file) and e2e_<tag>.tex (one per end-to-end file),
plus guard_summary.json with the headline numbers (geometric-mean change and worst regression per end-to-end run).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

OUT = sys.argv[1] if len(sys.argv) > 1 else "report/final/tables"


def tex(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace("{", r"\{").replace("}", r"\}"))


def write(name, body):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body.rstrip("\n") + "%\n")
    print("wrote", name)


def table_env(rows, colspec, header, caption, label="", size="small", pos="h"):
    ncol = sum(colspec.count(c) for c in "lrcp")
    body = "\n".join(rows) if rows else "\\multicolumn{%d}{c}{(no data yet)} \\\\" % ncol
    lab = f"\\label{{{label}}}" if label else ""
    return (f"\\begin{{table}}[{pos}]\\centering\\{size}\n\\begin{{tabular}}{{{colspec}}}\\toprule\n{header} \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\n\\caption{{{caption}}}{lab}\n\\end{{table}}\n")


os.makedirs(OUT, exist_ok=True)
FAMILY = {"mlp": "decode MLP (decompose\\_mm)", "conv": "conv layout (channels-last)", "attn": "attention fusion (SDPA)"}
MACHINE = {"results": "Apple M4 Pro", "results/vm": "Xeon (x86, oneDNN)"}
MODEL = {"inductor-heuristic": "Inductor's rule", "tree": "decision tree", "gbt": "gradient-boosted trees",
         "logreg": "logistic regression", "decisions-api+context": "Decisions API (+context)", "decisions-api": "Decisions API",
         "encoder": "Laya encoder (fine-tuned)"}


def dataset_dist():
    rows = []
    for fam in ("mlp", "conv", "attn"):
        for d, mach in MACHINE.items():
            p = f"{d}/dataset_{fam}.jsonl"
            if not os.path.exists(p):
                continue
            recs = [json.loads(l) for l in open(p)]
            c = Counter(r["label"] for r in recs)
            taus = sorted(r.get("tau_pct", 0) for r in recs)
            med_tau = taus[len(taus) // 2] if taus else 0
            rows.append(f"{FAMILY[fam]} & {mach} & {len(recs)} & {c.get('harmful',0)} & {c.get('helpful',0)} & {c.get('neutral',0)} & {med_tau:.1f}\\\\")
    write("dataset_dist.tex", table_env(rows, "llrrrrr", "family & machine & rows & harmful & helpful & neutral & median $\\tau$ (\\%)",
                                        "The firing dataset: one row per sweep configuration, label from two confirmed on/off comparisons.", "tab:dataset"))


def guard_cv(path: str, tag: str):
    d = json.load(open(path))
    rows = []
    last = None
    for r in d["results"]:
        fam = FAMILY[r["family"]]
        if last and last != r["family"]:
            rows.append("\\midrule")
        last = r["family"]
        name = MODEL.get(r["model"].split(" (")[0], r["model"])
        if "refusals" in r["model"]:
            name += " (" + r["model"].split("(")[1]
        rows.append(f"{fam if rows == [] or rows[-1] == '\\midrule' else ''} & {tex(name)} & {r['accuracy']:.2f} & {r['recall']:.2f} & {r['false_alarm_rate']:.2f} & {r['cost_weighted']:.2f} & {r['default_cost_weighted']:.1f}\\\\")
    write(f"guard_cv_{tag}.tex", table_env(rows, "llrrrrr",
          "family & guard & accuracy & harmful caught & false alarms & cost \\%/firing & always-fire cost",
          f"Guard quality under grouped cross-validation ({tag.replace('_', ' ')}). Cost = mean \\% of runtime lost per firing by following the guard; always-fire is Inductor's current behaviour.",
          f"tab:guardcv-{tag}", size="footnotesize"))


def e2e(path: str, tag: str):
    d = json.load(open(path))
    recs = d["rows"] if isinstance(d, dict) else d
    labels = []
    for r in recs:
        for s in r.get("steps", []):
            if s.get("guards") and s["guards"] not in labels:
                labels.append(s["guards"])
    rows, summary = [], {}
    for lab in labels:
        summary[lab] = {"log_sum": 0.0, "n": 0, "worst": 0.0, "worst_model": "", "faster": 0, "slower": 0, "blocked": 0}
    for r in recs:
        cells = []
        for lab in labels:
            s = next((s for s in r["steps"] if s.get("guards") == lab), None)
            if not s or "delta_pct" not in s:
                cells.append("--")
                continue
            mark = {"faster": "$\\downarrow$", "slower": "$\\uparrow$"}.get(s["verdict"], "")
            cells.append(f"{s['delta_pct']:+.1f}\\% {mark}" + (f" ({s['blocked']} blocked)" if s.get("blocked") else ""))
            sm = summary[lab]
            sm["log_sum"] += math.log(max(s["median_ms"], 1e-9) / r["default_ms"]); sm["n"] += 1
            sm["faster"] += s["verdict"] == "faster"; sm["slower"] += s["verdict"] == "slower"; sm["blocked"] += s.get("blocked", 0)
            if s["verdict"] == "slower" and s["delta_pct"] > sm["worst"]:
                sm["worst"], sm["worst_model"] = s["delta_pct"], r["model"]
        rows.append(f"\\code{{{tex(r['model'])}}} & {r['default_ms']:.2f} & {100*r['tau_ms']/r['default_ms']:.1f} & " + " & ".join(cells) + "\\\\")
    rows.append("\\midrule")
    rows.append("geometric mean & & & " + " & ".join(f"{100*(math.exp(s['log_sum']/s['n'])-1):+.1f}\\%" if s["n"] else "--" for s in summary.values()) + "\\\\")
    rows.append("faster / slower (confirmed) & & & " + " & ".join(f"{s['faster']} / {s['slower']}" for s in summary.values()) + "\\\\")
    header = "model & stock ms & $\\tau$ \\% & " + " & ".join("guards: " + tex(l) for l in labels)
    write(f"e2e_{tag}.tex", table_env(rows, "lrr" + "l" * len(labels), header,
          f"End-to-end effect of adding the guards one family at a time ({tag.replace('_', ' ')}); change versus stock Inductor, arrows mark confirmed verdicts beyond $\\tau$.",
          f"tab:e2e-{tag}", size="footnotesize"))
    for s in summary.values():
        s["geomean_pct"] = 100 * (math.exp(s["log_sum"] / s["n"]) - 1) if s["n"] else None
        s.pop("log_sum")
    return summary


if __name__ == "__main__":
    dataset_dist()
    for p in sorted(glob.glob("results/guards_eval*.json")) + sorted(glob.glob("results/vm/guards_eval*.json")):
        tag = os.path.basename(p)[len("guards_eval"):-5].strip("_") or "mac"
        if p.startswith("results/vm"):
            tag = "vm_" + tag if tag else "vm"
        if "_e2e" in p:
            continue
        guard_cv(p, tag)
    heads = {}
    for p in sorted(glob.glob("results/guards_eval_e2e*.json")) + sorted(glob.glob("results/vm/guards_eval_e2e*.json")):
        tag = ("vm_" if p.startswith("results/vm") else "mac_") + (os.path.basename(p)[len("guards_eval_e2e"):-5].strip("_") or "all")
        heads[tag] = e2e(p, tag)
    json.dump(heads, open(os.path.join(OUT, "guard_summary.json"), "w"), indent=1)
    print(json.dumps(heads, indent=1))
