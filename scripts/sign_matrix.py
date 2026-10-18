"""Rule x machine sign matrix from sweep tables.

For every single-switch sweep row with a confirmed verdict, the effect of the switch being ON (relative to OFF)
is "hurts" or "helps", whichever direction the row was measured in. Per (switch, machine) we report the number
of models where ON hurts / helps and the largest confirmed effect.
Usage: python scripts/sign_matrix.py --machine "Mac M4 Pro=results/sweep_*.json" --machine "Xeon=results/vm/sweep_*.json" ...
       [--min-pct 3] [--top 25] [--out report/final/tables/sign_matrix]"""
import argparse, glob, json, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--machine", action="append", required=True, help='"label=glob" of sweep json files')
ap.add_argument("--min-pct", type=float, default=3.0); ap.add_argument("--top", type=int, default=25)
ap.add_argument("--out", default=None)
a = ap.parse_args()

cells = defaultdict(lambda: {"hurts": 0, "helps": 0, "max": 0.0})
machines = []
per_machine_models = defaultdict(set)
for spec in a.machine:
    label, pat = spec.split("=", 1); machines.append(label)
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f)); rows = d.get("rows", d if isinstance(d, list) else [])
        model = os.path.basename(f)[:-5].split("sweep2_")[-1].split("sweep_")[-1]
        per_machine_models[label].add(model)
        for r in rows:
            if not (r.get("slower") or r.get("faster")) or abs(r.get("delta_pct", 0)) < a.min_pct:
                continue
            on_hurts = (r["direction"] == "off" and r["faster"]) or (r["direction"] == "on" and r["slower"])
            c = cells[(r["switch"], label)]
            c["hurts" if on_hurts else "helps"] += 1
            c["max"] = max(c["max"], abs(r["delta_pct"]))
score = defaultdict(float)
for (sid, lab), c in cells.items():
    score[sid] += c["hurts"] + c["helps"]
top = sorted(score, key=lambda s: -score[s])[: a.top]

def cell(sid, lab):
    c = cells.get((sid, lab))
    if not c:
        return "·"
    parts = ([f"hurts {c['hurts']}"] if c["hurts"] else []) + ([f"helps {c['helps']}"] if c["helps"] else [])
    return ", ".join(parts) + f" ({c['max']:.0f}%)"

lines = ["| switch (ON vs OFF) | " + " | ".join(f"{m} ({len(per_machine_models[m])} models)" for m in machines) + " |", "|---|" + "---|" * len(machines)]
for sid in top:
    lines.append(f"| `{sid.split('/')[-1][:38]}` | " + " | ".join(cell(sid, m) for m in machines) + " |")
md = "\n".join(lines)
print(md)
flips = []
for sid in top:
    signs = {m: ("hurts" if cells[(sid, m)]["hurts"] > cells[(sid, m)]["helps"] else "helps") for m in machines if (sid, m) in cells}
    if len(set(signs.values())) > 1:
        flips.append((sid, signs))
print(f"\nswitches whose dominant sign differs between machines: {len(flips)} of {len(top)}")
for sid, signs in flips:
    print("  ", sid.split("/")[-1], signs)
if a.out:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out + ".md", "w").write(md + "\n")
    json.dump({"machines": machines, "cells": {f"{s}|{m}": c for (s, m), c in cells.items()}, "flips": flips}, open(a.out + ".json", "w"), indent=1)
