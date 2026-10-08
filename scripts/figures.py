"""Build the report figures from results/*.json. Usage: python scripts/figures.py [results_dir] [out_dir]"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "report/midterm/figures"
os.makedirs(OUT, exist_ok=True)


def load(pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        try:
            out[p] = json.load(open(p))
        except Exception as e:
            print("skip", p, e)
    return out


def fig_noise():
    files = load("noise_*.json")
    if not files:
        return
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(files) + 1.5))
    labels, data, taus = [], [], []
    for p, d in files.items():
        n = d["noise"]
        labels.append(d["model"]); data.append([100 * (m / n["center"] - 1) for m in n["medians"]]); taus.append(100 * n["tau"] / n["center"])
    y = range(len(labels))
    for i, (xs, t) in enumerate(zip(data, taus)):
        ax.scatter(xs, [i] * len(xs), s=18, zorder=3)
        ax.plot([-t, t], [i, i], color="0.6", lw=6, alpha=0.4, zorder=1)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("per-run median, % deviation from the centre (band = threshold tau)")
    ax.set_title("Baseline noise: repeated identical runs in separate processes")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "noise.pdf")); plt.close(fig)


def fig_sweeps():
    files = load("sweep_*.json")
    for p, d in files.items():
        rows = [r for r in d["rows"] if "error" not in r]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["delta_pct"])
        fig, ax = plt.subplots(figsize=(8, 0.28 * len(rows) + 1.5))
        names = [r["switch"].split("/")[-1][:40] + (" (on)" if r["direction"] == "on" else " (off)") for r in rows]
        vals = [r["delta_pct"] for r in rows]
        colors = ["#c0392b" if r["slower"] else ("#27ae60" if r["faster"] else "0.6") for r in rows]
        ax.barh(range(len(rows)), vals, color=colors)
        tau_pct = 100 * d["noise"]["tau"] / d["reference_ms"]
        ax.axvspan(-tau_pct, tau_pct, color="0.85", zorder=0)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("% change in median time vs. default (grey band = noise threshold)")
        ax.set_title(f"{d['model']}: toggling one switch at a time ({d['env'].split('|')[0]})")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"sweep_{d['model']}.pdf")); plt.close(fig)


def fig_attribution_trace():
    files = load("attribution_*.json")
    for p, d in files.items():
        tr = d["result"]["trace"]
        if not tr:
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        xs = range(1, len(tr) + 1)
        sizes = [len(t["candidate"]) for t in tr]
        cols = ["#c0392b" if t["verdict"] == "slow" else "#2980b9" for t in tr]
        ax.scatter(xs, sizes, c=cols, s=25)
        ax.set_xlabel("judge evaluation"); ax.set_ylabel("|changes tried|")
        ax.set_title(f"ddmin trace, {d['model']} (red = slow, blue = fast); culprits: {len(d['result']['culprits'])}")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"trace_{d['model']}.pdf")); plt.close(fig)


fig_noise(); fig_sweeps(); fig_attribution_trace()
print("figures written to", OUT)
