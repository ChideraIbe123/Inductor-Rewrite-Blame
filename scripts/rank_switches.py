"""Rank switches by confirmed effect across a directory of sweep_*.json files.
Usage: python scripts/rank_switches.py <results_dir> [--top K] [--min-pct P] [--exclude-master]
Prints one switch id per line (most impactful first): the sum over models of confirmed |delta%| >= P."""
import argparse, glob, json, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("results_dir"); ap.add_argument("--top", type=int, default=5); ap.add_argument("--min-pct", type=float, default=3.0)
ap.add_argument("--exclude-master", action="store_true"); ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()
score = defaultdict(float); hits = defaultdict(int); sign = defaultdict(int)
for f in sorted(glob.glob(os.path.join(a.results_dir, "sweep_*.json"))):
    d = json.load(open(f)); rows = d.get("rows", d if isinstance(d, list) else [])
    for r in rows:
        if not (r.get("slower") or r.get("faster")) or abs(r.get("delta_pct", 0)) < a.min_pct:
            continue
        sid = r["switch"]
        if a.exclude_master and sid.startswith("master/"):
            continue
        score[sid] += abs(r["delta_pct"]); hits[sid] += 1; sign[sid] += 1 if r.get("slower") else -1
ranked = sorted(score, key=lambda s: -score[s])
for sid in ranked[: a.top]:
    if a.verbose:
        print(f"{sid}\t{score[sid]:.1f}\t{hits[sid]} models\tnet sign {sign[sid]:+d}")
    else:
        print(sid)
