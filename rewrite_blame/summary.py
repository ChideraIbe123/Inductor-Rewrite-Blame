"""Overview across all result files: one line per (model, machine) with the switches that matter."""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict


def _host(env: str) -> str:
    return "Mac" if "Apple" in env or "Mac" in env else ("VM" if "Intel" in env or "fa26" in env else env.split("|")[0])


def _short(sid: str) -> str:
    return sid.rsplit("/", 1)[-1]


def load_results(dirs=("results", "results/vm")) -> dict:
    out = defaultdict(list)
    for d in dirs:
        for kind in ("sweep", "attribution", "attribution_all", "interactions", "repeat", "tau_scan", "noise", "verify"):
            for f in sorted(glob.glob(os.path.join(d, f"{kind}_*.json"))):
                base = os.path.basename(f)
                if kind == "attribution" and base.startswith("attribution_all_"):
                    continue
                try:
                    out[kind].append(json.load(open(f)))
                except Exception:
                    pass
    return out


def summarize(results: dict) -> dict:
    """Return {"sweeps": [...], "attributions": [...], "interactions": [...]} rows for rendering."""
    sweeps = []
    for s in results.get("sweep", []):
        rows = [r for r in s["rows"] if "error" not in r]
        slower = [(r["switch"], r["direction"], r["delta_pct"]) for r in rows if r.get("slower")]
        faster = [(r["switch"], r["direction"], r["delta_pct"]) for r in rows if r.get("faster")]
        sweeps.append({"model": s["model"], "host": _host(s["env"]), "baseline_ms": s["reference_ms"],
                       "tau_pct": 100 * s["noise"]["tau"] / s["reference_ms"], "n_toggled": len(rows),
                       "slower": sorted(slower, key=lambda x: -x[2]), "faster": sorted(faster, key=lambda x: x[2])})
    attrs = []
    for a in results.get("attribution", []):
        r = a["result"]
        attrs.append({"model": a["model"], "host": _host(a["env"]), "kind": a["kind"], "judge": a["judge"],
                      "n_candidates": len(r["candidates"]), "evaluations": r["judge_calls"], "culprits": r["culprits"],
                      "delta_pct": (100 * (a["slow"]["timing"]["median"] / a["fast"]["timing"]["median"] - 1)) if a["judge"] == "timing" and a["fast"].get("timing") else None})
    for a in results.get("attribution_all", []):
        it = a["iterative"]
        attrs.append({"model": a["model"], "host": _host(a["env"]), "kind": "iterative", "judge": "timing",
                      "n_candidates": it["rounds"][0]["candidates"] if it["rounds"] else 0, "evaluations": it["judge_calls"],
                      "culprits": [c for g in a["groups"] for c in g["culprits"]],
                      "delta_pct": 100 * a["total_ms"] / a["fast_ms"] if a["fast_ms"] else None,
                      "explained_pct": 100 * a["explained_ms"] / a["fast_ms"] if a["fast_ms"] else None,
                      "groups": [(g["culprits"], g["delta_pct"]) for g in a["groups"]], "residual_ok": it["residual_explained"]})
    inter = []
    for i in results.get("interactions", []):
        inter.append({"model": i["model"], "host": _host(i["env"]), "n_switches": len(i["ids"]), "n_pairs": len(i["pairs"]),
                      "measurements": i["measurements"], "superadditive": [(r["a"], r["b"], r["interaction_ms"], r["delta_ms"]) for r in i["superadditive"][:5]],
                      "n_super": len(i["superadditive"]), "n_mask": len(i["masking"]), "base_ms": i["base_ms"],
                      "drift_events": i.get("drift_events", 0)})
    taus = [{"model": t["model"], "host": _host(t["env"]), "stable": t["stable"],
             "verdicts": [(row["k"], row["kind"], len(row["culprits"])) for row in t["rows"]]} for t in results.get("tau_scan", [])]
    reps = [{"model": r["model"], "agreement": r["agreement"], "n_baselines": len(r["baselines"])} for r in results.get("repeat", [])]
    return {"sweeps": sweeps, "attributions": attrs, "interactions": inter, "tau_scans": taus, "repeats": reps}


def render_summary(sm: dict) -> str:
    L = ["# Results overview", "", "## Leave-one-out sweeps", "",
         "| model | machine | baseline ms | tau % | toggled | slower (switch, toggled, %) | faster |", "|---|---|---|---|---|---|---|"]
    for s in sorted(sm["sweeps"], key=lambda s: (s["model"], s["host"])):
        sl = "; ".join(f"{_short(a)} {b} {c:+.1f}" for a, b, c in s["slower"][:4]) or "-"
        fa = "; ".join(f"{_short(a)} {b} {c:+.1f}" for a, b, c in s["faster"][:4]) or "-"
        L.append(f"| {s['model']} | {s['host']} | {s['baseline_ms']:.3f} | {s['tau_pct']:.1f} | {s['n_toggled']} | {sl} | {fa} |")
    L += ["", "## Attributions", "", "| model | machine | kind | slow vs fast % | candidates | judge evals | culprits |", "|---|---|---|---|---|---|---|"]
    for a in sorted(sm["attributions"], key=lambda a: (a["model"], a["host"], a["kind"])):
        d = f"{a['delta_pct']:+.1f}" if a.get("delta_pct") is not None else "-"
        cul = ", ".join(_short(c) for c in a["culprits"]) or "none"
        if a["kind"] == "iterative":
            cul = " | ".join(f"[{', '.join(_short(c) for c in g)}: {p:+.1f}%]" for g, p in a["groups"]) or "none"
            cul += "" if a.get("residual_ok") else " (residual not explained)"
        L.append(f"| {a['model']} | {a['host']} | {a['kind']} | {d} | {a['n_candidates']} | {a['evaluations']} | {cul} |")
    L += ["", "## Pairwise interactions", "", "| model | machine | switches | pairs | measurements | super-additive | masking | drift events | top pair (interaction ms) |", "|---|---|---|---|---|---|---|---|---|"]
    for i in sorted(sm["interactions"], key=lambda i: (i["model"], i["host"])):
        top = f"{_short(i['superadditive'][0][0])} + {_short(i['superadditive'][0][1])} ({i['superadditive'][0][2]:+.3f})" if i["superadditive"] else "-"
        L.append(f"| {i['model']} | {i['host']} | {i['n_switches']} | {i['n_pairs']} | {i['measurements']} | {i['n_super']} | {i['n_mask']} | {i['drift_events']} | {top} |")
    L += ["", "## Threshold sensitivity", "", "| model | machine | stable | verdict per k |", "|---|---|---|---|"]
    for t in sm["tau_scans"]:
        L.append(f"| {t['model']} | {t['host']} | {t['stable']} | " + "; ".join(f"k={k}: {kind} ({n})" for k, kind, n in t["verdicts"]) + " |")
    L += ["", "## Repeatability", "", "| model | baselines | same machine agree/disagree | across machines agree/disagree |", "|---|---|---|---|"]
    for r in sm["repeats"]:
        a = r["agreement"]
        L.append(f"| {r['model']} | {r['n_baselines']} | {a['same_machine']['agree']}/{a['same_machine']['disagree']} | {a['cross_machine']['agree']}/{a['cross_machine']['disagree']} |")
    return "\n".join(L) + "\n"
