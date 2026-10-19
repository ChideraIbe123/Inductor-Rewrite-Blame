"""Why does a rule help on one machine and hurt on another? Join each machine's confirmed sweep verdicts with the
code-level change the same switch makes (kernel count, external library calls, static allocation), and summarise
per (switch, machine, sign). Sources: verify_*.json where present, else the pod store's compile-only rows.
Usage: python scripts/mechanism.py --machine "Mac M4=results" --machine "Xeon=results/vm" --machine "4090 bf16=results/gpu/rtx4090:results/gpu/measurements_1mvamzh7yxhrqn.sqlite" ... [--out docs/mechanism.md]"""
import argparse, glob, json, os, sqlite3, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--machine", action="append", required=True, help='"label=dir[:store.sqlite]" (dir holds sweep_*.json and maybe verify_*.json)')
ap.add_argument("--switches", nargs="*", default=["lowering/layout_optimization", "lowering/freezing", "sched/max_fusion_size", "optimus/decompose_mm_pass",
                                                  "master/pattern_matcher", "master/post_grad_passes", "sched/inplace_buffers", "lowering/conv_1x1_as_mm"])
ap.add_argument("--min-pct", type=float, default=3.0); ap.add_argument("--out", default=None)
a = ap.parse_args()

def code_deltas_from_verify(d):
    """{switch: (dkernels, dextern, dalloc, names_base, names_toggled)} with base - toggled orientation = effect of turning the switch OFF (default-on) / ON (default-off)."""
    out = {}
    eff = d["effects"]; items = list(eff.items()) if isinstance(eff, dict) else [(e.get("switch"), e) for e in eff]
    for sid, v in items:
        if not isinstance(v, dict) or not v.get("kernel_count"):
            continue
        kb, kt = v["kernel_count"]; eb, et = v.get("extern_call_count", [None, None]); ab, at = v.get("alloc_bytes", [None, None])
        out[sid] = {"dk": kt - kb, "de": (et - eb) if eb is not None and et is not None else None, "da": (at - ab) if ab is not None and at is not None else None,
                    "only_base": v.get("kernels_only_in_base", [])[:3], "only_toggled": v.get("kernels_only_in_toggled", v.get("kernels_only_in_new", []))[:3]}
    return out

def code_deltas_from_store(path):
    con = sqlite3.connect(path); base = {}; tog = defaultdict(dict)
    for model, proto, payload in con.execute("SELECT model, protocol, payload FROM measurements WHERE protocol = 'compile_only'"):
        d = json.loads(payload); diff = d.get("diff") or {}; code = d.get("code") or {}
        if not code: continue
        on, off = diff.get("on_extra", []), diff.get("off_defaults", [])
        if not on and not off: base[model] = code
        elif len(on) + len(off) == 1: tog[model][(on or off)[0]] = code
    out = defaultdict(dict)
    for model, sws in tog.items():
        b = base.get(model)
        if not b: continue
        for sid, c in sws.items():
            out[model][sid] = {"dk": (c.get("kernel_count") or 0) - (b.get("kernel_count") or 0),
                               "de": (c.get("extern_call_count") or 0) - (b.get("extern_call_count") or 0),
                               "da": (c.get("alloc_bytes") or 0) - (b.get("alloc_bytes") or 0), "only_base": [], "only_toggled": []}
    return out

report = defaultdict(lambda: defaultdict(list))   # (switch, machine) -> sign -> list of dicts
for spec in a.machine:
    label, rest = spec.split("=", 1); d, _, store = rest.partition(":")
    deltas = code_deltas_from_store(store) if store else {}
    for f in sorted(glob.glob(os.path.join(d, "sweep_*.json")) + glob.glob(os.path.join(d, "sweep2_*.json"))):
        model = os.path.basename(f)[:-5].split("_", 1)[1]
        sw = json.load(open(f)); rows = sw.get("rows", [])
        vf = os.path.join(d, f"verify_{model}.json")
        dl = code_deltas_from_verify(json.load(open(vf))) if os.path.exists(vf) and not store else deltas.get(model, {})
        for r in rows:
            sid = r["switch"]
            if sid not in a.switches or not (r["slower"] or r["faster"]) or abs(r["delta_pct"]) < a.min_pct:
                continue
            on_hurts = (r["direction"] == "off" and r["faster"]) or (r["direction"] == "on" and r["slower"])
            cd = dl.get(sid)
            # orient code deltas as "ON minus OFF"
            sign = 1 if r["direction"] == "on" else -1
            rec = {"model": model, "pct": abs(r["delta_pct"]), "dk": sign * cd["dk"] if cd else None, "de": sign * cd["de"] if cd and cd["de"] is not None else None,
                   "da": sign * cd["da"] if cd and cd["da"] is not None else None, "names": (cd["only_base"] if (cd and r["direction"] == "off") else (cd["only_toggled"] if cd else []))}
            report[(sid, label)]["ON hurts" if on_hurts else "ON helps"].append(rec)

def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):+.0f}" if xs else "·"
lines = ["| switch | machine | ON helps: n, median %, Δkernels, Δextern calls, Δalloc MB | ON hurts: n, median %, Δkernels, Δextern calls, Δalloc MB |", "|---|---|---|---|"]
for sid in a.switches:
    for spec in a.machine:
        label = spec.split("=", 1)[0]
        cell = report.get((sid, label))
        if not cell: continue
        def fmt(k):
            rs = cell.get(k, [])
            if not rs: return "·"
            return f"{len(rs)}, {statistics.median(r['pct'] for r in rs):.0f}%, {med([r['dk'] for r in rs])}, {med([r['de'] for r in rs])}, {med([(r['da'] or 0)/2**20 if r['da'] is not None else None for r in rs])}"
        lines.append(f"| `{sid.split('/')[-1]}` | {label} | {fmt('ON helps')} | {fmt('ON hurts')} |")
md = "\n".join(lines); print(md)
# kernel-name evidence for the layout rule: what kinds of kernels appear when it is ON
print("\nKernel names that exist only with the switch ON (layout rule), by machine:")
for spec in a.machine:
    label = spec.split("=", 1)[0]
    names = [n for sgn in ("ON helps", "ON hurts") for r in report.get(("lowering/layout_optimization", label), {}).get(sgn, []) for n in r["names"]]
    if names: print(f"  {label}: {sorted(set(n[:70] for n in names))[:4]}")
if a.out:
    open(a.out, "w").write(md + "\n")
