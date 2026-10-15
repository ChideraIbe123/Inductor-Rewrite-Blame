"""Does 'take the min' beat 'take the median'? Re-run the null calibration and the sweep verdicts on the
stored per-call samples with each statistic. Usage: python scripts/noise_study_min.py"""
import json, sqlite3, numpy as np
from collections import defaultdict

STORES = {"mac": "results/measurements.sqlite", "vm": "results/vm/measurements.sqlite", "gpu": "results/gpu/measurements.sqlite"}
STATS = {"median": np.median, "min": np.min, "p10": lambda s: np.percentile(s, 10), "mean": np.mean,
         "median_of_last_half": lambda s: np.median(s[len(s)//2:])}

def session_of(d):
    pk = d.get("protocol_key") or d.get("protocol", {}).get("key") or ""
    return d.get("session") or (pk.split("@", 1)[1].split("/")[0].split("+")[0] if "@" in pk else "old")

def load(path):
    con = sqlite3.connect(path); out = []
    for (proto, payload) in con.execute("SELECT protocol, payload FROM measurements"):
        d = json.loads(payload)
        if d.get("timing") and d["timing"].get("samples") and not d.get("error"):
            sess = proto.split("@", 1)[1].split("/")[0].split("+")[0] if "@" in proto else "old"
            d["_session"] = sess; d["_rep"] = "/rep" in proto; d["_guards"] = "+guards" in proto
            out.append(d)
    return out

def tau_rule(ref_vals):
    c = np.median(ref_vals); mad = np.median(np.abs(np.array(ref_vals) - c))
    return c, max(3 * mad, 0.01 * c)

print(f"{'machine':<6} {'stat':<20} {'groups':>6} {'median CV%':>10} {'false alarms (1 run)':>21} {'false alarms (AND 2)':>21}")
for mach, path in STORES.items():
    try: rows = load(path)
    except Exception as e: print(mach, "no store", e); continue
    groups = defaultdict(list)
    for d in rows:
        if d["_rep"] and not d["_guards"]:
            groups[(d["model"], d["_session"], d["state_hash"])].append(np.array(d["timing"]["samples"]))
    groups = {k: v for k, v in groups.items() if len(v) >= 7}
    # within-session single-run comparisons between the null groups: hold one run out as the "candidate"
    for name, f in STATS.items():
        cvs, fa1, n1, fa2, n2 = [], 0, 0, 0, 0
        for k, runs in groups.items():
            vals = [float(f(s)) for s in runs]
            cvs.append(100 * np.std(vals) / np.mean(vals))
            for i in range(len(vals)):
                ref = vals[:i] + vals[i + 1:]
                c, tau = tau_rule(ref[:7])
                n1 += 1; fa1 += vals[i] - c > tau
            if len(vals) >= 9:
                for i in range(len(vals) - 1):
                    ref = vals[:i] + vals[i + 2:]
                    c, tau = tau_rule(ref[:7])
                    n2 += 1; fa2 += (vals[i] - c > tau) and (vals[i + 1] - c > tau)
        print(f"{mach:<6} {name:<20} {len(groups):>6} {np.median(cvs):>10.2f} {100*fa1/max(n1,1):>19.1f}% {100*fa2/max(n2,1):>19.1f}%")
    # sweep verdicts: do median and min agree on which single-switch rows are beyond tau?
    by = defaultdict(dict)
    for d in rows:
        if d["_guards"]: continue
        by[(d["model"], d["_session"])].setdefault(d["state_hash"], []).append(d)
    agree = disagree = 0; flips = []
    for (model, sess), states in by.items():
        base = [d for h, ds in states.items() for d in ds if d["_rep"]]
        if len(base) < 7: continue
        base_hash = base[0]["state_hash"]
        for h, ds in states.items():
            if h == base_hash or len(ds) == 0: continue
            d = ds[0]
            df = d.get("diff") or {}
            if len(df.get("on_extra", [])) + len(df.get("off_defaults", [])) != 1: continue
            vmed = {}; 
            for name in ("median", "min"):
                f = STATS[name]; refv = [float(f(np.array(b["timing"]["samples"]))) for b in base[:7]]
                c, tau = tau_rule(refv); v = float(f(np.array(d["timing"]["samples"])))
                vmed[name] = "slower" if v - c > tau else "faster" if c - v > tau else "same"
            if vmed["median"] == vmed["min"]: agree += 1
            else: disagree += 1; flips.append((model, sess, vmed["median"], vmed["min"]))
    print(f"{mach:<6} single-switch rows: median and min agree on {agree}, disagree on {disagree}")
    from collections import Counter
    print("      disagreement kinds (median->min):", Counter((a, b) for _, _, a, b in flips).most_common(4))
