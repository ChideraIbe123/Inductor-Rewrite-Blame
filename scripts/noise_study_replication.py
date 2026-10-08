import json, sqlite3, numpy as np, itertools
from collections import defaultdict
from scipy import stats
rng = np.random.default_rng(1)
rows = []
for path in ["results/measurements.sqlite", "results/vm/measurements.sqlite"]:
    con = sqlite3.connect(path)
    for (payload,) in con.execute("SELECT payload FROM measurements"):
        d = json.loads(payload)
        if d.get("timing") and d["timing"].get("samples") and not d.get("error"):
            rows.append(d)
def session_of(r):
    pk = r.get("protocol_key") or ""
    return r.get("session") or (pk.split("@",1)[1].split("/")[0] if "@" in pk else "old")
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["env"].split(" ")[0], session_of(r), r["state_hash"])].append(np.array(r["timing"]["samples"]))
big = {k: v for k, v in groups.items() if len(v) >= 6}

def knee(s, w=5):
    tail = s[len(s)//2:]; m = np.median(tail); iqr = np.subtract(*np.percentile(tail, [75, 25]))
    for k in range(0, len(s) - w + 1):
        if np.median(s[k:k+w]) <= m + max(iqr, 1e-9): return k
    return len(s) - w
def kneedle(s):
    cm = np.cumsum(s) / np.arange(1, len(s)+1); x = np.linspace(0, 1, len(s))
    y = (cm - cm.min()) / (cm.max() - cm.min() + 1e-12); return int(np.argmax((y[0] + (y[-1]-y[0]) * x) - y))
POLICIES = {"raw": lambda s: s, "drop10": lambda s: s[10:], "knee": lambda s: s[knee(s):], "kneedle": lambda s: s[kneedle(s):],
            "last25": lambda s: s[25:]}

# ---------- H. precision of each warm-up policy: between-run CV of run medians (lower = better) and FP (MAD k=3, m=1)
print("== H. warm-up policy vs precision: median over identical groups of (CV of run medians), and leave-one-out FP with MAD k=3")
for name, pol in POLICIES.items():
    cvs = []; f = n = 0
    for k, runs in big.items():
        rm = np.array([np.median(pol(s)) for s in runs]); cvs.append(rm.std(ddof=1) / rm.mean())
        for i in range(len(rm)):
            ref = np.delete(rm, i); c = np.median(ref); mad = 1.4826 * np.median(np.abs(ref - c)); tau = max(3*mad, 0.01*c)
            n += 1; f += int(rm[i] - c > tau)
    print(f"  {name:8s} median between-run CV={100*np.median(cvs):.2f}%  mean={100*np.mean(cvs):.2f}%   FP={f}/{n}={100*f/n:.1f}%")

# ---------- E. are FP candidates 'disturbed' processes? within-run signatures of FP vs non-FP runs
print("\n== E. within-run signature of false-positive runs (MAD k=3) vs others")
sig_fp, sig_ok = [], []
for k, runs in big.items():
    rm = np.array([np.median(s) for s in runs])
    for i, s in enumerate(runs):
        ref = np.delete(rm, i); c = np.median(ref); mad = 1.4826 * np.median(np.abs(ref - c)); tau = max(3*mad, 0.01*c)
        q1, q3 = np.percentile(s, [25, 75]); iqr_rel = (q3 - q1) / np.median(s)
        grp_iqr = np.median([(np.percentile(t,75)-np.percentile(t,25))/np.median(t) for j,t in enumerate(runs) if j != i])
        feat = dict(iqr_ratio=iqr_rel / max(grp_iqr, 1e-9), outl=np.mean(s > q3 + 1.5*(q3-q1)), knee=knee(s),
                    minratio=np.min(s)/np.median(np.delete(rm, i)), excess=(rm[i]-c)/c)
        (sig_fp if rm[i] - c > tau else sig_ok).append(feat)
for key in ("iqr_ratio", "outl", "knee", "minratio"):
    a = np.array([d[key] for d in sig_fp]); b = np.array([d[key] for d in sig_ok])
    print(f"  {key:10s} FP: median={np.median(a):.3f} p25={np.percentile(a,25):.3f}   others: median={np.median(b):.3f} p75={np.percentile(b,75):.3f}   n_fp={len(a)}")
print("  FP excess over centre (%):", np.round(100*np.sort([d['excess'] for d in sig_fp]), 1))
print("  'min of run' vs reference centre for FP runs (if a disturbed process, even its fastest call is slow):",
      np.round(np.sort([d['minratio'] for d in sig_fp]), 3))

# ---------- F/G. candidate replication and empirical nulls
print("\n== F/G. replicated candidates: FP (null, leave-m-out) and TP (known effects) for several decision rules")
def rules(cand_meds, ref_meds, boots_cache=None):
    c = np.median(ref_meds); mad = 1.4826 * np.median(np.abs(ref_meds - c)); m = len(cand_meds)
    cm = np.median(cand_meds) if m % 2 else np.mean(cand_meds)
    out = {}
    tau = max(3 * mad, 0.01 * c)
    out["MAD3 agg"] = cm - c > tau
    out["MAD3 agg, tau/sqrt(m)"] = cm - c > max(3 * mad / np.sqrt(m), 0.01 * c)
    out["MAD3 all-exceed"] = all(x - c > tau for x in cand_meds)
    # between-run bootstrap null: aggregate of m draws from the reference medians
    B = 2000; draws = rng.choice(ref_meds, size=(B, m), replace=True)
    agg = np.median(draws, axis=1) if m % 2 else draws.mean(axis=1)
    out["boot p99 (+1% floor)"] = cm - c > max(np.percentile(agg - c, 99), 0.01 * c)
    out["boot p95 (+1% floor)"] = cm - c > max(np.percentile(agg - c, 95), 0.01 * c)
    # rank test on run medians (exact small-sample), one-sided
    if m >= 2:
        p = stats.mannwhitneyu(cand_meds, ref_meds, alternative="greater").pvalue
        out["MWU on run medians p<.05 & 1%"] = (p < 0.05) and (cm - c > 0.01 * c)
    return out
for m in (1, 2, 3):
    fp = defaultdict(lambda: [0, 0]); tp = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for k, runs in big.items():
        rm = np.array([np.median(s) for s in runs])
        for idx in itertools.combinations(range(len(rm)), m):
            cand = rm[list(idx)]; ref = np.delete(rm, list(idx))
            if len(ref) < 5: continue
            for name, slow in rules(cand, ref).items():
                fp[name][1] += 1; fp[name][0] += int(slow)
    # TP: candidate = m runs of a different state in the same model/env/session
    for (model, mach, sess, sh), refs in big.items():
        ref = np.array([np.median(s) for s in refs]); c = np.median(ref)
        for (m2, ma2, se2, sh2), cands in groups.items():
            if (m2, ma2, se2) != (model, mach, sess) or sh2 == sh or len(cands) < m: continue
            cm_all = np.array([np.median(s) for s in cands]); eff = np.median(cm_all) / c - 1
            if abs(eff) < 0.015: continue
            bucket = ("faster " if eff < 0 else "") + ("1.5-3%" if abs(eff) < 0.03 else "3-6%" if abs(eff) < 0.06 else ">6%")
            for idx in itertools.combinations(range(len(cm_all)), m):
                for name, slow in rules(cm_all[list(idx)], ref).items():
                    tp[bucket][name][1] += 1; tp[bucket][name][0] += int(slow)
    print(f"\n  --- m={m} candidate run(s) ---")
    for name in fp:
        f, n = fp[name]
        parts = []
        for b in ("1.5-3%", "3-6%", ">6%", "faster >6%"):
            h, nn = tp[b].get(name, [0, 0]); parts.append(f"{b}: {h}/{nn}")
        print(f"   {name:28s} FP={100*f/n:4.1f}% ({f}/{n})   TP " + "  ".join(parts))
