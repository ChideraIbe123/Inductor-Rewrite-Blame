import json, sqlite3, numpy as np, itertools
from collections import defaultdict
from scipy import stats
rng = np.random.default_rng(0)
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

# ---------- D. steady-state / knee inside the timed window ----------
def knee(s, w=5):
    """first index k such that median(s[k:k+w]) <= median(last half) + 1.0*IQR(last half)"""
    tail = s[len(s)//2:]
    m = np.median(tail); iqr = np.subtract(*np.percentile(tail, [75, 25]))
    for k in range(0, len(s) - w + 1):
        if np.median(s[k:k+w]) <= m + max(iqr, 1e-9):
            return k
    return len(s) - w
def kneedle(s):
    """kneedle on the cumulative mean curve (normalized), returns the elbow index"""
    cm = np.cumsum(s) / np.arange(1, len(s)+1)
    x = np.linspace(0, 1, len(s)); y = (cm - cm.min()) / (cm.max() - cm.min() + 1e-12)
    d = (y[0] + (y[-1]-y[0]) * x) - y   # distance below the chord (decreasing curve)
    return int(np.argmax(d))
allk, allkd = [], []
print("== D. knee (steady-state start) within the 50 timed samples, per model/machine: frac k>0, frac k>=10, median k, kneedle median, median change if trimmed")
per = defaultdict(list)
for (model, mach, sess, sh), runs in groups.items():
    for s in runs:
        k = knee(s); kd = kneedle(s)
        trimmed = np.median(s[k:]) / np.median(s)
        per[(model, mach)].append((k, kd, trimmed))
for key, v in sorted(per.items()):
    ks = np.array([a for a,_,_ in v]); kds = np.array([b for _,b,_ in v]); tr = np.array([c for _,_,c in v])
    print(f"  {key[0]:28s} {key[1]:6s} n={len(v):3d} k>0: {np.mean(ks>0):.2f}  k>=10: {np.mean(ks>=10):.2f}  median k={np.median(ks):.0f}  kneedle={np.median(kds):.0f}  trimmed/raw median={np.median(tr):.4f} (min {tr.min():.3f})")

# ---------- C. null calibration: leave-one-out within identical-config groups ----------
def hl_shift(a, b):
    """Hodges-Lehmann estimate of median(a_i - b_j) (positive = a slower)"""
    d = a[:, None] - b[None, :]
    return np.median(d)
def methods(cand, refs, center_floor=0.01):
    """cand: samples of one run; refs: list of sample arrays. returns dict method -> (slow?: bool, stat, threshold)"""
    out = {}
    rm = np.array([np.median(r) for r in refs]); c = np.median(cand)
    center = np.median(rm)
    # 1 MAD on run medians
    mad = 1.4826 * np.median(np.abs(rm - center))
    for k in (2, 3, 4):
        tau = max(k * mad, center_floor * center)
        out[f"MAD k={k}"] = (c - center > tau, c - center, tau)
    # 2 IQR on run medians (sigma ~ IQR/1.349)
    q1, q3 = np.percentile(rm, [25, 75]); sig = (q3 - q1) / 1.349
    tau = max(3 * sig, center_floor * center); out["IQR k=3"] = (c - center > tau, c - center, tau)
    # 3 std (CV) on run medians
    tau = max(3 * rm.std(ddof=1), center_floor * center); out["SD k=3"] = (c - center > tau, c - center, tau)
    # 4 Mann-Whitney on pooled samples + 1% HL floor
    pooled = np.concatenate(refs)
    p = stats.mannwhitneyu(cand, pooled, alternative="greater").pvalue
    shift = hl_shift(cand, pooled)
    out["MWU p<.01 & HL>1%"] = (p < 0.01 and shift > center_floor * center, shift, center_floor * center)
    out["MWU p<.001 & HL>2%"] = (p < 0.001 and shift > 2 * center_floor * center, shift, 2 * center_floor * center)
    # 5 hierarchical bootstrap of single-run medians under the null (resample runs, then samples)
    B = 400; boots = np.empty(B)
    for i in range(B):
        r = refs[rng.integers(len(refs))]
        boots[i] = np.median(rng.choice(r, size=len(r), replace=True))
    thr = np.percentile(boots, 99)
    out["HB p99 & 1% floor"] = (c > max(thr, center * (1 + center_floor)), c - center, max(thr, center*(1+center_floor)) - center)
    # 6 Wasserstein distance vs each ref, compared with ref-vs-ref null (95th pct), directional via HL shift
    dnull = [stats.wasserstein_distance(a, b) for a, b in itertools.combinations(refs, 2)]
    dcand = np.median([stats.wasserstein_distance(cand, r) for r in refs])
    out["W1 > p95(null) & HL>0"] = (dcand > np.percentile(dnull, 95) and shift > center_floor * center, dcand, np.percentile(dnull, 95))
    # 7 MAD k=3 after trimming to steady state (knee)
    tr = lambda s: s[knee(s):]
    rm2 = np.array([np.median(tr(r)) for r in refs]); c2 = np.median(tr(cand)); cen2 = np.median(rm2)
    mad2 = 1.4826 * np.median(np.abs(rm2 - cen2)); tau = max(3 * mad2, center_floor * cen2)
    out["MAD k=3 +knee-trim"] = (c2 - cen2 > tau, c2 - cen2, tau)
    # 8 MAD k=3 on run medians but candidate = mean of 2 runs (confirmation), emulated by averaging with another ref? (skip)
    return out

fp = defaultdict(lambda: [0, 0]); tau_rel = defaultdict(list)
for key, runs in groups.items():
    if len(runs) < 6: continue
    for i in range(len(runs)):
        cand = runs[i]; refs = runs[:i] + runs[i+1:]
        for name, (slow, stat, tau) in methods(cand, refs).items():
            fp[name][1] += 1; fp[name][0] += int(slow)
            tau_rel[name].append(tau / np.median([np.median(r) for r in refs]))
print("\n== C. null (leave-one-out over identical runs): false-positive rate per method, median relative threshold")
for name, (f, n) in sorted(fp.items(), key=lambda kv: kv[1][0]/kv[1][1]):
    print(f"  {name:24s} FP={f:3d}/{n} = {100*f/n:4.1f}%   median tau/center={100*np.median(tau_rel[name]):.2f}%  p90={100*np.percentile(tau_rel[name],90):.2f}%")

# ---------- TP: known effects: reference group vs a different-state group in same model/env/session ----------
print("\n== TP on measured effects (candidate = each run of state B vs reference = runs of state A, same model/env/session); effect = median shift")
# pick pairs of groups with >=6 refs and >=1 candidate runs
tp = defaultdict(lambda: defaultdict(lambda: [0, 0]))
effects = []
for (model, mach, sess, sh), refs in groups.items():
    if len(refs) < 6: continue
    center = np.median([np.median(r) for r in refs])
    for (m2, ma2, se2, sh2), cands in groups.items():
        if (m2, ma2, se2) != (model, mach, sess) or sh2 == sh: continue
        eff = np.median([np.median(c) for c in cands]) / center - 1
        if abs(eff) < 0.015: continue   # only clear effects as 'truth'
        bucket = "1.5-3%" if abs(eff) < 0.03 else ("3-6%" if abs(eff) < 0.06 else ">6%")
        if eff < 0: bucket = "faster " + bucket
        for c in cands:
            for name, (slow, _, _) in methods(c, refs).items():
                tp[bucket][name][1] += 1; tp[bucket][name][0] += int(slow)
for bucket in sorted(tp):
    print(f"  effect {bucket}:")
    for name, (h, n) in sorted(tp[bucket].items()):
        print(f"     {name:24s} flagged slow {h:3d}/{n}")
