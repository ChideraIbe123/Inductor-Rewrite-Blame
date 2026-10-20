"""Paper figures (matplotlib, PDF): F1 sign-matrix heatmap, F2 attribution cost (judge evaluations vs number of
candidate changes) with the null-calibration false-alarm bars, F3 guard gain vs oracle gain, F4 freezing gain
distribution per GPU. Usage: python scripts/figures_paper.py [out_dir=report/final/figures]"""
import glob, json, os, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "report/final/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "pdf.fonttype": 42})

# ---------- F1: sign matrix heatmap ----------
sm = json.load(open("report/final/tables/sign_matrix.json"))
machines = sm["machines"]
cells = {tuple(k.split("|", 1)): v for k, v in sm["cells"].items()}
switches = sorted({s for s, _ in cells}, key=lambda s: -sum(cells.get((s, m), {"hurts": 0, "helps": 0})["hurts"] + cells.get((s, m), {"hurts": 0, "helps": 0})["helps"] for m in machines))[:14]
M = np.full((len(switches), len(machines)), np.nan)
for i, s in enumerate(switches):
    for j, m in enumerate(machines):
        c = cells.get((s, m))
        if c and (c["hurts"] or c["helps"]):
            M[i, j] = (c["helps"] - c["hurts"]) / (c["helps"] + c["hurts"])   # +1 = ON helps on every model, -1 = hurts
fig, ax = plt.subplots(figsize=(3.6, 3.4))
im = ax.imshow(M, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(machines))); ax.set_xticklabels(machines, rotation=35, ha="right")
ax.set_yticks(range(len(switches))); ax.set_yticklabels([s.split("/")[-1][:28] for s in switches])
for i, s in enumerate(switches):
    for j, m in enumerate(machines):
        c = cells.get((s, m))
        if c and (c["hurts"] or c["helps"]):
            ax.text(j, i, f"{c['helps']}/{c['hurts']}", ha="center", va="center", fontsize=6, color="black")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("ON helps  ←  →  ON hurts", fontsize=7); cb.set_ticks([-1, 0, 1]); cb.set_ticklabels(["hurts all", "split", "helps all"])
ax.set_title("Confirmed effect of each switch being ON (helps/hurts model counts)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sign_matrix.pdf")); plt.close(fig); print("sign_matrix.pdf")

# ---------- F2: attribution cost + null calibration ----------
pts = []
for f in glob.glob("results/attribution*.json") + glob.glob("results/vm/attribution*.json") + glob.glob("results/gpu/attribution*.json"):
    d = json.load(open(f)); r = d.get("result", d)
    n = r.get("n_candidates") or d.get("n_candidates") or len(r.get("changes", []) or []) or None
    k = r.get("judge_calls") or d.get("judge_calls")
    if not n:
        trace = r.get("trace") or d.get("trace") or []
        n = max((len(t.get("changes", [])) for t in trace), default=None)
    if n and k:
        pts.append((n, k, bool(r.get("interaction"))))
fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.8))
ax = axes[0]
if pts:
    xs, ys, inter = zip(*pts)
    ax.scatter(xs, ys, s=12, c=["tab:red" if i else "tab:blue" for i in inter])
    g = np.linspace(1, max(xs), 50); ax.plot(g, 2 * np.log2(g) + 2, "k--", lw=0.7, label="2 log2 n + 2")
    ax.legend(fontsize=6)
ax.set_xlabel("candidate changes (n)"); ax.set_ylabel("judge evaluations"); ax.set_title("Attribution cost")
ax = axes[1]
rules = ["single run\n(MAD k=3)", "+ steady trim", "Wasserstein", "3 SD", "AND of 2\n(adopted)", "AND of 3"]
fa = [8.9, 9.4, 6.1, 6.7, 1.5, 0.9]      # from docs/noise_method.md (480 null runs)
ax.bar(range(len(rules)), fa, color=["tab:gray"] * 4 + ["tab:green", "tab:green"])
ax.set_xticks(range(len(rules))); ax.set_xticklabels(rules, fontsize=5.5); ax.set_ylabel("false alarms (%)"); ax.set_title("Null calibration (480 runs)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "attribution_cost.pdf")); plt.close(fig); print("attribution_cost.pdf", len(pts), "cases")

# ---------- F3: guard gain vs oracle gain (where oracle columns exist) ----------
rows = []
for f in glob.glob("results/**/guards_eval_e2e*.json", recursive=True):
    d = json.load(open(f))
    for r in d.get("rows", []):
        o = r.get("oracle", {}).get("best_delta_pct")
        if o is None or not r.get("steps"): continue
        st = r["steps"][-1]
        if "delta_pct" in st: rows.append((o, st["delta_pct"], r["model"], os.path.basename(f)))
fig, ax = plt.subplots(figsize=(1.8, 1.8))
if rows:
    ax.scatter([-o for o, _, _, _ in rows], [-g for _, g, _, _ in rows], s=10)
    lim = max(1, max(-o for o, _, _, _ in rows)); ax.plot([0, lim], [0, lim], "k--", lw=0.7)
ax.set_xlabel("oracle gain (%)"); ax.set_ylabel("guard gain (%)"); ax.set_title("Guards vs per-model oracle")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "guard_vs_oracle.pdf")); plt.close(fig); print("guard_vs_oracle.pdf", len(rows), "points")

# ---------- F4: freezing gain distribution per GPU ----------
fig, ax = plt.subplots(figsize=(3.6, 1.7))
for G, lab in (("rtx4090", "RTX 4090"), ("a40", "A40"), ("l4", "L4")):
    p = f"results/gpu/{G}/dataset_switch_lowering_freezing.jsonl"
    if not os.path.exists(p): continue
    rs = [json.loads(l) for l in open(p)]
    gains = sorted(100 * (1 - r["ratio"]) for r in rs if r["label"] != "no_effect")
    ax.plot(gains, np.linspace(0, 1, len(gains)), label=f"{lab} (n={len(gains)})")
ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("speed-up with freezing ON (%)"); ax.set_ylabel("fraction of networks"); ax.legend(fontsize=6); ax.set_title("freezing over 163 real networks per GPU (bf16)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "freezing_cdf.pdf")); plt.close(fig); print("freezing_cdf.pdf")
