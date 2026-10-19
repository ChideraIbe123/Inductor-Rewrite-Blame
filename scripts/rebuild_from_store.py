"""Rebuild per-machine sweep tables and switch datasets from a pod's measurement store.

The stores are the raw evidence (one row per measured configuration, with the 50 per-call samples, generated
program hash, fired rules and protocol). Summary files can always be regenerated from them; this script does so
for a pod whose summary files were lost or overwritten.

Usage: python scripts/rebuild_from_store.py <store.sqlite> <label> [--registry results/gpu/switches.json] [--out results/gpu/<label>]
Writes <out>/noise_<model>.json, <out>/sweep_<model>.json (rank_switches.py-compatible rows) and
<out>/dataset_switch_<slug>.jsonl (rows with the whole-model profile computed locally)."""
from __future__ import annotations

import argparse, json, os, re, sqlite3, sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())
from rewrite_blame.stats import noise_model          # noqa: E402
from rewrite_blame.dataset import label_pair          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("store"); ap.add_argument("label")
ap.add_argument("--registry", default="results/gpu/switches.json"); ap.add_argument("--out", default=None)
ap.add_argument("--noise-config", default="zoo_resnet34_b32"); ap.add_argument("--no-profiles", action="store_true")
a = ap.parse_args()
out = a.out or os.path.join("results", "gpu", a.label); os.makedirs(out, exist_ok=True)

reg = json.load(open(a.registry)); sw = reg["switches"] if isinstance(reg, dict) else reg
default_on = {s["id"]: bool(s.get("default_on", True)) for s in sw}

con = sqlite3.connect(a.store)
rows = []
for model, env, proto, payload in con.execute("SELECT model, env, protocol, payload FROM measurements"):
    d = json.loads(payload)
    if d.get("error") or not d.get("timing"):
        continue
    m = re.match(r"^(?P<p>[^@]+)@(?P<sess>[^/+]+)(?:/rep(?P<rep>\d+))?(?P<rest>.*)$", proto)
    if not m:
        continue
    diff = d.get("diff") or {}
    rows.append({"model": model, "env": env, "session": m.group("sess"), "rep": int(m.group("rep") or 0),
                 "extra": m.group("rest"), "on": tuple(diff.get("on_extra", [])), "off": tuple(diff.get("off_defaults", [])),
                 "median": d["timing"]["median"], "hash": d.get("program_hash"), "kernels": (d.get("code") or {}).get("kernel_count"),
                 "alloc": (d.get("code") or {}).get("alloc_bytes"), "extern": (d.get("code") or {}).get("extern_call_count"),
                 "fired": d.get("fired") or {}, "correct": d.get("correct"), "compile_s": d.get("compile_s"), "peak_mem_mb": d.get("peak_mem_mb")})
print(f"{a.label}: {len(rows)} timed measurements, env {rows[0]['env'] if rows else '?'}")
env = rows[0]["env"] if rows else ""
by = defaultdict(list)
for r in rows:
    by[(r["session"], r["model"])].append(r)

def verdict(d, d2, tau):
    if d2 is None:
        return "same"
    if d > tau and d2 > tau:
        return "slower"
    if d < -tau and d2 < -tau:
        return "faster"
    return "same"

# ---------------- sweeps (sessions without a -sw/-attr/-e2e suffix) ----------------
n_sweeps = 0
for (sess, model), rs in sorted(by.items()):
    if sess.endswith(("-sw", "-attr", "-e2e")) or any(r["extra"] for r in rs):
        continue
    base = sorted([r for r in rs if not r["on"] and not r["off"] and 1 <= r["rep"] <= 7], key=lambda r: r["rep"])
    if len(base) < 5:
        continue
    nm = noise_model([r["median"] for r in base])
    base0 = next((r for r in rs if not r["on"] and not r["off"] and r["rep"] == 0), base[0])
    json.dump({"model": model, "env": env, "session": sess, "noise": {"center": nm.center, "tau": nm.tau, "medians": [r["median"] for r in base]}},
              open(os.path.join(out, f"noise_{model}.json"), "w"), indent=1)
    srows = []
    singles = defaultdict(dict)
    for r in rs:
        if len(r["on"]) + len(r["off"]) == 1:
            singles[(r["on"], r["off"])][r["rep"]] = r
    for (on, off), reps in singles.items():
        r0 = reps.get(0)
        if r0 is None:
            continue
        sid, direction = (on[0], "on") if on else (off[0], "off")
        d = r0["median"] - nm.center
        r2 = reps.get(2); d2 = r2["median"] - nm.center if r2 else None
        same = bool(base0["hash"]) and r0["hash"] == base0["hash"]
        v = "same" if same else verdict(d, d2, nm.tau)
        dd = 0.5 * (d + d2) if (d2 is not None and v != "same") else d
        srows.append({"switch": sid, "direction": direction, "median_ms": r0["median"], "delta_ms": dd, "delta_pct": 100 * dd / nm.center,
                      "beyond_noise": abs(d) > nm.tau, "slower": v == "slower", "faster": v == "faster",
                      "confirmation": (f"second run {r2['median']:.3f} ms" if r2 else ""), "same_program": same,
                      "kernel_count": r0["kernels"], "baseline_ms": nm.center, "alloc_bytes": r0["alloc"], "correct": r0["correct"]})
    json.dump({"model": model, "env": env, "session": sess, "base_center_ms": nm.center, "tau_ms": nm.tau, "rows": srows},
              open(os.path.join(out, f"sweep_{model}.json"), "w"), indent=1)
    n_sweeps += 1
print(f"  sweeps rebuilt: {n_sweeps} models")

# ---------------- switch datasets (sessions ending in -sw) ----------------
sw_sessions = sorted({s for (s, _) in by if s.endswith("-sw")})
if sw_sessions and not a.no_profiles:
    from rewrite_blame import models as _models
for sess in sw_sessions:
    nc = sorted([r for r in by.get((sess, a.noise_config), []) if not r["on"] and not r["off"] and 1 <= r["rep"] <= 7], key=lambda r: r["rep"])
    tau_pct = max(100 * noise_model([r["median"] for r in nc]).tau / noise_model([r["median"] for r in nc]).center, 2.0) if len(nc) >= 2 else 2.0
    print(f"  session {sess}: tau {tau_pct:.2f}% from {a.noise_config} ({len(nc)} runs)")
    tau = tau_pct / 100
    per_switch = defaultdict(list)
    for (s2, model), rs in sorted(by.items()):
        if s2 != sess:
            continue
        default = {r["rep"]: r for r in rs if not r["on"] and not r["off"]}
        toggled = defaultdict(dict)
        for r in rs:
            if len(r["on"]) + len(r["off"]) == 1:
                toggled[(r["on"] or r["off"])[0]][r["rep"]] = r
        for sid, reps in toggled.items():
            if 0 not in reps or 0 not in default:
                continue
            don = default_on.get(sid, True)
            on0, off0 = (default[0], reps[0]) if don else (reps[0], default[0])
            ratios = [on0["median"] / off0["median"]]
            if 2 in reps and 2 in default:
                on2, off2 = (default[2], reps[2]) if don else (reps[2], default[2])
                ratios.append(on2["median"] / off2["median"])
            changed = bool(on0["hash"]) and on0["hash"] != off0["hash"]
            label, confirmed = label_pair(ratios, tau) if changed else ("no_effect", True)
            prof = {}
            if not a.no_profiles:
                try:
                    prof = dict(_models.profile_of(model))
                except Exception as e:
                    print(f"    profile failed for {model}: {str(e)[:60]}")
            row = {"family": f"switch:{sid}", "rule_family": sid, "switch": sid, "default_on": don, "model": model, "env": env,
                   "session": sess, "x86": 0, "cuda": int("cuda" in env), "threads": 4, **prof,
                   "ms_on": on0["median"], "ms_off": off0["median"], "ratio": ratios[0], "ratios": ratios, "tau_pct": tau_pct,
                   "label": label, "confirmed": confirmed, "program_changed": changed,
                   "fired": {k: v for k, v in on0["fired"].items() if k == sid}, "kernels_on": on0["kernels"], "kernels_off": off0["kernels"],
                   "alloc_on": on0["alloc"], "alloc_off": off0["alloc"], "extern_on": on0["extern"], "extern_off": off0["extern"],
                   "compile_s_on": on0.get("compile_s"), "compile_s_off": off0.get("compile_s"),
                   "peak_mem_mb_on": on0.get("peak_mem_mb"), "peak_mem_mb_off": off0.get("peak_mem_mb"),
                   "rebuilt_from_store": True}
            per_switch[sid].append(row)
    for sid, rws in per_switch.items():
        path = os.path.join(out, "dataset_switch_" + sid.replace("/", "_") + ".jsonl")
        with open(path, "w") as f:
            for r in rws:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        print(f"    {sid}: {len(rws)} rows {dict(Counter(r['label'] for r in rws))}")
