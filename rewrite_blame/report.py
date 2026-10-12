"""Human-readable reports (Markdown) for attribution, sweeps and switch verification."""

from __future__ import annotations

from .codegen_stats import CodeStats, diff_stats


def _fmt_ms(x):
    return "-" if x is None else f"{x:.3f}"


def render_attribution(a: dict) -> str:
    r = a["result"]
    lines = [f"# Attribution report: {a['model']}", ""]
    lines.append(f"- environment: `{a['env']}`")
    lines.append(f"- judge: {a['judge']} ({a['metric']})")
    if a.get("noise"):
        n = a["noise"]
        lines.append(f"- reference (fast) median: {n['center']:.3f} ms from {len(n['medians'])} repeated runs; "
                     f"MAD {n['spread']:.3f} ms; threshold tau = {n['tau']:.3f} ms (k={n['k']}, floor {n['floor_frac']*100:.0f}%); "
                     f"max |diff| between identical runs {n['pairwise_max']:.3f} ms")
    fast, slow = a["fast"], a["slow"]
    if a["judge"] == "timing":
        lines.append(f"- fast state: {_fmt_ms(fast['timing'].get('median'))} ms, slow state: {_fmt_ms(slow['timing'].get('median'))} ms "
                     f"(+{slow['timing'].get('median', 0) - fast['timing'].get('median', 0):.3f} ms)")
    lines.append(f"- candidate changes: {len(r['candidates'])}; judge evaluations: {r['judge_calls']}; wall {a['wall_s']:.0f} s; "
                 f"new measurements {a['new_measurements']}, cache hits {a['cache_hits']}")
    lines.append("")
    lines.append(f"## Verdict: {a['kind']}")
    if r["notes"]:
        for n in r["notes"]:
            lines.append(f"- note: {n}")
    if r["culprits"]:
        lines.append("")
        lines.append("Culprit change(s):")
        for c in r["culprits"]:
            single = r["singles"].get(c)
            lines.append(f"- `{c}`" + (f"  (alone: {single})" if single else ""))
        if a.get("culprit_state"):
            cs = a["culprit_state"]
            lines.append("")
            lines.append("## Kernel-level difference (fast -> fast + culprits)")
            d = diff_stats(CodeStats.from_dict(fast["code"]), CodeStats.from_dict(cs["code"]))
            if a["judge"] == "timing":
                lines.append(f"- median ms: {_fmt_ms(fast['timing'].get('median'))} -> {_fmt_ms(cs['timing'].get('median'))}")
            lines.append(f"- fused kernels: {d['kernel_count'][0]} -> {d['kernel_count'][1]}")
            lines.append(f"- extern/fallback calls: {d['extern_call_count'][0]} -> {d['extern_call_count'][1]}")
            lines.append(f"- intermediate allocation bytes: {d['alloc_bytes'][0]:,} -> {d['alloc_bytes'][1]:,}")
            lines.append(f"- static loads/stores: {d['total_loads'][0]}/{d['total_stores'][0]} -> {d['total_loads'][1]}/{d['total_stores'][1]}")
            if d["kernels_only_in_a"]:
                lines.append(f"- kernels only in fast: {', '.join(d['kernels_only_in_a'])}")
            if d["kernels_only_in_b"]:
                lines.append(f"- kernels only with culprits: {', '.join(d['kernels_only_in_b'])}")
            if d["externs_only_in_a"]:
                lines.append(f"- externs only in fast: {', '.join(d['externs_only_in_a'])}")
            if d["externs_only_in_b"]:
                lines.append(f"- externs only with culprits: {', '.join(d['externs_only_in_b'])}")
            f0, f1 = fast["fired"], cs["fired"]
            ch = {k: (f0.get(k, 0), f1.get(k, 0)) for k in set(f0) | set(f1) if f0.get(k, 0) != f1.get(k, 0)}
            if ch:
                lines.append("- rule firings changed: " + ", ".join(f"{k}: {v[0]}->{v[1]}" for k, v in sorted(ch.items())))
            if cs.get("suppressed"):
                lines.append("- rules suppressed by the culprit state: " + ", ".join(f"{k} x{v}" for k, v in cs["suppressed"].items()))
    lines.append("")
    lines.append("## Search trace")
    lines.append("| step | changes tried | verdict | value | reference | note |")
    lines.append("|---|---|---|---|---|---|")
    for t in r["trace"]:
        lines.append(f"| {len(t['candidate'])} | {', '.join(t['candidate'])[:120]} | {t['verdict']} | "
                     f"{'' if t['value'] is None else round(t['value'], 3)} | {'' if t['reference'] is None else round(t['reference'], 3)} | {t.get('note','')} |")
    return "\n".join(lines) + "\n"


def render_sweep(s: dict) -> str:
    n = s["noise"]
    lines = [f"# Leave-one-out sweep: {s['model']}", "",
             f"- environment: `{s['env']}`",
             f"- baseline median {s['reference_ms']:.3f} ms; tau {n['tau']:.3f} ms ({100*n['tau']/s['reference_ms']:.1f}%); "
             f"identical-run max |diff| {n['pairwise_max']:.3f} ms"
             + (f"; interleaved baselines: {', '.join(f'{b:.3f}' for b in s.get('baselines', []))} ms, {s.get('drift_events', 0)} drift event(s)" if s.get("baselines") else ""), "",
             "| switch | toggled | median ms | delta ms | delta % | beyond noise | kernels | correct |", "|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        if "error" in r:
            lines.append(f"| `{r['switch']}` | | error | | | | | {r['error']} |")
            continue
        flag = "SLOWER" if r["slower"] else ("faster" if r["faster"] else "")
        if r.get("confirmation") and not (r["slower"] or r["faster"]):
            flag = "unconfirmed"
        lines.append(f"| `{r['switch']}` | {r['direction']} | {r['median_ms']:.3f} | {r['delta_ms']:+.3f} | {r['delta_pct']:+.1f} | "
                     f"{flag} | {r['kernel_count']} | {r['correct']} |")
    return "\n".join(lines) + "\n"


def render_verify(v: dict) -> str:
    lines = [f"# Switch verification: {v['model']}", "",
             f"- baseline fused kernels: {v['baseline']['kernel_count']}",
             f"- rules fired in baseline: {len(v['baseline']['fired'])}",
             f"- candidates: {len(v['universe']['pattern'])} fired patterns, {len(v['universe']['config'])} config flags, "
             f"{len(v['universe']['option'])} opt-in options", "",
             "| switch | kind | toggled | status | kernels | externs | allocation bytes | ops changed |", "|---|---|---|---|---|---|---|---|"]
    for r in v["effects"]:
        if "error" in r:
            lines.append(f"| `{r['switch']}` | | {r['direction']} | error: {r['error'][:80]} | | | | |")
            continue
        ops = ", ".join(f"{k.replace('aten.', '')} {a}->{b}" for k, (a, b) in sorted(r["ops_changed"].items())[:6])
        lines.append(f"| `{r['switch']}` | {r['kind']} | {r['direction']} | {r['status']} | {r['kernel_count'][0]}->{r['kernel_count'][1]} | "
                     f"{r['extern_call_count'][0]}->{r['extern_call_count'][1]} | {r['alloc_bytes'][0]:,}->{r['alloc_bytes'][1]:,} | {ops} |")
    return "\n".join(lines) + "\n"


def render_iterative(a: dict) -> str:
    it = a["iterative"]
    lines = [f"# Iterative attribution: {a['model']}", "", f"- environment: `{a['env']}`",
             f"- fast {a['fast_ms']:.3f} ms, slow {a['slow_ms']:.3f} ms (+{a['total_ms']:.3f} ms); "
             f"explained by found causes: {a['explained_ms']:.3f} ms; residual within noise: {it['residual_explained']}",
             f"- rounds: {len(it['rounds'])}; judge evaluations: {it['judge_calls']}; wall {a['wall_s']:.0f} s", ""]
    if it["notes"]:
        lines += [f"- note: {n}" for n in it["notes"]]
    lines += ["| round | cause (change set) | kind | delta ms | delta % | kernels | alloc bytes |", "|---|---|---|---|---|---|---|"]
    for g in a["groups"]:
        lines.append(f"| {g['round']} | {', '.join('`'+c+'`' for c in g['culprits'])} | {g['kind']} | {g['delta_ms']:+.3f} | {g['delta_pct']:+.1f} | "
                     f"{g['kernels'][0]}->{g['kernels'][1]} | {g['alloc_bytes'][0]}->{g['alloc_bytes'][1]} |")
    return "\n".join(lines) + "\n"


def render_interactions(a: dict) -> str:
    lines = [f"# Pairwise interactions: {a['model']}", "", f"- environment: `{a['env']}`",
             f"- base {a['base_ms']:.3f} ms, tau {a['tau']:.3f} ms; {len(a['ids'])} switches, {len(a['pairs'])} pairs, {a['measurements']} measurements", "",
             "## Singles", "| switch | delta ms |", "|---|---|"]
    for k, v in sorted(a["singles"].items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"| `{k}` | {v:+.3f} |")
    lines += ["", f"## Super-additive pairs ({len(a['superadditive'])}): the pair is worse than the sum of its parts", "| a | b | kind | pair delta ms | a alone | b alone | interaction ms |", "|---|---|---|---|---|---|---|"]
    for r in a["superadditive"]:
        lines.append(f"| `{r['a']}` | `{r['b']}` | {r.get('kind','')} | {r['delta_ms']:+.3f} | {r['delta_a']:+.3f} | {r['delta_b']:+.3f} | {r['interaction_ms']:+.3f} |")
    if a.get("baselines"):
        nb, nd = len(a["baselines"]), a.get("drift_events", 0)
        lines.insert(4, f"- interleaved baselines: {nb} runs, {nd} disturbed, {a.get('level_shifts', 0)} level shift(s); unmeasured switches: {a.get('unmeasured', [])}")
        if nb and nd / nb > 0.3:
            lines.insert(5, f"- **WARNING: the machine was unstable during this scan ({nd}/{nb} baseline runs disturbed); "
                            "treat flagged pairs as provisional and repeat the scan in a quiet session.**")
    lines += ["", f"## Masking pairs ({len(a['masking'])}): the pair is faster than its slower member", "| a | b | pair delta ms | a alone | b alone | interaction ms |", "|---|---|---|---|---|---|"]
    for r in a["masking"]:
        lines.append(f"| `{r['a']}` | `{r['b']}` | {r['delta_ms']:+.3f} | {r['delta_a']:+.3f} | {r['delta_b']:+.3f} | {r['interaction_ms']:+.3f} |")
    lines += ["", "## All pairs (by |interaction|)", "| a | b | pair delta ms | interaction ms |", "|---|---|---|---|"]
    for r in a["pairs"][:40]:
        if r.get("delta_ms") is None:
            lines.append(f"| `{r['a']}` | `{r['b']}` | unmeasured | |")
            continue
        lines.append(f"| `{r['a']}` | `{r['b']}` | {r['delta_ms']:+.3f} | {r['interaction_ms']:+.3f} |")
    return "\n".join(lines) + "\n"
