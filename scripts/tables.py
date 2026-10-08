"""Generate LaTeX table snippets from results JSON. Usage: python scripts/tables.py [results_dir] [out_dir]

Writes: corpus.tex, noise.tex, verify_summary.tex, sweep_<model>.tex, attribution_<model>.tex, tests.tex
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "report/midterm/tables"
os.makedirs(OUT, exist_ok=True)


def tex(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace("{", r"\{").replace("}", r"\}").replace("[", "{[}").replace("]", "{]}"))


def short(sid: str, n=46) -> str:
    s = sid.split("/")[-1] if sid.count("/") >= 2 else sid
    return s if len(s) <= n else s[: n - 1] + "~"


def write(name, body):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)
    print("wrote", name)


def load_glob(pat):
    out = []
    for p in sorted(glob.glob(os.path.join(RES, pat))):
        try:
            out.append(json.load(open(p)))
        except Exception as e:
            print("skip", p, e)
    return out


def corpus():
    sys.path.insert(0, ".")
    from rewrite_blame import models
    rows = []
    for n in models.names():
        s = models.get(n)
        rows.append(f"{tex(n)} & {s.family} & {s.size} & {tex(s.description)} \\\\")
    write("corpus.tex", "\n".join(rows) + "\n")


def noise():
    rows = []
    for d in load_glob("noise_*.json") + load_glob("vm/noise_*.json"):
        n = d["noise"]
        host = d["env"].split("|")[0]
        rows.append(f"{tex(d['model'])} & {tex(host)} & {len(n['medians'])} & {n['center']:.3f} & {n['spread']:.3f} & "
                    f"{n['tau']:.3f} & {100*n['tau']/n['center']:.1f} & {n['pairwise_max']:.3f} \\\\")
    write("noise.tex", "\n".join(rows) + "\n")


def verify_summary():
    rows = []
    for d in load_glob("verify_*.json") + load_glob("vm/verify_*.json"):
        from collections import Counter
        c = Counter(r.get("status", "error") for r in d["effects"])
        uni = d["universe"]
        host = "Mac" if "MacBook" in str(d.get("base", "")) else ""
        rows.append(f"{tex(d['model'])} & {d['baseline']['kernel_count']} & {len(d['baseline']['fired'])} & "
                    f"{len(uni['pattern'])}/{len(uni['config'])}/{len(uni['option'])} & {c.get('changes_graph',0)} & "
                    f"{c.get('fires_only',0)} & {c.get('no_effect',0)} & {c.get('error',0)} \\\\")
    write("verify_summary.tex", "\n".join(rows) + "\n")


def sweeps():
    for d in load_glob("sweep_*.json") + load_glob("vm/sweep_*.json"):
        host = d["env"].split("|")[0]
        rows = []
        for r in d["rows"]:
            if "error" in r:
                continue
            flag = r"\textbf{slower}" if r["slower"] else ("faster" if r["faster"] else "")
            rows.append(f"{tex(short(r['switch']))} & {r['direction']} & {r['median_ms']:.3f} & {r['delta_pct']:+.1f} & "
                        f"{r['kernel_count']} & {flag} \\\\")
        tag = re.sub(r"[^A-Za-z0-9]", "", d["model"] + host)
        write(f"sweep_{tag}.tex", "\n".join(rows) + "\n")


def attributions():
    for d in load_glob("attribution_*.json") + load_glob("vm/attribution_*.json"):
        r = d["result"]
        host = d["env"].split("|")[0]
        tag = re.sub(r"[^A-Za-z0-9]", "", d["model"] + host)
        lines = [f"\\textbf{{model}} & {tex(d['model'])} ({tex(host)}) \\\\",
                 f"judge & {d['judge']} ({tex(d['metric'])}) \\\\"]
        if d.get("noise"):
            n = d["noise"]
            lines.append(f"fast reference & {n['center']:.3f} ms, $\\tau$ = {n['tau']:.3f} ms ({100*n['tau']/n['center']:.1f}\\%) \\\\")
            lines.append(f"slow state & {d['slow']['timing'].get('median', float('nan')):.3f} ms \\\\")
        lines.append(f"candidate changes & {len(r['candidates'])} \\\\")
        lines.append(f"judge evaluations & {r['judge_calls']} \\\\")
        lines.append(f"verdict & {d['kind']} \\\\")
        lines.append("culprits & " + ", ".join(f"\\texttt{{{tex(short(c, 60))}}}" for c in r["culprits"]) + " \\\\")
        if d.get("culprit_state"):
            cs, fa = d["culprit_state"], d["fast"]
            lines.append(f"kernels fast $\\rightarrow$ culprits & {fa['code']['kernel_count']} $\\rightarrow$ {cs['code']['kernel_count']} \\\\")
            lines.append(f"extern calls & {fa['code']['extern_call_count']} $\\rightarrow$ {cs['code']['extern_call_count']} \\\\")
            lines.append(f"intermediate bytes & {fa['code']['alloc_bytes']:,} $\\rightarrow$ {cs['code']['alloc_bytes']:,} \\\\")
            if d["judge"] == "timing":
                lines.append(f"median ms & {fa['timing']['median']:.3f} $\\rightarrow$ {cs['timing']['median']:.3f} \\\\")
        write(f"attribution_{tag}.tex", "\n".join(lines) + "\n")


def tests():
    try:
        out = subprocess.run([".venv/bin/python", "-m", "pytest", "--collect-only", "-q", "-m", "unit or inductor"],
                             capture_output=True, text=True, timeout=300).stdout
    except Exception as e:
        print("collect failed", e)
        return
    from collections import Counter
    c = Counter()
    for line in out.splitlines():
        if "::" in line:
            c[line.split("::")[0].split("/")[-1]] += 1
    rows = [f"{tex(k)} & {v} \\\\" for k, v in sorted(c.items())]
    rows.append(f"\\textbf{{total}} & \\textbf{{{sum(c.values())}}} \\\\")
    write("tests.tex", "\n".join(rows) + "\n")


corpus(); noise(); verify_summary(); sweeps(); attributions(); tests()
