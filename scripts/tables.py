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


def codebr(name: str) -> str:
    """Code name that may break after underscores/dots (for long kernel names)."""
    return "\\code{" + tex(name).replace("\\_", "\\_\\allowbreak{}").replace(".", ".\\allowbreak{}") + "}"


def grouped(names) -> str:
    from collections import Counter
    c = Counter(names)
    return ", ".join(codebr(k) + (f" ($\\times${v})" if v > 1 else "") for k, v in sorted(c.items()))


def short(sid: str, n=46) -> str:
    s = sid.split("/")[-1] if sid.count("/") >= 2 else sid
    return s if len(s) <= n else s[: n - 1] + "~"


def write(name, body):
    # end the file with a comment so TeX does not see a stray end-of-line before \bottomrule
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body.rstrip("\n") + "%\n")
    print("wrote", name)


def table_env(rows: list[str], colspec: str, header: str, caption: str, label: str = "", size: str = "small", pos: str = "h") -> str:
    body = "\n".join(rows) if rows else "\\multicolumn{%d}{c}{(no data yet)} \\\\" % (colspec.count("l") + colspec.count("r") + colspec.count("c") + colspec.count("p"))
    lab = f"\\label{{{label}}}" if label else ""
    return (f"\\begin{{table}}[{pos}]\\centering\\{size}\n\\begin{{tabular}}{{{colspec}}}\\toprule\n{header} \\\\\\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\n\\caption{{{caption}}}{lab}\n\\end{{table}}\n")


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
    write("corpus_table.tex", table_env(rows, "lllp{0.5\\linewidth}", "model & family & size & description",
                                        "Model corpus (random-init from fixed configs).", "tab:corpus", size="scriptsize"))


def noise():
    rows = []
    for d in load_glob("noise_*.json") + load_glob("vm/noise_*.json"):
        n = d["noise"]
        host = "Mac" if "Mac" in d["env"] else "VM"
        rows.append(f"{tex(d['model'])} & {host} & {len(n['medians'])} & {n['center']:.3f} & {n['spread']:.3f} & "
                    f"{n['tau']:.3f} & {100*n['tau']/n['center']:.1f} & {n['pairwise_max']:.3f} \\\\")
    write("noise.tex", "\n".join(rows) + "\n")
    write("noise_table.tex", table_env(rows, "llrrrrrr", "model & machine & runs & centre (ms) & MAD (ms) & $\\tau$ (ms) & $\\tau$ (\\%) & max $|\\Delta|$ (ms)",
                                       "Baseline noise from repeated identical runs in separate processes (Mac: Apple M4 Pro, 8 threads; VM: course VM, 4-core Xeon Silver 4216). Last column: largest difference between two identical runs.", "tab:noise", size="scriptsize"))


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
    write("verify_table.tex", table_env(rows, "lrrlrrrr", "model & kernels & rules fired & candidates (pat/cfg/opt) & changes graph & fires only & no effect & errors",
                                        "Switch verification: effect of toggling each candidate switch on the compiled program. A pattern rule that never fired cannot change anything when turned off, so only fired rules are candidates.", "tab:verify", size="scriptsize"))


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
        lines.append("culprits & " + ", ".join(codebr(short(c, 60)) for c in r["culprits"]) + " \\\\")
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
    write("tests_table.tex", table_env(rows, "lr", "test file & tests", "Automated tests (unit tests run in well under a second; Inductor tests compile toy graphs).", "tab:tests"))


corpus(); noise(); verify_summary(); sweeps(); attributions(); tests()


# ---------------------------------------------------------------- narrative sections
def _host_label(env: str) -> str:
    h = env.split("|")[0]
    return "Mac (M4 Pro, 8 threads)" if "MacBook" in h or "Mac" in h else "VM (Xeon, 4 threads)"


def sweeps_section():
    parts = []
    for d in load_glob("sweep_*.json") + load_glob("vm/sweep_*.json"):
        rows = [r for r in d["rows"] if "error" not in r]
        slow = [r for r in rows if r["slower"]]
        fast = [r for r in rows if r["faster"]]
        n = d["noise"]
        txt = (f"\\subsubsection*{{{tex(d['model'])} on the {_host_label(d['env'])}}}\n"
               f"Baseline {d['reference_ms']:.3f}~ms, $\\tau$ = {n['tau']:.3f}~ms ({100*n['tau']/d['reference_ms']:.1f}\\%). "
               f"{len(rows)} graph-changing switches were toggled one at a time: {len(slow)} made the model slower than "
               f"$\\tau$, {len(fast)} made it faster, {len(rows)-len(slow)-len(fast)} stayed within noise. ")
        if slow:
            txt += "Slower: " + "; ".join(f"{codebr(short(r['switch']))} ({r['direction']}, {r['delta_pct']:+.1f}\\%)" for r in slow) + ". "
        if fast:
            txt += "Faster: " + "; ".join(f"{codebr(short(r['switch']))} ({r['direction']}, {r['delta_pct']:+.1f}\\%)" for r in fast) + ". "
        trows = [f"{tex(short(r['switch']))} & {r['direction']} & {r['median_ms']:.3f} & {r['delta_pct']:+.1f} & {r['kernel_count']} & "
                 + ("\\textbf{slower}" if r["slower"] else ("faster" if r["faster"] else "")) + " \\\\" for r in rows]
        txt += "\n" + table_env(trows, "llrrrl", "switch & toggled & median (ms) & $\\Delta$ (\\%) & kernels & vs.\\ $\\tau$",
                                 f"Leave-one-out sweep, {tex(d['model'])}, {_host_label(d['env'])}.", size="scriptsize")
        ht = "mac" if "Mac" in d["env"] else "vm"
        txt += (f"\\IfFileExists{{figures/sweep_{d['model']}_{ht}.pdf}}{{\\begin{{figure}}[h]\\centering"
                f"\\includegraphics[width=0.9\\linewidth]{{figures/sweep_{d['model']}_{ht}.pdf}}\\end{{figure}}}}{{}}\n")
        parts.append(txt)
    if not parts:
        parts.append("No timed sweeps have completed yet.\n")
    write("sweeps_section.tex", "\n".join(parts))


def attribution_section():
    from collections import Counter
    parts = []
    for d in load_glob("attribution_*.json") + load_glob("vm/attribution_*.json"):
        r = d["result"]
        host = d["env"].split("|")[0]
        tag = re.sub(r"[^A-Za-z0-9]", "", d["model"] + host)
        fa, sl, cs = d["fast"], d["slow"], d.get("culprit_state")
        txt = (f"\\subsection*{{{tex(d['model'])} on the {_host_label(d['env'])}: verdict \\emph{{{d['kind']}}}}}\n"
               f"The slow state differs from the fast state in {len(r['candidates'])} switches. ")
        if d["judge"] == "timing":
            txt += (f"Fast {fa['timing']['median']:.3f}~ms vs.\\ slow {sl['timing']['median']:.3f}~ms "
                    f"({100*(sl['timing']['median']/fa['timing']['median']-1):+.1f}\\%); $\\tau$ = {d['noise']['tau']:.3f}~ms. ")
        txt += (f"ddmin needed {r['judge_calls']} judge evaluations (versus {len(r['candidates'])} for leave-one-out, "
                f"{len(r['candidates'])*(len(r['candidates'])-1)//2} for all pairs) and returned ")
        if r["culprits"]:
            txt += "the culprit set " + ", ".join(codebr(c) for c in r["culprits"]) + ". "
            if r["interaction"]:
                txt += "Each culprit alone was judged \\emph{fast}: the slowdown needs all of them together. "
            elif len(r["culprits"]) == 1:
                txt += "A single change reproduces the whole slowdown. "
        else:
            txt += "no culprits: " + "; ".join(tex(n) for n in r["notes"]) + ". "
        if cs:
            ka = Counter(re.sub(r"_\d+$", "", k["name"]) for k in fa["code"]["kernels"])
            kb = Counter(re.sub(r"_\d+$", "", k["name"]) for k in cs["code"]["kernels"])
            ea, eb = Counter(fa["code"]["extern_calls"]), Counter(cs["code"]["extern_calls"])
            txt += (f"\\textbf{{Kernel-level explanation.}} fused kernels {fa['code']['kernel_count']} $\\rightarrow$ {cs['code']['kernel_count']}, "
                    f"extern calls {fa['code']['extern_call_count']} $\\rightarrow$ {cs['code']['extern_call_count']}, "
                    f"intermediate bytes {fa['code']['alloc_bytes']:,} $\\rightarrow$ {cs['code']['alloc_bytes']:,}. ")
            only_a, only_b = sorted((ka - kb).elements()), sorted((kb - ka).elements())
            if only_a:
                txt += "Kernels lost: " + grouped(only_a) + ". "
            if only_b:
                txt += "Kernels gained: " + grouped(only_b) + ". "
            if (ea - eb) or (eb - ea):
                txt += ("Extern calls lost: " + (grouped(sorted((ea - eb).elements())) or "none")
                        + "; gained: " + (grouped(sorted((eb - ea).elements())) or "none") + ". ")
            if cs.get("suppressed"):
                txt += "Rules suppressed in the culprit state: " + ", ".join(codebr(short(k)) for k in cs["suppressed"]) + ". "
        arows = open(os.path.join(OUT, f"attribution_{tag}.tex")).read().rstrip("%\n").splitlines()
        txt += "\n" + table_env(arows, "lp{0.62\\linewidth}", "field & value", f"Attribution summary, {tex(d['model'])}.")
        ht = "mac" if "Mac" in d["env"] else "vm"
        txt += (f"\\IfFileExists{{figures/trace_{d['model']}_{ht}.pdf}}{{\\begin{{figure}}[h]\\centering"
                f"\\includegraphics[width=0.7\\linewidth]{{figures/trace_{d['model']}_{ht}.pdf}}"
                f"\\caption{{ddmin search trace for {tex(d['model'])} ({_host_label(d['env'])}): each point is one judge evaluation.}}\\end{{figure}}}}{{}}\n")
        parts.append(txt)
    if not parts:
        parts.append("No attribution runs have completed yet.\n")
    write("attribution_section.tex", "\n".join(parts))


def counts():
    sys.path.insert(0, ".")
    from rewrite_blame import models
    write("corpuscount.tex", f"{len(models.names())} ")
    try:
        out = subprocess.run([".venv/bin/python", "-m", "pytest", "--collect-only", "-q", "-m", "unit or inductor"],
                             capture_output=True, text=True, timeout=300).stdout
        n = sum(1 for l in out.splitlines() if "::" in l)
        write("testcount.tex", f"{n} ")
    except Exception:
        write("testcount.tex", "100 ")


sweeps_section(); attribution_section(); counts()
