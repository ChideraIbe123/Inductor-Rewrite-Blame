"""LaTeX tables for the final report / paper from the analysis outputs.
Usage: python scripts/final_tables.py  -> report/final/tables/{sign_matrix,mechanism,freezing_zoo,datasets_gpu,pretrained}.tex"""
import glob, json, os, re, statistics, collections

OUT = "report/final/tables"; os.makedirs(OUT, exist_ok=True)

def tex(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
            .replace("·", r"$\cdot$").replace("Δ", r"$\Delta$").replace("→", r"$\rightarrow$").replace("`", ""))

def md_table_to_tex(md_path, caption, label, size="footnotesize", colspec=None, max_cols=None):
    rows = [l.strip() for l in open(md_path) if l.strip().startswith("|")]
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows if not set(r.replace("|", "").strip()) <= set("-: ")]
    if max_cols: cells = [c[:max_cols] for c in cells]
    n = len(cells[0])
    if colspec is None:
        width = {6: "2.1cm", 7: "1.85cm", 8: "1.6cm"}.get(n - 1, "1.4cm")
        colspec = "l" + f"p{{{width}}}" * (n - 1)
    body = [" & ".join(tex(c) for c in cells[0]) + r" \\\midrule"] + [" & ".join(tex(c) for c in r) + r" \\" for r in cells[1:]]
    return (f"\\begin{{table*}}[t]\\centering\\{size}\n\\begin{{tabular}}{{{colspec}}}\\toprule\n" + "\n".join(body) +
            f"\n\\bottomrule\\end{{tabular}}\n\\caption{{{caption}}}\\label{{{label}}}\n\\end{{table*}}\n")

def write(name, body):
    open(os.path.join(OUT, name), "w").write(body.rstrip("\n") + "%\n"); print("wrote", name)

if os.path.exists("report/final/tables/sign_matrix.md"):
    write("sign_matrix.tex", md_table_to_tex("report/final/tables/sign_matrix.md",
          "Rule $\\times$ machine sign matrix. For each switch, the number of corpus models on which having the switch ON is confirmed to hurt or help (two time-separated runs beyond the machine's noise band, effect $\\geq 3\\%$), with the largest confirmed effect. Eight of the sixteen most impactful switches have the opposite dominant sign on different machines.",
          "tab:signmatrix", size="scriptsize", colspec=None))   # colspec derived from the column count
if os.path.exists("docs/mechanism.md"):
    write("mechanism.tex", md_table_to_tex("docs/mechanism.md",
          "Mechanism: confirmed verdicts joined with the code-level change the same switch makes (ON minus OFF: kernels, external library calls, static allocation in MB; medians over the affected models). The structural change is the same on every machine; its runtime sign is not.",
          "tab:mechanism", size="scriptsize", colspec="llp{5.2cm}p{5.2cm}"))

# freezing over the real-network zoo per GPU
rows = []
for G, lab in (("rtx4090", "RTX 4090 (bf16)"), ("l4", "L4 (bf16)"), ("a40", "A40 (bf16)")):
    p = f"results/gpu/{G}/dataset_switch_lowering_freezing.jsonl"
    if not os.path.exists(p): continue
    rs = [json.loads(l) for l in open(p)]; c = collections.Counter(r["label"] for r in rs)
    g = sorted(100 * (1 - r["ratio"]) for r in rs if r["label"] == "helpful")
    conv = [r for r in rs if r["model"].startswith("zoo_")]; hf = [r for r in rs if r["model"].startswith("hfzoo_")]
    rows.append(f"{lab} & {len(rs)} & {c['helpful']} & {c['harmful']} & {c['neutral']} & {statistics.median(g):.1f} & {g[int(0.9*len(g))]:.1f} & {g[-1]:.1f} & "
                f"{sum(1 for r in conv if r['label']=='helpful')}/{len(conv)} & {sum(1 for r in hf if r['label']=='helpful')}/{len(hf)} \\\\")
if rows:
    write("freezing_zoo.tex", "\\begin{table}[t]\\centering\\footnotesize\n\\begin{tabular}{lrrrrrrrrr}\\toprule\nGPU & nets & faster & slower & neutral & median gain \\% & p90 & max & conv nets & transformers \\\\\\midrule\n" + "\n".join(rows) +
          "\n\\bottomrule\\end{tabular}\n\\caption{\\texttt{freezing} (off by default) on 163 real networks per GPU: confirmed faster/slower with freezing ON, gains among the faster ones, and the faster fraction by family.}\\label{tab:freezing}\n\\end{table}\n")

# GPU switch datasets summary
rows = []
for G, lab in (("rtx4090", "4090"), ("l4", "L4"), ("a40", "A40")):
    for f in sorted(glob.glob(f"results/gpu/{G}/dataset_switch_*.jsonl")):
        rs = [json.loads(l) for l in open(f)]; c = collections.Counter(r["label"] for r in rs)
        if c.get("no_effect", 0) == len(rs): continue
        eff = sorted(abs(100 * (r["ratio"] - 1)) for r in rs if r["label"] in ("harmful", "helpful"))
        rows.append(f"{lab} & \\texttt{{{tex(rs[0]['switch'].split('/')[-1])}}} & {len(rs)} & {c.get('harmful',0)} & {c.get('helpful',0)} & {c.get('neutral',0)} & {c.get('no_effect',0)} & {statistics.median(eff) if eff else 0:.1f} \\\\")
if rows:
    write("datasets_gpu.tex", "\\begin{table}[t]\\centering\\footnotesize\n\\begin{tabular}{llrrrrrr}\\toprule\nGPU & switch & rows & harmful & helpful & neutral & no effect & median $|$effect$|$ \\% \\\\\\midrule\n" + "\n".join(rows) +
          "\n\\bottomrule\\end{tabular}\n\\caption{Real-network switch datasets on three GPUs (bf16): each row is one network measured with the switch on and off, twice when beyond the noise band.}\\label{tab:datasetsgpu}\n\\end{table}\n")

# pretrained vs random twins (A5000)
p = "results/gpu/dataset_switch_lowering_freezing_pretrained.jsonl"
if os.path.exists(p):
    rs = {r["model"]: r for r in (json.loads(l) for l in open(p))}
    lines = []
    for base in ["resnet18","resnet50","mobilenet_v3_small","efficientnet_b0","convnext_tiny","bert_base_eagerattn","distilbert","gpt2_prefill","gpt2_decode","smollm_135m_decode","smollm_135m_prefill","qwen2_0_5b_decode"]:
        a, b = rs.get(base), rs.get(base + "_pretrained")
        if not (a and b): continue
        lines.append(f"\\texttt{{{tex(base)}}} & {100*(1-a['ratio']):+.1f} & {tex(a['label'])} & {100*(1-b['ratio']):+.1f} & {tex(b['label'])} & {a['peak_mem_mb_off'] or 0:.0f} $\\rightarrow$ {a['peak_mem_mb_on'] or 0:.0f} & {a['compile_s_off']:.0f} $\\rightarrow$ {a['compile_s_on']:.0f} \\\\")
    write("pretrained.tex", "\\begin{table}[t]\\centering\\footnotesize\n\\begin{tabular}{lrlrlcc}\\toprule\nmodel & \\multicolumn{2}{c}{random init} & \\multicolumn{2}{c}{trained weights} & peak MB off$\\rightarrow$on & compile s \\\\\n & gain \\% & label & gain \\% & label & & \\\\\\midrule\n" + "\n".join(lines) +
          "\n\\bottomrule\\end{tabular}\n\\caption{\\texttt{freezing} on the corpus models with random and with trained weights (RTX A5000, bf16): the verdicts and magnitudes agree; the price is peak memory and compile time.}\\label{tab:pretrained}\n\\end{table}\n")
