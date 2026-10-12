"""Command line: python -m rewrite_blame <command> ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import Runner, RunnerConfig, parse_state_spec


def _runner(a) -> Runner:
    cfg = RunnerConfig(registry_path=Path(a.registry), store_path=Path(a.store), threads=a.threads,
                       warmup=a.warmup, rounds=a.rounds, iters=a.iters, timeout_s=a.timeout, verbose=not a.quiet,
                       session=a.session, cooldown_factor=a.cooldown, retry_unsteady=not a.no_retry_unsteady)
    return Runner(cfg)


def _write(a, payload: dict, text: str | None = None, default_name: str = "out") -> None:
    out = Path(a.out) if a.out else Path("results") / f"{default_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str))
    if text is not None:
        out.with_suffix(".md").write_text(text)
        print(text)
    print(f"[wrote {out}]", file=sys.stderr)


def cmd_list_models(a):
    from . import models
    for n in models.names():
        s = models.get(n)
        print(f"{n:32s} {s.family:7s} {s.size:7s} {s.description}")


def cmd_discover(a):
    from .discover import discover, summarize
    reg = discover()
    Path(a.registry).parent.mkdir(parents=True, exist_ok=True)
    reg.save(a.registry)
    print(json.dumps(summarize(reg), indent=1))
    print(f"[wrote {a.registry} with {len(reg)} switches]", file=sys.stderr)


def cmd_measure(a):
    r = _runner(a)
    st = parse_state_spec(r.reg, a.state)
    m = r.measure(a.model, st, time_it=not a.no_time, force=a.force)
    d = m.to_dict()
    d["timing"].pop("samples", None) if d.get("timing") else None
    print(json.dumps({k: d[k] for k in ("model", "state_hash", "compile_s", "timing", "correct", "error", "fired", "counters", "metrics")}, indent=1, default=str))
    print("kernels:", [k["name"] for k in m.code.get("kernels", [])])
    print("externs:", m.code.get("extern_calls"))


def cmd_noise(a):
    r = _runner(a)
    st = parse_state_spec(r.reg, a.state)
    nm, ms = r.noise(a.model, st, runs=a.runs, k=a.k, floor_frac=a.floor)
    payload = {"model": a.model, "env": r.env, "state": sorted(st), "noise": nm.to_dict(),
               "runs": [{"median": m.median_ms, "mad": m.timing["mad"], "min": m.timing["min"], "p90": m.timing["p90"]} for m in ms]}
    print(json.dumps(payload["noise"], indent=1))
    _write(a, payload, default_name=f"noise_{a.model}")


def cmd_verify(a):
    from .report import render_verify
    r = _runner(a)
    base = parse_state_spec(r.reg, a.state)
    v = r.verify_switches(a.model, base, include_options=not a.no_options)
    _write(a, v, render_verify(v), default_name=f"verify_{a.model}")


def cmd_sweep(a):
    from .report import render_sweep
    from .stats import NoiseModel
    r = _runner(a)
    base = parse_state_spec(r.reg, a.state)
    noise = None
    if a.noise_file:
        noise = NoiseModel.from_dict(json.loads(Path(a.noise_file).read_text())["noise"])
    s = r.sweep(a.model, base, include_options=not a.no_options, only_graph_changing=not a.all_switches, noise=noise)
    _write(a, s, render_sweep(s), default_name=f"sweep_{a.model}")


def cmd_attribute(a):
    from .report import render_attribution
    from .stats import NoiseModel
    r = _runner(a)
    fast = parse_state_spec(r.reg, a.fast)
    slow = parse_state_spec(r.reg, a.slow)
    noise = None
    if a.noise_file:
        noise = NoiseModel.from_dict(json.loads(Path(a.noise_file).read_text())["noise"])
    res = r.attribute(a.model, fast, slow, judge=a.judge, noise=noise, metric=a.metric, noise_runs=a.runs)
    _write(a, res, render_attribution(res), default_name=f"attribution_{a.model}")


def cmd_attribute_all(a):
    from .report import render_iterative
    from .stats import NoiseModel
    r = _runner(a)
    fast = parse_state_spec(r.reg, a.fast)
    slow = parse_state_spec(r.reg, a.slow)
    noise = NoiseModel.from_dict(json.loads(Path(a.noise_file).read_text())["noise"]) if a.noise_file else None
    res = r.attribute_all(a.model, fast, slow, noise=noise, noise_runs=a.runs, max_rounds=a.max_rounds)
    _write(a, res, render_iterative(res), default_name=f"attribution_all_{a.model}")


def cmd_interactions(a):
    from .report import render_interactions
    from .stats import NoiseModel
    r = _runner(a)
    base = parse_state_spec(r.reg, a.state)
    noise = NoiseModel.from_dict(json.loads(Path(a.noise_file).read_text())["noise"]) if a.noise_file else None
    ids = [parse_state_spec(r.reg, "none+" + t) for t in a.ids.split(",")] if a.ids else None
    if ids is not None:
        ids = [next(iter(st)) for st in ids]
    res = r.interactions(a.model, base, ids=ids, noise=noise, max_pairs=a.max_pairs, only_graph_changing=not a.all_switches)
    _write(a, res, render_interactions(res), default_name=f"interactions_{a.model}")


def cmd_tau_scan(a):
    r = _runner(a)
    fast = parse_state_spec(r.reg, a.fast)
    slow = parse_state_spec(r.reg, a.slow)
    res = r.tau_sensitivity(a.model, fast, slow, ks=[float(x) for x in a.ks.split(",")], noise_runs=a.runs)
    for row in res["rows"]:
        print(f"k={row['k']:<4} tau={row['tau']:.3f} ms  verdict={row['kind']:<12} culprits={row['culprits']}  calls={row['judge_calls']}")
    print("stable across k:", res["stable"])
    _write(a, res, default_name=f"tau_scan_{a.model}")


def cmd_repeat(a):
    """Compare baseline speed and per-switch verdicts across sessions and machines (from the stores)."""
    from .repeat import backfill_diffs, render_repeat, repeatability_by_diff
    from .store import Store
    from .switches import Registry
    r = _runner(a)
    rows = r.store.all(model=a.model)
    regs = [r.reg]
    for extra in a.extra_store or []:
        if Path(extra).exists():
            with Store(extra) as st:
                rows += st.all(model=a.model)
            reg_path = Path(extra).with_name("switches.json")
            if reg_path.exists():
                regs.append(Registry.load(reg_path))
    for extra_reg in a.extra_registry or []:
        if Path(extra_reg).exists():
            regs.append(Registry.load(extra_reg))
    backfill_diffs(rows, regs)
    from .repeat import filter_sessions
    rows, dropped = filter_sessions(rows, exclude=a.exclude_session or [], max_tau_pct=a.max_tau_pct)
    rep = repeatability_by_diff(rows, a.model)
    rep.dropped_sessions = dropped
    _write(a, rep.to_dict(), render_repeat(rep), default_name=f"repeat_{a.model}")


def cmd_guards(a):
    """Build the labelled-firing dataset from stores + sweep files and evaluate stump guards per rule family."""
    import glob
    from .guards import build_dataset, leave_one_model_out, render_guards
    from .store import Store
    from .switches import Registry, state_hash
    r = _runner(a)
    stores = [(r.store, r.reg)]
    for extra in a.extra_store or []:
        if Path(extra).exists():
            reg_path = Path(extra).with_name("switches.json")
            stores.append((Store(extra), Registry.load(reg_path) if reg_path.exists() else r.reg))
    default_rows = []
    for st, reg in stores:
        bh = state_hash(reg.default_state())
        for row in st.all():
            if row.get("state_hash") == bh and row.get("firings") and not row.get("error"):
                default_rows.append(row)
    sweeps = []
    for pat in (a.sweeps or ["results/sweep_*.json", "results/vm/sweep_*.json"]):
        for f in sorted(glob.glob(pat)):
            try:
                sweeps.append(json.loads(Path(f).read_text()))
            except Exception:
                pass
    data = build_dataset(default_rows, sweeps)
    ev = leave_one_model_out(data)
    _write(a, {"dataset": data, "evaluation": ev, "n_default_rows": len(default_rows), "n_sweeps": len(sweeps)},
           render_guards(data, ev), default_name="guards")


def cmd_summary(a):
    from .summary import load_results, render_summary, summarize
    res = load_results(a.dirs or ("results", "results/vm"))
    sm = summarize(res)
    text = render_summary(sm)
    out = Path(a.out) if a.out else Path("results/summary.md")
    out.write_text(text)
    print(text)
    print(f"[wrote {out}]", file=sys.stderr)


def cmd_dataset(a):
    """Collect labelled (rule on vs off) rows for a sweep family."""
    import random
    from .dataset import collect
    from .models import sweeps
    r = _runner(a)
    names = sweeps.grid(a.family, a.level)
    if a.shuffle:
        random.Random(a.seed).shuffle(names)
    if a.limit:
        names = names[: a.limit]
    out = Path(a.out) if a.out else Path("results") / f"dataset_{a.family}.jsonl"
    rows = collect(r, a.family, names, out, tau_pct=a.tau_pct, noise_config=a.noise_config, log=lambda *x: print(*x, file=sys.stderr, flush=True))
    from collections import Counter
    print(json.dumps({"family": a.family, "new_rows": len(rows), "labels": Counter(x["label"] for x in rows), "file": str(out)}, default=str))


def cmd_guards_train(a):
    """Evaluate guard models per family on the sweep datasets (grouped CV), fit on all rows, save a bundle."""
    from collections import Counter
    from .guard_models import GuardBundle, evaluate, fit_all, load_rows, render_eval, tree_rules
    paths = [Path(p) for p in (a.data or [])] or sorted(Path("results").glob("dataset_*.jsonl")) + sorted(Path("results/vm").glob("dataset_*.jsonl"))
    results, rules, counts = [], {}, {}
    bundle = GuardBundle()
    for fam in a.families or ["mlp", "conv", "attn"]:
        rows = load_rows(paths, fam)
        if len(rows) < 8:
            print(f"[guards-train] {fam}: only {len(rows)} rows, skipping", file=sys.stderr)
            continue
        counts[fam] = Counter(r["label"] for r in rows)
        from .guard_models import evaluate_policy
        results.append(evaluate_policy(rows, fam))          # Inductor's own hand-written rule as the baseline
        for kind in a.kinds:
            if kind in ("tree", "gbt", "logreg"):
                results.append(evaluate(rows, fam, kind, n_folds=a.folds))
            elif kind == "encoder":
                from .guard_encoder import DEFAULT_CHECKPOINT, evaluate_encoder
                from .guard_models import EvalResult
                d = evaluate_encoder(rows, fam, checkpoint=a.checkpoint or DEFAULT_CHECKPOINT, n_folds=a.folds, epochs=a.epochs,
                                     log=lambda *x: print(*x, file=sys.stderr, flush=True))
                results.append(EvalResult(**d))
            elif kind in ("api", "api-context"):
                from .guard_api import load_api_key
                if not load_api_key():
                    print(f"[guards-train] {kind}: no API key (OPENAI_API_KEY or ~/.openai_key), skipping", file=sys.stderr)
                    continue
                from .guard_api import evaluate_api
                from .guard_models import EvalResult
                d = evaluate_api(rows, fam, with_context=(kind == "api-context"), log=lambda *x: print(*x, file=sys.stderr, flush=True))
                d.pop("probs", None)
                results.append(EvalResult(**d))
        clf, names = fit_all(rows, fam, a.bundle_kind)
        if a.bundle_kind == "tree":
            rules[fam] = tree_rules(clf, names)
        bundle.add(fam, a.bundle_kind, clf, names, threshold=a.threshold)
    out = Path(a.out) if a.out else Path("results/guards_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": [r.to_dict() for r in results], "counts": {k: dict(v) for k, v in counts.items()},
                               "rules": rules, "data": [str(p) for p in paths]}, indent=1))
    text = render_eval(results, rules, counts)
    out.with_suffix(".md").write_text(text)
    print(text)
    bpath = Path(a.bundle) if a.bundle else Path("results/guards.pkl")
    if bundle.guards:
        bundle.save(bpath)
        print(f"[guards-train] bundle with {sorted(bundle.guards)} saved to {bpath}", file=sys.stderr)


def cmd_guards_eval(a):
    """End-to-end: real models, stock vs guards added one family at a time."""
    import math
    r = _runner(a)
    res = r.guards_eval(a.models, a.bundle, a.families, noise_runs=a.runs)
    lines = [f"# Guards end-to-end on real models ({res['env']})", "", "| model | default ms | " + " | ".join(f"+{f}" for f in a.families) + " |",
             "|---|---|" + "---|" * len(a.families)]
    speedups = {f: [] for f in a.families}
    for row in res["rows"]:
        cells = []
        for st in row["steps"]:
            if "error" in st:
                cells.append("error")
                continue
            cells.append(f"{st['delta_pct']:+.1f}% {st['verdict']} (blocked {st['blocked']})")
            fam = st["guards"].split("+")[-1]
            speedups[fam].append(st["delta_pct"])
        lines.append(f"| {row['model']} | {row['default_ms']:.3f} | " + " | ".join(cells) + " |")
    lines.append("")
    for st_label in ["+".join(a.families[:i]) for i in range(1, len(a.families) + 1)]:
        ds = [st["delta_pct"] for row in res["rows"] for st in row["steps"] if st.get("guards") == st_label and "error" not in st]
        if ds:
            gm = (math.exp(sum(math.log(1 + d / 100) for d in ds) / len(ds)) - 1) * 100
            lines.append(f"- guards {st_label}: geometric-mean change {gm:+.2f}%, worst {max(ds):+.1f}%, best {min(ds):+.1f}% over {len(ds)} models")
    text = "\n".join(lines) + "\n"
    _write(a, res, text, default_name="guards_eval_e2e")


def cmd_show(a):
    from .store import Store
    st = Store(a.store)
    rows = st.all(model=a.model or None)
    print(f"{len(rows)} measurements in {a.store}")
    for d in rows[-a.n:]:
        t = d.get("timing") or {}
        print(f"{d['model']:28s} {d['state_hash']} {d.get('protocol_key','?'):18s} kernels={d.get('code',{}).get('kernel_count')} "
              f"median={t.get('median') and round(t['median'],3)} err={'yes' if d.get('error') else 'no'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rewrite_blame")
    ap.add_argument("--registry", default="results/switches.json")
    ap.add_argument("--store", default="results/measurements.sqlite")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--cooldown", type=float, default=0.5, help="pause between timed measurements as a fraction of the previous one's wall time (0 on servers)")
    ap.add_argument("--no-retry-unsteady", action="store_true")
    ap.add_argument("--session", default=None, help="tag for timed measurements (default: today's date); "
                    "timings are only reused within the same session")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-models").set_defaults(fn=cmd_list_models)
    sub.add_parser("discover").set_defaults(fn=cmd_discover)

    p = sub.add_parser("measure"); p.set_defaults(fn=cmd_measure)
    p.add_argument("--model", required=True); p.add_argument("--state", default="default")
    p.add_argument("--no-time", action="store_true"); p.add_argument("--force", action="store_true")

    p = sub.add_parser("noise"); p.set_defaults(fn=cmd_noise)
    p.add_argument("--model", required=True); p.add_argument("--state", default="default")
    p.add_argument("--runs", type=int, default=7); p.add_argument("--k", type=float, default=3.0)
    p.add_argument("--floor", type=float, default=0.01); p.add_argument("--out")

    p = sub.add_parser("verify"); p.set_defaults(fn=cmd_verify)
    p.add_argument("--model", required=True); p.add_argument("--state", default="default")
    p.add_argument("--no-options", action="store_true"); p.add_argument("--out")

    p = sub.add_parser("sweep"); p.set_defaults(fn=cmd_sweep)
    p.add_argument("--model", required=True); p.add_argument("--state", default="default")
    p.add_argument("--no-options", action="store_true"); p.add_argument("--all-switches", action="store_true")
    p.add_argument("--noise-file"); p.add_argument("--out")

    p = sub.add_parser("attribute"); p.set_defaults(fn=cmd_attribute)
    p.add_argument("--model", required=True); p.add_argument("--fast", required=True); p.add_argument("--slow", required=True)
    p.add_argument("--judge", choices=["timing", "metric"], default="timing"); p.add_argument("--metric", default="kernel_count")
    p.add_argument("--noise-file"); p.add_argument("--runs", type=int, default=7); p.add_argument("--out")

    p = sub.add_parser("attribute-all", help="iterate attribution until the residual is within noise"); p.set_defaults(fn=cmd_attribute_all)
    p.add_argument("--model", required=True); p.add_argument("--fast", required=True); p.add_argument("--slow", required=True)
    p.add_argument("--noise-file"); p.add_argument("--runs", type=int, default=7); p.add_argument("--max-rounds", type=int, default=6); p.add_argument("--out")

    p = sub.add_parser("interactions", help="toggle switches alone and in pairs; flag super-additive pairs"); p.set_defaults(fn=cmd_interactions)
    p.add_argument("--model", required=True); p.add_argument("--state", default="default"); p.add_argument("--ids", help="comma-separated switch ids (default: graph-changing ones)")
    p.add_argument("--all-switches", action="store_true"); p.add_argument("--max-pairs", type=int); p.add_argument("--noise-file"); p.add_argument("--out")

    p = sub.add_parser("tau-scan", help="re-run attribution for several noise multipliers k"); p.set_defaults(fn=cmd_tau_scan)
    p.add_argument("--model", required=True); p.add_argument("--fast", required=True); p.add_argument("--slow", required=True)
    p.add_argument("--ks", default="2,3,4,6"); p.add_argument("--runs", type=int, default=7); p.add_argument("--out")

    p = sub.add_parser("repeat", help="repeatability of verdicts across sessions/machines from the store"); p.set_defaults(fn=cmd_repeat)
    p.add_argument("--model", required=True); p.add_argument("--state", default="default"); p.add_argument("--out")
    p.add_argument("--extra-store", action="append", help="additional measurement stores (e.g. pulled from another machine)")
    p.add_argument("--extra-registry", action="append", help="additional switch registries used to interpret old rows")
    p.add_argument("--exclude-session", action="append", help="session name or glob to leave out, e.g. 2026-10-10 or '*pairs*'")
    p.add_argument("--max-tau-pct", type=float, default=None, help="drop sessions whose baseline tau exceeds this percent")

    p = sub.add_parser("guards", help="labelled rule firings + per-family stump guards (stretch goal)"); p.set_defaults(fn=cmd_guards)
    p.add_argument("--extra-store", action="append"); p.add_argument("--sweeps", action="append", help="glob(s) of sweep json files"); p.add_argument("--out")

    p = sub.add_parser("summary", help="one overview of all result files"); p.set_defaults(fn=cmd_summary)
    p.add_argument("--dirs", nargs="*"); p.add_argument("--out")

    p = sub.add_parser("dataset", help="collect labelled rule-on/off rows for a sweep family (guard training data)"); p.set_defaults(fn=cmd_dataset)
    p.add_argument("--family", required=True, choices=["attn", "conv", "mlp"]); p.add_argument("--level", default="full", choices=["full", "small"])
    p.add_argument("--shuffle", action="store_true"); p.add_argument("--seed", type=int, default=0); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tau-pct", type=float, default=None); p.add_argument("--noise-config"); p.add_argument("--out")

    p = sub.add_parser("guards-train", help="evaluate and fit per-family guard models on the sweep datasets"); p.set_defaults(fn=cmd_guards_train)
    p.add_argument("--data", action="append", help="dataset jsonl file(s); default: results/dataset_*.jsonl and results/vm/dataset_*.jsonl")
    p.add_argument("--families", nargs="*"); p.add_argument("--kinds", nargs="*", default=["tree", "gbt", "logreg"])
    p.add_argument("--bundle-kind", default="tree"); p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--out"); p.add_argument("--bundle")
    p.add_argument("--checkpoint", help="encoder checkpoint for --kinds encoder (default: Laya)"); p.add_argument("--epochs", type=int, default=6)

    p = sub.add_parser("guards-eval", help="end-to-end: stock vs guards on real models, one family at a time"); p.set_defaults(fn=cmd_guards_eval)
    p.add_argument("--models", nargs="+", required=True); p.add_argument("--bundle", default="results/guards.pkl")
    p.add_argument("--families", nargs="*", default=["mlp", "conv", "attn"]); p.add_argument("--runs", type=int, default=7); p.add_argument("--out")

    p = sub.add_parser("show"); p.set_defaults(fn=cmd_show)
    p.add_argument("--model", default=""); p.add_argument("-n", type=int, default=40)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
