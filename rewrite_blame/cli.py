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
                       session=a.session)
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
    rep = repeatability_by_diff(rows, a.model)
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

    p = sub.add_parser("guards", help="labelled rule firings + per-family stump guards (stretch goal)"); p.set_defaults(fn=cmd_guards)
    p.add_argument("--extra-store", action="append"); p.add_argument("--sweeps", action="append", help="glob(s) of sweep json files"); p.add_argument("--out")

    p = sub.add_parser("summary", help="one overview of all result files"); p.set_defaults(fn=cmd_summary)
    p.add_argument("--dirs", nargs="*"); p.add_argument("--out")

    p = sub.add_parser("show"); p.set_defaults(fn=cmd_show)
    p.add_argument("--model", default=""); p.add_argument("-n", type=int, default=40)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
