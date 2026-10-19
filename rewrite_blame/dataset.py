"""Build the guard dataset: for every configuration of a sweep family, measure the model with the
rewrite on and off (confirmed by the two-run rule) and write one labelled row.

Rows are appended to a JSONL file as they are produced, so a collection can be interrupted and
resumed; configurations already in the file are skipped.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from .guards import firing_features, family_of
from .models import sweeps
from .pipeline import Runner, parse_state_spec


def _rule_switch_ids(reg, family: str) -> list[str]:
    fam = sweeps.FAMILY_RULE[family]["family"]
    return [s.id for s in reg.of_kind("pattern") if family_of(s.id) == fam or fam in s.id]


def label_pair(ratios: list[float], tau_frac: float) -> tuple[str, bool]:
    """``ratios`` = time(rule on) / time(rule off) from independent run pairs. harmful/helpful need
    every pair to agree beyond tau (AND rule); otherwise neutral. Returns (label, confirmed)."""
    if not ratios:
        return "neutral", False
    if all(r > 1 + tau_frac for r in ratios):
        return ("harmful", len(ratios) >= 2)
    if all(r < 1 - tau_frac for r in ratios):
        return ("helpful", len(ratios) >= 2)
    return "neutral", len(ratios) >= 2


def measure_pair(runner: Runner, name: str, on_state, off_state, tau: float, rule_ids: set[str]) -> dict | None:
    """Measure one model with the rule on and off (a second independent pair when the first is beyond tau),
    and label it. Returns the row fields shared by the sweep-family and the switch datasets, or None on error."""
    m_on = runner.measure(name, on_state)
    m_off = runner.measure(name, off_state)
    if m_on.error or m_off.error:
        return None
    program_changed = bool(m_on.program_hash) and m_on.program_hash != m_off.program_hash
    fired = {k: v for k, v in m_on.fired.items() if k in rule_ids}
    ratios = [m_on.median_ms / m_off.median_ms]
    if program_changed and abs(ratios[0] - 1) > tau:
        m_on2 = runner.measure(name, on_state, rep=2)
        m_off2 = runner.measure(name, off_state, rep=2)
        if not (m_on2.error or m_off2.error):
            ratios.append(m_on2.median_ms / m_off2.median_ms)
    label, confirmed = label_pair(ratios, tau) if program_changed else ("no_effect", True)
    feats = firing_features(m_on.firings, next(iter(fired), "")) if fired else None
    return {"ms_on": m_on.median_ms, "ms_off": m_off.median_ms, "ratio": ratios[0], "ratios": ratios,
            "tau_pct": 100 * tau, "label": label, "confirmed": confirmed, "program_changed": program_changed,
            "fired": fired, "kernels_on": m_on.kernel_count, "kernels_off": m_off.kernel_count,
            "alloc_on": m_on.code.get("alloc_bytes"), "alloc_off": m_off.code.get("alloc_bytes"),
            "extern_on": m_on.code.get("extern_call_count"), "extern_off": m_off.code.get("extern_call_count"),
            "compile_s_on": m_on.compile_s, "compile_s_off": m_off.compile_s,
            "peak_mem_mb_on": m_on.peak_mem_mb, "peak_mem_mb_off": m_off.peak_mem_mb,
            "firing_features": feats}


def family_tau(runner: Runner, label: str, names: list[str], on_state, tau_pct: float | None, noise_config: str | None, log) -> float:
    """Relative noise threshold (percent) for a dataset: given, or measured on ``noise_config``, or the median
    over three representative configurations (one unlucky noise run must not set the bar); floor 2%."""
    if tau_pct is not None:
        return tau_pct
    reps = [noise_config] if noise_config else sorted({names[0], names[len(names) // 2], names[-1]})
    taus = []
    for rep_name in reps:
        nm, _ = runner.noise(rep_name, on_state)
        taus.append(100 * nm.tau / nm.center)
        log(f"[dataset] {label}: noise on {rep_name}: tau {taus[-1]:.2f}%")
    t = max(sorted(taus)[len(taus) // 2], 2.0)
    log(f"[dataset] {label}: tau = {t:.2f}% (median of {len(taus)})")
    return t


def collect_switch(runner: Runner, switch_id: str, names: list[str], out_path: Path, tau_pct: float | None = None,
                   noise_config: str | None = None, log=print) -> list[dict]:
    """Real-model rows for ONE switch: each model measured with the switch on and off (twice when beyond tau),
    features = the whole-model profile (``models.profile_of``). ``label`` keeps the family convention:
    harmful = the switch ON is slower than OFF. ``default_on`` records the switch's default so training can
    choose between a blocking guard (default on) and an enabling guard (default off)."""
    from .models import profile_of
    reg = runner.reg
    if switch_id not in reg:
        raise KeyError(f"unknown switch {switch_id!r}")
    default = reg.default_state()
    default_on = switch_id in default
    on_state = frozenset(default | {switch_id})
    off_state = frozenset(default - {switch_id})
    rule_ids = {switch_id}
    family = f"switch:{switch_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["model"])
            except Exception:
                pass
    tau_pct = family_tau(runner, family, names, default, tau_pct, noise_config, log)
    tau = tau_pct / 100.0
    rows = []
    for i, name in enumerate(names):
        if name in done:
            continue
        t0 = time.time()
        try:
            prof = profile_of(name)
        except Exception as e:        # a zoo model that does not build here: skip, do not abort the run
            log(f"[dataset] {name}: profile failed ({str(e)[:80]})")
            continue
        pair = measure_pair(runner, name, on_state, off_state, tau, rule_ids)
        if pair is None:
            log(f"[dataset] {name}: error")
            continue
        row = {"family": family, "rule_family": switch_id, "switch": switch_id, "default_on": default_on,
               "model": name, "env": runner.env, "session": runner.cfg.session_tag,
               "x86": int("x86" in runner.env and "cuda" not in runner.env), "cuda": int("cuda" in runner.env), "threads": runner.threads,
               **prof, **pair}
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        log(f"[dataset] {i+1}/{len(names)} {name}: on {pair['ms_on']:.3f} off {pair['ms_off']:.3f} ratio {pair['ratio']:.3f} -> {pair['label']}"
            f"{' (confirmed)' if pair['confirmed'] else ''} [{time.time()-t0:.0f}s]")
    return rows


def collect(runner: Runner, family: str, names: list[str], out_path: Path, tau_pct: float | None = None,
            noise_config: str | None = None, log=print) -> list[dict]:
    reg = runner.reg
    spec = sweeps.FAMILY_RULE[family]
    on_state = parse_state_spec(reg, spec["on_spec"])
    off_state = parse_state_spec(reg, spec["off_spec"])
    rule_ids = set(_rule_switch_ids(reg, family))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["model"])
            except Exception:
                pass
    tau_pct = family_tau(runner, family, names, on_state, tau_pct, noise_config, log)
    tau = tau_pct / 100.0
    rows = []
    for i, name in enumerate(names):
        if name in done:
            continue
        t0 = time.time()
        pair = measure_pair(runner, name, on_state, off_state, tau, rule_ids)
        if pair is None:
            log(f"[dataset] {name}: error")
            continue
        row = {
            "family": family, "rule_family": spec["family"], "model": name, "env": runner.env, "session": runner.cfg.session_tag,
            "x86": int("x86" in runner.env and "cuda" not in runner.env), "cuda": int("cuda" in runner.env), "threads": runner.threads,
            **{f"cfg_{k}": v for k, v in sweeps.config_of(name).items()},
            **pair,
        }
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        log(f"[dataset] {i+1}/{len(names)} {name}: on {pair['ms_on']:.3f} off {pair['ms_off']:.3f} ratio {pair['ratio']:.3f} "
            f"-> {pair['label']}{' (confirmed)' if pair['confirmed'] else ''} [{time.time()-t0:.0f}s]")
    return rows


def load(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows
