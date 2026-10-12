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
    # noise threshold for the family: measured once on a representative configuration
    if tau_pct is None:
        rep_name = noise_config or names[len(names) // 2]
        nm, _ = runner.noise(rep_name, on_state)
        tau_pct = max(100 * nm.tau / nm.center, 2.0)
        log(f"[dataset] {family}: tau = {tau_pct:.2f}% (from {rep_name})")
    tau = tau_pct / 100.0
    rows = []
    for i, name in enumerate(names):
        if name in done:
            continue
        t0 = time.time()
        m_on = runner.measure(name, on_state)
        m_off = runner.measure(name, off_state)
        if m_on.error or m_off.error:
            log(f"[dataset] {name}: error ({(m_on.error or m_off.error).splitlines()[0][:100]})")
            continue
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
        row = {
            "family": family, "rule_family": spec["family"], "model": name, "env": runner.env, "session": runner.cfg.session_tag,
            "x86": int("x86" in runner.env), "threads": runner.threads,
            **{f"cfg_{k}": v for k, v in sweeps.config_of(name).items()},
            "ms_on": m_on.median_ms, "ms_off": m_off.median_ms, "ratio": ratios[0],
            "ratios": ratios, "tau_pct": tau_pct, "label": label, "confirmed": confirmed,
            "program_changed": program_changed, "fired": fired,
            "kernels_on": m_on.kernel_count, "kernels_off": m_off.kernel_count,
            "alloc_on": m_on.code.get("alloc_bytes"), "alloc_off": m_off.code.get("alloc_bytes"),
            "extern_on": m_on.code.get("extern_call_count"), "extern_off": m_off.code.get("extern_call_count"),
            "firing_features": feats,
        }
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        log(f"[dataset] {i+1}/{len(names)} {name}: on {m_on.median_ms:.3f} off {m_off.median_ms:.3f} ratio {ratios[0]:.3f} "
            f"-> {label}{' (confirmed)' if confirmed else ''} [{time.time()-t0:.0f}s]")
    return rows


def load(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows
