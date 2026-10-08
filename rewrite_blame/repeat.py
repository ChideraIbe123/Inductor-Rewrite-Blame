"""Repeatability: do baseline speed and per-switch verdicts agree across sessions and machines?

Pure functions over stored measurement rows (dicts), so they are testable without torch.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Iterable

from .stats import median, noise_model


def session_of(row: dict) -> str:
    """Session tag of a timed row; rows from before session scoping get the date of their timestamp."""
    s = row.get("session")
    if s:
        return s
    pk = row.get("protocol_key") or ""
    if "@" in pk:
        return pk.split("@", 1)[1].split("/")[0]
    ts = row.get("timestamp") or 0
    return datetime.date.fromtimestamp(ts).isoformat() if ts else "unknown"


@dataclass
class RepeatReport:
    model: str
    baselines: list[dict]          # per (env, session): n runs, centre, tau
    switches: list[dict]           # per switch: per-(env, session) delta %, verdict, agreement
    agreement: dict                # counts of agree / disagree across sessions per machine, and across machines

    def to_dict(self) -> dict:
        return asdict(self)


def repeatability(rows: Iterable[dict], model: str, base_state_hash: str, toggles: dict[str, str],
                  k: float = 3.0, floor_frac: float = 0.01) -> RepeatReport:
    """``toggles`` maps a switch id to the state hash of "base with that switch toggled"."""
    rows = [r for r in rows if r.get("model") == model and r.get("timing") and not r.get("error")]
    # group medians by (env, session, state_hash)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["env"], session_of(r), r["state_hash"])].append(r["timing"]["median"])
    baselines, noise_by = [], {}
    for (env, sess, sh), meds in sorted(groups.items()):
        if sh != base_state_hash:
            continue
        if len(meds) >= 2:
            nm = noise_model(meds, k=k, floor_frac=floor_frac)
            centre, tau = nm.center, nm.tau
        else:
            centre, tau = meds[0], max(floor_frac * meds[0], 0.0)
        noise_by[(env, sess)] = (centre, tau)
        baselines.append({"env": env, "session": sess, "runs": len(meds), "centre_ms": centre, "tau_ms": tau,
                          "tau_pct": 100 * tau / centre})
    switches = []
    agree_same_machine = disagree_same_machine = agree_cross = disagree_cross = 0
    for sid, sh in sorted(toggles.items()):
        cells = []
        for (env, sess), (centre, tau) in sorted(noise_by.items()):
            meds = groups.get((env, sess, sh))
            if not meds:
                continue
            d = median(meds) - centre
            verdict = "slower" if d > tau else ("faster" if d < -tau else "same")
            cells.append({"env": env, "session": sess, "delta_pct": 100 * d / centre, "verdict": verdict})
        # agreement across sessions on the same machine, and across machines (any session)
        by_env: dict[str, set] = defaultdict(set)
        for c in cells:
            by_env[c["env"]].add(c["verdict"])
        same = [len(v) == 1 for v in by_env.values() if v]
        cross_verdicts = {next(iter(v)) for v in by_env.values() if len(v) == 1}
        n_env_with_cells = sum(1 for e, c in by_env.items() if c)
        row = {"switch": sid, "cells": cells,
               "same_machine_agree": all(same) if same else None,
               "cross_machine_agree": (len(cross_verdicts) == 1) if n_env_with_cells >= 2 else None}
        for ok in same:
            if ok:
                agree_same_machine += 1
            else:
                disagree_same_machine += 1
        if row["cross_machine_agree"] is True:
            agree_cross += 1
        elif row["cross_machine_agree"] is False:
            disagree_cross += 1
        switches.append(row)
    agreement = {"same_machine": {"agree": agree_same_machine, "disagree": disagree_same_machine},
                 "cross_machine": {"agree": agree_cross, "disagree": disagree_cross}}
    return RepeatReport(model, baselines, switches, agreement)


def render_repeat(rep: RepeatReport) -> str:
    lines = [f"# Repeatability: {rep.model}", "", "## Baselines per machine and session",
             "| machine | session | runs | centre ms | tau ms | tau % |", "|---|---|---|---|---|---|"]
    for b in rep.baselines:
        lines.append(f"| {b['env']} | {b['session']} | {b['runs']} | {b['centre_ms']:.3f} | {b['tau_ms']:.3f} | {b['tau_pct']:.1f} |")
    lines += ["", "## Per-switch verdicts (delta % vs that session's baseline)", "| switch | cells | same-machine agree | cross-machine agree |", "|---|---|---|---|"]
    for s in rep.switches:
        cells = "; ".join(f"{c['env'].split('|')[0].split(' ')[0]}/{c['session'][-5:]}: {c['delta_pct']:+.1f}% {c['verdict']}" for c in s["cells"])
        lines.append(f"| `{s['switch']}` | {cells} | {s['same_machine_agree']} | {s['cross_machine_agree']} |")
    a = rep.agreement
    lines += ["", f"Same machine, different sessions: {a['same_machine']['agree']} agree, {a['same_machine']['disagree']} disagree. "
                  f"Across machines: {a['cross_machine']['agree']} agree, {a['cross_machine']['disagree']} disagree."]
    return "\n".join(lines) + "\n"
