"""Run one measurement in a fresh subprocess and return it to the parent as JSON."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .measure import Measurement
from .switches import Registry

SENTINEL = "@@REWRITE_BLAME_RESULT@@"


def _child_main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--state-file", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--time-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from . import env
    threads = a.threads or env.default_threads()
    env.setup_threads(threads)
    from .discover import discover
    from .measure import measure
    reg = discover()
    saved = Registry.load(a.registry)
    missing = sorted(set(saved.ids()) - set(reg.ids()))
    extra = sorted(set(reg.ids()) - set(saved.ids()))
    state = json.loads(Path(a.state_file).read_text())
    m = measure(reg, a.model, state, threads=threads, warmup=a.warmup, rounds=a.rounds, iters=a.iters,
                time_it=not a.no_time, check_correct=not a.no_check, time_eager=a.time_eager, seed=a.seed)
    d = m.to_dict()
    d["registry_drift"] = {"missing_in_process": missing, "extra_in_process": extra}
    sys.stdout.write(SENTINEL + json.dumps(d) + SENTINEL + "\n")
    sys.stdout.flush()
    return 0


def run(model: str, state: Iterable[str], registry_path: str | Path, *, threads: int = 0, warmup: int = 10,
        rounds: int = 5, iters: int = 10, time_it: bool = True, check_correct: bool = True,
        time_eager: bool = False, seed: int = 0, timeout_s: int = 3600, cache_dir: str | None = None,
        python: str | None = None, extra_env: dict | None = None) -> Measurement:
    from . import env
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sorted(state), f)
        state_file = f.name
    cmd = [python or sys.executable, "-m", "rewrite_blame.worker", "--model", model, "--state-file", state_file,
           "--registry", str(registry_path), "--threads", str(threads), "--warmup", str(warmup),
           "--rounds", str(rounds), "--iters", str(iters), "--seed", str(seed)]
    if not time_it:
        cmd.append("--no-time")
    if not check_correct:
        cmd.append("--no-check")
    if time_eager:
        cmd.append("--time-eager")
    e = dict(os.environ)
    e["TORCHINDUCTOR_CACHE_DIR"] = env.inductor_cache_dir(cache_dir)
    e.setdefault("TOKENIZERS_PARALLELISM", "false")
    if threads:
        e["OMP_NUM_THREADS"] = str(threads)
    e.update(extra_env or {})
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=e)
    finally:
        try:
            os.unlink(state_file)
        except OSError:
            pass
    out = p.stdout
    if SENTINEL in out:
        payload = out.split(SENTINEL)[1]
        d = json.loads(payload)
        drift = d.pop("registry_drift", None)
        m = Measurement.from_dict(d)
        if drift and drift["missing_in_process"]:
            m.error = (m.error or "") + f"\nregistry names switches this process does not have: {drift['missing_in_process'][:5]}"
        elif drift and drift["extra_in_process"]:
            # the process knows more rules than the saved registry (e.g. platform-specific oneDNN
            # patterns): not fatal, but the universe used by the parent is incomplete -> recorded
            m.protocol["registry_drift_extra"] = drift["extra_in_process"]
        return m
    from . import env as _env
    import time
    from .switches import state_hash
    m = Measurement(model=model, state=sorted(state), state_hash=state_hash(state), env="?", threads=threads,
                    torch_version="?", timestamp=time.time())
    m.error = f"worker failed (rc={p.returncode}).\nSTDOUT tail:\n{p.stdout[-1500:]}\nSTDERR tail:\n{p.stderr[-3000:]}"
    return m


if __name__ == "__main__":
    sys.exit(_child_main(sys.argv[1:]))
