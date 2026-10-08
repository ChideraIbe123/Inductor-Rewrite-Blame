"""Environment setup and fingerprinting (so results from different machines never mix)."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys


def cpu_brand() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown-cpu"


def default_threads() -> int:
    env = os.environ.get("REWRITE_BLAME_THREADS")
    if env:
        return int(env)
    n = os.cpu_count() or 4
    # leave headroom for the OS/compiler on big machines; use everything on small ones
    return min(n, 8) if n > 4 else n


def setup_threads(threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import torch
    torch.set_num_threads(threads)


def fingerprint(threads: int) -> str:
    import torch
    host = socket.gethostname().split(".")[0]
    return f"{host}|{platform.machine()}|torch{torch.__version__}|T{threads}"


def describe(threads: int) -> dict:
    import torch
    return {
        "fingerprint": fingerprint(threads),
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "cpu": cpu_brand(),
        "cpu_count": os.cpu_count(),
        "threads": threads,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def inductor_cache_dir(root: str | None = None) -> str:
    d = root or os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(os.getcwd(), ".inductor_cache")
    os.makedirs(d, exist_ok=True)
    return d
