"""Static statistics over Inductor's generated wrapper code.

Works on the Python "output_code" strings Inductor emits for both the C++/OpenMP backend
(``cpp_fused_* = async_compile.cpp_pybinding(...)``) and the Triton backend
(``triton_* = async_compile.triton(...)``), so the same counters serve a CPU run today and a
GPU run later. No torch import.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field, asdict

_DTYPE_BYTES = {
    "torch.float64": 8, "torch.int64": 8, "torch.complex64": 8,
    "torch.float32": 4, "torch.int32": 4, "torch.float": 4, "torch.int": 4,
    "torch.float16": 2, "torch.bfloat16": 2, "torch.int16": 2, "torch.half": 2,
    "torch.int8": 1, "torch.uint8": 1, "torch.bool": 1,
    "torch.float8_e4m3fn": 1, "torch.float8_e5m2": 1,
}

# kernel definitions in the wrapper
_CPP_DEF = re.compile(r"^(?P<name>cpp_\w+)\s*=\s*async_compile\.cpp_pybinding\(", re.M)
_TRITON_DEF = re.compile(r"^(?P<name>triton_\w+)\s*=\s*async_compile\.triton\(", re.M)
_MPS_DEF = re.compile(r"^(?P<name>mps_lib_\w+)\s*=\s*compile_mps_shader\(", re.M)
# allocations of intermediates / outputs:  empty_strided_cpu((8, 16), (16, 1), torch.float32)
_ALLOC = re.compile(r"empty_strided_(?P<dev>cpu|cuda|xpu|mps|mtia)\w*\(\((?P<shape>[^)]*)\),\s*\((?P<stride>[^)]*)\),\s*(?P<dtype>torch\.\w+)")
# extern / fallback kernels invoked from the wrapper's call()
_EXTERN = re.compile(r"\bextern_kernels\.(\w+)\(")
_FALLBACK = re.compile(r"\btorch\.ops\.(\w+\.\w+)(?:\.\w+)?\(")
_ATEN_CALL = re.compile(r"(?<![\w.])aten\.(\w+(?:\.\w+)?)\(")


@dataclass
class KernelInfo:
    name: str
    backend: str  # cpp | triton | mps
    category: str  # pointwise | reduction | template | unknown
    n_in_ptr: int = 0
    n_out_ptr: int = 0
    n_inout_ptr: int = 0
    n_loads: int = 0
    n_stores: int = 0
    vectorized: bool = False
    n_omp_parallel: int = 0
    n_calls: int = 0
    lines: int = 0


@dataclass
class CodeStats:
    n_sources: int = 0
    kernel_count: int = 0
    kernel_calls: int = 0
    kernels: list[KernelInfo] = field(default_factory=list)
    extern_calls: dict[str, int] = field(default_factory=dict)
    extern_call_count: int = 0
    alloc_count: int = 0
    alloc_bytes: int = 0
    total_loads: int = 0
    total_stores: int = 0
    pointwise_kernels: int = 0
    reduction_kernels: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CodeStats":
        d = dict(d)
        d["kernels"] = [KernelInfo(**k) for k in d.get("kernels", [])]
        return cls(**d)

    def kernel_names(self) -> list[str]:
        return [k.name for k in self.kernels]


def _kernel_bodies(src: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Return (name, body) for each kernel definition; body runs to the next top-level statement."""
    out = []
    matches = list(pattern.finditer(src))
    for i, m in enumerate(matches):
        start = m.start()
        # body ends at the next kernel def or at the wrapper's "async_compile.wait" / "def call("
        end = len(src)
        nxt = re.compile(r"^(?:\w+\s*=\s*async_compile\.|async_compile\.wait|def call\(|mps_lib_\w+\s*=)", re.M)
        for n in nxt.finditer(src, m.end()):
            end = n.start()
            break
        out.append((m.group("name"), src[start:end]))
    return out


def _category(name: str, body: str, backend: str) -> str:
    if backend == "triton":
        if "triton_tem_" in name or "_template" in name:
            return "template"
        if name.startswith("triton_red_") or name.startswith("triton_per_"):
            return "reduction"
        if name.startswith("triton_poi_"):
            return "pointwise"
        return "unknown"
    # C++: reductions carry accumulators; templates come from cpp gemm templates
    if "cpp_fused_" not in name and ("gemm" in name or "template" in name):
        return "template"
    if "tmp_acc" in body or "reduction" in body or "_vec = at::vec::Vectorized" in body and "tmp_acc" in body:
        return "reduction"
    return "pointwise"


def analyze_source(src: str) -> CodeStats:
    stats = CodeStats(n_sources=1)
    found: list[tuple[str, str, str]] = []
    for name, body in _kernel_bodies(src, _CPP_DEF):
        found.append(("cpp", name, body))
    for name, body in _kernel_bodies(src, _TRITON_DEF):
        found.append(("triton", name, body))
    for name, body in _kernel_bodies(src, _MPS_DEF):
        found.append(("mps", name, body))

    # the wrapper's call() region (after the last kernel definition) tells us launches/externs
    call_idx = src.find("def call(")
    call_region = src[call_idx:] if call_idx >= 0 else src

    for backend, name, body in found:
        info = KernelInfo(name=name, backend=backend, category=_category(name, body, backend))
        info.lines = body.count("\n")
        if backend == "cpp":
            sig = re.search(r"kernel\((?P<args>[^)]*)\)", body, re.S)
            args = sig.group("args") if sig else ""
            info.n_in_ptr = len(re.findall(r"\bin_ptr\d+\b", args))
            info.n_out_ptr = len(re.findall(r"\bout_ptr\d+\b", args))
            info.n_inout_ptr = len(re.findall(r"\bin_out_ptr\d+\b", args))
            after_sig = body[sig.end():] if sig else body
            info.n_loads = len(re.findall(r"::loadu\(|\bin_ptr\d+\[|\bin_out_ptr\d+\[(?!.*=\s*tmp)", after_sig))
            info.n_stores = len(re.findall(r"\.store\(|\bout_ptr\d+\[[^\]]*\]\s*=|\bin_out_ptr\d+\[[^\]]*\]\s*=", after_sig))
            info.vectorized = "at::vec::Vectorized" in body
            info.n_omp_parallel = body.count("#pragma omp parallel")
        elif backend == "triton":
            sig = re.search(r"def\s+\w+\((?P<args>[^)]*)\)", body, re.S)
            args = sig.group("args") if sig else ""
            info.n_in_ptr = len(re.findall(r"\bin_ptr\d+\b", args))
            info.n_out_ptr = len(re.findall(r"\bout_ptr\d+\b", args))
            info.n_inout_ptr = len(re.findall(r"\bin_out_ptr\d+\b", args))
            info.n_loads = len(re.findall(r"\btl\.load\(", body))
            info.n_stores = len(re.findall(r"\btl\.store\(", body))
        info.n_calls = len(re.findall(r"(?<![\w.])%s\s*\(" % re.escape(name), call_region))
        # cpp kernels are invoked as name(args) too; the `.run(` form is Triton's launch
        info.n_calls += len(re.findall(r"(?<![\w.])%s\.run\(" % re.escape(name), call_region))
        stats.kernels.append(info)

    stats.kernel_count = len(stats.kernels)
    stats.kernel_calls = sum(k.n_calls for k in stats.kernels)
    stats.total_loads = sum(k.n_loads for k in stats.kernels)
    stats.total_stores = sum(k.n_stores for k in stats.kernels)
    stats.pointwise_kernels = sum(k.category == "pointwise" for k in stats.kernels)
    stats.reduction_kernels = sum(k.category == "reduction" for k in stats.kernels)

    externs: Counter = Counter()
    for m in _EXTERN.finditer(call_region):
        externs[f"extern_kernels.{m.group(1)}"] += 1
    for m in _FALLBACK.finditer(call_region):
        externs[f"torch.ops.{m.group(1)}"] += 1
    for m in _ATEN_CALL.finditer(call_region):
        externs[f"aten.{m.group(1)}"] += 1
    stats.extern_calls = dict(sorted(externs.items()))
    stats.extern_call_count = sum(externs.values())

    for m in _ALLOC.finditer(call_region):
        stats.alloc_count += 1
        shape = [s.strip() for s in m.group("shape").split(",") if s.strip()]
        numel = 1
        ok = True
        for s in shape:
            try:
                numel *= int(s)
            except ValueError:  # symbolic shape
                ok = False
                break
        if ok:
            stats.alloc_bytes += numel * _DTYPE_BYTES.get(m.group("dtype"), 4)
    return stats


def analyze_sources(sources: list[str]) -> CodeStats:
    """Aggregate over all captured sources (Inductor emits one per compiled graph)."""
    total = CodeStats()
    externs: Counter = Counter()
    for src in sources:
        s = analyze_source(src)
        total.n_sources += 1
        total.kernel_count += s.kernel_count
        total.kernel_calls += s.kernel_calls
        total.kernels.extend(s.kernels)
        externs.update(s.extern_calls)
        total.extern_call_count += s.extern_call_count
        total.alloc_count += s.alloc_count
        total.alloc_bytes += s.alloc_bytes
        total.total_loads += s.total_loads
        total.total_stores += s.total_stores
        total.pointwise_kernels += s.pointwise_kernels
        total.reduction_kernels += s.reduction_kernels
    total.extern_calls = dict(sorted(externs.items()))
    return total


def diff_stats(a: CodeStats, b: CodeStats) -> dict:
    """Kernel-level difference between two compilations (b relative to a)."""
    def sig(k: KernelInfo) -> str:
        # strip the trailing numeric suffix so identical fusions in different positions compare equal
        return re.sub(r"_\d+$", "", k.name)

    ca = Counter(sig(k) for k in a.kernels)
    cb = Counter(sig(k) for k in b.kernels)
    return {
        "kernel_count": (a.kernel_count, b.kernel_count),
        "kernel_calls": (a.kernel_calls, b.kernel_calls),
        "extern_call_count": (a.extern_call_count, b.extern_call_count),
        "alloc_count": (a.alloc_count, b.alloc_count),
        "alloc_bytes": (a.alloc_bytes, b.alloc_bytes),
        "total_loads": (a.total_loads, b.total_loads),
        "total_stores": (a.total_stores, b.total_stores),
        "kernels_only_in_a": sorted((ca - cb).elements()),
        "kernels_only_in_b": sorted((cb - ca).elements()),
        "externs_only_in_a": sorted((Counter(a.extern_calls) - Counter(b.extern_calls)).elements()),
        "externs_only_in_b": sorted((Counter(b.extern_calls) - Counter(a.extern_calls)).elements()),
    }


_VOLATILE = [
    re.compile(r"^#\s*AOT ID:.*$", re.M),                 # per-process compile counter
    re.compile(r"/(?:tmp|private|var|Users|home)/[^\s'\"]*"),   # temp / cache paths
    re.compile(r"\b[0-9a-f]{32,}\b"),                    # cache keys
]


def normalize_source(src: str) -> str:
    for rx in _VOLATILE:
        src = rx.sub("", src)
    return src


def program_hash(sources: list[str]) -> str:
    """Hash of the generated program with volatile tokens removed. Two configurations with the
    same program hash compiled to byte-identical kernels and wrapper, so any timing difference
    between them is noise by construction."""
    h = hashlib.sha256()
    for src in sources:
        h.update(normalize_source(src).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]
