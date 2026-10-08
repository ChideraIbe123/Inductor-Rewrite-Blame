# rewrite-blame: which TorchInductor graph rewrite made my model slower?

A measurement harness and an attribution tool for TorchInductor (the `torch.compile` backend).
The harness switches Inductor's graph rewrites on and off, one rule at a time or in sets, compiles
a model under each configuration, records the generated kernels and the run time, and the
attribution tool uses delta debugging over the rule set to name the rule (or pair of rules) that
reproduces a measured slowdown.

Status: midterm (Oct 2026). Everything runs on Inductor's **CPU C++/OpenMP backend** (no GPU was
available); the code parses Triton output too, so a GPU run is a config change.

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch==2.14.1 torchvision transformers pytest numpy matplotlib tabulate
# Linux CPU wheels: add  --index-url https://download.pytorch.org/whl/cpu  to the torch/torchvision install
.venv/bin/python -m rewrite_blame discover          # enumerate rewrite switches -> results/switches.json
make test-fast                                        # unit + small Inductor tests
```

## Concepts

* **Switch** – an on/off handle over one rewrite. Three kinds: a `config` flag
  (`sched/epilogue_fusion`), an opt-in `option` of Inductor's fusion-option dicts
  (`optimus/decompose_mm_pass`), or a `pattern` – one entry of a `PatternMatcherPass`
  (`joint/joint_graph.patterns/_sfdp_pattern_11_inference`). Pattern switches work by wrapping the
  entry's `extra_check`, so a disabled rule is matched but refuses to fire; the harness records
  how often each rule fired or was suppressed.
* **State** – the set of switches that are ON. `default` is Inductor out of the box.
* **Measurement** – one compile of one model under one state in a fresh subprocess: generated-code
  statistics (fused kernels, extern calls, intermediate allocations, loads/stores), Inductor
  metrics, rule firings, post-grad op histogram, correctness against eager, and timing
  (warm-up + repeated trials). Stored in `results/measurements.sqlite` keyed by model, machine,
  state hash and timing protocol so nothing is measured twice.
* **Noise model** – the baseline state is measured in `k` separate processes; the slowdown
  threshold is `max(3 * MAD, 1%)` of the per-run medians.
* **Attribution** – given a fast state and a slow state, ddmin over the signed differences
  (`+id` turn on, `-id` turn off) with a timing (or compile-metric) judge; returns a 1-minimal
  culprit set, checks whether the culprits hurt alone or only together, and prints the
  kernel-level diff that explains the slowdown.

## Commands

```bash
python -m rewrite_blame list-models
python -m rewrite_blame measure   --model attention_block --state "default-*_sfdp_pattern_*"
python -m rewrite_blame noise     --model resnet18 --runs 7
python -m rewrite_blame verify    --model norm_mlp              # which switches change the graph
python -m rewrite_blame sweep     --model resnet18              # leave-one-out timings vs noise
python -m rewrite_blame attribute --model attention_block --fast default --slow "default-*_sfdp_pattern_*_inference"
python -m rewrite_blame attribute --model decode_mlp --fast default --slow default+optimus/decompose_mm_pass --judge timing
```

State specs: `default|all|none` followed by `+id`, `-id`, `~family` or glob terms such as
`-*_sfdp_pattern_*`.

## Layout

```
rewrite_blame/switches.py      switch data model, registry, built-in config/option switches
rewrite_blame/discover.py      warm-up compile, enumerate PatternMatcherPass entries, install gates
rewrite_blame/apply.py         apply a state (config.patch + gates + cache-key pass)
rewrite_blame/models/          corpus: toy graphs, torchvision, HuggingFace (random init)
rewrite_blame/measure.py       compile + capture + time one configuration
rewrite_blame/worker.py        subprocess isolation
rewrite_blame/codegen_stats.py parse generated wrapper code (C++ and Triton)
rewrite_blame/stats.py         medians, MAD, noise threshold
rewrite_blame/store.py         sqlite result cache
rewrite_blame/judge.py         timing / metric / synthetic judges
rewrite_blame/attribute.py     ddmin + pair check
rewrite_blame/pipeline.py      Runner: noise, verify, sweep, attribute
rewrite_blame/report.py        Markdown reports
tests/test_u_*.py              unit tests (no compile)      tests/test_i_*.py  Inductor tests
scripts/                       VM sync + tmux runners        report/midterm/     LaTeX report
```
