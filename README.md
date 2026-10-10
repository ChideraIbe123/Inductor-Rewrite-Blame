# inductor-rewrite-blame

Which graph rewrite made my `torch.compile` model slower?

TorchInductor applies hundreds of small graph rewrites before generating kernels. Each one looks
like a win on its own, but some make a model slower. This tool switches each rewrite on or off,
measures the result, and blames the rewrite (or pair of rewrites) behind a slowdown.

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch==2.14.1 torchvision transformers pytest numpy matplotlib tabulate
.venv/bin/python -m rewrite_blame discover     # find the rewrite switches
make test-fast                                   # run the tests
```

## Usage

```bash
# list the models
python -m rewrite_blame list-models

# how noisy is the baseline? (7 independent processes -> threshold tau)
python -m rewrite_blame noise --model resnet18

# which switches change the compiled program, and by how much?
python -m rewrite_blame verify --model resnet18
python -m rewrite_blame sweep  --model resnet18

# blame: which of the changes between the fast and the slow setup caused the slowdown?
python -m rewrite_blame attribute     --model attention_block --fast default --slow "default-*_sfdp_pattern_*_inference"
python -m rewrite_blame attribute-all --model resnet18 --fast default --slow "default-lowering/layout_optimization-sched/inplace_buffers"   # every independent cause
python -m rewrite_blame tau-scan      --model attention_block --fast default --slow "default-*_sfdp_pattern_*_inference"  # are the culprits stable across thresholds?

# pairs: which switches hurt more together than alone?
python -m rewrite_blame interactions --model decode_mlp

# do verdicts hold across days and machines?
python -m rewrite_blame repeat --model resnet18 --extra-store results/vm/measurements.sqlite
```

A state is written as `default`, `all` or `none` followed by `+switch` or `-switch` terms
(globs allowed, `~family` toggles a whole family). Results land in `results/` as JSON and Markdown.

How "slower" is decided (see `docs/noise_method.md` for the evidence): run medians over the
steady-state calls; a reference of 7 processes with tau = max(3 MAD, 1%); a slow verdict needs
two time-separated candidate runs that both exceed tau, with a baseline run in between to catch
disturbed machine time; configurations whose generated program is byte-identical are never
blamed; timings are only compared within one session (`--session`, default: today's date).

## Layout

```
rewrite_blame/   switches, discovery, measurement, attribution, CLI
tests/           unit tests (test_u_*) and Inductor tests (test_i_*)
scripts/         helpers for running sweeps on a remote machine and building reports
```

Everything runs on Inductor's CPU backend; the code parser also understands Triton output.
