# Which Rewrite Made My Model Slower?

Finding and blaming hardware-dependent graph rewrites in TorchInductor (CS598APE final project).

`torch.compile` (TorchInductor) applies hundreds of small graph rewrites before it generates kernels. Most of them
help, but some make a model slower, and which ones depends on the machine. This tool:

1. adds an on/off **switch** to every rewrite rule, without modifying PyTorch;
2. measures a model with switches on and off, using a **noise-calibrated** test for "slower";
3. **blames** the rule (or pair of rules) behind a slowdown with delta debugging;
4. trains small **guards** that turn a rule off on the shapes where it hurts.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- macOS (Apple Silicon) or Linux x86-64. Everything runs on the CPU by default.
- An NVIDIA GPU with CUDA is needed only for the GPU experiments.

## Quick start

```bash
git clone https://github.com/ChideraIbe123/Inductor-Rewrite-Blame
cd Inductor-Rewrite-Blame
make setup        # creates .venv, installs PyTorch 2.14.1 and the other packages, finds the rewrite switches
make test-fast    # 171 tests, about 3 minutes on a laptop
```

Without `make`/`uv`, the same thing with plain pip:

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.14.1 torchvision transformers scikit-learn numpy matplotlib tabulate pytest
.venv/bin/python -m rewrite_blame discover
.venv/bin/python -m pytest -m "unit or inductor" -q
```

## Try it (CPU, a few minutes)

```bash
source .venv/bin/activate

# the built-in models
python -m rewrite_blame list-models

# which switches change the compiled program of a small attention block? (about 2.5 min)
python -m rewrite_blame verify --model attention_block

# blame: turning off the attention-fusion rules makes this model slower; which rules matter? (about 3 min)
python -m rewrite_blame attribute --model attention_block --fast default --slow "default-*_sfdp_pattern_*_inference"
```

Each command prints a table and writes JSON and Markdown to `results/`. The `attribute` run finds the attention
rules behind the slowdown; Appendix A of the report walks through this same example.

## All commands

```bash
python -m rewrite_blame noise     --model resnet18   # noise of the baseline: 7 processes -> threshold tau
python -m rewrite_blame verify    --model resnet18   # which switches change the generated code
python -m rewrite_blame sweep     --model resnet18   # time each switch that changes the code
python -m rewrite_blame attribute --model M --fast STATE --slow STATE   # rules behind fast -> slow
python -m rewrite_blame attribute-all --model M --fast STATE --slow STATE   # repeat until nothing is left
python -m rewrite_blame interactions  --model decode_mlp   # pairs that hurt more together than alone
python -m rewrite_blame tau-scan  --model M --fast STATE --slow STATE   # are culprits stable across thresholds?
python -m rewrite_blame repeat    --model resnet18   # do verdicts hold across sessions/machines?
python -m rewrite_blame dataset       # labelled rule on/off rows (guard training data)
python -m rewrite_blame guards-train  # train per-rule guards on that data
python -m rewrite_blame guards-eval   # stock compiler vs guards on real models
python -m rewrite_blame summary                      # overview of all result files
python -m rewrite_blame <command> --help             # options for any command
```

**States.** A state is `default`, `all` or `none`, followed by `+switch` or `-switch` terms. Globs are allowed,
and `~family` toggles a whole family. Example: `default-lowering/layout_optimization`.

**How "slower" is decided** (details and evidence in `docs/noise_method.md`): the median over steady-state
calls per run; a reference of 7 separate processes gives the threshold tau = max(3 MAD, 1%); a slowdown must
show up in two runs taken at different times; two setups whose generated code is identical are never blamed.

**GPU.** Set the device and precision with environment variables:

```bash
REWRITE_BLAME_DEVICE=cuda REWRITE_BLAME_DTYPE=bf16 python -m rewrite_blame sweep --model resnet18
```

## Repository layout

```
rewrite_blame/   the package: switches, discovery, measurement, noise model, attribution, guards, CLI
tests/           unit tests (test_u_*, no compiling) and Inductor tests (test_i_*, compile small graphs)
scripts/         analysis and table/figure scripts, remote-machine and GPU-pod helpers
scripts/runs/    the exact run recipes behind each result (kept for traceability, see its README)
evidence/        per-network result files behind the report's tables (see evidence/README.md)
docs/            method notes: noise method, sign matrix, mechanism, results notes
```

## Main results

Measured on six targets (Apple M4 CPU, Intel Xeon CPU, A40 fp32, and A40 / L4 / RTX 4090 in bf16):

- 7 to 8 of the 16 most impactful switches are faster on some machines and slower on others, while making the
  same change to the generated code everywhere.
- The calibrated test lowers false slowdown reports from 8.9% to 1.5%.
- `torch._inductor.config.freezing` makes inference faster on 105 to 113 of 163 real networks per GPU, but is
  off by default (reported upstream as pytorch/pytorch#201502).

Full numbers are in the final report and in `docs/results_notes.md`.
