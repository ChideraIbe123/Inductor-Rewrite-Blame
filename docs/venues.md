# Venue research (2026-10-20) — for the paper submission

Decision: **ISPASS 2027** first (abstract Dec 11, paper Dec 18, 2026 AoE; 9 pages incl. figures, unlimited refs;
double-blind; rebuttal Feb 15–19; notification Mar 8, 2027; NYU, May 9–11, 2027; ~39% acceptance in 2026; explicit
tool/benchmark paper category and "confirmations or refutations of important prior results" topic; optional artifact
evaluation with IEEE badges; arXiv allowed, use a different title and notify the chair if posting after submission).
Fallbacks: **PACT 2027** (~mid/late April 2027, 10 pp, compiler venue, wants a technique; lead with the attribution
tool + guards), **IISWC 2027** (~mid/late May 2027, 10 pp, ~42%, characterization venue, tool/benchmark papers;
arXiv version must carry a different title), **ICPE 2027 Emerging Research Track** (Feb 1, 2027; short version),
EuroMLSys 2027 (~late Feb 2027, 6 pp, with EuroSys Apr 19 in Rabat), C4ML at CGO 2027 (~Jan 2027, 1–4 page
abstracts, no proceedings).

Too soon / poor fit: MLSys 2027 (Oct 30, 2026; rarely takes pure characterization), PLDI 2027 (Nov 12), OSDI 2027
(Dec 8; wants a system), ICS 2027 (Dec 7; HPC framing required), SIGMETRICS 2027 (Jan 11; wants a modeling
contribution), OOPSLA/ECOOP/CC (topical mismatch), ASPLOS 2028 (~Apr 2027 cycle, ~14.5%; the long shot if the second
PyTorch version and a second GPU class exist), CGO 2028 R1 (~June 2027; Tools & Practical Experience track requires
passing artifact evaluation).

## What "enough" looks like at these venues (accepted-paper calibration)
- Demystifying TensorRT (IISWC 2021): 12 CNNs, 2 Jetson platforms, 1 TensorRT version, 10 runs, mean + SD, numbered findings.
- NonGEMM Bench (ISPASS 2025): 17 models, workstation + datacenter, operator-level latency shares.
- Variability-Guided Performance Optimization (ICPE 2026, runner-up): ~12 apps, 4+ servers, n = 200–1000 repetitions, distributions.
- TorchBench (arXiv 2023): 84 models, A100 (+MI210), two PyTorch versions, median of 10 runs; 5 perf issues found upstream.
- PyTorch 2 (ASPLOS 2024): 180+ models, one A100 + one Xeon, 100 reps/3 warmups, geomean; Table 4 one-at-a-time ablation of 8 knobs, run with freezing ON.
Our 24 models x 6 machine/precision targets, calibrated noise oracle and 133 tests are within or above the envelope;
thin spots: one PyTorch version (fix: E1), mostly random weights (mitigated by the 12 pretrained twins; scope the claims).

## Reviewer expectations (SIGPLAN Empirical Evaluation Checklist, PLDI/ISPASS practice)
Report distributions and CIs, not point estimates; measure compile time and memory next to speed; scope claims
(inference, one version); eager and default `torch.compile` are the right baselines; declare limitations; versions and
hardware fully specified; artifact available (ISPASS tool papers must be open-sourced before the conference).
Not required: training workloads, other compilers (TensorRT/XLA/ORT), energy, significance tests (but variance yes).

## Double-blind and the public footprint
All candidate venues tolerate an arXiv preprint and public artifacts if the submission uses a different system name
and title, an anonymized repo, and does not cite the author-named GitHub issue (say "an upstream issue filed by the
authors, link withheld"). No promotion from two weeks before the deadline until notification.
