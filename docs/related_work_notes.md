# Related work notes (literature pass, 2026-10-20) — source material for the paper's related-work section

## Positioning (one paragraph)
Prior data-driven compilation either (i) replaces one hand-written heuristic inside one pass with a learned policy on
one target (MLGO's inliner/regalloc; Cummins et al. PACT'17 device mapping and thread coarsening; Inductor's in-tree
AutoHeuristic for `pad_mm`/mm ranking), (ii) learns a cost model to drive search over kernel schedules or candidate
graph substitutions (AutoTVM/Ansor/TenSet/TLP; Kaufman et al. MLSys'21 for XLA; TASO/PET/Tensat/X-RLflow/MCTS
e-graphs PACT'24), or (iii) diffs a compiler's recorded decisions to explain a regression after the fact (Pečimúth et
al. MPLR'23; "The Unseen Delta" ISSTA'26). The only prior per-optimization ablation of TorchInductor is Table 4 of the
PyTorch 2 paper (ASPLOS'24): eight coarse knobs, one A100, one suite, aggregate geomean, no noise model, run with
freezing ON. The only prior statement that pass effectiveness is "model- and device-dependent" is one sentence in the
Relay paper (MAPL'19) over four cumulative passes on two devices. We are the first to (1) make a production ML
compiler's ~300 rewrite rules independently switchable without modifying it, (2) attribute a measured slowdown to the
responsible rule(s) by delta debugging against a noise-calibrated oracle (counterfactual blame, not decision-log
diffing), (3) show at rule granularity across six CPU/GPU/precision targets that half of the most impactful rules
change sign while making the identical structural change, and (4) close the loop with learned per-rule guards and a
default-change finding (`freezing`) backed by 163 networks per GPU.

## Must-cite list by section

### Learned heuristics / autotuning
- MLGO (Trofin et al., arXiv 2021, in LLVM) — RL inlining-for-size and regalloc eviction.
- DeepTune / "End-to-End Deep Learning of Optimization Heuristics" (Cummins et al., PACT 2017) — learned binary heuristics (device mapping, coarsening).
- Magni, Dubach, O'Boyle (PACT 2014) — thread coarsening helps on some GPUs, hurts on others.
- AutoPhase (MLSys 2020), Liang et al. ICML 2023 (coreset pass ordering), CompilerGym (CGO 2022), Compiler-R1 (NeurIPS 2025), LLM compiler (Cummins 2023/24) — pass ordering/selection for size.
- Milepost GCC (IJPP 2011), COBAYN (TACO 2016), OpenTuner (PACT 2014), Pallister et al. (Comp. J. 2013: "no single optimisation is universally beneficial" across 5 embedded platforms).
- AutoTVM (NeurIPS 2018), Ansor (OSDI 2020), TenSet (NeurIPS 2021 D&B), TLP (ASPLOS 2023: cost models degrade across hardware), TCL (2026), Kaufman et al. TPU cost model (MLSys 2021), TpuGraphs (NeurIPS 2023), Halide autoscheduler (SIGGRAPH 2019).
- GO (NeurIPS 2020) and Learning to Fuse (2019) — learned fusion/placement decisions in a research fork.
- TASO (SOSP 2019), PET (OSDI 2021), Tensat (MLSys 2021), MCTS + equality saturation (PACT 2024), X-RLflow (MLSys 2023), Mirage (OSDI 2025) — rewrite search with cost models.
- MLIR Transform dialect (2024) — per-transformation control for search.
- Inductor AutoHeuristic (`torch/_inductor/autoheuristic/`, no paper) — regression-tree heuristics generated per GPU for 1–2 decisions.

### Attribution / localization
- Pečimúth, Leopoldseder, Tůma, "Diagnosing Compiler Performance by Comparing Optimization Decisions" (MPLR 2023).
- Yi, Ding, Shi, Gligoric, "Understanding and Finding JIT Compiler Performance Bugs" (OOPSLA 2026).
- Liu, Wang, Wang, "The Unseen Delta" (ISSTA 2026).
- Bisection effectiveness for performance regressions (EMSE 2022) — noisy oracles make naive bisection unreliable.
- BugLens (2025), MultiConf (TOSEM 2026), LLVM `-opt-bisect-limit`, FLiT Bisect (SC 2019) — ddmin/bisection over passes for miscompiles or numerical variability.
- Coz causal profiling (SOSP 2015); ServiceLab (OSDI 2024), FBDetect (SOSP 2024) — regression detection + bisection over code changes.
- Zeller & Hildebrandt ddmin (TSE 2002) — the algorithm.

### Hardware dependence of optimization benefit
- Relay (MAPL 2019): "The effectiveness of each pass is both model- and device-dependent."
- PyTorch 2 (ASPLOS 2024) Table 4 (ablation on A100; freezing row −0.06/−0.00; experiments run with freezing enabled).
- Channels-last on CPU (PyTorch/Intel blog), pytorch issue #50036 (NHWC conv slower on TITAN RTX), MosaicML channels-last card.
- TLP/ATFormer/TCL on cross-hardware cost-model transfer.
- Sinha et al., "Not All GPUs Are Created Equal" (SC 2022) — same-SKU variability.
- "Launch-Bound and Substitutable" (2026): removing graph breaks made an MoE model 3x slower; Snider & Liang on XLA fusion rules; edge inference strategy benchmarks (2026).

### Characterizations of ML compilers
- PyTorch 2 (ASPLOS 2024), TorchBench (arXiv 2023), Carmona-Martínez et al. (J. Supercomputing 2026: torch.compile cannot accelerate LLMs consistently), ParaDnn (MLSys 2020), GraphMend (2025), DL compiler bug studies (FSE 2021; 2026 frontend/silent-bug studies). No IISWC/ISPASS torch.compile characterization paper exists (state this).

### Measurement methodology
- Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems" (SC 2015); Mytkowicz et al. "Producing Wrong Data..." (ASPLOS 2009); Stabilizer (ASPLOS 2013); Georges et al. (OOPSLA 2007); Kalibera & Jones (ISMM 2013); Barrett et al. "Virtual Machine Warmup Blows Hot and Cold" (OOPSLA 2017); Chen & Revels robust benchmarking (2016); "Benchmarking Crimes"/SoK (EuroS&P 2019); MLSys 2021 "Accounting for Variance in ML Benchmarks"; `triton.testing.do_bench`; ISPASS 2025 "Use Equal-Work or Equal-Time Speedup, Not Geomean Speedup".

## The three objections to pre-empt, with the answer
1. "PyTorch 2 already ablates Inductor (Table 4)." — 8 knobs vs ~300 rules; one A100 vs six targets; geomean vs per-model/per-rule confirmed verdicts; no noise floor vs calibrated oracle; their runs had freezing ON, so its per-model effect across GPUs was never measured. Show that at their granularity the sign flips average away.
2. "Learned compiler heuristics are old news." — The contribution is the measurement/attribution substrate; guards are a downstream consumer. Decision type (rewrite presence vs schedule), scale (~300 rules), training signal (measured end-to-end deltas), external to the compiler, explicit cross-hardware sign result. Cite AutoHeuristic explicitly.
3. "ddmin over passes / decision-diffing already exists." — Boolean oracles for miscompiles vs a statistical oracle for slowdowns; diffing is correlational, we intervene; EMSE'22 shows naive performance bisection is unreliable; include the uncalibrated-bisection control experiment (E2) showing spurious blame at measured noise levels.
Secondary: "hardware-dependent benefit is known" — agree and cite; the contribution is quantifying it at rule granularity in the compiler most people use, naming the rules.
