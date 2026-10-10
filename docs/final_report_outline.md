# Final report outline (due Nov 19)

Working title: Which Rewrite Made My Model Slower? Finding and Blaming Bad Graph Rewrites in TorchInductor.

## 1. Problem and thesis
- Inductor's rewrites are local decisions; their value depends on what the kernel generator and the
  hardware do with the result; nothing in Inductor checks. Engineers find regressions by hand.
- Thesis: with per-rule switches, honest timing and a search over switch sets, a measured slowdown
  can be attributed automatically to the rule or rules that cause it, including pairs.

## 2. The switch mechanism (item 1)
- PatternMatcherPass / PatternEntry.extra_check gate; discovery via warm-up compile; 269 switches on
  the Mac build, 396 on the x86 build (oneDNN rules); config/option/master switches; fire and
  suppression counts; registry-independent state diffs; the FX-graph cache and lazy-init pitfalls.
- Evidence: verification tables (which switches change the compiled program); program hash.

## 3. Measurement (item 2) and the noise study
- Harness design (subprocess per configuration, generated-code statistics, correctness check, store).
- docs/noise_method.md: skew and outliers; steady-state trim measured; null calibration of ten
  detectors; fat-tailed between-process noise; AND rule over time-separated runs; interleaved
  baselines; identical-program guard; sessions. Tables from scripts/noise_study_*.py.

## 4. Attribution (item 5)
- Signed change tokens, ddmin, 1-minimality, pair check, iterative attribution for all causes,
  tau-scan for stability, pair scan for super-additive pairs, kernel-level explanation.
- Case studies: attention block (pair on Mac; single + opposite sign on VM), decode MLP
  (decompose_mm among 16 opt-in passes), ResNet-18 (layout optimisation among 6 flags; both
  machines), BERT (rule 28: win on Mac, loss on VM), the unstable cat/slice case caught by tau-scan,
  the fake pair interactions caught by the program hash.

## 5. Large programs and repeatability (item 6)
- Full corpus sweeps on both machines (24 models incl. ViT-B/16 and Qwen2.5-0.5B); per-machine
  tables; `repeat` agreement counts across sessions and machines; what did and did not reproduce
  (e.g. norm_mlp Mac effects day to day; decode_mlp freezing on VM).

## 6. Guards (stretch, item 7)
- Labelled firings dataset (features at each firing + measured effect of disabling the rule);
  per-family stump guard with leave-one-model-out evaluation vs always-fire. Report honestly how
  small the dataset is; the attention family across machines is the motivating case.

## 7. Testing (items 3 and 6)
- Test inventory (unit vs Inductor-compiling), planted-rewrite graphs, synthetic ddmin tests,
  noise-study scripts as reproducible evidence.

## 8. Limitations and future work
- CPU backend only; no KV-cache decode; small corpus for guards; VM is a shared host; attribution
  power for 2-5% effects is a measurement-budget question.

## Data to still collect
- [ ] Mac corpus chain (running 2026-10-10): all 24 models, session 2026-10-10; toys re-measured clean.
- [ ] VM: corrected pair scan (running); then corpus chain for the remaining models + second sessions.
- [ ] `repeat` and `summary` regenerated after both; `guards` after sweeps with firings exist on both machines.
