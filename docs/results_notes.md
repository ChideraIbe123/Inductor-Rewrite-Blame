# Results notes (running log for the final report)

Numbers are medians over steady-state calls; tau is the session's noise threshold. "Mac" = Apple
M4 Pro, 8 threads; "VM" = 4-core Xeon Silver 4216 with oneDNN. Sessions are dated; clean/cool
sessions were re-measured with cooldown pauses after a thermal-throttling episode on the Mac.

## Rewrites that hurt (default-on or opt-in), with the attribution that found them

| model | machine | finding | attribution |
|---|---|---|---|
| SmolLM-135M decode (1 token) | Mac | opt-in `decompose_mm_pass` +134% (6.6 -> 15.5 ms) | iterative, 16 opt-in passes on at once, 9 evaluations, single culprit, residual within noise |
| Qwen2.5-0.5B decode | Mac | `decompose_mm_pass` +39% | sweep |
| decode MLP toy | Mac | `decompose_mm_pass` +15% | single culprit among 16 (7 evaluations) |
| EfficientNet-B0 | Mac | channels-last `layout_optimization` +68% (325 -> 546 ms); fused conv+SiLU kernels split, intermediates 82 KB -> 2.5 MB | single culprit among 4 flags (5 evaluations) |
| MobileNet-V3-small | Mac | `layout_optimization` +41% (127 -> 180 ms); intermediates 112 KB -> 4.9 MB | single culprit (5 evaluations) |
| attention block | VM | attention fusion `_sfdp_pattern_11` +8% (fused CPU flash attention slower than bmm + fused softmax) | single culprit among 63 rules (8 evaluations) |
| BERT-base, eager attention | VM | `_sfdp_pattern_28` +2.4% | single culprit among 63 (11 evaluations, 41 min) |
| SmolLM-135M prefill | Mac | default-on `reorder_for_locality` costs 3.6% | sweep (cool session) |
| ResNet-50 | Mac | `freezing` +7% (no oneDNN on this build) | sweep; the iterative case with a helping flag added nets out within noise (see below) |
| cat/slice toy | VM | `batch_relu`+`batch_tanh` together +12.8% | iterative; verdict unstable across k (effect near the noise floor) |

## Rewrites whose loss hurts (the "upgrade silently stopped applying it" case)

| model | machine | finding | attribution |
|---|---|---|---|
| attention block | Mac | losing attention fusion +34%; needs both rules 11 and 12 off | interaction of 2 among 63 (16 evaluations); stable for k = 2..6 |
| ResNet-18 | Mac, VM | `layout_optimization` lost: +11% / +13% | single culprit among 6 flags; both machines agree |
| GPT-2 eager attention | Mac | `master/pattern_matcher` off +37% (throttled session; re-measured: within noise) | sweep |
| BERT-base eager | Mac | `_sfdp_pattern_28` off +4.5% | sweep |
| SmolLM prefill (both attention variants), T5-small | Mac | `max_fusion_size` cap removed: +3.4% / +6.9% / +1.3% | cool-session sweeps (the warm-session DistilBERT/BERT values of +9-11% did not survive re-measurement: nothing beyond noise at tau = 1%) |
| layer-norm MLP | Mac | `constant_folding` off +2.6% (day 1), within noise (day 2) | iterative; a non-repeatable marginal effect |

## Same rewrite, opposite sign

* Attention fusion: +34% win on the Mac, 8% loss on the VM (attention block); +4.5% win / 2.2% loss (BERT).
* `freezing`: -6% to -47% on the VM (oneDNN prepack), +1.5% to +7% on the Mac (no oneDNN).
* `layout_optimization`: +11% win on ResNet-18, 29-40% loss on MobileNet/EfficientNet (depthwise convolutions), on the same machine.
* `conv_1x1_as_mm` (opt-in): 3.5-5.6% win on EfficientNet/ResNet-50 (Mac), 1.8% on ResNet-50 (VM).

## Things the method caught rather than reported as findings

* Thermal throttling on the laptop: SmolLM prefill medians 37-149 ms within one session, GPT-2 toggles "+480%". Fixed by the unsteady-run detector and cooldown pauses; re-measured sessions give tau = 1-2%.
* Fake pair interactions on the VM (four "+70-150%" pairs): back-to-back confirmation runs during a disturbed stretch; fresh measurements identical to default; the generated program was byte-identical. Fixed by interleaved baselines and the program-hash guard.
* cat/slice on the VM: verdict changes with k (single at k=2, 8-way set at k=3, none at k>=4): tau-scan flags it as unstable.
* ResNet-50 iterative case: a hurting flag (+7%) and a helping flag (-5.6%) turned on together net to +2.2%, within noise, so the tool correctly reports nothing to blame; cancelling pairs are the pair scan's job, not ddmin's.
* Day-to-day: the layer-norm MLP's 3.6% slow state did not reproduce the next day; the `repeat` command reports such disagreements.

## Cool-session corrections (2026-10-11, Mac, tau = 1-2% everywhere, cooldown pauses on)

Re-measured after the thermal episode. Verdicts that survived: ResNet-18 layout +10.3%; BERT eager
rule 28 +4.7% (attributed as the single culprit among 63 rules in 11 evaluations); SmolLM decode
decompose_mm +134.9%; MobileNet/EfficientNet layout -29% / -40% when off; SmolLM prefill
reorder_for_locality -3.6% when off and max_fusion_size +3.4%; cat/slice toy: opt-in
normalization_pass +15.8%, pattern matcher off +17.8%. Verdicts retracted: DistilBERT (+9-11%
for max_fusion_size/inplace_buffers in the warm session; nothing beyond noise now, and the
iterative case over the same flags finds nothing to blame), GPT-2 eager "+480%" rows, BERT-base
max_fusion_size +2.5%. Lesson recorded in docs/noise_method.md: a baseline measured minutes
before the toggles is stale on a laptop; sweeps now re-baseline every eight toggles.

Mac corpus status: all 24 models measured; 15 attribution runs; `results/summary.md` is the overview.

## Pair scan, ResNet-50 (Mac, 2026-10-11, 8 switches, 28 pairs, 86 measurements)

* Robust finding: `conv_1x1_as_mm` on (-1.2 ms alone) and `layout_optimization` off (-1.1 ms
  alone) give -0.02 ms together (confirmed by a time-separated second run): the two gains are the
  same gain (both avoid channels-last 1x1 convolutions). Labelled "redundant gains".
* Provisional: `inplace_buffers` off + `max_fusion_size` cap removed: +4.2 ms together vs +0.4 and
  +0.6 alone ("compounding costs", confirmed by a second run at +5.3), but 18 of 33 interleaved
  baseline runs were disturbed during this scan (night-time machine activity), two pairs that
  compile to a byte-identical program still showed +1.7 and +3.6 ms, and several flags were
  dropped by the second run. The report now prints an instability warning when more than 30% of
  baseline runs are disturbed; this scan must be repeated in a quiet session before the pair is
  claimed.

## Repeatability (item 6), as of 2026-10-12

Sessions used: cool/clean Mac sessions and the VM's overnight sessions; excluded by name the warm
Mac session of 2026-10-10 and the pair-scan sessions, and automatically any session whose baseline
tau exceeds 5% (`repeat --exclude-session ... --max-tau-pct 5`).

* Same machine, different days: 213 of 220 per-switch verdicts agree (96.8%). All 7 disagreements
  are effects of 1-2.6%, i.e. at the noise threshold (norm_mlp constant folding and in-place
  buffers, ResNet-18 freezing at -1.1% vs -1.2%, three cat/slice switches at -2%).
* Across machines (Mac vs VM): 42 agree, 47 disagree. The disagreements are not noise: they are
  the opposite-sign cases (attention fusion, freezing, oneDNN-dependent passes) where the same
  rewrite genuinely helps on one CPU and hurts on the other.

Reading: once sessions are measured cleanly, the harness's verdicts are stable day to day; what
varies is the hardware's answer, which is the point of the project.

## VM Qwen2.5-0.5B prefill: a retraction the method produced itself (2026-10-12)

The VM sweep (single measurement per toggle, baseline measured once) showed the opt-in
`normalization_pass` at +217% (886 -> 2810 ms) and five unrelated default-on switches at
"-4.8% when off". Neither survived confirmation:

* iterative attribution with six opt-in passes on: slow state 847 ms vs default 847 ms, nothing to blame;
* confirmed single-switch check of `normalization_pass` alone: first run +5.9%, independent second
  run disagreed, verdict "not reproduced";
* "fast" state without node reordering and constant folding: 907 ms, i.e. 7% *slower* than
  default (so those two rewrites help on the VM; the sweep's baseline had been measured during a
  loaded minute).

Consequence: sweeps now confirm every flagged row with an independent second run (the same AND
rule as the judge) and mark rows whose second run disagrees as "unconfirmed". The Mac corpus rows
were cross-checked instead by their second sessions (96.8% agreement, see above).

ResNet-50 on the VM (same session): removing layout optimisation costs 16% (255 vs 220 ms), so on
the Xeon that rewrite helps even for this model; the `conv_1x1_as_mm` + `normalization_pass` state
is not slower than default.

## Guards, MLP family (decompose_mm on decode-shaped matmuls), Mac, 2026-10-12

Dataset: 161 configurations (M in 1..128, in 512..4096, hidden 1024..11008) with Inductor's CPU
shape thresholds lifted, each labelled by a confirmed on/off pair: 107 harmful, 50 helpful, 4
neutral. Decomposition is catastrophic at large M (up to 13x) and *helpful* (10-60% faster) for
many M <= 2 shapes with large hidden sizes, exactly the region Inductor's hard-coded rule
(M <= 1 and dims <= 2048) excludes, while that rule still lets through small-dimension M=1 cases
that cost 2-4x.

Grouped 5-fold cross-validation (held-out configuration slices):

| guard | accuracy | harmful caught | false alarms | cost %/firing |
|---|---|---|---|---|
| Inductor hand-written rule | 0.63 | 0.94 | 0.98 | 14.8 |
| decision tree (depth 4) | 0.95 | 0.95 | 0.06 | 1.3 |
| gradient-boosted trees | 0.93 | 0.95 | 0.11 | 0.95 |
| logistic regression | 0.91 | 0.89 | 0.04 | 6.0 |
| Decisions API zero-shot | 0.66 | 0.99 | 1.00 | 20.6 |
| Decisions API + 40 measured examples in context | 0.89 | 0.93 | 0.20 | 5.2 |

Cost = mean % runtime lost per firing by following the guard (harmful allowed or helpful blocked);
"always fire" would cost 341%. Zero-shot, the API calls everything harmful; with context it is a
slow, expensive nearest-neighbour over our own data. Encoder (Laya) results pending CPU time.

## Guards on the full Mac dataset (2026-10-13, 737 configurations, grouped 5-fold CV)

| family | rows (harmful / helpful / neutral) | Inductor's rule: acc, cost | tree: acc, caught, false alarms, cost | GBT cost | API+context: acc, cost |
|---|---|---|---|---|---|
| decode MLP | 161 (107 / 48 / 6) | 0.63, 14.8 | 0.95, 0.95, 0.06, 1.3 | 0.95 | 0.89, 5.2 |
| conv (channels-last) | 163 (127 / 20 / 16) | 0.22, 390 | 0.85, 0.94, 0.50, 45 | 95 | 0.79, 6.8 |
| attention fusion | 414 (9 / 386 / 19) | 0.98, 0.93 | 0.96, 0.33, 0.03, 0.81 | 0.49 | 0.98, 0.93 |

Cost = mean % runtime lost per firing when following the policy (harmful allowed or helpful blocked).

Reading per family, Mac only (no oneDNN):
* MLP: the learned guard replaces Inductor's hard-coded M <= 1 / dims <= 2048 rule with a boundary
  that costs a tenth as much; the tree's top split is arithmetic intensity (M/2 <= 3), then weight
  size, then M.
* Conv: Inductor applies channels-last unconditionally and on this machine that is wrong in 78% of
  configurations (depthwise always, dense often) at a mean cost of 390% per firing. Every learned
  guard is far better; the API-with-context scores lowest cost because "block almost everything" is
  nearly right here. The VM (oneDNN) data is needed before this guard can be called general.
* Attention: fusion is helpful in 93% of Mac configurations, so always-fire is already a 0.98
  policy; the 9 harmful cases are too few for the trees to learn (3 of 9 caught). The harmful
  region exists on the Xeon (attention block +8%, BERT +2.4%); the VM's dataset supplies it.

Encoder (Laya, ModernBERT-large fine-tuned per fold, 4 epochs, CPU, 36 min for all families):
MLP 0.88 accuracy / 0.93 caught / 0.24 false alarms / cost 4.4; conv 0.76 / 0.79 / 0.33 / 8.7;
attention 0.95 / 0.11 / 0.03 / 0.92. Ranking on every family: trees > fine-tuned encoder ~ API
with context > Inductor's hand-written rule, with the trees also the only option that answers in
microseconds offline. End-to-end real-model results: pending (running).

## Guards end to end on real models (Mac, 2026-10-13, night session on a low battery budget)

Stock Inductor vs Inductor consulting the trained guards, one family added at a time, confirmed
by the two-run rule against each model's noise threshold (results/guards_eval_e2e.md):

* **channels-last guard**: MobileNet-V3-small **-29.0%**, EfficientNet-B0 **-41.2%** (the guard
  disabled layout optimisation for exactly these two models; it left ResNet-18, ResNet-50 and all
  transformers alone: worst change +1.2%, within noise). Trained only on synthetic conv stacks.
* **matmul-decomposition guard**: no effect on the default pipeline because `decompose_mm_pass` is
  opt-in and never fires by default; it is evaluated separately with the pass enabled (queued).
* **attention guard**: no firings blocked on the Mac (fusion is helpful here), no change.
* Geometric-mean change over the 12 models: -10.7% with the first two guards, worst +1.2%.

The night session ran on a 33% battery with a reduced power budget: small decode models measured
2-3x slower than usual with wide noise; those rows (and the matmul-guard world) are re-measured
in a plugged-in daytime session.

## VM dataset (x86, oneDNN) as of 2026-10-13 (attention family partial, 202/414)

| family | rows | harmful | helpful | neutral | Mac for comparison |
|---|---|---|---|---|---|
| decode MLP | 160 | 147 | 10 | 3 | 106 / 48 / 6 |
| conv (channels-last) | 200 | 24 | 124 | 52 | 127 / 20 / 16 |
| attention fusion | 202 | 47 | 70 | 85 | 9 / 386 / 19 |

The two machines are near mirror images: decomposition is almost always harmful on the Xeon,
channels-last is mostly helpful there (oneDNN) and mostly harmful on the Mac, and attention fusion
is harmful in a quarter of the x86 configurations versus 2% on the Mac. This is why the guards
carry the machine as a feature and are trained on both machines' data.

Corrected pair scan on the VM (cat/slice toy, 43 switches, 903 pairs, 1,183 measurements, 144
interleaved baselines, none disturbed): exactly one super-additive pair survives confirmation,
`master/pattern_matcher` off + `batch_tanh` on: +11.6 ms (12x) together, ~0 alone. The earlier
scans' hundreds of flags were artefacts of a stale reference, now prevented by the start-up
reference check and consolidated-baseline re-expression.

## Guards trained on both machines' data (2026-10-13, VM attention 202/414 rows)

Grouped cross-validation (whole configuration slices held out), cost = mean % of runtime lost by
following the guard's decisions per firing. The hand-written Inductor rule is the baseline where one exists.

| family | rows | harmful | Inductor rule cost | tree | GBT | logistic | always-fire cost |
|---|---|---|---|---|---|---|---|
| decode MLP | 321 | 254 | 7.67 | 2.74 | **1.39** | 4.42 | 261 |
| conv layout | 363 | 151 | 176 (no rule) | 58.3 | **1.20** | 3.22 | 176 |
| attention | 616 | 56 | 1.57 (no rule) | 1.70 | **0.68** | 1.83 | 1.57 |

Once both machines are mixed into one dataset a depth-4 tree is no longer enough for the conv family
(it cannot carve the two machines' opposite behaviours apart with four splits), while gradient-boosted
trees stay at about 1% lost per firing on all three families. `guards-train` therefore now defaults to
`--bundle-kind best`: per family it puts the kind with the lowest cross-validated cost into the live
bundle (`results/guards_combined.pkl` = GBT for all three families). The attention guard is the only
one that beats "always fire" by less than a factor of two (0.68 vs 1.57), because only 9% of the
attention firings are harmful and they are mostly on the Xeon.

## Daytime redo of the end-to-end guard evaluation (Mac, 2026-10-13, session 2026-10-13-guards2, 100% battery, awake)

Last night's small-model rows were taken under the laptop's battery budget and are 2.5-3x too slow
(decode MLP 0.59 -> 0.20 ms, GPT-2 decode 25.5 -> 9.2 ms, SmolLM decode 14.9 -> 6.4 ms today). They are
superseded by this session; every measurement now records the power state.

Conv + attention guards on the six small models (stock base): every row "same" within tau, no guard fired
wrongly (geometric mean -0.5% / -0.2%, worst +2.7% unconfirmed).

Matmul guard with the decomposition pass enabled (base = `default+optimus/decompose_mm_wide`, the world
of a user who turned the opt-in pass on):

| model | base ms | with mlp guard | blocked firings |
|---|---|---|---|
| decode_mlp | 0.220 | -11.3% | 3 |
| smollm_135m_decode | 15.80 | -58.1% | 210 |
| qwen2_0_5b_decode | 20.98 | -24.5% | 96 |
| gpt2_decode | 22.27 | -4.1% | 1 |
| smollm_135m_prefill | 234.3 | -83.2% | 211 |

Geometric mean -46% over five models, none slower. The guard blocks exactly the decompositions the
attribution runs had blamed (decompose_mm on decode-shaped matmuls) and lets the rest through.

## First GPU results (NVIDIA A40, RunPod, torch 2.14.1+cu126, session 2026-10-13-gpu)

The harness runs unchanged on the Triton backend (device chosen by `REWRITE_BLAME_DEVICE=cuda`, timing
synchronised on the stream). The GPU build registers 398 switches (355 pattern rules): the CUDA-only
rules (pad_mm, unfuse_bias_add_to_pointwise, _fuse_addcdiv_to_fma, the half-precision sfdp variants) join
the common set.

- Toy models run in 0.15-0.27 ms on the A40, which is launch overhead; their noise thresholds are 5-20%
  and no toy switch is confirmed in either direction. The GPU story has to come from the real models.
- Toy attention block: the sfdp rules never match on CUDA; `div_softmax_pattern` and
  `unfuse_bias_add_to_pointwise` (GPU-only) fire instead. Six CPU-specific planted tests are now @cpu_only.
- ResNet-18 (2.10 ms, tau 1.0%): **layout optimisation OFF is 9.4% faster** (confirmed, second run agrees)
  and freezing ON is 12.0% faster (confirmed). On both CPUs the layout optimisation *helped* ResNet-18,
  so this is the third sign flip of that rule and the first on the GPU. Noise/verify/sweep took 16 min.

## VM dataset complete and x86 end-to-end guards (2026-10-13 evening)

VM attention family finished: 419 rows, 93 harmful / 127 helpful / 199 neutral (Mac: 9 / 386 / 19 of 414).
Combined two-CPU dataset: 1,516 rows. Grouped cross-validation, cost in % runtime lost per firing:

| family | rows | GBT cost | best other | always-fire |
|---|---|---|---|---|
| decode MLP | 320 | 0.98 | tree 6.3 | 262 |
| conv layout | 363 | 1.20 | logreg 3.2 | 176 |
| attention | 833 | 0.58 | tree 1.8 | 1.74 |

End-to-end on the Xeon (11 real models, stock base, all three guards): every row "same" within tau and
no firing blocked. That is the correct outcome on this machine: with the current x86 data the guards
judge BERT's attention firings (B=8, T=128, H=12, D=64, rule 28) not harmful, and the latest VM sweep
agrees (rule 28 is 2.6% *helpful* on BERT in session 2026-10-12; the midterm's 8% loss came from a warm
session and was retracted). Channels-last is helpful on the Xeon, so the layout guard lets it through.

Matmul guard with the decomposition pass enabled on the Xeon: gpt2_decode -73.4% (49 firings blocked),
qwen2_0_5b_decode -2.9% (73 blocked), smollm decode +0.1%, decode_mlp +6.9% unconfirmed; geometric mean
-27%. Together with the Mac (-46%), the guard removes most of the damage of the opt-in pass on both CPUs
while never confirmed slower.

## GPU phase 1 complete (NVIDIA A40, 23 models, session 2026-10-13-gpu, 2026-10-14)

Real models measure with tau 1.0-1.5% on the A40 (toys under 0.3 ms are launch-bound and 5-30% noisy;
EfficientNet and the cat/slice toy swung between runs and are re-measured in the repeat session).

- **Freezing ON is a confirmed win on 13 of 17 real models**: SmolLM decode -21%, SmolLM prefill -15/-18%,
  Qwen prefill -16%, ResNet-18 -12%, BERT -8.5%, ResNet-50 -7.6%, DistilBERT -7.5%, Qwen decode -7.9%,
  MobileNet -6.5%, T5 -4.6%. Freezing is off by default: a missed win rather than a harmful rewrite.
- **Layout optimisation is harmful on the GPU**: ResNet-18 -9.4% with it off (it helped ResNet-18 on both
  CPUs); the conv/bn stack -25%. `conv_1x1_as_mm` (opt-in) costs ResNet-50 +26%; the opt-in
  `normalization_pass` costs GPT-2 decode +11%.
- The defaults that are right: the fusion-size limit (off = +2 to +14% on ConvNeXt, ViT, BERT, DistilBERT,
  GPT-2, T5, SmolLM, Qwen decode) and the GPU-only `unfuse_bias_add_to_pointwise` (off = +12% ConvNeXt).
- **Attention fusion never fires on CUDA fp32**: not on the toy block, BERT, GPT-2 or SmolLM. The GPU-only
  `mul_softmax_pattern`/`div_softmax_pattern` fire first; the sfdp patterns then do not match. All 414
  attention dataset rows on the GPU are no_effect (identical program). So the rule is a 34% win on the Mac,
  neutral on the Xeon, and never applied on the A40.

Phase 2 (the CPU case-study pairs) correctly reported "slow state is not slower" or "identical program"
on the GPU: the tool does not invent culprits. GPU-specific cases (phase 2b) are built from the sweep:
first result, ResNet-50 default vs all opt-in passes + conv_1x1_as_mm: culprit `+conv_1x1_as_mm` alone,
+24.8% (+0.76 ms of a 0.755 ms gap), 24->27 kernels, allocations 18 MB -> 63 MB, 9 judge evaluations.

Incident: two copies of the phase 2-4 chain ran concurrently (a hung ssh launch had started one), so the
first GPU conv dataset (178 configs written twice, labels agree on only 121) and the repeat session are
contaminated and were moved to results/contaminated/. Both are being redone alone after phase 2b.

## GPU attribution case studies (A40, phase 2b, 2026-10-14, all run alone on the GPU)

| case (fast vs slow) | gap | found causes (rounds) | judge evals | kernel-level change |
|---|---|---|---|---|
| ResNet-50: default vs all opt-in + conv_1x1_as_mm | +24.6% | `+conv_1x1_as_mm` alone, +24.8% | 9 | 24->27 kernels, alloc 18 MB -> 63 MB |
| ResNet-18: best (layout off, freezing on) vs default | +10.7% | `-freezing` +1.2% then `+layout_optimization` +9.4% | 6 | 13->22 kernels, alloc 6.5 -> 30 MB |
| ConvNeXt-T: default vs five flags off | +7.4% | `-inplace_buffers` +1.3%, `-max_fusion_size` +7.1% | 12 | 25->42 kernels |
| BERT: default vs five flags off | +7.6% | `-max_fusion_size` alone, +7.9% | 7 | 7->16 kernels |
| SmolLM prefill: default+freezing vs default-max_fusion_size | +26.6% | `-freezing` +24.1%, `-max_fusion_size` +2.6% | 5 | 12->20 kernels |
| GPT-2 decode: default vs all opt-in passes | +2.8% (< tau 7.5%) | none: slow state not slower this session | 2 | identical kernel count |

Five of six cases: the culprit(s) planted by the sweep were recovered exactly, each explaining the whole gap
(residual within noise), with 5-12 judge evaluations. The sixth is an honest "no slowdown to attribute":
GPT-2 decode's +11% normalization_pass row from phase 1 did not reproduce in this session (its noise band is
7.5%), which the repeat session will settle. tau-scan on the ResNet-50 case: the same single culprit at
k = 2, 3, 4, 6 (stable).

## GPU redo (alone on the A40): conv dataset, repeat session, guards (2026-10-14 afternoon)

Clean GPU conv dataset: 178 configurations, 35 harmful / 1 helpful / 142 neutral (107 of the neutrals
unconfirmed), median tau 14% because the sweep models run in 0.2-2 ms on the GPU. Compared with the
contaminated double run (150 "harmful" of 356 rows) the direction is the same but the confidence is lower:
on the A40, channels-last never helps and hurts in a fifth of the configurations.

Attention family on the GPU: all 414 rows no_effect (the sfdp rules never fire on CUDA fp32), so there is
nothing for an attention guard to decide there.

Guard quality with the GPU rows:
- GPU-only conv guard: logistic regression 1.38% lost per firing vs 6.71% always-fire (tree 2.45, GBT 4.34).
- Three-machine conv guard (541 rows): logistic 3.42% vs 120.5% always-fire; GBT falls to 14.95% here, so
  `--bundle-kind best` now picks logistic regression for conv and GBT for the other two families
  (results/guards_3machines.pkl).

GPU repeatability (two sessions, both alone on the GPU): 188 of 221 verdicts agree (85%, vs 97% on the
CPUs). Most flips are toys and GPT-2 decode (2 ms, tau 7.5%), plus Qwen prefill effects of about 2% that sit
on the 1.5% threshold. The headline GPU findings reproduce: freezing wins, layout harm on ResNet-18, the
fusion-limit effects. GPT-2 decode's +11% normalization_pass flipped to -11%: unstable, not reported.

## GPU conv dataset v2 (GPU-sized batches) and the GPU guard: a negative result (2026-10-14 evening)

Recollected with the batch raised until each activation holds >=64M elements (calibrated: the sweep
models then run 2.5-33 ms instead of 0.2-2 ms). Labels became decisive: 99 harmful / 57 helpful / 22
neutral of 178 (small-batch run: 35 / 1 / 142). On the A40 channels-last is harmful in 71% of dense and
46% of depthwise configurations, with little dependence on channels or resolution.

GPU-only conv guard (grouped CV): logistic 2.30% lost per firing vs 8.76% always-fire (tree 2.79, GBT 2.71).

End-to-end on the seven GPU image models (stock base): the guard was consulted on every model and
blocked none, so every row is "same"; ResNet-18's 9.4% layout slowdown was not removed. Diagnosis: the
model-level features of real networks (20-81 convolutions, max channels 512-2048 at 224x224) lie far
outside the sweep grid (1-3 layers, 16-256 channels, 14-112 px), and the guard extrapolates to
"harmless" (P <= 0.17 even after clipping every feature to the training range). The two CPU guards work
on real models because depthwise=1 is decisive there and the CPU batch regime matches the sweeps.

Conclusion for the report: a guard learned from synthetic sweeps transfers only when the real firings
look like the sweep configurations. For the GPU layout decision that is not the case; a guard there needs
training rows taken from real-network firings (or a sweep whose layer counts and activation budgets match
real networks). Kept as evidence: results/gpu/dataset_conv_smallbatch.jsonl (noisy labels),
results/gpu/guards_eval_e2e.json (three-machine guard), results/gpu/guards_eval_e2e_gpu2.json (GPU-only).

GPU campaign cost: $18.60 of the $188 RunPod credit (A40 at $0.59/h for ~31 h). Pod terminated 2026-10-14.

## GPU layout guard trained on real networks (2026-10-15, second A40 pod)

The synthetic-sweep guard did not transfer to real GPU models (above). Replacing the training set with real
networks fixes it. New family `net`: 20 torchvision architectures (models/zoo.py; the seven corpus image models
are excluded and held out) at batch 8, 32 and 64, features from one forward pass with hooks on every
convolution (batch, resolution, number of convs, total MACs, fraction of MACs in depthwise and in 1x1 convs,
work-weighted mean and max channels, mean kernel, total activation volume). Noise band from one stable
batch-32 network (ResNet-34: 1.0%, floor 2%). A first attempt used the default median-of-3 representative
configurations, two of which were batch-1 networks with 36-43% noise bands on the GPU; it labelled all 60 rows
neutral and is kept as results/gpu/dataset_net_tau36_*.jsonl. Batch 1 is now excluded from the GPU grid.

Dataset: 60 rows, 10 harmful / 30 helpful / 20 neutral. The layout rule hurts at batch 8 (DenseNet, VGG,
RegNet-Y +20%, ResNet-34 +9%, ShuffleNet at batch 64 +26%) and helps most networks at batch 32-64 (AlexNet
-34%, ResNeXt -11%, ResNet-101 -8%). Grouped CV by architecture: logistic regression 1.65% lost per firing vs
1.98% always-fire (recall 0.80, false alarms 0.20); trees do worse on 60 rows.

End-to-end on the seven held-out corpus image models (stock base, A40):

| model | stock ms | with the net guard | blocked |
|---|---|---|---|
| resnet18 | 2.10 | **-8.2% faster (confirmed)** | yes |
| resnet50 | 3.13 | **-3.7% faster (confirmed)** | yes |
| conv_bn_relu_stack | 0.27 | **-21.0% faster (confirmed)** | yes |
| efficientnet_b0 | 2.82 | -19.4% (not confirmed) | yes |
| mobilenet_v3_small | 1.66 | +5.9% (not confirmed) | yes |
| convnext_tiny | 4.23 | -0.2% | no |
| vit_b_16 | 11.76 | +0.2% | no |

Geometric mean -7.1%, three confirmed speed-ups, no confirmed regression; the 9.4% ResNet-18 slowdown the
sweep found is removed by the guard. Compared with the synthetic-sweep guard (blocked nothing): same
compiler hook, same decision point, different training distribution. Conclusion for the report: the guard
mechanism works on all three machines; what it needs is training rows whose features look like the firings
it will judge. Second pod cost about $3 (A40, 4 h).

## Campaign 2 (bf16) on four pods, 2026-10-16/17: what ran, what broke, what it cost

Four pods in parallel, all bf16: RTX 4090 (full chain), A40 (sweep), A40 (seven switch datasets over the
zoos), L4 (full chain, top-3). Everything finished between 01:33 and 18:47 UTC on 2026-10-17.

**Cost incident.** The controlling session ended and the pods idled 20-40 h at full price: about $138 spent
against an estimate of $35 and the agreed $80 cap; $29 of credit remains. All pods were terminated on
2026-10-18. Every pod chain now self-terminates at its end (`trap` in `scripts/gpu_*.sh`), and pulls keep one
store per pod. The merged pull also overwrote same-named summary files between pods; the per-pod stores are
complete (A40 sweep 1,355 measurements, A40 datasets 1,715, L4 2,650, 4090 3,010), and
`scripts/rebuild_from_store.py` regenerates each machine's noise, sweep and dataset tables from its store.

**Phase 3 did not run.** `guards-eval` crashed on every pod before measuring (temp-file name built from a
family id containing `/`; fixed). The bf16 end-to-end guard evaluation is still to be done (one pod per GPU,
about $3 each, with self-termination).

**bf16 changes the attention story.** In bf16 BERT's attention fusion fires on the GPUs through
`_sfdp_pattern_28_half_inference` (12 firings); in fp32 no attention rule ever fired on the GPU. The fp32
"never fires" result was a precision artefact. On the L4 the rule is worth 7.5%: the attribution case
BERT default vs {rule 28 off, fusion limit off, ...} found `-rule 28` (+7.5%, 5->7 kernels) and
`-max_fusion_size` (+19.0%, 7->16 kernels), explaining the whole 0.85 ms gap in 13 judge evaluations.

**bf16 attribution case studies (L4, auto-seeded from the sweep).** attention block: a genuine interaction
`-master/pattern_matcher` x `-master/post_grad_passes` (+6.3%) plus `-max_fusion_size` (+14.2%); conv stack:
`-layout_optimization` alone (+16.3%); DistilBERT: `-inplace_buffers` (+3.2%) and `-max_fusion_size` (+3.5%);
ConvNeXt and decode MLP: slow state not slower (correctly no culprit). On the 4090 the automatic case builder
mixed `+freezing` (a win) into the "slow" specs, so four of six cases were not slower and the tool said so;
GPT-2 decode: `-max_fusion_size` alone (+13.8%, 11->24 kernels).

**bf16 switch ranking (confirmed |delta| >= 3%, summed over models).** 4090: freezing on 19 of 22 models (all
faster), max_fusion_size 12, constant_folding, inplace_buffers, layout_optimization 4. A40: freezing 14,
max_fusion_size 13, layout 2, constant_folding 3, rule 28 (half) 1.

## Rebuilt from the stores: bf16 switch datasets on three GPUs and the six-machine sign matrix (2026-10-18)

Per-GPU tables are in results/gpu/{a40,l4,rtx4090}/ (regenerated by `scripts/rebuild_from_store.py`; model
profiles recomputed locally). Real-network rows per switch (163 = 114 torchvision + 49 transformer zoo models):

| GPU | switch | harmful | helpful | neutral | no effect | median confirmed effect |
|---|---|---|---|---|---|---|
| 4090 | freezing (default off) | 1 | 113 | 49 | 0 | 10.3% |
| L4 | freezing | 1 | 105 | 57 | 0 | 6.7% |
| A40 | freezing | 0 | 112 | 51 | 0 | 8.5% |
| 4090 | layout_optimization | 21 | 62 | 28 | 52 | 16.5% |
| L4 | layout_optimization | 21 | 83 | 7 | 52 | 24.2% |
| A40 | layout_optimization | 18 | 69 | 24 | 52 | 18.6% |
| 4090 | max_fusion_size | 12 | 52 | 99 | 0 | 10.9% |
| L4 | max_fusion_size | 4 | 76 | 83 | 0 | 6.6% |
| A40 | max_fusion_size | 10 | 71 | 82 | 0 | 8.4% |
| 4090 | inplace_buffers | 12 | 30 | 121 | 0 | 4.8% |
| A40 | inplace_buffers | 19 | 10 | 134 | 0 | 6.9% |
| A40 | conv_1x1_as_mm (default off) | 18 | 14 | 55 | 76 | 6.1% |
| 4090 | constant_folding | 5 | 1 | 27 | 130 | 6.8% |
| A40 | rule 28 (half) and unfuse_bias_add over the transformer zoo | 0 | 0 | 0 | 49 | the zoo uses sdpa attention: the eager-attention rules never match |

**What this says about guards on GPUs in bf16.** Where a rule is harmful on a minority of real networks and
helpful on the majority (layout: 18-21 harmful vs 62-83 helpful), the learned guards do *not* beat
"always on" in grouped cross-validation (GBT cost 4.2 / 4.3 / 3.3% lost per firing vs 3.6 / 3.6 / 2.9% for
always-on on the 4090 / L4 / A40): the false alarms on helpful networks cost more than the harm avoided.
The guard approach pays off where harm is frequent and shape-predictable (CPU matmul decomposition
261% -> 1%, CPU layout 176% -> 1%, A40 fp32 layout end-to-end). Where a rule is right most of the time on a
machine, the measured answer is a per-machine default, and the datasets quantify it:

- **Freezing should be ON for GPU inference**: helpful on 105-113 of 163 real networks on each of three GPUs
  (median 7-10%, up to 49%), harmful on at most 1. It is off by default in Inductor. This is the strongest
  upstream item we have, backed by ~490 confirmed measurements.
- Layout optimisation and the fusion-size limit: keep ON on GPUs (helpful 3-7x more often than harmful).

**Six-machine sign matrix** (`scripts/sign_matrix.py`, docs/sign_matrix.md, report/final/tables/sign_matrix.*):
Mac M4, Xeon, A40 fp32, A40 bf16, L4 bf16, 4090 bf16. Of the 16 most impactful switches, **8 have the opposite
dominant sign on different machines**: layout optimisation (hurts on Mac and A40 fp32, helps on Xeon and all bf16
GPUs), the fusion limit (helps everywhere but the Xeon), the pattern matcher and post-grad passes as a whole
(help on Mac/GPUs, hurt on the Xeon), matmul decomposition (hurts Mac, helps Xeon), normalization pass,
reorder_for_locality and split_reductions. Freezing helps on every machine (Xeon up to 95%).

Guard bundles per GPU: results/gpu/<gpu>/guards_switches.pkl (block mode for default-on switches, enable mode for
freezing / conv_1x1_as_mm); their bf16 end-to-end run is still pending a pod.

## Why rules flip: same code change, different hardware (2026-10-19, `scripts/mechanism.py`, docs/mechanism.md)

Joining every confirmed sweep verdict with the code-level change the same switch makes (kernel count, external
library calls, static allocation) shows that **the compiler makes the same structural change on every
machine; only the runtime consequence differs**:

- `decompose_mm_pass` ON removes 96-106 external BLAS calls and 3-4 kernels on *both* CPUs; that is 39% slower
  on the Mac (Accelerate's small-matmul path is fast, the generated loops are not) and 53% faster on the Xeon.
- `layout_optimization` ON adds about 2 kernels and 4-22 MB of static allocation on both CPUs (the batch-norm/
  ReLU epilogues stop fusing into the convolution and become `cpp_fused__native_batch_norm_*` kernels); that costs
  28% on the Mac and saves 14% on the Xeon, whose oneDNN convolutions prefer channels-last.
- `max_fusion_size` ON (the default limit) removes 8-12 kernels on the GPUs (fewer, larger Triton kernels): 7-12%
  faster on 11-13 models per GPU, while on the Xeon the one confirmed effect goes the other way.
- `freezing` ON on the A40 removes 18 external calls per model (constant-folded weight transforms), on the 4090
  about 1, and on the Xeon it removes 18 MB of allocation; it helps everywhere the generated code is
  cheaper than the library call it replaces, and the one Mac loss (ResNet-50, 7%) is a case where it is not.

Hence the thesis in one sentence: a graph rewrite fixes the *shape* of the generated program, and whether that
shape is faster depends on the relative cost of library calls, generated kernels and memory traffic on the
target, which no rule written once can know.

Freezing package, GPU part (A5000, bf16, 2026-10-19): trained weights agree with random init (labels 10/12 for
freezing, 11/12 for layout, equal magnitudes); peak memory +45-51% for LLM decode/prefill with freezing (frozen
constants beside the original parameters; `freezing_discard_parameters` exists for this), 0 for attention
encoders; compile +10-30%; real workloads SmolLM `generate` 8-22% faster (271 -> 676 MB), ResNet-50 batch-32
3.6% faster (179 -> 240 MB), identical outputs. Issue draft: docs/pytorch_issue_freezing.md.

## Freezing package, Xeon part (2026-10-19, overnight, pinned 1-3, min statistic)

Freezing on the 12 corpus models, random and trained weights: 10/12 faster with both weight sets (labels agree
12/12): MobileNet-V3 -28% random / -36% pretrained, GPT-2 decode -37% / -39%, SmolLM prefill -18% / -16%,
EfficientNet -17%, ResNet-50 -13%, encoders -4 to -7%; SmolLM and Qwen decode neutral. Real workloads on the Xeon:
ResNet-50 batch-32 classify +10-11% faster, SmolLM generate -1% (no gain) with peak RSS 1.3 -> 3.3 GB.
Real-network zoo (batch 8) on the Xeon: layout ON helpful on 35/52 conv nets, harmful 0 (median 13.5%);
`decompose_mm_wide` ON harmful on 13/22 transformers (median 133%!), helpful 0: the CPU matmul guard case stands.
CPU peak RSS numbers for random-init rows are not usable (constant 3.7 GB, dominated by the build); the GPU
`max_memory_allocated` numbers are the memory evidence.

## Freezing package, Mac part (2026-10-19, battery 56%, 8 threads)
Freezing on the Apple M4 Pro: 11/12 corpus models neutral (SmolLM decode 2-4% faster), random and trained weights
agree 12/12; real workloads within 1% (generate 245 -> 243 ms, classify 1.61 s both). Layout: harmful on 4 of the 10
conv rows as before. Evidence extracts for the issue are now tracked in `evidence/`.

## Upstream (2026-10-20)
Freezing issue filed by Chidera: https://github.com/pytorch/pytorch/issues/201502 (no labels yet; the triage bot
usually adds `module: inductor` within a day). Replies to maintainers are written by Chidera (PyTorch AI policy).

## Second PyTorch version (E1), first result: the rule set drifts between releases (2026-10-20)
PyTorch 2.13.0 on the Mac registers 267 switches vs 269 under 2.14.1. The four rules that exist only in 2.14 are
`reciprocal_sqrt_to_rsqrt`, `_fuse_addcdiv_to_fma`, `unfuse_bias_baddbmm_to_pointwise` and one more sfdp replacement
variant; nothing was removed. The 10-model subset sweeps under 2.13 (Mac and VM, `scripts/runs/pt213_subset.sh`)
give the second-version columns of the sign matrix. Practical note: torch 2.13's C++ builder does not quote paths, so
its venv must live at a path without spaces (~/venvs/rb-pt213 on the Mac).

## E3 on an A40 (bf16, 2026-10-20 night, $2.50): switch guards end-to-end, freezing_discard_parameters, max-autotune

- **`freezing_discard_parameters=True` removes the duplicate memory** (base = freezing ON, A40): ResNet-18 72 -> 51 MB
  (unfrozen baseline 48), ResNet-50 121 -> 75 (73), BERT 280 -> 235 (241), DistilBERT 172 -> 149 (155), MobileNet 24 -> 21,
  EfficientNet 38 -> 31; speed unchanged within noise. So the memory cost of freezing reported in #201502 is avoidable.
- **It crashes on every tied-embedding LLM** (GPT-2 prefill/decode, SmolLM decode/prefill, Qwen2.5-0.5B decode, random and
  pretrained): `AssertionError: Guard failed on the same frame it was created. This is a bug - please create an issue.`
  That is a second, independent PyTorch issue (Dynamo guard bug under freezing + discard_parameters); reproducer below.
- **max-autotune could not be measured**: our harness kept the first output tensor and compared it after the timing loop,
  which under CUDA graphs (enabled by max-autotune) reads a buffer the next call overwrote (`accessing tensor output of
  CUDAGraphs that has been overwritten`). Fixed in measure.py (outputs cloned when a compile mode is set); needs a rerun.
- **Switch guards on the A40 held-out models**: all "same" (unconfirmed), geometric mean -0.3% to +2% across the five
  steps; the guards blocked or enabled almost nothing (0-1 per model). The per-model oracle says freezing is the best
  toggle on 5 of 11 models (Qwen decode -23.8%, ResNet-18 -13.7%, BERT -11.7%, DistilBERT -7.7%, ConvNeXt -4.3%), so the
  freezing *enable* guard should have fired and did not: under investigation (suspect: the live profile carries
  `cfg_bf16=1` while the dataset rows were profiled in fp32 on the CPU, an out-of-distribution feature). Two decode
  models had unusable noise bands in this session (GPT-2 decode 48%, SmolLM decode 28%).
