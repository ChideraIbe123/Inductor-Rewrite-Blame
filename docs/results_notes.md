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
