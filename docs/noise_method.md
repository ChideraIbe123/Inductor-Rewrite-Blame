# How the harness decides that a configuration is slower

Timing noise is the main threat to blaming a rewrite. This note records how the decision rule was
chosen: not by convention, but by calibrating candidate rules on the harness's own stored
measurements (482 runs with raw per-call samples, two machines: Apple M4 Pro and a 4-core Xeon VM).
The scripts are `scripts/noise_study_methods.py` and `scripts/noise_study_replication.py`; they
read `results/measurements.sqlite` and `results/vm/measurements.sqlite`.

## What the raw samples look like

* Per-call times are strongly right-skewed (skewness 1 to 7) with 5-10% Tukey outliers. Means are
  useless; medians are used throughout.
* Warm-up leaks into the timed window for some runs: the first five timed calls are ~1% above
  the rest at the median, and up to 17% at the 90th percentile; 20-75% of runs still trend
  downward (Spearman < -0.3), depending on the model.
* Variance lives at two levels. Within a run the relative IQR is 0.4-5%; between independent
  processes the CV of run medians is 0.1-7%. Identical configurations occasionally differ by
  22-24% between processes: whole processes that run slow (shared host, background load).

## Steady state (warm-up) detection: measured, not assumed

Policies compared by the between-run spread of run medians (lower = more precise) and by the
false-positive rate of the MAD rule on identical runs:

| policy | median between-run CV | false positives (single run) |
|---|---|---|
| raw (median of all 50 timed calls) | 1.10% | 16/180 = 8.9% |
| drop first 10 timed calls | 1.03% | 17/180 |
| conservative knee (adopted) | 0.93% | 17/180 |
| kneedle elbow on the cumulative mean | 1.70% | 22/180 |
| keep last half only | 1.20% | 23/180 |

The adopted rule: the steady state starts at the first index `k` where the median of calls
`k..k+5` is within one IQR of the median of the second half of the run. For most runs `k = 0`
(the ten warm-up calls were enough); it changes a run's median by 0.00% at the median and by
up to 11% for the rare run whose warm-up leaked. It buys a 15% reduction in spread and never
hurts. The more aggressive elbow rules discard good samples and make precision worse. Warm-up is
therefore handled, but it is *not* what causes false verdicts.

## Where false verdicts come from

Leave-one-out over identical runs (each run in turn plays the "candidate", the other 6-7 the
reference). False-positive rate of a single candidate run, by rule:

| rule | false positives | median threshold |
|---|---|---|
| Wasserstein-1 distance above the 95th percentile of identical-run distances, plus positive shift | 6.1% | 7.2% |
| 3 standard deviations of run medians | 6.7% | 3.4% |
| hierarchical bootstrap of one run's median, 99th percentile | 7.8% | 2.6% |
| MAD, k = 4 | 8.3% | 2.8% |
| IQR-based sigma, k = 3 | 8.3% | 1.8% |
| **MAD, k = 3 (floor 1%)** | **8.9%** | **2.1%** |
| MAD, k = 3 after steady-state trim | 9.4% | 1.8% |
| Mann-Whitney on pooled calls, p < 0.001 and shift > 2% | 10.0% | 2.0% |
| MAD, k = 2 | 12.8% | 1.4% |
| Mann-Whitney on pooled calls, p < 0.01 and shift > 1% | 16.1% | 1.0% |

Every rule is wrong 6-16% of the time with one candidate run, and the rules that reach 6% do so
only by raising the threshold to 3-7%, which loses the 2-5% effects the project cares about
(the SD rule detects 1 of 33 effects between 1.5% and 3%, MAD k=3 detects 11). The reason is
the fat right tail of between-process noise: the false positives are processes that ran 1.2% to
22% slow as a whole. Their within-run spread is 2.5x the typical (a signature), but a third of
genuinely slower configurations share that signature, so it cannot be used to discard runs.

## What fixes it: replication with an AND rule

Null false-positive rate when the candidate is measured in `m` independent processes:

| rule | m = 1 | m = 2 | m = 3 |
|---|---|---|---|
| mean of candidate medians above tau | 8.9% | 9.5% | 6.0% |
| mean above tau / sqrt(m) | 8.9% | 12.9% | 9.5% |
| **every candidate run above tau (AND)** | **8.9%** | **1.5%** | **0.9%** |
| Mann-Whitney on run medians, p < 0.05 | - | 2.7% (power 7/35) | 2.1% |
| between-run bootstrap, 99th percentile | 10.0% | 12.6% | 5.8% |

Averaging does not help because one 20% outlier process dominates the mean. The AND rule does,
and keeps 100% detection of effects above 6% (35/35 at m = 2, 56/56 at m = 3). Power for 3-6%
effects is roughly the single-run power squared (about 22% at m = 2); detecting such small
effects reliably needs more reference and candidate runs, which is a measurement-budget choice,
not a statistics choice.

## The adopted procedure

1. One measurement = one fresh process: 10 warm-up calls, 50 timed calls, run statistic = median
   over the steady-state calls.
2. Reference = 7 independent processes of the fast state; centre = median of their medians;
   tau = max(3 x MAD of their medians, 1% of the centre).
3. A candidate is SLOW only if two independent processes each exceed the centre by more than
   tau. A candidate whose first process is within tau is FAST without a second run (a false
   FAST only costs ddmin a missed reduction; a false SLOW sends it down the wrong branch).
4. Pairwise interaction flags are confirmed the same way (both runs must agree).
5. Timed measurements are compared only within one session (same day, same machine) because
   machine speed drifts between sessions; the `repeat` command reports how verdicts hold up
   across sessions and machines.
6. The `tau-scan` command re-runs an attribution for k = 2, 3, 4, 6 and reports whether the
   culprits are stable; unstable culprits mean the effect is near the noise floor.

## Interleaving: the two confirmation runs must be separated in time

A 903-pair interaction scan on the VM (4.4 hours, 1,416 measurements) showed two artefacts even
with the AND rule in place:

* 220 pairs looked *faster* than their singles. The singles were measured in the first hour, the
  pairs later, and the machine was about 1% faster in hours 2-4 (median of all run medians per
  2-hour bin: 0.957, 0.950, 0.956 ms). A stale baseline turns machine drift into fake "masking".
* Four pairs of switches that cannot plausibly interact showed +70% to +150% together, confirmed
  by both runs. The two runs were back-to-back, so one disturbed stretch of machine time passed
  both. (Re-measuring those states in a fresh session is the direct check; see the results file.)

Both are the known failure modes that interleaved designs exist for, so the harness now:

1. takes a fresh **baseline run between the two candidate runs** of a slow verdict; if that
   baseline deviates from the session reference by more than tau the machine is disturbed, the
   attempt is discarded and retried later with fresh processes (twice at most), and if it never
   settles the verdict is FAST with a note rather than a blame;
2. in long scans, **re-measures the base state every 20 measurements** and compares every
   candidate with the running baseline (median of the last three baseline runs) instead of the
   one taken at the start; flagged pairs are confirmed only after a baseline run that is within
   tau and by a second run that agrees.

Cost: one extra baseline run per slow verdict (ddmin issues O(log n) of them) and 5% more runs
in a scan.

## A guard that needs no statistics: identical generated programs

Re-measuring the four "giant interaction" pair states in a fresh session gave 0.95-0.98 ms
against a 0.97 ms default, and all of them compile to the very same kernel as the default. That
suggests a filter stronger than any threshold: every measurement now records a hash of the
generated program (wrapper plus kernels, with the per-process compile counter and cache paths
stripped). If a candidate's program hash equals the reference's, the switches in question did
nothing to the compiled code, so a timing difference between them cannot be a rewrite effect:
the judge returns FAST with a note, the pair scan never flags such a pair, and the sweep marks
the row. The hash is stable across processes and changes whenever a rewrite actually fires
(tested on the layer-norm MLP: disabling a rule that fires changes it, disabling one that never
fires does not).
