# Meeting script, 15 Oct 2026 (about 5 minutes spoken)

## 1. The problem (30 seconds)
torch.compile speeds up a PyTorch model by applying a few hundred small rewrite rules. Each rule finds a
pattern in the model and replaces it with something it expects to be faster. The compiler applies a rule
every time the pattern appears. It never checks whether that made the model faster on this hardware.
Sometimes a rule makes the model slower, and today there is no way to find out which one.

My project answers two questions. First: when a compiled model is slower, which rule did it? Second: can
we stop that rule from firing in the cases where it hurts?

## 2. What I built (90 seconds)
Four pieces.

One. A switch for every rule. Inside the compiler each rule has a small check function it calls before
applying the rule. I replace that function with mine: if the rule's switch is off, mine says no and the
compiler skips the rule. Nothing in PyTorch is modified. I also count every time a rule fires.

Two. A measuring tool. It compiles a model with any set of switches in a fresh process, checks the output
still matches the uncompiled model, records what code came out, and times it. I use 24 models, from small
test cases to ResNet, BERT, GPT-2 and Qwen.

Three. A rule for separating a real slowdown from timing noise. I ran the same configuration hundreds of
times with nothing changed, tried ten different statistical tests, and picked the one with the fewest
false alarms. A change counts only if it beats the noise band in two separate runs taken at different
times. If the compiler produced identical code, nothing can be blamed.

Four. The blame tool. You give it a fast setup and a slow setup. It splits the list of differences in half,
measures each half, keeps the half that is still slow, and repeats until no single change can be removed.
That is delta debugging. It also tests whether two rules only hurt together, and it keeps going until the
remaining gap is inside the noise band, so it finds several causes if there are several.

Everything is tested: about 140 automated tests, including small models with a planted slowdown where the
tool must name the right rule.

## 3. What I found (90 seconds)
I ran this on three machines: my Apple M4 Pro laptop, the course Linux machine with an Intel Xeon, and a
rented NVIDIA A40 GPU.

The main finding: the same rule helps on one machine and hurts on another. The attention fusion rule is a
34 percent win on the Mac, neutral on the Xeon, and on the GPU it never even fires because a GPU-only rule
changes the pattern first. The channels-last layout rule makes MobileNet 41 percent slower on the Mac but
helps it on the Xeon, and on the GPU it makes ResNet-18 9 percent slower, the same model it helps on both
CPUs.

Second: the blame tool works. On the GPU I hid a known 25 percent slowdown among seventeen changes and it
named the one rule, with the explanation: kernels went from 24 to 27 and memory allocations tripled.
Across all the case studies it recovers the planted cause, and when nothing is actually slower it says so
instead of guessing. It also found a real pair: two attention rules that change nothing alone and only
change the code together.

Third: results repeat. On the same machine, 97 percent of verdicts agree across days on the CPUs and 85
percent on the GPU. Across machines they mostly disagree, and that disagreement is the hardware, not noise.

One more thing worth saying: on the GPU, a setting called freezing, which is off by default, makes 13 of 17
real models faster, by 5 to 21 percent. That is a free win the compiler leaves on the table.

## 4. The guards (60 seconds)
The stretch goal was to act on the blame. A guard is a small learned model that sits in that same check
function and decides, each time a rule is about to fire, whether to allow it. I built a dataset of about
1,700 rule firings across the machines, each labelled by measuring the model with the rule on and off, and
trained one guard per rule family.

On the two CPUs this works. Per firing, a guard loses about 1 percent of run time, where always applying
the rule loses 176 to 261 percent for the two worst families. On real models the guard made MobileNet 29
percent faster and EfficientNet 41 percent faster on the Mac, and GPT-2 decode 73 percent faster on the
Xeon when the matmul pass is on. Nothing got slower.

On the GPU it did not work, and I know why. The guard is trained on small synthetic networks with one to
three layers. Real networks have twenty to eighty layers and far more channels, so the guard is asked about
inputs unlike anything it was trained on and it answers "harmless". I fixed the dataset's noise problem,
retrained, and it still did not transfer. So the honest conclusion is: a guard learned from synthetic
sweeps only transfers when real firings look like the sweep, and for the GPU layout decision they do not.

## 5. Where it stands and what is next (30 seconds)
All measurement is done: three machines, 24 models, the noise study, the case studies, the repeat runs,
and the guard datasets and evaluations. The GPU work cost 19 dollars of credits. Code and notes are on
GitHub. What is left is the final report, due November 19. Its argument: the per-rule switches and the
blame tool are the contribution, the cross-hardware sign flips are why the tool is needed, and the guards
show that once you know the harmful rule you can fix it, with a clear statement of where that stops working.

## Likely questions and short answers
- Why do the rule counts differ per machine (269, 396, 398)? The compiler only registers rules that can
  run there: the Xeon build has Intel's oneDNN fusion rules, the GPU build has CUDA-only rules.
- How do you know a slowdown is not noise? Seven baseline runs give a noise band, three times the median
  deviation with a 1 percent floor. A change must beat it in two time-separated runs, and identical
  generated code is never blamed.
- Why synthetic sweeps for the guard data? To cover shapes systematically with confident labels. The GPU
  result shows the limit of that choice.
- Could this go upstream? The switch mechanism answers an open PyTorch request for pattern-matcher
  observability, and the freezing and layout findings are concrete issues to file.

---

# Detail blocks (say these when asked, or fold them in)

## How the switch works and how I decide what to switch
Every rewrite rule object inside Inductor has a field called `extra_check`, a function the compiler calls
after the rule's pattern matched and before it applies the rule. The rule runs only if that function says
yes. I replace that field on every rule with a small object of my own. When the compiler calls it, my object
looks up the rule's switch. Off: it answers no, the compiler moves on, and I count one "suppressed". On: it
calls the rule's original function and passes its answer through, and I count one "fired".

To find all the rules I first compile a tiny warm-up model that contains attention, a convolution with batch
norm, a split and a concatenate, a matmul and a type cast, because Inductor only registers its rules the
first time it compiles. Then I walk through Inductor's modules, find every pattern-matcher object and every
rule inside it, and give each a name such as `joint_graph.patterns/_sfdp_pattern_28_inference`.
Compiler settings such as freezing or the fusion-size limit, and optional passes such as the matmul
decomposition, are switches too, set through Inductor's configuration object.

How I decide which to turn off: a "verify" step. Starting from the default, I flip each switch alone,
compile, and compare the generated code by a hash. Most switches change nothing for a given model, because
their pattern never appears in it. Only the switches that actually change the generated code go into the
timing sweep, where each is flipped alone and timed. For BERT that is about ten switches out of 398.

## The noise method, with the numbers
One measurement is: compile, 10 warm-up calls, then 50 timed calls. The 50 times are right-skewed: most
calls are close to the fastest, a few are much slower when the operating system or another process
interrupts. The mean is pulled up by those few, so I take the median of the 50.

Then I run that whole measurement seven times, each in a fresh process, spread out in time, and get seven
medians. The noise band is three times the median absolute deviation of those seven, with a floor of one
percent. The median absolute deviation is: take the median of the seven, take each one's distance from it,
take the median of those distances.

Why this rule and not another: I ran 480 measurements where nothing was changed, so any detected slowdown
was a false alarm, and compared ten decision rules. Three standard deviations, interquartile range,
Wasserstein distance between the sample distributions, bootstrap intervals, Mann-Whitney tests on the
pooled calls, and MAD with different multipliers. All of them gave 6 to 16 percent false alarms from a
single comparison, because between-process noise has fat tails. What actually fixed it was replication:
require two separate candidate runs, taken at different times with a fresh baseline between them, to both
exceed the band. That took false alarms from 9 percent to 1.5 percent. I also tested whether trimming the
warm-up portion of each run helps instead of assuming it; it changed little, so the warm-up is kept simple.
Finally, if the compiler produced byte-identical code for two setups they cannot differ in speed, so the
tool never blames a switch that did not change the code.

## Why delta debugging
Between a fast setup and a slow setup there can be a hundred or more differences. For BERT on the GPU the
list was 117 changes. Testing each change alone needs 117 compiles and still misses cases where two changes
only hurt together. Testing every combination is impossible. Delta debugging splits the list in half, tests
each half, keeps a half that is still slow, and when neither half alone is slow it splits finer and tries
removing one piece at a time. It ends with the smallest set that still reproduces the slowdown, usually in
5 to 16 measurements, and it finds pairs. Each measurement costs seconds to minutes, and results are stored
so nothing is measured twice.

## What one dataset row looks like
A row is one synthetic model, one machine, the rule on versus off, measured twice. A real row from the Mac:

  model: conv sweep, 128 channels, 28x28 input, 1x1 kernels, 3 layers, depthwise
  machine: Mac, 8 threads
  time with the layout rule on: 12.71 ms;  with it off: 10.33 ms;  ratio 1.23 (second run 1.21)
  noise band: 2.7 percent;  kernels: 3 with the rule on, 1 with it off
  label: harmful, confirmed

The features the guard sees are the configuration numbers (channels, resolution, kernel, layers,
depthwise, activation size) plus which machine and the thread count. The label is harmful if both ratios
exceed one plus the band, helpful if both are below one minus the band, neutral otherwise. Rows where the
code did not change are dropped: there is nothing to decide. About 1,700 rows across the three machines.

## How the guard was determined
For each rule family I compared several models on identical data splits: a single decision tree,
gradient-boosted trees, logistic regression, a fine-tuned text encoder, and OpenAI's Decisions API given
the same numbers as text. The split holds out whole groups of configurations the model has never seen,
so the score is about generalising, not memorising. The score is the run time lost per firing by
following the model's decisions, compared with always applying the rule, which is what Inductor does today.
The best model per family goes into the live bundle: gradient-boosted trees for attention and the matmul
family, logistic regression for the layout family. In the compiler the guard runs inside the same check
function described above: when a rule is about to fire, the shapes at the match are turned into the
feature vector, the model gives a probability of harm, and above one half the check answers no.

## Why BERT gets different rewrites on the GPU than on the CPUs
The model is identical. Two things differ. First, each rule's own check inspects the device of the tensors
it matched, and many rules exist for one backend only. On the GPU build there are rules that pad matrix
sizes for the tensor cores, a rule that moves a bias addition out of a matmul so it can fuse into the next
Triton kernel, and a rule that folds a scaling multiply into the softmax. On the Xeon build there are
oneDNN fusion rules that merge a linear layer with the operation after it. Second, the rules run in a fixed
order and each changes the graph the next one sees. On BERT the same four rules fire on the Mac and the
Xeon, including the attention fusion rule number 28. On the GPU the softmax-scaling rule fires first on the
same twelve attention blocks, rewrites exactly the part the fusion rule was looking for, and the fusion rule
then matches nothing. So the difference is not the architecture of BERT; it is that the compiler keeps a
different rule set per backend because the kernel generator is different, Triton on the GPU versus C++ and
OpenMP on the CPUs, and a graph shape that is best for one is not best for the other.
