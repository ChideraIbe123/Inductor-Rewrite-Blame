# Meeting script, 15 Oct 2026 (about five minutes spoken)

I built a tool that finds which rewrite rule in the PyTorch compiler made a model slower. That part works
on all three machines I tested: my Mac, the course Xeon, and a rented GPU.

**How the switch works.** Inside the compiler, every rewrite rule has a small check function. The compiler
calls it right before applying the rule, and only applies the rule if the function says yes. I swap that
function for my own. If I have turned the rule off, mine says no. If it is on, mine calls the original and
passes the answer through, and counts that the rule fired. Nothing in PyTorch is changed. To get the list of
rules, I compile one tiny model first, because the compiler only sets up its rules the first time it runs,
and then I read them out of the compiler's memory and give each one a name.

**How I decide what to turn off.** Most rules never touch a given model, so flipping them does nothing. So
for each model I first flip every switch one at a time and check whether the generated code changed. Only
the switches that change the code are worth timing. For BERT that is about ten out of nearly four hundred.
Then I time each of those with the switch off and compare to the default.

**How I handle noise.** If I run the same model twice, I get two different times. So before I can say
"this switch made the model slower," I need to know how much the time moves on its own.

Here is what I do. To time a model once, I call it fifty times and take the middle value, the median. I use
the median and not the average because a few of the fifty calls are always much slower, when the operating
system interrupts, and those few would drag an average up.

Then I do that whole thing seven times, each in a fresh process, with gaps in between. That gives me seven
medians for the same unchanged model. How far apart those seven are tells me the normal wobble. I set a
limit of three times that wobble, and at least one percent. A switch is only a suspect if it moves the
time by more than that limit.

How I know this is the right limit: I ran about five hundred measurements where I changed nothing at all.
Any slowdown the tool reported there was wrong by definition. I tried ten different statistical rules on
those runs. Every one of them was wrong about ten percent of the time if it looked at just one measurement.
So I added one more rule: the tool measures the suspect setup twice, at two different times, with a fresh
measurement of the default in between, and it only calls it slower if both measurements are over the limit.
That brought the wrong calls down to about one percent.

One last check that needs no statistics: if the compiler produced exactly the same code with a switch on
and off, the speed cannot differ, so the tool never blames that switch.

**Why delta debugging.** When I have a fast setup and a slow setup, the difference can be a hundred switches.
For BERT on the GPU it was 117. Testing each one alone would take 117 compiles and would still miss the case
where two rules only hurt together. So the tool splits the list in half, measures each half, keeps the half
that is still slow, and repeats, and at the end tries removing single switches. It finishes in five to
sixteen measurements with the smallest set that reproduces the slowdown, and it catches pairs. It found a real
one: two attention rules that change nothing alone and only change the code together.

**The main finding** is that the same rule helps on one machine and hurts on another. The attention fusion
rule is a 34 percent win on the Mac, neutral on the Xeon, and never fires on the GPU. The layout rule slows
MobileNet by 41 percent on the Mac but helps it on the Xeon, and on the GPU it slows ResNet-18, the model it
helps on both CPUs.

**Why the same model gets different rewrites on different hardware.** BERT is the same model everywhere,
but the compiler keeps a different set of rules for each backend, because the code it generates is
different: Triton kernels on the GPU, C++ on the CPUs, and a graph shape that is good for one is not good
for the other. Each rule also checks what device its tensors are on. So on the GPU there are rules that only
exist there, and the rules run in a fixed order, each changing the graph the next one sees. On BERT, a
GPU-only rule that folds a scaling step into the softmax fires first, rewrites exactly the piece the
attention fusion rule was looking for, and the fusion rule then finds nothing. On the two CPUs that first
rule does not exist, so fusion fires.

**The stretch goal: guards.** A guard is a small learned model that sits in that same check function and
blocks a rule where it will hurt. To train it I needed data. I generated small synthetic networks across a
grid of sizes, and for each one measured the model with the rule on and off, twice. One row of the dataset
looks like this: a convolution network with 128 channels, 28 by 28 input, three depthwise layers, on the
Mac; 12.7 milliseconds with the layout rule on, 10.3 with it off, so 23 percent slower, confirmed in both
runs; label: harmful. The inputs to the guard are those size numbers plus which machine. About 1,700 rows
across the three machines.

**How I chose the guard.** I compared a decision tree, gradient-boosted trees, logistic regression, a
fine-tuned text model, and OpenAI's decision API, all on the same held-out data, scoring each by how much
run time is lost per firing if you follow its decisions, compared with always applying the rule, which is
what the compiler does now. The best per rule family went into the compiler: boosted trees for two families,
logistic regression for the layout family. If the guard says more than fifty percent chance of harm, the rule
is skipped.

On the two CPUs the guards work. Per firing they lose about one percent, where always applying the rule
loses up to 260 percent. On real models: MobileNet 29 percent faster and EfficientNet 41 percent faster on the
Mac, GPT-2 decode 73 percent faster on the Xeon, nothing slower.

On the GPU the guard did not transfer, and I know why. The synthetic networks have one to three layers. Real
networks have twenty to eighty. The guard was asked about inputs unlike anything it had seen and answered
"harmless". I fixed a noise problem in the GPU dataset and retrained, and it still did not transfer. That is
the limitation I will state in the report: a guard trained on synthetic data only works when the real cases
look like the training cases.

**Where it stands.** All the measuring is done on three machines. The GPU work cost nineteen dollars of
credits. Code and notes are on GitHub. What is left is the final report, due November 19.

## If asked
- Why do the rule counts differ per machine (269, 396, 398)? The compiler only sets up the rules that can run
  there: the Xeon build has Intel oneDNN fusion rules, the GPU build has CUDA-only rules.
- How many tests? About 140, including small models with a planted slowdown where the tool must name the
  right rule.
- Could this go upstream? The switch mechanism answers an open PyTorch request for pattern-matcher
  observability, and the freezing and layout findings are concrete issues to file.
