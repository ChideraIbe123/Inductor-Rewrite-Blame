# Progress check, 14 Oct 2026
Project: Which Rewrite Made My Model Slower? (Chidera Ibe)

## 1. The problem
PyTorch has a compiler called torch.compile. You give it a model and it produces faster code for it.
Part of how it does this is a list of a few hundred small "rewrite rules". Each rule looks for a pattern
in the model and replaces it with something it expects to be faster. One rule, for example, finds the
math of attention and replaces it with a single combined operation.

The compiler applies a rule every time its pattern appears. It never checks whether the change made the
model faster on the computer it is running on. Sometimes a rule makes the model slower. When that
happens, the person using the compiler cannot tell which rule caused it. That is the problem.

## 2. The switches: how I turn one rule on or off
Inside the compiler, every rule has a small function attached to it called the "extra check". After the
compiler finds a match for the rule's pattern, it calls this function, and the rule is applied only if
the function answers yes.

I replace that function with my own. Mine first looks at a list of switches. If the rule's switch is
off, it answers no, and the compiler skips the rule. If the switch is on, it calls the original function
and passes the answer through. It also counts: once for every time the rule was applied, and once for
every time it would have been applied but was blocked. No PyTorch code is changed.

Some things that are not pattern rules are controlled differently: compiler settings such as "fuse
operations after a matmul" are set through the compiler's configuration object, and optional passes are
turned on through the compiler's options dictionary. In total each of these is one switch.

To find all the rules, I first compile a tiny model that contains attention, a convolution, a split and
concatenate, a matmul and a type conversion. This makes the compiler register all of its rules. Then my
code walks through the compiler's modules, finds every rule object, and gives it a name.

Every measurement runs in its own fresh process, so switch settings never leak from one measurement to
the next. The compiler's cache of previously compiled code is turned off during measurement, so the
compiler really runs every time.

## 3. Why the number of rules differs by computer
The compiler only registers rules that can work on the computer it is running on.

- Mac (Apple M4 Pro): 269 switches. This PyTorch build has no Intel math library and no GPU.
- Linux machine (Intel Xeon): 396 switches. This build includes Intel's oneDNN library, and the
  compiler adds rules that merge a convolution or a linear layer with the operation that follows it
  into a single oneDNN call. Those rules do not exist on the Mac.
- GPU (NVIDIA A40): 398 switches. This build adds GPU-only rules: padding matrix sizes so the GPU's
  matrix units can be used, moving a bias addition out of a matmul so it fuses with the next operation,
  and a second copy of the attention rules written for GPU and for half precision.

Because the lists differ, my tool compares setups by their difference from the default, not by rule
numbers, so results from the three computers can be put side by side.

## 4. The measuring tool
It compiles a model with any set of switches, then:
- checks that the output matches the uncompiled model, so a "fast" setup is never a wrong one;
- records the code the compiler produced: how many kernels (the small programs it generated), how
  much memory they touch, and which operations were left to outside libraries;
- times the model: 10 warm-up calls, then 50 timed calls, and keeps the median.

The corpus is 24 models: six small hand-made ones, image models (ResNet-18, ResNet-50, MobileNet,
EfficientNet, ConvNeXt, ViT), and language models (BERT, DistilBERT, GPT-2, T5, SmolLM, Qwen2.5), the
language models both for processing a prompt and for generating one token. The models use random
weights, so nothing has to be downloaded, and the speed does not depend on the weights.

## 5. The noise limit: how I decided when a slowdown is real
Running the same model twice never gives exactly the same time. So I needed a limit, and a change only
counts if it is bigger than the limit.

How the limit is set for one model on one computer:
1. Measure the default setup 7 times, each in its own process, spaced out in time. Each measurement is
   the median of 50 timed calls, so I have 7 numbers.
2. Take the median of the 7 numbers. For each number, take its distance from that median. The median of
   those 7 distances is called the MAD.
3. The limit is 3 times the MAD, or 1% of the run time, whichever is larger.

How I chose this rule instead of others: I ran 480 measurements where nothing was changed at all, so any
"slowdown" found would be a false alarm. I tried ten different ways to decide (standard deviation,
coefficient of variation, interquartile range, comparing single runs, and so on) and counted the false
alarms of each. I also tested whether cutting off the first slow calls of a run helps, instead of
assuming it. The rule above had the fewest false alarms while still catching the slowdowns I had planted.

Two more safety rules. First, a setup is called slower only if two separate measurements, taken at
different times with a fresh measurement of the default between them, both exceed the limit. Second, if
the compiler produced byte-for-byte the same code for two setups, they cannot differ in speed, so the
tool never blames a rule whose switch did not change the code.

## 6. The blame tool: how attribution works
Input: a fast setup and a slow setup. The difference between them is a list of changes, for example
"rule A turned off, rule B turned off, pass C turned on". The question is which of those changes cause
the slowdown.

The method is delta debugging. It works like this:
1. Split the list of changes into two halves. Apply the first half to the fast setup and measure. Apply
   the second half and measure.
2. If one half alone makes it slow, keep only that half and go back to step 1 with the shorter list.
3. If neither half alone does, split into smaller pieces and try removing one piece at a time. Keep
   any removal after which the setup is still slow.
4. Stop when no single change can be removed without the slowdown disappearing. What is left is the
   smallest set of changes that reproduces the slowdown.

Every "measure" in these steps is decided by the noise rule from section 5, comparing against the fast
setup. Results are stored, so a setup is never measured twice.

After that:
- If the final set has two or more changes, each one is tested alone. If none of them alone makes it
  slow, the tool reports them as a pair that only hurts together.
- The tool then adds the found cause to the fast side and runs again, to find the next cause, until the
  remaining gap is inside the noise limit. This is how it finds several causes in one slow setup.
- For the final answer it shows which kernels appeared or disappeared, how memory traffic changed, and
  which operations moved to outside libraries, so you can see why the slow setup is slow.

A typical case needed 5 to 16 measurements and under a few minutes on the CPUs.

## 7. The guards: what they are and how they are built
A guard is a small decision model that runs inside the compiler. It sits in the same "extra check"
function from section 2. When a rule is about to be applied, the guard looks at the situation and
predicts whether this firing would make the model slower. If it predicts harm, the function answers no
and the rule is skipped.

There is one guard per rule family. The three families are:
- attention fusion (the rule that combines the attention math into one operation),
- image layout change (the rule that stores images channels-last in memory),
- matmul split (the rule that breaks a large matrix multiplication into smaller pieces).

What the guard sees (its inputs): the sizes of the tensors at the point where the rule matched. For
attention: batch size, sequence length, number of heads, head size. For layout: channels, image size,
kernel size, number of layers, whether convolutions are depthwise. For matmul: the three matrix
dimensions. Plus which computer it is (Mac, x86, GPU) and the thread count.

What the guard is: gradient-boosted decision trees, which is a set of small decision trees whose
votes are added up. I also tried a single decision tree, logistic regression, a fine-tuned text model
(Laya), and OpenAI's Decisions API given the same numbers as text. The boosted trees had the lowest
measured cost, so they are what is in the live guard.

How the training data was made: for each family I generated many small models with different sizes,
and for each one measured the model with the rule on and with the rule off, twice, in separate runs.
Each example is labelled harmful, helpful or neutral using the noise limit. About 1,500 examples across
the two CPUs so far; the GPU set is being collected now.

How the guards are scored: I hold out whole groups of sizes the guard has never seen and measure the
run time lost per firing by following its decisions, compared with the compiler's current behaviour of
always applying the rule.

## 8. What was found
- The same rule helps on one computer and hurts on another. The attention rule makes a model 34%
  faster on the Mac and 8% slower on the Linux machine. The layout rule makes MobileNet 41% slower and
  EfficientNet 68% slower on the Mac, but mostly helps on the Linux machine.
- Two rules can hurt only together. Attention rules 11 and 12 match the same part of the model.
  Turning off one changes nothing. Turning off both changes the code. The blame tool finds this pair.
- The worst single rule is the matmul split. When it is on, SmolLM token generation is 135% slower and
  Qwen is 39% slower.
- The results repeat. On the same computer, 213 of 220 verdicts came out the same on another day.
  Across the two computers most verdicts differ, and that difference is real, not noise.
- The guards work. Per rule firing, the guard loses about 1% of run time. Always applying the rule,
  which is what the compiler does today, loses 176% to 261% for the two worst families. On real models
  the guard made MobileNet 29% faster, EfficientNet 41% faster, and SmolLM token generation 58% faster.
  No model became slower.
- First GPU result, from today: on the GPU, the attention rule never fires on my small attention model.
  Two rules that only exist for GPUs fire first and change the pattern before it can match.

## 9. Testing
About 140 automated tests. The fast ones check the pieces on their own: the delta debugging search on
made-up data where the answer is known, the noise rule, the code statistics, the storage. The slower ones
compile small models where I planted a known slowdown, and check that the tool blames the right rule,
that each switch really changes the code, and that outputs stay correct under every switch setting.

## 10. What is running right now
- On a rented GPU (NVIDIA A40, about 60 cents an hour, about $20 in total): the measuring tool is going
  through all 23 models overnight. After that it will run the blame tool on the same cases I ran on the
  two CPUs, build a GPU dataset for the guards, and repeat the measurements a second time.
- On the course Linux machine: it is finishing the last part of the guard dataset. Then it trains the
  guards on data from both CPUs and tests them on real models.

## 11. What comes next
Collect the GPU results, test the guards across all three computers, and write the final report, due
19 November. The report will say: the switches and the blame tool are the main contribution; the finding
that rules flip between helpful and harmful across hardware is the reason the tool is needed; and the
guards show that once you know which rule is harmful, you can fix it.

## 12. Words used above
- Kernel: one small program the compiler generates to run part of the model.
- Fired: a rule matched and was applied.
- Default setup: the compiler with all its normal settings, nothing changed.
- Sweep: turning switches off one at a time for one model and measuring each.
- Session: a block of measurements taken on one day; times are only compared inside a session.
