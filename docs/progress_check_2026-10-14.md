# Progress check, 14 Oct 2026
Project: Which Rewrite Made My Model Slower? (Chidera Ibe)

## The problem in plain words
PyTorch has a compiler called torch.compile. You give it a model, and it produces faster code for that
model. Part of how it does this is a list of a few hundred small "rewrite rules". Each rule looks for a
certain pattern in the model and replaces it with something it thinks is faster. For example, one rule
finds the math of attention and replaces it with one combined operation.

The problem: the compiler applies a rule every time the pattern appears. It never checks whether the
change actually made the model faster on the computer it is running on. Sometimes a rule makes the model
slower. When that happens, the person using the compiler cannot tell which rule caused it.

## What I built
1. **An on/off switch for every rule.** I can turn any single rule off and compile the model again. This
   did not need any changes to PyTorch itself. The switch also counts how many times each rule was used.
   There are 269 rules on my Mac, 396 on the course Linux machine, and 398 on the GPU.
2. **A measuring tool.** It compiles a model with any set of rules turned on or off, checks that the
   result is still correct, records what code was produced, and measures how long the model takes to run.
   I test it on 24 models: small hand-made ones, image models like ResNet, and language models like BERT,
   GPT-2 and Qwen.
3. **A rule for telling real slowdowns from random noise.** Running the same model twice never gives
   exactly the same time. I ran the same setup hundreds of times to see how much the times move on their
   own. From that I set a limit. A change only counts as a slowdown if it is bigger than that limit in
   two separate runs taken at different times.
4. **The blame tool.** You give it a fast setup and a slow setup. It turns rules on and off in a planned
   order, measuring each time, until it finds the smallest set of rules that makes the model slow. It
   also notices when two rules are only harmful together. It then shows what changed in the produced code.
5. **Guards.** A guard is a small learned model that sits in the compiler and decides, each time a rule
   is about to fire, whether to let it. I built a dataset of about 1,500 rule firings with the measured
   effect of each one, and trained one guard for each of three rule families.
6. About 140 automated tests. Some tests use small models where I planted a known slowdown and check
   that the tool finds the right rule.

## What I found
- **The same rule helps on one computer and hurts on another.** The attention rule makes a model 34%
  faster on my Mac and 8% slower on the Linux machine. The rule that changes how images are stored in
  memory makes MobileNet 41% slower and EfficientNet 68% slower on the Mac, but helps on the Linux machine.
- **Two rules can hurt only together.** Attention rules number 11 and 12 match the same part of the
  model. Turning off one changes nothing. Turning off both changes the code. The tool finds this pair.
- **The worst single rule** is the one that splits large matrix multiplications. When it is on, SmolLM
  text generation is 135% slower and Qwen is 39% slower.
- **The results repeat.** On the same computer, 213 of 220 verdicts came out the same on a different day.
  Across the two computers most verdicts differ, and that difference is real, not noise.
- **The guards work.** Per rule firing, a guard loses about 1% of run time. Letting every rule fire, which
  is what the compiler does today, loses 176% to 261% for the two worst families. On real models the
  guard made MobileNet 29% faster, EfficientNet 41% faster, and SmolLM text generation 58% faster.
  No model became slower.
- **First GPU result, from today.** On the GPU, the attention rule never fires on my small attention
  model. Two other rules that only exist for GPUs fire first and change the pattern before it can match.

## What is running right now
- **On a rented GPU (NVIDIA A40, about 60 cents an hour, about $20 in total).** The measuring tool is
  going through all 23 models overnight. After that it will run the blame tool on the same cases I ran on
  the two CPUs, build a GPU dataset for the guards, and repeat the measurements a second time.
- **On the course Linux machine.** It is finishing the last part of the guard dataset. Then it will train
  guards on the data from both CPUs and test them on real models.

## What comes next
Collect the GPU results, test the guards across all three computers, and write the final report, due
19 November. The report will say: the switches and the blame tool are the main contribution, the
finding that rules flip between helpful and harmful across hardware is the reason the tool is needed,
and the guards show that once you know which rule is harmful you can fix it.
