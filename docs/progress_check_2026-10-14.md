# Progress check, 14 Oct 2026
Project: Which Rewrite Made My Model Slower? Finding and Blaming Bad Graph Rewrites in TorchInductor (Chidera Ibe)

## The problem
torch.compile rewrites a model's graph with a few hundred small rules (fuse attention into one kernel,
change a tensor's memory layout, split a matmul, and so on). Each rule fires whenever its pattern
matches. Nothing checks whether the rewrite actually made the model faster on this machine. When a
compiled model is slower than expected, an engineer has no way to find out which rule did it.

## What is built (all code, tested, on GitHub)
1. **Per-rule switches.** Every one of Inductor's pattern rules can be turned off individually without
   patching PyTorch, by wrapping the hook Inductor already calls before applying a rule. The harness
   also counts how often each rule fired or was suppressed. 269 switches on the Mac build, 396 on the
   x86 VM, 398 on the GPU.
2. **A measurement harness** that compiles a model under any set of switches in a fresh process,
   records the generated kernels, checks the output against eager mode, and times it. A corpus of
   24 models: toys, ResNet/MobileNet/EfficientNet/ConvNeXt/ViT, BERT/DistilBERT/GPT-2/T5, SmolLM and
   Qwen2.5 prefill and decode.
3. **A noise method** chosen by calibration, not assumption: 480 repeated runs of identical
   configurations showed which statistics give false alarms. The result: median per run, threshold
   = 3 x MAD over 7 runs (floor 1%), and a slowdown counts only if two time-separated runs both
   exceed it. Byte-identical generated programs are never blamed.
4. **The blame tool.** Given a fast and a slow switch set, delta debugging over the differences finds
   the smallest set of rule changes that reproduces the slowdown, flags pairs that only hurt together,
   iterates until the remaining gap is within noise, and explains the result at kernel level.
5. **Guards (the stretch goal).** A dataset of ~1,500 labelled rule firings across both CPUs
   (shape features at the firing, measured effect of blocking it), and a small decision model per
   rule family that runs inside the compiler and blocks a firing it predicts to be harmful.
6. About 140 automated tests, including graphs with planted slowdowns whose culprit is known.

## What was found
- **The same rewrite flips sign across machines.** Inductor's attention fusion is a 34% win on the
  Apple M4 Pro and an 8% loss on the Xeon VM. Channels-last layout costs MobileNet 41% and
  EfficientNet 68% on the Mac but helps ResNet, and is mostly helpful on the Xeon.
- **A real pair interaction exists:** attention rules 11 and 12 match the same graph, so turning off
  either alone changes nothing; only both together change the program. The tool finds it.
- **The matmul decomposition pass is the biggest single offender:** +135% on SmolLM decode, +39% on
  Qwen decode when enabled.
- **Repeatability:** 213 of 220 verdicts agree across sessions on the same machine; across machines
  42 of 47 disagree, and those disagreements are real hardware differences, not noise.
- **Guards work.** Per firing, the learned guard loses about 1% of runtime versus 261% (matmul
  decomposition) and 176% (channels-last) for Inductor's always-apply behaviour. End to end on real
  models: MobileNet 29% faster, EfficientNet 41% faster, and with the decomposition pass enabled
  SmolLM decode 58% and Qwen decode 24% faster. No model got slower.
- **First GPU result (today):** on an NVIDIA A40 the toy attention block is never fused into SDPA at
  all; two GPU-only rewrites fire first. Six planted tests that encode CPU behaviour had to be marked
  CPU-only.

## What is running right now
- **GPU campaign on RunPod (A40, about $0.60/h, ~$20 total):** noise, verify and sweep over the 23
  corpus models (overnight), then the same attribution case studies as the CPUs, a GPU guard dataset
  for the conv and attention families, and a repeat session. This turns the two-CPU comparison into a
  two-CPU-plus-GPU one.
- **VM:** finishing the attention half of the x86 guard dataset, then retraining the guards on both
  CPUs' data and running the end-to-end evaluation on x86 models.

## Next
Pull the GPU results, train and evaluate guards on three machines, and write the final report
(due 19 Nov). The write-up argues the per-rule switch mechanism and the blame tool as the contribution,
the cross-hardware sign flips as the motivating finding, and the guards as proof that the blame is
actionable.
