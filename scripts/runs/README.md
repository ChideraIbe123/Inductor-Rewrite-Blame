# Run recipes (provenance of the results, not reusable tools)

Each script here is the exact command sequence behind one block of measurements in `docs/results_notes.md`.
They are kept so every number in the report can be traced to how it was produced. The reusable tools live
one directory up (`scripts/`).

| script | machine | what it produced |
|---|---|---|
| mac_chain*.sh, mac_cool3.sh, mac_toys_clean.sh, mac_redo_hot*.sh | Mac | corpus sweeps (noise, verify, sweep) in cool sessions; re-measurements after throttling |
| mac_item5.sh, mac_item5b.sh, mac_pairs_quiet.sh | Mac | attribution case studies, pair scans, tau-scans |
| mac_dataset.sh, mac_guards_final.sh, mac_e2e_small_redo.sh | Mac | guard dataset, guard training, end-to-end guard evaluation (daytime redo) |
| vm_run.sh, vm_chain2.sh, vm_item5.sh, vm_item6*.sh | Xeon VM | corpus sweeps, case studies, repeat sessions, pair scan |
| vm_dataset.sh, vm_dataset_resume.sh, vm_guards_final.sh | Xeon VM | guard dataset (resumable across the daily reboot), two-CPU training, x86 end-to-end |
| vm_pinning_study.sh, vm_pinning_study2.sh | Xeon VM | core-pinning noise study (study 1 invalid: OMP_PROC_BIND; study 2 taskset only) |
| gpu_after_p1.sh, gpu_phase2b.sh, gpu_phase34_redo.sh, gpu_dataset_v2.sh | A40 pod | phases 2-4 chaining, GPU-specific attributions, clean redo, GPU-sized conv dataset + guard + end-to-end |
