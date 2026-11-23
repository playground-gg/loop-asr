# Decisions

Owner decisions that override or settle points in `plan.md`. Newest first. `plan.md`,
`brief.md` and `research.md` are kept as written; where they disagree with this file, this
file wins.

## 2026-11-23

**(a) Arm B samples K from {2, 4, 6, 8, 10}, K_max = 10.**
plan.md section 3 says "per K_max=8, campiona K in {2,4,6,8,10}", which contradicts itself;
10 wins. Weights stay 1:2:6:2:1. `configs/arm_b.yaml`: `default_loops: 10`,
`loops_sample: [2, 4, 6, 8, 10]`. Arm C stays at 18 blocks, matched to B at **K=8** as the
plan's table says (B at K_max=10 runs 22 blocks per frame).

**(b) The A LR sweep runs at half steps; the winner is rerun at full steps (`A_best`) before
the go test.**
Sweep values 3e-4, 7e-4, 1.5e-3 (plan.md section 4) in `configs/arm_a.yaml`
(`lr_sweep`, `lr_sweep_steps_fraction: 0.5`). `configs/arm_a_best.yaml` is the full-step rerun,
`lr_peak: null` until the sweep has run. Go criterion H1 compares B against A_best.

**(c) The pilot reports dev WER only.**
test-clean / test-other and int8 quantisation come once, only after a go. Every arm config's
`eval.splits` is dev-clean and dev-other.
