# Decisions

Decisions that override or settle points in `plan.md`, newest first. `brief.md`, `plan.md`,
`research.md` and `idea.md` change only by PR, and every such change gets an entry here.
Where a source doc and this file disagree, this file wins.

## 2026-11-23, plan v2: lead decisions 1–8

The owner asked to "develop the whole plan, the idea and the execution plan, merge; we run it
another time; we are pushing the frontier of tiny STT". The owner did not pick the ambition
level or the fixes when asked. The project lead chose the defaults below.
**Each one is a lead default, and the owner can overturn it.** They come from the frontier
research in `docs/research/` (summary in `research.md`). brief.md, plan.md and research.md were
rewritten on top of them, and idea.md was added. The previous research.md (the Inkling 2 Mini
study, Italian) moved unchanged to `docs/research/inkling_2_mini_study.md`. The new docs are in
English, like the research they build on.

**1. Framing.** *Lead default, owner can overturn.*
- The claim: one tiny looped model (~3.6M), trained once with sampled K, usable at any k at
  inference. Komatsu 2022 and LARM 2026 already published the generic "looped beats plain on
  100h", so that alone is not the claim.
- It is compared head-to-head with Komatsu/LARM-style baselines and measured as real CPU and
  MPS wall-clock, not only FLOPs.
- In the matched setup, the head-to-head is a new arm, **B_fix**: B's encoder trained at a
  fixed K=6 and decoded at any k, which is Komatsu-style folding and LARM-style fixed K
  (`configs/arm_b_fix.yaml`). Published LARM/Komatsu numbers are external references on test
  sets. A full LARM-recipe arm (carried state, FiLM, feedback) is stage 2.
- Target output: an honest blog post plus an HF release. A workshop paper is stage 2, only
  after a go. It needs 3 seeds, C with early exits and per-k specialists.

**2. Fair size.** *Lead default, owner can overturn.*
- PR #1's B used one gate for the whole core: +16,896 params (+0.47% vs A). **The research's
  "+0.5M / +15%" concern does not apply to our code.** It holds only for an Inkling-faithful
  gate: 4 gates per core layer, rank 64, a 2d input.
- Applied as written: **one gate per core layer, rank 32.** B = 3,608,513 against A =
  3,574,721, i.e. **+33,792 params (+0.95%)**.
- `tests/test_smoke.py::test_b_within_3_percent_of_a_params` enforces ≤ +3%.
- The gate keeps PR #1's additive form (`h + up(silu(down(LN(m))))`, `up` zero-init). Each
  core layer reads the previous pass's output of the same layer. Inkling's tanh
  forget/message form is a stage-2 ablation.

**3. Fair tuning.** *Lead default, owner can overturn.* Supersedes (b) below where they differ.
- Every arm (A, B, C) gets the same sweep: LR ∈ {3e-4, 7e-4, 1.5e-3} at half steps, chosen on
  dev-clean. Each winner is rerun at full steps: `arm_{a,b,c}_best.yaml`, `lr_peak: null`
  until the sweep has run.
- B_fix reuses B's winner and has no sweep of its own. That is a small bias in B's favour,
  written down in plan.md §4.
- The total fits the 25 h pilot (plan.md §6), so the "A and B only" fallback is not used. It is
  the first cut if step 0 projects over 30 h, and the C-tuning bias must then be written down.

**4. Fair compute.** *Lead default, owner can overturn.* Supersedes the "C = 18 blocks, matched
at K=8" part of (a).
- Picked: **C set to B's mean-K FLOPs.** The weights 1:2:6:2:1 on {2,4,6,8,10} have mean = mode
  = 6, so C has 2 + 2·6 = **14 blocks** (11,472,321 params).
- Why not move the mean to 8:
  - The symmetric 1:2:6:2:1 shape needs the support {4,…,12}. That drops k=2 from training
    and moves K_max to 12, but "any k" needs low-k mass.
  - Skewed weights on {2,…,10} (e.g. 1:1:2:6:2) reach a mean of only 7.2.
  - Keeping the weights also keeps Inkling's schedule and decision (a)'s K set.
- Ratios, counted on CPU (`forward_flops`, 15 s utterance) [M]:
  - **Training compute** (forward FLOPs per step, expected over K): B : C = **1.011** without
    C's exit heads, **1.002** with them (as C actually trains). A : C = 0.340.
  - **Inference compute:** B(k) : C@(2+2k) = **1.005 / 1.009 / 1.011** at k = 2 / 4 / 6.
    B(6) : C = 1.011, B(4) : C = 0.743, B(8) : C = 1.280, B(10) : C = 1.548.

**5. Fair speed test.** *Lead default, owner can overturn.*
- Implemented as an evaluation of the **same C checkpoint**, not a separate arm C'. C has CTC
  exits at blocks 6/10/14 through its shared final norm + head, which are B's depths at
  k = 2/4/6. It trains on the mean of the three exit losses.
- An exit truly stops the computation (`depth=`), so its CPU latency is what a deployed
  early-exit model pays.
- The exit loss is a deliberate exception to "arms differ only in the encoder". Each
  any-depth model gets its native any-depth training: sampled K for B, joint exits for C.
- B(k) at k < 6 is compared with C@(2+2k) at equal FLOPs, in WER and measured CPU latency.

**6. Go criterion.** *Lead default, owner can overturn.* Supersedes plan v1 §7. Dev-clean and
dev-other, greedy CTC, CPU latency at 4 threads, batch 1, median over 200 fixed dev-clean
utterances:
> **GO** if all three hold:
> 1. **Size (primary):** WER(B(6)) ≤ 0.80 × WER(A_best) on dev-clean **and** ≤ 0.85 × WER(A_best) on dev-other.
> 2. **Any k:** on dev-clean, WER(B(4)) ≤ 0.85 × WER(A_best) **and** WER(B(4)) ≤ 1.15 × WER(B(6)).
> 3. **Equal compute:** for both k = 4 and k = 6, on dev-clean, WER(B(k)) ≤ 1.10 × WER(C@(2+2k)) (C@10, C@14), **and** B(k)'s measured CPU latency ≤ 1.10 × C@(2+2k)'s.

- Latency bar raised from 1.05 to 1.10 after review (2026-11-23): B's gate ops at batch 1 plus
  its +1% FLOPs are predicted to cost ~4% (1.00–1.10), so a 1.05 bar would test op overhead,
  not the idea.
- Re-derived from the research's predictions (plan.md §7): P(go) ≈ 0.25 (0.2–0.3) = 0.8 (1)
  × 0.75 (2 given 1) × 0.5 (3 WER, k=6 at 1.08 against 1.10) × 0.85 (3 latency), against
  ≈ 0.15–0.2 for v1. With the 1.05 latency bar it would have been ≈ 0.15.
- "B clearly beats size-matched A" stays the primary claim.

**7. Predictions.** *Lead default, owner can overturn.*
- `experiments/pilot/README.md` has per-arm predictions with 80% ranges from the research
  (A 32, A_best 30, B(4) 21, B(8) 19, C 18 dev-clean), plus what would be surprising. They are
  committed before any run.
- C keeps 18 even though it is now 14 blocks deep, not 18. The shallower depth (≈ +3%) is
  offset by the exit loss (InterCTC: −5 to −15%).

**8. Step 0.** *Lead default, owner can overturn.* Before any training run, plan.md §6:
- a torch 2.15 MPS ctc_loss check (MPS vs CPU gradients on real batches; a random-data
  pre-check here gave 2.4e-5);
- FLOPs counted on CPU only (`loop_asr.forward_flops`);
- the reference deps (ffmpeg for Whisper, moonshine-voice);
- `get_device` now raises when MPS is requested but missing, and warns on an automatic CPU
  fallback on a Mac;
- clips under 7 frames, or failing `ctc_feasible`, are filtered out;
- the budget reconciled to **one number: 25 h** of Mac time (step 0 3 h, sweeps 10.6 h, full
  runs 10.1 h, eval 1.5 h). The cut order applies above 30 h projected.

## 2026-11-23

**(a) Arm B samples K from {2, 4, 6, 8, 10}, K_max = 10.**
plan.md section 3 says "per K_max=8, campiona K in {2,4,6,8,10}", which contradicts itself;
10 wins. Weights stay 1:2:6:2:1. `configs/arm_b.yaml`: `default_loops: 10`,
`loops_sample: [2, 4, 6, 8, 10]`. Arm C stays at 18 blocks, matched to B at **K=8** as the
plan's table says (B at K_max=10 runs 22 blocks per frame).
*Partly superseded by decision 4: C is now 14 blocks, matched at B's mean K = 6.*

**(b) The A LR sweep runs at half steps; the winner is rerun at full steps (`A_best`) before
the go test.**
Sweep values 3e-4, 7e-4, 1.5e-3 (plan.md section 4) in `configs/arm_a.yaml`
(`lr_sweep`, `lr_sweep_steps_fraction: 0.5`). `configs/arm_a_best.yaml` is the full-step rerun,
`lr_peak: null` until the sweep has run. Go criterion H1 compares B against A_best.
*Extended by decision 3: B and C get the same sweep.*

**(c) The pilot reports dev WER only.**
test-clean / test-other and int8 quantisation come once, only after a go. Every arm config's
`eval.splits` is dev-clean and dev-other.
