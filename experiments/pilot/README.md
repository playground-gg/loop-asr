# pilot: A, B, B_fix and C on train-clean-100, dev only

Status: **not run.** These predictions were written and committed on 2026-11-23, before any
run (CLAUDE.md rule; DECISIONS.md decision 7). After the runs, the outcomes go next to them.
The predictions themselves do not change.

## Goal

Test H1–H6 of `docs/plan.md` §1 and decide go or no-go against `docs/plan.md` §7.

## Predictions

Dev WER %, greedy CTC, 15 epochs, light SpecAugment, BPE-256, at each arm's swept LR. The
point is the median, and the range is an 80% interval. Sources: `docs/research/frontier.md` §4
(A 32, A_best 30, B(4) 21, B(8) 19, C 18), extended to the exits and to B_fix by the same
reasoning. All [E].

| arm / read-out | FLOPs vs C@14 | dev-clean | 80% range | dev-other | 80% range |
|---|---|---|---|---|---|
| A at LR 7e-4, full steps (exists only if 7e-4 wins the sweep) | 0.34 | 32 | 26–40 | 58 | 50–68 |
| **A_best** | 0.34 | **30** | 25–37 | **55** | 47–65 |
| B(1) (never trained at k=1) | 0.34 | 50 | 30–80 | 72 | 55–95 |
| B(2) | 0.47 | 27 | 21–35 | 52 | 44–62 |
| **B(4)** | 0.74 | **21** | 17–27 | **45** | 38–54 |
| **B(6)** (mean and mode of training K) | 1.01 | **19.5** | 15.5–25 | **43** | 36–51 |
| **B(8)** | 1.28 | **19** | 15–24 | 42 | 35–50 |
| B(10) | 1.55 | 19 | 15–25 | 42 | 35–51 |
| B(12), B(14), B(16) (extrapolation) | 1.8–2.4 | 19–23 | 15–30 | 42–48 | 35–60 |
| C@6 | 0.47 | 26 | 20–34 | 52 | 43–62 |
| **C@10** | 0.74 | **21** | 16.5–27 | 46 | 38–55 |
| **C@14** (full C) | 1 | **18** | 14–23 | **40** | 33–48 |
| B_fix(2) | 0.47 | 40 | 25–60 | 65 | 48–85 |
| B_fix(4) | 0.74 | 25 | 19–35 | 50 | 41–62 |
| B_fix(6) (its training K) | 1.01 | 19 | 15–24 | 42 | 35–50 |
| B_fix(8) | 1.28 | 22 | 17–32 | 46 | 37–58 |
| B_fix(10) | 1.55 | 26 | 18–40 | 50 | 39–65 |

**Reasoning, short (full version in frontier.md §4):**
- **A:** LARM's 4-block model, scaled for 15 vs 50 epochs and d=256 vs 384. The sweep buys
  3–10%.
- **B(6) and B(8):** naive looping gave LARM −53% from 4 unique blocks. We have less depth
  and only 2 shared core blocks, and B starts life as A. So −35% vs A_best.
- **The k-curve is flat from 6 to 10:** Inkling varloop is flat above its mode, and LARM
  saturates. k=4 is +8% (1/6 of the training mass). k=2 is bad (1/12 of the mass, like
  Inkling's 1-pass collapse). k=1 is out of distribution.
- **C:** 14 blocks rather than the research's 18 (≈ +3%), offset by the exit loss (InterCTC
  −5 to −15%), so 18 is kept. The exits are scaled from LARM T4 and Wright 2309.09546.
- **B_fix:** at its own K it is the specialist, so slightly better than B(6). Away from K=6
  it degrades as in Komatsu ("decode repeats ≠ train repeats is worse") and in LARM's
  K=24 exit-4 result. It has no intermediate loss, so it degrades faster than LARM's exits.

**Go-criterion read-outs (plan.md §7):**

| criterion | predicted | threshold | P(pass) |
|---|---|---|---|
| 1. B(6)/A_best | 0.65 clean, 0.78 other | ≤ 0.80, ≤ 0.85 | 0.8 |
| 2. B(4)/A_best; B(4)/B(6) | 0.70; 1.08 | ≤ 0.85; ≤ 1.15 | 0.75 given 1 |
| 3. B(4)/C@10; B(6)/C@14 (WER) | 1.00; 1.08 | ≤ 1.10 each | 0.5 given 1–2 |
| 3. CPU latency, B(k)/C@(2+2k) | 0.97 (0.85–1.10) | ≤ 1.05 | (included above) |
| **go** | | | **≈ 0.3** |

**Other predictions:**
- **H4:** B(4)/B_fix(4) = 0.84 and B(8)/B_fix(8) = 0.86, both ≤ 0.90. B(6)/B_fix(6) = 1.03,
  so sampling costs ≤ 5% at the specialist's own depth.
- **H5:** B(12) ≤ 1.05 × B(10) (P ≈ 0.6). B(16) is worse than B(10).
- **H6:** the gate ratio rises from 0 to 0.1–0.5 within the first 2 epochs and stays there.
- **CPU latency:**
  - B(4) vs full C: 1.25–1.35× faster. The FLOP ratio is 1.35, diluted by per-op overhead.
  - MPS: smaller gaps than CPU, because launch overhead dominates at batch 1.
- **Sweep winners:** 1.5e-3 for A (the shallow model tolerates a higher LR), 7e-4 for B and
  C. Low confidence.
- **References (dev-clean):** Whisper tiny.en ≈ 5.5 and Moonshine Tiny ≈ 4.5, both about 5–20×
  slower than B(4) on CPU per utterance.

## What would be surprising

- B(4) ≤ 0.95 × C@10, or B(6) ≤ 0.90 × C@14: looping beats plain exits at equal compute.
- B_fix(4) ≤ B(4): sampling K buys nothing over fixed-K folding.
- B improves past k=10 (extrapolation works).
- B(6) ≤ 15 on dev-clean at 3.6M.
- A_best ≤ 22. LARM's baseline would then be badly tuned and its looping gain inflated.
- B(6) > 0.80 × A_best. That contradicts LARM and Komatsu. Suspect a bug or a gate that never
  opens.
- The gate ratio stays near 0 while B still beats A. Under the restart rule that is
  impossible, so it means a bug.
- B(k)'s CPU latency > 1.10 × C@(2+2k). Weight reuse would then cost wall-clock time.

## Setup

- Configs: `configs/arm_{a,b,c}.yaml` (sweeps), `configs/arm_{a,b,c}_best.yaml` (full steps),
  `configs/arm_b_fix.yaml`, at the commit that adds this file. The `lr_peak` values get filled
  after the sweeps, in a separate commit.
- Hardware and software: Apple M5, 24 GB, MPS, torch 2.15.0, fp32 eager, seed 0.
- Budget and run table: `docs/plan.md` §6, 25 h in total.

## Outcome

*(empty until the runs)*
