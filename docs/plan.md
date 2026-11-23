# Experiment plan v2: one tiny looped CTC model, any k, on a Mac

v2, 2026-11-23. It replaces plan v1 (Italian, in git history at commit fe708ca) and builds on
the frontier research (`research.md`, `research/frontier.md`). It folds in
`DECISIONS.md` items (a)–(c) and lead decisions 1–8. This file changes only by PR, with a
DECISIONS entry. No run has happened yet.
Tags: **[V]** verified in a primary source; **[M]** measured in this repo; **[E]** estimate;
**[C]** conjecture.

## Summary

- **Claim:** a ~3.6M looped encoder (Inkling-style restart + closed gate), trained once with K
  sampled from {2,4,6,8,10} at 1:2:6:2:1, works at any k. It clearly beats a plain model with the
  same parameters (A). It holds up against the same encoder trained at a fixed K (B_fix, the
  Komatsu/LARM-style baseline). It holds up against a 3.2× larger plain model (C) exiting at
  the same FLOPs, timed as real CPU wall-clock.
- **Pilot:** A, B, C each get a 3-LR sweep at half steps. Each winner is rerun at full steps,
  and B_fix runs at B's LR. Dev only, 15 epochs. **25 h of Mac time in total** (§6).
- **Go** (§7): B(6) ≤ 0.80 × A_best, B(4) is usable, and B(4)/B(6) are within 10% of C's
  equal-FLOP exits at no more than 10% extra CPU latency. Prior P(go) ≈ 0.25 [E].
- **Output:** a blog post plus an HF release. A workshop paper is stage 2, only after a go (§8).

---

## 1. Hypotheses

The predictions, with 80% ranges, are in `experiments/pilot/README.md`. They are committed
before any run. B means B_best, the full-step rerun at its swept LR; the same goes for A and C.
B(k) is B run at k loops. C@d is C's CTC exit after d blocks.

| # | hypothesis | test (dev-clean unless stated) | role |
|---|---|---|---|
| H1 | Looping beats a plain model of the same size | B(6) vs A_best, dev-clean and dev-other | **primary**, go criterion 1 |
| H2 | One sampled-K model is usable at low k | B(4) vs A_best and vs B(6) | go criterion 2 |
| H3 | At equal FLOPs the loop is no worse than a plain early-exit model, in accuracy or CPU wall-clock | B(4) vs C@10, B(6) vs C@14 | go criterion 3 |
| H4 | Sampled K beats fixed K away from the trained depth, and costs little at it | B(k) vs B_fix(k), k = 2, 4, 6, 8 | reported; the novelty claim |
| H5 | B extrapolates past K_max | B(12), B(14), B(16) vs B(10) | reported |
| H6 | The closed gate opens | mean ‖gate message‖ / ‖h0‖ per core layer, logged through training | diagnostic; a bug if it stays ~0 |

## 2. Data

| split | use | size |
|---|---|---|
| train-clean-100 (100.6 h, 28.5k utterances) | training, BPE-256 | 6.3 GB |
| dev-clean / dev-other | sweep selection, every pilot metric | 337 / 314 MB |
| test-clean / test-other | **not downloaded**; only after a go and the owner's OK | 346 / 328 MB |

- Source: openslr.org/12, CC BY 4.0 [V]. Total download ≈ 7 GB, under the 20 GB ask line.
- Features: 80-dim Kaldi fbank (`torchaudio.compliance.kaldi.fbank`), 10 ms hop, computed once
  on CPU and stored as float16 (~5.8 GB) [E]. All arms read the same files.
- Tokens: sentencepiece BPE-256 trained on the train-clean-100 transcripts only. CTC blank = 256.
- Filtering (step 0): drop clips with fewer than 7 frames, which the 4× frontend reduces to 0,
  and any clip where `loop_asr.ctc_feasible` fails (T' < labels + repeats). Count the dropped
  clips. Assert max target length < 512 (MPS CTC bug #203356, not fixed in 2.15 [V]).
- The same text normaliser is used for every arm and for the references (Whisper-style English
  normaliser).

## 3. Arms

Shared by all arms: pre-norm Transformer, RoPE, d=256, 4 heads, FFN 1024, dropout 0.1, 4× conv2d
subsampling (64 ch, 25 frames/s), final LayerNorm + linear CTC head, BPE-256, greedy decoding.

| arm | encoder | params | block applications per frame | training loss | config |
|---|---|---|---|---|---|
| A | 4 plain blocks | 3,574,721 | 4 | CTC | `arm_a.yaml`, `arm_a_best.yaml` |
| B | prelude 1 + shared core 2 × k + coda 1; one gate per core layer, rank 32, closed at init; K ~ {2,4,6,8,10} at 1:2:6:2:1 per step | 3,608,513 (+0.95% vs A) | 2 + 2k (mean 14) | CTC after the coda at the sampled K | `arm_b.yaml`, `arm_b_best.yaml` |
| B_fix | B's encoder at a fixed K = 6 | 3,608,513 | 14 | CTC at K = 6 | `arm_b_fix.yaml` |
| C | 14 plain blocks, CTC exits at 6 / 10 / 14 through the shared head | 11,472,321 | 14 (exits stop at 6 or 10) | mean of the 3 exit CTC losses | `arm_c.yaml`, `arm_c_best.yaml` |

**B in detail.**
- Every pass restarts from the prelude output h0. Core layer j of pass p reads the output
  of core layer j of pass p−1 through its gate: `h ← h + up(silu(down(LN(m))))`, with `up`
  zero-initialised.
- At init every pass computes the same thing, so B(k) equals the 4-block path for any k
  (tested). All of B's gain must come through the gates opening, so H6 is logged.
- There is no step embedding (Inkling has none, and Geiping finds it hurts extrapolation).
- This is a simplified Inkling: an additive low-rank message per layer instead of Inkling's
  tanh forget/message gate over an attention read. The research finds gate types perform about
  the same [V-s]. The Inkling-form gate is a stage-2 ablation.
- B at k=1 never occurs in training, so it is out of distribution and reported only.

**C in detail.** Exits at 6/10/14 are exactly B's depths at k = 2/4/6. Those three are the
equal-FLOP comparison points. An exit stops the forward pass there (`depth=`), so an exit's
measured latency is what a deployed early-exit model would pay. Training C on its exits is the
one deliberate break of "arms differ only in the encoder". Each any-depth model gets its
native any-depth training: sampled K for B, joint exits for C. The main support is Wright
2309.09546: joint exit training from scratch improves the final and middle exits over
single-exit models (6.5 / 17.7 → 5.1 / 15.1), but is worse at the shallow CTC exits 2 and 4.
So C should be strong at the exits the go criterion uses (blocks 10 and 14) and may be weaker
at block 6. InterCTC (9.2 → 7.8 in Komatsu) is an intermediate-loss result and only weaker
support. A stronger C makes a B win more meaningful.

**Compute, counted on CPU** (`loop_asr.forward_flops`, one 15 s utterance, forward
matmul/conv FLOPs) [M]:

| | GFLOP | ratio to C@14 | | GFLOP | ratio to equal-depth C exit |
|---|---|---|---|---|---|
| A = B(1) | 3.77 | 0.340 | B(2) | 5.25 | 1.005 (C@6 5.23) |
| C@14 | 11.08 | 1 | B(4) | 8.23 | 1.009 (C@10 8.15) |
| B(8) | 14.18 | 1.280 | B(6) | 11.20 | 1.011 (C@14) |
| B(10) | 17.15 | 1.548 | B(16) | 26.08 | 2.354 |

- **Training-compute ratio** (forward FLOPs per step, expected over K): B : C = 1.011 against
  C without its exit heads, **1.002 with them**, as C actually trains (the two extra exit heads
  add ~0.9%). A : C = 0.340, B_fix : C = 1.011 (1.002). So B and C train at the same compute. A trains on a third of it, which is the A-long caveat in §4.
- **Inference-compute ratio:** B(k) : C@(2+2k) = 1.005–1.011 at k = 2, 4, 6, the gates being
  the only extra matmuls. B(6) : C (full) = 1.011. B(4) : C (full) = 0.743, so 1.35× fewer FLOPs.

## 4. Controls and the biases left

- **Matched:** steps, epochs (15), schedule (AdamW, 1k warmup, cosine), seed 0, batch order,
  batch size, light SpecAugment, features, tokenizer, normaliser, decoding.
- **Size:** B is within +3% of A's parameters (test `test_b_within_3_percent_of_a_params`).
- **Tuning:** A, B and C each get the same sweep, LR ∈ {3e-4, 7e-4, 1.5e-3} at half steps,
  picked on dev-clean. The winner is rerun at full steps. B_fix uses B's winning LR (same
  encoder, so no own sweep): that is a small bias in B's favour, written down here.
- **Compute:** C is matched to B's mean training K (§3). B at k > 6 spends more than C.
- **Speed:** B(k) is compared with C exiting at the same FLOPs, never only with full-depth C.
- **Biases left on purpose (stage 2):**
  - A-long: A trained at B's training FLOPs, ~3× the steps.
  - One seed: B–C gaps under 5% relative are reported as ties.
  - Dev is used both for the sweep and for the report, so the numbers are optimistic.
  - Light SpecAugment favours looping's regularisation edge at 100h.

## 5. Metrics and measurement protocol

- **WER:** greedy CTC, jiwer, one normaliser, on dev-clean and dev-other. B and B_fix are
  read at k ∈ {1, 2, 4, 6, 8, 10, 12, 14, 16}, C at exits {6, 10, 14}, A at 4.
- **CPU latency (used by the go criterion):**
  - Eager PyTorch 2.15, `torch.inference_mode()`, `torch.set_num_threads(4)` (the M5 P-cores),
    batch 1.
  - A fixed list of 200 dev-clean utterances, drawn with seed 0 and committed.
  - 10 warmup utterances, then the encoder + head time per utterance, features excluded and
    timed separately.
  - Report the median and p90, plus RTF = compute time / audio time. Also 1 thread (reported
    only).
  - The same machine state for every arm: on mains power, no other heavy jobs, the arms
    interleaved in one session.
- **MPS latency:** the same protocol with `torch.mps.synchronize()` before each timer stop.
  Reported, not in the go criterion.
- **ONNX Runtime CPU** (fp32, then int8 dynamic after a go): reported, exploratory.
- **FLOPs:** `loop_asr.forward_flops`, always on CPU (FlopCounterMode counts MPS attention as 0).
- **Size:** unique parameters, fp32/fp16 state_dict bytes. int8 only after a go.
- **Gate opening (H6):** mean ‖gate_j(m)‖ / ‖h0‖ per core layer, every 500 steps.
- **Exploratory, inference only:** adaptive k per utterance. Stop when the greedy output is
  unchanged between k and k+2. Report the mean k and WER.
- **References on the same Mac:** Whisper tiny.en and Moonshine Tiny, dev-clean/dev-other WER
  and the same CPU latency protocol. The comparison is for scale only, since they saw
  2–3 orders of magnitude more data.

## 6. Pilot

### Step 0 (before any training run, ~3 h)

1. Environment: `uv sync` (torch 2.15.0 is locked). Record the torch, macOS and chip versions.
   `get_device("mps")` must return MPS. It now raises if MPS is missing, and an automatic CPU
   fallback on a Mac warns.
2. Download train-clean-100, dev-clean and dev-other (~7 GB). Compute the fbank features, train
   BPE-256, and filter clips (§2). Record the dropped clips.
3. **MPS ctc_loss check on torch 2.15:** on 8 real batches, with dropout off and fp32, compare
   the MPS and CPU loss and gradients. They must agree to a relative grad difference < 1e-3.
   Assert max target length < 512. A pre-check on random data in this PR gave 2.4e-5 [M].
4. **FLOPs counted on CPU** for every arm, k and exit with `loop_asr.forward_flops`. They
   must match §3.
5. **Reference deps:** `brew install ffmpeg` (Whisper needs it), plus openai-whisper and
   moonshine-voice in a separate dependency group. Each must transcribe one dev-clean clip.
6. **Calibration:** 200 training steps per arm on MPS, measuring seconds per step, then the
   projected hours in the run table below.
   - Up to 30 h projected: run as planned.
   - Over 30 h: apply the cut order and write the cut into `experiments/pilot/README.md`
     before launching.
7. Commit `experiments/p0-calibration/README.md` (predictions: s/step per arm) before
   calibrating.

### Runs

The cost model is minutes/epoch ≈ 0.9 + 0.8 × block applications per frame [E]. It is
calibrated in step 0 from plan v1's 3–5 min/epoch for 4 blocks and 10–20 for 18.

| # | run | config | epochs | min/epoch [E] | hours |
|---|---|---|---|---|---|
| 0 | step 0: install, download, features, checks, calibration | – | – | – | 3.0 |
| 1 | A sweep: 3 LRs × half steps | `arm_a.yaml` | 3 × 7.5 | 4.1 | 1.5 |
| 2 | B sweep: 3 LRs × half steps | `arm_b.yaml` | 3 × 7.5 | 12.1 | 4.5 |
| 3 | C sweep: 3 LRs × half steps | `arm_c.yaml` | 3 × 7.5 | 12.3 | 4.6 |
| 4 | A_best, full steps | `arm_a_best.yaml` | 15 | 4.1 | 1.0 |
| 5 | B_best, full steps | `arm_b_best.yaml` | 15 | 12.1 | 3.0 |
| 6 | C_best, full steps | `arm_c_best.yaml` | 15 | 12.3 | 3.1 |
| 7 | B_fix, full steps at B's LR | `arm_b_fix.yaml` | 15 | 12.1 | 3.0 |
| 8 | eval: WER at every k and exit, CPU/MPS latency, references | – | – | – | 1.5 |
| | **total** | | | | **25.2 h ≈ 25 h** |

- **Order:** 0 → 1–3 → 4–7 → 8.
- **Owner's go:** every run over ~2 h needs one. Ask once per overnight batch, e.g. batch 1
  = runs 1–3 (10.6 h), batch 2 = runs 4–7 (10.1 h).
- **Cut order** if step 0 projects over 30 h:
  1. Sweep A and B only. C runs once at 7e-4, and the C-tuning bias is written down (decision 3).
     This saves ~4.6 h.
  2. Move B_fix to stage 2. This saves 3.0 h, and H4 goes unanswered in the pilot.
  3. Drop to 12 epochs for every arm.

## 7. Go criterion (fixed before the pilot)

Measured on dev-clean and dev-other, greedy CTC, after runs 4–6. CPU latency is the median of
§5 at 4 threads, batch 1.

> **GO** if all three hold:
> 1. **Size (primary):** WER(B(6)) ≤ 0.80 × WER(A_best) on dev-clean **and** ≤ 0.85 × WER(A_best) on dev-other.
> 2. **Any k:** on dev-clean, WER(B(4)) ≤ 0.85 × WER(A_best) **and** WER(B(4)) ≤ 1.15 × WER(B(6)).
> 3. **Equal compute:** for both k = 4 and k = 6, on dev-clean, WER(B(k)) ≤ 1.10 × WER(C@(2+2k)) (C@10, C@14), **and** B(k)'s measured CPU latency ≤ 1.10 × C@(2+2k)'s.

**Sanity check (not a no-go; failing it means a bug to fix and rerun):** A_best < 45 and
B(6) < 35 on dev-clean, and H6's gate ratio > 0.05 in at least one core layer at the end of
training.

**Predicted:** see `experiments/pilot/README.md` and frontier.md §4. All [E].

| criterion | predicted ratio | P(pass) |
|---|---|---|
| 1 | 19.5/30 = 0.65 (clean), 43/55 = 0.78 (other) | 0.8 |
| 2 | 21/30 = 0.70, 21/19.5 = 1.08 | 0.75 given 1 |
| 3, WER | 21/21 = 1.00 (k=4), 19.5/18 = 1.08 (k=6), bar 1.10 | 0.5 given 1–2 (k=6 is a coin flip; k=4 nearly sure once k=6 passes) |
| 3, latency | 1.04 (1.00–1.10) at k = 4 and 6, bar 1.10 | 0.85 (≈ 0.5 at a 1.05 bar) |
| **go** | 0.8 × 0.75 × 0.5 × 0.85 | **≈ 0.25** (0.2–0.3; ≈ 0.15 at a 1.05 latency bar) |

The factors are correlated (a strong B helps 1, 2 and 3 together), so the product is a rough
estimate. Latency is predicted above 1.0 because B runs one gate (a rank-32 projection pair
and a multiply) per core layer per pass: at batch 1 each small op pays a fixed per-op cost
larger than its FLOPs, on top of B's +0.5–1.1% FLOPs.

**Why these thresholds.** Plan v1's speed test needed B(k) to match full-depth C within 5%
while being 1.5× faster on CPU. Only k ≤ 4 could pass that, at 56% of C's depth, which gave
P ≈ 0.15–0.2. That test also compared looping with "no exits", not with exits. Now:
- Criterion 1 keeps v1's 0.80 bar (LARM and Komatsu make it likely) and adds dev-other,
  where looping's edge is smaller.
- Criterion 2 makes "usable at any k" concrete at the lowest k that carries real training
  mass (2/12).
- Criterion 3 asks the real question: at the same FLOPs and the same wall-clock, does a 3.6M
  looped model come within 10% of a 3.2× larger plain model with exits? A tie is the claim,
  and a 10% margin is about the one-seed noise at this WER plus C's size advantage. The
  latency bar is 10% too (1.05 in the first v2 draft): the gates' per-op overhead at batch 1
  is expected to cost a few percent, and a 5% bar would fail B on that overhead rather than
  on the idea.
- Overall, a go stays a real result while not being near-impossible: P ≈ 0.25 against
  0.15–0.2 for v1. It is still less likely than not.

**Outcomes:**
- **Go (1 ∧ 2 ∧ 3):** write the blog post and HF release, and propose stage 2 (§8) to the owner.
- **1 ∧ 2, not 3:** "smaller, not more compute-efficient". Write it up as a size-vs-WER post
  with the wall-clock numbers. No stage 2 paper.
- **1, not 2:** looping helps only near the trained depth. Check H4 against B_fix, then
  report. No stage 2.
- **Not 1:** no-go. Check H6 and the sanity checks for a bug first, then report a negative
  result at 3.5M.

H4 and H5 are reported in every outcome. They decide what the blog post can claim beyond
the go.

## 8. After the pilot

- **Stage 2, only after a go and the owner's OK (~4–6 days of Mac time):** the workshop-paper
  package.
  - 50 epochs, 3 seeds, test-clean/test-other once.
  - Per-k specialists at k = 2, 4 (B_fix at K=6 already exists) and A-long.
  - C without exit heads (final-layer CTC only), to measure what joint exit training does to
    C at full depth.
  - A LARM-style carry-state arm (B2: state carried, input injection, intermediate CTC).
  - The Inkling-form tanh gate as an ablation, plus a closed vs open gate init.
  - ESPnet-strength SpecAugment in all arms.
  - Extrapolation and adaptive k in full.
- **Stage 3, only if stage 2 shows a clean B ≥ C:** train-960 at 3.5M for B, C and B2
  (~60 GB download, owner's OK).
- **Release:** int8 ONNX under 5 MB, a k-slider demo, a Pareto plot against Whisper tiny.en and
  Moonshine Tiny on the same Mac, and a blog post. Distillation from Moonshine pseudo-labels
  is a separate "product" variant, kept apart from the architecture claim.

## 9. Code, stack and risks

- **Code now:** `src/loop_asr/` holds the encoders, the per-layer gates, C's exits
  (`depth=`, `forward_exits`), `forward_flops` (CPU), `ctc_feasible` and a strict
  `get_device`. A training loop of ~400 lines in plain PyTorch comes with step 0. Copy
  ESPnet's hyperparameters and normaliser, not its framework.
- **Stack:** torch 2.15.0, fp32 eager, bool masks (bf16 softmax NaN on M5 with fully masked
  rows), no torch.compile unless validated against eager, torchaudio 2.11 `kaldi.fbank`,
  sentencepiece, jiwer.
- **Risks:**
  - MPS throughput is unknown, so every hour figure is [E] until step 0. The cut order caps
    the damage.
  - MPS CTC: the ≥512-label fix is in no stable release. BPE-256 stays far below that, and it
    is asserted.
  - Wall-clock at 3.5M, batch 1 may be dominated by per-op overhead, not FLOPs. That
    hurts B and C equally per block, but the gates add small ops to B.
  - At 100h looping wins partly as a regulariser (LARM's 960h reversal), so a pilot go does
    not transfer to more data.
  - LARM and Komatsu were partly read through summaries, and LARM's code is unreleased or
    unclear.
  - One seed.
