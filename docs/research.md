# Research summary: where loop-asr stands against the frontier

Written 2026-11-23 from the frontier research in `docs/research/`. It replaces the earlier
research.md, a study of Inkling 2 Mini in Italian, which now lives unchanged at
[`research/inkling_2_mini_study.md`](research/inkling_2_mini_study.md).

Tags: **[V]** read in the primary source; **[V-s]** primary source read through a summary;
**[U]** unverified; **[E]** estimate. Each line cites the detail file that holds the URLs:
[frontier.md](research/frontier.md) (main report), [frontier_tables.md](research/frontier_tables.md),
[looping_lit.md](research/looping_lit.md), [novelty.md](research/novelty.md),
[practical.md](research/practical.md).

## 1. The core claim is already published

- **Komatsu 2022** (arXiv 2202.08474, ICASSP) [V-s]: a fully shared (folded) conformer-CTC,
  **6.8M parameters, trained on train-clean-100 only, 9.5 / 25.6** test-clean/test-other
  greedy. The repeat count can be changed at decoding. Decoding with a different count from the
  one used in training makes WER worse. A plain 18-layer CTC with 30.5M parameters gets 9.2 / 24.0.
- **LARM 2026** (arXiv 2606.04678) [V]: 4 shared Transformer blocks × K=12, d=384, 7.7M params,
  100h, 11.34 / 31.84 greedy. It beats a plain 48-block model with the same block applications
  (85.7M, 12.24 / 34.06). It loses to that model on 960h (4.59 vs 3.87), so at 100h part of
  looping's win is regularisation. K is **fixed per model**. Exits at loops 4/8/12 give
  15.20 / 11.57 / 11.34. The exit at loop 4 of a K=24 model gets 20.05, against 14.07 for a
  model trained at K=4. Its plain 4-block baseline (26.78) is weak.
- **Rolland et al. 2026** (arXiv 2606.09357) [V-s]: Prelude/Recurrent/Coda ASR on 960h with
  AED+CTC, down to 3.1M parameters (2.78 / 6.49). The loop count varies in training by
  utterance length or by curriculum, but it is fixed at inference.

**Correction to the old plan.** plan.md v1 said: "I found no published model under 10M
parameters trained only on 100h, apart from LARM". That is **wrong**. Komatsu's 6.8M shared
conformer (9.5 / 25.6) was trained on 100h alone and is far stronger than LARM's 7.7M. It is the
real bar under 10M on 100h. Better-tuned ~26–30M CTC encoders on 100h reach about 9–10 / 23–24
(ESPnet E-Branchformer-CTC 9.6 / 23.1; dev 9.2 / 22.4) [V].

## 2. What is still new (novelty.md)

Not found anywhere:
1. **One model trained with sampled K and evaluated at every k**, compared with fixed-K models
   and per-k specialists. LARM T4 shows that exits of a fixed-K model are poor, and sampling K is
   the obvious fix. It is untested in ASR.
2. **A closed-init gated message** (Inkling's restart-from-prelude rule plus a gate that starts
   closed) in a speech encoder.
3. **Measured CPU wall-clock against k** for a looped ASR encoder, compared with a plain model
   with early exits. Looped-ASR papers give FLOPs or RTF only. The one wall-clock precedent for
   dynamic depth, blank-triggered layer skipping (arXiv 2401.02046), uses no sharing.
4. CTC-only at ~3.5M on 100h, a new scale point. Rolland's 3.1M model is AED on 960h.
5. Per-utterance adaptive k chosen from intermediate CTC confidence. LARM lists this as future
   work.

**Publishability:** as scoped, this is a blog post plus an HF release. A workshop paper needs
a fixed-K/per-k-specialist comparison, C with early exits, a measured CPU Pareto, 3 seeds and
extrapolation past the trained K. A full Interspeech paper needs one of these on top: adaptive
halting with real latency savings, a gate ablation, or 960h.

## 3. Inkling 2 Mini, as verified in code (looping_lit.md)

Read in transformers v5.24.0, `modeling_inkling2_loop.py` [V]:
- 24 layers: prelude 0–5, core 6–17 run `num_passes` times, coda 18–23. **Each pass
  restarts from the prelude output.** Only a message crosses passes. LARM and Geiping do the
  opposite: they carry the state forward.
- The message is per-token attention over the previous pass's core block states. Its queries
  are zero-initialised and indexed by layer, not by pass. There is **no step embedding**.
- The gate is `(1 − tanh(W_f g))·h + tanh(W_m g)·m`, `g = down(cat[rms(h), rms(m)])`, rank 64, with
  **4 gates per core layer** (q, k, v, MoE input), tanh and no bias. It is exactly closed at zero
  weights. TML's training init is not published [U].
- Production runs a fixed 3 passes. Only `TML-d24-varloop-research` was trained with 1–5 passes
  sampled 1:2:6:2:1. One pass collapses (SWE-V 56.3 vs 78.7 at 3), and 4–5 passes add little.
- **Consequence for arm B [E]:** under the restart rule with a closed gate, every pass computes
  the same function. At init B(k) is a 4-block model for any k, while paying for 2+2k blocks.
  All of B's gain has to come through the message opening, so the opening must be logged.

## 4. Other literature that shapes the design

- ALBERT [V]: sharing the FFN is what costs accuracy. Sharing attention alone costs nothing.
- "How to Loop Your Transformer" (Meta, 2610.40325) [V-s]: middle-stack looping is best on a
  low FLOP budget, gate types perform about the same, and loop-specific low-rank adapters help.
- Geiping 2025 [V-s]: step conditioning hurts extrapolation to more loops. A run can "learn
  early to ignore the incoming state".
- Early exits without sharing: InterCTC (2102.03216). Wright et al. (2309.09546): a 31M
  conformer on 960h, test-other 43.8 / 25.7 / 18.1 / 15.1 at exits 2/4/6/12, and joint exit
  training beats single-exit models. **A plain model with exits is the fair speed control.**

## 5. Where a 3.5M Transformer-CTC on 100h should land

The frontier.md §4 predictions (dev-clean greedy, 15 epochs, d=256, BPE-256, light
SpecAugment, all [E]): A ≈ 32, A_best ≈ 30, B(4) ≈ 21, B(8) ≈ 19, C ≈ 18. The test-other bar
from Komatsu (25.6 at 6.8M) is out of reach at 3.5M with a plain Transformer. External
products (Whisper tiny.en 39M, 5.66 / 15.45; Moonshine Tiny 27M, 4.52 / 11.71) train on 2–3
orders of magnitude more data. They tell us how far we are from a product, not whether looping
works. The per-arm numbers and 80% ranges are in `experiments/pilot/README.md`.

## 6. Flaws the research found in plan v1, and where each is fixed

| flaw (frontier.md §6) | fix | where |
|---|---|---|
| C has no early exits, so the speed test compares small k with a full-depth model | C trained with CTC exits at 6/10/14, read at equal FLOPs | DECISIONS 5 |
| K weights 1:2:6:2:1 have mean 6, but C was matched to K=8 | C at B's mean K: 14 blocks | DECISIONS 4 |
| Inkling-faithful gates (+0.5M, +15%) break the size match | the concern does not apply to our code (one gate at +16.9k). Now one gate per core layer, rank 32: +33.8k, +0.95%, tested ≤ +3% | DECISIONS 2 |
| only A gets an LR sweep | the same 3-LR half-step sweep for A, B and C | DECISIONS 3 |
| go criterion near-impossible (P ≈ 0.15–0.2) | re-derived, P ≈ 0.3 | DECISIONS 6 |
| FLOPs on MPS count attention as 0 | `loop_asr.forward_flops` counts on a CPU copy | DECISIONS 8 |

Left open on purpose (plan.md §9): stronger SpecAugment, an Inkling-form tanh gate, a LARM-style
carry-state arm, and A-long (A trained at B's training FLOPs).

## 7. Practical facts for the Mac (practical.md)

- torch **2.15.0** (2026-10-28) is current [V]. MPS CTC forward and backward need ≥2.14. The
  inf-gradient fix for targets of ≥512 labels (#203360) is in **no stable release** [V].
  BPE-256 targets on LS-100 stay around 200 tokens or fewer [E], but step 0 asserts < 512.
  Measured in this PR: MPS-vs-CPU CTC gradient relative difference 2.4e-5 on random data, fp32
  (`.kit/evidence/plan-v2/`).
- Train in fp32 with bool masks: bf16 softmax NaNs on M5 come from fully masked rows. There is
  no float64 on MPS, and determinism is not guaranteed. Treat torch.compile on MPS as
  unvalidated.
- `FlopCounterMode` has no formula for the MPS SDPA op, so attention counts as 0 there.
  **Count FLOPs on CPU.**
- torchaudio 2.11.1 keeps `compliance.kaldi.fbank`. Precompute features once on CPU.
- int8: ONNX Runtime dynamic quantisation is the most reliable CPU path on a Mac. Eager
  `quantize_dynamic` is being deprecated. Core ML W8A8 is for the Neural Engine.
- RTF protocol: batch 1, fixed thread count (4 P-cores, plus 1 thread), warmup, median and p90,
  `torch.mps.synchronize()` on MPS. At 3.5M and batch 1, per-op overhead can dominate the FLOPs.
- ESPnet LS-100 CTC recipe [V]: SpecAugment with 2 frequency masks (F ≤ 27) and 5 time masks
  (≤ 5%), 70 epochs, BPE-5000. SpecAugment at 100h is worth ~13% relative on clean and ~29% on
  other (2210.08520) [V].
