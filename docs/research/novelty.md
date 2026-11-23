# Novelty search: looped / shared / adaptive-depth speech encoders (as of 2026-11-23)

## Bottom line
The core claim, that a small looped shared-block CTC encoder on LibriSpeech-100h can match or beat deeper unshared models and that loop count works as an inference-time knob, is **already published twice**: Komatsu 2022 (folded encoders, CTC, 100h, conformer, repeat count changeable at decoding) and LARM 2026 (looped Transformer, 100h/960h, test-time loop scaling, including a 48-block baseline that is apparently FLOP-matched). Rolland et al. 2026 covers Prelude-Recurrent-Coda ASR and has already tried variable loop counts during training. What is left is narrow: one model trained with **sampled k** and evaluated at any k, a **closed-init gate**, **measured CPU wall-clock**, and the **~3.5M** scale. Realistically this is a strong blog post or HF release. It becomes a workshop paper only with the controls listed at the end.

## Closest prior work (verified unless marked)

### 1. LARM, arXiv 2606.04678 (Kaloga et al., Idiap, Jun 2026); code at github.com/idiap/LARM (MIT, checkpoints "TBD", needs a CUDA GPU)
- N=4 shared Transformer blocks, d=384, K=12 loops (48 block applications). CTC loss only at sparse checkpoints (loops 4/8/12). Also uses clock embeddings, FiLM depth conditioning, and delayed soft-posterior feedback (α, β initialised to 0.5, so not a closed gate). 50 epochs, AdamW.
- **K is fixed per trained model, not sampled.** Intermediate exits are read at inference. There is no adaptive halting; the authors list it as future work. **No wall-clock or CPU numbers.** FLOPs are not reported explicitly.

| 100h, greedy / +4-gram | Params | test-clean | test-other |
|---|---|---|---|
| Plain 4-block | 7.6M | 26.78 / 13.44 | 52.82 / 36.60 |
| Plain 16-block | 28.9M | 14.43 / 9.97 | 37.23 / 28.68 |
| Plain 48-block | 85.7M | 12.24 / 9.03 | 34.06 / 27.07 |
| LARM d=384 | 7.7M | 11.34 / 8.66 | 31.84 / 26.28 |
| LARM d=1024 | 85.7M | 8.95 / 7.22 | 27.93 / 23.79 |

- My inference: the plain 48-block model is 48×d=384 blocks, so it has the **same block applications (about the same FLOPs)** as LARM d=384 with 11× the parameters. LARM already contains a matched-FLOP deep control, at 100h.
- Per-exit WER at 100h, test-clean, for the K=12 model: 15.20 at loop 4, 11.57 at loop 8, 11.34 at loop 12. Gains continue up to K=20 (11.02); K=24 degrades (13.3). Ablation: naive looping gives 12.70 against 11.34 for the full LARM.
- Weak spot: the absolute WERs are poor (a 7.7M Transformer-CTC at 31.8% on test-other) compared with conformer baselines at the same data size (see Komatsu below).

### 2. Komatsu, "Non-Autoregressive ASR with Self-Conditioned Folded Encoders", ICASSP 2022, arXiv 2202.08474
- Conformer-CTC on LibriSpeech-100h (and TED2). Nb base layers, then Nf folded layers repeated r times, with self-conditioned CTC. **The paper states that the repeat count can be changed at decoding.** Running more repeats at decoding than in training degrades WER; the best results come when training and decoding repeats match. Reports RTF.

| 100h, no LM | Params | test-clean | test-other | RTF |
|---|---|---|---|---|
| CTC, 18-layer | 30.5M | 9.2 | 24.0 | 0.058 |
| InterCTC, 18-layer | 30.5M | 7.8 | 21.7 | 0.058 |
| Self-cond. CTC, 18-layer | 30.6M | 7.3 | 20.5 | 0.058 |
| Nb=0, Nf=3, r=6 (fully shared) | **6.8M** | 9.5 | 25.6 | 0.058 |
| Nb=3, Nf=3, r=5 | 11.6M | 7.6 | 21.0 | 0.058 |
| Nb=6, Nf=3, r=6 | 16.3M | 7.2 | 20.1 | 0.075 |

- This is the **real sub-10M, 100h prior**. A 6.8M fully shared conformer reaches 9.5/25.6, far better than LARM's 7.7M Transformer. Any 3.5M result has to be compared against it.

### 3. Rolland, Carvalho, Abad, "Rethinking Depth: Recursive-Transformer for ASR", arXiv 2606.09357 (Jun 2026)
- AED+CTC on LS-960 and AISHELL. Prelude-Recurrent-Coda configs written {shared layers n_r, loops L, prelude, coda}. 24-layer baseline: 2.12/4.76 at 75.6M. {4,6,0,0}: 2.32/5.24 at 12.6M. {1,24}: 2.78/6.49 at 3.1M. L1 {4,5,2,2}: 2.16/4.92 at 25.2M.
- **Already tried variable loops in training.** Length-based loop count per utterance (1-20) gave 2.22/5.05. A curriculum 1→20 gave 2.13/4.97. Loops are fixed at inference and there is no early stop. FLOPs are matched across configs; no wall-clock.
- Earlier work by the same group: Rolland & Abad, Interspeech 2025, Shared-Conformer, 63% fewer parameters (isca-archive.org/interspeech_2025/rolland25_interspeech.html; I found no absolute numbers).

### 4. Other weight sharing in ASR
- Hernandez et al. (Google), ICASSP 2023, arXiv 2303.08343. Repeats whole conformer blocks, shares modules and sub-components, and shares low-rank factors. A **5M** model reaches 2.84/2.94 on dev-clean/test-clean (large-scale training data; test-other and repeat counts not found). Fixed depth.
- Tang et al., arXiv 2303.13072, "Beyond Universal Transformer". On AISHELL-1, one encoder block repeated 12 times plus a linear+ReLU adapter per repeat: 7.6-8.5M params, CER 6.63 against 5.30 for the 29M baseline. Pure reuse without adapters gives 9.30. Going to 18 repeats did not help. Inference depth was not varied.
- "Depth through Recurrence: Towards Ultra-Efficient On-Device ASR", NeurIPS 2026 poster 149074 (Feng et al.). Two shared recurrent blocks plus knowledge distillation recover 97.4% of large-model accuracy on the Open ASR Leaderboard; adds MoE-LoRA personalisation. Parameter counts and whether k varies at inference: **UNVERIFIED** (no arXiv version found).
- Also cited by Rolland and not checked: Sim-T (Wei 2023), ResidualTransformer (Wang & Li 2024), Audio ALBERT (Chi 2021). Universal-Transformer-style ASR (Zhou et al. 2018): **UNVERIFIED**, not located.

### 5. Early exit and dynamic depth (no sharing)
- InterCTC, Lee & Watanabe 2021, arXiv 2102.03216. Uses intermediate CTC plus stochastic depth; the follow-up arXiv 2106.09216 uses this to prune layers at inference. Self-conditioned CTC: Nozaki & Komatsu 2021 (numbers in the Komatsu table above).
- Wright et al., arXiv 2309.09546 (ICASSP 2024 workshop). 31M Conformer-CTC trained on LS-960 with exits every 2 layers; test-other at exits 2/4/6/12 is 43.8/25.7/18.1/15.1 with joint exit training. Joint training of all exits beats single-exit models. Exit choice by posterior confidence beats entropy.
- Splitformer, arXiv 2506.18035 (EUSIPCO 2025). 36.7M, LS-960, early exit with parallel downsampled layers. Layer-2 test-clean is 28.1 against 31.0 for the baseline (from a secondary summary, **UNVERIFIED**).
- HuBERT-EE, arXiv 2204.06328 (Interspeech 2024): confidence-based early exit on HuBERT-large.
- I3D, Peng, Lee & Watanabe, ICASSP 2023, arXiv 2303.07624. Gumbel gates on each MHA/FFN, depth chosen at inference by threshold β. On LS-100h AED, 27 average layers give 8.2 against 8.0 for 36 layers.
- Dynamic Encoder Transducer, arXiv 2104.02176 (layer dropout, depth chosen on demand). RWTH Dynamic Encoder Size, arXiv 2407.18930 (Interspeech 2024, subnet supernet).
- CTC blank-triggered layer skipping, arXiv 2401.02046. 12-layer d=256 conformer, LS-960: RTF 0.078→0.048 with WER 3.65/9.35→3.70/9.34. This is a real **wall-clock** precedent.

### 6. Tiny edge STT releases, 2025-26
- I found no sub-10M public release that uses recurrence. Moonshine flavours (arXiv 2509.02523) are 27M with no sharing. Hernandez 2023 is the only 5M weight-sharing system, and it is not released.

## What is actually new in the proposed project

| Element | Already done? |
|---|---|
| Looped shared encoder on 100h beats deeper plain | **Yes**: LARM, Komatsu |
| Loop count set at inference | **Yes**: LARM (exits), Komatsu (repeat count) |
| Matched-FLOP deep plain control | **Mostly yes**: LARM's 48-block model (my inference, 100h only, 85.7M) |
| Intermediate CTC across loops | **Yes**: LARM, Komatsu |
| Loop count **sampled per batch** in training, one model for all k | **Partly**: Rolland used length-based and curriculum schedules, but no one shows a single model evaluated across k against per-k specialists |
| Gate **initialised closed** (ReZero/SkipInit-style) on the loop residual | **Not found in ASR** |
| Measured **CPU wall-clock versus k** curve | **Not in looped-ASR work** (Komatsu gives RTF only; 2401.02046 for non-looped) |
| ~3.5M params, below every looped-ASR point so far | **New scale point** (Rolland's 3.1M is a degenerate {1,24} on 960h) |
| Consumer Apple Silicon training | Reproducibility angle, not a research contribution |

## Publishability (honest)
- **As scoped, this is a blog post or HF release.** It would be a nice reproduction of LARM/Komatsu at 3.5M with a wall-clock curve. A reviewer would say "LARM plus Komatsu already show this."
- **Workshop paper** (an ICASSP/Interspeech satellite, or an efficient-ML workshop at NeurIPS/ICLR) if it adds all of these:
  1. A conformer baseline on par with Komatsu's numbers, so the absolute WERs are credible.
  2. Head-to-head comparisons against fixed-K LARM-style training and Komatsu-style folding at the same parameter count.
  3. A Pareto plot of sampled-k "any-k" against per-k specialists, plus extrapolation to k beyond training (Komatsu and LARM both see degradation there; fixing it would be a result).
  4. Measured CPU latency against WER.
  5. Three or more seeds.
- **Interspeech full paper** needs one more substantive piece:
  - Adaptive per-utterance halting (LARM's stated future work), trained from the closed gate or the intermediate-CTC confidence, with real latency savings.
  - Or streaming or chunked looping.
  - Or a convincing ablation showing the closed-init gate is what makes sampled-k training stable or lets it extrapolate.
  - Scaling to 960h at 3.5M would also help.
- Cite at least: 2606.04678, 2202.08474, 2606.09357, 2303.08343, 2303.13072, 2102.03216, 2309.09546, 2303.07624, 2401.02046, NeurIPS 2026 poster 149074.
