# loop-asr: frontier, novelty, predictions and plan review

Written 2026-11-23. Research only: no code and no runs. Tags: **[V]** read in the primary source this session; **[V-s]** primary source read through a summarising fetcher; **[U]** unverified (memory, abstract or snippet); **[E]** my own estimate or arithmetic. Raw notes with every URL are next to this file in `frontier_tables.md`, `looping_lit.md`, `novelty.md` and `practical.md`.

## Summary

- **The idea is not new.** Komatsu 2022 already ran a fully shared 6.8M conformer-CTC on 100h, choosing the repeat count at decoding (9.5/25.6). LARM 2026 has a looped 7.7M model that beats a 48-block plain model at equal FLOPs (11.34 vs 12.24). What is left for loop-asr: one model trained with sampled K and run at any k, a gate that starts closed, measured CPU wall-clock, and the 3.5M size.
- **The real comparison bar is not LARM's 7.6M model** (26.78/52.82 is a weak baseline). It is Komatsu's 6.8M shared conformer at 9.5/25.6. At 3.5M, a Transformer-CTC on 100h should land at about 17–24% WER greedy on dev-clean.
- **Predictions (dev-clean, greedy, 15 epochs):** A ≈ 32, A_best ≈ 30, B(8) ≈ 19, C ≈ 18, B(4) ≈ 21.
- **What that means for go:** H1 (B ≤ 0.8×A_best) is very likely. H2 (B ≤ 1.05×C) is a coin toss. The speed criterion is unlikely (~20%). Overall P(go) ≈ 15–20%.
- **Biggest design flaw:** C as specified has no early exits. The fair speed control is C with intermediate CTC (exits at depth 6/10/14/18). Without it, criterion 4 measures "small k against a full-depth model", not looping.
- **Fixes before compute is spent:**
  - The 1:2:6:2:1 weights put their mode on K=6, but C is matched to K=8.
  - Inkling-style gates (4 per layer, rank 64) add ~0.5M parameters (+15%), which breaks the parameter match with A.
  - Under Inkling's restart rule, B at init is a 4-block model whatever k is.

---

## 1. The frontier map of tiny ASR

### 1a. General frontier (any data). tc/to = LibriSpeech test-clean/test-other WER %

| model | params | train hours | tc/to no LM | tc/to with LM | latency / device | source |
|---|---|---|---|---|---|---|
| Moonshine Tiny v1 | 27.1M | ~200k | 4.52 / 11.71 | – | 27 ms response, 5.9% load, Apple M3 | arXiv 2410.15608 T1,T3; 2602.12241 T2 [V-s] |
| Moonshine v2 Tiny (streaming) | 33.6M (enc 7.4M) | ~300k | 4.49 / 12.09 | – | 50 ms, 8.0%, M3 | arXiv 2602.12241 T1–T3 [V-s] |
| Moonshine Base v1 | 61.5M | ~200k | 3.23 / 8.18 | – | 44 ms, M3 | 2410.15608; 2602.12241 [V-s] |
| Whisper tiny.en | 39M (37.8M per Moonshine) | 680k | 5.6 / 14.6 (paper); 5.66 / 15.45 (Moonshine re-eval) | – | 289 ms, 8.5%, M3 | 2212.04356 App. D.1.1; 2410.15608 T3; 2602.12241 T2 [V-s] |
| Whisper tiny | 39M | 680k | 7.6 / 16.9 | – | same | 2212.04356 [V-s] |
| distil-small.en | 166M | ~22k | not on LS test | – | "5.6× faster than large-v2" | HF card [V-s] |
| QuartzNet 5x5 CTC | 6.7M | 960 | dev only: 5.39 / 15.69 | – | – | 1910.10261 T2 [V-s] |
| Citrinet-256 CTC | 9.8M | 960 | 3.78 / 9.6 | 3.65 / 8.06 (6-gram) | – | 2104.01721 T3 [V-s] |
| NeMo conformer_ctc_small | ~13M | several k | 3.7 / 8.1 | 2.6 / 6.0 | – | HF nvidia/stt_en_conformer_ctc_small [V-s] |
| Conformer-CTC-S (Squeezeformer repro) | 8.7M | 960, 500 ep | 4.06 / 10.58 | – | 26.2 GFLOPs | 2206.00888 T3 [V-s] |
| Squeezeformer-XS CTC | 9.0M | 960, 500 ep, SPM-128 | 3.74 / 9.09 | – | 15.8 GFLOPs | 2206.00888 T3 [V-s] |
| Efficient Conformer CTC S | 13.2M | 960, BPE-256 | 3.57 / 8.99 | 2.72 / 6.66 | – | 2109.01163 T2 [V-s] |
| Conformer(S) RNN-T | 10.3M | 960 | 2.7 / 6.3 | 2.1 / 5.0 | – | 2005.08100 T2 [V-s] |
| ContextNet(S) RNN-T | 10.8M | 960 | 2.9 / 7.0 | 2.3 / 5.5 | – | 2005.03191 T2 [V-s] |
| Google weight-shared conformer | 5M | 960 | dev/test-clean 2.84/2.94 (no test-other found) | – | – | arXiv 2303.08343 [V-s] |
| icefall Zipformer-S + CR-CTC | 22.1M | 960 | 2.57 / 5.95 | – | – | icefall RESULTS.md [V-s] |
| icefall streaming Zipformer small | 6.1M | LS + GigaSpeech | 5.95 / 15.03 (320 ms chunk) | – | – | icefall RESULTS.md [V-s] |
| Parakeet TDT-CTC 110M (smallest Parakeet) | 114M | 36k | 2.4 / 5.2 | – | RTFx ~5300, A100 | HF card [V-s] |
| Canary-180M-flash | 182M | 85k | 1.87 / 3.83 (card metadata says 2.87 for to) | – | RTFx 1233 A100 | HF card [V-s] |
| LARM 7.7M (looped) | 7.7M | 960 | 4.59 / 11.75 | 3.51 / 9.38 | – | 2606.04678 T1 [V] |
| Recursive Transformer {1,24,0,0} (AED+CTC) | 3.1M | 960 | 2.78 / 6.49 | – | – | arXiv 2606.09357 [V-s] |
| Kyutai stt-1b-en_fr | ~1B | 2.5M h | not on the card | – | 0.5 s delay | HF card [U] |

**UNVERIFIED:**
- Any 2026 English release under 10M parameters with LibriSpeech numbers. I found none in a primary source.
- Picovoice and Liquid STT.
- A 65.7M ESP32-P4 transducer (arXiv 2610.38739), seen only in a snippet.

**Reading the table:**
- At under 10M parameters, 960h and a 500-epoch TPU budget, CTC reaches ~3.7/9.
- Weight sharing has already gone to 3.1M (Rolland, AED+CTC, 960h, 2.78/6.49). So "looped at 3.5M" is not a new size point on 960h, only on 100h with CTC-only.

### 1b. The real bar: train-clean-100 only, no extra data, no SSL

| model | params | loss / decoding | tokens | ep | tc/to no LM | with LM | source |
|---|---|---|---|---|---|---|---|
| plain Transformer-CTC 4 blk, d=384 | 7.6M | CTC greedy | char-30 | 50 | 26.78 / 52.82 | 13.44 / 36.60 | LARM T1 [V] |
| plain 16 blk | 28.9M | CTC | char | 50 | 14.43 / 37.23 | 9.97 / 28.68 | LARM T1 [V] |
| plain 48 blk (FLOP-matched to LARM) | 85.7M | CTC | char | 50 | 12.24 / 34.06 | 9.03 / 27.07 | LARM T2 [V] |
| naive loop 4×12 (no FiLM, no feedback) | ~7.6M | CTC | char | 50 | 12.70 / – | – | LARM T3a [V] |
| LARM 4×12, d=384 | 7.7M | CTC | char | 50 | 11.34 / 31.84 | 8.66 / 26.28 | LARM T1 [V] |
| LARM N6K8 / N8K6, d=384 | n/r | CTC | char | 50 | 10.86 / 10.85 (tc) | 8.25 / 8.34 | LARM T5 [V] |
| **Komatsu folded conformer, fully shared (Nf=3, r=6)** | **6.8M** | CTC greedy, self-cond. | n/r | n/r | **9.5 / 25.6** | – | arXiv 2202.08474 [V-s] |
| Komatsu Nb=3, Nf=3, r=5 | 11.6M | CTC | – | – | 7.6 / 21.0 | – | 2202.08474 [V-s] |
| Komatsu plain 18-layer CTC | 30.5M | CTC | – | – | 9.2 / 24.0 | – | 2202.08474 [V-s] |
| Conformer-CTC 18L d=256 (+SelfCond) | ~30M | CTC greedy | SPM-300 | 50 | 9.08 / 23.68 (7.48 / 21.31) | – | arXiv 2204.00174 [V-s] |
| ESPnet E-Branchformer CTC e12 | 26.4M | CTC greedy | BPE-5000 | 70 | 9.6 / 23.1 (dev 9.2 / 22.4) | – | ESPnet egs2/librispeech_100 README [V] |
| ESPnet E-Branchformer CTC+att | 38.5M | hybrid | BPE-5000 | – | 6.3 / 17.0 | – | same [V] |
| icefall pruned_transducer_stateless2 | ~80M [U] | RNN-T | BPE-500 | 40 | 6.64 / 17.19 | – | icefall RESULTS.md [V-s] |
| Transformer, no aug vs SpecAugment | n/r | – | – | – | 12.1 / 32.7 → 10.5 / 23.2 | – | arXiv 2210.08520 [V-s] |

**Corrections to plan.md §3:**
- "I found no models under 10M trained only on 100h apart from LARM" is wrong: Komatsu's 6.8M shared conformer exists and is far stronger than LARM's.
- LARM's plain 4-block baseline (26.78) is weak, probably too shallow and lightly augmented. A 4-block d=384 Transformer-CTC is not a tuned baseline. A ×2.3 gap against plain conformers of similar size is suspicious, and some of LARM's "looping gain" may be "deeper network on a weak recipe".
- Best LARM-like number at <10M on 100h: ~11/32 greedy. Best known at <10M on 100h: 9.5/25.6 (conformer, shared). Our 3.5M Transformer should sit above both.

## 2. Weight sharing and looping: what the literature actually says

### 2a. What Inkling 2 Mini does, verified in code

Source: transformers v5.24.0, `models/inkling2_loop/modeling_inkling2_loop.py`, plus configs on HF [V].

- **Layout:**
  - 24 layers in 4 blocks of 6: prelude 0–5, core 6–17 run `num_passes` times, coda 18–23.
  - Config `loop_config={core_start:6, core_end:18, num_passes:3, gate_rank:64}`.
- **Each pass restarts from the prelude output** (code comment ~L1397: "Remaining passes restart from the pre loop states and receive messages from the previous pass").
  - The state is not carried; only a message crosses passes.
  - This is the opposite of LARM and Geiping (carry the state and re-inject the input).
- **The message:**
  - A per-token softmax attention over the previous pass's core block states, with learned queries initialised to `torch.zeros` (a uniform average at init).
  - The queries are indexed by layer and by {q,k,v,mlp}, **not by pass**.
  - **There is no loop/step embedding anywhere.**
- **The gate:**
  - `(1 − tanh(W_f g))·h + tanh(W_m g)·m`, with `g = down(cat[rms(h), rms(m)])`; 2d→64→d, no bias, fp32.
  - Four gates per core layer: on the q, k and v inputs and on the MoE input.
  - Weights at zero give exactly h, so the gate is closed. The blog says it "initially blocks the entire message" [V-s].
  - The HF init (normal, std 0.02) is a loading default. **TML's training init is not published [U].**
- **The pass count is fixed at 3 in production.** "Only supports a fixed number of looping passes" [V-s].
  - Only `TML-d24-varloop-research` was trained with 1–5 passes at 1:2:6:2:1, and then post-trained at 3 passes.
  - Its card warns it "may behave unpredictably at pass counts other than 3" [V-s].
  - SWE-bench Verified for 1–5 passes: 56.3 / 76.5 / 78.7 / 79.3 / 79.6. One pass collapses; 4–5 add little [V-s].
  - The no-CoT time horizon still holds at 10 passes, 2× the trained maximum [V-s].
- **Loss:** final loss only in the released code. Intermediate-pass supervision in training is [U].
- **Middle vs last looping:** looping the middle layers gave 1.51–1.91× compute-equivalent gain, against 1.24–1.36× for the last layers [V-s].

**Consequence for arm B [E]:**
- With restart-from-prelude and a closed gate, every pass computes the same function on the same input.
- So **at init B(k) is functionally a 4-block model** (prelude, core×2, coda) for any k, while paying 2+2k blocks of compute.
- All of B's advantage must come through the message path opening. This is cleaner than LARM (B starts as A), but it is also the failure mode Geiping §4.3 describes: a run that "learned early to ignore the incoming state".
- The gate opening (mean |tanh(W_m g)|) must be logged and has to be a pilot diagnostic.

### 2b. LARM (arXiv 2606.04678v2), everything that matters [V]

- **Encoder:**
  - Pre-norm Transformer with RoPE, head dim 64, FFN ×4.
  - Conv2d 4× frontend (64 channels).
  - d=384, N=4 shared blocks, K=12 (48 block applications), 7.7M parameters.
  - Character CTC (30 symbols).
- **Recurrence:**
  - The state is carried, with input injection: a(k) = z(k) + β·h⁰ + α·r̄(k).
  - r̄ is the CTC posterior fed back with a one-frame delay. α and β start at **0.5, not closed**.
  - Clock embedding plus FiLM depth conditioning on d̄ = (k−1)/(K−1).
  - CTC averaged over loops {4, 8, 12}.
  - **K is fixed per model, not sampled.**
- **Training:**
  - AdamW, weight decay 5e-3, peak LR 7e-4, cosine, 1k warmup, batch 32, 50 epochs.
  - Light SpecAugment: 1 frequency mask with F=15, 2 time masks at p=0.02.
  - **Not reported:** hardware, time, FLOPs, latency, dev WER.
- **Ablations (T3, test-clean):**
  - naive loop 12.70 vs full LARM 11.34, so **~90% of the gain over the plain 4-block (26.78) comes from plain looping**;
  - FiLM vs no depth conditioning 11.34 vs 12.13;
  - no feedback 12.31.
- **Loop exits (T4):**
  - K=12 model: 15.20 / 11.57 / 11.34 at exits 4 / 8 / 12.
  - A K=4-trained model at exit 4 gets 14.07, while exit 4 of the K=24 model gets 20.05. **Exits of a deep-trained model are worse than a specialist at that depth.**
  - Gains saturate at K = 16–20; K=24 degraded (13.3).
- **Unique blocks (T5):** N8K6 10.85 vs N4K12 11.34 at equal compute, so more unique blocks helps a little.
- **960h reversal:** plain 48-block 3.87 vs LARM 4.59. At 100h looping wins partly as a regulariser [V].
- **Code:** the paper says "will be released soon" [V]. One search found github.com/idiap/LARM (MIT, checkpoints TBD) [U, conflicting].
- **Inconsistency:** the d=1024 model is listed at 85.7M, but 4 blocks at d=1024 should be ~50M [E].

### 2c. Rest of the literature, one line each

- **ALBERT** (T4) [V]: sharing everything costs −1.5 to −2.5 average points. Sharing only attention costs nothing (81.7 vs 81.6). **The loss comes from sharing the FFN.**
- **Universal Transformer** (1807.03819) [U]: one shared block plus a per-step coordinate embedding, optional ACT. 28.9 vs 28.4 BLEU on En-De.
- **ACT / PonderNet** [U]: per-token halting needs a ponder-cost hyperparameter (τ, or a KL to a geometric prior). PonderNet extrapolates where ACT fails.
- **Recurrent depth, Geiping 2025** (2502.05171) [V-s]:
  - Prelude/recurrent/coda.
  - The input is injected at every step by concatenation through a 2h→h adapter; the initial state is random.
  - Recurrence sampled from a log-normal Poisson around r̄=32; truncated backprop through 8 steps.
  - Step-conditioning "interacts badly with path independence, leading to models that cannot extrapolate".
  - GSM8K is 0.00 at r=1 and 34.8 at r=32.
- **Mixture-of-Recursions** (2507.10524) [V]: middle-cycle sharing is best. At equal FLOPs it beats vanilla (43.1 vs 42.3 few-shot) with ~50% fewer parameters. Its load balancing struggles for about half of training.
- **Relaxed Recursive** (2410.20672) [U]: one LoRA per loop recovers most of the gap to the unshared model.
- **"How to Loop Your Transformer"** (Meta, 2610.40325) [V-s]:
  - Middle-stack looping is best at a low FLOP budget.
  - Loop-specific low-rank adapters (+1–4% parameters) help at L of 4–8.
  - **Gate types perform about the same.** A separate RMSNorm of the two streams matters.
  - It does not test a loop count different from training.
- **Theory** [U]: Saunshi 2025 says k layers looped L times nearly match kL layers on reasoning, but not on memorisation. Yang 2023 uses input injection plus random loop counts.
- **Loop-count robustness:**
  - Kuo, NeurIPS 2026 (2606.29983) [U]: randomising K reduces out-of-distribution variance.
  - Yang (2610.14456) [U]: there is a finite window for extrapolating to more loops.
- **Speech-specific sharing:**
  - Komatsu folded encoders (above) [V-s].
  - Rolland et al. 2606.09357 [V-s]: a Prelude/Recurrent/Coda ASR encoder, loop count varied in training by utterance length or by curriculum (2.13/4.97 at 960h), but fixed at inference.
  - Tang 2303.13072 [V-s]: one block ×12 plus per-repeat adapters, 7.6M, AISHELL CER 6.63 vs 5.30 for a 29M model; 9.30 without adapters.
  - "Depth through Recurrence: Towards Ultra-Efficient On-Device ASR", NeurIPS 2026 poster 149074 [U]: no arXiv version found. Not the same paper as the TTS arXiv 2609.29768 that research.md cites.
- **Early exit / dynamic depth without sharing:**
  - InterCTC 2102.03216 [V-s].
  - Wright et al. 2309.09546 [V-s]: 31M conformer, test-other 43.8 / 25.7 / 18.1 / 15.1 at exits 2 / 4 / 6 / 12. Joint exit training beats single-exit models.
  - I3D 2303.07624 [V-s]: 100h, 27 average layers at 8.2 vs 36 layers at 8.0.
  - Blank-triggered layer skipping 2401.02046 [V-s]: RTF 0.078→0.048 at equal WER, a **real wall-clock precedent**.

## 3. Novelty: what would actually be new

**Already shown:**
- A looped/shared encoder on 100h beats a deeper plain one: LARM, Komatsu.
- Depth chosen at inference: LARM exits, Komatsu repeat count.
- A FLOP-matched deep plain control: LARM's 48-block model (my inference: 48 × d=384 = the same block applications).
- Intermediate CTC across loops: LARM, Komatsu.
- Sharing at ~3M: Rolland, 960h, AED.

**Not found anywhere (the residual contribution):**
1. **One model trained with sampled K and evaluated at every k, compared with per-k specialists.** LARM T4 shows that exits of a fixed-K model are poor (20.05 vs 14.07); sampled-K training is the obvious fix and is untested in ASR.
2. **A closed-init gated message (Inkling restart+message) in a speech encoder.**
3. **Measured CPU wall-clock vs k for a looped ASR encoder**, against a plain model with early exits.
4. CTC-only at ~3.5M on 100h.
5. **Per-utterance adaptive k** chosen from intermediate CTC confidence, which LARM lists as future work.

**Publishability:**
- **As scoped (A/B/C, one seed): a blog post plus HF release.** A reviewer would say "LARM + Komatsu".
- **Workshop paper** (an Interspeech/ICASSP satellite, or an efficient-ML workshop) needs:
  - (i) B against a fixed-K LARM-style arm and against per-k specialists, at equal parameters;
  - (ii) C with early exits;
  - (iii) a measured CPU WER/latency Pareto;
  - (iv) 3 seeds;
  - (v) extrapolation to k > 10.
- **Full Interspeech paper:** add one of:
  - adaptive per-utterance halting with real latency savings;
  - a clean ablation showing the closed gate is what makes sampled K stable or able to extrapolate;
  - 960h at 3.5M.
- **What makes a blog/HF release worth it:** a <5 MB int8 English CTC model with a k-slider and a curve that is honestly worse than Moonshine Tiny but 8× smaller. Today no open model under 10M has published LibriSpeech numbers [U].

## 4. Expected outcomes, as concrete numbers

**Assumed setup:**
- d=256, 4 heads, FFN 1024, 4× conv frontend, BPE-256, greedy CTC, no LM.
- 15 pilot epochs, "light" SpecAugment as in plan.md.
- All numbers are dev-clean WER % with an 80% interval. **[E] throughout.**

**Anchors and adjustments:**
- Anchors: LARM 4-block 26.78 (7.6M, 50 epochs); naive loop 12.70 (48 applications); LARM 16-block 14.43; Komatsu shared 6.8M 9.5.
- Adjustments:
  - fewer epochs, 15 vs 50: +10–25% relative;
  - narrower, d=256 vs 384: +5–15%;
  - BPE-256 vs characters: about ±5%, with CTC on 100h slightly favouring characters [U];
  - B executes 18 blocks vs 48 for LARM and has only 2 shared core blocks (LARM T5: fewer unique blocks is worse).
- dev-clean runs ~0.3–0.5 lower than test-clean (ESPnet 9.2 vs 9.6). dev-other ≈ 2.2–2.8× dev-clean at this quality (LARM ratios 1.97–2.8) [V].

| arm | blocks executed | params [E] | dev-clean pred. | 80% range | dev-other pred. |
|---|---|---|---|---|---|
| A (LR 7e-4) | 4 | 3.56M | 32 | 26–40 | 58 |
| A_best (best of the 3-LR sweep, full steps) | 4 | 3.56M | 30 | 25–37 | 55 |
| B k=2 | 6 | 3.6–4.1M* | 27 | 21–35 | 52 |
| B k=4 | 10 | " | 21 | 17–27 | 45 |
| B k=6 (training mode) | 14 | " | 19.5 | 15.5–25 | 43 |
| B k=8 (FLOP-matched to C) | 18 | " | 19 | 15–24 | 42 |
| B k=10 | 22 | " | 19 | 15–25 | 42 |
| B k=12–16 (extrapolation) | 26–34 | " | 19–23 | 15–30 | – |
| C, 18 plain | 18 | 14.6M | 18 | 14–23 | 40 |
| C with exits at 10 (if intermediate CTC is added) | 10 | 14.6M | 22 | 17–28 | – |

\*The parameter count depends on the gate design (see §5).

**Reasoning:**
- **A:** LARM's 4-block result scaled for epochs and width. The LR sweep usually buys 3–10% relative on a mis-set baseline.
- **B(8):** "naive looping" gave LARM −53% relative from 4 unique blocks. Ours has less depth (18 vs 48) and fewer unique looped blocks (2), and B starts life as A. I predict −35 to −45% vs A_best.
- **The k-curve:** flat from 6 to 10, because Inkling varloop is flat above its mode and LARM saturates. +10% at k=4, because LARM exit 8/12 = +2% but exit 4/12 = +34%, and 4 is below our mode. Bad at k=2, which gets 1/12 of the training mass (Inkling's 1-pass collapse at 8.3% mass).
- **C vs B(8):** LARM's FLOP-matched 48-block plain model lost by 8% at 100h. But our C has only 4× B's parameters (LARM had 11×), so it is less over-parameterised for 100h. B also pays for training across k. **Median B(8)/C = 1.05, range 0.90–1.20.**
- **Effect of stronger SpecAugment (§5):** all numbers drop by ~15–30% relative (arXiv 2210.08520: 32.7→23.2 on test-other at 100h). The B–C gap likely narrows, because part of looping's 100h edge is regularisation (LARM's 960h reversal).

**Go criteria vs this evidence:**

| criterion | needed | predicted | P(pass) [E] |
|---|---|---|---|
| 1 sanity (B < 30, A < 45) | – | 19 / 32 | 0.9 |
| 2: B ≤ 0.8×A_best | ≤ 24 | 19/30 = 0.63 | 0.85 |
| 3: B ≤ 1.05×C | ≤ 18.9 | 19/18 = 1.05 | 0.45–0.5 |
| 4: some k<K with WER ≤ 1.05×C **and** ≥1.5× faster than C on CPU | k must be ≤4 | B(4)/C ≈ 1.17 | 0.2 |
| **go = 2 ∧ 3 ∧ 4** | | | **≈ 0.15–0.2** |

**Arithmetic on criterion 4 [E]:**
- B(k) executes 2+2k blocks against C's 18, plus ~1 block-equivalent of frontend and head. So the speedup is (19)/(3+2k):
  - k=4: 1.73×;
  - k=5: 1.46× (fails);
  - k=6: 1.27×.
- **Only k ≤ 4 can pass**, and k=4 must be within 5% of C while having only 56% of C's depth.
- LARM's closest analogue (exit 4 of K=12, 33% depth) was +34%. Exit 8 (67%) was +2%.
- So criterion 4 hinges on whether sampled-K training makes the curve much flatter than LARM's fixed-K exits. That is exactly the open question, which is good: criterion 4 is the informative test.
- **Caveat:** at 3.5M and batch 1, PyTorch eager CPU time may be dominated by per-op overhead rather than FLOPs. Measured speedups may be well below the FLOP ratio. Measure with ONNX Runtime as well as eager.

**What would be surprising (worth a post either way):**
- B(4) ≤ 1.05×C, i.e. the speed criterion passes.
- B(8) < 0.9×C.
- B improving beyond k=10 (extrapolation).
- B ≤ 15 dev-clean at 3.5M.
- A_best ≤ 22, which would mean LARM's baseline was badly tuned and its looping gain inflated.
- B > 0.8×A_best, which would contradict both LARM and Komatsu; suspect a bug or a gate that never opens.
- The gate staying near closed while B still beats A. That would be impossible under restart semantics, so it means a bug.

## 5. Cheap design improvements, ranked by expected gain vs risk to the comparison

| # | change | expected gain | risk to the comparison | verdict |
|---|---|---|---|---|
| 1 | **Same intermediate CTC in all deep arms.** B at k ∈ {2,4,6,8,10}; C at layers 6/10/14/18; shared head | Large at small k (LARM T4; Wright: joint exits beat single-exit models) | Low, if C gets the same treatment. **Without it C is a strawman for criterion 4** | **Do (must)** |
| 2 | **Stronger SpecAugment, same in all arms** (2 freq masks F=27, 5–10 time masks ≤5%, as in the ESPnet 100h recipe) | 15–30% relative absolute WER (2210.08520) | Low. It shrinks the regularisation advantage of looping, which makes a win more meaningful | **Do** |
| 3 | **Fix K sampling vs FLOP match.** Either put weights on {2,4,6,8,10} with mode 8 (1:1:2:6:2) or match C to the mode (14 blocks) | Removes a confound | None | **Do** |
| 4 | **Gate parameter budget.** One gate per core layer (not 4), rank 32: ~2×(2·256·32 + 2·32·256) ≈ 66k (+1.8%). Otherwise Inkling-faithful 4 gates × rank 64 ≈ 0.52M (+15%), and A must be widened to match | Keeps parameter matching honest | Medium if ignored | **Do** |
| 5 | **Gate init:** W_f = W_m = 0, down-projection random (std 0.02), message queries 0; log mean \|tanh(W_m g)\| per layer | Faithful to Inkling. The gradient on W_m is nonzero at 0, so the gate opens | Low | **Do** |
| 6 | **Carry-state + input-injection variant** (LARM/Geiping style: h_k = F(h_{k−1}) + β·h⁰, β init 0.5 or a closed gate) as arm B2 | Proven in ASR, where restart+message is not | Adds an arm. Cheap if the pilot is extended to 4 runs | Do if budget allows. It answers "is Inkling's restart the right choice from scratch" |
| 7 | **Per-utterance adaptive k at inference:** stop when greedy CTC output (or mean max-posterior) is unchanged between exits k and k+2 | Free; plausibly 20–40% average compute saved at ~equal WER [E]; the novelty lever | None (inference only) | **Do in eval** |
| 8 | No loop/step embedding (Inkling has none; it hurts extrapolation per Geiping §3.1) | – | – | Keep out; FiLM only as a later ablation |
| 9 | 8× frontend (12.5 fps) for all arms | ~2× speed, 4× cheaper attention | WER ±; CTC margin with BPE-256 drops to ~2.5× [E] | Not in the pilot |
| 10 | int8 (ONNX Runtime dynamic; Core ML W8A8) | 4× size; QAT int8 costs ~0.0–0.1 abs on a 960h conformer (2203.15952); PTQ on tiny models 0.1–0.5 [U] | None (post hoc) | Post-pilot measurement |
| 11 | Distillation from Moonshine/Whisper pseudo-labels on 360h | Large absolute | High: changes data, not architecture | Only after go, as its own arm |

## 6. Threats to validity

1. **No early exits for C.** The speed criterion compares B at small k against C at full depth. Plain models with exits also give a speed knob (Wright 2309.09546; I3D). Fix: #1 above.
2. **K mode ≠ FLOP-match point.** 1:2:6:2:1 on {2..10} has its mode at 6 (14 blocks); C matches k=8 (18). B is trained mostly at a depth C does not represent.
3. **Parameter match broken by gates.** A faithful Inkling gate adds ~15% parameters [E arithmetic]. Report unique parameters per arm and keep them within ±3%.
4. **Tuning budget favours A.** Only A gets an LR sweep, and C (14.6M, 18 deep) at LR 7e-4 may be mis-tuned in either direction. Give C and B at least a 2-point LR check at half steps (7e-4, plus 1.5e-3 or 3e-4), or the H2 verdict is not trustworthy.
5. **Training-compute confound.** A trains with 4 blocks; B averages 14 blocks per step; C 18. The plan's "A-long" control (equal training FLOPs) is necessary, not optional, for any claim about A vs B.
6. **Regularisation vs efficiency.** At 100h looping wins partly as a regulariser (LARM 960h reversal: 3.87 vs 4.59). A 100h win does not transfer automatically.
7. **Seeds.** One seed per arm. Seed-to-seed spread for small CTC models is typically a few tenths of absolute WER, up to ~1 at this WER level [U]. Treat B–C differences under ~5% relative as ties in the pilot.
8. **MPS numerics:**
   - CTC MPS backward was merged with `-f`, bypassing CI [V]. The ≥512-label inf-gradient fix (#203360) is in **no stable release**; 2.15.0 shipped 2026-10-28 [V]. BPE-256 targets stay far below 512, but assert it.
   - Run one MPS-vs-CPU gradient comparison on real batches in step 0.
   - bf16/fp16 softmax NaN on M5 with fully masked rows or a −1e9 constant (Apple forum 841213) [V-s]: train in fp32, use bool padding masks.
   - No float64; no guaranteed determinism.
   - torch.compile on MPS had wrong-result bugs fixed in 2.15 [V]: use eager, or validate against it.
9. **FLOP accounting.** `FlopCounterMode` has no formula for the MPS SDPA op, so attention counts 0 on MPS. Elementwise ops are always 0 and masks are ignored [V]. Count on CPU or meta and report attention separately: at T=375 frames attention is ~24% of block FLOPs [E].
10. **Wall-clock ≠ FLOPs at 3.5M.** At batch 1 the Python and op-launch overhead per block is comparable to the math. Use a fixed thread count (4 P-cores, plus 1 thread), warmup, median and p90, and `torch.mps.synchronize()`. Report eager plus ONNX Runtime.
11. **Dev reuse.** Model selection and the LR sweep happen on dev, so dev numbers are optimistic. Test stays closed until the final run, as planned.
12. **Text normalisation and tokenizer.** BPE-256 trained on train-clean-100 transcripts only. Identical normaliser for all arms and for the Whisper/Moonshine references (those need Whisper-style normalisation; otherwise their WER is inflated).
13. **torchaudio.** `compliance.kaldi.fbank` is kept in 2.9+ [V module-level]; the current version is 2.11.1 [V]. Precompute features once on CPU so frontend differences cannot leak between arms.

## 7. Staged roadmap

- **Stage 0, about half a day:**
  - Install torch 2.15.0.
  - Features.
  - MPS-vs-CPU CTC gradient check.
  - Throughput calibration for A, B, C: the M5 GPU is ~5 TFLOPS FP32 on paper [U]; expect 1–2 effective. One epoch of C ≈ 6 × 14.6M × 9M frames ≈ 7.9e14 FLOP, i.e. 7–13 min [E].
  - Write predictions.md with the table in §4.
- **Stage 1, pilot (~1 day):**
  - A, A_best rerun, B, C (with exits), and optionally B2 carry-state, all at 15 epochs, dev only.
  - Full k-curve 1–16, adaptive-k eval, CPU RTF.
- **Stage 2, if go or a "2 ∧ 3 only" outcome (~4–6 days of Mac time):**
  - 50 epochs, 3 seeds.
  - A-long; per-k specialists at k = 4 and 8; B2; gate ablation (closed vs open init).
  - Test set once.
  - This is the workshop-paper package.
- **Stage 3:**
  - Train-960 at 3.5M for B, C and B2 only. It is ~9.6× the data; ~20 epochs ≈ 2–4 days per arm [E], and a 60 GB download.
  - This answers the regularisation question. Do it only if Stage 2 shows a clean B ≥ C.
- **Stage 4, release:**
  - HF model card with int8 ONNX (<5 MB), a k-slider demo, a Pareto plot against Whisper tiny.en and Moonshine Tiny measured on the same Mac, and a blog post.
  - Distillation from Moonshine (360h of pseudo-labels) as the "product" variant, kept clearly separate from the architecture claim.
- **What a no-go teaches:**
  - **Fails 2:** looping does not help even at equal parameters at 3.5M. That contradicts LARM and Komatsu, so check the gate opening and look for bugs before concluding.
  - **Fails 3 only:** loops buy parameter efficiency, not compute efficiency. That is useful for flash/RAM-bound targets: report size-vs-WER and stop.
  - **Fails 4 only:** a sampled-K looped model is no better a speed knob than early exits. That is a publishable negative for the blog, since nobody has measured it.

## What to change in brief.md / plan.md

1. brief.md: LARM's looped 7.7M **beats** the 85.7M FLOP-matched model at 100h (11.34 vs 12.24) and loses at 960h. Cite Komatsu 2022 (6.8M shared, 9.5/25.6) as the real bar.
2. plan.md §3: delete "no <10M 100h-only models except LARM". Add Komatsu, Rolland 2606.09357 and the ESPnet E-Branchformer-CTC (9.6/23.1) to the reference table.
3. plan.md §3 arm B: say explicitly "restart from prelude + gated message (tanh, zero-init, no bias, no step embedding)". Note that B ≡ a 4-block model at init. Add the gate-opening log.
4. Gate budget: one gate per core layer, rank ≤32 (~+2% parameters). Report unique parameters for every arm.
5. K sampling: change the weights so the mode is 8 (e.g. 1:1:2:6:2 on {2,4,6,8,10}), or match C to 14 blocks. State which k the FLOP match uses.
6. Give C intermediate CTC exits at 6/10/14/18 (and B at every trained k). Criterion 4 compares B(k) with C at an exit of equal FLOPs, not only full-depth C.
7. Give B and C a 2-point LR check. Make A-long mandatory, not post-go.
8. Raise SpecAugment to the ESPnet-100h policy in all arms. Keep LARM's light policy only as a footnote ablation.
9. Replace H1–H5 predictions with §4's numbers (A_best 30, B(8) 19, C 18, B(4) 21) and P(go) ≈ 15–20%. Add adaptive-k and extrapolation (k=12–16) to the evaluation.
10. Stack: torch 2.15.0, fp32 eager, bool masks, MPS-vs-CPU CTC gradient check, FLOPs counted on CPU/meta, RTF in eager **and** ONNX Runtime (also the int8 path; dynamic quantization on qnnpack is deprecated).
