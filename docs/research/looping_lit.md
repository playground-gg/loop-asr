# Looped / weight-shared depth: literature findings for loop-asr (compiled 2026-11-23)

How these were read: VERIFIED means I read the primary source this session, either as raw HTML or code that I grepped myself, or through a fetch summarizer, marked (S). UNVERIFIED means from memory or from an abstract only.

## Part 1: Inkling 2 Mini (Thinking Machines, 2026-11-20)

Sources:
- Blog (S): https://thinkingmachines.ai/news/inkling-2-mini/
- Model card (S): https://thinkingmachines.ai/model-card/inkling-2-mini/
- Varloop card (S): https://thinkingmachines.ai/model-card/tml-d24-varloop-research/
- HF docs (raw): https://huggingface.co/docs/transformers/en/model_doc/inkling2_loop
- HF blog (S): https://huggingface.co/blog/inkling2mini
- Code (raw, read by me): transformers v5.24.0 `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`
- Configs (raw): `thinkingmachines/Inkling-2-Mini/config.json` and `TML-d24-varloop-research/config.json`

**Layout.** 24 physical layers in 4 attention-residual blocks of 6 (`n_layers_per_attn_res_block=6`).
- Prelude: layers 0–5, run once.
- Core: layers 6–17, i.e. 2 blocks or 12 layers, run `num_passes` times.
- Coda: layers 18–23, run once.
- Model card: "The middle 12 layers run three times with shared weights, giving 48 layer applications per token from 24 layers of weights."
- Size: hidden 2304, MoE with 8 of 128 experts, 26.4B total / 3.4B active parameters. Each extra pass adds about 1.0B effective active parameters.
- Your (B) layout (1 prelude block + 2 shared core blocks + 1 coda block) matches it structurally.

**Config.** Both checkpoints have `loop_config = {core_start:6, core_end:18, num_passes:3, gate_rank:64}`. Only `is_varloop` differs (false/true). In the HF docs, `loop_gate_rank` (default 64) is the "Rank of the message gate projections that mix the residual blocks of the previous pass into the current one."

**Each pass restarts from the prelude output (VERIFIED, code ~L1397).**
- Code comment: `# Remaining passes restart from the pre loop states and receive messages from the previous pass` followed by `hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states`.
- So the hidden state is not carried from one pass to the next. Cross-pass information flows only through the message. This differs from LARM and Geiping, which carry the state forward and inject the input.

**What the message is (VERIFIED, code L145–173).**
- `Inkling2LoopMessageAttentionResidual` is a softmax attention, done per token, over the previous pass's core block states: the 2 block-boundary residuals plus the final partial hidden state.
- The query is a learned vector. There is one per (loop-layer, {q,k,v,mlp}), with shape `(12, 4, hidden)`, and it is initialised with `torch.zeros`. At init the read is therefore a uniform average.
- Keys and values are RMS-normed with no weights.
- The query is indexed by loop layer, not by pass. There is **no per-pass or step embedding** anywhere in the code.
- HF blog (S): "During passes 2 and 3, each core layer attends over two residual blocks from the previous pass."

**Gate (VERIFIED, code L79–96).** `Inkling2LoopMessageGate` "Mixes hidden states and message as `(1 - forget_gate) * hidden_states + message_gate * message`."
- `g = down_proj(cat[rmsnorm(h), rmsnorm(m)])`, with `down_proj` 2d→64 and no bias.
- `forget = tanh(W_f g)`, `msg = tanh(W_m g)`, with W_f and W_m 64→d, fp32, no bias.
- There are 4 such gates per core layer, for the q, k and v inputs and the MoE input. They are applied after the attention-residual input mix and before the layer norm.
- The gate is **tanh, not sigmoid, and there is no bias**. If W_f = W_m = 0, the output is exactly h (gate fully closed).
- Blog (S): the gate "initially blocks the entire message", so each loop starts out acting like a non-looped model.
- In HF `_init_weights`, the gate projections get no special init, so they fall back to the generic normal init with std 0.02. That is only the init used when loading the weights; TML's own training init is not public (UNVERIFIED: exact zero-init of W_m).
- The architecture can be closed exactly at zero weights, and the blog says it starts closed: **"starts closed" is consistent and supported, but the exact initial values are UNVERIFIED.**

**Pass-count schedule.**
- Production Inkling 2 Mini uses a **fixed 3 passes**. Blog: "Inkling 2 Mini only supports a fixed number of looping passes." Your belief is confirmed.
- The varloop research model is the variable one. Its card says it was trained on "1 to 5 passes… sampled in a 1:2:6:2:1 ratio" (8.3/16.7/50/16.7/8.3%; the summarizer's "10/20/60/20/10%" is wrong arithmetic) during the rest of midtraining once looping was introduced, "and it was then post-trained with 3 passes". RL was done at 3 passes only.
- The card warns it "may behave unpredictably at pass counts other than 3."
- Inference: `text_config={"loop_num_passes": N}`, with N ≥ 1.
- Blog (S): it held up at 10 passes on no-CoT tasks, which is beyond the training maximum of 5. No-CoT 50% time horizon: 1p 16 s, 2p 26 s, 3p 37 s, 5p 42 s, 10p 40 s.

**Loss.** The code computes only the final-layer LM loss; nothing is computed at intermediate passes. Training-time deep supervision is not documented (UNVERIFIED).

**Other blog figures (S).**
- Looping the middle layers gave 1.51–1.91× compute-equivalent gain, against 1.24–1.36× for looping the last layers. Middle won 27 of 28 evaluations.
- A non-looped model needed "7.5% less compute to train, but required 140% more inference compute to match" the looped model's loss.

**Varloop card, scores by pass count (1p / 2p / 3p / 4p / 5p):**

| Benchmark | 1p | 2p | 3p | 4p | 5p | Inkling 2 Mini (3p) |
|---|---|---|---|---|---|---|
| SWE-bench Verified | 56.3 | 76.5 | 78.7 | 79.3 | 79.6 | 79.2 |
| SWE-bench Pro v2 (Hard) | 39.7 | 69.1 | 78.9 | 76.0 | 78.4 | 77.9 |
| SWE-bench Multilingual | 40.8 | 68.5 | 71.8 | 72.8 | 72.0 | 73.7 |
| DeepSWE | 0.7 | 28.3 | 47.4 | 46.9 | 48.2 | – |
| GPQA-D | 69.6 | 79.2 | 84.2 | 81.9 | 83.1 | – |
| MRCR v2 | 27.2 | 65.2 | 89.2 | 93.4 | 92.1 | – |

The pattern:
- Big gains from 1→2→3 passes.
- Roughly flat or noisy at 4–5 passes, past the mode of the training distribution.
- 1 pass collapses badly even though it got 8.3% of the training mass.

## Part 2: LARM (arXiv 2606.04678v2; VERIFIED, raw HTML grepped)

- Paper: Kaloga, Kumar, Sheikh, Khalil, Motlicek, Kodrasi, "Test-Time Compute Scaling for ASR with Depth-Conditioned Looped Transformers". v1 2026-06-03, v2 2026-06-15. https://arxiv.org/html/2606.04678v2
- **Code:** "The code and selected checkpoints will be released soon". Not released at v2.

**Architecture (§3, App. B.1).**
- Pre-norm Transformer, not a Conformer.
- Input: 80-dim fbank (25/10 ms).
- Frontend: 2× Conv2d (64 channels, 3×3, stride 2, SiLU), giving 4× time subsampling. Its 64×20 = 1280-dim output goes through a linear layer to d, then dropout 0.1.
- RoPE (base 10k), head dim 64 (H = d/64), FFN ratio 4.
- Default: d = 384, N = 4 shared blocks, K = 12 loops, which is 48 block passes. 7.7M parameters.
- Tokens: character CTC with 30 symbols (letters, apostrophe, word boundary, `<unk>`, blank).

**Recurrence (§3).**
- Shared encoder: z(k) = F_θ(h(k−1)).
- Shared CTC head at every loop: p(k) = softmax(z(k)Wψ + b).
- **Feedback:** r_t(k) = p_t(k)·W_ρ (|V|→d), delayed by one frame: r̄_t = r_{t−1}, zero at t = 0.
- **Aggregation:** a(k) = z(k) + β·h(0) + α·r̄(k). This is input injection of the frontend output. α and β are learned scalars initialised to 0.5.
- **Supervision-clock embedding:** e(k) = W_c[(k−1) mod c], added to the state.
- **FiLM depth conditioning:** h(k) = γ(d̄)⊙â(k) + β_film(d̄), with d̄ = (k−1)/(K−1). γ and β_film are MLPs ℝ→ℝ^d with hidden size 64.
- **Sparse CTC checkpoints** at S = {c, 2c, …, K}. With c = 4 and K = 12 these are {4, 8, 12}.
- Loss: the mean CTC loss over S.
- **K is fixed per model during training; there is no sampling.**
- d̄ depends on K, so the depth conditioning is tied to the trained K.

**Training (App. B.2).**
- AdamW (0.9, 0.999, 1e-8), weight decay 5e-3, gradient clip 1.0.
- Cosine schedule, peak LR 7e-4, minimum 0.03×peak. Warmup 1k steps (100h) or 10k steps (960h).
- 50 epochs; some runs 100.
- Effective batch 32 (with gradient accumulation).
- Light SpecAugment: 1 frequency mask with F = 15, 2 time masks with p = 0.02, filled with the utterance mean.
- Decoding: greedy CTC, plus 4-gram KenLM (beam 100, LM weight 0.5, word insertion bonus 1.0).
- **Not reported:** hardware, training time, FLOPs, latency, dev-set WER.

**Table 1: test-clean / test-other WER, greedy, then +4-gram LM.**

| Data | Model | Params | Greedy clean / other | +LM clean / other |
|---|---|---|---|---|
| 100h | 4-block | 7.6M | 26.78 / 52.82 | 13.44 / 36.60 |
| 100h | 16-block | 28.9M | 14.43 / 37.23 | 9.97 / 28.68 |
| 100h | LARM (4 blocks, K=12) | 7.7M | **11.34 / 31.84** | 8.66 / 26.28 |
| 960h | 4-block | 7.6M | 14.07 / 29.82 | 6.77 / 17.39 |
| 960h | 16-block | 28.9M | 4.79 / 13.26 | 3.51 / 9.87 |
| 960h | LARM (4 blocks, K=12) | 7.7M | 4.59 / 11.75 | 3.51 / 9.38 |

**Table 2: width scaling (same column layout).**

| Data | Model | Params | Greedy clean / other | +LM clean / other |
|---|---|---|---|---|
| 100h | 48-block | 85.7M | 12.24 / 34.06 | 9.03 / 27.07 |
| 100h | LARM d=768 | 28.9M | 9.58 / 28.25 | 7.57 / 23.89 |
| 100h | LARM d=1024 | 85.7M | 8.95 / 27.93 | 7.22 / 23.79 |
| 960h | 48-block | 85.7M | 3.87 / 10.58 | 3.20 / 8.56 |
| 960h | LARM d=768 | 28.9M | 3.45 / 9.44 | 2.93 / 7.93 |
| 960h | LARM d=1024 | 85.7M | 3.39 / 9.01 | 2.83 / 7.67 |

Caveat: the parameter counts look inconsistent. 4 blocks at d = 1024 should be about 50M parameters, not 85.7M.

**Table 3: 100h ablations, test-clean, greedy.**
- (a) 4-block 26.78; naive looped 12.70; LARM 11.34. **Most of the gain comes from plain looping.**
- (b) Depth conditioning: none 12.13, embedding 11.98, MLP 11.77, FiLM 11.34.
- (c) Feedback: current-frame 11.43, no feedback 12.31, fixed α/β 11.91, LARM 11.34.

**Table 4: loop exits, 100h, c = 4, greedy test-clean at each exit.** Best LM is the best LM WER across all exits, with the exit in parentheses.

| Trained K | Exit 4 | Exit 8 | Exit 12 | Exit 16 | Exit 20 | Exit 24 | Best LM | Final LM |
|---|---|---|---|---|---|---|---|---|
| 4 | 14.07 | – | – | – | – | – | 9.11 (4) | 9.11 |
| 8 | 14.22 | 12.07 | – | – | – | – | 8.67 (8) | 8.67 |
| 12 | 15.20 | 11.57 | 11.34 | – | – | – | 8.38 (7) | 8.66 |
| 16 | 16.12 | 11.94 | 11.30 | 11.24 | – | – | 8.28 (7) | 8.66 |
| 20 | 16.95 | 11.83 | 11.17 | 11.05 | **11.02** | – | 8.03 (7) | 8.43 |
| 24 | 20.05 | 14.33 | 13.52 | 13.39 | 13.31 | 13.34 | 9.02 (11) | 9.45 |

What Table 4 shows:
- Early exits from a large-K model are worse than a model trained for that K. For example, exit 4 is 20.05 in the K = 24 model against 14.07 for a K = 4 model.
- Gains saturate around 16–20 loops; the K = 24 model got worse.
- With an LM, the best result often comes from an unsupervised intermediate loop (7).
- The paper does not evaluate looping beyond the trained K.

**Table 5: extended runs, 100h, test-clean.**
- d768 / N4 / K16 / c4 / 100 epochs: 8.39 greedy, best LM 6.83 (exit 9).
- d1024 / K16 / 100 epochs: 8.27 / 6.93.
- d384 with N6/K8: 10.86. With N8/K6/c3: 10.85. With N4/K12: 11.34.
- So more unique blocks at the same 48 block passes helps slightly.

**For your project:** at 100h, a 7.7M looped model already roughly halves the WER of a 4-block plain model (26.8 → 11.3) and beats a 16-block model (14.4). Your ~3.5M model has about half of LARM's width or parameters.

## Part 3: Related work, in brief

- **Universal Transformer** (Dehghani 2018, arXiv 1807.03819; UNVERIFIED). Shared block, adds position plus timestep "coordinate" embeddings at each step, optional ACT. WMT14 En-De 28.9 BLEU against 28.4 for the base Transformer; bAbI-10k solved, average error about 0.2%.
- **ALBERT** (Lan 2019, Table 4; VERIFIED, ar5iv). Base model, average over 5 tasks:
  - E = 768: all-shared 31M → 79.8, not-shared 108M → 82.3 (−2.5).
  - E = 128: all-shared 12M → 80.1, not-shared 89M → 81.6 (−1.5).
  - Sharing only attention costs nothing (E = 128: 81.7). **The loss comes from sharing the FFN** (shared-FFN 80.2).
- **ACT** (Graves 2016, arXiv 1603.08983; UNVERIFIED). Per-step halting probability, with ponder cost τ·(N+R). It is sensitive to τ, and the halting signal's gradient is biased.
- **PonderNet** (Banino 2021, arXiv 2107.05424; UNVERIFIED). Probabilistic halting with a geometric prior and a KL term. On parity it extrapolates to harder inputs where ACT fails; bAbI comes with lower compute than UT.
- **Recurrent depth** (Geiping 2025, arXiv 2502.05171; VERIFIED (S) + raw grep).
  - Prelude/core/coda (2, 4, 2), h = 5280, 3.5B parameters.
  - **The input e is injected at every step** by concatenation with an adapter 2h→h.
  - s0 ~ truncated normal (σ² = 2/5).
  - Training r ~ Poisson(e^τ) + 1 with τ ~ N(log r̄ − σ²/2, σ), σ = ½, r̄ = 32.
  - Truncated backprop through the last k = 8 steps; about 800B tokens.
  - §3.1: a step-conditioned core block "interacts badly with path independence, leading to models that cannot extrapolate."
  - §4.3: one failed run "learned early to ignore the incoming state" (same perplexity at r = 1 and 32).
  - GSM8K CoT 34.8 / 42.1 at r = 32; 0.00 at r = 1.
- **Mixture-of-Recursions** (Bae 2025, arXiv 2507.10524; VERIFIED raw).
  - Per-token router over recursion depth.
  - Middle-Cycle sharing works best (§4.1).
  - At an equal 16.5e18 FLOPs, Nr = 2 MoR beats vanilla (43.1 vs 42.3 average few-shot) with about 50% fewer parameters.
  - Up to 2.06× throughput with depth-wise batching.
- **Relaxed Recursive Transformers** (Bae 2024, arXiv 2410.20672; abstract VERIFIED). Per-loop, layer-wise LoRA relaxes the weight tying. Recursive Gemma 1B beats TinyLlama 1.1B and Pythia 1B and recovers most of Gemma 2B. Throughput gain of 2–3× is theoretical only.
- **Looped-transformer theory.**
  - Giannou 2023 (arXiv 2301.13196; UNVERIFIED): a 13-layer looped transformer is a programmable computer.
  - Yang 2023 (arXiv 2311.12424; UNVERIFIED): in-context learning matches a standard transformer with under 10% of the parameters. It uses input injection and trains with randomly sampled loop counts using a truncated loss window.
  - Saunshi 2025 (arXiv 2502.17416; abstract VERIFIED): a k-layer model looped L times "nearly matches" a kL-layer model on reasoning tasks but is worse on memorization. T loops can simulate T steps of CoT.
- **"How to Loop Your Transformer"** (Ekbote et al., Meta, arXiv 2610.40325, 2026-10-29; VERIFIED (S)).
  - Covers models up to 1B parameters, 155B tokens, L ≤ 32.
  - Looping the full stack is best at fixed parameters; looping the middle stack is best at a low FLOP budget.
  - Gains saturate or become unstable at high L.
  - Loop-specific low-rank adapters (rank ≤ 8, +1.1–4.3% parameters) help at L ∈ {4, 8} but only marginally at L = 2.
  - Fixed 50/50, scalar, linear and MLP gates perform about the same. Separately RMS-norming the two streams before mixing matters.
  - It does not study training at one loop count and testing at another.
- **LoopMDM** (arXiv 2605.26106; abstract only). Loops the early-middle layers of a masked diffusion LM. Matches same-size models with up to 3.3× fewer training FLOPs; +8.5 GSM8K. The loop count can vary at inference.
- **Depth through recurrence, TTS** (arXiv 2609.29768; abstract only). Flow-matching TTS at 18 block calls. The best layout runs 9 blocks twice each, consecutively (−47.1% parameters, competitive). Cycling 6 blocks 3 times gives higher WER. Middle placement is the most robust.
- **Loop-count robustness.**
  - Kuo et al., NeurIPS 2026 (arXiv 2606.29983; abstract only): randomising the loop count during training "sharply reduces OOD variance"; learned RL-Halting improves on that.
  - LoopFormer (arXiv 2602.11451; abstract only): trains on variable-length trajectories with time/step-size conditioning and a shortcut-consistency loss.
  - Yang et al. (arXiv 2610.14456; abstract only): there is a finite window of extrapolation depth; too many loops "create false connections."

## Part 4: Design lessons for your models (A), (B) and (C)

**(a) Input injection.**
- Strong evidence that it matters for stable recurrence that carries state:
  - Geiping (§3.1, citing Bansal 2022 and Anil 2022): needed for path independence.
  - LARM: β·h0, with α/β learned rather than fixed 11.34 against 11.91.
  - Ekbote: removing the embedding makes runs worse than no looping, and some diverge.
- Inkling avoids the issue differently: every pass *restarts* from the prelude output, and the previous pass arrives only as a gated message.
- Your plan B ("cross-loop message through a gate") is Inkling-like, so input injection is built in by construction.
- Decide explicitly whether the state carries over: restart plus message, or carry plus h0.

**(b) Loop or step embeddings.**
- They give a small gain at a fixed K: LARM FiLM 11.34 against none 12.13; Ekbote's adapters.
- They hurt extrapolation: Geiping §3.1.
- Inkling has none.
- Since you choose K at inference over {2, …, 10}, avoid absolute step conditioning such as LARM's d̄ = (k−1)/(K−1), which needs K known in advance. If you add one, ablate it.

**(c) Zero-init or closed residual branches.**
- Inkling: tanh gates without bias, so zero weights mean an identity map with the message ignored. The message-read queries are zero-initialised.
- ReZero (Bachlechner 2020, arXiv 2003.04887; UNVERIFIED): α = 0 per residual branch, 56% faster convergence on enwik8.
- SkipInit (De & Smith 2020; UNVERIFIED) and Fixup (Zhang 2019, arXiv 1901.09321; UNVERIFIED): train 10,000-layer ResNets without normalisation.
- Risk, from Geiping §4.3: a model can learn to ignore the recurrent input. So monitor the gate magnitude, i.e. the mean |tanh(W_m g)|, during training.

**(d) Sampling K during training.**
- Inkling varloop: the 1:2:6:2:1 schedule gives large losses at the rare low end (SWE-V 56.3 at 1 pass) and a plateau above the mode (79.3 / 79.6 at 4 / 5 passes). It survived 10 passes on a no-CoT task.
- Geiping: heavy-tailed sampling plus truncated BPTT (k = 8).
- Kuo 2026: randomising K reduces OOD variance.
- LARM, with a fixed K, has early exits much worse than dedicated models (exit 4 = 20.05 against 14.07).
- Expect your K = 2 and K = 10 to be weak, since each gets only 1/12 of the mass. Consider deep supervision: CTC at several K, as in LARM's checkpoints.

**(e) Per-token adaptive halting.**
- ACT and PonderNet need a ponder-cost hyperparameter.
- MoR is the strongest recent result: −50% parameters, +0.8 few-shot accuracy, up to 2.06× throughput. But the routing is unstable ("struggles to balance loads… nearly half of the training steps").
- Kuo: learned stopping beats heuristic randomisation.
- For CTC, frames are independent, so halting per frame is feasible. The cost is padding and masking: no real FLOP savings without depth-wise batching.
- Recommendation: keep halting out of scope and pick K per utterance or globally.

## Your FLOP-matched baseline (C) against LARM's numbers

- B's maximum depth is 1 + 2·10 + 1 = 22 block applications; the weighted mean K is 6, giving 14.
- Your C uses 18 blocks, which equals B at K = 8 (1 + 16 + 1), not at the mean K = 6 (14). State which K the FLOP match refers to.
- LARM's closest analogue to your C is its 48-block baseline (85.7M). It costs the same FLOPs as LARM with N = 4 and K = 12, which is 48 block passes.
- At 100h, LARM (7.7M) beat that FLOP-matched 48-block model: 11.34 against 12.24 greedy, 8.66 against 9.03 with the LM.
- At 960h, the 48-block model won: 3.87 against 4.59.
- So at 100h, an unshared deep stack is not better than a looped one with 11× fewer parameters; at 960h it is. Expect your C to be roughly even with B on train-clean-100.
