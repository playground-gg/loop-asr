# Tiny English ASR frontier map (checked 2026-11-23)

Notation: "greedy" means no LM. tc/to = LibriSpeech test-clean/test-other WER (%). Every number below was read on the cited page during this session unless it is marked UNVERIFIED.

## Table 1: General frontier (any training data)

| Model | Params | Train data | tc / to (no LM) | tc / to (with LM) | Latency / RTF | Source |
|---|---|---|---|---|---|---|
| Moonshine Tiny (v1) | 27.1M | ~200K h | 4.52 / 11.71 | - | 27 ms response latency, 5.91% compute load, Apple M3 (v2 paper T2); 0.7x Whisper tiny.en FLOPs (T1) | arXiv 2410.15608 T1, T3; arXiv 2602.12241 T2 |
| Moonshine Base (v1) | 61.5M | ~200K h | 3.23 / 8.18 | - | 44 ms, 7.34%, Apple M3 | arXiv 2410.15608 T1, T2; 2602.12241 T2 |
| Moonshine v2 Tiny (streaming; HF `moonshine-ai/moonshine-streaming-tiny`) | 33.57M (enc 7.39M) | ~300K h | 4.49 / 12.09 (Open ASR avg 12.01) | - | 50 ms, 8.03%, Apple M3 | arXiv 2602.12241 T1, T2, T3 |
| Moonshine v2 Small | 123.36M | ~300K h | 2.49 / 6.78 | - | 148 ms, 17.97%, M3 | 2602.12241 T1-T3 |
| Moonshine v2 Medium | 244.93M | ~300K h | 2.08 / 5.00 | - | 258 ms, 28.95%, M3 | 2602.12241 T1-T3 |
| Whisper tiny.en | 39M (Whisper T1); 37.8M in Moonshine T1 | 680K h | 5.6 / 14.6 (Whisper paper); 5.66 / 15.45 (Moonshine T3 re-eval) | - | Moonshine v2 T2 "Whisper Tiny": 289 ms, 8.46%, Apple M3. Moonshine v1 intro: ">=500 ms lower bound" on a low-cost ARM chip | arXiv 2212.04356 App. D.1.1 (greedy); 2410.15608 T3 |
| Whisper tiny | 39M | 680K h | 7.6 / 16.9 | - | (same as above) | 2212.04356 App. D.1.1 |
| Whisper base.en | 74M; 72.6M in Moonshine T1 | 680K h | 4.2 / 10.2 (paper); 4.25 / 10.35 (Moonshine T2) | - | Whisper Base: 553 ms, 16.19%, M3 | 2212.04356 D.1.1; 2410.15608 T2; 2602.12241 T2 |
| distil-small.en | 166M | ~22K h | only "3.43 on validation.clean" (example script output); short-form avg 12.1 | - | "5.6x faster than large-v2" (device not stated) | huggingface.co/distil-whisper/distil-small.en |
| QuartzNet 5x5 (CTC) | 6.7M | LS-960 | dev only: 5.39 / 15.69 | - | - | arXiv 1910.10261 T2 |
| QuartzNet 15x5 (CTC) | 18.9M | LS-960 | 3.90 / 11.28 | 2.96 / 8.07 (6-gram); 2.69 / 7.25 (TXL) | - | 1910.10261 T4 |
| Citrinet-256 (CTC) | 9.8M (10.2M in T9) | LS | 3.78 / 9.6 | 3.65 / 8.06 (6-gram); 2.75 / 6.87 (Transf.) | - | arXiv 2104.01721 T3 |
| Citrinet-384 (CTC) | 21.0M | LS | 3.20 / 7.90 | 2.94 / 6.71; 2.52 / 5.95 | - | 2104.01721 T3 |
| NeMo stt_en_citrinet_256 (NGC) | not stated | ASRSet >7K h | 4.4 / 10.7 (decoding not stated) | - | - | catalog.ngc.nvidia.com .../stt_en_citrinet_256 |
| NeMo stt_en_conformer_ctc_small | ~13M | NeMo ASRSet, several K h | 3.7 / 8.1 (greedy) | 2.6 / 6.0 (n-gram); 2.2 / 5.2 (n-gram + rescorer) | Open ASR mean WER 11.16 | huggingface.co/nvidia/stt_en_conformer_ctc_small |
| Parakeet TDT-CTC 110M | ~114M | 36K h | 2.4 / 5.2 (TDT greedy; table row is labelled v2.0) | - | RTFx ~5300 (5,345.14), A100 | huggingface.co/nvidia/parakeet-tdt_ctc-110m |
| Canary-180M-flash | 182M | 85K h | 1.87 / 3.83 (Open ASR table; card metadata says 2.87 for to) | - | RTFx 1233 (A100), 2041 (H100) | huggingface.co/nvidia/canary-180m-flash |
| Conformer-CTC-S (Squeezeformer authors' reproduction) | 8.7M | LS-960, 500 ep | 4.06 / 10.58 | - | 26.2 GFLOPs | arXiv 2206.00888 T3 |
| Squeezeformer-XS (CTC) | 9.0M | LS-960, 500 ep, SPM-128 | 3.74 / 9.09 | - | 15.8 GFLOPs | 2206.00888 T3 |
| Squeezeformer-S | 18.6M | LS-960 | 3.08 / 7.47 | - | 26.3 GFLOPs | 2206.00888 T3 |
| Squeezeformer-SM | 28.2M | LS-960 | 2.79 / 6.89 | - | 42.7 GFLOPs | 2206.00888 T3 |
| Efficient Conformer CTC (S) | 13.2M | LS (~1000 h), BPE-256 | 3.57 / 8.99 | 2.72 / 6.66 (6-gram) | - | arXiv 2109.01163 T2 |
| Efficient Conformer RNN-T (S) | 10.8M | LS, 1k vocab | 3.28 / 8.03 | 2.83 / 7.05 | - | 2109.01163 T2 |
| Conformer (S), RNN-T | 10.3M | LS-960, 1k WPM | 2.7 / 6.3 | 2.1 / 5.0 (LSTM LM) | - | arXiv 2005.08100 T2 |
| ContextNet (S), RNN-T | 10.8M | LS-960 | 2.9 / 7.0 | 2.3 / 5.5 | - | arXiv 2005.03191 T2 |
| HyperConformer small (CTC+attn, beam 6) | 7.9M | LS-960 | 5.3 / 12.3 | 2.9 / 7.0 | - | arXiv 2305.18281 T1 |
| Conformer small (SpeechBrain, same paper) | 8.7M | LS-960 | 5.0 / 11.3 | 3.1 / 6.8 | - | 2305.18281 T1 |
| icefall Zipformer-S (transducer) | 23.3M | LS-960, 50 ep | 2.46 / 5.86 (greedy); 2.42 / 5.73 (MBS) | - | - | icefall egs/librispeech/ASR/RESULTS.md |
| icefall Zipformer-S + CR-CTC | 22.1M | LS-960 | 2.57 / 5.95 (CTC greedy); 2.52 / 5.85 (prefix beam) | - | - | same RESULTS.md |
| icefall streaming Zipformer, "smaller" | ~20M | (data not stated) | 3.94 / 9.79 (greedy, 320 ms chunk, simulated streaming) | - | - | same RESULTS.md |
| icefall streaming Zipformer small | 6.1M | LS + GigaSpeech (giga-prob 0.9) | 5.95 / 15.03 (greedy, 320 ms) | - | - | same RESULTS.md |
| LARM 4-block looped, K=12 (CTC, chars) | 7.7M | LS-960, 50 ep | 4.59 / 11.75 | 3.51 / 9.38 (4-gram) | - | arXiv 2606.04678 T1 |
| Plain 4-block Transformer-CTC (LARM baseline) | 7.6M | LS-960 | 14.07 / 29.82 | 6.77 / 17.39 | - | 2606.04678 T1 |
| Vosk small-en-us-0.15 (Kaldi) | 40M is the file size, not a parameter count | n/a | 9.85 (tc; LM in graph) | - | runs on Android and Raspberry Pi | alphacephei.com/vosk/models |
| Kyutai stt-1b-en_fr | ~1B | 2.5M h pretraining + fine-tuning | UNVERIFIED (not on the model card) | - | 0.5 s delay | huggingface.co/kyutai/stt-1b-en_fr |
| Moonshine "micro" SpellingCNN | ~1.3 MiB flash | n/a | no WER; command/spelling recognition, not open-vocabulary LibriSpeech ASR | - | RP2350 MCU, about 36 MMAC/s | github.com/moonshine-ai/moonshine micro/README.md |

## Table 2: The comparison bar (train-clean-100 only, no extra data, no SSL)

| Model | Params | Loss / decoding | Tokenizer | Epochs | tc / to (no LM) | tc / to (with LM) | Source |
|---|---|---|---|---|---|---|---|
| Plain Transformer-CTC, 4 blocks, d=384 | 7.6M | CTC greedy | chars (30) | 50 | 26.78 / 52.82 | 13.44 / 36.60 (4-gram KenLM, beam 100) | arXiv 2606.04678 T1 |
| Plain Transformer-CTC, 16 blocks | 28.9M | CTC | chars | 50 | 14.43 / 37.23 | 9.97 / 28.68 | 2606.04678 T1 |
| Plain Transformer-CTC, 48 blocks | 85.7M | CTC | chars | 50 | 12.24 / 34.06 | 9.03 / 27.07 | 2606.04678 T2 |
| Naive looped, 4 blocks (no depth conditioning) | ~7.6M | CTC | chars | 50 | 12.70 (tc only) | - | 2606.04678 T3a |
| LARM, 4 shared blocks, K=12, d=384 | 7.7M | CTC | chars | 50 | 11.34 / 31.84 | 8.66 / 26.28 | 2606.04678 T1 |
| LARM, N=6, K=8, d=384 | n/r | CTC | chars | 50 | 10.86 (tc) | 8.25 | 2606.04678 T5 |
| LARM, N=8, K=6, d=384 | n/r | CTC | chars | 50 | 10.85 (tc) | 8.34 | 2606.04678 T5 |
| LARM, K=12, d=768 | 28.9M | CTC | chars | 50 | 9.58 / 28.25 | 7.57 / 23.89 | 2606.04678 T2 |
| LARM, K=16, d=1024 | n/r | CTC | chars | 100 | 8.27 (tc) | 6.95 | 2606.04678 T5 |
| ESPnet Conformer-CTC, 18 blocks, d=256 (egs3 PR) | 31.7M | CTC | n/r | 180 | 10.72 / 25.18 | - | github.com/espnet/espnet/pull/6944 |
| Conformer-CTC, 18 layers, d=256 (InterAug paper) | n/r (likely ~30M) | CTC greedy | SPM-300 | 50 | 9.08 / 23.68 | - | arXiv 2204.00174, results table |
| + SelfCond (intermediate CTC) | same | CTC greedy | SPM-300 | 50 | 7.48 / 21.31 | - | 2204.00174 |
| + InterAug (token substitution) | same | CTC greedy | SPM-300 | 50 | 7.23 / 20.34 | - | 2204.00174 |
| ESPnet E-Branchformer-CTC, e12 | 26.43M | CTC | BPE-5000 | n/r | 9.6 / 23.1 | - | espnet egs2/librispeech_100/asr1/README.md |
| ESPnet Conformer (CTC 0.3 + attention), beam 1 | n/r | hybrid | n/r | n/r | 8.4 / 19.4 | - | same README |
| ESPnet Transformer (CTC 0.3 + attention), beam 20 | n/r | hybrid | n/r | n/r | 8.4 / 20.5 | - | same README |
| ESPnet E-Branchformer (CTC + attention) | 38.47M | hybrid | BPE-5000 | n/r | 6.3 / 17.0 | - | same README |
| ESPnet E-Branchformer transducer | 30.01M | RNN-T, beam 10 | n/r | n/r | 6.8 / 18.0 | - | same README |
| ESPnet Conformer transducer | 30.53M | RNN-T, beam 10 | n/r | n/r | 6.9 / 18.1 | - | same README |
| ESPnet Multiconvformer | 37.21M | hybrid | BPE-5000 | n/r | - | 6.2 / 17.0 (LM) | same README |
| icefall pruned_transducer_stateless2 (`--full-libri 0`) | ~80M (inferred from the recipe family) | RNN-T greedy / fast beam | BPE-500 (UNVERIFIED) | 40 | 6.64 / 17.19 (greedy); 6.53 / 16.82 (fast beam) | - | icefall RESULTS.md, "Training on train-clean-100" |
| icefall pruned_transducer_stateless4 (100h) | n/r | RNN-T | n/r | 30 | 6.92 / 18.65 (greedy, averaged) | - | same RESULTS.md |
| LAS-6-1280 + SpecAugment (supervised baseline) | n/r | attention | n/r | n/r | 5.5 / 16.9 | - | arXiv 2005.09629 T1 |
| Hybrid DNN/HMM (Lüscher et al.) | n/r | hybrid | - | - | - | 5.8 / 18.6 (4-gram) | arXiv 2006.11477 T1 |
| SpeechBrain Conformer small / HyperConformer small (100h) | ~8.7M / ~7.9M | CTC+attn, beam, Transformer LM | n/r | 110 | - | 8.29 / 6.76 (split ambiguous, see notes) | arXiv 2305.18281 T2 |

n/r = not reported.

## Notes, discrepancies and UNVERIFIED items

- **The LARM numbers check out.** The brief's numbers (7.6M at 26.78/52.82 and 7.7M at 11.34/31.84) match Table 1 exactly. The setup:
  - Character CTC with 30 symbols.
  - 80-dim fbank features.
  - Light SpecAugment: one frequency mask with F=15, two time masks at ratio 0.02.
  - AdamW, peak LR 7e-4, cosine schedule, 1k warmup steps, effective batch 32, 50 epochs.
  - LM decoding uses a 4-gram KenLM with beam 100 and LM weight 0.5.
  - The paper does not state the hardware.
- **Where a 3.5M CTC model would land.** No 100h-only result under 7M parameters exists in a primary source. On 100h, the bar is roughly 27/53 (plain 4-layer, greedy) to 11/32 (looped 7.7M, greedy). Well-tuned ~26-32M CTC encoders reach about 9-11 / 23-25.
- **Strongest 100h numbers.** Hybrid CTC+attention and transducer models (6.3-6.9 / 17-18) are not comparable to greedy pure CTC.
- **HyperConformer Table 2 is ambiguous.** The caption says "relative WER reduction on test-other with LM", but 6.76 on 100h would beat the 960h test-other. A SpeechBrain user who reproduced it on train-clean-100 got 7.06 tc / 16.74 to, which suggests the column may actually be test-clean. Treat it as UNVERIFIED (github.com/speechbrain/speechbrain/discussions/2955).
- **Discarded search claim.** A search snippet credited the icefall conformer_ctc recipe with 3.26 / 8.21 on train-clean-100. That number comes from a 109M model, and the doc page does not say 100h. I excluded it.
- **Small Zipformer results in icefall.** I found no train-clean-100 Zipformer results in icefall's RESULTS.md (both halves of the 169K-character file were scanned by the fetch tool). The only 100h results are pruned_transducer_stateless2 and stateless4. The 6.1M streaming Zipformer uses GigaSpeech, so it is not LibriSpeech-only.
- **Whisper tiny.en numbers differ by source.** The Whisper paper reports 5.6 / 14.6, while Moonshine's re-evaluation reports 5.66 / 15.45. Parameter counts also differ: 39M in the Whisper paper versus 37.8M in the Moonshine paper.
- **Moonshine latency figures.** The Moonshine v2 paper reports "compute load %", which it describes as the inverse of RTF, measured on an Apple M3 during live transcription. v1 Moonshine Tiny has lower latency (27 ms) than v2 Tiny (50 ms) but is not streaming. The v1 paper gives no absolute latency in text, only Figure 5 (H100) plus claims of "5x less compute" and "up to 3x lower latency".
- **Canary-180M-flash test-other conflict.** The model card's table says 3.83, but its metadata says 2.87.
- **Citrinet details.**
  - Parameter counts differ between the paper's tables: 9.8M in Table 3 versus 10.2M in Table 9.
  - The paper does not say "960h" explicitly.
  - The NGC stt_en_citrinet_256 card (4.4 / 10.7) is a different model, trained on more than 7K hours of mixed data.
- **QuartzNet 5x5** has dev numbers only, with no test-set results.
- **NVIDIA FastConformer.** I found no public FastConformer-CTC "small" English checkpoint. The smallest public Parakeet is the 110M TDT-CTC. **UNVERIFIED** that no smaller one exists.
- **UNVERIFIED items:**
  - Kyutai LibriSpeech WER.
  - Open ASR leaderboard per-model LibriSpeech values. I did not fetch the leaderboard; only numbers quoted on model cards are used.
  - Picovoice and Liquid tiny STT.
  - Any 2026 sub-10M English release with LibriSpeech WER. The searches surfaced none in a primary source. "Flavors of Moonshine" (arXiv 2509.02523) covers non-English tiny models.
  - The 2026 ESP32-P4 transducer (arXiv 2610.38739, 65.7M params, about 4.6-4.8 tc) comes from a search snippet only.
- **Secondary 100h sources I did not use (UNVERIFIED):**
  - HF voidash/tinyconformer-ctc-ls100: 978K params, 68.45 / 82.50 greedy. The page returned 401.
  - github.com/annapo28/asr: a Conformer-CTC on 100h, 19.68 / 42.67, with unknown parameter count.
  - The Multi-Convformer paper (arXiv 2407.03718) reportedly has an E-Branchformer CTC on 100h at 10.05 / 23.87. Its PDF was unreadable.
