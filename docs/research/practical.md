# Practical facts: small CTC on LS-100, Apple M5 + PyTorch MPS (researched 2026-11-23)

Tags: [V] I checked this against a primary source. [U] Unverified or from memory. [E] My own estimate.
Local machine: Apple M5 (4P+6E CPU), 24 GB RAM, macOS 27.0.1. Torch and torchaudio are **not installed**, so I ran nothing locally.

## 1. PyTorch MPS
- **Latest stable is 2.15.0, released 28 Oct 2026 [V]** (https://github.com/pytorch/pytorch/releases). Before it: 2.14.1 (30 Sep), 2.14.0 (2 Sep), 2.13.0 (8 Jul). **2.15 drops macOS 14: MPS now needs macOS 15+ [V]**.
- **CTC on MPS:**
  - Forward was added in PR #187716, merged 26 Jun 2026 [V]. Backward was added in PR #188187, merged 29 Jun 2026 with `-f`, which bypassed CI [V].
  - The 2.14.0 notes say: "Add MPS forward and backward support for `torch.nn.functional.ctc_loss`" [V]. **Native MPS CTC therefore needs torch ≥2.14.**
- **Issue #203356** ("ctc_loss backward returns wrong (inf) gradients when any target has >=512 labels"):
  - Closed 7 Nov 2026 by PR #203360, commit b6472a3 [V].
  - That is after 2.15.0 shipped, the PR has no milestone, and I found no 2.15.1 or cherry-pick. **So no stable release contains the fix [V/inferred]**. It will most likely land in 2.16 [U].
  - Impact for this project: BPE-256 targets on LS-100 run about 5–6 tokens/s [E]. Even 35 s utterances stay at roughly 200 tokens or fewer, well under 512, so **the bug should not trigger here**. A cheap safeguard: assert `max(target_len) < 512`, or check on CPU.
  - Reviewers flagged that multi-block kernels rely on barrier ordering. Gradcheck MPS against CPU on a few batches [suggestion].
- **SDPA:**
  - On MPS, SDPA dispatches to a separate op, `_scaled_dot_product_attention_math_for_mps` [V, native_functions.yaml].
  - 2.15 fixed MPS SDPA output shape with broadcast batch/head dims [V].
  - FlexAttention on MPS arrived in 2.13 [V].
- **Autocast / bf16 / fp16:**
  - The amp docs (2.15) list no MPS device type or MPS op lists [V] (https://docs.pytorch.org/docs/2.15/amp.html).
  - MPS autocast exists in recent releases (fp16 since about 2.5, bf16 later) [U]. Check with `torch.amp.autocast_mode.is_autocast_available("mps")`.
  - An Apple forum thread (Aug 2026) reports bf16 softmax NaN on M5 Max: https://developer.apple.com/forums/thread/841213. Apple DTS traced it to fully masked rows (−inf − −inf) and to fp16 overflow of a `-1e9` mask constant.
  - **Practical advice:** use a finite `torch.finfo(dtype).min` mask, or a bool mask with no fully masked rows. **Train fp32 as the baseline.**
- **Determinism:**
  - 2.15 now raises under `torch.use_deterministic_algorithms(True)` for nondeterministic MPS ops [V].
  - There is no general "deterministic MPS" guarantee. Use `torch.mps.manual_seed` and expect run-to-run float noise [U].
- **float64 is unsupported on MPS** (raises TypeError) [V] (https://docs.pytorch.org/docs/2.15/notes/mps.html). `PYTORCH_ENABLE_MPS_FALLBACK=1` sends unsupported ops to CPU.
- **torch.compile on MPS:** Inductor-MPS is under active work. 2.15 fixes wrong argmax/argmin, channels_last conv backward strides, and int64 constants truncated to 32 bits [V]. One forum thread reports NaN loss with compile on MPS and CPU: https://discuss.pytorch.org/t/224290. **Treat compile as optional and validate it against eager** [U on "prototype" label].
- **LayerNorm numerics issues on MPS:** none found in 2.14/2.15 notes [not found].
- **MPS vs CUDA divergence for speech models:** no specific report found [not found]. Old forum threads report NaN/non-convergence on early MPS, from 2022–23.
- **M5 hardware:**
  - Apple's specs: 10-core GPU with a Neural Accelerator per core, "over 4x the peak GPU compute performance compared to M4" for AI, 153 GB/s bandwidth, 16-core Neural Engine [V] (https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/).
  - Third-party theoretical shader numbers: FP32 ~5.1 TFLOPS, FP16 ~10.3 TFLOPS [U] (cpu-monkey).
  - A MacRumors user measured ~1.5 TFLOPS fp16 matmul per core via the Neural Accelerators, so ~15 TFLOPS for 10 cores [U].
  - The accelerators are reached through Metal 4 tensor APIs / Metal Performance Primitives. **I found no evidence that PyTorch MPS matmul uses them [U]**.
  - I found no credible M4/M5 PyTorch-MPS training throughput numbers for small transformers. Benchmark it yourself.
- **MLX:** `mlx.nn.losses` has **no CTC loss** [V] (https://ml-explore.github.io/mlx/build/html/python/nn/losses.html). There is a third-party alpha package, `mlx-ctc` (https://github.com/djphoenix/mlx-ctc).

## 2. torchaudio
- **Latest is 2.11.1 (28 Oct 2026)**. It declares `torch>=2.11`, so it no longer pins to each torch release [V] (https://github.com/pytorch/audio/releases).
- **What was removed:** 2.9.0 removed most APIs marked "drop". `load`/`save` still exist but now run on TorchCodec [V].
- **What stays (issue #3902, https://github.com/pytorch/audio/issues/3902):**
  - The `transforms`, `functional`, `compliance.kaldi`, `models` and `pipelines` modules. That keeps `compliance.kaldi.fbank`, `transforms.MelSpectrogram` and `models.Conformer` [V at module level].
  - The C++ extensions `lfilter`, `RNNTLoss`, `CUCTC`, `forced_align` and `overdrive` (decision in 2.10) [V].
- **`models.decoder.ctc_decoder`** (flashlight) is still in the stable docs, but its status after 2.9 is unclear [U]. Alternatives: `pyctcdecode` + KenLM [U].
- **Alternatives for features:**
  - `kaldi-native-fbank` (k2/sherpa, matches Kaldi fbank)
  - lhotse (`Fbank`, which also handles LibriSpeech prep and SpecAugment)
  - librosa (mel only, not Kaldi-compatible)
  - **Recommendation:** precompute 80-dim fbank once on CPU [suggestion].

## 3. Recipes for LS-100
- **ESPnet egs2/librispeech_100/asr1** (https://github.com/espnet/espnet/tree/master/egs2/librispeech_100/asr1):
  - `run.sh` settings: **nbpe=5000** (not 300), `speed_perturb_factors "0.9 1.0 1.1"`, `use_lm false` [V].
  - **CTC-only E-Branchformer**, `train_asr_ctc_e_branchformer_e12_mlp1024_linear1024.yaml`:
    - 12 blocks, d=256, 4 heads, linear units 1024, cgMLP kernel 31, conv2d input, dropout 0.1, ctc_weight 1.0 [V]
    - Adam lr 2e-3, wd 1e-6, warmuplr 15k steps; batch_bins 16M (numel), accum_grad 4, **70 epochs**, AMP; keeps the 10 best by cer_ctc [V]
    - SpecAug: time warp window 5, **2 freq masks width 0–27**, **5 time masks with width ratio 0–0.05** [V]
    - Results (26.4M params, greedy CTC, no LM): **dev-clean 9.2, dev-other 22.4, test-clean 9.6, test-other 23.1** [V]
  - Hybrid CTC/attention E-Branchformer, 38.5M params: 6.3 / 17.0 test. Conformer: 6.5 / 17.3. Transformer: 8.4 / 20.5 [V].
- **icefall:** the 100h runs (`--full-libri 0`) are pruned-transducer only. `pruned_transducer_stateless2` (BPE 500, 40 epochs) reaches **6.64 / 17.19** with greedy decoding and epoch-39 avg-10 [V]. I found no 100h CTC result in RESULTS.md (https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md).
- **SpeechBrain:** I did not find a 100h CTC recipe [not found].
- **SpecAugment:**
  - Park et al. 2019 (arXiv 1904.08779) is **960h only, with no 100h experiment** [V]. Its policies [U, from memory of the paper]:
    - LB: W=80, F=27, mF=1, T=100, mT=1, p=1.0
    - LD: same but mF=2, mT=2
  - Park et al. 2019b (arXiv 1912.05533) adds adaptive masking: pM=pS=0.04, up to 20 masks [V].
  - **Measured gain at 100h** (ESPnet-style CTC/attention, arXiv 2210.08520, Table 1) [V]:

    | Setting | test-clean | test-other |
    |---|---|---|
    | No augmentation | 12.1 | 32.7 |
    | SpecAugment | 10.5 | 23.2 |

    That is about 13% relative on clean and **about 29% relative on other**.
- **Speed perturbation (0.9/1.0/1.1):** Ko et al. 2015 reported about 3–6% relative WER gain, and it triples epoch cost [U].
- **Epochs:** ESPnet CTC uses 70 and icefall transducers use 30–40 [V]. Zipformer CTC on 960h uses 100 [V, arXiv 2310.11230]. **For 100h, plan 50–100 epochs plus checkpoint averaging [E]**.
- **BPE size at 100h:**
  - I found no clean sweep of character vs 256 vs 1k vs 5k for CTC on LS-100 [not found].
  - One ESPnet-related study found BPE did not beat characters at 100h [U, via search summary].
  - Zeyer/Hori-style results at 960h favour about 500–1000 units for hybrid CTC/attention [U].
  - Small vocabularies such as 256–500 are the common choice for CTC on small data. icefall's 100h runs use 500 [V].

## 4. CTC specifics
- **Length constraint:** CTC needs T ≥ L + (number of adjacent repeated labels). Otherwise the loss is inf; use `zero_infinity=True`.
  - At 25 fps (4x subsampling, 10 ms hop), LibriSpeech speaks about 2.7 words/s [E: ~9.4M words / 960h].
  - BPE-256 uses about 1.7–2 tokens/word, so about **4.5–6 tokens/s [E]**. That gives a margin of about 4–5x, so this is not a concern.
  - Count and drop utterances where T_sub < L + repeats during data prep. Note that conv2d 4x subsampling gives T' = ((T−1)//2−1)//2.
- **LM gain:**
  - A 960h self-conditioned CTC paper (arXiv 2204.00176) shows a 4-gram beam improving test-other by about 10% relative [U-partial, from search snippet]. Its 100h plain-CTC greedy baseline is ~10.65 / 25.0 [U].
  - Expect 15–25% relative from a 4-gram on weak 100h models [E]. I found no clean greedy-vs-4-gram table at 100h.

## 5. int8 on Apple Silicon
- **`torch.ao.quantization.quantize_dynamic`:**
  - The docs mark eager-mode quantization for migration to torchao `quantize_` (https://docs.pytorch.org/docs/2.15/quantization.html) [V].
  - The stale "delete in 2.10" plan has not happened; the 2.15 docs still list it [V].
  - On arm64 macOS it works with `torch.backends.quantized.engine="qnnpack"`, quantizing only `nn.Linear`/LSTM; Conv2d stays fp32 [U, from experience]. Verify after installing torch.
- **torchao** `Int8DynamicActivationInt8WeightConfig`: I found no Apple-CPU benchmarks [not found]. Its fast paths target x86/CUDA, so CPU latency on M5 may not improve [U].
- **ONNX Runtime** dynamic quantization (`quantize_dynamic`, QInt8 weights) is recommended for transformers [V] (https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html):
  - On Arm there is no U8S8 saturation issue, and dot-product instructions help [V].
  - **This is the most reliable int8 CPU path on Mac [E]**.
- **Core ML** (coremltools) [V] (https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html):
  - Supports 8/4-bit weight-only quantization and W8A8.
  - W8A8 runs fast int8-int8 on the Neural Engine on A17 Pro/M4+.
  - Weight-only mainly shrinks size.
- **WER cost of int8:**
  - Google's 4-bit Conformer paper (arXiv 2203.15952, QAT) [V]:

    | Model | Float | int8 weights | int4 weights + int8 activations |
    |---|---|---|---|
    | Large (test-clean/other) | 2.0 / 4.4 | 2.0 / 4.5 | 2.1 / 4.4 |
    | Small | 2.5 / 6.1 | not given | 2.8 / 6.6 |

  - Post-training dynamic int8 on small CTC models typically costs ≤0.1–0.5 absolute WER [U]. Size of int8 Linear weights is about 4x smaller than fp32.

## 6. FLOPs
- **`torch.utils.flop_counter.FlopCounterMode`** (source checked on main) [V] counts:
  - mm/addmm/bmm/baddbmm/_scaled_mm
  - convolution (forward and backward)
  - SDPA variants: efficient, flash, `flash_attention_for_cpu`, cudnn, plus flash/efficient forward and backward
- **Caveats:**
  - Elementwise ops, softmax, LayerNorm and activations count as 0.
  - Attention formulas ignore masks, so they are upper bounds.
  - Counts use 2·m·n·k (MAC×2).
  - Ops with no formula add 0. **`_scaled_dot_product_attention_math_for_mps` is not registered**, so attention likely shows 0 FLOPs on MPS [V registry / inferred effect]. **Count on CPU or the meta device.**
- **Alternatives:** fvcore (`FlopCountAnalysis`, counts MACs labelled as "flops"), ptflops (module hooks, misses functional attention), DeepSpeed profiler [U].

## 7. RTF practice
- **Open ASR Leaderboard:** RTFx = total audio duration / total transcription time, so higher is better. It can be measured per utterance or per batch [V] (arXiv 2510.06961; https://huggingface.co/blog/open-asr-leaderboard).
- A 2026 on-device ASR paper reports RTFx at batch 1 and stresses comparing on the same hardware [V] (arXiv 2604.14493).
- **Moonshine** reports compute (FLOPs) per 10 s segment and latency on low-cost ARM. It argues Whisper's 30 s padding wastes compute [V] (arXiv 2410.15608, 2509.02523).
- **Recommended protocol [E]:**
  - batch 1, with a fixed test set (test-clean)
  - include or exclude feature extraction explicitly
  - CPU: `torch.set_num_threads(4)` to match the M5's 4 P-cores (also report 1 thread), plus `torch.inference_mode()`
  - 5–10 warmup runs
  - MPS: call `torch.mps.synchronize()` before stopping timers
  - report RTF = compute/audio and RTFx = 1/RTF, as median and p90 over utterances

## Key unknowns to verify locally
- MPS autocast dtype support
- whether qnnpack is in `torch.backends.quantized.supported_engines`
- MPS-vs-CPU CTC gradcheck
- FlopCounter on MPS
- actual MPS step throughput
