status: project playground-gg/loop-asr

# Brief: a tiny looped STT model you can run at any depth

v2, 2026-11-23. Rewritten on top of the frontier research (`research.md`, `research/`). The
first version of this brief, in Italian, is in git history (commit fe708ca). Changes to this
file go by PR, with an entry in `DECISIONS.md`.

**Problem.** Can a speech-to-text model of a few million parameters use its compute
better by looping its middle blocks? The basic claim is already published. Komatsu 2022
(arXiv 2202.08474) ran a fully shared 6.8M conformer-CTC on 100h, with the repeat count set at
decoding: 9.5 / 25.6. LARM 2026 (arXiv 2606.04678) showed a looped 7.7M Transformer beating a
plain model with the same number of block applications on 100h (11.34 vs 12.24), though not on 960h. What is still
open is narrower and practical (`idea.md`, `research.md` §2):

**Claim to test.** One tiny looped model (~3.5M), trained once with the loop count K sampled
(Inkling 2 Mini's 1:2:6:2:1), is usable at any k at inference. It must:
1. clearly beat a plain model of the same size,
2. hold up against a fixed-K looped model, the Komatsu/LARM-style baseline,
3. hold up against a plain model with early exits at equal compute, measured as real CPU
   wall-clock on a Mac and not only as FLOPs.

**User.** The owner: an honest, reproducible experiment on the owner's Mac, in the style of
wsp-adaptive-decision-models (predictions written before every run, matched controls).

**First version (the pilot).**
- English, LibriSpeech train-clean-100, Mac only (Apple M5, 24 GB, MPS), dev sets only.
- Arms: A (4 plain blocks, ~3.57M), B (looped, prelude 1 + shared core 2 × k + coda 1, one
  closed gate per core layer, K sampled from {2,4,6,8,10}, ~3.61M), B_fix (B at fixed K=6),
  C (14 plain blocks, ~11.5M, CTC exits at 6/10/14, the same FLOPs as B at k=2/4/6).
- The same 3-LR sweep for A, B and C, then each winner rerun at full steps.
- Measures: dev-clean/dev-other WER at every k or exit, CPU and MPS wall-clock, FLOPs counted
  on CPU, size. Whisper tiny.en and Moonshine Tiny are measured on the same Mac as references.
- About 25 h of Mac time, with one go criterion fixed in advance (`plan.md` §7).

**Target output.** A blog post plus an HF release: an int8 model under 5 MB, a k-slider, and
an honest Pareto plot against Whisper tiny.en and Moonshine Tiny. A workshop paper is stage 2,
only after a go (3 seeds, C with exits, per-k specialists).

**Out.** Italian, cloud GPUs, TTS, streaming, an app. test-clean/test-other and int8 come only
after a go. No distillation in the pilot.

**Done when.** The pilot has run, and a report gives go or no-go against `plan.md` §7, with the
numbers of every arm and the predictions they were written against.

**Stack.** Python 3.12, PyTorch 2.15 (MPS, fp32, eager), soundfile, torchaudio `kaldi.fbank`,
sentencepiece (BPE 256), jiwer. References: openai-whisper (needs ffmpeg) and moonshine-voice.
