# loop-asr

Can a speech-to-text model of a few million parameters get close to a much larger one by
looping its middle blocks? Inkling 2 Mini (Thinking Machines, 2026-11-20) repeats its central
blocks with shared weights, behind a gate that starts closed; LARM (arXiv 2606.04678) reports
that on ASR a 7.7M looped encoder gets close to an 85.7M one at the same FLOPs. This repo tests
the idea on one Mac (Apple M5, 24 GB, MPS), on LibriSpeech train-clean-100, English only.

Status: scaffold only. No training run yet, no data downloaded.

## The question

Does a tiny CTC encoder with Inkling-style looped shared blocks beat a plain encoder **at the
same size**, hold up against one **at the same FLOPs**, and give a useful speed/accuracy knob
when the loop count is chosen at inference?

## Three arms

Pre-norm Transformer encoder (4x conv subsampling, d=256, 4 heads, FFN 1024, RoPE), CTC head,
BPE 256 with sentencepiece. Same steps, LR peak, schedule, seed, batch order and SpecAugment
across arms.

| arm | structure | params [E] | blocks run per frame | role | config |
|---|---|---|---|---|---|
| A | 4 plain blocks | ~3.5M | 4 | matched-parameter control | `configs/arm_a.yaml` |
| A_best | A at the best LR of its sweep, full steps | ~3.5M | 4 | the control H1 is judged against | `configs/arm_a_best.yaml` |
| B | prelude 1 + shared core 2 x K + coda 1; low-rank message gate initialised closed; K sampled from {2,4,6,8,10} in training | ~3.5M | 2 + 2K (18 at K=8) | test arm | `configs/arm_b.yaml` |
| C | 18 plain blocks | ~14.5M | 18 | matched-FLOP control for B at K=8 | `configs/arm_c.yaml` |

## Metrics

- WER (greedy CTC, `jiwer`, same text normalisation everywhere): **dev-clean / dev-other in the
  pilot**; test-clean / test-other once, only after a go.
- WER of B at each loop count k = 1, 2, 4, 6, 8, 10.
- Parameters, size on disk (fp32, fp16; int8 only after a go).
- RTF on CPU (1 thread and all threads) and MPS, batch 1, 200 dev-clean utterances; latency
  p50/p95 on a 10 s clip; analytic FLOPs per second of audio.

## Go criterion (pilot, dev-clean greedy; docs/plan.md section 7)

1. Sanity: B < 30% WER and A < 45%. Failing this is a bug to fix, not a no-go.
2. H1: B <= 0.8 x WER(A_best).
3. H2: B <= 1.05 x WER(C).
4. Speed: some k < K where B(k) has WER <= WER(C) + 5% relative **and** is at least 1.5x
   faster than C on CPU, wall clock.

**Go** if 2, 3 and 4 hold. 2 and 3 without 4 means "smaller, not faster": funded only if size
is the target. Anything else is a no-go.

## Out of scope

Italian, cloud GPUs, TTS, streaming, an app. For now also: Whisper tiny.en / Moonshine Tiny
reference runs and the optional distillation arm.

## Working style

As in `wsp-adaptive-decision-models`:
- **Predictions before runs.** Every run gets `experiments/<id>/README.md` with its predictions
  written and committed before it starts (see `experiments/README.md`).
- **Matched controls.** Arms differ only in the encoder; the baseline gets its own LR sweep so a
  win is not a badly tuned control.
- **Honest reports.** Failed runs and negative results stay in the summaries; differences under
  2x the seed spread are reported as "no difference"; no tuning to chase a paper's number.

## Layout

- `docs/` the brief, the experiment plan, the Inkling 2 Mini study and the decision log.
- `configs/` one YAML per arm.
- `src/loop_asr/` config loader, plain and looped encoders, CTC model, device helper.
- `tests/` CPU smoke tests (seconds).
- `experiments/` one folder per run, predictions first.
- `results/` compact summaries only; no weights, no data, no large logs.

## Setup

```bash
uv sync                      # Python 3.12, torch (MPS on macOS, CPU wheel on Linux)
uv run ruff check . && uv run ruff format --check . && uv run pytest
```
