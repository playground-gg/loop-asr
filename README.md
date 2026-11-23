# loop-asr

One tiny looped speech-to-text model, trained once, that you can run at any depth. A ~3.6M
CTC encoder whose two middle blocks share weights and repeat k times (Inkling 2 Mini's layout:
restart from the prelude, a gated message between passes, a gate that starts closed). It is
trained with k sampled each step, then tested on one Mac (Apple M5, 24 GB, MPS) on
LibriSpeech train-clean-100, English only.

The generic claim, that a looped encoder beats a plain one on 100h, is already published
(Komatsu 2022, arXiv 2202.08474; LARM 2026, arXiv 2606.04678). This repo tests what is still
open: one sampled-K model against a fixed-K one, and against a plain model with early exits at
equal FLOPs, timed as real CPU wall-clock. Start with [docs/idea.md](docs/idea.md).

Status: plan v2 written, no training run yet, no data downloaded.

## Arms

Pre-norm Transformer encoder (4x conv subsampling, d=256, 4 heads, FFN 1024, RoPE), CTC head,
BPE 256 with sentencepiece. Same steps, schedule, seed, batch order and SpecAugment across
arms. Each of A, B and C gets the same 3-LR sweep at half steps; its winner is rerun at full
steps (`*_best.yaml`).

| arm | structure | params | blocks run per frame | role | config |
|---|---|---|---|---|---|
| A | 4 plain blocks | 3.57M | 4 | matched-parameter control | `arm_a.yaml`, `arm_a_best.yaml` |
| B | prelude 1 + shared core 2 x k + coda 1; one closed low-rank gate per core layer; K sampled from {2,4,6,8,10} at 1:2:6:2:1 | 3.61M (+0.95%) | 2 + 2k (mean 14) | test arm | `arm_b.yaml`, `arm_b_best.yaml` |
| B_fix | B's encoder trained at fixed K=6 | 3.61M | 14 | Komatsu/LARM-style fixed-K baseline | `arm_b_fix.yaml` |
| C | 14 plain blocks, CTC exits at 6/10/14 | 11.47M | 14 (exits: 6, 10) | matched-FLOP control with early exits (= B at k=2/4/6) | `arm_c.yaml`, `arm_c_best.yaml` |

## Metrics

- WER (greedy CTC, `jiwer`, same text normalisation everywhere): **dev-clean / dev-other in the
  pilot**; test-clean / test-other once, only after a go.
- B and B_fix at k = 1, 2, 4, 6, 8, 10, 12, 14, 16; C at exits 6, 10, 14.
- CPU latency (4 threads, batch 1, median over 200 fixed dev-clean utterances), also MPS;
  FLOPs counted on CPU (`loop_asr.forward_flops`); parameters and size.
- Whisper tiny.en and Moonshine Tiny measured on the same Mac, for scale.

## Go criterion (pilot, docs/plan.md section 7)

**GO** if all three hold:
1. **Size (primary):** WER(B(6)) <= 0.80 x WER(A_best) on dev-clean and <= 0.85 x on dev-other.
2. **Any k:** on dev-clean, WER(B(4)) <= 0.85 x WER(A_best) and <= 1.15 x WER(B(6)).
3. **Equal compute:** for k = 4 and k = 6, on dev-clean, WER(B(k)) <= 1.10 x WER(C@(2+2k)),
   and B(k)'s measured CPU latency <= 1.10 x C@(2+2k)'s.

Prior P(go) about 0.25. The pilot costs about 25 h of Mac time (docs/plan.md section 6).

## Out of scope

Italian, cloud GPUs, TTS, streaming, an app. For now also the distillation arm (a separate
"product" variant, after a go).

## Working style

As in `wsp-adaptive-decision-models`:
- **Predictions before runs.** Every run gets `experiments/<id>/README.md` with its predictions
  written and committed before it starts (see `experiments/README.md`).
- **Matched controls.** Arms differ only in the encoder (C's exit loss is the one stated
  exception); every arm gets the same LR sweep so a win is not a badly tuned control.
- **Honest reports.** Failed runs and negative results stay in the summaries; differences under
  2x the seed spread are reported as "no difference"; no tuning to chase a paper's number.

## Layout

- `docs/` the idea, brief, experiment plan, research summary (+ full notes in `docs/research/`) and the decision log.
- `configs/` one YAML per arm.
- `src/loop_asr/` config loader, plain (with early exits) and looped encoders, CTC model, CPU FLOP counter, device helper.
- `tests/` CPU smoke tests (seconds).
- `experiments/` one folder per run, predictions first.
- `results/` compact summaries only; no weights, no data, no large logs.

## Setup

```bash
uv sync                      # Python 3.12, torch (MPS on macOS, CPU wheel on Linux)
uv run ruff check . && uv run ruff format --check . && uv run pytest
```
