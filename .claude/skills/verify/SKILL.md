---
name: verify
description: How to run and exercise loop-asr end to end to verify a change
---

# Verify a loop-asr change

**Setup (fresh worktree):** `uv sync` (Python 3.12 from uv; system python3 has no torch).
uv is at `~/.local/bin/uv` if not on PATH. Nothing else to copy: no env file, no data yet.

**Gate:** `uv run ruff check . && uv run ruff format --check . && uv run pytest` (about 5 s on CPU).
CI runs the same on ubuntu with the CPU-only torch wheel.

**Exercise it the way a user does** (there is no training loop or CLI yet):
- Build each arm from its config and print parameter counts against `target_params`:
  `uv run python -c "from loop_asr import *; [print(a, count_params(build_model(load_config(f'configs/arm_{a}.yaml').model))) for a in ('a','a_best','b','c')]"`
- Forward the looped model at several loop counts on random fbank-shaped input
  `(B, T, 80)` and check finite CTC loss; at init (gate closed) every K gives the same output.
- On the Mac, `get_device()` returns `mps`; force CPU with `LOOP_ASR_DEVICE=cpu`.

**Proof:** the gate's pass line plus the printed parameter counts, saved under
`.kit/evidence/<branch>/`.

**Traps:**
- PyYAML reads `3.5e6` as a string; `load_config` coerces `target_params`, but numeric
  fields in `train:` are raw YAML (write `7.0e-4`, with the dot).
- Never run training, download LibriSpeech, or open test-clean/test-other without the owner's
  go (docs/DECISIONS.md). Runs over ~2 h or downloads over ~20 GB: ask first.
- MPS `ctc_loss` had an inf-gradient bug for targets >= 512 labels (plan.md section 8); BPE 256 avoids it.
