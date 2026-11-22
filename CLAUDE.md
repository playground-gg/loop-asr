# loop-asr

Research repo: plain vs Inkling-style looped shared-block CTC encoders for tiny English STT,
on one Mac (24 GB, MPS). Read `README.md`, `docs/DECISIONS.md`, then `docs/plan.md`.

## Rules
- Runs only on the local Mac; never set up cloud GPUs. Device only through `loop_asr.get_device()`.
- Ask the owner before any run expected over ~2 h, any download over ~20 GB, or opening
  test-clean / test-other (only after a go).
- Predictions go in `experiments/<id>/README.md`, committed before the run.
- Matched controls: arms differ only in the encoder.
- `docs/brief.md`, `docs/plan.md`, `docs/research.md` are the owner's source docs: do not edit;
  record changes in `docs/DECISIONS.md`.
- Never commit data, features, checkpoints or large logs.
- Gate: `uv run ruff check . && uv run ruff format --check . && uv run pytest`.

# Compact instructions

Keep: the goal, constraints and decisions stated exactly (docs/DECISIONS.md items, go
criterion numbers), the owner's preferences, in-flight work and branches, open items, failing
checks, exact names, paths and numbers. Drop tool output already acted on.
