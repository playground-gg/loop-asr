# experiments

One folder per run: `experiments/<id>/README.md` (ids like `p0-calibration`, `p1-arm-b`).

## Rule: predictions before runs

The README is written and **committed before the run starts**, with:

1. **Goal** - the hypothesis it tests (H1-H5 in `docs/plan.md` section 1).
2. **Predictions** - the number you expect (e.g. "dev-clean WER 18-24"), and what would count
   as wrong. Corrections are allowed only before the run; after it, the prediction stays as it
   was and the outcome is written next to it.
3. **Setup** - exact command, config path (and its git commit), hardware, torch version,
   dtype, device, seed, steps.

After the run, add:

4. **Outcome** - the measured numbers, wall-clock time, where the results live in `results/`.
5. **What this shows / does not show** - two lines.

Failed and negative runs keep their folder. Controls are matched (same steps, LR peak,
schedule, seed, batch order, SpecAugment) unless the README says which one differs and why.
The pilot opens dev splits only; test-* is opened once, after a go (`docs/DECISIONS.md`).
