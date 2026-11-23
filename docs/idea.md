# The idea

**One tiny looped speech model, trained once, that you can run at any depth.**

A ~3.5M-parameter CTC encoder whose two middle blocks share weights and repeat k times
(Inkling 2 Mini's layout: restart from the prelude, a gated message between passes, a gate that
starts closed). It is trained with k sampled every step, so a single checkpoint serves every k.
On a phone or a laptop, k becomes a dial: few loops when the battery or the latency budget is
tight, more loops when accuracy matters. You ship one set of weights, under 5 MB in int8.

## What we expect to learn

1. **Does the loop buy accuracy at a fixed size?** We compare B with a plain 4-block model with
   the same parameters (A). LARM and Komatsu say yes at 7–8M. We test whether that holds at 3.5M
   with an Inkling-style gate. Prediction: B(6) about 35% better than the tuned A.
2. **Is one sampled-K model as good as models trained for one depth?** We compare B with the
   same encoder trained at a fixed K=6 (B_fix, the Komatsu/LARM-style baseline). Prediction:
   B is much better than B_fix away from k=6, and at most 5% worse at k=6.
3. **Is the loop a better speed dial than early exits?** We compare B with a plain model 3.2×
   larger (C, 14 blocks) that exits at the same FLOPs, and measure the time on the Mac's CPU,
   not only the FLOPs. Prediction: B is roughly tied with C at the same compute. Whether B ties
   or loses to C is the question that matters most.

## Why it matters for tiny on-device STT

- The open small models (Whisper tiny.en 39M, Moonshine Tiny 27M) are 8–10× larger than this
  one. No open English model under 10M parameters publishes LibriSpeech numbers.
- On-device, the scarce resources are flash, RAM and joules, and they shift at runtime. A model
  that trades accuracy for compute at inference, without a second checkpoint, fits that better
  than a ladder of fixed models.
- What looped-ASR papers report is FLOPs. What a device spends is wall-clock time. At 3.5M and
  batch 1, op overhead and cache can break the link between the two, and nobody has measured
  it for looped encoders.

## What a no-go teaches

- **B does not beat the size-matched A:** looping does not pay at 3.5M, which contradicts
  LARM and Komatsu. First check that the gate opened and look for bugs, then report it. Either
  way, it marks where the size floor for looping sits.
- **B beats A but loses to C's exits at equal compute:** loops buy parameters, not compute.
  That is still useful where flash or RAM is the binding limit: report size against WER and
  stop. It is also a negative result nobody has published with wall-clock numbers.
- **B ties C but is not better than B_fix:** sampling K adds nothing over Komatsu-style
  folding. The "train once, any k" pitch is then not new, only convenient.

The target output is an honest blog post and an HF release (int8 model, k-slider, a Pareto
plot against Whisper tiny.en and Moonshine Tiny on the same Mac). A workshop paper is stage 2,
and only after a go (plan.md §8).
