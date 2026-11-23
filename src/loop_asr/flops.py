"""Forward FLOPs per arm, counted on CPU.

`torch.utils.flop_counter` has no formula for the MPS attention op, so on MPS attention
counts as 0 (docs/research/practical.md section 6). Counting always happens on a CPU copy.
Elementwise ops, norms and softmax count 0 and attention ignores the padding mask, so these
are matmul/conv FLOPs (2 per multiply-add) for one unpadded utterance.
"""

from __future__ import annotations

import copy

import torch
from torch.utils.flop_counter import FlopCounterMode

from loop_asr.model import CTCModel

FRAMES_PER_SECOND = 100  # 10 ms fbank hop


def forward_flops(
    model: CTCModel,
    seconds: float = 15.0,
    loops: int | None = None,
    depth: int | None = None,
) -> int:
    """Forward FLOPs for one utterance of `seconds` of audio, on a CPU copy of `model`."""
    cpu = copy.deepcopy(model).to("cpu").eval()
    n = int(seconds * FRAMES_PER_SECOND)
    feats = torch.zeros(1, n, cpu.cfg.n_mels)
    lengths = torch.tensor([n])
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        cpu(feats, lengths, loops=loops, depth=depth)
    return counter.get_total_flops()
