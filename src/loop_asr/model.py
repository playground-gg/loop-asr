"""Minimal CTC encoders for the three arms of docs/plan.md section 3.

- PlainEncoder: N pre-norm Transformer blocks (arms A and C).
- LoopedEncoder: prelude + a core of shared blocks run K times + coda (arm B). Every pass
  restarts from the prelude output; the previous pass's core output enters through a
  low-rank gate whose output projection starts at zero, so at init every pass computes the
  same thing and the model equals the single-pass (no-loop) path for any K.

This is a stub to fix shapes, parameter counts and the gate's initial behaviour; no
training code lives here yet.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from loop_asr.config import ModelConfig


def rope(x: Tensor) -> Tensor:
    """Rotary position embedding over the last dim of x: (B, H, T, Dh)."""
    _, _, t, dh = x.shape
    half = dh // 2
    freqs = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    angles = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    """Pre-norm Transformer block: RoPE self-attention + GELU FFN."""

    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        attn_mask = None if mask is None else mask[:, None, None, :]  # True = attend
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.dropout(self.proj(a.transpose(1, 2).reshape(b, t, d)))
        return x + self.dropout(self.ff(self.norm2(x)))


class Subsampling(nn.Module):
    """Two stride-2 conv2d layers: 4x time reduction (100 -> 25 frames/s)."""

    def __init__(self, n_mels: int, d: int, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2),
            nn.ReLU(),
        )
        f = ((n_mels - 1) // 2 - 1) // 2
        self.out = nn.Linear(channels * f, d)

    @staticmethod
    def out_lengths(lengths: Tensor) -> Tensor:
        return ((lengths - 1) // 2 - 1) // 2

    def forward(self, feats: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(feats.unsqueeze(1))  # (B, C, T', F')
        b, c, t, f = x.shape
        x = self.out(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        return x, self.out_lengths(lengths)


def _blocks(n: int, cfg: ModelConfig) -> nn.ModuleList:
    return nn.ModuleList(Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(n))


class PlainEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.blocks = _blocks(cfg.n_blocks, cfg)

    def forward(self, x: Tensor, mask: Tensor | None, loops: int | None = None) -> Tensor:
        del loops  # plain encoders have no loop knob; accepted for a uniform interface
        for blk in self.blocks:
            x = blk(x, mask)
        return x


class LoopGate(nn.Module):
    """Low-rank message from the previous pass; the up-projection starts at zero (closed)."""

    def __init__(self, d: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, message: Tensor) -> Tensor:
        return self.up(F.silu(self.down(self.norm(message))))


class LoopedEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.prelude = _blocks(cfg.n_prelude, cfg)
        self.core = _blocks(cfg.n_core, cfg)
        self.coda = _blocks(cfg.n_coda, cfg)
        self.gate = LoopGate(cfg.d_model, cfg.gate_rank)
        self.default_loops = cfg.default_loops

    def forward(self, x: Tensor, mask: Tensor | None, loops: int | None = None) -> Tensor:
        k = self.default_loops if loops is None else loops
        if k < 1:
            raise ValueError(f"loops must be >= 1, got {k}")
        for blk in self.prelude:
            x = blk(x, mask)
        h0 = x
        message: Tensor | None = None
        for _ in range(k):
            h = h0 if message is None else h0 + self.gate(message)
            for blk in self.core:
                h = blk(h, mask)
            message = h
        x = message
        for blk in self.coda:
            x = blk(x, mask)
        return x


class CTCModel(nn.Module):
    """Subsampling -> encoder -> LayerNorm -> linear CTC head (blank = last index)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.subsample = Subsampling(cfg.n_mels, cfg.d_model, cfg.conv_channels)
        self.encoder: PlainEncoder | LoopedEncoder
        if cfg.kind == "plain":
            self.encoder = PlainEncoder(cfg)
        elif cfg.kind == "looped":
            self.encoder = LoopedEncoder(cfg)
        else:
            raise ValueError(f"unknown model kind {cfg.kind!r}")
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size + 1)

    @property
    def blank(self) -> int:
        return self.cfg.vocab_size

    def forward(
        self, feats: Tensor, lengths: Tensor, loops: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """feats (B, T, n_mels), lengths (B,) -> log-probs (B, T', V+1), out lengths (B,)."""
        x, out_lengths = self.subsample(feats, lengths)
        mask = torch.arange(x.shape[1], device=x.device)[None] < out_lengths[:, None]
        x = self.encoder(x, mask, loops=loops)
        return self.head(self.norm(x)).log_softmax(-1), out_lengths

    def ctc_loss(
        self,
        log_probs: Tensor,
        out_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        return F.ctc_loss(
            log_probs.transpose(0, 1),
            targets,
            out_lengths,
            target_lengths,
            blank=self.blank,
            zero_infinity=False,
        )


def build_model(cfg: ModelConfig) -> CTCModel:
    return CTCModel(cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
