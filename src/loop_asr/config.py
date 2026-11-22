"""YAML config loader for the three arms (A plain, B looped, C plain at matched FLOPs)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ModelConfig:
    kind: Literal["plain", "looped"] = "plain"
    n_mels: int = 80
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    conv_channels: int = 64
    vocab_size: int = 256  # BPE pieces; CTC blank is an extra index (vocab_size)
    # plain encoder
    n_blocks: int = 4
    # looped encoder: prelude + shared core repeated K times + coda
    n_prelude: int = 1
    n_core: int = 2
    n_coda: int = 1
    gate_rank: int = 32
    default_loops: int = 10  # K_max, docs/DECISIONS.md 2026-11-23 (a)


@dataclass
class Config:
    name: str
    arm: str
    model: ModelConfig
    target_params: float | None = None
    train: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _model_config(raw: dict[str, Any]) -> ModelConfig:
    known = {f.name for f in fields(ModelConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown model config keys: {sorted(unknown)}")
    return ModelConfig(**raw)


def _number(value: Any) -> float | None:
    # PyYAML (YAML 1.1) reads `3.5e6` as a string; accept it as a number.
    return None if value is None else float(value)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return Config(
        name=raw["name"],
        arm=raw["arm"],
        model=_model_config(raw.get("model", {})),
        target_params=_number(raw.get("target_params")),
        train=raw.get("train", {}) or {},
        eval=raw.get("eval", {}) or {},
        notes=raw.get("notes", "") or "",
    )
