"""Device selection: MPS on Apple Silicon when available, else CPU."""

from __future__ import annotations

import os

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the device to run on.

    `prefer` (or the env var LOOP_ASR_DEVICE) forces "cpu" or "mps"; asking for "mps"
    where it is unavailable falls back to CPU rather than failing.
    """
    choice = (prefer or os.environ.get("LOOP_ASR_DEVICE", "")).lower()
    if choice == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
