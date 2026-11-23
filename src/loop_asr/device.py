"""Device selection: MPS on Apple Silicon when available, else CPU.

Asking for MPS explicitly (argument or LOOP_ASR_DEVICE=mps) where it is missing raises
instead of silently falling back: a run that was meant for MPS but lands on CPU wastes hours
and corrupts any timing (plan.md step 0, DECISIONS.md decision 8). Choosing automatically on
macOS and getting CPU warns.
"""

from __future__ import annotations

import os
import sys
import warnings

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the device to run on.

    `prefer` (or the env var LOOP_ASR_DEVICE) forces "cpu" or "mps"; empty means automatic.
    """
    choice = (prefer or os.environ.get("LOOP_ASR_DEVICE", "")).lower()
    if choice not in {"", "cpu", "mps"}:
        raise ValueError(f"unknown device {choice!r}; use 'cpu', 'mps' or leave it empty")
    if choice == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if choice == "mps":
        raise RuntimeError(
            "MPS was requested but is not available "
            f"(built={torch.backends.mps.is_built()}); refusing to fall back to CPU"
        )
    if sys.platform == "darwin":
        warnings.warn("MPS is not available on this Mac; running on CPU", stacklevel=2)
    return torch.device("cpu")
