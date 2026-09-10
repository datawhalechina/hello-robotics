"""Thin integration layer around the public Motus repository (no model code copied)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from settings import MOTUS_ROOT


def install_motus_imports() -> None:
    if not MOTUS_ROOT.is_dir():
        raise FileNotFoundError(f"Motus repository not found: {MOTUS_ROOT}")
    # ``train/train.py`` imports its sibling ``sample.py`` as a top-level module,
    # exactly as when the official script is launched with ``python train/train.py``.
    # Add that directory as well as the repository and bundled WAN package.
    for path in (MOTUS_ROOT / "train", MOTUS_ROOT, MOTUS_ROOT / "bak"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def configure_attention() -> str:
    """Use official FlashAttention when installed; otherwise patch WAN to torch SDPA."""
    install_motus_imports()
    from wan.modules import attention as attention_module
    from wan.modules import model as model_module

    requested = os.getenv("MOTUS_ATTN_BACKEND", "auto").lower()
    flash_available = bool(
        attention_module.FLASH_ATTN_2_AVAILABLE or attention_module.FLASH_ATTN_3_AVAILABLE
    )
    if requested not in {"auto", "flash", "sdpa"}:
        raise ValueError("MOTUS_ATTN_BACKEND must be auto, flash or sdpa")
    if requested == "flash" and not flash_available:
        raise RuntimeError("FlashAttention requested but flash-attn is not installed")
    if requested == "sdpa" or (requested == "auto" and not flash_available):
        model_module.flash_attention = attention_module.attention
        backend = "torch-sdpa"
    else:
        backend = "flash-attention"
    print(f"attention backend: {backend}")
    return backend
