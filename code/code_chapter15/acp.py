"""Advantage-conditioned prompt rewriting; it runs before tokenization."""

from __future__ import annotations
import random
import re
from config import ACP_NEGATIVE, ACP_POSITIVE

_TAG = re.compile(r"\s*Advantage:\s*(positive|negative)\s*$", re.I)


def clean_task(task: str) -> str:
    return _TAG.sub("", str(task)).strip()


def tagged_task(task: str, positive: bool) -> str:
    return f"{clean_task(task)}\n{ACP_POSITIVE if positive else ACP_NEGATIVE}"


class ACPPrompt:
    def __init__(self, dropout: float = 0.3, seed: int = 1000):
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.dropout, self.rng = float(dropout), random.Random(seed)

    def __call__(self, task: str, indicator: int) -> str:
        if indicator not in (0, 1):
            raise ValueError("indicator must be 0 or 1")
        return (
            clean_task(task)
            if self.rng.random() < self.dropout
            else tagged_task(task, indicator == 1)
        )
