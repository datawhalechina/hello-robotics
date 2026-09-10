"""Load the exact Chapter-15 G2 pick/place scene from this chapter's local copy."""

from __future__ import annotations

import sys

from settings import RAW_FPS, TASK_RUNTIME_ROOT

if str(TASK_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_RUNTIME_ROOT))

from auto_expert import AutoExpert  # noqa: E402
from config import COLORS, POSITION_NOISE, SEED, TASK_TEMPLATE, SimulationConfig  # noqa: E402
from robot import G2Robot, closed_fraction  # noqa: E402
from simulation import G2Simulation  # noqa: E402


class MotusSimulationConfig(SimulationConfig):
    """Chapter-15 scene with 30 Hz raw recording, matching Motus downsample=3."""

    @property
    def record_every(self) -> int:
        return max(1, self.physics_hz // RAW_FPS)


__all__ = [
    "AutoExpert",
    "COLORS",
    "G2Robot",
    "G2Simulation",
    "MotusSimulationConfig",
    "POSITION_NOISE",
    "SEED",
    "TASK_TEMPLATE",
    "closed_fraction",
]
