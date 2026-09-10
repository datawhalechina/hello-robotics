"""Small G2 adapter for Motus' official RoboTwin Stage-3 dataset.

The official RoboTwin converter calls commanded joint targets ``qpos``.  Chapter
16 records both the measured state and the command, so they must stay separate:

* ``qpos/*.pt``: measured 16-D state used to condition Motus;
* ``commands/*.pt``: 16-D absolute joint targets used as action supervision.

This module patches only the dataset class imported from Motus.  Sampling,
video loading, UMT5/Qwen processing and Stage-3 losses remain official Motus.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

import torch

from settings import ACTION_DIM, TASK_NAMES

LOGGER = logging.getLogger(__name__)
DEFAULT_TASKS = tuple(TASK_NAMES.values())


def is_validation_episode(split: str, task: str, stem: str, fraction: float, seed: int) -> bool:
    """Deterministically assign a complete episode to train or validation."""
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    key = f"{int(seed)}:{split}/{task}/{stem}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    return bucket < float(fraction)


def command_path_from_qpos(qpos_path: str | Path) -> Path:
    qpos = Path(qpos_path)
    return qpos.parent.parent / "commands" / qpos.name


def load_state_action(
    qpos_path: str | Path,
    action_indices: Iterable[int],
    initial_state_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load measured condition state and commanded future absolute targets."""
    qpos_path = Path(qpos_path)
    command_path = command_path_from_qpos(qpos_path)
    measured = torch.as_tensor(torch.load(qpos_path, map_location="cpu"), dtype=torch.float32)
    commands = torch.as_tensor(torch.load(command_path, map_location="cpu"), dtype=torch.float32)

    if measured.ndim != 2 or measured.shape[1] != ACTION_DIM:
        raise ValueError(f"{qpos_path}: expected measured qpos [T,{ACTION_DIM}], got {tuple(measured.shape)}")
    if commands.ndim != 2 or commands.shape[1] != ACTION_DIM:
        raise ValueError(f"{command_path}: expected commands [T,{ACTION_DIM}], got {tuple(commands.shape)}")
    if len(measured) != len(commands):
        raise ValueError(f"unaligned state/action streams: {len(measured)} != {len(commands)}")

    state_index = min(max(int(initial_state_idx), 0), len(measured) - 1)
    indices = [int(index) for index in action_indices]
    if not indices:
        raise ValueError("action_indices must not be empty")
    if min(indices) < 0 or max(indices) >= len(commands):
        raise IndexError(f"action index range [{min(indices)}, {max(indices)}] outside T={len(commands)}")
    return measured[state_index].clone(), commands[indices].clone()


def install_g2_dataset_adapter() -> None:
    """Patch Motus' RoboTwin loader before the official trainer builds datasets."""
    from data.robotwin2 import robotwin_agilex_dataset as module

    dataset_class = module.RobotWinTaskDataset
    if getattr(dataset_class, "_chapter16_g2_adapter", False):
        return

    original_init = dataset_class.__init__

    def adapted_init(
        self,
        *args,
        validation_fraction: float = 0.1,
        split_seed: int = 16016,
        allowed_tasks: Iterable[str] | None = None,
        **kwargs,
    ):
        self.chapter16_validation_fraction = float(validation_fraction)
        self.chapter16_split_seed = int(split_seed)
        self.chapter16_allowed_tasks = set(allowed_tasks or DEFAULT_TASKS)
        original_init(self, *args, **kwargs)
        LOGGER.info(
            "Chapter-16 G2 %s split: %d episodes from tasks=%s",
            "validation" if self.val else "train",
            self.total_episodes,
            sorted(self.chapter16_allowed_tasks),
        )

    def scan_task_folder(self, task_path: Path):
        task_path = Path(task_path)
        if task_path.name not in self.chapter16_allowed_tasks:
            LOGGER.info("Skip non-Chapter-16 task: %s", task_path)
            return []

        required = {
            "qpos": task_path / "qpos",
            "commands": task_path / "commands",
            "videos": task_path / "videos",
            "umt5_wan": task_path / "umt5_wan",
            "metas": task_path / "metas",
        }
        if not all(path.is_dir() for path in required.values()):
            LOGGER.warning("Missing G2 Stage-3 directories in %s", task_path)
            return []

        split = task_path.parent.name
        episodes = []
        for qpos_file in sorted(required["qpos"].glob("*.pt")):
            stem = qpos_file.stem
            files = {
                "command_path": required["commands"] / f"{stem}.pt",
                "video_path": required["videos"] / f"{stem}.mp4",
                "lang_path": required["umt5_wan"] / f"{stem}.pt",
                "meta_path": required["metas"] / f"{stem}.txt",
            }
            if not all(path.is_file() for path in files.values()):
                continue
            belongs_to_val = is_validation_episode(
                split, task_path.name, stem,
                self.chapter16_validation_fraction,
                self.chapter16_split_seed,
            )
            if bool(self.val) != belongs_to_val:
                continue
            episodes.append(
                {
                    "episode_name": stem,
                    "task_name": task_path.name,
                    "qpos_path": str(qpos_file),
                    **{name: str(path) for name, path in files.items()},
                }
            )
        LOGGER.info(
            "Task %s/%s: %d complete %s episodes",
            split, task_path.name, len(episodes), "val" if self.val else "train",
        )
        return episodes

    def load_robot_data(self, qpos_path, action_indices, initial_state_idx=0):
        return load_state_action(qpos_path, action_indices, initial_state_idx)

    dataset_class.__init__ = adapted_init
    dataset_class._scan_task_folder = scan_task_folder
    dataset_class._load_robot_data = load_robot_data
    dataset_class._chapter16_g2_adapter = True
