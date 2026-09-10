"""Validate raw/processed state-command alignment for the three color tasks."""

from __future__ import annotations

import json
import numpy as np

from data_utils import dataset_summary
from g2_dataset_adapter import is_validation_episode
from settings import ACTION_DIM, PROCESSED_ROOT, RAW_ROOT, TASK_NAMES


def main() -> None:
    raw_rows = []
    failures = 0
    frames = 0
    command_delta = []
    for path in sorted(RAW_ROOT.glob("*/*/episode_*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            n = len(episode["qpos"])
            if episode["qpos"].shape != (n, ACTION_DIM):
                raise ValueError(f"{path}: qpos shape {episode['qpos'].shape}")
            if "command" not in episode or episode["command"].shape != (n, ACTION_DIM):
                raise ValueError(f"{path}: missing/invalid command stream")
            ok = bool(episode["success"].item())
            raw_rows.append(path)
            failures += int(not ok)
            frames += n
            if ok:
                command_delta.append(np.abs(episode["command"] - episode["qpos"]))

    summary = dataset_summary(PROCESSED_ROOT)
    split_counts = {"train": 0, "validation": 0}
    by_task = {}
    for group in summary["groups"]:
        task_dir = PROCESSED_ROOT / group["split"] / group["task"]
        if group["task"] not in TASK_NAMES.values():
            continue
        counts = {"train": 0, "validation": 0}
        for path in (task_dir / "qpos").glob("*.pt"):
            if not (task_dir / "commands" / path.name).is_file():
                continue
            key = "validation" if is_validation_episode(
                group["split"], group["task"], path.stem, 0.1, 16016
            ) else "train"
            counts[key] += 1
            split_counts[key] += 1
        by_task[f"{group['split']}/{group['task']}"] = counts

    deltas = np.concatenate(command_delta) if command_delta else np.empty((0, ACTION_DIM))
    result = {
        "raw_episodes": len(raw_rows),
        "raw_successes": len(raw_rows) - failures,
        "raw_failures": failures,
        "raw_frames_30hz": frames,
        "mean_abs_command_minus_measured": None if not len(deltas) else float(deltas.mean()),
        "processed": summary,
        "held_out_split": {"seed": 16016, "fraction": 0.1, **split_counts, "groups": by_task},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
