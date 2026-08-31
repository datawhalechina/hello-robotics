"""Compute native 16D state/action statistics before pi0.5 pads them to 32D."""

from __future__ import annotations

import argparse

import numpy as np

from config import PI05_ASSET_ROOT, demos_dir, labeled_dir, lerobot_repo
from dataset import episode_paths, load_episode, scalar
from openpi_runtime import prepare_openpi


def selected(stage: str, round_id: int):
    paths = (
        episode_paths([demos_dir()])
        if stage == "sft"
        else episode_paths([labeled_dir(round_id)])
    )
    for path in paths:
        episode = load_episode(path)
        if stage == "sft" and not (
            bool(scalar(episode.get("use_for_sft", False)))
            and bool(scalar(episode["success"]))
        ):
            continue
        yield episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sft", "acp"), required=True)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "acp" and args.round < 1:
        raise ValueError("ACP stats require --round >= 1")

    episodes = list(selected(args.stage, args.round))
    if not episodes:
        raise RuntimeError("no eligible episodes")
    state = np.concatenate([ep["state"] for ep in episodes], axis=0).astype(np.float32)
    action = np.concatenate([ep["action"] for ep in episodes], axis=0).astype(
        np.float32
    )
    if state.shape[1:] != (16,) or action.shape[1:] != (16,):
        raise ValueError("normalization must be computed in native 16D")

    prepare_openpi()
    from openpi.shared import normalize

    state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
    state_stats.update(state)
    action_stats.update(action)
    repo_id = lerobot_repo(args.stage, args.round)
    destination = PI05_ASSET_ROOT / f"g2_pi05_{args.stage}_{args.round:03d}" / repo_id
    normalize.save(
        destination,
        {
            "state": state_stats.get_statistics(),
            "actions": action_stats.get_statistics(),
        },
    )
    print(
        {
            "output": str(destination / "norm_stats.json"),
            "frames": len(state),
            "native_dim": 16,
        }
    )


if __name__ == "__main__":
    main()
