"""Export chapter NPZ episodes to the LeRobot schema consumed by OpenPI."""

from __future__ import annotations

import argparse
import shutil

import numpy as np

from config import DATASET_FPS, LEROBOT_ROOT, demos_dir, labeled_dir, lerobot_repo
from dataset import episode_paths, load_episode, scalar
from openpi_runtime import prepare_openpi

JOINT_NAMES = [
    *(f"left_arm_{index}" for index in range(1, 8)),
    *(f"right_arm_{index}" for index in range(1, 8)),
    "left_gripper",
    "right_gripper",
]


def source_episodes(stage: str, round_id: int):
    paths = (
        episode_paths([demos_dir()])
        if stage == "sft"
        else sorted(labeled_dir(round_id).glob("episode_*.npz"))
    )
    if not paths:
        raise RuntimeError("no source episodes")
    for path in paths:
        episode = load_episode(path)
        if stage == "sft":
            eligible = bool(scalar(episode.get("use_for_sft", False))) and bool(
                scalar(episode["success"])
            )
            if not eligible:
                continue
            indicators = np.ones(
                len(episode["state"]), dtype=np.int64
            )  # schema only; SFT adapter ignores it
        else:
            if "acp_indicator" not in episode:
                raise ValueError(
                    f"{path} has no acp_indicator; run label_advantages.py first"
                )
            indicators = np.asarray(episode["acp_indicator"], dtype=np.int64)
            if (
                indicators.shape != (len(episode["state"]),)
                or not np.isin(indicators, (0, 1)).all()
            ):
                raise ValueError(f"invalid acp_indicator in {path}")
        yield path, episode, indicators


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sft", "acp"), required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "acp" and args.round < 1:
        raise ValueError("ACP export requires --round >= 1")

    prepare_openpi()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    episodes = list(source_episodes(args.stage, args.round))
    if not episodes:
        raise RuntimeError("no eligible episodes")
    repo_id = lerobot_repo(args.stage, args.round)
    output = LEROBOT_ROOT / repo_id
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)

    first = episodes[0][1]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": [JOINT_NAMES],
        },
        "action": {"dtype": "float32", "shape": (16,), "names": [JOINT_NAMES]},
        "complementary_info.acp_indicator": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "complementary_info.is_intervention": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "complementary_info.policy_action": {
            "dtype": "float32",
            "shape": (16,),
            "names": [JOINT_NAMES],
        },
    }
    for camera, key in (
        ("head", "head_image"),
        ("left_wrist", "left_image"),
        ("right_wrist", "right_image"),
    ):
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": tuple(first[key].shape[1:]),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        robot_type="G2_omnipicker",
        fps=DATASET_FPS,
        features=features,
        use_videos=True,
        image_writer_threads=8,
    )
    frame_count = 0
    for path, episode, indicators in episodes:
        task = str(scalar(episode["task"]))
        for index in range(len(episode["state"])):
            dataset.add_frame(
                {
                    "observation.images.head": episode["head_image"][index],
                    "observation.images.left_wrist": episode["left_image"][index],
                    "observation.images.right_wrist": episode["right_image"][index],
                    "observation.state": episode["state"][index].astype(np.float32),
                    "action": episode["action"][index].astype(np.float32),
                    "complementary_info.acp_indicator": np.asarray(
                        [indicators[index]], dtype=np.int64
                    ),
                    "complementary_info.is_intervention": np.asarray(
                        [episode["is_intervention"][index]], dtype=np.int64
                    ),
                    "complementary_info.policy_action": episode["policy_action"][
                        index
                    ].astype(np.float32),
                    "task": task,
                }
            )
            frame_count += 1
        dataset.save_episode()
        print(
            f"exported {path.name}: frames={len(indicators)} positive={indicators.mean():.1%}"
        )
    print(
        {
            "repo_id": repo_id,
            "output": str(output),
            "episodes": len(episodes),
            "frames": frame_count,
        }
    )


if __name__ == "__main__":
    main()
