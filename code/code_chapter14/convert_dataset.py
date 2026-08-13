"""把采集的NPZ回合转换成openpi可读取的LeRobot数据集。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import numpy as np

try:
    from .config import DATASET_FPS, LEROBOT_HOME, RAW_DATA_DIR, REPO_ID
except ImportError:
    from config import DATASET_FPS, LEROBOT_HOME, RAW_DATA_DIR, REPO_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="NPZ -> LeRobot")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--lerobot-home", type=Path, default=LEROBOT_HOME)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    episodes = []
    for path in sorted(args.raw_dir.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if args.keep_failed or bool(data["success"]):
                episodes.append(path)
            else:
                print(f"[跳过] {path.name}未成功")
    if not episodes:
        raise RuntimeError("没有可转换的回合，请先运行collect_data.py")

    with np.load(episodes[0], allow_pickle=False) as first:
        for key in ("state", "actions"):
            if first[key].shape[-1] != 16:
                raise ValueError(f"{key}必须是完整G2 16维")
        image_shapes = {
            "top_head": tuple(first["head_image"].shape[1:]),
            "hand_left": tuple(first["left_image"].shape[1:]),
            "hand_right": tuple(first["right_image"].shape[1:]),
        }

    output = args.lerobot_home / args.repo_id
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output}已存在；如需重建请加--overwrite")
        shutil.rmtree(output)

    names = [
        "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6", "left_arm_7",
        "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6", "right_arm_7",
        "left_gripper", "right_gripper",
    ]
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": [names]},
        "action": {"dtype": "float32", "shape": (16,), "names": [names]},
    }
    for camera, shape in image_shapes.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "image", "shape": shape, "names": ["height", "width", "channel"]
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        robot_type="G2_omnipicker",
        fps=DATASET_FPS,
        features=features,
        use_videos=True,
        image_writer_threads=8,
    )
    for path in episodes:
        with np.load(path, allow_pickle=False) as data:
            prompt = str(data["prompt"])
            for head, left, right, state, action in zip(
                data["head_image"], data["left_image"], data["right_image"],
                data["state"], data["actions"], strict=True,
            ):
                dataset.add_frame({
                    "observation.images.top_head": head,
                    "observation.images.hand_left": left,
                    "observation.images.hand_right": right,
                    "observation.state": state,
                    "action": action,
                    "task": prompt,
                })
            dataset.save_episode()
    print(f"[完成] LeRobot数据集：{output}，共{len(episodes)}回合")
    if args.push_to_hub:
        dataset.push_to_hub(tags=["g2", "pi05", "isaac-sim"], private=True, push_videos=True)


if __name__ == "__main__":
    main()
