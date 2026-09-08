"""把采集的NPZ回合转换成openpi可读取的LeRobot数据集。"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .config import (
        DATASET_FPS,
        EXPECTED_EPISODE_FRAMES,
        EXPECTED_EPISODES,
        LEROBOT_HOME,
        RAW_DATA_DIR,
        REPO_ID,
    )
except ImportError:
    from config import (
        DATASET_FPS,
        EXPECTED_EPISODE_FRAMES,
        EXPECTED_EPISODES,
        LEROBOT_HOME,
        RAW_DATA_DIR,
        REPO_ID,
    )


FRAME_KEYS = ("head_image", "left_image", "right_image", "state", "actions")
IMAGE_KEYS = FRAME_KEYS[:3]
EXPECTED_IMAGE_SHAPE = (240, 320, 3)


def validate_timing(data, fps: int) -> None:
    """按真实仿真时间核验频率，拒绝仅标注 FPS 的旧轨迹。"""
    if "observation_time" not in data or "image_time" not in data:
        raise ValueError("缺少观测/图像时间戳，请使用同步采集代码重新采集")
    times = np.asarray(data["observation_time"])
    images = np.asarray(data["image_time"])
    frames = len(data["state"])
    if times.shape != (frames,) or images.shape != (frames, 3):
        raise ValueError("时间戳形状与轨迹帧数不一致")
    if not np.isfinite(times).all() or not np.isfinite(images).all():
        raise ValueError("时间戳包含非有限值")
    if not np.allclose(images, times[:, None], rtol=0, atol=1e-6):
        raise ValueError("相机与状态时间戳不同步")
    if not np.allclose(np.diff(times), 1.0 / fps, rtol=0, atol=1e-6):
        raise ValueError("真实采样频率与配置 FPS 不一致")


def validate_episode(
    data,
    *,
    path: Path,
    fps: int,
    expected_frames: int,
) -> None:
    """完整检查一个成功回合，避免坏数据进入视频编码阶段。"""
    missing = [
        key
        for key in (*FRAME_KEYS, "fps", "prompt", "target_color", "success")
        if key not in data
    ]
    if missing:
        raise ValueError(f"{path.name}缺少字段：{missing}")
    if not bool(data["success"]):
        raise ValueError(f"{path.name}不是成功轨迹")
    if int(data["fps"]) != fps:
        raise ValueError(
            f"{path.name}采集频率为{int(data['fps'])} Hz，转换配置为{fps} Hz"
        )

    lengths = {key: int(data[key].shape[0]) for key in FRAME_KEYS}
    if set(lengths.values()) != {expected_frames}:
        raise ValueError(f"{path.name}帧数合同错误：{lengths}")

    for key in ("state", "actions"):
        value = np.asarray(data[key])
        if value.shape != (expected_frames, 16) or not np.isfinite(value).all():
            raise ValueError(f"{path.name}的{key}维度或数值错误：{value.shape}")
    for key in IMAGE_KEYS:
        value = np.asarray(data[key])
        if (
            value.shape != (expected_frames, *EXPECTED_IMAGE_SHAPE)
            or value.dtype != np.uint8
        ):
            raise ValueError(
                f"{path.name}的{key}图像格式错误：{value.shape}/{value.dtype}"
            )

    prompt = str(data["prompt"]).strip()
    color = str(data["target_color"]).strip()
    if not prompt or not color:
        raise ValueError(f"{path.name}的任务指令或目标颜色为空")
    validate_timing(data, fps)


def resolve_dataset_output(lerobot_home: Path, repo_id: str) -> Path:
    """将数据集输出限制在 LeRobot 根目录内，防止 --overwrite 误删其他目录。"""
    root = Path(lerobot_home).expanduser().resolve()
    relative = Path(repo_id)
    if relative.is_absolute():
        raise ValueError("--repo-id必须是相对于--lerobot-home的仓库名称")
    output = (root / relative).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError(f"数据集输出必须位于{root}内，实际为{output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="NPZ -> LeRobot")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--lerobot-home", type=Path, default=LEROBOT_HOME)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    episodes = []
    for path in sorted(args.raw_dir.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if bool(data["success"]):
                episodes.append(path)
            else:
                print(f"[跳过] {path.name}未成功")
    if not episodes:
        raise RuntimeError("没有可转换的回合，请先运行 collect_demos.py")

    if len(episodes) != EXPECTED_EPISODES:
        raise ValueError(f"成功回合数应为{EXPECTED_EPISODES}，实际为{len(episodes)}")
    color_counts = Counter()
    for path in episodes:
        with np.load(path, allow_pickle=False) as data:
            validate_episode(
                data,
                path=path,
                fps=DATASET_FPS,
                expected_frames=EXPECTED_EPISODE_FRAMES,
            )
            color_counts[str(data["target_color"])] += 1
    if color_counts != Counter({"red": 20, "green": 20, "blue": 20}):
        raise ValueError(f"RGB成功回合分布错误：{dict(color_counts)}")

    with np.load(episodes[0], allow_pickle=False) as first:
        image_shapes = {
            "top_head": tuple(first["head_image"].shape[1:]),
            "hand_left": tuple(first["left_image"].shape[1:]),
            "hand_right": tuple(first["right_image"].shape[1:]),
        }

    output = resolve_dataset_output(args.lerobot_home, args.repo_id)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output}已存在；如需重建请加--overwrite")
        shutil.rmtree(output)

    names = [
        "left_arm_1",
        "left_arm_2",
        "left_arm_3",
        "left_arm_4",
        "left_arm_5",
        "left_arm_6",
        "left_arm_7",
        "right_arm_1",
        "right_arm_2",
        "right_arm_3",
        "right_arm_4",
        "right_arm_5",
        "right_arm_6",
        "right_arm_7",
        "left_gripper",
        "right_gripper",
    ]
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": [names]},
        "action": {"dtype": "float32", "shape": (16,), "names": [names]},
    }
    for camera, shape in image_shapes.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
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
                data["head_image"],
                data["left_image"],
                data["right_image"],
                data["state"],
                data["actions"],
                strict=True,
            ):
                dataset.add_frame(
                    {
                        "observation.images.top_head": head,
                        "observation.images.hand_left": left,
                        "observation.images.hand_right": right,
                        "observation.state": state,
                        "action": action,
                        "task": prompt,
                    }
                )
            dataset.save_episode()
    print(f"[完成] LeRobot数据集：{output}，共{len(episodes)}回合")
    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["g2", "pi05", "isaac-sim"], private=True, push_videos=True
        )


if __name__ == "__main__":
    main()
