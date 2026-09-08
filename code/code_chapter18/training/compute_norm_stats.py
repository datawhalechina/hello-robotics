"""计算 G2 数据的 ACoT state/coarse/fine 归一化统计。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tqdm

try:
    from .config import (
        GRIPPER_OPEN_RAD,
        PI05_BASE_CHECKPOINT,
        REPO_ID,
        TRAIN_ASSETS_DIR,
        TRAIN_CHECKPOINT_DIR,
    )
    from .openpi_runtime import prepare_openpi
except ImportError:
    from config import (
        GRIPPER_OPEN_RAD,
        PI05_BASE_CHECKPOINT,
        REPO_ID,
        TRAIN_ASSETS_DIR,
        TRAIN_CHECKPOINT_DIR,
    )
    from openpi_runtime import prepare_openpi


def main() -> None:
    parser = argparse.ArgumentParser(description="计算 G2 ACoT 数据归一化统计")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--coarse-horizon", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--min-std",
        type=float,
        default=1e-3,
        help="状态/动作标准差下限，防止近常量维度在归一化后被放大",
    )
    parser.add_argument(
        "--max-normalized-abs",
        type=float,
        default=10.0,
        help="统计完成后允许的最大归一化绝对值",
    )
    parser.add_argument(
        "--max-left-gripper-spread",
        type=float,
        default=2e-5,
        help="左夹爪常量标签允许的数值残差；默认保持严格检查",
    )
    parser.add_argument(
        "--lerobot-home", type=Path, help="覆盖LeRobot数据根目录，便于测试或迁移数据"
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=TRAIN_ASSETS_DIR)
    args = parser.parse_args()
    if args.min_std <= 0:
        raise ValueError("--min-std必须大于0")
    if args.coarse_horizon < 1 or args.action_horizon < 1:
        raise ValueError("horizon必须大于0")
    if args.max_normalized_abs <= 0:
        raise ValueError("--max-normalized-abs必须大于0")
    if args.max_left_gripper_spread < 0:
        raise ValueError("--max-left-gripper-spread不能小于0")
    # 归一化只遍历数据，不需要占用训练 GPU。必须在首次触发 JAX 后端前
    # 显式选择 CPU；仅隐藏 CUDA 设备会让 CUDA-only 插件初始化报错。
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    prepare_openpi(lerobot_home=args.lerobot_home)

    from g2_policy import make_finetune_config
    from openpi.shared import normalize
    from openpi.training import data_loader

    config = make_finetune_config(
        repo_id=args.repo_id,
        initial_checkpoint=str(PI05_BASE_CHECKPOINT),
        assets_base_dir=str(args.output_dir),
        checkpoint_base_dir=str(TRAIN_CHECKPOINT_DIR),
        batch_size=args.batch_size,
        train_steps=1000,
        lora=True,
        coarse_action_horizon=args.coarse_horizon,
        action_horizon=args.action_horizon,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    coarse_horizon = int(config.model.coarse_action_horizon)
    action_horizon = int(config.model.action_horizon)
    coarse_shift, action_shift = tuple(data_config.joint_action_shifts)
    coarse_required = (coarse_horizon - 1) * coarse_shift + 1
    action_required = (action_horizon - 1) * action_shift + 1

    dataset = data_loader.create_torch_dataset(data_config, config.model)
    # 统计只依赖 state/action。若沿用完整训练 transform，LeRobot 会把三路
    # 图像也逐帧解码，既不影响统计值又显著拖慢共享存储上的全量遍历。
    # 先去掉 PromptFromLeRobotTask 包装，再只保留统计所需的 Parquet 列。
    raw_dataset = getattr(dataset, "_dataset", dataset)
    required_columns = (
        "observation.state",
        "action",
        "episode_index",
        "task_index",
        "timestamp",
    )
    raw_dataset.hf_dataset = raw_dataset.hf_dataset.select_columns(required_columns)

    class StatsDataset:
        """等价执行 G2ACOTInputs + ACOTDeltaActions，但跳过图像与文本。"""

        def __len__(self) -> int:
            return len(raw_dataset)

        def __getitem__(self, index: int) -> dict[str, np.ndarray]:
            sample = raw_dataset[index]
            state = np.asarray(sample["observation.state"], dtype=np.float32).reshape(
                -1
            )
            actions = np.asarray(sample["action"], dtype=np.float32).copy()
            if state.size != 16 or actions.ndim != 2 or actions.shape[1] != 16:
                raise ValueError(
                    f"G2统计输入形状错误：state={state.shape}, actions={actions.shape}"
                )
            # 与 G2ACOTInputs 保持一致：左夹爪未参与任务，去掉首次冷启动
            # 造成的 PhysX 数值回弹，只保留标准张开动作监督。
            actions[:, 14] = -GRIPPER_OPEN_RAD
            required = max(coarse_required, action_required)
            if actions.shape[0] < required:
                raise ValueError(
                    f"ACoT动作窗口需要至少{required}帧，实际为{actions.shape[0]}"
                )

            state = np.pad(state, (0, 16))
            coarse = np.pad(actions[:coarse_required:coarse_shift], ((0, 0), (0, 16)))
            fine = np.pad(actions[:action_required:action_shift], ((0, 0), (0, 16)))
            # 与 G2ACOTDataConfig 的 make_bool_mask(14, -2) 一致：双臂转为
            # 相对当前状态的 delta，两个夹爪继续保留绝对动作。
            coarse[:, :14] -= state[:14]
            fine[:, :14] -= state[:14]
            return {"state": state, "coarse_actions": coarse, "actions": fine}

    dataset = StatsDataset()
    frame_count = (
        len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)
    )
    if frame_count < 1:
        raise RuntimeError("数据集为空，请先采集并转换数据")
    batch_size = min(args.batch_size, frame_count)
    num_batches = max(1, frame_count // batch_size)
    loader = data_loader.TorchDataLoader(
        # 统计脚本使用主进程读取，避免局部教学transform在spawn worker中无法序列化。
        dataset,
        local_batch_size=batch_size,
        num_workers=0,
        shuffle=args.max_frames is not None,
        num_batches=num_batches,
    )
    running = {
        key: normalize.RunningStats() for key in ("state", "coarse_actions", "actions")
    }
    observed_min = {key: None for key in running}
    observed_max = {key: None for key in running}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing G2 stats"):
        for key, stats in running.items():
            # 官方 RunningStats 使用 E[x²]-E[x]²。输入保持 float32 时，夹爪
            # 这类“绝对值约 0.785、方差约 1e-4”的维度容易发生数值抵消，
            # 因此统计阶段统一提升到 float64。
            values = np.asarray(batch[key], dtype=np.float64)
            values = values.reshape(-1, values.shape[-1])
            stats.update(values)
            batch_min = values.min(axis=0)
            batch_max = values.max(axis=0)
            observed_min[key] = (
                batch_min
                if observed_min[key] is None
                else np.minimum(observed_min[key], batch_min)
            )
            observed_max[key] = (
                batch_max
                if observed_max[key] is None
                else np.maximum(observed_max[key], batch_max)
            )

    # 本任务只使用右夹爪，左夹爪动作必须严格保持标准张开命令。若出现
    # 波动，说明动作标签包含 PhysX 状态误差，应先检查数据。
    for key in ("coarse_actions", "actions"):
        left_gripper_spread = float(observed_max[key][14] - observed_min[key][14])
        if left_gripper_spread > args.max_left_gripper_spread:
            raise RuntimeError(
                f"{key}左夹爪动作不是常量：range={left_gripper_spread:.8f}；"
                f"允许上限={args.max_left_gripper_spread:.8f}；"
                "请重新采集、修复动作标签或显式调整数值残差容差"
            )

    result = {}
    for key, stats in running.items():
        raw = stats.get_statistics()
        result[key] = normalize.NormStats(
            mean=np.asarray(raw.mean, dtype=np.float64),
            std=np.maximum(np.asarray(raw.std, dtype=np.float64), args.min_std),
            q01=np.asarray(raw.q01, dtype=np.float64),
            q99=np.asarray(raw.q99, dtype=np.float64),
        )
        denominator = result[key].std + 1e-6
        max_abs = float(
            np.maximum(
                np.abs((observed_min[key] - result[key].mean) / denominator),
                np.abs((observed_max[key] - result[key].mean) / denominator),
            ).max()
        )
        print(
            f"[归一化检查] {key}: max_abs={max_abs:.4f}, min_std={result[key].std.min():.6f}"
        )
        if max_abs > args.max_normalized_abs:
            raise RuntimeError(
                f"{key}归一化幅度异常：max_abs={max_abs:.4f} > "
                f"{args.max_normalized_abs:.4f}"
            )
    output = config.assets_dirs / data_config.asset_id
    normalize.save(output, result)
    print(f"[完成] 归一化统计：{output / 'norm_stats.json'}")
    print(
        f"[维度] state={result['state'].mean.shape[-1]} "
        f"coarse={result['coarse_actions'].mean.shape[-1]} actions={result['actions'].mean.shape[-1]}"
    )


if __name__ == "__main__":
    main()
