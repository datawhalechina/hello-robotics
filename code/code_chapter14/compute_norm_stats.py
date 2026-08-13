"""计算本章16维G2状态/动作的π0.5分位数归一化统计。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tqdm

try:
    from .config import REPO_ID, TRAIN_ASSETS_DIR
    from .openpi_runtime import prepare_openpi
    from .config import CHECKPOINTS, TRAIN_CHECKPOINT_DIR
except ImportError:
    from config import REPO_ID, TRAIN_ASSETS_DIR, CHECKPOINTS, TRAIN_CHECKPOINT_DIR
    from openpi_runtime import prepare_openpi


def main() -> None:
    parser = argparse.ArgumentParser(description="计算G2数据归一化统计")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--lerobot-home", type=Path, help="覆盖LeRobot数据根目录，便于测试或迁移数据")
    parser.add_argument("--output-dir", type=Path, default=TRAIN_ASSETS_DIR)
    args = parser.parse_args()
    if args.lerobot_home is not None:
        os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())
    prepare_openpi()

    from g2_policy import make_finetune_config
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training import data_loader

    config = make_finetune_config(
        repo_id=REPO_ID,
        initial_checkpoint=str(CHECKPOINTS["instruction"]),
        assets_base_dir=str(args.output_dir),
        checkpoint_base_dir=str(TRAIN_CHECKPOINT_DIR),
        batch_size=args.batch_size,
        train_steps=1000,
        lora=False,
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    class RemoveStrings(transforms.DataTransformFn):
        def __call__(self, values: dict) -> dict:
            return {key: value for key, value in values.items()
                    if not np.issubdtype(np.asarray(value).dtype, np.str_)}

    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.TransformedDataset(dataset, [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        RemoveStrings(),
    ])
    frame_count = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)
    if frame_count < 1:
        raise RuntimeError("数据集为空，请先采集并转换数据")
    batch_size = min(args.batch_size, frame_count)
    num_batches = max(1, int(np.ceil(frame_count / batch_size)))
    loader = data_loader.TorchDataLoader(
        # 统计脚本使用主进程读取，避免局部教学transform在spawn worker中无法序列化。
        dataset, local_batch_size=batch_size, num_workers=0,
        shuffle=args.max_frames is not None, num_batches=num_batches,
    )
    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing G2 stats"):
        for key, stats in running.items():
            values = np.asarray(batch[key]).reshape(-1, np.asarray(batch[key]).shape[-1])
            stats.update(values)
    result = {key: stats.get_statistics() for key, stats in running.items()}
    output = config.assets_dirs / data_config.asset_id
    normalize.save(output, result)
    print(f"[完成] 归一化统计：{output / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
