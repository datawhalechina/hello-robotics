"""调用openpi原生训练循环微调π0.5，不复制训练框架。"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util

try:
    from .config import CHECKPOINTS, REPO_ID, TRAIN_ASSETS_DIR, TRAIN_CHECKPOINT_DIR, OPENPI_ROOT
    from .openpi_runtime import prepare_openpi
except ImportError:
    from config import CHECKPOINTS, REPO_ID, TRAIN_ASSETS_DIR, TRAIN_CHECKPOINT_DIR, OPENPI_ROOT
    from openpi_runtime import prepare_openpi


def main() -> None:
    parser = argparse.ArgumentParser(description="微调π0.5完成G2彩色物块入盒")
    parser.add_argument("--init", choices=("instruction", "base"), default="instruction",
                        help="instruction已完成G2对齐，通常比原始base更容易训练")
    parser.add_argument("--exp-name", default="first_run")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite和--resume不能同时使用")

    prepare_openpi()
    from g2_policy import make_finetune_config

    initial = CHECKPOINTS[args.init]
    if not (initial / "params").is_dir():
        raise FileNotFoundError(f"找不到初始化权重：{initial / 'params'}")
    config = make_finetune_config(
        repo_id=REPO_ID,
        initial_checkpoint=str(initial),
        assets_base_dir=str(TRAIN_ASSETS_DIR),
        checkpoint_base_dir=str(TRAIN_CHECKPOINT_DIR),
        batch_size=args.batch_size,
        train_steps=args.steps,
        lora=args.lora,
    )
    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb,
        fsdp_devices=args.fsdp_devices,
    )
    script = OPENPI_ROOT / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("chapter14_openpi_train", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载openpi训练脚本：{script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(config)


if __name__ == "__main__":
    main()
