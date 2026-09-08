"""调用官方 ACoT-VLA 训练循环微调 G2 三色物块任务。"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
from pathlib import Path

try:
    from .config import (
        DATASET_VERSION,
        OPENPI_ROOT,
        PI05_BASE_CHECKPOINT,
        REPO_ID,
        TRAIN_ASSETS_DIR,
        TRAIN_CHECKPOINT_DIR,
    )
    from .openpi_runtime import prepare_openpi
except ImportError:
    from config import (
        DATASET_VERSION,
        OPENPI_ROOT,
        PI05_BASE_CHECKPOINT,
        REPO_ID,
        TRAIN_ASSETS_DIR,
        TRAIN_CHECKPOINT_DIR,
    )
    from openpi_runtime import prepare_openpi


def main() -> None:
    parser = argparse.ArgumentParser(description="基于官方 pi05_base 权重微调 ACoT-VLA")
    parser.add_argument(
        "--exp-name", default=f"acot_pi05_base_{DATASET_VERSION}_h50_f30_b1_20k_v1"
    )
    parser.add_argument("--steps", type=int, default=20_000, help="优化器更新次数")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10_000,
        help="checkpoint 保存间隔（step）；最终 step 始终额外保存",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="本地 24GB GPU 默认使用 batch 1；显存充足时可显式增大",
    )
    parser.add_argument("--coarse-horizon", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=200,
        help="π0.5 输入 token 上限；保持官方默认 200，避免截断离散状态与指令",
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=str(PI05_BASE_CHECKPOINT),
        help="ACoT-VLA 的 π0.5 初始化 checkpoint 根目录",
    )
    parser.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认冻结 Pi0.5 语言基座、训练 LoRA；ACoT 新模块始终全量训练",
    )
    parser.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="保留完整 EMA 参数副本；24GB 本地显卡默认关闭以降低显存峰值",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        help="覆盖LeRobot数据根目录；仅用于隔离的smoke test",
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=TRAIN_ASSETS_DIR,
        help="归一化统计根目录",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=TRAIN_CHECKPOINT_DIR,
        help="训练输出根目录",
    )
    args = parser.parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite和--resume不能同时使用")
    if args.save_interval < 1:
        raise ValueError("--save-interval必须大于0")
    if (
        args.batch_size < 1
        or args.steps < 1
        or args.coarse_horizon < 1
        or args.action_horizon < 1
        or args.max_token_len < 1
    ):
        raise ValueError("steps、batch-size、horizon和max-token-len必须大于0")

    # 必须在导入 JAX 前设置。platform allocator 会释放初始化临时块，
    # 否则即使 Adafactor 稳态可放入 24 GiB，官方初始化仍会因峰值 OOM。
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    prepare_openpi(lerobot_home=args.lerobot_home)
    from g2_policy import make_finetune_config

    initial = Path(args.initial_checkpoint).expanduser().resolve()
    if not (initial / "params").is_dir():
        raise FileNotFoundError(f"初始化 checkpoint 缺少 params：{initial}")
    config = make_finetune_config(
        repo_id=args.repo_id,
        initial_checkpoint=str(initial),
        assets_base_dir=str(args.assets_dir.resolve()),
        checkpoint_base_dir=str(args.checkpoint_dir.resolve()),
        batch_size=args.batch_size,
        train_steps=args.steps,
        lora=args.lora,
        coarse_action_horizon=args.coarse_horizon,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        ema_decay=0.999 if args.ema else None,
    )
    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb,
        fsdp_devices=1,
        save_interval=args.save_interval,
    )
    script = OPENPI_ROOT / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("acotvla_g2_train", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载openpi训练脚本：{script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(config)


if __name__ == "__main__":
    main()
