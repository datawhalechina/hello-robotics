"""启动instruction、manipulation或本章微调后的π0.5服务。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import checkpoint_path  # noqa: E402
from openpi_runtime import checkpoint_uses_lora, find_norm_stats, prepare_openpi  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="启动G2 π0.5 WebSocket服务")
    parser.add_argument("--model", choices=("instruction", "manipulation", "finetuned"), default="manipulation")
    parser.add_argument("--checkpoint", help="覆盖默认checkpoint；微调模型必须指定到具体step目录")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--lora", action=argparse.BooleanOptionalAction, default=None,
        help="微调checkpoint是否使用LoRA；默认从checkpoint自动判断",
    )
    args = parser.parse_args()

    prepare_openpi()
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared import normalize
    from g2_policy import make_runtime_config

    if args.model == "finetuned" and not args.checkpoint:
        raise ValueError("使用finetuned时请用--checkpoint指定训练产生的step目录")
    # 微调模型沿用本章instruction结构；checkpoint_path只负责本地路径检查。
    config_kind = "finetuned" if args.model == "finetuned" else args.model
    checkpoint = checkpoint_path(
        "instruction" if args.model == "finetuned" else args.model,
        args.checkpoint,
    )
    norm_dir = find_norm_stats(checkpoint)
    norm_stats = normalize.load(norm_dir)
    uses_lora = checkpoint_uses_lora(checkpoint) if args.lora is None else args.lora
    config = make_runtime_config(config_kind, lora=uses_lora)
    policy = policy_config.create_trained_policy(config, checkpoint, norm_stats=norm_stats)
    server = WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port, metadata=policy.metadata)
    logging.info("模型：%s", checkpoint)
    logging.info("归一化统计：%s", norm_dir)
    logging.info("LoRA结构：%s", "是" if uses_lora else "否")
    logging.info("G2 π0.5服务监听端口：%d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
