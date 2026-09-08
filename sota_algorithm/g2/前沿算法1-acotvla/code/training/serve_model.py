"""加载微调 checkpoint，启动 G2 ACoT-VLA WebSocket 服务。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import checkpoint_path
from openpi_runtime import (
    checkpoint_uses_lora,
    find_norm_stats,
    prepare_openpi,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 G2 ACoT-VLA WebSocket 服务")
    parser.add_argument("--checkpoint", required=True, help="训练产生的具体 step 目录")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--coarse-horizon", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="微调checkpoint是否使用LoRA；默认从checkpoint自动判断",
    )
    args = parser.parse_args()

    prepare_openpi()
    from g2_policy import make_runtime_config
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared import normalize

    checkpoint = checkpoint_path(args.checkpoint)
    norm_dir = find_norm_stats(checkpoint)
    norm_stats = normalize.load(norm_dir)
    uses_lora = checkpoint_uses_lora(checkpoint) if args.lora is None else args.lora
    config = make_runtime_config(
        lora=uses_lora,
        coarse_action_horizon=args.coarse_horizon,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
    )
    policy = policy_config.create_trained_policy(
        config, checkpoint, norm_stats=norm_stats
    )
    server = WebsocketPolicyServer(
        policy, host="127.0.0.1", port=args.port, metadata=policy.metadata
    )
    logger.info("模型：%s", checkpoint)
    logger.info("归一化统计：%s", norm_dir)
    logger.info("LoRA结构：%s", "是" if uses_lora else "否")
    logger.info("G2 ACoT-VLA 服务监听端口：%d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
