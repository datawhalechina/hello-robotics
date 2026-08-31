"""Serve a trained chapter-15 pi0.5 checkpoint over OpenPI's websocket protocol."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openpi_runtime import metadata, norm_directory, prepare_openpi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    prepare_openpi()
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared import normalize
    from pi05_adapter import runtime_config

    checkpoint = args.checkpoint.expanduser().resolve()
    info = metadata(checkpoint)
    if (
        info.get("contract") != "g2_pi05_v1"
        or info.get("action_transform") != "all_absolute"
    ):
        raise ValueError(f"incompatible checkpoint metadata: {info}")
    stats = normalize.load(norm_directory(checkpoint))
    if any(len(getattr(stats[key], "mean")) != 16 for key in ("state", "actions")):
        raise ValueError("checkpoint norm stats must be native 16D")
    cfg = runtime_config(bool(info.get("lora", False)), info)
    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint,
        norm_stats=stats,
        sample_kwargs={"num_steps": 10},
    )
    logging.basicConfig(level=logging.INFO)
    print(
        f"serving {checkpoint} at ws://{args.host}:{args.port}; inference=positive tag, flow_steps=10, CFG=off"
    )
    WebsocketPolicyServer(
        policy, host=args.host, port=args.port, metadata=info
    ).serve_forever()


if __name__ == "__main__":
    main()
