"""Run Stage-3 Motus on a GPU and serve 16x16 action chunks to Isaac Sim."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path
import traceback

import numpy as np

from motus_runtime import MotusPolicy
from policy_rpc import pack, receive, send, unpack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8616)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seed", type=int, default=16016,
        help="seed the policy process once, matching RoboTwin evaluation",
    )
    parser.add_argument(
        "--reseed-each-request", action="store_true",
        help="non-official deterministic ablation: reuse identical latent noise per request",
    )
    args = parser.parse_args()
    policy = MotusPolicy(
        args.checkpoint, args.device, args.inference_steps, args.seed,
        reseed_each_request=args.reseed_each_request,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(2)
        print(f"Motus Stage-3 policy ready at {args.host}:{args.port}", flush=True)
        while True:
            connection, address = server.accept()
            with connection:
                try:
                    request = unpack(receive(connection))
                    images = tuple(request[name] for name in ("head", "left", "right"))
                    result = policy.predict(images, request["state"], str(request["instruction"].item()))
                    response = pack(actions=result.actions, error=np.asarray(""))
                except Exception as exc:
                    traceback.print_exc()
                    response = pack(actions=np.empty((0, 16), np.float32), error=np.asarray(f"{type(exc).__name__}: {exc}"))
                send(connection, response)


if __name__ == "__main__":
    main()
