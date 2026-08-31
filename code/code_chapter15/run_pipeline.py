"""Thin command printer/runner for the optional-SFT workflow."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from config import CHECKPOINT_ROOT, PI05_BASE, ROOT


def command(script: str, *arguments) -> list[str]:
    """Build a command for data/model utilities in the chapter OpenPI venv."""

    return [sys.executable, str(ROOT / script), *(str(value) for value in arguments)]


def isaac_command(script: str, *arguments) -> list[str]:
    """Build a simulation command with Isaac Sim's Python launcher."""

    launcher = Path(
        os.getenv("CHAPTER15_ISAAC_PYTHON", "/home/robot/isaac-sim/python.sh")
    ).expanduser()
    return [str(launcher), str(ROOT / script), *(str(value) for value in arguments)]


def main() -> None:
    parser = argparse.ArgumentParser()
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument("--with-sft", action="store_true")
    route.add_argument("--no-sft", action="store_true")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--init", type=Path)
    parser.add_argument(
        "--fsdp-devices",
        type=int,
        default=1,
        help="number of GPUs used to shard pi0.5 policy training",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="use OpenPI low-memory LoRA policy training",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the no-SFT improvement phase; default only prints commands",
    )
    parser.add_argument(
        "--policy-server-ready",
        action="store_true",
        help="confirm that serve_policy.py is already running for --execute",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.round < 1:
        raise ValueError("round must be >= 1")
    if args.execute and args.with_sft:
        raise ValueError(
            "with-SFT crosses a policy-service boundary; print the plan and run sections 6-8 manually"
        )
    if args.execute and not args.policy_server_ready:
        raise ValueError("--execute requires --policy-server-ready")

    overwrite = ["--overwrite"] if args.overwrite else []
    lora = ["--lora"] if args.lora else []
    initial = args.init or PI05_BASE
    steps: list[list[str]] = []
    if args.with_sft:
        steps += [
            isaac_command("collect_demos.py", "--position-noise", "0.01", "--headless", *overwrite),
            command("export_dataset.py", "--stage", "sft", *overwrite),
            command("compute_norm_stats.py", "--stage", "sft"),
            command(
                "train_policy.py",
                "--stage",
                "sft",
                "--initial",
                initial,
                "--fsdp-devices",
                args.fsdp_devices,
                *lora,
                *overwrite,
            ),
        ]
        initial = CHECKPOINT_ROOT / "g2_pi05_sft_000/sft_round_000"

    steps += [
        ["#", "start serve_policy.py in another terminal with the current checkpoint"],
        isaac_command(
            "collect_rollouts.py",
            "--round",
            args.round,
            "--position-noise",
            "0.01",
            "--headless",
            *overwrite,
        ),
        command("train_value.py", "--round", args.round, *overwrite),
        command("label_advantages.py", "--round", args.round, *overwrite),
        command(
            "export_dataset.py", "--stage", "acp", "--round", args.round, *overwrite
        ),
        command("compute_norm_stats.py", "--stage", "acp", "--round", args.round),
        command(
            "train_policy.py",
            "--stage",
            "acp",
            "--round",
            args.round,
            "--initial",
            initial,
            "--fsdp-devices",
            args.fsdp_devices,
            *lora,
            *overwrite,
        ),
    ]
    for step in steps:
        print(" ".join(shlex.quote(part) for part in step), flush=True)
        if args.execute and step[0] != "#":
            subprocess.run(step, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
