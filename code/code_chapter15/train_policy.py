"""Train pi0.5 either as optional untagged SFT or as Evo-RL ACP improvement."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path

from config import PI05_BASE, PolicyTrainPreset, lerobot_repo
from openpi_runtime import latest, prepare_openpi


def load_openpi_train_main():
    from config import OPENPI_ROOT

    path = OPENPI_ROOT / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("chapter15_openpi_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def resolve_initial(stage: str, initial: Path | None) -> Path:
    value = (initial or PI05_BASE).expanduser().resolve()
    if value.name == "params" and value.is_dir():
        value = value.parent
    if not (value / "params").exists():
        try:
            value = latest(value)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"initial checkpoint needs params/: {value}"
            ) from exc
    if stage == "acp" and value == PI05_BASE.resolve():
        print("warning: ACP starts directly from pi0.5 base because SFT was skipped")
    return value


def pending_provenance_path(checkpoint_root: Path) -> Path:
    """Return a path OpenPI cannot remove when it overwrites the run directory."""

    return checkpoint_root.parent / f"{checkpoint_root.name}.chapter15_run.pending.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_checkpoint_provenance(checkpoint_root: Path, payload: dict) -> None:
    for checkpoint in sorted(
        path
        for path in checkpoint_root.glob("*")
        if path.is_dir() and path.name.isdigit()
    ):
        write_json(checkpoint / "chapter15_policy.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sft", "acp"), required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--steps", type=int, default=PolicyTrainPreset().steps)
    parser.add_argument(
        "--batch-size", type=int, default=PolicyTrainPreset().batch_size
    )
    parser.add_argument("--lora", action="store_true")
    parser.add_argument(
        "--fsdp-devices",
        type=int,
        default=1,
        help="shard model and optimizer state across this many GPUs",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "acp" and args.round < 1:
        raise ValueError("ACP training requires --round >= 1")
    if args.resume and args.overwrite:
        raise ValueError("choose only one of --resume/--overwrite")
    if args.fsdp_devices < 1:
        raise ValueError("--fsdp-devices must be >= 1")

    # pi0.5 LoRA with batch 32 peaks above JAX's default 75% GPU pool on 32GB cards.
    # Respect an explicit user setting, otherwise reserve enough memory before JAX imports.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    if not args.lora and args.fsdp_devices == 1:
        print(
            "warning: full pi0.5 fine-tuning replicates model and AdamW state on one GPU; "
            "32GB cards should use --lora or a supported multi-GPU FSDP setup"
        )

    prepare_openpi()
    from pi05_adapter import train_config

    initial = resolve_initial(args.stage, args.initial)
    repo_id = lerobot_repo(args.stage, args.round)
    name = f"g2_pi05_{args.stage}_{args.round:03d}"
    experiment = args.experiment or f"{args.stage}_round_{args.round:03d}"
    cfg = train_config(
        name,
        repo_id,
        initial,
        args.stage,
        args.batch_size,
        args.steps,
        args.lora,
        fsdp_devices=args.fsdp_devices,
    )
    cfg = dataclasses.replace(
        cfg, exp_name=experiment, resume=args.resume, overwrite=args.overwrite
    )
    train_main = load_openpi_train_main()
    run_root = cfg.checkpoint_dir
    payload = {
        "stage": args.stage,
        "round_id": args.round,
        "base_model": "pi0.5",
        "contract": "g2_pi05_v1",
        "action_transform": "all_absolute",
        "initialized_from": str(initial),
        "repo_id": repo_id,
        "lora": bool(args.lora),
        "fsdp_devices": int(args.fsdp_devices),
        "step": int(args.steps),
    }
    pending = pending_provenance_path(run_root)
    write_json(pending, payload)
    train_main(cfg)
    write_json(run_root / "chapter15_run.json", payload)
    write_checkpoint_provenance(run_root, payload)
    pending.unlink(missing_ok=True)
    print({"checkpoint_root": str(run_root), **payload})


if __name__ == "__main__":
    main()
