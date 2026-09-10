"""Convert fresh Chapter-16 NPZ trajectories to Motus' official RoboTwin layout."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from data_utils import cache_name, concatenate_views, scalar, write_mp4
from settings import (
    ACTION_DIM,
    MOTUS_ROOT,
    PROCESSED_ROOT,
    RAW_FPS,
    RAW_ROOT,
    ROBOTWIN_VIDEO_HEIGHT,
    ROBOTWIN_VIDEO_WIDTH,
    T5_CACHE_ROOT,
    WAN_ROOT,
    ensure_dirs,
    stage3_prompt,
)


def raw_episodes(root: Path):
    yield from sorted(root.glob("*/*/episode_*.npz"))


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def convert_episode(path: Path, raw_root: Path = RAW_ROOT, overwrite: bool = False) -> tuple[Path, str] | None:
    split, task = path.relative_to(raw_root).parts[:2]
    stem = path.stem
    target = PROCESSED_ROOT / split / task
    qpos_path = target / "qpos" / f"{stem}.pt"
    command_path = target / "commands" / f"{stem}.pt"
    video_path = target / "videos" / f"{stem}.mp4"
    meta_path = target / "metas" / f"{stem}.txt"

    with np.load(path, allow_pickle=False) as episode:
        if "success" in episode and not bool(scalar(episode["success"])):
            return None
        qpos = np.asarray(episode["qpos"], np.float32)
        if "command" not in episode:
            raise ValueError(f"{path}: missing commanded actions; recollect this legacy episode")
        command = np.asarray(episode["command"], np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != ACTION_DIM:
            raise ValueError(f"{path}: expected qpos [T,{ACTION_DIM}], got {qpos.shape}")
        if command.shape != qpos.shape:
            raise ValueError(f"{path}: command {command.shape} must match qpos {qpos.shape}")
        lengths = [len(qpos), len(command), len(episode["head_image"]), len(episode["left_image"]), len(episode["right_image"])]
        if len(set(lengths)) != 1 or lengths[0] < 49:
            raise ValueError(f"{path}: unaligned/short streams {lengths}; Stage 3 needs >=49 raw frames")
        fps = int(scalar(episode["raw_fps"])) if "raw_fps" in episode else RAW_FPS
        if fps != RAW_FPS:
            raise ValueError(f"{path}: raw_fps={fps}, expected {RAW_FPS}")
        raw_instruction = str(scalar(episode["instruction"]))
        model_instruction = stage3_prompt(raw_instruction)

        for folder in (qpos_path.parent, command_path.parent, video_path.parent, meta_path.parent, target / "umt5_wan"):
            folder.mkdir(parents=True, exist_ok=True)
        if overwrite or not qpos_path.exists():
            torch.save(torch.from_numpy(qpos.copy()), qpos_path)
        if overwrite or not command_path.exists():
            torch.save(torch.from_numpy(command.copy()), command_path)
        # Official Stage 3 feeds the same prefixed text to both UMT5 and Qwen-VL.
        # Refresh stale bare-text metadata even when --overwrite is not requested.
        expected_meta = model_instruction + "\n"
        if overwrite or not meta_path.exists() or meta_path.read_text(encoding="utf-8") != expected_meta:
            meta_path.write_text(expected_meta, encoding="utf-8")
        if overwrite or not video_path.exists():
            frames = (
                concatenate_views(episode["head_image"][i], episode["left_image"][i], episode["right_image"][i])
                for i in range(len(qpos))
            )
            count = write_mp4(
                video_path, frames, fps,
                height=ROBOTWIN_VIDEO_HEIGHT, width=ROBOTWIN_VIDEO_WIDTH,
            )
            if count != len(qpos):
                raise RuntimeError(f"{video_path}: wrote {count}/{len(qpos)} frames")
    return target / "umt5_wan" / f"{stem}.pt", model_instruction


def make_t5_encoder(device: str):
    bak = MOTUS_ROOT / "bak"
    if str(bak) not in sys.path:
        sys.path.insert(0, str(bak))
    from wan.modules.t5 import T5EncoderModel

    checkpoint = WAN_ROOT / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer = WAN_ROOT / "google" / "umt5-xxl"
    if not checkpoint.is_file() or not tokenizer.is_dir():
        raise FileNotFoundError(
            "WAN T5 files are incomplete. Expected:\n"
            f"  {checkpoint}\n  {tokenizer}"
        )
    return T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device(device),
        checkpoint_path=str(checkpoint),
        tokenizer_path=str(tokenizer),
    )


def encode_instructions(rows: list[tuple[Path, str]], device: str, overwrite: bool) -> None:
    unique = sorted({text for _, text in rows})
    missing = [text for text in unique if overwrite or not (T5_CACHE_ROOT / cache_name(text)).is_file()]
    encoder = make_t5_encoder(device) if missing else None
    for index, text in enumerate(missing, 1):
        print(f"T5 {index}/{len(missing)}: {text}")
        encoded = encoder([text], torch.device(device))
        values = encoded if isinstance(encoded, list) else list(encoded)
        tensors = [value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in values]
        cache = T5_CACHE_ROOT / cache_name(text)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, cache)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for output, text in rows:
        link_or_copy(T5_CACHE_ROOT / cache_name(text), output)


def main() -> None:
    parser = argparse.ArgumentParser(description="NPZ -> official Motus RoboTwin Stage-3 files")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--encode-t5", action="store_true", help="also create required UMT5 embeddings")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_dirs()

    paths = list(raw_episodes(args.raw_root))
    if not paths:
        raise FileNotFoundError(f"no episodes under {args.raw_root}/{{clean,randomized}}/<task>")
    rows: list[tuple[Path, str]] = []
    skipped = 0
    for index, path in enumerate(paths, 1):
        result = convert_episode(path, args.raw_root, args.overwrite)
        if result is None:
            skipped += 1
        else:
            rows.append(result)
        print(f"convert {index}/{len(paths)}: {path.name}", end="\r")
    print()
    if args.encode_t5:
        encode_instructions(rows, args.device, args.overwrite)
    else:
        print("Videos/qpos/commands/metas are ready. Run again with --encode-t5 before training.")
    print(f"processed={len(rows)} skipped_failed={skipped} output={PROCESSED_ROOT}")


if __name__ == "__main__":
    main()
