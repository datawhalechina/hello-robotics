"""Case 1: generate an 8-frame future video and a 16-step action chunk."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from data_utils import PIL_BILINEAR, stitch_views, write_mp4
from motus_runtime import MotusPolicy
from settings import STATE_DIM, TASK_TEXT, VIDEO_HEIGHT, VIDEO_WIDTH


def load_condition(args):
    if args.episode:
        with np.load(args.episode, allow_pickle=False) as episode:
            i = min(max(args.frame, 0), len(episode["qpos"]) - 1)
            image = stitch_views(episode["head_image"][i], episode["left_image"][i], episode["right_image"][i])
            state = np.asarray(episode["qpos"][i], np.float32)
            instruction = args.instruction or str(episode["instruction"].item())
    else:
        if not args.image:
            raise ValueError("provide --episode or --image")
        image = np.asarray(Image.open(args.image).convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT), PIL_BILINEAR))
        state = np.zeros(STATE_DIM, np.float32) if args.state is None else np.load(args.state).astype(np.float32)
        instruction = args.instruction
        if not instruction:
            raise ValueError("--image mode also requires --instruction")
    return image, state, instruction


def save_grid(path: Path, condition: np.ndarray, frames: np.ndarray) -> None:
    thumbs = [condition, *frames]
    rows = []
    for start in range(0, len(thumbs), 3):
        row = np.concatenate(thumbs[start:start + 3], axis=1)
        if row.shape[1] < VIDEO_WIDTH * 3:
            row = np.pad(row, ((0, 0), (0, VIDEO_WIDTH * 3 - row.shape[1]), (0, 0)))
        rows.append(row)
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="single image -> Motus future video")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--state", type=Path, help="optional .npy 16D state for --image")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--instruction")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=16016)
    parser.add_argument("--output", type=Path, default=Path("outputs/future_red"))
    args = parser.parse_args()
    image, state, instruction = load_condition(args)
    args.output.mkdir(parents=True, exist_ok=True)
    policy = MotusPolicy(args.checkpoint, args.device, args.steps, args.seed)
    result = policy.predict_composite(image, state, instruction)
    Image.fromarray(image).save(args.output / "condition.png")
    write_mp4(args.output / "future.mp4", result.frames, fps=5)
    imageio.mimsave(args.output / "future.gif", list(result.frames), duration=0.2, loop=0)
    save_grid(args.output / "future_grid.png", image, result.frames)
    np.save(args.output / "predicted_actions.npy", result.actions)
    print(f"saved {len(result.frames)} future frames and {result.actions.shape} actions to {args.output}")


if __name__ == "__main__":
    main()
