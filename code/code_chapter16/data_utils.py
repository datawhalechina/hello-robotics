"""Small data helpers shared by collection, conversion, training checks and inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import torch

from settings import (
    ACTION_DIM,
    STATE_DIM,
    ROBOTWIN_VIDEO_HEIGHT,
    ROBOTWIN_VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)

PIL_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def scalar(value):
    return np.asarray(value).item()


def rgb(image: np.ndarray) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"expected HWC RGB image, got {array.shape}")
    return Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB")


def concatenate_views(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Reproduce Motus' RoboTwin converter: full head over two half-size wrists."""
    head_image = rgb(head)
    source_w, source_h = head_image.size
    half_w, half_h = source_w // 2, source_h // 2
    combined = Image.new("RGB", (source_w, source_h + half_h))
    combined.paste(head_image, (0, 0))
    combined.paste(rgb(left).resize((half_w, half_h), PIL_BILINEAR), (0, source_h))
    combined.paste(
        rgb(right).resize((source_w - half_w, half_h), PIL_BILINEAR),
        (half_w, source_h),
    )
    # The official converter writes every combined frame as 360x320.
    combined = combined.resize((ROBOTWIN_VIDEO_WIDTH, ROBOTWIN_VIDEO_HEIGHT), PIL_BILINEAR)
    return np.asarray(combined, dtype=np.uint8)


def resize_with_padding(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Aspect-preserving resize and black padding, matching Motus image_utils."""
    source = rgb(image)
    source_w, source_h = source.size
    scale = min(height / source_h, width / source_w)
    resized_w, resized_h = int(source_w * scale), int(source_h * scale)
    resized = source.resize((resized_w, resized_h), PIL_BILINEAR)
    canvas = Image.new("RGB", (width, height))
    canvas.paste(resized, ((width - resized_w) // 2, (height - resized_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def stitch_views(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the final 384x320 model input produced by the official loader."""
    return resize_with_padding(concatenate_views(head, left, right), VIDEO_HEIGHT, VIDEO_WIDTH)


def image_tensor(image: np.ndarray | Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def cache_name(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] + ".pt"


def write_mp4(
    path: Path,
    frames: Iterable[np.ndarray],
    fps: int,
    height: int = VIDEO_HEIGHT,
    width: int = VIDEO_WIDTH,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    count = 0
    try:
        for frame in frames:
            frame = np.asarray(frame, np.uint8)
            if frame.shape != (height, width, 3):
                raise ValueError(f"wrong frame shape: {frame.shape}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        writer.release()
    return count


def find_checkpoint_file(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = (
        path / "mp_rank_00_model_states.pt",
        path / "pytorch_model" / "mp_rank_00_model_states.pt",
        path / "pytorch_model_0.bin",
        path / "pytorch_model.bin",
        path / "model.safetensors",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("checkpoint not found; tried:\n" + "\n".join(map(str, candidates)))


def load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    file = find_checkpoint_file(path)
    if file.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(file), device="cpu")
    payload = torch.load(file, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("module", "state_dict", "model"):
            if isinstance(payload.get(key), dict):
                return payload[key]
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint payload: {type(payload)}")
    return payload


def validate_vector(name: str, value: np.ndarray, dim: int) -> np.ndarray:
    value = np.asarray(value, np.float32).reshape(-1)
    if value.shape != (dim,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite {dim}D, got {value.shape}")
    return value


def dataset_summary(root: Path) -> dict:
    rows = []
    for split in ("clean", "randomized"):
        for task in sorted((root / split).glob("*")) if (root / split).exists() else []:
            if not task.is_dir():
                continue
            stems = {p.stem for p in (task / "qpos").glob("*.pt")}
            valid = [s for s in stems if (task / "commands" / f"{s}.pt").is_file()
                     and (task / "videos" / f"{s}.mp4").is_file()
                     and (task / "umt5_wan" / f"{s}.pt").is_file()
                     and (task / "metas" / f"{s}.txt").is_file()]
            rows.append({"split": split, "task": task.name, "episodes": len(valid)})
    return {"root": str(root), "episodes": sum(r["episodes"] for r in rows), "groups": rows}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def stage3_indices(condition: int = 0) -> tuple[list[int], list[int]]:
    """Return official Stage-3 action/video raw-frame indices for one condition frame."""
    from settings import ACTION_CHUNK_SIZE, GLOBAL_DOWNSAMPLE_RATE, NUM_VIDEO_FRAMES, VIDEO_ACTION_FREQ_RATIO
    actions = [condition + (i + 1) * GLOBAL_DOWNSAMPLE_RATE for i in range(ACTION_CHUNK_SIZE)]
    videos = [actions[(i + 1) * VIDEO_ACTION_FREQ_RATIO - 1] for i in range(NUM_VIDEO_FRAMES)]
    return actions, videos
