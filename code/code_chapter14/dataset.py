"""采集阶段使用的简单NPZ回合记录器。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EpisodeRecorder:
    def __init__(self, output_dir: Path, episode_id: int, prompt: str, color: str, sample_every: int) -> None:
        self.output_dir = Path(output_dir)
        self.episode_id = episode_id
        self.prompt = prompt
        self.color = color
        self.sample_every = max(1, int(sample_every))
        self.frames = {key: [] for key in ("head_image", "left_image", "right_image", "state", "actions")}
        self._step = 0

    def should_record(self) -> bool:
        self._step += 1
        return self._step % self.sample_every == 0

    def record(self, images, state, action) -> None:
        head, left, right = images
        for key, value in zip(("head_image", "left_image", "right_image"), (head, left, right), strict=True):
            self.frames[key].append(np.asarray(value, dtype=np.uint8))
        self.frames["state"].append(np.asarray(state, dtype=np.float32))
        self.frames["actions"].append(np.asarray(action, dtype=np.float32))

    def save(self, success: bool, overwrite: bool = False) -> Path:
        if not self.frames["state"]:
            raise RuntimeError("当前回合没有采集到任何帧")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"episode_{self.episode_id:04d}.npz"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path}已存在；如需覆盖请加--overwrite")
        np.savez_compressed(
            path,
            **{key: np.stack(value) for key, value in self.frames.items()},
            prompt=np.asarray(self.prompt),
            target_color=np.asarray(self.color),
            success=np.asarray(bool(success)),
        )
        return path


def update_manifest(output_dir: Path) -> None:
    episodes = []
    for path in sorted(Path(output_dir).glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            episodes.append({
                "file": path.name,
                "frames": int(data["state"].shape[0]),
                "target_color": str(data["target_color"]),
                "success": bool(data["success"]),
                "prompt": str(data["prompt"]),
            })
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "manifest.json").write_text(
        json.dumps({"episodes": episodes}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
