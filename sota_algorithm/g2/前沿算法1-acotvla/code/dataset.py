"""最小 NPZ 示教记录器，只保存 ACoT-VLA 训练需要的字段。"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRAME_KEYS = ("head_image", "left_image", "right_image", "state", "actions")


def prepare_episode_directory(directory: Path, *, overwrite: bool) -> None:
    directory = Path(directory)
    if directory.exists() and any(directory.glob("episode_*.npz")):
        if not overwrite:
            raise FileExistsError(f"{directory} 已有回合；需要重采请添加 --overwrite")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class EpisodeRecorder:
    output_dir: Path
    episode_id: int
    prompt: str
    target_color: str
    dataset_fps: int = 30
    iteration: int = 0
    collector_checkpoint: str = "scripted_expert"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.frames = {key: [] for key in FRAME_KEYS}
        self.observation_times = []
        self.image_times = []

    def record(
        self,
        images,
        state,
        action,
        *,
        observation_time=None,
        image_times=None,
        **_metadata,
    ) -> None:
        if observation_time is not None:
            times = np.asarray(image_times, dtype=np.float64)
            if (
                times.shape != (3,)
                or not np.isfinite(observation_time)
                or not np.allclose(times, observation_time, rtol=0, atol=1e-6)
            ):
                raise ValueError("必须记录同一时刻的三路相机与状态")
            self.observation_times.append(float(observation_time))
            self.image_times.append(times.copy())
        for key, value in zip(FRAME_KEYS[:3], images, strict=True):
            self.frames[key].append(np.asarray(value, dtype=np.uint8))
        self.frames["state"].append(np.asarray(state, dtype=np.float32))
        self.frames["actions"].append(np.asarray(action, dtype=np.float32))

    def save(
        self,
        success: bool,
        _episode_kind: str = "demonstration",
        *,
        overwrite: bool = False,
        **_metadata,
    ) -> Path:
        if not self.frames["state"]:
            raise RuntimeError("当前回合没有采集到帧")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"episode_{self.episode_id:04d}.npz"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} 已存在；需要覆盖时添加 --overwrite")
        timestamps = {}
        if self.observation_times:
            if len(self.observation_times) != len(self.frames["state"]):
                raise ValueError("时间戳数量与轨迹帧数不一致")
            timestamps = {
                "observation_time": np.asarray(
                    self.observation_times, dtype=np.float64
                ),
                "image_time": np.stack(self.image_times),
            }
        np.savez_compressed(
            path,
            **{key: np.stack(value) for key, value in self.frames.items()},
            **timestamps,
            prompt=np.asarray(self.prompt),
            target_color=np.asarray(self.target_color),
            success=np.asarray(bool(success)),
            fps=np.asarray(self.dataset_fps, dtype=np.int64),
        )
        return path


def update_manifest(directory: Path) -> Path:
    records = []
    for path in sorted(Path(directory).glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            records.append(
                {
                    "file": path.name,
                    "frames": int(data["state"].shape[0]),
                    "target_color": str(data["target_color"]),
                    "success": bool(data["success"]),
                    "prompt": str(data["prompt"]),
                    "fps": int(data["fps"]) if "fps" in data else None,
                }
            )
    output = Path(directory) / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "episodes": records,
                "summary": {
                    "episodes": len(records),
                    "frames": sum(item["frames"] for item in records),
                    "successes": sum(item["success"] for item in records),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
