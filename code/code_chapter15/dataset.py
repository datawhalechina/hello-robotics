"""Small, inspectable NPZ episode format used before exporting to LeRobot."""

from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
import shutil
import numpy as np
from config import ACTION_DIM, COLORS, STATE_DIM

FRAME_KEYS = (
    "head_image",
    "left_image",
    "right_image",
    "state",
    "action",
    "policy_action",
    "intervention_state",
    "is_intervention",
    "collector_policy_id",
    "episode_index",
    "frame_index",
    "task_index",
    "source",
)


def scalar(value):
    return np.asarray(value).item()


def prepare_dir(
    path: Path, overwrite: bool = False, *, resume: bool = False
) -> None:
    path = Path(path)
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    if path.exists() and any(path.glob("episode_*.npz")):
        if resume:
            return
        if not overwrite:
            raise FileExistsError(f"{path} already contains episodes")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def episode_paths(directories: Iterable[Path]) -> list[Path]:
    result = [p for d in directories for p in sorted(Path(d).glob("episode_*.npz"))]
    if not result:
        raise RuntimeError("no episode_*.npz files found")
    return result


class EpisodeRecorder:
    def __init__(
        self,
        output: Path,
        episode_index: int,
        task: str,
        target_color: str,
        task_index: int,
        collector_policy_id: str = "",
        max_frames: int | None = None,
    ):
        if target_color not in COLORS:
            raise ValueError(f"unknown color: {target_color}")
        self.output, self.episode_index, self.task = (
            Path(output),
            int(episode_index),
            str(task),
        )
        self.target_color, self.task_index, self.collector_policy_id = (
            target_color,
            int(task_index),
            str(collector_policy_id),
        )
        self.max_frames = max_frames
        self.frames = {key: [] for key in FRAME_KEYS}

    def __len__(self):
        return len(self.frames["state"])

    @property
    def full(self):
        return self.max_frames is not None and len(self) >= self.max_frames

    def add(
        self,
        images,
        state,
        action,
        *,
        intervention_state: int,
        source: str,
        policy_action=None,
    ):
        if self.full:
            raise RuntimeError("episode frame budget exhausted")
        head, left, right = images
        state = np.asarray(state, np.float32).reshape(STATE_DIM)
        action = np.asarray(action, np.float32).reshape(ACTION_DIM)
        active = int(intervention_state) == 1
        if policy_action is None:
            policy_action = np.zeros(ACTION_DIM, np.float32) if active else action
        values = {
            "head_image": np.asarray(head, np.uint8),
            "left_image": np.asarray(left, np.uint8),
            "right_image": np.asarray(right, np.uint8),
            "state": state,
            "action": action,
            "policy_action": np.asarray(policy_action, np.float32).reshape(ACTION_DIM),
            "intervention_state": np.int8(intervention_state),
            "is_intervention": np.int8(active),
            "collector_policy_id": self.collector_policy_id,
            "episode_index": np.int64(self.episode_index),
            "frame_index": np.int64(len(self)),
            "task_index": np.int64(self.task_index),
            "source": str(source),
        }
        for key, value in values.items():
            self.frames[key].append(value)

    def save(
        self,
        *,
        success: bool,
        episode_kind: str,
        stop_reason: str,
        correction_segments: list[str] | None = None,
        use_for_sft: bool = False,
        use_for_value: bool = True,
        overwrite: bool = False,
    ) -> Path:
        if not len(self):
            raise RuntimeError("cannot save an empty episode")
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"episode_{self.episode_index:06d}.npz"
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        data = {key: np.asarray(value) for key, value in self.frames.items()}
        data.update(
            task=np.asarray(self.task),
            target_color=np.asarray(self.target_color),
            success=np.asarray(bool(success)),
            episode_success=np.asarray("success" if success else "failure"),
            length=np.asarray(len(self), np.int64),
            episode_kind=np.asarray(episode_kind),
            stop_reason=np.asarray(stop_reason),
            correction_segments=np.asarray(correction_segments or [], dtype="U64"),
            use_for_sft=np.asarray(bool(use_for_sft)),
            use_for_value=np.asarray(bool(use_for_value)),
        )
        np.savez_compressed(path, **data)
        return path


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        ep = {k: np.asarray(source[k]) for k in source.files}
    missing = [
        k
        for k in (*FRAME_KEYS, "task", "target_color", "success", "length")
        if k not in ep
    ]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    length = int(scalar(ep["length"]))
    if length < 1 or any(len(ep[k]) != length for k in FRAME_KEYS):
        raise ValueError(f"{path}: inconsistent frame lengths")
    if (
        ep["state"].shape != (length, STATE_DIM)
        or ep["action"].shape != (length, ACTION_DIM)
        or ep["policy_action"].shape != (length, ACTION_DIM)
    ):
        raise ValueError(f"{path}: invalid state/action shape")
    states = ep["intervention_state"].astype(int)
    intervention = ep["is_intervention"].astype(bool)
    if not np.array_equal(intervention, states == 1):
        raise ValueError(f"{path}: intervention flags disagree")
    if np.any(ep["policy_action"][states == 1] != 0):
        raise ValueError(f"{path}: ACTIVE policy_action must be zero")
    return ep


def task_max_lengths(paths: list[Path]) -> dict[int, int]:
    result: dict[int, int] = {}
    for path in paths:
        ep = load_episode(path)
        task = int(ep["task_index"][0])
        result[task] = max(result.get(task, 0), len(ep["state"]))
    return result


def summarize(paths: list[Path]) -> dict:
    eps = [load_episode(p) for p in paths]
    frames = sum(len(e["state"]) for e in eps)
    return {
        "episodes": len(eps),
        "frames": frames,
        "successes": sum(int(scalar(e["success"])) for e in eps),
        "intervention_frames": sum(int(e["is_intervention"].sum()) for e in eps),
        "release_frames": sum(int((e["intervention_state"] == 2).sum()) for e in eps),
    }
