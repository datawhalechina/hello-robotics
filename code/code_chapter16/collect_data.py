"""Collect fresh 30 Hz G2 demonstrations for Motus Stage 3.

The task expert comes from this chapter's local G2 runtime.  The recorder stores
measured qpos and three synchronized cameras at 30 Hz, matching the official
RoboTwin loader's global_downsample_rate=3 sampling rule.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from environment import AutoExpert, COLORS, G2Robot, G2Simulation, MotusSimulationConfig
from settings import (
    CHAPTER16_CLEAN_EPISODES,
    CHAPTER16_RANDOMIZED_EPISODES,
    CHAPTER16_POSITION_NOISE,
    RAW_FPS,
    RAW_ROOT,
    TASK_NAMES,
    TASK_TEXT,
    ensure_dirs,
)


def log(message: str) -> None:
    """Print progress immediately, including when stdout is piped through tee."""
    print(message, flush=True)


@dataclass
class RawRecorder:
    split: str
    color: str
    episode_id: int
    head: list[np.ndarray] = field(default_factory=list)
    left: list[np.ndarray] = field(default_factory=list)
    right: list[np.ndarray] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    command: list[np.ndarray] = field(default_factory=list)

    def add(self, images, state, action, **_metadata) -> None:
        head, left, right = images
        self.head.append(np.asarray(head, np.uint8))
        self.left.append(np.asarray(left, np.uint8))
        self.right.append(np.asarray(right, np.uint8))
        self.qpos.append(np.asarray(state, np.float32))
        self.command.append(np.asarray(action, np.float32))

    def save(self, success: bool, reason: str) -> Path:
        folder = RAW_ROOT / self.split / TASK_NAMES[self.color]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"episode_{self.episode_id:06d}.npz"
        np.savez_compressed(
            path,
            head_image=np.asarray(self.head, np.uint8),
            left_image=np.asarray(self.left, np.uint8),
            right_image=np.asarray(self.right, np.uint8),
            qpos=np.asarray(self.qpos, np.float32),
            command=np.asarray(self.command, np.float32),
            instruction=np.asarray(TASK_TEXT[self.color]),
            color=np.asarray(self.color),
            split=np.asarray(self.split),
            success=np.asarray(bool(success)),
            stop_reason=np.asarray(reason),
            raw_fps=np.asarray(RAW_FPS, np.int64),
        )
        return path


def balanced_quotas(total: int, colors: list[str]) -> dict[str, int]:
    """Split an official total exactly and deterministically across selected tasks."""
    if total < 0:
        raise ValueError("episode total must be non-negative")
    if not colors:
        raise ValueError("at least one color is required")
    base, remainder = divmod(total, len(colors))
    return {color: base + int(index < remainder) for index, color in enumerate(colors)}


def episode_paths(split: str, color: str) -> list[Path]:
    return sorted((RAW_ROOT / split / TASK_NAMES[color]).glob("episode_*.npz"))


def next_episode_id(split: str, color: str) -> int:
    paths = episode_paths(split, color)
    return 0 if not paths else int(paths[-1].stem.rsplit("_", 1)[1]) + 1


def successful_episodes(split: str, color: str) -> int:
    count = 0
    for path in episode_paths(split, color):
        try:
            with np.load(path, allow_pickle=False) as episode:
                count += int(bool(episode["success"].item()))
        except Exception as exc:
            log(f"[warning] ignore unreadable episode {path}: {exc}")
    return count


def collect_group(
    sim,
    robot,
    rng,
    split: str,
    color: str,
    target_successes: int,
    noise: float,
    max_new_attempts: int,
) -> None:
    """Resume a group until its total number of successful episodes reaches target."""
    episode_id = next_episode_id(split, color)
    accepted = successful_episodes(split, color)
    attempts = 0
    log(f"[{split}/{color}] existing={accepted}, target={target_successes}")
    while accepted < target_successes:
        if max_new_attempts and attempts >= max_new_attempts:
            raise RuntimeError(
                f"{split}/{color}: reached --max-new-attempts={max_new_attempts}; "
                f"successful={accepted}/{target_successes}"
            )
        attempts += 1
        log(f"[{split}/{color}] attempt={attempts}: reset and randomize")
        robot.reset()
        sim.task.randomize(rng, noise)
        for _ in range(30):
            sim.step(True)
        recorder = RawRecorder(split, color, episode_id)
        log(f"[{split}/{color}] attempt={attempts}: expert motion started")
        expert = AutoExpert(sim, robot, recorder=recorder, intervention=False)
        reason = "success"
        try:
            expert.demonstrate(color)
        except Exception as exc:
            reason = f"expert_error:{type(exc).__name__}:{exc}"
        success = sim.task.success(color)
        if not success and reason == "success":
            reason = "task_not_completed"
        if len(recorder.qpos) < 49:
            success = False
            reason = "too_short"
        path = recorder.save(success, reason)
        accepted += int(success)
        log(
            f"{path.relative_to(RAW_ROOT)} frames={len(recorder.qpos)} "
            f"success={success} accepted={accepted}/{target_successes} new_attempts={attempts}"
        )
        episode_id += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect fresh Motus Stage-3 trajectories")
    parser.add_argument(
        "--clean-total", type=int, default=CHAPTER16_CLEAN_EPISODES,
        help="successful clean trajectories across selected colors",
    )
    parser.add_argument(
        "--randomized-total", type=int, default=CHAPTER16_RANDOMIZED_EPISODES,
        help="successful randomized trajectories across selected colors",
    )
    parser.add_argument("--randomized-noise", type=float, default=CHAPTER16_POSITION_NOISE)
    parser.add_argument("--colors", nargs="+", choices=COLORS, default=list(COLORS))
    parser.add_argument("--seed", type=int, default=16000)
    parser.add_argument(
        "--max-new-attempts", type=int, default=0,
        help="per split/color safety cap; 0 means unlimited",
    )
    parser.add_argument("--plan-only", action="store_true", help="print quotas without starting Isaac Sim")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.clean_total < 0 or args.randomized_total < 0 or args.max_new_attempts < 0:
        parser.error("episode counts must be non-negative")

    clean = balanced_quotas(args.clean_total, args.colors)
    randomized = balanced_quotas(args.randomized_total, args.colors)
    log(f"clean quotas: {clean} (total={sum(clean.values())})")
    log(f"randomized quotas: {randomized} (total={sum(randomized.values())})")
    if args.plan_only:
        return

    ensure_dirs()
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    log(f"starting Isaac Sim (headless={args.headless}) ...")
    sim = G2Simulation(MotusSimulationConfig(headless=args.headless))
    log("Isaac Sim ready; constructing G2 controller ...")
    try:
        robot = G2Robot(sim.articulation)
        log("G2 controller ready; collection started")
        for color in args.colors:
            collect_group(sim, robot, rng, "clean", color, clean[color], 0.0, args.max_new_attempts)
            collect_group(
                sim, robot, rng, "randomized", color, randomized[color],
                args.randomized_noise, args.max_new_attempts,
            )
    finally:
        sim.close()
    log(f"collection finished in {(time.monotonic() - started) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
