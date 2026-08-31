"""Collect successful scripted demonstrations for the optional initial SFT."""

from __future__ import annotations

import argparse

import numpy as np

from auto_expert import AutoExpert
from config import (
    COLORS,
    POSITION_NOISE,
    SEED,
    TASK_TEMPLATE,
    SimulationConfig,
    demos_dir,
)
from dataset import EpisodeRecorder, prepare_dir
from robot import G2Robot
from simulation import G2Simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-color", type=int, default=10)
    parser.add_argument("--position-noise", type=float, default=POSITION_NOISE)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    output = demos_dir()
    prepare_dir(output, args.overwrite)
    rng = np.random.default_rng(args.seed)
    sim = G2Simulation(SimulationConfig(headless=args.headless))
    try:
        robot = G2Robot(sim.articulation)
        episode = 0
        for color_index, color in enumerate(COLORS):
            for _ in range(args.episodes_per_color):
                robot.reset()
                sim.task.randomize(rng, args.position_noise)
                for _ in range(30):
                    sim.step(True)
                recorder = EpisodeRecorder(
                    output,
                    episode,
                    TASK_TEMPLATE.format(color=color),
                    color,
                    color_index,
                    collector_policy_id="scripted_demo_v1",
                )
                expert = AutoExpert(sim, robot, recorder=recorder, intervention=False)
                stop_reason = "completed"
                try:
                    expert.demonstrate(color)
                except Exception as exc:  # preserve failures for Value debugging
                    stop_reason = f"expert_error:{type(exc).__name__}"
                success = sim.task.success(color)
                path = recorder.save(
                    success=success,
                    episode_kind="demonstration",
                    stop_reason=stop_reason if not success else "success",
                    use_for_sft=success,
                    use_for_value=True,
                )
                print(
                    f"{path.name}: color={color} success={success} frames={len(recorder)}"
                )
                episode += 1
    finally:
        sim.close()


if __name__ == "__main__":
    main()
