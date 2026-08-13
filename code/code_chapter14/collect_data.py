"""在当前Isaac Sim场景中自动采集单物块入盒示教。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .config import DATASET_FPS, RAW_DATA_DIR, SimulationConfig, TaskConfig
    from .dataset import EpisodeRecorder, update_manifest
    from .expert import ScriptedExpert
    from .robot import G2Robot
    from .simulation import G2Simulation
except ImportError:
    from config import DATASET_FPS, RAW_DATA_DIR, SimulationConfig, TaskConfig
    from dataset import EpisodeRecorder, update_manifest
    from expert import ScriptedExpert
    from robot import G2Robot
    from simulation import G2Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="采集G2三色物块入盒示教")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--position-noise", type=float, default=0.0,
        help="物块水平随机范围（米）；先用0采集稳定示教，再逐步增大到0.01~0.02",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes必须大于0")

    rng = np.random.default_rng(args.seed)
    sim = G2Simulation(
        SimulationConfig(headless=args.headless),
        TaskConfig(),
    )
    try:
        robot = G2Robot(sim.robot)
        # 先让机器人稳定在公开G2数据的home姿态，再重置动态物块。
        class _NoRecord:
            @staticmethod
            def should_record() -> bool:
                return False

            @staticmethod
            def record(*_args) -> None:
                pass

        ScriptedExpert(sim, robot, _NoRecord()).move_action(
            robot.home_action, sim.config.warmup_steps / sim.config.physics_hz
        )
        for episode_id in range(args.start_id, args.start_id + args.episodes):
            sim.task.randomize(rng, args.position_noise)
            for _ in range(30):
                sim.step(render=True)
            color = ("red", "green", "blue")[episode_id % 3]
            prompt = f"Pick up the {color} block and place it into the empty box."
            recorder = EpisodeRecorder(
                args.output, episode_id, prompt, color,
                sample_every=max(1, sim.config.physics_hz // DATASET_FPS),
            )
            print(f"[采集] episode={episode_id}, target={color}", flush=True)
            ScriptedExpert(sim, robot, recorder).run(color)
            success = color in sim.task.completed
            path = recorder.save(success, overwrite=args.overwrite)
            print(f"[采集] 保存{path.name}, success={success}", flush=True)
        update_manifest(args.output)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
