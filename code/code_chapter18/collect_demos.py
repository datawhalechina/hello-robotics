"""采集高质量单色脚本专家演示，作为本地离线 D_demo。"""

from __future__ import annotations

import argparse

import numpy as np
from config import COLORS, DATASET_FPS, SimulationConfig, TaskConfig, demonstration_dir
from dataset import EpisodeRecorder, prepare_episode_directory, update_manifest
from expert import ScriptedExpert
from robot import G2Robot
from simulation import G2Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="采集 red/green/blue 单色成功演示")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument(
        "--dataset-fps",
        type=int,
        default=DATASET_FPS,
        help="专家轨迹保存频率；必须能整除物理仿真频率",
    )
    parser.add_argument("--position-noise", type=float, default=0.01)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes 必须大于 0")
    if args.start_id < 0:
        raise ValueError("--start-id 不能小于 0")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts 必须大于 0")

    simulation_config = SimulationConfig(headless=args.headless)
    if args.dataset_fps < 1 or simulation_config.physics_hz % args.dataset_fps != 0:
        raise ValueError(
            f"--dataset-fps 必须是 {simulation_config.physics_hz} Hz 的正整数因子"
        )
    physics_steps_per_action = simulation_config.physics_hz // args.dataset_fps
    print(
        f"[频率] physics={simulation_config.physics_hz} Hz, "
        f"dataset={args.dataset_fps} Hz, steps_per_action={physics_steps_per_action}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    output = demonstration_dir()
    if args.start_id == 0:
        prepare_episode_directory(output, overwrite=args.overwrite)
    else:
        output.mkdir(parents=True, exist_ok=True)
    sim = G2Simulation(simulation_config, TaskConfig())
    try:
        robot = G2Robot(sim.robot)
        for episode_id in range(args.start_id, args.start_id + args.episodes):
            color = COLORS[episode_id % len(COLORS)]
            prompt = f"Pick up the {color} block and place it into the empty box."
            for attempt in range(1, args.max_attempts + 1):
                robot.reset_episode()
                sim.task.randomize(rng, args.position_noise)
                for _ in range(40):
                    sim.step(render=True)
                recorder = EpisodeRecorder(
                    output,
                    episode_id,
                    prompt,
                    color,
                    dataset_fps=args.dataset_fps,
                    iteration=0,
                )
                print(
                    f"[演示] episode={episode_id}, target={color}, attempt={attempt}",
                    flush=True,
                )
                ScriptedExpert(sim, robot, recorder).run(color)
                success = color in sim.task.completed
                if success:
                    path = recorder.save(
                        True, "demonstration", overwrite=args.overwrite
                    )
                    print(f"[保存] {path.name}, success=True", flush=True)
                    break
                print("[重试] 脚本专家未完成，本次失败轨迹不写入 D_demo", flush=True)
            else:
                raise RuntimeError(
                    f"episode={episode_id} 连续 {args.max_attempts} 次专家演示失败"
                )
        print(f"[Manifest] {update_manifest(output)}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
