"""执行单色物块抓取。"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
from config import COLORS, ControlConfig, SimulationConfig, TaskConfig
from robot import G2Robot
from simulation import G2Simulation
from vla_client import ChunkRunner, VLAClient


def prompt(color: str) -> str:
    return f"Pick up the {color} block and place it into the empty box."


def main() -> None:
    parser = argparse.ArgumentParser(description="运行一个 ACoT-VLA 抓取回合")
    parser.add_argument("--target", choices=COLORS, default="red")
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--position-noise", type=float, default=0.01)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--execute-chunk", type=int, default=8)
    parser.add_argument("--max-replans-per-color", type=int, default=60)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.execute_chunk < 1 or args.max_replans_per_color < 1:
        raise ValueError("execute chunk 和 max replans 必须大于 0")
    if args.position_noise < 0:
        raise ValueError("--position-noise 不能小于 0")

    client = VLAClient(args.host, args.port)
    print(f"[模型] {client.metadata.get('checkpoint', 'unknown')}")

    sim = G2Simulation(SimulationConfig(headless=args.headless), TaskConfig())
    try:
        robot = G2Robot(sim.robot)
        robot.reset_episode()
        for _ in range(10):
            sim.step(render=True)
        sim.task.randomize(np.random.default_rng(args.seed), args.position_noise)
        for _ in range(30):
            sim.step(render=True)

        runner = ChunkRunner(
            robot,
            sim,
            replace(ControlConfig(), execute_chunk=args.execute_chunk),
        )
        runner.move_home()
        color = args.target
        used = 0
        for _ in range(args.max_replans_per_color):
            if color in sim.task.completed:
                break
            state, images = sim.observe(robot)
            actions = client.infer(state, images, prompt(color))
            runner.execute(actions)
            used += 1
        print(
            f"[任务] {color}: "
            f"{'成功' if color in sim.task.completed else '失败'}，replans={used}"
        )
        print(f"[结果] completed={sorted(sim.task.completed)}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
